"""
alcuin.cli.main - Research-First Command Line Interface for Alcuin
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from alcuin.agent.graph import ResearchAgent
from alcuin.storage.engine import VaultStorage
from alcuin.storage.vault import Vault, VaultRegistry

console = Console()


def resolve_or_prompt_vault(
    target: Optional[str] = None,
    cwd: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Vault:
    """Resolve vault from target flag, directory walk-up, or interactive picker."""
    # 1. Try direct resolution (flag or parent directory)
    if target:
        try:
            return Vault.resolve(target=target, overrides=overrides)
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            sys.exit(1)

    # 2. Check current directory walkup
    walkup_vault = Vault.resolve(cwd=cwd, overrides=overrides)
    if walkup_vault:
        return walkup_vault

    # 3. Fallback: check registered vaults
    registered = VaultRegistry.list_vaults()
    if not registered:
        console.print(
            "[bold red]No Alcuin vault detected in current directory and no registered vaults found.[/bold red]\n"
            "Run [bold cyan]alcuin init[/bold cyan] in your document folder to create one."
        )
        sys.exit(1)

    # If non-interactive shell, cannot prompt
    if not sys.stdin.isatty():
        console.print(
            "[bold red]Not inside an Alcuin vault and no --vault flag provided in non-interactive shell.[/bold red]\n"
            "Specify [bold cyan]-v <vault_name>[/bold cyan]."
        )
        sys.exit(1)

    # Interactive prompt
    console.print("\n[bold yellow]Not inside an Alcuin vault. Choose from registered vaults:[/bold yellow]")
    vault_names = list(registered.keys())
    for idx, name in enumerate(vault_names, 1):
        v_path = registered[name]
        console.print(f"  [bold cyan][{idx}][/bold cyan] [bold]{name}[/bold] ({v_path})")

    choice_str = click.prompt(f"\nSelect vault [1-{len(vault_names)}]", type=str, default="1")
    try:
        choice_idx = int(choice_str) - 1
        if 0 <= choice_idx < len(vault_names):
            chosen_name = vault_names[choice_idx]
            return Vault.resolve(target=chosen_name, overrides=overrides)
    except Exception:
        pass

    console.print("[bold red]Invalid selection.[/bold red]")
    sys.exit(1)


class DefaultResearchGroup(click.Group):
    """Click command group that routes unrecognized arguments directly to 'research'."""

    def parse_args(self, ctx, args):
        subcommands = set(self.commands.keys())
        has_subcommand = False
        for arg in args:
            if not arg.startswith("-"):
                if arg in subcommands:
                    has_subcommand = True
                break

        if not has_subcommand and args and not any(a in ("--help", "-h") for a in args):
            args = ["research"] + args

        return super().parse_args(ctx, args)


@click.group(cls=DefaultResearchGroup, invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Alcuin - Autonomous, agentic personal knowledge engine."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command(
    name="research",
    short_help="Research a question across vault documents (Default command)",
)
@click.argument("question", nargs=-1, required=True)
@click.option("-v", "--vault", "vault_target", help="Vault name or directory path")
@click.option("--max-hops", type=int, help="Maximum wikilink hop depth")
@click.option("--max-iterations", type=int, help="Maximum research iteration cycles")
@click.option("--model", "model_name", help="LLM model name (overrides vault and global config)")
@click.option("--model-endpoint", help="LLM model API endpoint URL")
@click.option("--temperature", type=float, help="LLM sampling temperature")
@click.option("--verbose", "-V", is_flag=True, help="Display live agent thought iterations and research steps")
def research_cmd(
    question: Tuple[str, ...],
    vault_target: Optional[str],
    max_hops: Optional[int],
    max_iterations: Optional[int],
    model_name: Optional[str],
    model_endpoint: Optional[str],
    temperature: Optional[float],
    verbose: bool = False,
):
    """Run autonomous multi-hop research on a question."""
    query = " ".join(question).strip()
    if not query:
        console.print("[bold red]Please supply a question to research.[/bold red]")
        sys.exit(1)

    overrides = {
        "max_hops": max_hops,
        "max_iterations": max_iterations,
        "model_name": model_name,
        "model_endpoint": model_endpoint,
        "temperature": temperature,
    }

    vault = resolve_or_prompt_vault(target=vault_target, overrides=overrides)
    if verbose:
        console.print(f"\n[dim]Using vault '[bold]{vault.name}[/bold]' ({vault.root_path})[/dim]")
        console.print(f"[bold cyan]Researching:[/bold cyan] {query}\n")

    agent = ResearchAgent(vault)

    status_text = "[bold green]Researching...[/bold green]"
    with console.status(status_text, spinner="dots") as status:
        final_state = None
        for node_name, state_update in agent.stream_research(query):
            if "thought_history" in state_update and state_update["thought_history"]:
                latest_thought = state_update["thought_history"][-1]
                if verbose:
                    status.update(f"[dim]{latest_thought}[/dim]")
                    console.print(f"  [dim]• {latest_thought}[/dim]")
            final_state = state_update

    if not final_state or not final_state.get("final_answer"):
        final_state = agent.research(query)

    answer = final_state.get("final_answer", "No answer could be synthesized.")
    confidence = final_state.get("confidence", 0.0)
    sources = final_state.get("sources", [])

    console.print("\n")
    console.print(Markdown(answer))

    if verbose:
        footer = f"\n[dim]Confidence: {int(confidence * 100)}%"
        if sources:
            footer += f" | Sources: {', '.join(sources)}"
        footer += "[/dim]"
        console.print(footer)


@cli.command(short_help="Initialize a new in-situ vault in target directory")
@click.argument("path", default=".", type=click.Path(file_okay=False))
@click.option("--name", help="Vault name (defaults to directory name)")
@click.option("--description", default="", help="Description of the vault")
def init(path: str, name: Optional[str], description: str):
    """Initialize a directory as an Alcuin knowledge vault."""
    target_path = Path(path).resolve()
    vault = Vault.init(target_path, name=name, description=description)
    console.print(
        f"[bold green]Initialized vault '[bold]{vault.name}[/bold]' at {target_path}[/bold green]\n"
        f"Metadata created at {vault.alcuin_dir}\n"
        f"Run [bold cyan]alcuin index[/bold cyan] to ingest existing documents."
    )


@cli.group(short_help="Manage system-wide registered vaults")
def vault():
    """Vault management commands."""
    pass


@vault.command(name="list", short_help="List all registered vaults")
def vault_list():
    """List all registered vaults across the system."""
    vaults = VaultRegistry.list_vaults()
    if not vaults:
        console.print("[dim]No vaults registered. Run 'alcuin init' to create one.[/dim]")
        return

    table = Table(title="Alcuin Registered Vaults")
    table.add_column("Vault Name", style="bold cyan")
    table.add_column("Path", style="dim")
    table.add_column("Status", style="green")

    for name, path in vaults.items():
        status = "Active" if (path / ".alcuin").is_dir() else "Missing .alcuin/"
        table.add_row(name, str(path), status)

    console.print(table)


@vault.command(name="register", short_help="Register an existing folder as a vault")
@click.argument("path", type=click.Path(exists=True, file_okay=False))
@click.option("--name", help="Vault name")
def vault_register(path: str, name: Optional[str]):
    """Register an existing vault into the system registry."""
    p = Path(path).resolve()
    v_name = name or p.name
    VaultRegistry.register(v_name, p)
    console.print(f"[bold green]Registered vault '[bold]{v_name}[/bold]' at {p}[/bold green]")


@vault.command(name="unregister", short_help="Remove a vault from registry")
@click.argument("name")
def vault_unregister(name: str):
    """Remove a vault from the system registry (local files are preserved)."""
    if VaultRegistry.unregister(name):
        console.print(f"[bold green]Unregistered vault '[bold]{name}[/bold]'. Local files preserved.[/bold green]")
    else:
        console.print(f"[bold red]Vault '{name}' not found in registry.[/bold red]")


@cli.command(short_help="Scan and incrementally index vault documents")
@click.option("-v", "--vault", "vault_target", help="Vault name or directory path")
@click.option("--force", is_flag=True, help="Force re-indexing of all documents")
def index(vault_target: Optional[str], force: bool):
    """Trigger manual indexing sync for a vault."""
    vault_obj = resolve_or_prompt_vault(target=vault_target)
    console.print(f"[bold cyan]Indexing vault '[bold]{vault_obj.name}[/bold]'...[/bold cyan]")
    storage = VaultStorage(vault_obj)

    report = storage.index(force=force)
    console.print(
        f"[bold green]Sync Complete ({report.duration_seconds}s):[/bold green]\n"
        f"  • Added: {len(report.added_files)}\n"
        f"  • Modified: {len(report.modified_files)}\n"
        f"  • Deleted: {len(report.deleted_files)}\n"
        f"  • Unchanged: {len(report.unchanged_files)}\n"
        f"  • Total Chunks Upserted: {report.total_chunks_indexed}"
    )


@cli.command(short_help="Start FastAPI Web UI and background periodic indexing daemon")
@click.option("--host", default="127.0.0.1", help="Server host address")
@click.option("--port", default=8000, type=int, help="Server port")
def serve(host: str, port: int):
    """Start the Alcuin Web UI dashboard and periodic indexing daemon."""
    import uvicorn
    console.print(f"[bold green]Starting Alcuin Server at http://{host}:{port}[/bold green]")
    console.print("[dim]Background periodic indexing daemon active. Press Ctrl+C to stop.[/dim]")
    uvicorn.run("alcuin.api.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    cli()

