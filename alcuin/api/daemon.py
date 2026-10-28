"""
alcuin.api.daemon - Background Periodic Indexing and Inbox Intake Daemon for Alcuin Vaults
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from alcuin.storage.engine import VaultStorage
from alcuin.storage.ingest import InboxIngestionEngine
from alcuin.storage.vault import Vault, VaultRegistry

logger = logging.getLogger("alcuin.daemon")


class PeriodicIndexer:
    """Monitors registered vaults in the background for both document intake and incremental indexing."""

    def __init__(self, check_interval: float = 5.0):
        self.check_interval = check_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_indexed: Dict[str, float] = {}
        self._last_inbox_check: Dict[str, float] = {}
        # Tracks (size, mtime) per candidate file to ensure write stability before intake
        self._file_stats: Dict[str, Tuple[int, float]] = {}
        self._lock = asyncio.Lock()

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Background indexing & intake daemon started.")

    async def stop(self):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Background indexing & intake daemon stopped.")

    async def _run_loop(self):
        while self._running:
            try:
                await self._check_vaults()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background daemon loop: {e}")

            await asyncio.sleep(self.check_interval)

    async def _check_vaults(self):
        vaults = VaultRegistry.list_vaults()
        now = time.time()

        for name, path in vaults.items():
            if not (path / ".alcuin").is_dir():
                continue

            try:
                vault = Vault.load(path)

                # 1. Automated Background Inbox Intake (SPEC-020)
                if vault.config.auto_ingest_inbox:
                    last_inbox = self._last_inbox_check.get(name, 0.0)
                    inbox_interval = vault.config.inbox_check_interval_seconds
                    if (now - last_inbox) >= inbox_interval:
                        self._last_inbox_check[name] = now
                        await self._process_vault_inbox(name, vault)

                # 2. Periodic LanceDB / BM25 Incremental Indexing
                if vault.config.periodic_indexing:
                    last_sync = self._last_indexed.get(name, 0.0)
                    interval = vault.config.index_interval_seconds

                    if (now - last_sync) >= interval:
                        async with self._lock:
                            storage = VaultStorage(vault)
                            report = await asyncio.to_thread(storage.index)
                            self._last_indexed[name] = now
                            if report.has_changes:
                                logger.info(
                                    f"Periodic sync for '{name}': +{len(report.added_files)} "
                                    f"~{len(report.modified_files)} -{len(report.deleted_files)} "
                                    f"({report.total_chunks_indexed} chunks in {report.duration_seconds}s)"
                                )
            except Exception as e:
                logger.warning(f"Failed background processing for vault '{name}': {e}")

    async def _process_vault_inbox(self, name: str, vault: Vault):
        """Probe inbox folder, ensure write stability, and process eligible documents."""
        inbox_dir = vault.root_path / vault.config.inbox_dir
        if not inbox_dir.is_dir():
            return

        try:
            candidates = [
                p
                for p in inbox_dir.iterdir()
                if p.is_file() and not p.name.startswith(".") and p.name != "_rejected"
            ]
        except OSError:
            return

        if not candidates:
            return

        stable_files = []
        for p in candidates:
            key = str(p)
            try:
                st = p.stat()
                curr_stat = (st.st_size, st.st_mtime)
                prev_stat = self._file_stats.get(key)
                if prev_stat is not None and prev_stat == curr_stat:
                    stable_files.append(p)
                else:
                    self._file_stats[key] = curr_stat
            except OSError:
                pass

        if stable_files:
            async with self._lock:
                engine = InboxIngestionEngine(vault)
                result = await asyncio.to_thread(engine.ingest_inbox)
                if result.ingested_notes:
                    logger.info(
                        f"Auto-ingested {len(result.ingested_notes)} document(s) in vault '{name}'"
                    )
                if result.rejected_files:
                    logger.info(
                        f"Quarantined {len(result.rejected_files)} unsupported file(s) in vault '{name}/inbox/_rejected'"
                    )

            # Cleanup stats for files no longer in inbox
            for sf in stable_files:
                self._file_stats.pop(str(sf), None)
