"""
alcuin.api.daemon - Background Periodic Indexing Daemon for Alcuin Vaults
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional

from alcuin.storage.engine import VaultStorage
from alcuin.storage.vault import Vault, VaultRegistry

logger = logging.getLogger("alcuin.daemon")


class PeriodicIndexer:
    """Monitors and incrementally indexes registered vaults in the background."""

    def __init__(self, check_interval: float = 10.0):
        self.check_interval = check_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_indexed: Dict[str, float] = {}

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Periodic indexing daemon started.")

    async def stop(self):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Periodic indexing daemon stopped.")

    async def _run_loop(self):
        while self._running:
            try:
                await self._check_vaults()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic indexing loop: {e}")

            await asyncio.sleep(self.check_interval)

    async def _check_vaults(self):
        vaults = VaultRegistry.list_vaults()
        now = time.time()

        for name, path in vaults.items():
            if not (path / ".alcuin").is_dir():
                continue

            try:
                vault = Vault.load(path)
                if not vault.config.periodic_indexing:
                    continue

                last_sync = self._last_indexed.get(name, 0.0)
                interval = vault.config.index_interval_seconds

                if (now - last_sync) >= interval:
                    # Run sync in thread to avoid blocking event loop
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
                logger.warning(f"Failed periodic index for vault '{name}': {e}")
