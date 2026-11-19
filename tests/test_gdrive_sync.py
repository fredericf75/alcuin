"""Unit tests for SPEC-040 Google Drive folder sync & export connector."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

from click.testing import CliRunner

from alcuin.cli.main import cli
from alcuin.connectors.gdrive import (
    FOLDER_MIME,
    SHORTCUT_MIME,
    sanitize_name,
    sync_gdrive_folder,
)


class _FakeRequest:
    def __init__(self, payload: bytes):
        self.payload = payload

    def execute(self):
        return self.payload


class _FakeMediaDownload:
    def __init__(self, fh: io.BytesIO, request: _FakeRequest):
        self.fh = fh
        self.request = request
        self._done = False

    def next_chunk(self):
        self.fh.write(self.request.payload)
        self._done = True
        return None, True


class _FakeFilesResource:
    def __init__(self, tree: Dict[str, List[Dict[str, Any]]], payloads: Dict[str, bytes]):
        self.tree = tree
        self.payloads = payloads
        self.export_calls: List[tuple] = []
        self.media_calls: List[str] = []

    def list(self, q: str, **kwargs):
        folder_id = q.split("'")[1]
        return _FakeRequest({"files": list(self.tree.get(folder_id, []))})

    def export_media(self, fileId: str, mimeType: str):
        self.export_calls.append((fileId, mimeType))
        return _FakeRequest(self.payloads[fileId])

    def get_media(self, fileId: str, **kwargs):
        self.media_calls.append(fileId)
        return _FakeRequest(self.payloads[fileId])


class _FakeDriveService:
    def __init__(self, tree: Dict[str, List[Dict[str, Any]]], payloads: Dict[str, bytes]):
        self._files = _FakeFilesResource(tree, payloads)

    def files(self):
        return self._files


def test_sanitize_name():
    assert sanitize_name('Taxes: 2025/2026? "Final".') == "Taxes_ 2025_2026_ _Final_"
    assert sanitize_name("...") == "untitled"


def test_sync_gdrive_folder_recursive_export_incremental_and_prune(tmp_path, monkeypatch):
    import googleapiclient.http as ghttp

    monkeypatch.setattr(ghttp, "MediaIoBaseDownload", _FakeMediaDownload)

    tree = {
        "root": [
            {
                "id": "doc1",
                "name": "Family Emergency Plan",
                "mimeType": "application/vnd.google-apps.document",
                "modifiedTime": "2026-11-18T10:00:00Z",
            },
            {
                "id": "sheet1",
                "name": "Household Budget",
                "mimeType": "application/vnd.google-apps.spreadsheet",
                "modifiedTime": "2026-11-18T10:00:00Z",
            },
            {
                "id": "img1",
                "name": "photo.png",
                "mimeType": "image/png",
                "modifiedTime": "2026-11-18T10:00:00Z",
            },
            {
                "id": "short1",
                "name": "Link",
                "mimeType": SHORTCUT_MIME,
                "modifiedTime": "2026-11-18T10:00:00Z",
            },
            {
                "id": "sub1",
                "name": "Medical: Records",
                "mimeType": FOLDER_MIME,
                "modifiedTime": "2026-11-18T10:00:00Z",
            },
        ],
        "sub1": [
            {
                "id": "pdf1",
                "name": "Insurance.pdf",
                "mimeType": "application/pdf",
                "modifiedTime": "2026-11-18T10:00:00Z",
            },
            {
                "id": "doc2_dup1",
                "name": "Notes",
                "mimeType": "application/vnd.google-apps.document",
                "modifiedTime": "2026-11-18T10:00:00Z",
            },
            {
                "id": "doc2_dup2",
                "name": "Notes",
                "mimeType": "application/vnd.google-apps.document",
                "modifiedTime": "2026-11-18T10:00:00Z",
            },
        ],
    }
    payloads = {
        "doc1": b"# Emergency Plan\nCall 911.",
        "sheet1": b"%PDF-1.4 sheet",
        "pdf1": b"%PDF-1.4 insurance",
        "doc2_dup1": b"Note A",
        "doc2_dup2": b"Note B",
    }

    service = _FakeDriveService(tree, payloads)
    vault_dir = tmp_path / "family_vault"

    # Initial sync
    r1 = sync_gdrive_folder(service, "root", vault_dir)
    assert len(r1.downloaded) == 5
    assert len(r1.skipped) == 2
    assert (vault_dir / "Family Emergency Plan.md").read_text(encoding="utf-8") == "# Emergency Plan\nCall 911."
    assert (vault_dir / "Household Budget.pdf").read_bytes() == b"%PDF-1.4 sheet"
    assert (vault_dir / "Medical_ Records" / "Insurance.pdf").read_bytes() == b"%PDF-1.4 insurance"
    assert (vault_dir / "Medical_ Records" / "Notes.md").read_text(encoding="utf-8") == "Note A"
    assert (vault_dir / "Medical_ Records" / "Notes_doc2_d.md").read_text(encoding="utf-8") == "Note B"

    # Second sync with no changes -> all 5 unchanged
    service2 = _FakeDriveService(tree, payloads)
    r2 = sync_gdrive_folder(service2, "root", vault_dir)
    assert len(r2.unchanged) == 5
    assert r2.downloaded == []
    assert service2._files.export_calls == []
    assert service2._files.media_calls == []

    # Modify doc1 and delete doc2_dup2 with prune=True
    tree["root"][0]["modifiedTime"] = "2026-11-18T11:00:00Z"
    payloads["doc1"] = b"# Updated Emergency Plan"
    tree["sub1"] = [x for x in tree["sub1"] if x["id"] != "doc2_dup2"]

    service3 = _FakeDriveService(tree, payloads)
    r3 = sync_gdrive_folder(service3, "root", vault_dir, prune=True)
    assert r3.updated == ["Family Emergency Plan.md"]
    assert len(r3.unchanged) == 3
    assert r3.pruned == ["Medical_ Records/Notes_doc2_d.md"]
    assert not (vault_dir / "Medical_ Records" / "Notes_doc2_d.md").exists()
    assert (vault_dir / "Family Emergency Plan.md").read_text(encoding="utf-8") == "# Updated Emergency Plan"


def test_cli_gdrive_sync_help():
    runner = CliRunner()
    res = runner.invoke(cli, ["gdrive-sync", "--help"])
    assert res.exit_code == 0
    assert "--folder-id" in res.output
    assert "--index" in res.output
