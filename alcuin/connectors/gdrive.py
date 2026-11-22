"""Google Drive folder sync & export connector (SPEC-040)."""

from __future__ import annotations

import io
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_MIME = "application/vnd.google-apps.folder"
SHORTCUT_MIME = "application/vnd.google-apps.shortcut"

GOOGLE_EXPORT_MAP: Dict[str, Tuple[str, str]] = {
    "application/vnd.google-apps.document": ("text/markdown", ".md"),
    "application/vnd.google-apps.spreadsheet": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.presentation": ("application/pdf", ".pdf"),
}

SUPPORTED_BINARY_EXTS: Set[str] = {".pdf", ".docx", ".md", ".txt", ".rfc"}
_ILLEGAL_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass
class GDriveSyncReport:
    """Summary of a Google Drive folder sync run."""

    downloaded: List[str] = field(default_factory=list)
    updated: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    pruned: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def sanitize_name(name: str) -> str:
    """Replace OS-illegal filename characters, strip trailing dots/spaces, and neutralize reserved dot-dirs."""
    cleaned = _ILLEGAL_PATH_CHARS.sub("_", name).strip().rstrip(".")
    if not cleaned or cleaned in (".", ".."):
        return "untitled"
    if cleaned.lower() in (".alcuin", ".git"):
        cleaned = f"_{cleaned.lstrip('.')}"
    return cleaned


def default_token_path() -> Path:
    """Return the default OAuth token cache path outside any vault."""
    return Path.home() / ".alcuin" / "gdrive_token.json"


def default_credentials_path() -> Path:
    """Return the default OAuth client_secret.json path outside any vault."""
    return Path.home() / ".alcuin" / "gdrive_credentials.json"


def get_drive_service(credentials_path: Path, token_path: Path) -> Any:
    """Authenticate read-only via Desktop OAuth2 and return a Drive v3 service."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds: Optional[Credentials] = None
    if token_path.is_file():
        creds = Credentials.from_authorized_user_file(str(token_path), DRIVE_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path.is_file():
                raise FileNotFoundError(
                    f"Google OAuth credentials file not found: {credentials_path}. "
                    "Download a Desktop OAuth client_secret JSON from Google Cloud Console "
                    "and pass --credentials <path>."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), DRIVE_SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(token_path.parent, 0o700)
        except OSError:
            pass
        tmp_token = token_path.with_suffix(".json.tmp")
        fd = os.open(str(tmp_token), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(creds.to_json())
        except Exception:
            tmp_token.unlink(missing_ok=True)
            raise
        os.replace(tmp_token, token_path)
        try:
            os.chmod(token_path, 0o600)
        except OSError:
            pass

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _list_folder_children(service: Any, folder_id: str) -> List[Dict[str, Any]]:
    """List all non-trashed children inside a Google Drive folder."""
    items: List[Dict[str, Any]] = []
    page_token: Optional[str] = None
    query = f"'{folder_id}' in parents and trashed = false"
    while True:
        resp = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime, md5Checksum)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                pageSize=1000,
                pageToken=page_token,
            )
            .execute()
        )
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def _download_request_to_file(request: Any, dest: Path) -> None:
    """Stream a googleapiclient media/export request to disk atomically."""
    from googleapiclient.http import MediaIoBaseDownload

    dest.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    tmp_path = dest.with_name(f".{dest.name}.tmp")
    tmp_path.write_bytes(buf.getvalue())
    tmp_path.replace(dest)


def _resolve_target_path(
    rel_dir: Path,
    raw_name: str,
    mime_type: str,
    file_id: str,
    claimed_paths: Set[str],
) -> Optional[Tuple[Path, Optional[str]]]:
    """Determine (relative_local_path, export_mime_or_None) or None if unsupported."""
    safe = sanitize_name(raw_name)
    if mime_type in GOOGLE_EXPORT_MAP:
        export_mime, target_ext = GOOGLE_EXPORT_MAP[mime_type]
        stem = Path(safe).stem if Path(safe).suffix.lower() == target_ext else safe
        candidate = rel_dir / f"{stem}{target_ext}"
    elif mime_type.startswith("application/vnd.google-apps."):
        return None
    else:
        ext = Path(safe).suffix.lower()
        if ext not in SUPPORTED_BINARY_EXTS:
            return None
        export_mime = None
        candidate = rel_dir / safe

    rel_key = candidate.as_posix().lower()
    if rel_key in claimed_paths:
        candidate = candidate.with_name(f"{candidate.stem}_{file_id[:6]}{candidate.suffix}")
        rel_key = candidate.as_posix().lower()
    claimed_paths.add(rel_key)
    return candidate, export_mime


def sync_gdrive_folder(
    service: Any,
    folder_id: str,
    vault_dir: Path,
    prune: bool = False,
) -> GDriveSyncReport:
    """Recursively mirror & export a Google Drive folder into a local Alcuin vault."""
    vault_dir = Path(vault_dir)
    vault_dir.mkdir(parents=True, exist_ok=True)
    state_path = vault_dir / ".alcuin" / "gdrive_sync_state.json"

    prev_state: Dict[str, Dict[str, str]] = {}
    if state_path.is_file():
        try:
            prev_state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not parse %s, starting fresh: %s", state_path, exc)

    report = GDriveSyncReport()
    new_state: Dict[str, Dict[str, str]] = {}
    claimed_paths: Set[str] = set()
    visited_folders: Set[str] = set()

    def _walk(cur_folder_id: str, rel_dir: Path) -> None:
        if cur_folder_id in visited_folders:
            return
        visited_folders.add(cur_folder_id)

        for item in _list_folder_children(service, cur_folder_id):
            fid = item["id"]
            name = item.get("name", "untitled")
            mime = item.get("mimeType", "")
            mod_time = item.get("modifiedTime", "")
            md5 = item.get("md5Checksum", "")

            if mime == FOLDER_MIME:
                _walk(fid, rel_dir / sanitize_name(name))
                continue
            if mime == SHORTCUT_MIME:
                report.skipped.append(f"{(rel_dir / name).as_posix()} (shortcut)")
                continue

            resolved = _resolve_target_path(rel_dir, name, mime, fid, claimed_paths)
            if resolved is None:
                report.skipped.append((rel_dir / name).as_posix())
                continue

            rel_path, export_mime = resolved
            vault_root = vault_dir.resolve()
            alcuin_meta = (vault_root / ".alcuin").resolve()
            dest = (vault_root / rel_path).resolve()
            if not dest.is_relative_to(vault_root) or dest.is_relative_to(alcuin_meta):
                report.errors.append(f"Blocked unsafe target path: {rel_path.as_posix()}")
                continue
            rel_str = rel_path.as_posix()

            prior = prev_state.get(fid)
            if (
                prior
                and prior.get("modifiedTime") == mod_time
                and prior.get("local_path") == rel_str
                and dest.is_file()
            ):
                new_state[fid] = prior
                report.unchanged.append(rel_str)
                continue

            try:
                if export_mime:
                    req = service.files().export_media(fileId=fid, mimeType=export_mime)
                else:
                    req = service.files().get_media(fileId=fid, supportsAllDrives=True)
                _download_request_to_file(req, dest)
                new_state[fid] = {
                    "modifiedTime": mod_time,
                    "md5Checksum": md5,
                    "local_path": rel_str,
                    "mimeType": mime,
                }
                if prior:
                    report.updated.append(rel_str)
                else:
                    report.downloaded.append(rel_str)
            except Exception as exc:
                logger.warning("Failed syncing Drive file %s (%s): %s", name, fid, exc)
                report.errors.append(f"{rel_str}: {exc}")
                if prior:
                    new_state[fid] = prior

    _walk(folder_id, Path("."))

    if prune:
        vault_root = vault_dir.resolve()
        alcuin_meta = (vault_root / ".alcuin").resolve()
        active_ids = set(new_state.keys())
        for old_id, info in prev_state.items():
            if old_id not in active_ids:
                old_rel = info.get("local_path")
                if old_rel:
                    old_file = (vault_root / old_rel).resolve()
                    if (
                        old_file.is_relative_to(vault_root)
                        and not old_file.is_relative_to(alcuin_meta)
                        and old_file.is_file()
                    ):
                        old_file.unlink()
                        report.pruned.append(old_rel)

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(new_state, indent=2), encoding="utf-8")
    return report
