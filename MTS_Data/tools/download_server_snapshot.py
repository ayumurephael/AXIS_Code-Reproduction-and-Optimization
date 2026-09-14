"""Download the code-only snapshot of the multiaxis generation pipeline.

Credentials are read at runtime from THU_IE_GPU.md.  They are never written to
the snapshot or its manifest.  The remote side is accessed through SFTP only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import paramiko


ALLOWED_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
ALLOWED_NAMES = {"LICENSE", "Makefile"}
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}


def parse_credentials(path: Path) -> tuple[str, str, str]:
    text = path.read_text(encoding="utf-8")

    def value(label: str) -> str:
        match = re.search(rf"^-\s*{re.escape(label)}[：:]\s*`?([^`\s]+)`?\s*$", text, re.M)
        if not match:
            raise ValueError(f"Cannot find {label!r} in {path}")
        return match.group(1)

    account_block = re.search(
        r"^-\s*账号[：:]\s*`?([^`\s]+)`?\s*\r?\n"
        r"^-\s*密码[：:]\s*`?([^`\s]+)`?\s*$",
        text,
        re.M,
    )
    if account_block:
        username, password = account_block.groups()
    else:
        username, password = value("用户名"), value("密码")
    return value("服务器地址"), username, password


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def should_copy(relative: PurePosixPath) -> bool:
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    return relative.suffix.lower() in ALLOWED_SUFFIXES or relative.name in ALLOWED_NAMES


def walk_files(sftp: paramiko.SFTPClient, root: PurePosixPath):
    pending = [root]
    while pending:
        current = pending.pop()
        try:
            entries = sftp.listdir_attr(str(current))
        except OSError as exc:
            print(f"warning: cannot list {current}: {exc}")
            continue
        for entry in entries:
            remote = current / entry.filename
            if stat.S_ISDIR(entry.st_mode):
                if entry.filename not in EXCLUDED_PARTS:
                    pending.append(remote)
            elif stat.S_ISREG(entry.st_mode):
                yield remote, entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credential-file", type=Path, required=True)
    parser.add_argument("--host-port", type=int, default=2228)
    parser.add_argument("--remote-root", default="/home/zhangchen/axis_project_server3/multiaxis_runtime_train")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    host, username, password = parse_credentials(args.credential_file)
    args.output.mkdir(parents=True, exist_ok=True)
    roots = ("configs", "scripts", "src", "legacy/AXIS_repo")
    manifest_files: list[dict[str, object]] = []

    transport = paramiko.Transport((host, args.host_port))
    transport.connect(username=username, password=password)
    try:
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            for subtree in roots:
                remote_subtree = PurePosixPath(args.remote_root) / subtree
                try:
                    candidates = walk_files(sftp, remote_subtree)
                    for remote, attrs in candidates:
                        relative = remote.relative_to(PurePosixPath(args.remote_root))
                        if not should_copy(relative):
                            continue
                        local = args.output.joinpath(*relative.parts)
                        local.parent.mkdir(parents=True, exist_ok=True)
                        sftp.get(str(remote), str(local))
                        manifest_files.append(
                            {
                                "path": relative.as_posix(),
                                "size": attrs.st_size,
                                "remote_mtime": attrs.st_mtime,
                                "sha256": sha256(local),
                            }
                        )
                except OSError as exc:
                    print(f"warning: remote subtree unavailable: {remote_subtree}: {exc}")
    finally:
        transport.close()

    manifest = {
        "source_host_port": args.host_port,
        "source_root": args.remote_root,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": list(roots),
        "file_count": len(manifest_files),
        "files": sorted(manifest_files, key=lambda item: str(item["path"])),
    }
    manifest_path = args.output / "SOURCE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"downloaded {len(manifest_files)} files")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
