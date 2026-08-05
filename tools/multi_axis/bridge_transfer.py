from __future__ import annotations

import argparse
import json
import os
import posixpath
import stat
import time
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Optional, Tuple

import paramiko


def connect(host: str, port: int, username: str, password: str):
    transport = paramiko.Transport(
        (host, port),
        default_window_size=128 * 1024 * 1024,
        default_max_packet_size=1024 * 1024,
    )
    transport.use_compression(True)
    transport.connect(username=username, password=password)
    return transport, paramiko.SFTPClient.from_transport(transport)


def walk(sftp, root: str, relative: str, allowed: Optional[set[str]]):
    todo = [relative]
    while todo:
        current = todo.pop()
        for item in sftp.listdir_attr(posixpath.join(root, current)):
            child = posixpath.join(current, item.filename)
            if stat.S_ISDIR(item.st_mode):
                todo.append(child)
            elif stat.S_ISREG(item.st_mode) and (allowed is None or item.filename in allowed):
                yield child, int(item.st_size)


def ensure_remote_directory(sftp, path: str) -> None:
    parts = PurePosixPath(path).parts
    current = "/" if path.startswith("/") else ""
    for part in parts:
        if part == "/":
            continue
        current = posixpath.join(current, part)
        try:
            sftp.stat(current)
        except IOError:
            sftp.mkdir(current)


def inventory(sftp, source_root: str, selection: dict) -> Dict[str, int]:
    files: Dict[str, int] = {}
    for relative in selection["exact_files"]:
        files[relative] = int(sftp.stat(posixpath.join(source_root, relative)).st_size)
    for tree in selection["trees"]:
        allowed = None if tree["basenames"] is None else set(tree["basenames"])
        for relative, size in walk(sftp, source_root, tree["path"], allowed):
            files[relative] = size
    return files


def transfer_file(source, target, source_path: str, target_path: str, size: int) -> str:
    ensure_remote_directory(target, posixpath.dirname(target_path))
    try:
        if int(target.stat(target_path).st_size) == size:
            return "skip"
    except IOError:
        pass
    part_path = target_path + ".part"
    try:
        offset = int(target.stat(part_path).st_size)
    except IOError:
        offset = 0
    if offset > size:
        target.remove(part_path)
        offset = 0
    with source.open(source_path, "rb") as reader, target.open(
        part_path, "ab" if offset else "wb"
    ) as writer:
        if offset:
            reader.seek(offset)
        try:
            reader.prefetch(file_size=size, max_concurrent_requests=64)
        except (AttributeError, TypeError):
            reader.prefetch(file_size=size)
        if hasattr(writer, "set_pipelined"):
            writer.set_pipelined(True)
        copied = offset
        last_report = time.monotonic()
        while copied < size:
            block = reader.read(min(8 * 1024 * 1024, size - copied))
            if not block:
                raise IOError(f"Unexpected EOF at {copied}/{size}")
            writer.write(block)
            copied += len(block)
            if time.monotonic() - last_report >= 30:
                print(json.dumps({"event": "progress", "file": source_path, "bytes": copied, "size": size}), flush=True)
                last_report = time.monotonic()
    target.rename(part_path, target_path)
    return "copied"


def main() -> None:
    parser = argparse.ArgumentParser(description="Credential-safe, resumable two-SFTP bridge")
    parser.add_argument("--selection", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--target-directory", default="multi-axis-assets")
    parser.add_argument("--source-port", type=int, default=2228)
    parser.add_argument("--target-port", type=int, default=2229)
    parser.add_argument("--username", default="zhangchen")
    args = parser.parse_args()
    host = os.environ.get("AXIS_GPU_HOST")
    password = os.environ.pop("AXIS_GPU_PASSWORD", None)
    if not host or not password:
        raise RuntimeError("AXIS_GPU_HOST and AXIS_GPU_PASSWORD must be provided through the environment")
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    source_transport, source = connect(host, args.source_port, args.username, password)
    target_transport, target = connect(host, args.target_port, args.username, password)
    try:
        target_root = posixpath.join(target.normalize("."), args.target_directory)
        ensure_remote_directory(target, target_root)
        files = inventory(source, args.source_root, selection)
        print(json.dumps({"event": "inventory", "files": len(files), "bytes": sum(files.values())}), flush=True)
        records = []
        for index, (relative, size) in enumerate(sorted(files.items()), 1):
            source_path = posixpath.join(args.source_root, relative)
            target_path = posixpath.join(target_root, relative)
            for attempt in range(1, 6):
                try:
                    action = transfer_file(source, target, source_path, target_path, size)
                    break
                except Exception as exc:
                    print(json.dumps({"event": "retry", "file": relative, "attempt": attempt, "error": type(exc).__name__}), flush=True)
                    try:
                        source.close(); target.close(); source_transport.close(); target_transport.close()
                    except Exception:
                        pass
                    time.sleep(min(30, 2 ** attempt))
                    source_transport, source = connect(host, args.source_port, args.username, password)
                    target_transport, target = connect(host, args.target_port, args.username, password)
            else:
                raise RuntimeError(f"Transfer failed after retries: {relative}")
            records.append({"path": relative, "size": size, "action": action})
            print(json.dumps({"event": "file_done", "i": index, "n": len(files), "file": relative, "size": size, "action": action}), flush=True)
        manifest = json.dumps({"completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": records, "total_bytes": sum(x["size"] for x in records)}, indent=2).encode()
        with target.open(posixpath.join(target_root, "transfer_manifest.json"), "wb") as handle:
            handle.write(manifest)
        print(json.dumps({"event": "complete", "files": len(records), "bytes": sum(x["size"] for x in records)}), flush=True)
    finally:
        source.close(); target.close(); source_transport.close(); target_transport.close()


if __name__ == "__main__":
    main()
