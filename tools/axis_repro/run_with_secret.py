"""Run a subprocess with a DeepSeek credential injected only into its environment."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path


def extract_key(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    patterns = [
        r"(?is)deepseek.{0,160}?(?:api[_ -]?key|key).{0,30}?[:=]\s*[`'\"]?([A-Za-z0-9_-]{20,})",
        r"(?i)(sk-[A-Za-z0-9_-]{16,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    raise RuntimeError("No DeepSeek credential pattern found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credential-file", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise ValueError("missing subprocess command")
    environment = os.environ.copy()
    environment["DEEPSEEK_API_KEY"] = extract_key(args.credential_file)
    raise SystemExit(subprocess.run(command, env=environment, check=False).returncode)


if __name__ == "__main__":
    main()
