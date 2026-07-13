"""Stable CLI wrapper; injects append-only JSONL IO into the copied common module."""
from . import common
from .io_utils import append_jsonl

common.append_jsonl = append_jsonl
from .run_inference import main  # noqa: E402

if __name__ == "__main__":
    main()
