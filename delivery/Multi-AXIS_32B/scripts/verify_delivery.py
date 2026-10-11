from __future__ import annotations

import sys
from pathlib import Path


delivery_root = Path(__file__).resolve().parents[1]
source_root = delivery_root / "source" / "checkout"
if not source_root.is_dir():
    raise RuntimeError(f"Source checkout is missing: {source_root}")
sys.path.insert(0, str(source_root))

from tools.multi_axis.delivery_manifest import main


if __name__ == "__main__":
    main()
