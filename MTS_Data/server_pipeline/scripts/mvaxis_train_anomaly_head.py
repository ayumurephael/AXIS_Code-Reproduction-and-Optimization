from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("train_axis_interval_proposal.py")), run_name="__main__")
