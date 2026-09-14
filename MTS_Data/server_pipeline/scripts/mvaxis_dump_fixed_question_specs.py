from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.question_provider import fixed_question_specs, grouped_question_counts
from src.mvaxis.utils import relative_to_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/runs/fixed_question_specs_v2/questions_90.json")
    args = parser.parse_args()

    output = Path(relative_to_root(ROOT, args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    specs = fixed_question_specs()
    output.write_text(
        json.dumps([spec.__dict__ for spec in specs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "num_questions": len(specs),
                "question_counts": grouped_question_counts(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
