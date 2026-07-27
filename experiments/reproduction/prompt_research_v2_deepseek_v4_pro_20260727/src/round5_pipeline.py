"""Prepare, assemble, and analyze Round-5 OE refinement experiments."""

from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src import (
    round1_pipeline,
)
from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src.round5_selection import (
    ACTIVE_FAMILY,
    prompt_changed,
)


round1_pipeline.ACTIVE_FAMILY = ACTIVE_FAMILY
round1_pipeline.prompt_changed = prompt_changed


if __name__ == "__main__":
    round1_pipeline.main()
