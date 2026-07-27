"""Prepare, assemble, and analyze Round-4 OE component experiments."""

from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src import (
    round1_pipeline,
)
from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src.round4_selection import (
    ACTIVE_FAMILY,
    prompt_changed,
)


# Round 1's pipeline is deliberately generic apart from these two globals.
# Rebind them before delegating so its integrity checks and reporting operate
# over exactly the preregistered Round-4 modes.
round1_pipeline.ACTIVE_FAMILY = ACTIVE_FAMILY
round1_pipeline.prompt_changed = prompt_changed


if __name__ == "__main__":
    round1_pipeline.main()
