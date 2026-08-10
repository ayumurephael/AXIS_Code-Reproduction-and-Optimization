from __future__ import annotations

from typing import Iterable, Sequence


OFFICIAL_BIAS_NEUTRALIZED_DATASETS = (
    "478new",
    "SMD",
    "SWaT",
    "LEMMA-RCA",
    "VTA",
)
OPTIONAL_EVALUATION_VIEWS = ("478",)
SUPPORTED_EVALUATION_DATASETS = (
    "478new",
    "478",
    "SMD",
    "SWaT",
    "LEMMA-RCA",
    "VTA",
)

EXPECTED_DATASET_COUNTS = {
    "478new": 478,
    "478": 478,
    "SMD": 200,
    "SWaT": 184,
    "LEMMA-RCA": 12,
    "VTA": 200,
}
EXPECTED_TYPE_COUNTS = {
    "478new": {"MC": 176, "OE": 139, "TF": 163},
    "478": {"MC": 176, "OE": 139, "TF": 163},
    "SMD": {"MC": 66, "OE": 67, "TF": 67},
    "SWaT": {"MC": 61, "OE": 62, "TF": 61},
    "LEMMA-RCA": {"MC": 4, "OE": 4, "TF": 4},
    "VTA": {"MC": 67, "OE": 67, "TF": 66},
}
DATASET_DISPLAY_NAMES = {
    "478new": "Teacher-Eval 478new (bias-neutralized wording)",
    "478": "Teacher-Eval 478 (original wording)",
    "SMD": "SMD",
    "SWaT": "SWaT",
    "LEMMA-RCA": "LEMMA-RCA",
    "VTA": "VTA Articulary",
}


def validate_datasets(
    datasets: Sequence[str] | Iterable[str],
    *,
    allow_duplicate_views: bool = True,
) -> tuple[str, ...]:
    selected = tuple(datasets)
    if not selected:
        raise ValueError("At least one evaluation dataset is required")
    unknown = sorted(set(selected) - set(SUPPORTED_EVALUATION_DATASETS))
    if unknown:
        raise ValueError(f"Unsupported evaluation datasets: {unknown}")
    if len(selected) != len(set(selected)):
        raise ValueError("Evaluation dataset names must be unique")
    if not allow_duplicate_views and {"478", "478new"}.issubset(selected):
        raise ValueError(
            "478 and 478new are two wording views of the same 478 samples and "
            "must not both contribute to an independent-dataset macro average"
        )
    return selected
