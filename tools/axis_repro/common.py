import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class QARecord:
    record_id: str
    series_file: str
    series_index: int
    window_index: int
    sample_id: Any
    time_series: List[float]
    question: str
    answer: str
    question_type: str
    start_index: int
    end_index: int
    has_anomaly: Optional[bool] = None


def load_axis_records(
    dataset_dir: str,
    subset: str = "full",
    seed: int = 42,
    paper_train_ratio: float = 0.02,
    paper_series_limit: int = 70,
    max_records: Optional[int] = None,
) -> List[QARecord]:
    """Load AXIS QA records.

    subset="full" loads all series in sorted order.
    subset="paper140" mimics AXIS_test.py: shuffle with seed, drop the
    first int(n * 0.02) files, then evaluate the first 70 series.
    """
    root = Path(dataset_dir)
    series_dir = root / "series"
    files = sorted(series_dir.glob("series_*.json"))
    if not files:
        raise FileNotFoundError(f"No series_*.json files found under {series_dir}")

    if subset == "paper140":
        files = files[:]
        random.Random(seed).shuffle(files)
        split_idx = int(len(files) * paper_train_ratio)
        files = files[split_idx : split_idx + paper_series_limit]
    elif subset == "full":
        pass
    else:
        raise ValueError(f"Unknown subset: {subset}")

    records: List[QARecord] = []
    for series_index, path in enumerate(files):
        data = json.loads(path.read_text(encoding="utf-8"))
        windows = data.get("windows", [])
        for window_index, item in enumerate(windows):
            window_range = item.get("window_range", {})
            records.append(
                QARecord(
                    record_id=f"{path.stem}:{window_index}",
                    series_file=path.name,
                    series_index=series_index,
                    window_index=window_index,
                    sample_id=data.get("sample_id"),
                    time_series=data["original_data"]["time_series"],
                    question=item.get("question", ""),
                    answer=item.get("answer", ""),
                    question_type=item.get("question_type", "unknown"),
                    start_index=int(window_range.get("start", 0)),
                    end_index=int(window_range.get("end", 0)),
                    has_anomaly=item.get("has_anomaly"),
                )
            )
            if max_records is not None and len(records) >= max_records:
                return records
    return records


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9.+\-/%]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9.+\-/%]+", normalize_text(text))


def token_f1(prediction: str, reference: str) -> float:
    pred = tokenize(prediction)
    ref = tokenize(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    pred_counts = Counter(pred)
    ref_counts = Counter(ref)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred = tokenize(prediction)
    ref = tokenize(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    lcs = _lcs_len(pred, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def extract_choice(text: str) -> Optional[str]:
    if not text:
        return None
    stripped = text.strip()
    match = re.match(r"^\s*([A-D])\s*[\).:\-]", stripped, flags=re.I)
    if match:
        return match.group(1).upper()
    patterns = [
        r"\b(?:answer|option|choice)\s*(?:is|:)?\s*([A-D])\b",
        r"\b([A-D])\s*(?:is correct|is the correct|seems correct)\b",
        r"\bthe correct answer is\s*([A-D])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, stripped, flags=re.I)
        if match:
            return match.group(1).upper()
    # Last resort: use the first isolated option letter in short answers.
    if len(stripped) <= 20:
        match = re.search(r"\b([A-D])\b", stripped, flags=re.I)
        if match:
            return match.group(1).upper()
    return None


def extract_true_false(text: str) -> Optional[bool]:
    if not text:
        return None
    norm = normalize_text(text)
    patterns = [
        (r"\b(?:answer|statement|claim)\s*(?:is|:)?\s*true\b", True),
        (r"\b(?:answer|statement|claim)\s*(?:is|:)?\s*false\b", False),
        (r"^\s*true\b", True),
        (r"^\s*false\b", False),
        (r"\byes\b", True),
        (r"\bno\b", False),
        (r"\btrue\b", True),
        (r"\bfalse\b", False),
    ]
    for pattern, value in patterns:
        if re.search(pattern, norm):
            return value
    return None


def extract_numbers(text: str) -> List[str]:
    return re.findall(r"[-+]?\d+(?:\.\d+)?", text or "")


def numeric_f1(prediction: str, reference: str) -> Optional[float]:
    pred = extract_numbers(prediction)
    ref = extract_numbers(reference)
    if not pred and not ref:
        return None
    if not pred or not ref:
        return 0.0
    pred_counts = Counter(pred)
    ref_counts = Counter(ref)
    overlap = sum((pred_counts & ref_counts).values())
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_prediction(row: Dict[str, Any]) -> Dict[str, Any]:
    qtype = row.get("question_type", "unknown")
    expected = row.get("expected_answer") or row.get("answer") or ""
    generated = row.get("generated_response") or row.get("prediction") or ""
    scores: Dict[str, Any] = {
        "record_id": row.get("record_id"),
        "question_type": qtype,
        "loss": row.get("loss"),
    }

    if qtype == "multiple_choice":
        gold = extract_choice(expected)
        pred = extract_choice(generated)
        correct = None if gold is None else float(pred == gold)
        scores.update(
            {
                "gold_choice": gold,
                "pred_choice": pred,
                "choice_correct": correct,
                "primary_score": correct,
            }
        )
    elif qtype == "true_false":
        gold_tf = extract_true_false(expected)
        pred_tf = extract_true_false(generated)
        correct = None if gold_tf is None else float(pred_tf == gold_tf)
        scores.update(
            {
                "gold_true_false": gold_tf,
                "pred_true_false": pred_tf,
                "true_false_correct": correct,
                "primary_score": correct,
            }
        )
    else:
        tf1 = token_f1(generated, expected)
        rouge = rouge_l_f1(generated, expected)
        nf1 = numeric_f1(generated, expected)
        scores.update(
            {
                "token_f1": tf1,
                "rouge_l_f1": rouge,
                "numeric_f1": nf1,
                "length_ratio": len(tokenize(generated)) / max(len(tokenize(expected)), 1),
                "primary_score": 0.5 * tf1 + 0.5 * rouge,
            }
        )
    return scores


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return None if not clean else sum(clean) / len(clean)


def aggregate_scored_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row.get("question_type", "unknown")].append(row)

    per_type: Dict[str, Any] = {}
    for qtype, items in sorted(by_type.items()):
        metrics = {
            "count": len(items),
            "primary_score": mean(item.get("primary_score") for item in items),
            "loss": mean(item.get("loss") for item in items),
        }
        if qtype == "multiple_choice":
            metrics["choice_accuracy"] = mean(item.get("choice_correct") for item in items)
            metrics["unparsed"] = sum(1 for item in items if item.get("pred_choice") is None)
        elif qtype == "true_false":
            metrics["true_false_accuracy"] = mean(item.get("true_false_correct") for item in items)
            metrics["unparsed"] = sum(1 for item in items if item.get("pred_true_false") is None)
        else:
            metrics["token_f1"] = mean(item.get("token_f1") for item in items)
            metrics["rouge_l_f1"] = mean(item.get("rouge_l_f1") for item in items)
            metrics["numeric_f1"] = mean(item.get("numeric_f1") for item in items)
        per_type[qtype] = metrics

    macro_primary = mean(m["primary_score"] for m in per_type.values())
    return {
        "count": len(rows),
        "macro_primary_score": macro_primary,
        "loss": mean(row.get("loss") for row in rows),
        "per_type": per_type,
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    n_bootstrap: int = 1000,
    seed: int = 72,
    alpha: float = 0.05,
) -> Optional[Dict[str, float]]:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return None
    if len(clean) == 1:
        return {"mean": clean[0], "low": clean[0], "high": clean[0]}
    rng = random.Random(seed)
    samples = []
    for _ in range(n_bootstrap):
        draw = [clean[rng.randrange(len(clean))] for _ in clean]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    low_idx = max(0, int((alpha / 2) * len(samples)))
    high_idx = min(len(samples) - 1, int((1 - alpha / 2) * len(samples)))
    return {"mean": sum(clean) / len(clean), "low": samples[low_idx], "high": samples[high_idx]}


def record_to_dict(record: QARecord) -> Dict[str, Any]:
    return asdict(record)


