from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .common import bootstrap_mean_ci, read_jsonl

ORDER = ["multiple_choice", "open_ended", "true_false"]
DIMS = {"multiple_choice": ["correctness", "reasoning_quality"],
        "open_ended": ["accuracy", "completeness", "relevance"],
        "true_false": ["correctness", "justification_quality"]}
LABEL = {"base": "AXIS", "wo_fixed_hint": "w/o-task-hint",
         "wo_local_hint": "w/o-context-hint", "wo_windows": "w/o-windows"}


def aggregate(rows):
    per_record = defaultdict(dict); meta = {}
    for x in rows:
        key = (x["record_id"], x["mode"]); per_record[key][x["dimension"]] = float(x["score"])
        meta[key] = x["question_type"]
    values = defaultdict(lambda: defaultdict(list)); record_final = {}
    for key, scores in per_record.items():
        qtype = meta[key]
        expected = DIMS[qtype]
        if any(d not in scores for d in expected): continue
        weights = {x["dimension"]: float(x["weight"]) for x in rows if (x["record_id"], x["mode"]) == key}
        final = sum(scores[d]*weights[d] for d in expected)
        values[key[1]][qtype+"/final"].append(final); record_final[key] = final
        for d in expected: values[key[1]][qtype+"/"+d].append(scores[d])
    result = {}
    for mode, metrics in values.items():
        result[mode] = {k: sum(v)/len(v) for k,v in metrics.items()}
        present = [result[mode][q+"/final"] for q in ORDER if q+"/final" in result[mode]]
        if present:
            result[mode]["macro_final"] = sum(present)/len(present)
    paired = {}
    for mode in result:
        if mode == "base": continue
        diffs = [record_final[(rid, mode)]-score for (rid,m),score in record_final.items()
                 if m == "base" and (rid, mode) in record_final]
        paired[mode] = bootstrap_mean_ci(diffs, n_bootstrap=10000, seed=72)
    return result, paired


def markdown(result, paired):
    heads = ["Model"]
    for q in ORDER: heads += [q+" Final"] + [q+" "+d for d in DIMS[q]]
    lines = ["| "+" | ".join(heads)+" |", "|"+"---|"*len(heads)]
    for mode in ["base", "wo_fixed_hint", "wo_local_hint", "wo_windows"]:
        if mode not in result: continue
        vals = [LABEL.get(mode, mode)]
        for q in ORDER: vals += [f"{result[mode][q+'/final']:.2f}"]+[f"{result[mode][q+'/'+d]:.2f}" for d in DIMS[q]]
        lines.append("| "+" | ".join(vals)+" |")
    lines += ["", "Paired difference vs AXIS (variant − AXIS), 95% bootstrap CI:"]
    for mode, ci in paired.items(): lines.append(f"- {LABEL.get(mode,mode)}: {ci['mean']:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]")
    return "\n".join(lines)+"\n"


def main():
    p=argparse.ArgumentParser(); p.add_argument("--scores", required=True); p.add_argument("--output-prefix", required=True); a=p.parse_args()
    result, paired = aggregate(read_jsonl(a.scores)); out=Path(a.output_prefix); out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"models": result, "paired_vs_axis": paired}, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(markdown(result, paired), encoding="utf-8")
    print(out.with_suffix(".md"))


if __name__ == "__main__": main()



