from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .common import bootstrap_mean_ci, read_jsonl
from .build_tables import DIMS, LABEL, ORDER, markdown


def aggregate(rows):
    per_record, meta, weights = defaultdict(dict), {}, defaultdict(dict)
    for x in rows:
        key=(x["record_id"],x["mode"]); per_record[key][x["dimension"]]=float(x["score"])
        weights[key][x["dimension"]]=float(x["weight"]); meta[key]=x["question_type"]
    values=defaultdict(lambda:defaultdict(list)); record_final={}
    for key,scores in per_record.items():
        q=meta[key]
        if q not in DIMS or any(d not in scores for d in DIMS[q]): continue
        final=sum(scores[d]*weights[key][d] for d in DIMS[q]); record_final[key]=final
        values[key[1]][q+"/final"].append(final)
        for d in DIMS[q]: values[key[1]][q+"/"+d].append(scores[d])
    result={}
    for mode,metrics in values.items():
        result[mode]={k:sum(v)/len(v) for k,v in metrics.items()}
        present=[result[mode][q+"/final"] for q in ORDER if q+"/final" in result[mode]]
        result[mode]["macro_final"]=sum(present)/len(present)
    paired={}
    for mode in result:
        if mode=="base": continue
        diffs=[record_final[(rid,mode)]-s for (rid,m),s in record_final.items() if m=="base" and (rid,mode) in record_final]
        paired[mode]=bootstrap_mean_ci(diffs,n_bootstrap=10000,seed=72)
    return result,paired


def main():
    p=argparse.ArgumentParser(); p.add_argument("--scores",required=True); p.add_argument("--output-prefix",required=True); a=p.parse_args()
    result,paired=aggregate(read_jsonl(a.scores)); out=Path(a.output_prefix); out.parent.mkdir(parents=True,exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"models":result,"paired_vs_axis":paired},indent=2),encoding="utf-8")
    if all(all(q+"/final" in result[m] for q in ORDER) for m in result):
        out.with_suffix(".md").write_text(markdown(result,paired),encoding="utf-8")
    print(json.dumps({"models":result,"paired_vs_axis":paired}))


if __name__=="__main__": main()
