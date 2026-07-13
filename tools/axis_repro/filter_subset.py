import argparse, json
from pathlib import Path
from .common import read_jsonl

p=argparse.ArgumentParser();p.add_argument("--predictions",required=True);p.add_argument("--manifest",required=True);p.add_argument("--output",required=True);p.add_argument("--modes",nargs="+");a=p.parse_args()
ids=set(json.loads(Path(a.manifest).read_text(encoding="utf-8"))["record_ids"])
payload=lambda x:x.get("row",x)
rows=[x for x in read_jsonl(a.predictions) if payload(x)["record_id"] in ids and (not a.modes or payload(x)["mode"] in a.modes)]
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
print(json.dumps({"rows":len(rows),"records":len({payload(x)["record_id"] for x in rows}),"modes":sorted({payload(x)["mode"] for x in rows})}))

