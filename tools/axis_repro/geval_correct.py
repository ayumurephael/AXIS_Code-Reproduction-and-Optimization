from __future__ import annotations
import concurrent.futures,hashlib,math,re
from . import geval_deepseek as g

SCORE_RE=re.compile(r"(?:\*\*)?Score(?:\*\*)?\s*:\s*(?:\*\*)?\s*\[?([1-5])",re.I)

def final_score_distribution(response):
    choice=response.get("choices",[{}])[0];message=choice.get("message",{});text=message.get("content","") or ""
    hits=list(SCORE_RE.finditer(text))
    if not hits:return None
    offset=hits[-1].start(1);entries=choice.get("logprobs",{}).get("content",[]) or []
    cursor=0;target=None
    for entry in entries:
        token=str(entry.get("token",""));end=cursor+len(token)
        if cursor<=offset<end:target=entry;break
        cursor=end
    if target is None:return None
    logs={}
    for alt in target.get("top_logprobs",[]) or []:
        token=str(alt.get("token","")).strip().strip("[]* ")
        if len(token)==1 and token in "12345":logs[int(token)]=max(logs.get(int(token),-math.inf),float(alt["logprob"]))
    if len(logs)!=5:return None
    peak=max(logs.values());z=sum(math.exp(v-peak) for v in logs.values());probs={str(k):math.exp(v-peak)/z for k,v in logs.items()}
    return sum(int(k)*v for k,v in probs.items()),probs

def exact_samples(key,model,prompt,count):
    scores=[];attempts=0
    def one(_):
        response=g.api_call(key,model,prompt,1,False)
        content=response.get("choices",[{}])[0].get("message",{}).get("content","")
        return g.score_from_text(content)
    while len(scores)<count and attempts<100:
        batch=min(5,count-len(scores));attempts+=batch
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch) as ex:
            scores.extend(x for x in ex.map(one,range(batch)) if x is not None)
    if len(scores)<count:raise RuntimeError(f"Only {len(scores)}/{count} valid fallback scores after {attempts} calls")
    return scores[:count]

def judge_one(key,model,row,dim,spec,fallback_samples):
    weight,desc,guides=spec;prompt=g.prompt_for(row,dim,desc,guides);primary=g.api_call(key,model,prompt,0,True)
    dist=final_score_distribution(primary);message=primary.get("choices",[{}])[0].get("message",{});raw=message.get("content","")
    raw_score=g.score_from_text(raw)
    if dist:score,probs=dist;samples=[];method="final_score_top_logprobs"
    else:
        samples=exact_samples(key,model,prompt,fallback_samples);score=sum(samples)/len(samples);probs=None;method=f"exact_sample_mean_{fallback_samples}"
    return {"record_id":row["record_id"],"mode":row["mode"],"question_type":g.normalize_type(row["question_type"]),
            "dimension":dim,"weight":weight,"score":score,"method":method,"raw_score":raw_score,
            "distribution":probs,"fallback_scores":samples,"judge_content":raw,
            "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),"model":model,
            "system_fingerprint":primary.get("system_fingerprint"),"usage":primary.get("usage")}
