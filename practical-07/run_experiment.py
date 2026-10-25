"""Practical 7: hallucination and response-quality evaluation (automated metrics + LLM judge + sampling consistency)."""
from __future__ import annotations

import argparse, json, re, string, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "practical-04"))
import numpy as np  # noqa: E402
from common import ResourceMonitor, contains_key, snapshot_env, write_json  # noqa: E402

GEN_MODEL, JUDGE_MODEL = "llama3.2:3b", "qwen2.5:3b"
SYSTEM_GEN = "You are a helpful assistant. Answer the question concisely in one or two sentences."
SYSTEM_JUDGE = ('You are a strict fact-checker. Compare the RESPONSE with the REFERENCE answer for the QUESTION. Reply with JSON only: '
                '{"verdict": "correct" | "partially_correct" | "incorrect" | "declined", "hallucination": true | false, "unsupported_claims": ["..."]}. '
                'Set hallucination to true if the response states facts that contradict the reference or are fabricated.')
ABSTAIN = re.compile(
    r"(?:don'?t|do not|doesn'?t|does not) (?:know|have|mention|specify|provide|contain|say)"
    r"|(?:isn'?t|is not|aren'?t|are not|wasn'?t|not) (?:mentioned|provided|specified|stated|available|found|familiar|aware|sure|publicly|yet|been)"
    r"|no (?:information|mention|details?|data|record|evidence|such|winner)|(?:cannot|can'?t|couldn'?t|could not|unable to) (?:be known|find|answer|determine|provide|verify|confirm|locate|predict)"
    r"|fictitious|does not exist|doesn'?t exist|has not (?:been|yet)|hasn'?t (?:been|yet)|(?:future|not yet happened|not occurred)", re.I)
PACKAGES = ["ollama", "sentence-transformers", "sacrebleu", "rouge-score", "scipy", "numpy", "torch"]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def norm_tokens(s: str) -> list[str]:
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in s.split() if t not in {"a", "an", "the"}]


def token_f1(pred: str, ref: str) -> float:
    p, r = norm_tokens(pred), norm_tokens(ref)
    common = sum(min(p.count(t), r.count(t)) for t in set(p))
    if not p or not r or common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def chat(client, model, system, user, **kw):
    t0 = time.perf_counter()
    r = client.chat(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
    g = lambda n: getattr(r, n, None) or 0  # noqa: E731
    return {"text": r.message.content.strip(), "seconds": round(time.perf_counter() - t0, 2), "tokens": g("eval_count")}


def auroc(scores, labels) -> float | None:
    """P(score of a positive > score of a negative); positives = the class we want to detect."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    return round(float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])), 3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/prac07"); args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    qs = json.loads((HERE / "questions.json").read_text())
    import ollama
    from rouge_score import rouge_scorer
    import sacrebleu
    from sentence_transformers import SentenceTransformer, util
    from scipy.stats import spearmanr
    client = ollama.Client(host="http://127.0.0.1:11434")
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    rs = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    enc = lambda t: emb.encode(t, normalize_embeddings=True, convert_to_tensor=True)  # noqa: E731

    rows = []
    with ResourceMonitor() as mon:
        log("warm-up"); chat(client, GEN_MODEL, SYSTEM_GEN, "Say hello.", options={"num_predict": 8})
        for i, q in enumerate(qs, 1):
            log(f"generate {i}/{len(qs)} [{q['id']}] {q['question']}")
            main_r = chat(client, GEN_MODEL, SYSTEM_GEN, q["question"], options={"temperature": 0, "seed": 42, "num_predict": 120})
            samples = [chat(client, GEN_MODEL, SYSTEM_GEN, q["question"], options={"temperature": 0.8, "seed": s, "num_predict": 120}) for s in (1, 2, 3)]
            rows.append({"q": q, "response": main_r, "samples": samples})
            log(f"  -> {main_r['text'][:120]!r}")
        log("judging"); 
        for r in rows:
            q = r["q"]
            u = f"QUESTION: {q['question']}\nREFERENCE: {q['reference']}\nRESPONSE: {r['response']['text']}"
            j = chat(client, JUDGE_MODEL, SYSTEM_JUDGE, u, format="json", options={"temperature": 0, "seed": 42, "num_predict": 200})
            try:
                p = json.loads(j["text"]); r["judge"] = {"verdict": str(p.get("verdict", "")).lower(), "hallucination": bool(p.get("hallucination")),
                                                         "unsupported_claims": [str(x) for x in p.get("unsupported_claims", [])][:4], "valid_json": True}
            except Exception:
                r["judge"] = {"verdict": "invalid", "hallucination": None, "unsupported_claims": [], "valid_json": False, "raw": j["text"][:200]}
            log(f"  judge {q['id']}: {r['judge']['verdict']} halluc={r['judge']['hallucination']}")

    items = []
    for r in rows:
        q, resp = r["q"], r["response"]["text"]
        ref = q["reference"]
        abstained = bool(ABSTAIN.search(resp.replace("’", "'")))
        has_key = any(contains_key(resp, k) for k in q["keys"])
        correct = abstained if q["kind"] == "unanswerable" else (has_key or (abstained if q["kind"] == "false_premise" else False))
        halluc = (not correct) and (not abstained or q["kind"] in ("known", "hard"))
        if q["kind"] == "unanswerable":
            halluc = not abstained
        e_main, e_ref = enc(resp), enc(ref)
        cons = [float(util.cos_sim(e_main, enc(s["text"]))) for s in r["samples"]]
        sample_ok = [abstained_or_key(s["text"], q) for s in r["samples"]]
        m = {"exact_match": int(" ".join(norm_tokens(resp)) == " ".join(norm_tokens(ref))), "key_fact": int(has_key),
             "token_f1": round(token_f1(resp, ref), 4), "bleu": round(sacrebleu.sentence_bleu(resp, [ref]).score / 100, 4),
             "rouge1": round(rs.score(ref, resp)["rouge1"].fmeasure, 4), "rougeL": round(rs.score(ref, resp)["rougeL"].fmeasure, 4),
             "semantic_sim": round(float(util.cos_sim(e_main, e_ref)), 4), "consistency": round(float(np.mean(cons)), 4),
             "sample_agreement": round(float(np.mean(sample_ok)), 3)}
        items.append({"id": q["id"], "kind": q["kind"], "question": q["question"], "reference": ref, "keys": q["keys"], "response": resp,
                      "response_tokens": r["response"]["tokens"], "seconds": r["response"]["seconds"], "samples": [s["text"] for s in r["samples"]],
                      "abstained": abstained, "correct": bool(correct), "hallucinated": bool(halluc), "metrics": m, "judge": r["judge"]})

    def corr(metric):
        x, y = [it["metrics"][metric] for it in items], [int(it["correct"]) for it in items]
        rho = spearmanr(x, y).correlation if len(set(x)) > 1 else float("nan")
        return None if np.isnan(rho) else round(float(rho), 3)
    names = ["exact_match", "key_fact", "token_f1", "bleu", "rouge1", "rougeL", "semantic_sim", "consistency", "sample_agreement"]
    by = lambda f: [it for it in items if f(it)]  # noqa: E731
    agg = {n: {"overall": round(float(np.mean([it["metrics"][n] for it in items])), 4),
               "correct": round(float(np.mean([it["metrics"][n] for it in by(lambda i: i["correct"])] or [0])), 4),
               "incorrect": round(float(np.mean([it["metrics"][n] for it in by(lambda i: not i["correct"])] or [0])), 4),
               "spearman_with_correct": corr(n)} for n in names}
    kinds = {k: {"n": len(by(lambda i: i["kind"] == k)), "correct": sum(i["correct"] for i in by(lambda i: i["kind"] == k)),
                 "hallucinated": sum(i["hallucinated"] for i in by(lambda i: i["kind"] == k))} for k in ("known", "hard", "unanswerable", "false_premise")}
    jv = [it for it in items if it["judge"]["valid_json"]]
    judge_correct = [it["judge"]["verdict"] in ("correct", "declined") if it["kind"] == "unanswerable" else it["judge"]["verdict"] == "correct" for it in jv]
    agree = [bool(a) == bool(it["correct"]) for a, it in zip(judge_correct, jv)]
    po = float(np.mean(agree)) if agree else 0.0
    p_j, p_k = float(np.mean(judge_correct)) if jv else 0, float(np.mean([it["correct"] for it in jv])) if jv else 0
    pe = p_j * p_k + (1 - p_j) * (1 - p_k)
    summary = {"n": len(items), "correct": sum(i["correct"] for i in items), "hallucinated": sum(i["hallucinated"] for i in items),
               "abstained": sum(i["abstained"] for i in items), "kinds": kinds, "metrics": agg,
               "judge": {"valid_json": len(jv), "agreement_with_key_grading": round(po, 3), "cohen_kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else None,
                         "judge_flagged_hallucination": sum(bool(i["judge"]["hallucination"]) for i in jv),
                         "judge_hallucination_agrees_with_auto": sum(bool(i["judge"]["hallucination"]) == i["hallucinated"] for i in jv)},
               "detection_auroc": {"low_consistency_flags_incorrect": auroc([-it["metrics"]["consistency"] for it in items], [not it["correct"] for it in items]),
                                   "low_sample_agreement_flags_incorrect": auroc([-it["metrics"]["sample_agreement"] for it in items], [not it["correct"] for it in items]),
                                   "low_semantic_sim_flags_incorrect": auroc([-it["metrics"]["semantic_sim"] for it in items], [not it["correct"] for it in items])},
               "mean_seconds": round(float(np.mean([i["seconds"] for i in items])), 2)}
    write_json(out / "results.json", {"env": snapshot_env(PACKAGES), "models": {"generator": GEN_MODEL, "judge": JUDGE_MODEL, "embedder": "all-MiniLM-L6-v2"},
                                      "summary": summary, "items": items, "resources": mon.result})
    with open(out / "human_rating_sheet.csv", "w") as f:
        f.write("id,question,response,factual_accuracy_1to5,relevance_1to5,coherence_1to5,completeness_1to5,helpfulness_1to5,hallucination_yes_no\n")
        for it in items:
            f.write(",".join('"' + str(x).replace('"', '""') + '"' for x in (it["id"], it["question"], it["response"])) + ",,,,,,\n")
    md = ["## Practical 7 - hallucination and response-quality evaluation", "", f"{summary['correct']}/{summary['n']} correct, {summary['hallucinated']} hallucinated, {summary['abstained']} declined.", "",
          "| metric | correct | incorrect | spearman vs correctness |", "|---|---|---|---|"]
    md += [f"| {n} | {a['correct']} | {a['incorrect']} | {a['spearman_with_correct']} |" for n, a in agg.items()]
    (out / "summary.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


def abstained_or_key(text: str, q: dict) -> bool:
    if q["kind"] == "unanswerable":
        return bool(ABSTAIN.search(text))
    return any(contains_key(text, k) for k in q["keys"]) or (q["kind"] == "false_premise" and bool(ABSTAIN.search(text)))


if __name__ == "__main__":
    main()
