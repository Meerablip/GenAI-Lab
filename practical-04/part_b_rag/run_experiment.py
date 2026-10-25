"""Practical 4B: Retrieval-Augmented Generation with embeddings, vector databases and semantic search.

Stages
  1. Chunk a fictional knowledge base (ZIT handbook) at three granularities.
  2. Embed with two sentence-embedding models and index in ChromaDB and FAISS; BM25 is the lexical baseline.
  3. Evaluate retrieval (Hit@k, MRR) on 22 answerable questions.
  4. Generate answers with a local LLM (Ollama) closed-book, with retrieved context (RAG) and with oracle context.

The headline configuration below is fixed *before* any result is seen.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # for common.py
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import rag  # noqa: E402
from common import ResourceMonitor, contains_key, load_json, snapshot_env, write_json  # noqa: E402

CHUNK_SIZES = [25, 50, 100]
EMBEDDERS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "bge-small": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: "),
}
PRIMARY = {"embedder": "minilm", "chunk_words": 50, "top_k": 3}
LLM = "llama3.2:3b"
LLM_OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 160, "num_ctx": 2048}
PACKAGES = ["sentence-transformers", "transformers", "torch", "chromadb", "faiss-cpu", "rank-bm25",
            "numpy", "ollama", "psutil"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- retrieval experiment
def run_retrieval(docs, questions, out: Path) -> dict:
    answerable = [q for q in questions if q["gold_docs"]]
    results = {"configs": [], "per_question": {}, "latency_ms": {}, "build": {}}
    embedders = {}
    for key, (model_id, prefix) in EMBEDDERS.items():
        log(f"loading embedder {key}: {model_id}")
        embedders[key] = rag.Embedder(key, model_id, prefix)
    results["embedders"] = {k: {"model_id": e.model_id, "dim": e.dim, "load_s": round(e.load_s, 2)}
                            for k, e in embedders.items()}

    stores_for_primary = {}
    for size in CHUNK_SIZES:
        chunks = rag.build_chunks(docs, size)
        words = [c.n_words for c in chunks]
        log(f"chunk size {size}: {len(chunks)} chunks (mean {np.mean(words):.1f} words)")
        results.setdefault("chunking", {})[str(size)] = {
            "n_chunks": len(chunks), "mean_words": round(float(np.mean(words)), 1),
            "max_words": int(max(words))}

        # lexical baseline
        bm25 = rag.BM25Store()
        bm25.add(chunks)
        ranks = []
        for q in answerable:
            hits = [cid for cid, _ in bm25.search(q["question"], 10)]
            ranks.append(rag.first_relevant_rank(hits, q["gold_docs"]))
            results["per_question"].setdefault(q["id"], {})[f"bm25@{size}"] = ranks[-1]
        results["configs"].append({"method": "bm25", "store": "bm25", "chunk_words": size,
                                   "n_chunks": len(chunks), **_by_type(ranks, answerable)})

        for key, emb in embedders.items():
            t0 = time.perf_counter()
            vectors = emb.encode_docs([c.embed_text for c in chunks])
            embed_s = time.perf_counter() - t0
            qvecs = {q["id"]: emb.encode_query(q["question"]) for q in answerable}

            chroma = rag.ChromaStore(out / "chroma_db" / f"{key}_{size}", f"zit_{key}_{size}")
            t0 = time.perf_counter()
            chroma.add(chunks, vectors)
            chroma_build_s = time.perf_counter() - t0
            faiss_store = rag.FaissStore(emb.dim)
            t0 = time.perf_counter()
            faiss_store.add(chunks, vectors)
            faiss_build_s = time.perf_counter() - t0
            results["build"][f"{key}@{size}"] = {
                "embed_s": round(embed_s, 3), "chunks_per_s": round(len(chunks) / embed_s, 1),
                "chroma_build_s": round(chroma_build_s, 3), "faiss_build_s": round(faiss_build_s, 4),
                "chroma_disk_mb": round(chroma.disk_mb(), 2), "n_vectors": len(chunks), "dim": emb.dim}

            per_store = {}
            for store in (chroma, faiss_store):
                ranks, tops = [], []
                for q in answerable:
                    found = store.search(qvecs[q["id"]], 10)
                    ids = [cid for cid, _ in found]
                    tops.append(ids)
                    ranks.append(rag.first_relevant_rank(ids, q["gold_docs"]))
                    results["per_question"].setdefault(q["id"], {})[f"{key}-{store.name}@{size}"] = ranks[-1]
                per_store[store.name] = (ranks, tops)
                results["configs"].append({"method": key, "store": store.name, "chunk_words": size,
                                           "n_chunks": len(chunks), **_by_type(ranks, answerable)})
            same_top1 = np.mean([a[0] == b[0] for a, b in zip(per_store["chroma"][1], per_store["faiss"][1])])
            same_top5 = np.mean([a[:5] == b[:5] for a, b in zip(per_store["chroma"][1], per_store["faiss"][1])])
            results.setdefault("store_agreement", {})[f"{key}@{size}"] = {
                "top1_identical": round(float(same_top1), 4), "top5_identical": round(float(same_top5), 4)}

            if key == PRIMARY["embedder"] and size == PRIMARY["chunk_words"]:
                stores_for_primary = {"chunks": chunks, "chroma": chroma, "faiss": faiss_store, "emb": emb, "bm25": bm25}

    # recall@k curve at the primary chunk size
    curve = {}
    for method, store in [("bm25", "bm25"), ("minilm", "chroma"), ("bge-small", "chroma")]:
        ranks = [results["per_question"][q["id"]].get(
            f"bm25@{PRIMARY['chunk_words']}" if method == "bm25" else f"{method}-{store}@{PRIMARY['chunk_words']}")
            for q in answerable]
        curve[method] = [round(sum(r is not None and r <= k for r in ranks) / len(ranks), 4) for k in range(1, 11)]
    results["recall_curve"] = curve

    # latency at primary config
    p = stores_for_primary
    probe = answerable[0]["question"]
    qv = p["emb"].encode_query(probe)
    results["latency_ms"] = {
        "query_encode": rag.median_ms(lambda: p["emb"].encode_query(probe)),
        "chroma_search": rag.median_ms(lambda: p["chroma"].search(qv, 3)),
        "faiss_search": rag.median_ms(lambda: p["faiss"].search(qv, 3)),
        "bm25_search": rag.median_ms(lambda: p["bm25"].search(probe, 3)),
    }
    log(f"latency (ms): {results['latency_ms']}")
    results["_primary"] = p
    return results


def _by_type(ranks, questions) -> dict:
    out = {"overall": rag.summarise_ranks(ranks)}
    for t in ("keyword", "paraphrase"):
        sub = [r for r, q in zip(ranks, questions) if q["type"] == t]
        out[t] = rag.summarise_ranks(sub)
    return out


# --------------------------------------------------------------------------- generation experiment
def grade(q: dict, answer: str, context_docs: list[str], all_ids: list[str]) -> dict:
    abstained = rag.is_abstention(answer)
    cited = [d for d in all_ids if f"[{d}" in answer]
    if q["type"] == "unanswerable":
        correct = abstained  # the right behaviour is to decline
        hallucinated = not abstained
    else:
        correct = any(contains_key(answer, k) for k in q["answer_keys"])
        hallucinated = not correct and not abstained  # wrong, yet stated confidently
    return {
        "correct": bool(correct), "abstained": bool(abstained), "hallucinated": bool(hallucinated),
        "cited": cited, "cited_gold": any(d in q["gold_docs"] for d in cited) if q["gold_docs"] else None,
        "context_docs": context_docs,
    }


def run_generation(docs, questions, primary) -> dict:
    import ollama

    client = ollama.Client(host="http://127.0.0.1:11434")
    info = {"model": LLM, "options": LLM_OPTIONS}
    try:
        show = client.show(LLM)
        details = getattr(show, "details", None)
        info["details"] = details.model_dump() if hasattr(details, "model_dump") else str(details)
        for m in client.list().models:
            if m.model == LLM:
                info["digest"] = m.digest[:12]
                info["size_gb"] = round(m.size / 1e9, 2)
    except Exception as exc:  # metadata is optional
        info["metadata_error"] = repr(exc)

    log("warming up the LLM")
    warm = rag.ask_ollama(client, LLM, rag.SYSTEM_CLOSED, "Say hello.", LLM_OPTIONS)
    info["warmup"] = warm

    by_id = {d.doc_id: d for d in docs}
    all_ids = list(by_id)
    chunks_by_id = {c.chunk_id: c for c in primary["chunks"]}
    rows = []
    for n, q in enumerate(questions, start=1):
        log(f"question {n}/{len(questions)}: {q['question']}")
        gold = q["gold_docs"] or q.get("related_docs", [])
        row = {"id": q["id"], "type": q["type"], "question": q["question"], "gold_docs": q["gold_docs"],
               "answer_keys": q["answer_keys"]}

        # closed-book
        r = rag.ask_ollama(client, LLM, rag.SYSTEM_CLOSED, q["question"], LLM_OPTIONS)
        row["closed_book"] = {**r, **grade(q, r["answer"], [], all_ids)}

        # RAG: retrieve top-k chunks from ChromaDB
        t0 = time.perf_counter()
        qv = primary["emb"].encode_query(q["question"])
        found = primary["chroma"].search(qv, PRIMARY["top_k"])
        retrieval_ms = (time.perf_counter() - t0) * 1000
        retrieved = [chunks_by_id[cid] for cid, _ in found]
        user = f"Context:\n{rag.format_context(retrieved)}\n\nQuestion: {q['question']}\nAnswer:"
        r = rag.ask_ollama(client, LLM, rag.SYSTEM_RAG, user, LLM_OPTIONS)
        row["rag"] = {**r, **grade(q, r["answer"], [c.doc_id for c in retrieved], all_ids),
                      "retrieval_ms": round(retrieval_ms, 2),
                      "retrieved": [{"chunk": cid, "score": round(s, 4)} for cid, s in found],
                      "gold_retrieved": any(c.doc_id in q["gold_docs"] for c in retrieved)}

        # oracle: the full gold document(s)
        oracle_chunks = [rag.Chunk(d, d, by_id[d].title, by_id[d].text) for d in gold]
        user = f"Context:\n{rag.format_context(oracle_chunks)}\n\nQuestion: {q['question']}\nAnswer:"
        r = rag.ask_ollama(client, LLM, rag.SYSTEM_RAG, user, LLM_OPTIONS)
        row["oracle"] = {**r, **grade(q, r["answer"], gold, all_ids)}
        rows.append(row)
        log("  closed={!r} | rag={!r} | oracle={!r}".format(
            row["closed_book"]["answer"][:70], row["rag"]["answer"][:70], row["oracle"]["answer"][:70]))

    return {"llm": info, "rows": rows, "summary": summarise_generation(rows)}


def summarise_generation(rows) -> dict:
    ans = [r for r in rows if r["type"] != "unanswerable"]
    unans = [r for r in rows if r["type"] == "unanswerable"]
    out = {"n_answerable": len(ans), "n_unanswerable": len(unans)}
    for cond in ("closed_book", "rag", "oracle"):
        correct = sum(r[cond]["correct"] for r in ans)
        out[cond] = {
            "correct": correct,
            "accuracy": round(correct / len(ans), 4),
            "wrong_but_confident": sum(r[cond]["hallucinated"] for r in ans),
            "abstained_on_answerable": sum(r[cond]["abstained"] and not r[cond]["correct"] for r in ans),
            "unanswerable_abstained": sum(r[cond]["correct"] for r in unans),
            "unanswerable_hallucinated": sum(r[cond]["hallucinated"] for r in unans),
            "mean_wall_s": round(float(np.mean([r[cond]["wall_s"] for r in rows])), 2),
            "mean_prompt_tokens": round(float(np.mean([r[cond]["prompt_tokens"] for r in rows])), 1),
            "mean_gen_tokens": round(float(np.mean([r[cond]["gen_tokens"] for r in rows])), 1),
            "mean_tok_per_s": round(float(np.mean([r[cond]["tok_per_s"] for r in rows])), 2),
        }
        if cond != "closed_book":
            out[cond]["citation_rate"] = round(sum(bool(r[cond]["cited_gold"]) for r in ans) / len(ans), 4)
    hit = [r for r in ans if r["rag"]["gold_retrieved"]]
    miss = [r for r in ans if not r["rag"]["gold_retrieved"]]
    out["rag_by_retrieval"] = {
        "gold_in_top_k": {"n": len(hit), "correct": sum(r["rag"]["correct"] for r in hit)},
        "gold_not_in_top_k": {"n": len(miss), "correct": sum(r["rag"]["correct"] for r in miss)},
    }
    for t in ("keyword", "paraphrase"):
        sub = [r for r in ans if r["type"] == t]
        if sub:
            out[f"accuracy_{t}"] = {c: round(sum(r[c]["correct"] for r in sub) / len(sub), 4)
                                    for c in ("closed_book", "rag", "oracle")}
    out["mean_retrieval_ms"] = round(float(np.mean([r["rag"]["retrieval_ms"] for r in rows])), 2)
    return out


# --------------------------------------------------------------------------- main
def markdown_summary(res: dict) -> str:
    lines = ["## Practical 4B - RAG pipeline", ""]
    primary = [c for c in res["retrieval"]["configs"] if c["chunk_words"] == PRIMARY["chunk_words"]]
    lines += ["| method | store | Hit@1 | Hit@3 | Hit@5 | MRR@10 |", "|---|---|---|---|---|---|"]
    for c in primary:
        o = c["overall"]
        lines.append(f"| {c['method']} | {c['store']} | {o['hit@1']} | {o['hit@3']} | {o['hit@5']} | {o['mrr@10']} |")
    gen = res.get("generation")
    if gen:
        s = gen["summary"]
        lines += ["", f"LLM `{LLM}` on {s['n_answerable']} answerable questions:", "",
                  "| condition | correct | wrong-but-confident | mean latency (s) |", "|---|---|---|---|"]
        for cond in ("closed_book", "rag", "oracle"):
            c = s[cond]
            lines.append(f"| {cond} | {c['correct']}/{s['n_answerable']} | {c['wrong_but_confident']} | {c['mean_wall_s']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/part_b")
    ap.add_argument("--skip-llm", action="store_true", help="retrieval experiments only")
    ap.add_argument("--smoke", action="store_true", help="3 questions only (wiring check)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    docs = rag.load_documents(HERE / "knowledge_base")
    questions = load_json(HERE / "questions.json")
    if args.smoke:
        questions = [questions[0], questions[8], questions[-1]]  # keyword, paraphrase, unanswerable
    log(f"{len(docs)} documents, {sum(len(d.text.split()) for d in docs)} words, {len(questions)} questions")

    result: dict = {"env": snapshot_env(PACKAGES), "primary_config": PRIMARY, "llm_name": LLM,
                    "kb": {"n_docs": len(docs), "n_words": sum(len(d.text.split()) for d in docs),
                           "n_questions": len(questions),
                           "n_answerable": sum(bool(q["gold_docs"]) for q in questions)}}
    with ResourceMonitor() as mon:
        retrieval = run_retrieval(docs, questions, out)
    primary = retrieval.pop("_primary")
    retrieval["resources"] = mon.result
    result["retrieval"] = retrieval

    if not args.skip_llm:
        with ResourceMonitor() as mon:
            result["generation"] = run_generation(docs, questions, primary)
        result["generation"]["resources"] = mon.result

    write_json(out / "results.json", result)
    summary = markdown_summary(result)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    log(f"wrote {out/'results.json'}")


if __name__ == "__main__":
    main()
