"""Post-run analysis of Practical 4B: did the retrieved context actually contain the answer?

``results.json`` records which *documents* the top-k chunks came from. A document-level hit can still be a miss
at chunk level (the answer sentence may sit in a neighbouring chunk), so this script re-creates the chunks
deterministically from the knowledge base and checks the answer keys against the retrieved chunk text.

usage: python analyse_results.py results/part_b/results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import rag  # noqa: E402
from common import contains_key, load_json  # noqa: E402


def answer_in_context(results: dict, kb_dir: Path = HERE / "knowledge_base",
                      questions_path: Path = HERE / "questions.json") -> dict:
    chunk_words = results["primary_config"]["chunk_words"]
    chunks = {c.chunk_id: c for c in rag.build_chunks(rag.load_documents(kb_dir), chunk_words)}
    questions = {q["id"]: q for q in load_json(questions_path)}
    rows = []
    for row in results["generation"]["rows"]:
        q = questions[row["id"]]
        if not q["gold_docs"]:
            continue
        ids = [r["chunk"] for r in row["rag"]["retrieved"]]
        rows.append({
            "id": row["id"],
            "gold_document_in_top_k": bool(row["rag"]["gold_retrieved"]),
            "answer_chunk_in_top_k": any(contains_key(chunks[i].text, k) for i in ids for k in q["answer_keys"]),
            "rag_correct": bool(row["rag"]["correct"]),
        })
    with_ans = [r for r in rows if r["answer_chunk_in_top_k"]]
    without = [r for r in rows if not r["answer_chunk_in_top_k"]]
    return {
        "n": len(rows),
        "gold_document_in_top_k": sum(r["gold_document_in_top_k"] for r in rows),
        "answer_chunk_in_top_k": len(with_ans),
        "rag_correct_given_answer_chunk": sum(r["rag_correct"] for r in with_ans),
        "n_without_answer_chunk": len(without),
        "rag_correct_without_answer_chunk": sum(r["rag_correct"] for r in without),
        "rows": rows,
    }


if __name__ == "__main__":
    path = Path(sys.argv[1])
    out = answer_in_context(json.loads(path.read_text()))
    (path.parent / "answer_context.json").write_text(json.dumps(out, indent=2) + "\n")
    print({k: v for k, v in out.items() if k != "rows"})
