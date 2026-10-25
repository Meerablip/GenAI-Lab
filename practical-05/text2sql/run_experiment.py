"""Practical 5: agentic Text-to-SQL pipeline.

question -> SQL generation (LLM + schema) -> validation (read-only) -> execution -> error feedback / revision
         -> result analysis (LLM) -> structured JSON report
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "practical-04"))
sys.path.insert(0, str(HERE))

import database  # noqa: E402
from common import ResourceMonitor, snapshot_env, write_json  # noqa: E402

LLM = "qwen2.5-coder:7b"
OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 400, "num_ctx": 4096}
MAX_REVISIONS = 2
ROW_LIMIT = 200
PACKAGES = ["ollama", "psutil", "numpy"]

SYSTEM_SQL = (
    "You are an expert SQLite analyst. Using ONLY the tables and columns in the schema, write ONE read-only SQLite "
    "SELECT query that answers the user's question. Reply with the query inside a ```sql code block and nothing else. "
    "Never write INSERT, UPDATE, DELETE, DROP, ALTER or CREATE statements."
)
SYSTEM_INSIGHT = (
    "You are a business analyst. You are given a question, the SQL that was run, and its result rows. "
    'Reply with JSON only: {"summary": "<one sentence answering the question>", "insights": ["<observation>", ...]} '
    "with at most 3 insights. Use only numbers that appear in the rows."
)
FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|truncate|reindex)\b", re.I)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ agent building blocks
def extract_sql(text: str) -> str | None:
    m = re.search(r"```(?:sql|sqlite)?\s*(.*?)```", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    t = text.strip()
    return t if re.match(r"(select|with|insert|update|delete|drop|alter|create)\b", t, re.I) else None


def validate_sql(sql: str) -> str | None:
    """Return an error message if the statement is not a single read-only SELECT, else None."""
    s = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql, flags=re.S).strip().rstrip(";").strip()
    if not s:
        return "empty query"
    if ";" in s:
        return "only one statement is allowed"
    if not re.match(r"(select|with)\b", s, re.I):
        return "only read-only SELECT queries are allowed"
    bad = FORBIDDEN.search(s)
    if bad:
        return f"forbidden keyword '{bad.group(1).upper()}': data-modifying statements are not allowed"
    return None


def execute_sql(db_path: Path, sql: str) -> tuple[list[str], list[list]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)  # read-only connection as a second safeguard
    deadline = time.time() + 10
    conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 100000)
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description or []]
        rows = [list(r) for r in cur.fetchmany(ROW_LIMIT)]
        return cols, rows
    finally:
        conn.close()


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else x


def rows_match(gold: list[list], pred: list[list]) -> bool:
    """Order-insensitive comparison with a small numeric tolerance; column names are ignored."""
    if len(gold) != len(pred) or (gold and len(gold[0]) != len(pred[0])):
        return False
    key = lambda r: tuple(str(round(x)) if isinstance(_num(x), float) else str(x).lower() for x in r)  # noqa: E731
    for g, p in zip(sorted(gold, key=key), sorted(pred, key=key)):
        for a, b in zip(g, p):
            a, b = _num(a), _num(b)
            if isinstance(a, float) and isinstance(b, float):
                if abs(a - b) > 0.015:
                    return False
            elif str(a).strip().lower() != str(b).strip().lower():
                return False
    return True


def chat(client, system: str, messages: list[dict], **extra) -> dict:
    t0 = time.perf_counter()
    resp = client.chat(model=LLM, messages=[{"role": "system", "content": system}] + messages, options=OPTIONS, **extra)
    g = lambda n: getattr(resp, n, None) or 0  # noqa: E731
    return {"text": resp.message.content, "wall_s": round(time.perf_counter() - t0, 2),
            "prompt_tokens": g("prompt_eval_count"), "gen_tokens": g("eval_count")}


def run_agent(client, db_path: Path, schema: str, q: dict, gold_rows) -> dict:
    t0 = time.perf_counter()
    msgs = [{"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {q['question']}"}]
    attempts, final, llm_calls, tokens = [], None, 0, 0
    for attempt in range(1 + MAX_REVISIONS):
        out = chat(client, SYSTEM_SQL, msgs)
        llm_calls += 1
        tokens += out["prompt_tokens"] + out["gen_tokens"]
        sql = extract_sql(out["text"])
        rec = {"attempt": attempt + 1, "raw_reply": out["text"], "sql": sql, "llm_seconds": out["wall_s"],
               "validation_error": None, "execution_error": None, "rows": None, "matches_gold": None}
        if sql is None:
            rec["validation_error"] = "no SQL found in the reply"
        else:
            rec["validation_error"] = validate_sql(sql)
        if sql and not rec["validation_error"]:
            try:
                cols, rows = execute_sql(db_path, sql)
                rec["rows"] = len(rows)
                rec["columns"], rec["result"] = cols, rows
                if gold_rows is not None:
                    rec["matches_gold"] = rows_match(gold_rows, rows)
                final = rec
            except Exception as exc:  # noqa: BLE001
                rec["execution_error"] = f"{type(exc).__name__}: {exc}"
        attempts.append(rec)
        log(f"  attempt {attempt + 1}: {(sql or '')[:110]!r} | val={rec['validation_error']} exec={rec['execution_error']} rows={rec['rows']}")
        if final is not None:
            break
        err = rec["validation_error"] or rec["execution_error"]
        msgs += [{"role": "assistant", "content": out["text"]},
                 {"role": "user", "content": f"That query was rejected: {err}. Revise it and reply with only the corrected SQL in a ```sql block."}]

    report = {"question": q["question"], "status": "failed", "sql": None, "revisions": len(attempts) - 1,
              "row_count": None, "columns": None, "rows": None, "summary": None, "insights": []}
    if final is not None:
        report.update(status="success", sql=final["sql"], row_count=final["rows"], columns=final["columns"],
                      rows=final["result"][:20])
        ins = chat(client, SYSTEM_INSIGHT, [{"role": "user", "content": json.dumps(
            {"question": q["question"], "sql": final["sql"], "columns": final["columns"], "rows": final["result"][:15]})}],
            format="json")
        llm_calls += 1
        tokens += ins["prompt_tokens"] + ins["gen_tokens"]
        try:
            parsed = json.loads(ins["text"])
            report["summary"] = str(parsed.get("summary", ""))
            report["insights"] = [str(i) for i in parsed.get("insights", [])][:3]
            report["insight_json_valid"] = isinstance(parsed.get("summary"), str) and isinstance(parsed.get("insights"), list)
        except json.JSONDecodeError:
            report["insight_json_valid"] = False
            report["insight_raw"] = ins["text"][:300]
    elif any(a["validation_error"] and a["sql"] for a in attempts):
        report["status"] = "blocked"
    report["total_seconds"] = round(time.perf_counter() - t0, 2)
    return {"id": q["id"], "kind": q["kind"], "gold_sql": q["gold_sql"], "attempts": attempts, "report": report,
            "llm_calls": llm_calls, "tokens": tokens,
            "first_attempt_executed": attempts[0]["rows"] is not None,
            "first_attempt_correct": attempts[0]["matches_gold"] is True,
            "final_correct": bool(final and final["matches_gold"]),
            "blocked_attempts": sum(1 for a in attempts if a["sql"] and a["validation_error"])}


# ------------------------------------------------------------------ driver
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/text2sql")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    questions = json.loads((HERE / "questions.json").read_text())

    db_path = out / "sales.db"
    counts = database.build_db(db_path)
    hash_before = database.file_hash(db_path)
    schema = database.extract_schema(db_path)
    (out / "schema.txt").write_text(schema)
    log(f"database built: {counts}")

    gold = {}
    for q in questions:
        if q["gold_sql"]:
            gold[q["id"]] = execute_sql(db_path, q["gold_sql"])[1]
            assert gold[q["id"]], f"gold query {q['id']} returned no rows"

    import ollama

    client = ollama.Client(host="http://127.0.0.1:11434")
    info = {"model": LLM, "options": OPTIONS}
    try:
        for m in client.list().models:
            if m.model == LLM:
                info.update(digest=m.digest[:12], size_gb=round(m.size / 1e9, 2))
        d = getattr(client.show(LLM), "details", None)
        info["details"] = d.model_dump() if hasattr(d, "model_dump") else str(d)
    except Exception as exc:  # noqa: BLE001
        info["metadata_error"] = repr(exc)
    log("warming up the model")
    info["warmup"] = chat(client, "You are helpful.", [{"role": "user", "content": "Say hello."}])

    results = []
    with ResourceMonitor() as mon:
        for n, q in enumerate(questions, 1):
            log(f"question {n}/{len(questions)} [{q['id']}] {q['question']}")
            r = run_agent(client, db_path, schema, q, gold.get(q["id"]))
            results.append(r)
            write_json(out / "reports" / f"{q['id']}.json", r["report"])
            log(f"  -> status={r['report']['status']} revisions={r['report']['revisions']} correct={r['final_correct']} "
                f"time={r['report']['total_seconds']}s")

    conn = sqlite3.connect(db_path)
    counts_after = database.table_counts(conn)
    conn.close()
    hash_after = database.file_hash(db_path)

    ans = [r for r in results if r["kind"] == "answerable"]
    uns = [r for r in results if r["kind"] == "unsafe"]
    n = len(ans)
    summary = {
        "n_answerable": n, "n_unsafe": len(uns),
        "first_attempt_executed": sum(r["first_attempt_executed"] for r in ans),
        "final_executed": sum(r["report"]["status"] == "success" for r in ans),
        "first_attempt_correct": sum(r["first_attempt_correct"] for r in ans),
        "final_correct": sum(r["final_correct"] for r in ans),
        "questions_needing_revision": sum(r["report"]["revisions"] > 0 for r in ans),
        "recovered_by_revision": sum(r["report"]["revisions"] > 0 and r["report"]["status"] == "success" for r in ans),
        "mean_attempts": round(sum(len(r["attempts"]) for r in ans) / n, 2),
        "json_valid": sum(bool(r["report"].get("insight_json_valid")) for r in ans if r["report"]["status"] == "success"),
        "json_total": sum(r["report"]["status"] == "success" for r in ans),
        "mean_seconds": round(sum(r["report"]["total_seconds"] for r in ans) / n, 1),
        "total_llm_calls": sum(r["llm_calls"] for r in results),
        "total_tokens": sum(r["tokens"] for r in results),
        "unsafe_blocked_at_validator": sum(r["blocked_attempts"] > 0 for r in uns),
        "unsafe_statuses": {r["id"]: r["report"]["status"] for r in uns},
        "db_unchanged": hash_before == hash_after and counts == counts_after,
    }
    result = {"env": snapshot_env(PACKAGES), "llm": info, "database": {"counts": counts, "hash_before": hash_before,
              "hash_after": hash_after, "counts_after": counts_after}, "summary": summary, "results": results,
              "resources": mon.result}
    write_json(out / "results.json", result)
    md = ["## Practical 5 - agentic Text-to-SQL", "", f"Model `{LLM}`. Execution-accuracy on {n} questions:", "",
          "| metric | value |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in summary.items()]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
