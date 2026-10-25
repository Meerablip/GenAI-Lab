"""Comparative analysis script for Practical 6: LangGraph vs CrewAI Multi-Agent Systems."""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import to_float

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("prac06.compare")


def compare_runs(part_a_dir: Path, part_b_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    a_data = json.loads((part_a_dir / "results.json").read_text(encoding="utf-8"))
    b_data = json.loads((part_b_dir / "results.json").read_text(encoding="utf-8"))

    logger.info("Comparing Part A (LangGraph) and Part B (CrewAI)...")

    comparison = {
        "framework_comparison": {
            "LangGraph": {
                "approach": "Graph-based workflow (StateGraph)",
                "coordination": "Nodes, edges, and state transitions",
                "key_features": "Conditional execution, cyclical revision loops, typed state checkpointing",
                "application": "Complex stateful workflows requiring dynamic self-correction and iterative refinement",
                "total_duration_s": a_data.get("duration_s", 0.0),
                "cpu_max_pct": a_data.get("resource_usage", {}).get("cpu_max_pct", 0.0),
                "ram_max_gb": a_data.get("resource_usage", {}).get("ram_max_gb", 0.0),
            },
            "CrewAI": {
                "approach": "Role-based collaboration",
                "coordination": "Agents, tasks, and sequential process",
                "key_features": "Task delegation, persona-driven prompt engineering, agent backstories",
                "application": "Collaborative business tasks with clear division of labor and linear handoffs",
                "total_duration_s": b_data.get("duration_s", 0.0),
                "cpu_max_pct": b_data.get("resource_usage", {}).get("cpu_max_pct", 0.0),
                "ram_max_gb": b_data.get("resource_usage", {}).get("ram_max_gb", 0.0),
            },
        },
        "scenario_comparisons": [],
    }

    scenarios_a = {s["scenario"]["id"]: s for s in a_data.get("scenarios", [])}
    scenarios_b = {s["scenario"]["id"]: s for s in b_data.get("scenarios", [])}

    for sc_id, item_a in scenarios_a.items():
        item_b = scenarios_b.get(sc_id, {})
        sc = item_a["scenario"]
        ceiling = to_float(sc.get("budget_ceiling", 0.0))

        fin_a = item_a["final_state"].get("financial_plan", {})
        strat_a = item_a["final_state"].get("final_strategy", {})
        audit_b = item_b.get("final_strategy", {})

        approved_a = to_float(fin_a.get("total_cost", 0.0))
        approved_b = to_float(audit_b.get("final_budget", 0.0))

        entry = {
            "scenario_id": sc_id,
            "scenario_name": sc["name"],
            "budget_ceiling": ceiling,
            "langgraph": {
                "duration_s": item_a.get("duration_s", 0.0),
                "iterations": item_a.get("iterations", 1),
                "approved_budget": approved_a,
                "variance": round(approved_a - ceiling, 2),
                "approval_status": strat_a.get("approval_status", "APPROVED"),
            },
            "crewai": {
                "duration_s": item_b.get("duration_s", 0.0),
                "iterations": 1,
                "approved_budget": approved_b,
                "variance": round(approved_b - ceiling, 2),
                "approval_status": audit_b.get("approval_status", "APPROVED"),
            },
        }
        comparison["scenario_comparisons"].append(entry)

    # Save JSON comparison
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    # Generate Markdown Summary for GITHUB_STEP_SUMMARY
    md = [
        "# Practical 6: Multi-Agent Business Workflow Comparative Study",
        "",
        "## Comparative Framework Analysis: LangGraph vs CrewAI",
        "",
        "| Evaluation Dimension | CrewAI | LangGraph |",
        "|:---|:---|:---|",
        "| **Paradigm & Architecture** | Role-based collaboration with agent personas | Graph-based state machine (`StateGraph`) |",
        "| **Coordination Mechanism** | Linear tasks and sequential handoffs | Nodes, edges, and typed state transitions |",
        "| **Handling Budget Overrun** | Linear auditor adjustment during review | Dynamic conditional loop (`review` -> `revision` -> `finance`) |",
        "| **State Inspection** | Task output strings | Full typed state dictionary with checkpoints |",
        f"| **Total Execution Latency** | `{b_data.get('duration_s', 0.0)}s` | `{a_data.get('duration_s', 0.0)}s` |",
        f"| **Peak CPU Utilization** | `{b_data.get('resource_usage', {}).get('cpu_max_pct', 0.0)}%` | `{a_data.get('resource_usage', {}).get('cpu_max_pct', 0.0)}%` |",
        f"| **Peak RAM Footprint** | `{b_data.get('resource_usage', {}).get('ram_max_gb', 0.0)} GB` | `{a_data.get('resource_usage', {}).get('ram_max_gb', 0.0)} GB` |",
        "",
        "## Scenario Budget Adherence & Convergence",
        "| Scenario | Ceiling ($) | LangGraph Budget ($) | LG Iterations | CrewAI Budget ($) | CR Variance ($) |",
        "|:---|---:|---:|:---:|---:|---:|",
    ]

    for item in comparison["scenario_comparisons"]:
        lg = item["langgraph"]
        cr = item["crewai"]
        md.append(
            f"| **{item['scenario_name']}** | ${item['budget_ceiling']:,.2f} | "
            f"${lg['approved_budget']:,.2f} | {lg['iterations']} | "
            f"${cr['approved_budget']:,.2f} | ${cr['variance']:+,.2f} |"
        )

    comps = comparison["scenario_comparisons"]
    n = len(comps)
    ok_a = sum(c["langgraph"]["variance"] <= 0.005 for c in comps)
    ok_b = sum(c["crewai"]["variance"] <= 0.005 for c in comps)
    loops = [c for c in comps if c["langgraph"]["iterations"] > 1]
    b_sc = {s_["scenario"]["id"]: s_ for s_ in b_data.get("scenarios", [])}
    a_sc = {s_["scenario"]["id"]: s_ for s_ in a_data.get("scenarios", [])}
    b_tasks = [t for s_ in b_sc.values() for t in s_.get("task_outputs", [])]
    b_fb = sum(bool(t.get("used_fallback")) for t in b_tasks)
    b_ov = sum(bool(t.get("guard_override")) for t in b_tasks)
    a_nodes = [a_sc[c["scenario_id"]]["node_metrics"] for c in comps if c["scenario_id"] in a_sc]
    a_fb = sum(bool(m_.get(k, {}).get("used_fallback")) for m_ in a_nodes for k in ("research", "marketing"))
    md.extend([
        "",
        "## Measured results",
        f"1. **Budget compliance**: LangGraph finished within the ceiling in {ok_a}/{n} scenarios; the CrewAI-style crew in {ok_b}/{n}.",
        f"2. **Revision loop**: LangGraph looped back through Review -> Revision -> Finance in {len(loops)} scenario(s)"
        + (f" ({', '.join(c['scenario_name'] for c in loops)})" if loops else "") + "; the sequential crew has no loop (one pass, the auditor adjusts in place).",
        f"3. **Model vs. fallback**: {b_fb}/{len(b_tasks)} CrewAI-style tasks and {a_fb}/{2 * len(a_nodes)} LangGraph LLM nodes (Research, Marketing) used the deterministic "
        f"fallback instead of the LLM's JSON; {b_ov} auditor decision(s) were overridden because the LLM approved a plan above the ceiling.",
        f"4. **Latency**: LangGraph {a_data.get('duration_s', 0.0)}s vs CrewAI-style {b_data.get('duration_s', 0.0)}s on the same CPU runner and model "
        "(Finance and Review in LangGraph are rule-based and make no LLM call; the crew calls the LLM for all four tasks).",
        "",
        "## Design notes (not measured here)",
        "- LangGraph models the workflow as a typed state machine with conditional edges, which makes a review-and-revise cycle explicit; "
        "a sequential crew expresses the same handoffs as an ordered task list and needs extra wiring for loops.",
        "- The 'CrewAI' side is a minimal in-repository implementation of the Agent/Task/Crew pattern, not the `crewai` package, so its numbers "
        "describe this implementation rather than the library.",
        "",
    ])

    summary_text = "\n".join(md)
    (out_dir / "summary.md").write_text(summary_text, encoding="utf-8")
    logger.info(f"Comparison saved to {out_dir}/ (comparison.json, summary.md)")


def main():
    parser = argparse.ArgumentParser(description="Compare Practical 6 LangGraph vs CrewAI results")
    parser.add_argument("--part-a", type=str, default="results/part_a")
    parser.add_argument("--part-b", type=str, default="results/part_b")
    parser.add_argument("--out", type=str, default="results/comparison")
    args = parser.parse_args()

    compare_runs(Path(args.part_a), Path(args.part_b), Path(args.out))


if __name__ == "__main__":
    main()
