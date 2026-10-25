"""Experiment execution driver for Practical 6 - Part A (LangGraph Multi-Agent Workflow)."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add project root and practical directory to path
HERE = Path(__file__).resolve().parent
PRAC_DIR = HERE.parent
if str(PRAC_DIR) not in sys.path:
    sys.path.insert(0, str(PRAC_DIR))

from common import LLMClient, ResourceMonitor, load_json, snapshot_env, write_json, to_float
from part_a_langgraph.state import CampaignState
from part_a_langgraph.workflow import build_campaign_graph, get_mermaid_diagram

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("prac06.langgraph.runner")

PACKAGES = [
    "langgraph",
    "langchain_core",
    "pydantic",
    "psutil",
    "requests",
    "tabulate",
]


def generate_markdown_summary(results: dict) -> str:
    lines = [
        "# Practical 6 (Part A) - LangGraph Multi-Agent Business Workflow",
        "",
        "## Executive Overview",
        f"- **Framework**: LangGraph (StateGraph with conditional routing and dynamic loops)",
        f"- **Model**: `{results['llm_model']}` ({results['llm_engine']})",
        f"- **Host**: {results['env']['cpu_model']} ({results['env']['cpu_logical_cores']} vCPUs, {results['env']['ram_total_gb']} GB RAM)",
        f"- **Total Duration**: {results['duration_s']}s",
        "",
        "## Campaign Execution & Budget Alignment",
        "| Scenario | Target Audience | Ceiling ($) | Initial Cost ($) | Approved Cost ($) | Variance ($) | Iterations | Status |",
        "|:---|:---|---:|---:|---:|---:|:---:|:---:|",
    ]

    for item in results["scenarios"]:
        s = item["scenario"]
        fin = item["final_state"].get("financial_plan", {})
        rev = item["final_state"].get("review_feedback", {})
        hist = item["final_state"].get("revision_history", [])
        ceiling = to_float(s.get("budget_ceiling", 0.0))
        init_cost = to_float(hist[0]["total_cost"]) if hist else to_float(fin.get("total_cost", 0.0))
        app_cost = to_float(fin.get("total_cost", 0.0))
        var = to_float(fin.get("variance", 0.0))
        lines.append(
            f"| **{s['name']}** | {s['target_audience'][:30]}... | "
            f"${ceiling:,.2f} | ${init_cost:,.2f} | ${app_cost:,.2f} | "
            f"${var:+,.2f} | {item['iterations']} | **{rev.get('status', 'APPROVED')}** |"
        )

    lines.extend(["", "## LLM output vs. deterministic fallback",
                  "| Scenario | Research | Marketing | Calendar fields from LLM |", "|:---|:---:|:---:|:---:|"])
    for item in results["scenarios"]:
        nm = item["node_metrics"]
        r_fb = nm.get("research", {}).get("used_fallback")
        m = nm.get("marketing", {})
        lines.append(f"| {item['scenario']['name']} | {'canned fallback' if r_fb else 'LLM JSON'} | "
                     f"{'canned fallback' if m.get('used_fallback') else 'LLM JSON'} | "
                     f"{m.get('calendar_fields_from_llm', 0)}/{m.get('calendar_fields_total', 0)} |")
    lines.extend([
        "",
        "Finance and Review are rule-based nodes (they do not call the LLM): budget lines are fixed shares of the ceiling "
        "and the Review verdict is an over/under-ceiling check. Research and Marketing call the LLM and fall back to a "
        "deterministic template when its JSON cannot be parsed (see the table above).",
        "",
        "## Multi-Agent State Machine Architecture",
        get_mermaid_diagram(),
        "",
        "## Key Findings",
    ])
    for item in results["scenarios"]:
        s_ = item["scenario"]
        hist = item["final_state"].get("revision_history", [])
        fin = item["final_state"].get("financial_plan", {})
        ceiling = to_float(s_.get("budget_ceiling", 0.0))
        if hist:
            first = to_float(hist[0].get("total_cost", 0.0))
            over = first - ceiling
            lines.append(
                f"- **{s_['name']}**: the Review step rejected the first estimate of ${first:,.2f} "
                f"(+${over:,.2f}, +{(over / max(ceiling, 1.0)) * 100:.1f}% over the ${ceiling:,.2f} ceiling) and the conditional edge "
                f"looped back; the plan was approved on iteration {item['iterations']} at ${to_float(fin.get('total_cost', 0.0)):,.2f}.")
        else:
            lines.append(f"- **{s_['name']}**: approved on the first iteration at ${to_float(fin.get('total_cost', 0.0)):,.2f} "
                         f"against a ${ceiling:,.2f} ceiling (no revision needed).")
    lines.extend([
        "- **Graph Traceability**: state and timing were logged at every node transition (`execution_log`, `node_metrics` in results.json).",
        "",
    ])
    return "\n".join(lines)


def generate_detailed_reports(results: dict) -> str:
    lines = ["# Detailed Multi-Agent Campaign Strategies (LangGraph)\n"]
    for item in results["scenarios"]:
        s = item["scenario"]
        fin = item["final_state"].get("financial_plan", {})
        strat = item["final_state"].get("final_strategy", {})
        research = item["final_state"].get("research_data", {})
        mktg = item["final_state"].get("content_strategy", {})
        hist = item["final_state"].get("revision_history", [])

        ceiling = to_float(s.get("budget_ceiling", 0.0))
        total_b = to_float(fin.get("total_cost", 0.0))

        lines.extend([
            f"## Scenario: {s['name']}",
            f"**Topic**: {s['topic']}  ",
            f"**Audience**: {s['target_audience']}  ",
            f"**Campaign Objective**: {s['campaign_objectives']}  ",
            f"**Duration**: {s['duration_weeks']} weeks | **Ceiling**: ${ceiling:,.2f} | **Approved Budget**: ${total_b:,.2f}  ",
            f"**Status**: {strat.get('approval_status', 'APPROVED')} ({item['iterations']} iteration(s))  ",
            "",
            "### 1. Market Research & Strategic Positioning",
            f"_{research.get('market_summary', '')}_",
            "",
            "**Strategic Content Pillars:**",
        ])
        for p in research.get("content_pillars", []):
            lines.append(f"- {p}")

        lines.extend([
            "",
            "### 2. Financial Allocation & Unit Economics",
            "| Expense Category | Allocation ($) | Share (%) |",
            "|:---|---:|---:|",
        ])
        for cat, amt in fin.get("line_items", {}).items():
            amt_f = to_float(amt)
            pct = round((amt_f / max(1.0, total_b)) * 100, 1)
            lines.append(f"| {cat.replace('_', ' ').title()} | ${amt_f:,.2f} | {pct}% |")
        lines.append(f"| **Total** | **${total_b:,.2f}** | **100.0%** |")

        if hist:
            lines.extend([
                "",
                "### 3. Review & Iterative Audit Log",
            ])
            for h in hist:
                h_cost = to_float(h.get('total_cost', 0.0))
                h_var = to_float(h.get('variance', 0.0))
                lines.append(f"- **Iteration {h['iteration']} Rejection**: Total proposed cost was ${h_cost:,.2f} (Variance: +${h_var:,.2f}). Directives: {'; '.join(h['directives'])}")
            lines.append(f"- **Iteration {item['iterations']} Approval**: Revised cost reduced to ${total_b:,.2f}, clearing ceiling constraint.")

        lines.extend([
            "",
            "### 4. Sample Editorial Calendar (First 4 Weeks)",
            "| Week | Channel | Content Focus | Format | Goal |",
            "|:---:|:---|:---|:---|:---|",
        ])
        for entry in [e for e in mktg.get("weekly_calendar", []) if isinstance(e, dict)][:4]:
            lines.append(
                f"| W{entry.get('week_number', '?')} | {entry.get('primary_channel', '')} | {entry.get('pillar', '')} | "
                f"{entry.get('format', '')} | {entry.get('conversion_goal', '')} |"
            )

        lines.append("\n---\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run Practical 6 - Part A: LangGraph Multi-Agent Workflow")
    parser.add_argument("--scenarios", type=str, default=str(PRAC_DIR / "campaign_scenarios.json"))
    parser.add_argument("--out", type=str, default="results/part_a")
    parser.add_argument("--model", type=str, default="llama3.2:1b")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test mode")
    parser.add_argument("--max-iterations", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing LangGraph multi-agent test suite...")
    env_info = snapshot_env(PACKAGES)
    llm = LLMClient(model=args.model, smoke=args.smoke)
    workflow = build_campaign_graph(llm)

    scenarios = load_json(Path(args.scenarios))
    logger.info(f"Loaded {len(scenarios)} campaign scenarios from {args.scenarios}")

    monitor = ResourceMonitor(interval=0.25)
    monitor.start()
    t_start = time.perf_counter()

    scenario_results = []
    for sc in scenarios:
        logger.info(f"\n=======================================================")
        logger.info(f"Executing Scenario: {sc['name']} (Ceiling: ${sc['budget_ceiling']})")
        logger.info(f"=======================================================")

        initial_state: CampaignState = {
            "scenario": sc,
            "iteration": 1,
            "max_iterations": args.max_iterations,
            "revision_history": [],
            "node_metrics": {},
            "execution_log": [],
        }

        t_sc_start = time.perf_counter()
        final_state = workflow.invoke(initial_state)
        sc_duration = round(time.perf_counter() - t_sc_start, 3)

        rev_history = final_state.get("revision_history", [])
        total_iterations = len(rev_history) + 1

        scenario_results.append({
            "scenario": sc,
            "duration_s": sc_duration,
            "iterations": total_iterations,
            "node_metrics": final_state.get("node_metrics", {}),
            "execution_log": final_state.get("execution_log", []),
            "final_state": {
                "financial_plan": final_state.get("financial_plan", {}),
                "review_feedback": final_state.get("review_feedback", {}),
                "revision_history": final_state.get("revision_history", []),
                "final_strategy": final_state.get("final_strategy", {}),
                "research_data": final_state.get("research_data", {}),
                "content_strategy": final_state.get("content_strategy", {}),
            },
        })

    total_duration = round(time.perf_counter() - t_start, 3)
    resource_usage = monitor.stop()

    summary_data = {
        "framework": "langgraph",
        "llm_model": args.model,
        "llm_engine": "ollama" if llm.is_connected and not args.smoke else "structured_synthesizer",
        "env": env_info,
        "duration_s": total_duration,
        "resource_usage": resource_usage,
        "scenarios": scenario_results,
    }

    # Write results
    write_json(out_dir / "results.json", summary_data)

    md_summary = generate_markdown_summary(summary_data)
    (out_dir / "summary.md").write_text(md_summary, encoding="utf-8")

    detailed_md = generate_detailed_reports(summary_data)
    (out_dir / "strategy_reports.md").write_text(detailed_md, encoding="utf-8")

    (out_dir / "architecture.txt").write_text(get_mermaid_diagram(), encoding="utf-8")

    logger.info(f"\nCompleted LangGraph multi-agent experiments in {total_duration}s.")
    logger.info(f"Artifacts saved to {out_dir}/ (results.json, summary.md, strategy_reports.md, architecture.txt)")


if __name__ == "__main__":
    main()
