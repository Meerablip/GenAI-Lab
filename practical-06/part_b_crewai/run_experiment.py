"""Experiment execution driver for Practical 6 - Part B (CrewAI Multi-Agent Workflow)."""
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
from part_b_crewai.crew_workflow import build_crew

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("prac06.crewai.runner")

PACKAGES = [
    "pydantic",
    "psutil",
    "requests",
    "tabulate",
]


def generate_markdown_summary(results: dict) -> str:
    lines = [
        "# Practical 6 (Part B) - CrewAI Multi-Agent Business Workflow",
        "",
        "## Executive Overview",
        f"- **Framework**: CrewAI (Role-based collaboration with sequential task handoffs)",
        f"- **Model**: `{results['llm_model']}` ({results['llm_engine']})",
        f"- **Host**: {results['env']['cpu_model']} ({results['env']['cpu_logical_cores']} vCPUs, {results['env']['ram_total_gb']} GB RAM)",
        f"- **Total Duration**: {results['duration_s']}s",
        "",
        "## CrewAI Agent Team",
        "1. **Lead Market Research Analyst**: Identifies market demand, buyer personas, and content pillars.",
        "2. **Content Marketing Strategist**: Designs weekly editorial roadmap and channel deliverables.",
        "3. **Budget & Financial Controller**: Formulates line-item cost projections and checks ceiling.",
        "4. **Executive Strategy Director & Auditor**: Performs fiduciary audit and certifies campaign sign-off.",
        "",
        "## Scenario Execution Summary",
        "| Scenario | Ceiling ($) | Initial Cost ($) | Approved Budget ($) | Variance ($) | Auditor Decision |",
        "|:---|---:|---:|---:|---:|:---:|",
    ]

    for item in results["scenarios"]:
        s = item["scenario"]
        fin = item.get("financial_plan", {})
        audit = item.get("final_strategy", {})
        ceiling = to_float(s.get("budget_ceiling", 0.0))
        init_cost = to_float(fin.get("total_cost", ceiling))
        final_cost = to_float(audit.get("final_budget", init_cost))
        var = to_float(audit.get("variance", final_cost - ceiling))
        status = audit.get("approval_status", "APPROVED")
        lines.append(
            f"| **{s['name']}** | ${ceiling:,.2f} | ${init_cost:,.2f} | "
            f"${final_cost:,.2f} | ${var:+,.2f} | **{status}** |"
        )

    lines.extend(["", "## LLM output vs. deterministic fallback (per task)",
                  "| Scenario | Research | Content | Finance | Audit | Auditor decision overridden by the budget rule |",
                  "|:---|:---:|:---:|:---:|:---:|:---:|"])
    for item in results["scenarios"]:
        by = {t["task_name"]: t for t in item.get("task_outputs", [])}
        cell = lambda n: "fallback" if by.get(n, {}).get("used_fallback") else "LLM JSON"  # noqa: E731
        ov = by.get("audit_finalization", {}).get("guard_override")
        lines.append(f"| {item['scenario']['name']} | {cell('research')} | {cell('content_strategy')} | {cell('financial_planning')} | "
                     f"{cell('audit_finalization')} | {('yes - LLM said ' + str(ov['llm_status']) + ' at $' + format(ov['llm_final_budget'], ',.2f')) if ov else 'no'} |")
    lines.extend([
        "",
        "Note: the `Agent`/`Task`/`Crew` classes in `crew_workflow.py` are a minimal in-repository implementation of the CrewAI "
        "role/task/sequential-crew pattern; the `crewai` package itself is not used. The LLM's answer is accepted only if it meets a "
        "minimal schema; otherwise a deterministic template is used (table above).",
    ])
    lines.extend([
        "",
        "## Architectural Observations",
        "- **Linear Role Delegation**: CrewAI processes tasks through sequential agent specialization. The financial controller hands off the cost model to the auditor, who either gives straight approval or applies adjustments directly during fiduciary review.",
        "- **Comparison Note**: Unlike LangGraph's dynamic StateGraph conditional loop (which cycles back from review to revision), CrewAI executes a linear task sequence unless custom tool loops or recursive crews are instantiated.",
        "",
    ])
    return "\n".join(lines)


def generate_detailed_reports(results: dict) -> str:
    lines = ["# Detailed Multi-Agent Campaign Strategies (CrewAI)\n"]
    for item in results["scenarios"]:
        s = item["scenario"]
        fin = item.get("financial_plan", {})
        audit = item.get("final_strategy", {})
        research = item.get("research_data", {})
        mktg = item.get("content_strategy", {})

        ceiling = to_float(s.get("budget_ceiling", 0.0))
        final_b = to_float(audit.get("final_budget", fin.get("total_cost", 1.0)))

        lines.extend([
            f"## Scenario: {s['name']}",
            f"**Topic**: {s['topic']}  ",
            f"**Audience**: {s['target_audience']}  ",
            f"**Campaign Objective**: {s['campaign_objectives']}  ",
            f"**Budget Ceiling**: ${ceiling:,.2f} | **Approved Budget**: ${final_b:,.2f}  ",
            f"**Auditor Sign-off**: {audit.get('approval_status', 'APPROVED')}  ",
            f"**Auditor Notes**: _{audit.get('auditor_notes', '')}_  ",
            "",
            "### 1. Market Research Findings",
            f"_{research.get('market_summary', '')}_",
            "",
            "**Strategic Content Pillars:**",
        ])
        for p in research.get("content_pillars", []):
            lines.append(f"- {p}")

        lines.extend([
            "",
            "### 2. Approved Budget Distribution",
            "| Expense Category | Allocation ($) | Share (%) |",
            "|:---|---:|---:|",
        ])
        items = audit.get("adjusted_line_items", fin.get("line_items", {}))
        for cat, amt in items.items():
            amt_f = to_float(amt)
            pct = round((amt_f / max(1.0, final_b)) * 100, 1)
            lines.append(f"| {cat.replace('_', ' ').title()} | ${amt_f:,.2f} | {pct}% |")
        lines.append(f"| **Total** | **${final_b:,.2f}** | **100.0%** |")

        lines.extend([
            "",
            "### 3. Agent Task Handoff Timings",
            "| Task Name | Responsible Agent Role | Execution Time (s) |",
            "|:---|:---|---:|",
        ])
        for t in item.get("task_outputs", []):
            lines.append(f"| {t['task_name']} | {t['agent_role']} | {t['elapsed_s']}s |")

        lines.append("\n---\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run Practical 6 - Part B: CrewAI Multi-Agent Workflow")
    parser.add_argument("--scenarios", type=str, default=str(PRAC_DIR / "campaign_scenarios.json"))
    parser.add_argument("--out", type=str, default="results/part_b")
    parser.add_argument("--model", type=str, default="llama3.2:1b")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test mode")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing CrewAI multi-agent test suite...")
    env_info = snapshot_env(PACKAGES)
    llm = LLMClient(model=args.model, smoke=args.smoke)
    crew = build_crew(llm)

    scenarios = load_json(Path(args.scenarios))
    logger.info(f"Loaded {len(scenarios)} campaign scenarios from {args.scenarios}")

    monitor = ResourceMonitor(interval=0.25)
    monitor.start()
    t_start = time.perf_counter()

    scenario_results = []
    for sc in scenarios:
        logger.info(f"\n=======================================================")
        logger.info(f"Kicking Off CrewAI Scenario: {sc['name']} (Ceiling: ${sc['budget_ceiling']})")
        logger.info(f"=======================================================")

        t_sc_start = time.perf_counter()
        result = crew.kickoff({"scenario": sc})
        sc_duration = round(time.perf_counter() - t_sc_start, 3)

        result["duration_s"] = sc_duration
        scenario_results.append(result)

    total_duration = round(time.perf_counter() - t_start, 3)
    resource_usage = monitor.stop()

    summary_data = {
        "framework": "crewai",
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

    logger.info(f"\nCompleted CrewAI multi-agent experiments in {total_duration}s.")
    logger.info(f"Artifacts saved to {out_dir}/ (results.json, summary.md, strategy_reports.md)")


if __name__ == "__main__":
    main()
