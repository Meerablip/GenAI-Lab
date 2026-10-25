"""CrewAI Role-Based Multi-Agent Marketing & Finance Workflow for Practical 6."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from common import LLMClient, clean_string_list, extract_json_block, to_float

logger = logging.getLogger("prac06.crewai.workflow")


def _has_schema(task_name: str, d: Any) -> bool:
    """Minimal schema an LLM answer must meet to be used instead of the deterministic fallback."""
    if not isinstance(d, dict):
        return False
    if task_name == "research":
        return bool(clean_string_list(d.get("content_pillars")))
    if task_name == "content_strategy":
        cal = d.get("weekly_calendar")
        return isinstance(cal, list) and any(isinstance(e, dict) for e in cal)
    if task_name == "financial_planning":
        return to_float(d.get("total_cost")) > 0 and isinstance(d.get("line_items"), dict)
    if task_name == "audit_finalization":
        return isinstance(d.get("approval_status"), str) and to_float(d.get("final_budget")) > 0
    return True


def _normalise(task_name: str, d: Dict[str, Any], ceiling: float) -> Dict[str, Any]:
    """Coerce an accepted LLM answer to the types the rest of the pipeline expects."""
    d = dict(d)
    if task_name == "research":
        d["market_summary"] = d["market_summary"] if isinstance(d.get("market_summary"), str) else ""
        d["content_pillars"] = clean_string_list(d.get("content_pillars"))
        d["target_personas"] = d["target_personas"] if isinstance(d.get("target_personas"), list) else []
    elif task_name == "content_strategy":
        d["strategy_narrative"] = d["strategy_narrative"] if isinstance(d.get("strategy_narrative"), str) else ""
        d["weekly_calendar"] = [e for e in d["weekly_calendar"] if isinstance(e, dict)]
    elif task_name == "financial_planning":
        total = to_float(d.get("total_cost"))
        d["budget_ceiling"] = ceiling
        d["total_cost"] = total
        d["variance"] = round(total - ceiling, 2)
        d["is_over_budget"] = total > ceiling
        d["line_items"] = {str(k): to_float(v) for k, v in d["line_items"].items()}
    elif task_name == "audit_finalization":
        final = to_float(d.get("final_budget"))
        d["final_budget"] = final
        d["variance"] = round(final - ceiling, 2)
        d["approval_status"] = d["approval_status"].strip().upper().replace(" ", "_")
        d["auditor_notes"] = d["auditor_notes"] if isinstance(d.get("auditor_notes"), str) else ""
    return d


class Agent:
    """CrewAI Agent abstraction with role, goal, and backstory."""

    def __init__(self, role: str, goal: str, backstory: str, verbose: bool = True, allow_delegation: bool = False):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.verbose = verbose
        self.allow_delegation = allow_delegation

    def __repr__(self) -> str:
        return f"Agent(role='{self.role}')"


class Task:
    """CrewAI Task abstraction with description, expected output, and assigned agent."""

    def __init__(self, description: str, expected_output: str, agent: Agent, name: str = ""):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent
        self.name = name
        self.output: Any = None

    def __repr__(self) -> str:
        return f"Task(name='{self.name}', agent='{self.agent.role}')"


class Crew:
    """CrewAI Crew orchestrator managing sequential task delegation across agents."""

    def __init__(self, agents: List[Agent], tasks: List[Task], llm: LLMClient, verbose: bool = True):
        self.agents = agents
        self.tasks = tasks
        self.llm = llm
        self.verbose = verbose

    def kickoff(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the sequential CrewAI multi-agent pipeline."""
        scenario = inputs["scenario"]
        logger.info(f"[CrewAI] Kicking off crew for scenario: {scenario['name']}")

        context: Dict[str, Any] = {"scenario": scenario}
        task_outputs: List[Dict[str, Any]] = []

        for idx, task in enumerate(self.tasks, 1):
            t0 = time.perf_counter()
            logger.info(f"[CrewAI] Agent '{task.agent.role}' starting Task {idx}: '{task.name}'")

            system_prompt = (
                f"You are {task.agent.role}.\n"
                f"Your Goal: {task.agent.goal}\n"
                f"Your Backstory: {task.agent.backstory}\n"
                f"Deliver structured, rigorous, professional output matching: {task.expected_output}"
            )

            # Build prompt incorporating previous task context
            prompt = (
                f"Task Description:\n{task.description}\n\n"
                f"Current Campaign Context:\n"
                f"Topic: {scenario['topic']}\n"
                f"Target Audience: {scenario['target_audience']}\n"
                f"Campaign Objectives: {scenario['campaign_objectives']}\n"
                f"Duration: {scenario['duration_weeks']} weeks\n"
                f"Budget Ceiling: ${scenario['budget_ceiling']:,.2f}\n"
            )

            if "research_output" in context:
                prompt += f"\nMarket Research Findings from Research Agent:\n{context['research_output'].get('market_summary', '')}\n"
                prompt += f"Identified Pillars: {', '.join(clean_string_list(context['research_output'].get('content_pillars')))}\n"

            if "content_output" in context:
                prompt += f"\nContent Roadmap from Marketing Strategist:\n{context['content_output'].get('strategy_narrative', '')}\n"

            if "finance_output" in context:
                fin = context["finance_output"]
                prompt += f"\nBudget Proposal from Financial Controller:\nTotal Cost: ${to_float(fin.get('total_cost')):,.2f} vs Ceiling: ${scenario['budget_ceiling']:,.2f}\n"

            raw_resp, telem = self.llm.generate(prompt, system=system_prompt)
            parsed_data = extract_json_block(raw_resp)
            ceiling = to_float(scenario["budget_ceiling"])

            # Use the LLM's answer only if it meets the minimal schema; otherwise fall back (and record that we did)
            used_fallback = not _has_schema(task.name, parsed_data)
            guard_override = None
            if used_fallback:
                parsed_data = self._synthesize_task_output(task.name, scenario, context)
            else:
                parsed_data = _normalise(task.name, parsed_data, ceiling)
                # Rule from the brief: a plan above the ceiling must not be signed off
                if task.name == "audit_finalization" and parsed_data["final_budget"] > ceiling + 0.005:
                    guard_override = {"llm_status": parsed_data["approval_status"], "llm_final_budget": parsed_data["final_budget"]}
                    parsed_data = {**self._synthesize_task_output(task.name, scenario, context), "llm_decision_overridden": guard_override}

            elapsed = round(time.perf_counter() - t0, 3)
            task.output = parsed_data

            if task.name == "research":
                context["research_output"] = parsed_data
            elif task.name == "content_strategy":
                context["content_output"] = parsed_data
            elif task.name == "financial_planning":
                context["finance_output"] = parsed_data
            elif task.name == "audit_finalization":
                context["audit_output"] = parsed_data

            task_outputs.append({
                "task_name": task.name,
                "agent_role": task.agent.role,
                "elapsed_s": elapsed,
                "output": parsed_data,
                "telemetry": telem,
                "used_fallback": used_fallback,
                "guard_override": guard_override,
            })
            logger.info(f"[CrewAI] Agent '{task.agent.role}' completed '{task.name}' in {elapsed}s")

        return {
            "scenario": scenario,
            "task_outputs": task_outputs,
            "research_data": context.get("research_output", {}),
            "content_strategy": context.get("content_output", {}),
            "financial_plan": context.get("finance_output", {}),
            "final_strategy": context.get("audit_output", {}),
        }

    def _synthesize_task_output(self, task_name: str, scenario: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide domain-rich fallback outputs matching CrewAI's role expectations."""
        ceiling = float(scenario["budget_ceiling"])
        weeks = scenario["duration_weeks"]

        if task_name == "research":
            return {
                "market_summary": f"Comprehensive market analysis for {scenario['topic']}. Unmet demand identified across {scenario['target_audience']}.",
                "target_personas": [
                    {"persona": "Primary Enterprise Buyer", "needs": "Rapid time-to-value, automated compliance"},
                    {"persona": "Operational Champion", "needs": "Reduced manual workload, streamlined workflows"},
                ],
                "content_pillars": [
                    "Empirical Problem Diagnosis & Industry Standards",
                    "Architectural Reliability & Core Differentiators",
                    "Enterprise Case Proof & Measured ROI",
                    "Visionary Future Trends & Category Leadership",
                ],
                "recommended_channels": scenario["channels"],
            }

        elif task_name == "content_strategy":
            pillars = context.get("research_output", {}).get("content_pillars", [
                "Problem Diagnosis", "Architectural Reliability", "Case Proof", "Future Trends"
            ])
            calendar = []
            for w in range(1, weeks + 1):
                p = pillars[(w - 1) % len(pillars)]
                calendar.append({
                    "week": w,
                    "pillar": p,
                    "channel": scenario["channels"][(w - 1) % len(scenario["channels"])],
                    "asset": f"Executive Deep-Dive on {p}",
                    "format": "Technical Whitepaper & Carousel" if w % 2 == 1 else "Executive Roundtable Digest",
                })
            return {
                "strategy_narrative": f"Sequential role-delegated campaign spanning {weeks} weeks across {len(scenario['channels'])} target channels.",
                "weekly_calendar": calendar,
                "asset_summary": {"whitepapers": max(2, weeks // 3), "briefs": weeks, "videos": max(2, weeks // 2)},
            }

        elif task_name == "financial_planning":
            # CrewAI sequential process allocates within ceiling (or simulates initial over-budget if stress test)
            is_stress = scenario.get("force_initial_overbudget", False)
            if is_stress:
                # In sequential CrewAI, the Finance agent models initial cost request
                total = float(scenario.get("initial_unconstrained_estimate", 24800.0))
                prod = round(total * 0.35, 2)
                media = round(total * 0.35, 2)
                inf = round(total * 0.18, 2)
                sw = round(total * 0.08, 2)
                cont = round(total - (prod + media + inf + sw), 2)
            else:
                target = round(ceiling * 0.965, 2)
                prod = round(target * 0.30, 2)
                media = round(target * 0.40, 2)
                inf = round(target * 0.15, 2)
                sw = round(target * 0.07, 2)
                cont = round(target - (prod + media + inf + sw), 2)
                total = round(prod + media + inf + sw + cont, 2)

            return {
                "budget_ceiling": ceiling,
                "total_cost": total,
                "variance": round(total - ceiling, 2),
                "is_over_budget": total > ceiling,
                "line_items": {
                    "content_production": prod,
                    "media_distribution_ads": media,
                    "influencer_partnerships": inf,
                    "tools_software": sw,
                    "contingency_reserve": cont,
                },
                "unit_economics": {
                    "projected_reach": int(scenario.get("key_kpis", {}).get("projected_reach", total * 3.5)),
                    "estimated_cac_or_cpa": round(total / max(1, scenario.get("key_kpis", {}).get("demo_requests", 50)), 2),
                },
            }

        elif task_name == "audit_finalization":
            fin = context.get("finance_output", {})
            total = fin.get("total_cost", ceiling)
            is_over = fin.get("is_over_budget", False)

            if is_over:
                # In CrewAI sequential delegation, the Auditor adjusts the plan in-place during the final review
                adjusted_total = round(ceiling * 0.97, 2)
                adjusted_items = {
                    "content_production": round(adjusted_total * 0.30, 2),
                    "media_distribution_ads": round(adjusted_total * 0.40, 2),
                    "influencer_partnerships": round(adjusted_total * 0.15, 2),
                    "tools_software": round(adjusted_total * 0.07, 2),
                    "contingency_reserve": round(adjusted_total * 0.08, 2),
                }
                return {
                    "approval_status": "APPROVED_WITH_MODIFICATIONS",
                    "auditor_notes": f"Auditor intercepted ${total:.2f} draft, flagged +${fin.get('variance', 0):.2f} overrun, and applied direct sequential budget capping to ${adjusted_total:.2f}.",
                    "final_budget": adjusted_total,
                    "adjusted_line_items": adjusted_items,
                    "variance": round(adjusted_total - ceiling, 2),
                }
            else:
                return {
                    "approval_status": "APPROVED",
                    "auditor_notes": f"Auditor verified budget of ${total:.2f} satisfies authorized ceiling ${ceiling:.2f}.",
                    "final_budget": total,
                    "adjusted_line_items": fin.get("line_items", {}),
                    "variance": round(total - ceiling, 2),
                }
        return {}


def build_crew(llm: LLMClient) -> Crew:
    """Instantiate the 4 specialized CrewAI agents and sequential tasks."""
    # 1. Research Agent
    researcher = Agent(
        role="Lead Market Research Analyst",
        goal="Uncover high-leverage market angles, competitive whitespace, and core audience pain points",
        backstory="Senior market intelligence strategist with 15+ years experience in B2B tech and consumer markets.",
        allow_delegation=True,
    )

    # 2. Marketing Agent
    strategist = Agent(
        role="Content Marketing Strategist",
        goal="Architect an end-to-end editorial calendar and multi-channel asset roadmap aligned with audience intent",
        backstory="Accomplished marketing director specialized in full-funnel content architectures that drive pipeline.",
        allow_delegation=True,
    )

    # 3. Finance Agent
    controller = Agent(
        role="Budget & Financial Controller",
        goal="Construct rigorous unit-cost budget models and maximize ROI within authorized expenditure ceilings",
        backstory="Financial analyst with deep domain expertise in digital media spend, production economics, and conversion modeling.",
        allow_delegation=False,
    )

    # 4. Review Agent
    auditor = Agent(
        role="Executive Strategy Director & Auditor",
        goal="Audit content and financial proposals, enforce budget compliance, and authorize executive sign-off",
        backstory="Managing partner with ultimate fiduciary and strategic authority over cross-functional marketing campaigns.",
        allow_delegation=True,
    )

    # Define tasks
    t1 = Task(
        name="research",
        description="Conduct comprehensive market, competitor, and audience analysis for the campaign topic.",
        expected_output="JSON report with market_summary, target_personas, content_pillars, and recommended_channels.",
        agent=researcher,
    )

    t2 = Task(
        name="content_strategy",
        description="Develop weekly publishing calendar, asset deliverables, and channel distribution matrix.",
        expected_output="JSON object with strategy_narrative, weekly_calendar, and asset_summary.",
        agent=strategist,
    )

    t3 = Task(
        name="financial_planning",
        description="Estimate line-item expenditures (production, media, influencers, tools, contingency) and test against ceiling.",
        expected_output="JSON object with line_items, total_cost, variance, and unit_economics.",
        agent=controller,
    )

    t4 = Task(
        name="audit_finalization",
        description="Conduct fiduciary audit, check budget compliance, and produce final executive sign-off.",
        expected_output="JSON object with approval_status, auditor_notes, final_budget, and variance.",
        agent=auditor,
    )

    return Crew(agents=[researcher, strategist, controller, auditor], tasks=[t1, t2, t3, t4], llm=llm)
