"""Live end-to-end run of the graph: real models, fixture search. Run with:
RUN_LIVE=1 pytest -m live -s tests/live/test_live_graph.py

Capped at $0.25 by the run budget itself (a typical run costs a few cents).
"""

from __future__ import annotations

import pytest

from backend.graph import Budget, GraphDeps, memory_checkpointer, run_research
from backend.llm import AnthropicLLM
from backend.tools.search import FixtureSearchTool

pytestmark = pytest.mark.live


def test_live_graph_end_to_end():
    budget = Budget(max_usd=0.25, max_sub_questions=3, wall_clock_s=240)
    deps = GraphDeps(llm=AnthropicLLM(), search_tool=FixtureSearchTool.from_file())
    r = run_research(
        "I'm a 28 year old woman, lifting for about 3 years, 4 days a week. How many sets per week for hypertrophy?",
        deps=deps,
        budget=budget,
        checkpointer=memory_checkpointer(),
    )
    print(f"\noutcome: {r.outcome.status} - {r.outcome.message}")
    for s in r.trace:
        print(
            f"{s.index:>2} {s.node:<12} {s.sub_question_id or '':<4} {s.model or '':<18} "
            f"{s.status:<8} ${s.cost_usd:.4f} {s.latency_ms:>6}ms  {s.output_summary[:90]}"
        )
    print(f"spend: ${r.spend.cost_usd:.4f}, {r.spend.total_tokens} tokens, {r.spend.llm_calls} calls")
    if r.answer:
        print("recommendation:", r.answer.recommendation)

    assert r.outcome.status in ("answered", "degraded"), r.outcome
    assert r.answer is not None and r.answer.all_cited_ids() <= {f.id for f in r.findings}
    assert r.spend.cost_usd <= budget.max_usd
    assert [s.index for s in r.trace] == list(range(len(r.trace)))
