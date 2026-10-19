"""Deterministic fakes for graph tests.

Researchers run in parallel threads, so model responses are callables keyed on
the request content (which records / finding ids are in the prompt), never on
call order.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Iterable
from typing import Any

from backend.agents.researcher import ExtractionBatch, StudyExtraction
from backend.agents.writer import WriterDraft
from backend.llm import FakeLLM, LLMRequest
from backend.schemas import SearchResponse, SearchState, StudyRecord
from backend.tests.conftest import load_fixture
from backend.tools.search import FixtureSearchTool

STUDY_TYPE: dict[str, str] = {
    "FIXTURE-0001": "meta_analysis",
    "FIXTURE-0002": "rct",
    "FIXTURE-0003": "rct",
    "FIXTURE-0004": "meta_analysis",  # metadata turns it into a preprint
    "FIXTURE-0005": "rct",
    "FIXTURE-0007": "animal",
    "FIXTURE-0008": "meta_analysis",
    "FIXTURE-0009": "rct",
    "FIXTURE-0010": "meta_analysis",
    "FIXTURE-0011": "cohort",
}
POPULATION: dict[str, str] = {
    "FIXTURE-0001": "390 adults, mostly young untrained and recreationally trained men and women",
    "FIXTURE-0002": "34 resistance-trained men",
    "FIXTURE-0003": "40 trained adults (22 men, 18 women)",
    "FIXTURE-0004": "67 studies, 2058 participants",
    "FIXTURE-0005": "28 untrained young women",
    "FIXTURE-0007": "rats",
    "FIXTURE-0008": "1863 adults across 49 trials",
    "FIXTURE-0009": "40 overweight young men in an energy deficit",
    "FIXTURE-0010": "25 studies of trained and untrained adults",
    "FIXTURE-0011": "2066 adults aged 70-79",
}
_RECORD = re.compile(r'<record source_id="([^"]+)">')
_FINDING = re.compile(r"^\[([^\]]+)\] \((\w+), grade (\w+)", re.M)

INTAKE = load_fixture("intake_sets.json")  # in scope; trained woman, 4 days a week

SUB_QUESTIONS: dict[str, dict[str, Any]] = {
    "volume": {
        "question": "Is there a dose-response relationship between weekly sets per muscle and hypertrophy in resistance-trained adults?",
        "rationale": "Core question.",
        "search_terms": ["resistance training volume hypertrophy meta-analysis", "weekly sets muscle thickness trained"],
    },
    "plateau": {
        "question": "At what weekly set volume do hypertrophy gains plateau or diminish in trained individuals?",
        "rationale": "Upper bound.",
        "search_terms": ["high volume resistance training plateau hypertrophy", "diminishing returns weekly sets muscle growth"],
    },
    "frequency": {
        "question": "Does weekly training frequency change hypertrophy when weekly set volume is equated?",
        "rationale": "The user trains 4 days a week.",
        "search_terms": ["training frequency volume equated hypertrophy meta-analysis"],
    },
    "animal": {
        "question": "Does progressive overload volume increase muscle hypertrophy in a ladder-climbing model?",
        "rationale": "Mechanism.",
        "search_terms": ["rat ladder climbing hypertrophy"],
    },
    "empty": {
        "question": "Does creatine timing relative to training change kidney function markers in lifters?",
        "rationale": "Nothing in the fixtures answers this.",
        "search_terms": ["creatine kidney function lifters"],
    },
}


def plan(*keys: str) -> dict[str, Any]:
    return {"sub_questions": [SUB_QUESTIONS[k] for k in keys]}


def _snippet(rec: StudyRecord) -> str:
    text = rec.full_text or rec.abstract
    m = re.search(r"Results: (.+?\.)(\s|$)", text)
    return m.group(1) if m else re.split(r"(?<=\.)\s", rec.abstract)[0]


def extractor(
    records: Iterable[StudyRecord],
    population: dict[str, str] | None = None,
    on_call: Callable[[LLMRequest[Any]], None] | None = None,
    use_max_tokens: bool = False,
) -> Callable[[LLMRequest[Any]], ExtractionBatch | str]:
    """Researcher response: one relevant extraction per record in the prompt.

    `use_max_tokens` pads the JSON so the call bills ~its whole max_tokens, like a verbose model.
    """
    by_id = {r.source_id: r for r in records}
    pops = {**POPULATION, **(population or {})}

    def respond(request: LLMRequest[Any]) -> ExtractionBatch | str:
        if on_call:
            on_call(request)
        studies = []
        for sid in _RECORD.findall(request.user):
            rec = by_id[sid]
            snippet = _snippet(rec)
            studies.append(
                StudyExtraction(
                    source_id=sid,
                    relevant=True,
                    claim=snippet,
                    snippet=snippet,
                    study_type=STUDY_TYPE.get(sid, "rct"),  # type: ignore[arg-type]
                    population=pops.get(sid, "adults"),
                    sample_size=40,
                    effect=None,
                    funding_conflict=False,
                )
            )
        batch = ExtractionBatch(studies=studies)
        if use_max_tokens:
            text = batch.model_dump_json()
            return text + " " * max(0, request.max_tokens * 4 - len(text) - 4)
        return batch

    return respond


def writer_response(
    status: str = "consensus", confidence: str = "moderate", only_types: set[str] | None = None
) -> Callable[[LLMRequest[Any]], WriterDraft]:
    """Writer response citing the strongest findings that appear in the prompt."""

    def respond(request: LLMRequest[Any]) -> WriterDraft:
        found = _FINDING.findall(request.user)
        if only_types is not None:
            ids = [fid for fid, st, _ in found if st in only_types]
        else:
            ids = [fid for fid, st, _ in found if st not in ("animal", "in_vitro", "preprint")]
        ids = ids[:3] or [found[0][0]]
        return WriterDraft(
            status=status,  # type: ignore[arg-type]
            recommendation="Aim for 10-20 hard sets per muscle per week.",
            confidence=confidence,  # type: ignore[arg-type]
            why="More weekly sets produced more growth, with smaller gains at high volumes.",
            positions=[
                {
                    "stance": "More weekly volume produces more hypertrophy",
                    "supporting_findings": ids,
                    "evidence_grade": "moderate",
                    "best_counterargument": "Returns diminish at high volumes.",
                }
            ],
            what_would_change_this="Trials in trained women at high volumes.",
            applies_to_you="You are trained, so the trained-lifter trials apply.",
            practical_protocol="Do 12-14 sets per muscle per week across 4 sessions.",
            whats_new=None,
            cited_finding_ids=ids,
        )

    return respond


BAD_WRITER_DRAFT = {
    # Every position cites an unknown id -> WriterError("no position backed by a known finding").
    "status": "consensus",
    "recommendation": "x",
    "confidence": "low",
    "why": "x",
    "positions": [
        {"stance": "s", "supporting_findings": ["made-up"], "evidence_grade": "low", "best_counterargument": "c"}
    ],
    "what_would_change_this": "x",
    "applies_to_you": "x",
    "practical_protocol": "x",
    "whats_new": None,
    "cited_finding_ids": [],
}


def fake_llm(search: FixtureSearchTool, sub_questions: Iterable[str] = ("volume", "plateau", "frequency"), **over: Any) -> FakeLLM:
    responses: dict[str, Any] = {
        "intake": INTAKE,
        "planner": plan(*sub_questions),
        "researcher": extractor(search.records),
        "writer": writer_response(),
    }
    responses.update(over)
    return FakeLLM(responses)


class ScriptedSearch:
    """Fixture records, with per-query failures that can heal after N calls.

    `failures` maps a query substring to (state, times); times=None fails forever.
    """

    name = "scripted"

    def __init__(self, base: FixtureSearchTool, failures: dict[str, tuple[SearchState, int | None]] | None = None, raise_on: str | None = None):
        self.base = base
        self.records = base.records
        self.failures = dict(failures or {})
        self.raise_on = raise_on
        self.queries: list[str] = []
        self._lock = threading.Lock()
        self._count: dict[str, int] = {}

    def search(self, query: str, max_results: int = 10) -> SearchResponse:
        with self._lock:
            self.queries.append(query)
            if self.raise_on and self.raise_on in query:
                raise RuntimeError("search backend exploded")
            for pattern, (state, times) in self.failures.items():
                if pattern in query:
                    n = self._count.get(pattern, 0)
                    self._count[pattern] = n + 1
                    if times is None or n < times:
                        return SearchResponse(query=query, state=state, message=f"injected {state}")
        return self.base.search(query, max_results=max_results)


class FakeClock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t
        self._lock = threading.Lock()

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.t += seconds
