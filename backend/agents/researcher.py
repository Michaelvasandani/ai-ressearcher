"""Researcher: one sub-question in, graded Finding records + a SearchStatus out.

Flow (search and filtering are code; only extraction uses the model):
1. Run the sub-question's search terms through the SearchTool (capped).
2. Deduplicate by source id; drop retracted papers and records without
   provenance, recording each drop and its reason.
3. One model call extracts a claim, study design, population, sample size,
   effect and a verbatim snippet from each remaining record.
4. Code attaches provenance (ids, URLs, dates come from the search record,
   never from the model) and a deterministic evidence grade.

Expected failures never raise: they come back in SearchStatus.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from backend.agents.grading import grade_finding
from backend.llm import LLMClient, LLMError, LLMRequest
from backend.schemas import (
    DroppedRecord,
    Finding,
    ResearchResult,
    SearchResponse,
    SearchStatus,
    StudyRecord,
    StudyType,
    SubQuestion,
    UserProfile,
    utcnow,
)
from backend.tools.search import SearchTool

AGENT = "researcher"
MAX_SEARCHES = 3
MAX_RESULTS_PER_SEARCH = 10
MAX_RECORDS_TO_EXTRACT = 8
MAX_TEXT_CHARS = 6_000  # per record; methods + results fit well within this


class StudyExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    relevant: bool
    claim: str
    snippet: str
    study_type: StudyType
    population: str
    sample_size: int | None
    effect: str | None
    funding_conflict: bool | None


class ExtractionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    studies: list[StudyExtraction]


SYSTEM_PROMPT = """\
You extract findings from study records for an evidence-based training and nutrition coach. You \
get one research sub-question and several study records. Return one entry per record, with its \
source_id copied exactly.

For each record:
- relevant: true only if the study's results bear directly on the sub-question.
- claim: one sentence stating what this study found that answers the sub-question, with its \
numbers. Do not generalize beyond the study's population or design.
- snippet: a verbatim quote (one or two sentences) from the record text that states the result \
in the claim, including its numbers. Prefer the full text's results over the abstract. Copy it \
exactly; do not paraphrase. Never quote funding or methods boilerplate as the snippet.
- study_type: the design. meta_analysis (pooled analysis of studies), rct (randomized trial), \
cohort, case_control, case_report, review (narrative review, no pooling), animal, in_vitro, \
preprint (anything not yet peer reviewed, whatever its design).
- population: who was studied: number, sex, age, training status, and condition (e.g. energy deficit).
- sample_size: total participants, or null if not stated. For a meta-analysis, the pooled n if \
stated.
- effect: direction and size of the main effect with units, or null if not reported.
- funding_conflict: true if industry funding or a conflict of interest is stated, false if the \
text states no funding or no conflict, null if not mentioned.

Use only the record text. Do not add outside knowledge."""


def _render_record(r: StudyRecord) -> str:
    text = r.full_text or ""
    body = f"Abstract: {r.abstract}"
    if text:
        body += f"\nFull text (methods/results): {text[:MAX_TEXT_CHARS]}"
    types = ", ".join(r.publication_types) or "unknown"
    return (
        f'<record source_id="{r.source_id}">\nTitle: {r.title}\nPublished: {r.published.isoformat()}\n'
        f"Publication types: {types}\nPreprint: {r.is_preprint}\n{body}\n</record>"
    )


def _render_user(sq: SubQuestion, records: list[StudyRecord], profile: UserProfile | None) -> str:
    parts = [f"Sub-question: {sq.question}"]
    if profile and not profile.is_empty():
        parts.append(f"User profile (context only): {profile.model_dump_json(exclude_none=True)}")
    parts.append("Records:")
    parts.extend(_render_record(r) for r in records)
    return "\n\n".join(parts)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


_NUMBER = re.compile(r"\d+(?:\.\d+)?%?")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(text))


def _choose_snippet(snippet: str, claim: str, record: StudyRecord) -> str:
    """Return a verbatim snippet from the record that supports the claim.

    The model's snippet is kept when it is really in the record and, if the
    claim has numbers, shares at least one of them. Otherwise the record
    sentence sharing the most numbers with the claim is used (results text
    preferred), so later citation checks can find the claim's numbers.
    """
    source = f"{record.full_text or ''} {record.abstract}"
    claim_numbers = _numbers(claim)
    snippet = snippet.strip()
    if snippet and _norm(snippet) in _norm(source):
        if not claim_numbers or claim_numbers & _numbers(snippet):
            return snippet
    sentences = [s.strip() for s in _SENTENCE_END.split(source) if s.strip()]
    if not sentences:
        return source.strip()[:600]

    def score(sentence: str) -> int:
        # Effect sizes (decimals, percentages) identify a result better than counts like "8 weeks".
        return sum(2 if ("." in n or "%" in n) else 1 for n in claim_numbers & _numbers(sentence))

    best = max(sentences, key=score)  # first wins ties
    return best[:600]


def _study_type(extracted: StudyType, record: StudyRecord) -> StudyType:
    """Source metadata beats the model for study design labels."""
    if record.is_preprint:
        return "preprint"
    types = {t.lower() for t in record.publication_types}
    if "meta-analysis" in types:
        return "meta_analysis"
    if "randomized controlled trial" in types and extracted not in ("animal", "in_vitro"):
        return "rct"
    return extracted


def _aggregate_state(responses: list[SearchResponse], n_findings: int) -> str:
    failures = [r.state for r in responses if r.state not in ("ok", "no_results")]
    if n_findings and failures:
        return "partial"
    if n_findings:
        return "ok"
    if failures:
        return failures[0]
    return "no_results"


class ResearcherAgent:
    def __init__(
        self,
        llm: LLMClient,
        search_tool: SearchTool,
        model: str | None = None,
        max_searches: int = MAX_SEARCHES,
        max_records: int = MAX_RECORDS_TO_EXTRACT,
    ):
        self.llm = llm
        self.search_tool = search_tool
        self.model = model
        self.max_searches = max_searches
        self.max_records = max_records

    def _search(self, query: str) -> SearchResponse:
        try:
            return self.search_tool.search(query, max_results=MAX_RESULTS_PER_SEARCH)
        except Exception as e:  # a tool bug must not kill the run
            return SearchResponse(query=query, state="error", message=f"{type(e).__name__}: {e}")

    def run(self, sub_question: SubQuestion, profile: UserProfile | None = None) -> ResearchResult:
        queries = sub_question.search_terms[: self.max_searches]
        responses = [self._search(q) for q in queries]
        status = SearchStatus(
            sub_question_id=sub_question.id,
            state="no_results",
            queries=queries,
            searches_run=len(responses),
            records_seen=sum(len(r.records) for r in responses),
            errors=[f"{r.query}: {r.state}" + (f" ({r.message})" if r.message else "") for r in responses if r.state not in ("ok", "no_results")],
        )

        candidates: list[StudyRecord] = []
        seen: set[str] = set()
        for resp in responses:
            for rec in resp.records:
                if rec.source_id in seen:
                    continue  # same paper from another query: not worth a trace entry
                seen.add(rec.source_id)
                if rec.retracted:
                    status.dropped.append(DroppedRecord(source_id=rec.source_id, reason="retracted", detail=rec.title))
                elif not rec.source_id.strip() or not rec.source_url.strip():
                    status.dropped.append(DroppedRecord(source_id=rec.source_id or "?", reason="no_provenance"))
                else:
                    candidates.append(rec)
        candidates = candidates[: self.max_records]

        findings: list[Finding] = []
        if candidates:
            findings = self._extract(sub_question, candidates, profile, status)

        status.findings_kept = len(findings)
        status.state = _aggregate_state(responses, len(findings))  # type: ignore[assignment]
        extraction_failed = any(d.reason == "extraction_failed" for d in status.dropped)
        if status.state == "no_results" and extraction_failed:
            status.state = "error"
        return ResearchResult(sub_question_id=sub_question.id, findings=findings, status=status)

    def _extract(
        self,
        sq: SubQuestion,
        records: list[StudyRecord],
        profile: UserProfile | None,
        status: SearchStatus,
    ) -> list[Finding]:
        request = LLMRequest.for_agent(
            AGENT,
            system=SYSTEM_PROMPT,
            user=_render_user(sq, records, profile),
            output_type=ExtractionBatch,
            model=self.model,
        )
        try:
            batch = self.llm.complete(request).output
        except LLMError as e:
            status.errors.append(f"extraction failed: {e}")
            for r in records:
                status.dropped.append(DroppedRecord(source_id=r.source_id, reason="extraction_failed"))
            return []

        by_id = {e.source_id: e for e in batch.studies}
        retrieved_at = utcnow()
        findings: list[Finding] = []
        for rec in records:
            ex = by_id.get(rec.source_id)
            if ex is None:
                status.dropped.append(
                    DroppedRecord(source_id=rec.source_id, reason="extraction_failed", detail="missing from model output")
                )
                continue
            if not ex.relevant:
                status.dropped.append(DroppedRecord(source_id=rec.source_id, reason="irrelevant", detail=ex.claim[:200]))
                continue
            study_type = _study_type(ex.study_type, rec)
            grade, reasons = grade_finding(study_type, ex.sample_size, ex.funding_conflict)
            findings.append(
                Finding(
                    id=f"{sq.id}:{rec.source_id}",
                    sub_question_id=sq.id,
                    claim=ex.claim.strip(),
                    source_id=rec.source_id,
                    source_url=rec.source_url,
                    title=rec.title,
                    study_type=study_type,
                    population=ex.population.strip(),
                    sample_size=ex.sample_size,
                    effect=ex.effect,
                    published=rec.published,
                    funding_conflict=ex.funding_conflict,
                    retracted=rec.retracted,
                    snippet=_choose_snippet(ex.snippet, ex.claim, rec),
                    retrieved_at=retrieved_at,
                    grade=grade,
                    grade_reasons=reasons,
                )
            )
        return findings
