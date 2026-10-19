"""Researcher: one sub-question in, graded Finding records + a SearchStatus out.

Flow (search and filtering are code; only extraction uses the model):
1. Run the sub-question's search terms through the SearchTool (capped).
2. Deduplicate by DOI / PMID / source id across queries; drop retracted
   papers and records without provenance, recording each drop and its reason.
   Drops the search tool made (duplicates merged across sources, per-journal
   and per-lab caps) are carried into the status too.
3. One model call extracts a claim, study design, population, sample size,
   effect and a verbatim snippet from each remaining record.
4. Code attaches provenance (ids, URLs, dates, source and endpoint, whether
   the text was full text or abstract: all from the search record, never from
   the model) and a deterministic evidence grade. The snippet is checked to be
   verbatim, preferring the methods / results sections of the full text.

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
    Provenance,
    ResearchResult,
    SearchResponse,
    SearchStatus,
    StudyRecord,
    StudyType,
    SubQuestion,
    UserProfile,
)
from backend.tools.ranking import record_keys
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
in the claim, including its numbers. When the record has full-text sections, quote the Results \
section, not the abstract. Copy it exactly; do not paraphrase. Never quote funding or methods \
boilerplate as the snippet.
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
    body = f"Abstract: {r.abstract}"
    if r.full_text:
        basis = "open-access full text (methods and results sections)"
        body += f"\nFull text sections:\n{r.full_text[:MAX_TEXT_CHARS]}"
    else:
        basis = "abstract only (no open-access full text)"
    types = ", ".join(r.publication_types) or "unknown"
    mesh = f"\nMeSH: {', '.join(r.mesh_terms[:12])}" if r.mesh_terms else ""
    return (
        f'<record source_id="{r.source_id}">\nTitle: {r.title}\nPublished: {r.published.isoformat()}\n'
        f"Publication types: {types}\nPreprint: {r.is_preprint}{mesh}\nText available: {basis}\n{body}\n</record>"
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


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _section_text(full_text: str) -> str:
    """Methods and results only: funding statements are never a result snippet."""
    return full_text.split("Funding and conflicts:", 1)[0]


def _choose_snippet(snippet: str, claim: str, record: StudyRecord) -> str:
    """Return a verbatim snippet from the record that supports the claim.

    With full text, the snippet should come from its methods / results
    sections: the model's snippet is kept if it is verbatim there; otherwise
    the section sentence sharing the most numbers with the claim is used, and
    only if no section sentence shares any falls back to the abstract. Without
    full text, the same rules apply to the abstract. A snippet that has no
    number from a numeric claim is replaced, so later citation checks can find
    the claim's numbers.
    """
    claim_numbers = _numbers(claim)
    snippet = snippet.strip()

    def supports(text: str) -> bool:
        return not claim_numbers or bool(claim_numbers & _numbers(text))

    def score(sentence: str) -> int:
        # Effect sizes (decimals, percentages) identify a result better than counts like "8 weeks".
        return sum(2 if ("." in n or "%" in n) else 1 for n in claim_numbers & _numbers(sentence))

    sources = [_section_text(record.full_text)] if record.full_text else []
    sources.append(record.abstract)
    for source in sources:
        if snippet and _norm(snippet) in _norm(source) and supports(snippet):
            return snippet
        sentences = _sentences(source)
        if sentences and (best := max(sentences, key=score)) and score(best) > 0:
            return best[:600]
    # Nothing shares a number with the claim: first sentence of the best available text.
    for source in sources:
        if snippet and _norm(snippet) in _norm(source):
            return snippet
    fallback = _sentences(sources[0]) or [sources[0].strip()]
    return fallback[0][:600]


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


TRANSIENT_STATES = frozenset({"rate_limited", "timed_out", "error"})


def _aggregate_state(responses: list[SearchResponse], findings: list[Finding], dropped: list[DroppedRecord]) -> str:
    transient = [r.state for r in responses if r.state in TRANSIENT_STATES]
    if findings:
        if transient:
            return "partial"
        abstract_only = all(f.provenance and f.provenance.text_source == "abstract" for f in findings)
        if abstract_only and any(r.state == "paywalled" for r in responses):
            return "paywalled"
        return "ok"
    if transient:
        return transient[0]
    if any(d.reason == "extraction_failed" for d in dropped):
        return "error"
    reasons = {d.reason for d in dropped if d.reason not in ("duplicate", "journal_cap", "author_cap")}
    if reasons == {"retracted"}:
        return "retracted"
    if any(r.state == "paywalled" for r in responses):
        return "paywalled"
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

        for resp in responses:
            status.dropped.extend(resp.dropped)
        candidates: list[StudyRecord] = []
        seen: set[str] = set()
        for resp in responses:
            for rec in resp.records:
                keys = record_keys(rec)
                if seen & set(keys):
                    continue  # same paper from another query: not worth a trace entry
                seen.update(keys)
                if rec.retracted:
                    detail = rec.title + (f" | {rec.retraction_notice}" if rec.retraction_notice else "")
                    status.dropped.append(DroppedRecord(source_id=rec.source_id, reason="retracted", detail=detail[:300]))
                elif not rec.has_provenance():
                    status.dropped.append(DroppedRecord(source_id=rec.source_id or "?", reason="no_provenance"))
                else:
                    candidates.append(rec)
        candidates = candidates[: self.max_records]

        findings: list[Finding] = []
        if candidates:
            findings = self._extract(sub_question, candidates, profile, status)

        status.findings_kept = len(findings)
        status.abstract_only = [
            f.source_id for f in findings if f.provenance is not None and f.provenance.text_source == "abstract"
        ]
        status.state = _aggregate_state(responses, findings, status.dropped)  # type: ignore[assignment]
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
            provenance = Provenance.of(rec)
            if provenance is None:  # unreachable after the candidate filter; never emit one without it
                status.dropped.append(DroppedRecord(source_id=rec.source_id, reason="no_provenance"))
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
                    retrieved_at=provenance.retrieved_at,
                    grade=grade,
                    grade_reasons=reasons,
                    provenance=provenance,
                )
            )
        return findings
