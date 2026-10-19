"""Pydantic models for every handoff between agents.

These are the records that flow through the graph and get persisted. Models
that an LLM fills in directly (drafts) live next to their agent; the agent
code turns a draft into one of these records, adding the fields that must
not be model-generated (ids, provenance, timestamps, the disclaimer).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --- Shared literals --------------------------------------------------------

StudyType = Literal[
    "meta_analysis",
    "rct",
    "cohort",
    "case_control",
    "case_report",
    "review",
    "animal",
    "in_vitro",
    "preprint",
]
EvidenceGrade = Literal["high", "moderate", "low", "very_low"]  # GRADE-style
Confidence = Literal["high", "moderate", "low"]
EvidenceStatus = Literal["consensus", "leaning", "genuinely_contested"]

ScopeCategory = Literal[
    "in_scope",
    "injury",
    "medical_condition",
    "medication",
    "pregnancy",
    "eating_disorder",
    "diagnosis",
    "off_topic",
]
OUT_OF_SCOPE_CATEGORIES: frozenset[str] = frozenset(
    {"injury", "medical_condition", "medication", "pregnancy", "eating_disorder", "diagnosis"}
)

DISCLAIMER = (
    "This is general, evidence-based training and nutrition information, not medical advice. "
    "Talk to a doctor or registered dietitian before making changes if you have a health "
    "condition, injury, or take medication."
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Record(BaseModel):
    """Base for handoff records: strict about unknown fields."""

    model_config = ConfigDict(extra="forbid")


# --- Intake -----------------------------------------------------------------


class UserProfile(Record):
    """What we know about the user. Every field is optional; intake fills what it can."""

    goal: str | None = None  # e.g. "hypertrophy", "fat loss", "strength"
    sex: Literal["male", "female", "other"] | None = None
    age: int | None = None
    bodyweight_kg: float | None = None
    training_age_years: float | None = None  # years of consistent resistance training
    training_status: Literal["untrained", "novice", "intermediate", "advanced"] | None = None
    days_per_week: int | None = None
    session_minutes: int | None = None
    equipment: str | None = None  # e.g. "full gym", "dumbbells only"
    dietary_pattern: str | None = None  # e.g. "vegan", "omnivore"
    notes: str | None = None

    def merged_with(self, other: "UserProfile") -> "UserProfile":
        """Fields set on `other` win over fields set here."""
        data = self.model_dump()
        data.update({k: v for k, v in other.model_dump().items() if v is not None})
        return UserProfile(**data)

    def is_empty(self) -> bool:
        return all(v is None for v in self.model_dump().values())


class Intake(Record):
    original_question: str
    scoped_question: str  # rewritten as one clear, answerable question
    in_scope: bool
    scope_category: ScopeCategory
    scope_reason: str
    classified_by: Literal["rule", "llm"]
    profile: UserProfile = Field(default_factory=UserProfile)
    referral_message: str | None = None  # fixed text when out of scope

    @model_validator(mode="after")
    def _scope_consistency(self) -> "Intake":
        if self.in_scope != (self.scope_category == "in_scope"):
            raise ValueError("in_scope must be True exactly when scope_category == 'in_scope'")
        if not self.in_scope and not self.referral_message:
            raise ValueError("out-of-scope intake must carry a referral_message")
        return self


# --- Planning ---------------------------------------------------------------


class SubQuestion(Record):
    id: str  # "sq1", "sq2", ...
    question: str
    rationale: str  # why answering this helps answer the main question
    search_terms: list[str] = Field(min_length=1)  # literature-database queries


class ResearchPlan(Record):
    question: str  # the scoped question being answered
    sub_questions: list[SubQuestion] = Field(min_length=3, max_length=6)

    @field_validator("sub_questions")
    @classmethod
    def _unique_ids(cls, v: list[SubQuestion]) -> list[SubQuestion]:
        ids = [sq.id for sq in v]
        if len(ids) != len(set(ids)):
            raise ValueError("sub-question ids must be unique")
        return v


# --- Research ---------------------------------------------------------------


SourceName = Literal["pubmed", "europepmc", "crossref", "sportrxiv", "fixture"]
TextSource = Literal["full_text", "abstract"]
Access = Literal["open_access", "paywalled", "unavailable", "unknown"]


class StudyRecord(Record):
    """A raw search hit, as returned by a SearchTool. Provenance lives here."""

    source_id: str  # "PMID:123", "DOI:10.x/y", or a fixture / preprint id
    source_url: str
    title: str
    abstract: str
    full_text: str | None = None  # methods + results sections from open-access full text, if fetched
    journal: str | None = None
    authors: list[str] = Field(default_factory=list)  # "Last FM", in byline order
    published: date
    publication_types: list[str] = Field(default_factory=list)  # as reported by the source
    retracted: bool = False
    is_preprint: bool = False
    # Identifiers (used to deduplicate across sources)
    pmid: str | None = None
    doi: str | None = None
    pmcid: str | None = None
    mesh_terms: list[str] = Field(default_factory=list)
    # Provenance: which database and endpoint served the record, and when
    source: SourceName = "fixture"
    endpoint: str | None = None  # e.g. "pubmed:efetch", "europepmc:search"
    retrieved_at: datetime | None = None
    also_found_in: list[str] = Field(default_factory=list)  # other "source:endpoint"s that returned it
    access: Access = "unknown"  # why full_text is or is not present
    full_text_url: str | None = None  # where full_text was fetched from
    retraction_checked_by: list[str] = Field(default_factory=list)  # e.g. ["pubmed:publication_type", "crossref"]
    retraction_notice: str | None = None  # DOI / PMID / text of the notice when retracted

    @property
    def text_source(self) -> TextSource:
        """Whether this record's best text is the full-text sections or only the abstract."""
        return "full_text" if self.full_text else "abstract"

    @property
    def last_author(self) -> str | None:
        return self.authors[-1] if self.authors else None

    def has_provenance(self) -> bool:
        return bool(self.source_id.strip() and self.source_url.strip() and self.retrieved_at)


SearchState = Literal["ok", "no_results", "rate_limited", "paywalled", "timed_out", "error"]


class SearchResponse(Record):
    """What a SearchTool returns for one query. Never raised as an exception.

    `paywalled` with records means every record came back abstract-only (no
    open-access full text); the records are still usable and labeled.
    """

    query: str
    state: SearchState
    records: list[StudyRecord] = Field(default_factory=list)
    message: str | None = None
    source_states: dict[str, SearchState] = Field(default_factory=dict)  # per source, for composite tools
    dropped: list["DroppedRecord"] = Field(default_factory=list)  # merged duplicates, diversity caps, ...


DropReason = Literal[
    "retracted",
    "duplicate",
    "irrelevant",
    "no_provenance",
    "extraction_failed",
    "journal_cap",  # too many results from one journal
    "author_cap",  # too many results from one author group (same last author)
]


class DroppedRecord(Record):
    source_id: str
    reason: DropReason
    detail: str | None = None


SearchStatusState = Literal[
    "ok",
    "partial",  # findings, but some searches failed transiently
    "no_results",
    "rate_limited",
    "paywalled",  # findings rest on abstracts only (no open-access full text)
    "timed_out",
    "retracted",  # everything relevant that came back was retracted
    "error",
]


class SearchStatus(Record):
    """Structured outcome of researching one sub-question; the supervisor acts on it."""

    sub_question_id: str
    state: SearchStatusState
    queries: list[str] = Field(default_factory=list)
    searches_run: int = 0
    records_seen: int = 0
    findings_kept: int = 0
    abstract_only: list[str] = Field(default_factory=list)  # kept source ids with no full text
    dropped: list[DroppedRecord] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class Provenance(Record):
    """Where a finding came from: enough to re-fetch and audit it."""

    source: SourceName  # database that served the metadata
    endpoint: str  # e.g. "pubmed:efetch"
    source_id: str
    url: str
    retrieved_at: datetime
    text_source: TextSource  # the snippet came from full-text sections or only the abstract
    full_text_url: str | None = None
    also_found_in: list[str] = Field(default_factory=list)
    retraction_checked_by: list[str] = Field(default_factory=list)

    @field_validator("endpoint", "source_id", "url")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("provenance fields must be non-empty")
        return v

    @classmethod
    def of(cls, record: StudyRecord) -> "Provenance | None":
        """Provenance for a search record, or None if the record lacks it."""
        if not record.has_provenance():
            return None
        return cls(
            source=record.source,
            endpoint=record.endpoint or record.source,
            source_id=record.source_id,
            url=record.source_url,
            retrieved_at=record.retrieved_at,  # type: ignore[arg-type]
            text_source=record.text_source,
            full_text_url=record.full_text_url,
            also_found_in=record.also_found_in,
            retraction_checked_by=record.retraction_checked_by,
        )


class Finding(Record):
    id: str
    sub_question_id: str | None = None
    claim: str
    source_id: str  # PMID / DOI / NCT id
    source_url: str
    title: str | None = None
    study_type: StudyType
    population: str  # who was studied (age, sex, training status)
    sample_size: int | None
    effect: str | None  # direction + size
    published: date
    funding_conflict: bool | None
    retracted: bool
    snippet: str  # from the full text where available, not the search snippet
    retrieved_at: datetime
    grade: EvidenceGrade  # certainty of this single finding (deterministic, see grading.py)
    grade_reasons: list[str] = Field(default_factory=list)
    # Optional in the schema so a finding without it can be represented and then rejected by
    # the provenance gate (graph.state.usable_findings) before it reaches the writer.
    provenance: Provenance | None = None

    @field_validator("source_id", "source_url")
    @classmethod
    def _provenance_required(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("findings must carry provenance")
        return v

    def provenance_problem(self) -> str | None:
        """Why this finding is unusable for the writer, or None if its provenance is complete."""
        p = self.provenance
        if p is None:
            return "no provenance record"
        if p.source_id != self.source_id:
            return f"provenance is for {p.source_id}, finding is {self.source_id}"
        if p.url != self.source_url:
            return "provenance URL does not match the finding"
        return None


class ResearchResult(Record):
    sub_question_id: str
    findings: list[Finding]
    status: SearchStatus


# --- Positions, debate, verdict ----------------------------------------------


class Position(Record):
    stance: str
    supporting_findings: list[str] = Field(min_length=1)  # Finding ids
    evidence_grade: EvidenceGrade
    best_counterargument: str


class PositionMap(Record):
    status: EvidenceStatus
    positions: list[Position] = Field(min_length=1)  # leading position first; the sides worth debating
    proposed_status: EvidenceStatus | None = None  # what the mapper model proposed, before the code rules
    status_reasons: list[str] = Field(default_factory=list)  # how code set the status (false-balance rule)
    set_aside: list[Position] = Field(default_factory=list)  # weak dissent: shown, never debated
    unassigned: list[str] = Field(default_factory=list)  # finding ids the mapper put in no position


class ArgumentClaim(Record):
    text: str
    finding_ids: list[str] = Field(min_length=1)  # a claim without a finding id is rejected


class Argument(Record):
    stance: str
    claims: list[ArgumentClaim] = Field(min_length=1)
    rebuttal_of: str | None = None  # stance this argument rebuts, if a rebuttal round

    def cited_ids(self) -> set[str]:
        return {fid for c in self.claims for fid in c.finding_ids}

    def check_citations(self, known_ids: set[str]) -> None:
        unknown = self.cited_ids() - known_ids
        if unknown:
            raise ValueError(f"argument cites unknown finding ids: {sorted(unknown)}")


class RejectedClaim(Record):
    """An advocate claim that did not cite valid finding ids, kept for the trace."""

    text: str
    finding_ids: list[str] = Field(default_factory=list)
    reason: str


class DebateTurn(Record):
    """One advocate's argument in one round: 1 = opening, 2 = the single rebuttal round."""

    position: int  # 1-based index into PositionMap.positions ("P1", "P2", ...)
    stance: str
    round: Literal[1, 2]
    argument: Argument | None = None  # None when the advocate failed or every claim was rejected
    rejected_claims: list[RejectedClaim] = Field(default_factory=list)
    error: str | None = None


class Verdict(Record):
    status: EvidenceStatus
    recommendation: str
    confidence: Confidence
    rationale: str
    positions: list[Position] = Field(min_length=1)  # the recommended position first
    downweighted_findings: dict[str, str] = Field(default_factory=dict)  # finding id -> reason
    confidence_cap: Confidence | None = None  # highest confidence the evidence allowed (code)
    adjustments: list[str] = Field(default_factory=list)  # where code overrode the judge, and why


# --- Final answer -----------------------------------------------------------


class CoachAnswer(Record):
    status: EvidenceStatus
    recommendation: str
    confidence: Confidence
    why: str
    positions: list[Position] = Field(min_length=1)  # 1 if consensus, 2+ if contested
    what_would_change_this: str
    applies_to_you: str  # how the user's profile changes things
    practical_protocol: str  # what to actually do on Monday
    whats_new: str | None  # if recent research changed the answer
    cited_finding_ids: list[str] = Field(default_factory=list)
    disclaimer: str  # always present: not medical advice

    @field_validator("disclaimer")
    @classmethod
    def _disclaimer_present(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("disclaimer is required")
        return v

    @model_validator(mode="after")
    def _positions_match_status(self) -> "CoachAnswer":
        if self.status == "genuinely_contested" and len(self.positions) < 2:
            raise ValueError("a contested answer must show at least two positions")
        return self

    def all_cited_ids(self) -> set[str]:
        ids = set(self.cited_finding_ids)
        for p in self.positions:
            ids.update(p.supporting_findings)
        return ids
