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


class StudyRecord(Record):
    """A raw search hit, as returned by a SearchTool. Provenance lives here."""

    source_id: str  # PMID / DOI / NCT id
    source_url: str
    title: str
    abstract: str
    full_text: str | None = None  # methods + results where available
    journal: str | None = None
    authors: list[str] = Field(default_factory=list)
    published: date
    publication_types: list[str] = Field(default_factory=list)  # as reported by the source
    retracted: bool = False
    is_preprint: bool = False


SearchState = Literal["ok", "no_results", "rate_limited", "paywalled", "timed_out", "error"]


class SearchResponse(Record):
    """What a SearchTool returns for one query. Never raised as an exception."""

    query: str
    state: SearchState
    records: list[StudyRecord] = Field(default_factory=list)
    message: str | None = None


class DroppedRecord(Record):
    source_id: str
    reason: Literal["retracted", "duplicate", "irrelevant", "no_provenance", "extraction_failed"]
    detail: str | None = None


class SearchStatus(Record):
    """Structured outcome of researching one sub-question; the supervisor acts on it."""

    sub_question_id: str
    state: Literal["ok", "partial", "no_results", "rate_limited", "paywalled", "timed_out", "error"]
    queries: list[str] = Field(default_factory=list)
    searches_run: int = 0
    records_seen: int = 0
    findings_kept: int = 0
    dropped: list[DroppedRecord] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


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

    @field_validator("source_id", "source_url")
    @classmethod
    def _provenance_required(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("findings must carry provenance")
        return v


class ResearchResult(Record):
    sub_question_id: str
    findings: list[Finding]
    status: SearchStatus


# --- Positions, debate, verdict (used from Phase 4; defined now) ------------


class Position(Record):
    stance: str
    supporting_findings: list[str] = Field(min_length=1)  # Finding ids
    evidence_grade: EvidenceGrade
    best_counterargument: str


class PositionMap(Record):
    status: EvidenceStatus
    positions: list[Position] = Field(min_length=1)


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


class Verdict(Record):
    status: EvidenceStatus
    recommendation: str
    confidence: Confidence
    rationale: str
    positions: list[Position] = Field(min_length=1)
    downweighted_findings: dict[str, str] = Field(default_factory=dict)  # finding id -> reason


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
