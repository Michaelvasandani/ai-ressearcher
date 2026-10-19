"""Supervisor review, in deterministic code.

After each researcher pass the supervisor checks every sub-question against
the plan and the user's profile:
- Is it answered? (at least one finding from a human study)
- Are the findings only animal or in-vitro studies?
- Does the studied population match the user? (sex, training status, age)
- Is another pass justified? (fixable problem, send-back unused, searches and
  budget left, and a query that has not been tried yet)

After the writer it checks the answer: it cites real findings, the
recommendation does not rest on preprints / animal / in-vitro work alone, and
confidence is not higher than the cited evidence allows.

These functions are pure; the graph nodes in nodes.py call them and turn the
result into routing decisions, tasks and trace steps.
"""

from __future__ import annotations

import re

from backend.graph.budget import BudgetKind
from backend.graph.state import (
    MAX_SEND_BACKS,
    IssueCode,
    ResearchRecord,
    ReviewReport,
    SubQuestionReview,
)
from backend.schemas import CoachAnswer, Finding, SubQuestion, UserProfile

NON_HUMAN = frozenset({"animal", "in_vitro"})
CANNOT_CARRY_ALONE = frozenset({"animal", "in_vitro", "preprint"})
SEARCH_FAILURE_STATES = frozenset({"rate_limited", "timed_out", "error", "paywalled"})

# --- Population matching ----------------------------------------------------

_UNTRAINED = re.compile(r"\b(untrained|sedentary|novices?|beginners?|inactive)\b", re.I)
_TRAINED = re.compile(r"\b(trained|experienced|athletes?|bodybuilders?|powerlifters?|lifters)\b", re.I)
_MEN = re.compile(r"\b(men|males?|boys)\b", re.I)
_WOMEN = re.compile(r"\b(women|females?|girls)\b", re.I)
_OLDER = re.compile(
    r"\b(older|elderly|seniors?|postmenopausal)\b|\baged?\s+(6[5-9]|[7-9]\d)\b|\b(6[5-9]|[7-9]\d)\s*(-|–|to)\s*\d{2}\b",
    re.I,
)
_YOUNG = re.compile(r"\b(young|college|university students?|adolescents?)\b", re.I)


def population_mismatch(profile: UserProfile | None, population: str) -> list[str]:
    """Reasons the studied population differs from the user. Empty = matches or unknown."""
    if profile is None or profile.is_empty():
        return []
    reasons: list[str] = []
    trained = bool(_TRAINED.search(population))
    untrained = bool(_UNTRAINED.search(population))
    if profile.training_status in ("intermediate", "advanced") and untrained and not trained:
        reasons.append("studied untrained people; user is trained")
    if profile.training_status in ("untrained", "novice") and trained and not untrained:
        reasons.append("studied trained lifters; user is new to training")
    men, women = bool(_MEN.search(population)), bool(_WOMEN.search(population))
    if profile.sex == "female" and men and not women:
        reasons.append("studied men only; user is female")
    if profile.sex == "male" and women and not men:
        reasons.append("studied women only; user is male")
    older, young = bool(_OLDER.search(population)), bool(_YOUNG.search(population))
    if profile.age is not None and profile.age < 50 and older:
        reasons.append(f"studied older adults; user is {profile.age}")
    if profile.age is not None and profile.age >= 65 and young and not older:
        reasons.append(f"studied young adults; user is {profile.age}")
    return reasons


def profile_qualifiers(profile: UserProfile | None) -> list[str]:
    """Search words that steer a query toward people like the user."""
    if profile is None:
        return []
    words: list[str] = []
    if profile.training_status in ("intermediate", "advanced"):
        words.append("trained")
    elif profile.training_status in ("untrained", "novice"):
        words.append("untrained")
    if profile.sex == "female":
        words.append("women")
    elif profile.sex == "male":
        words.append("men")
    if profile.age is not None and profile.age >= 60:
        words.append("older adults")
    return words


# --- Research review --------------------------------------------------------


def _issues(record: ResearchRecord | None, mismatches: dict[str, str], human: list[Finding]) -> list[IssueCode]:
    if record is None or not record.passes:
        return ["not_researched"]
    issues: list[IssueCode] = []
    if not record.findings:
        issues.append("no_findings")
    elif not human:
        issues.append("only_animal_or_in_vitro")
    status = record.last_status
    if status is not None:
        if status.state in SEARCH_FAILURE_STATES or status.state == "partial":
            issues.append("search_failed")
        if any(d.reason == "extraction_failed" for d in status.dropped):
            issues.append("extraction_failed")
    if human and len(mismatches) == len(human):
        issues.append("population_mismatch")
    return issues


def refine_queries(
    sq: SubQuestion,
    record: ResearchRecord | None,
    profile: UserProfile | None,
    issues: list[IssueCode],
    limit: int,
) -> list[str]:
    """Deterministic queries for another pass, most promising first, never repeating a query
    that already ran (except retrying one that failed for a transient reason)."""
    tried = set(record.queries_run) if record else set()
    failed = list(record.passes[-1].failed_queries) if record and record.passes else []
    base = sq.search_terms
    candidates: list[str] = []
    if "search_failed" in issues:
        candidates += failed
    if "only_animal_or_in_vitro" in issues:
        candidates += [f"{t} humans randomized controlled trial" for t in base]
    if "population_mismatch" in issues:
        qualifiers = " ".join(profile_qualifiers(profile))
        if qualifiers:
            candidates += [f"{t} {qualifiers}" for t in base]
    if {"no_findings", "not_researched", "extraction_failed"} & set(issues):
        candidates += [t for t in base if t not in tried]
        candidates.append(sq.question.rstrip("?").strip())
    out: list[str] = []
    for q in candidates:
        if q in out or (q in tried and q not in failed):
            continue
        out.append(q)
    return out[: max(limit, 0)]


def review_sub_question(
    sq: SubQuestion,
    record: ResearchRecord | None,
    profile: UserProfile | None,
    *,
    searches_left: int,
    send_back_available: bool,
    budget_block: BudgetKind | None,
) -> SubQuestionReview:
    findings = record.findings if record else []
    human = [f for f in findings if f.study_type not in NON_HUMAN]
    mismatches = {
        f.id: "; ".join(reasons) for f in human if (reasons := population_mismatch(profile, f.population))
    }
    issues = _issues(record, mismatches, human)
    answered = bool(human)
    status = record.last_status if record else None

    needs_pass = not answered or "population_mismatch" in issues
    queries: list[str] = []
    blocked_by = None
    if not needs_pass:
        why = "answered by human studies matching the user"
    elif not send_back_available:
        why, blocked_by = "another pass would help, but the single send-back is already used", "send_back_used"
    elif budget_block is not None:
        why, blocked_by = f"another pass would help, but the {budget_block} budget is exhausted", budget_block
    elif searches_left <= 0:
        why, blocked_by = "another pass would help, but max_searches for this sub-question is used up", "max_searches"
    else:
        queries = refine_queries(sq, record, profile, issues, searches_left)
        if queries:
            why = f"send back: {', '.join(issues)}"
        else:
            why, blocked_by = "another pass would help, but there is no untried query", "no_untried_query"
    # Re-running searches that failed for transient reasons needs no stronger model;
    # anything about the quality of what was extracted does.
    model_issues = set(issues) - {"search_failed", "not_researched"}
    if "search_failed" in issues and status is not None and status.records_seen == 0:
        model_issues.discard("no_findings")  # nothing came back to extract: a search problem
    return SubQuestionReview(
        sub_question_id=sq.id,
        answered=answered,
        findings=len(findings),
        human_findings=len(human),
        issues=issues,
        search_state=status.state if status else None,
        population_mismatches=mismatches,
        another_pass=bool(queries),
        another_pass_reason=why,
        refined_queries=queries,
        escalate=bool(queries) and bool(model_issues),
        blocked_by=blocked_by,
    )


def can_carry_recommendation(findings: list[Finding]) -> bool:
    """At least one peer-reviewed human finding: preprints and animal work never carry it alone."""
    return any(f.study_type not in CANNOT_CARRY_ALONE for f in findings)


def review_research(
    sub_questions: list[SubQuestion],
    research: dict[str, ResearchRecord],
    profile: UserProfile | None,
    *,
    pass_no: int,
    send_backs_used: int,
    max_searches_per_sub_question: int,
    budget_block: BudgetKind | None,
) -> ReviewReport:
    send_back_available = send_backs_used < MAX_SEND_BACKS
    reviews = [
        review_sub_question(
            sq,
            research.get(sq.id),
            profile,
            searches_left=max_searches_per_sub_question - (research[sq.id].searches_used if sq.id in research else 0),
            send_back_available=send_back_available,
            budget_block=budget_block,
        )
        for sq in sub_questions
    ]
    send_back = [r.sub_question_id for r in reviews if r.another_pass]
    answered = sum(r.answered for r in reviews)
    if send_back:
        return ReviewReport(
            pass_no=pass_no,
            decision="send_back",
            reason=f"{answered} of {len(reviews)} sub-questions answered; sending back {', '.join(send_back)}",
            sub_questions=reviews,
            send_back_ids=send_back,
        )
    findings = [f for sq in sub_questions if sq.id in research for f in research[sq.id].findings]
    if can_carry_recommendation(findings):
        return ReviewReport(
            pass_no=pass_no,
            decision="proceed",
            reason=f"{answered} of {len(reviews)} sub-questions answered; writing with {len(findings)} findings",
            sub_questions=reviews,
        )
    why = "no findings" if not findings else "only preprint, animal or in-vitro findings"
    return ReviewReport(
        pass_no=pass_no,
        decision="fail",
        reason=f"cannot answer: {why}, and no further pass is possible",
        sub_questions=reviews,
    )


# --- Answer review ----------------------------------------------------------

_GRADE_RANK = {"very_low": 0, "low": 1, "moderate": 2, "high": 3}


def review_answer(answer: CoachAnswer | None, findings: list[Finding], writer_error: str | None) -> list[str]:
    """Problems with the writer's output. Empty = accept."""
    if writer_error:
        return [f"writer failed: {writer_error}"]
    if answer is None:
        return ["writer produced no answer"]
    by_id = {f.id: f for f in findings}
    cited = answer.all_cited_ids()
    problems: list[str] = []
    unknown = cited - by_id.keys()
    if unknown:
        problems.append(f"cites unknown finding ids: {sorted(unknown)}")
    known = [by_id[i] for i in cited if i in by_id]
    if not known:
        problems.append("cites no findings")
        return problems
    if not can_carry_recommendation(known):
        problems.append("the recommendation rests only on preprint, animal or in-vitro findings")
    best = max(_GRADE_RANK[f.grade] for f in known)
    if answer.confidence == "high" and best < _GRADE_RANK["moderate"]:
        problems.append("confidence is high but every cited finding is low or very low grade")
    return problems
