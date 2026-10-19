"""Citation validation: the final gate before an answer is returned.

Every claim and every number in a `CoachAnswer` must link to a real finding
record that contains it. All of this is deterministic code, no model:

Citation format
    The writer cites inline with finding ids in brackets (`[sq1:PMID:123]`);
    `writer.number_citations` turns those into numbered markers, so the final
    answer text reads `... 0.37% per set [1].` and `[n]` points at
    `answer.cited_finding_ids[n - 1]` (the UI's citation chips).

Claims
    The answer's text is split into sentences per field. In the evidence
    fields (`recommendation`, `why`, `whats_new`) every sentence is a claim and
    must carry at least one marker. In the advice fields (`practical_protocol`,
    `applies_to_you`, `what_would_change_this`) a sentence needs a marker only
    when it states a number. Position stances are backed by their
    `supporting_findings`; counterarguments by any finding the answer cites.

Numbers
    Digits (`12`, `0.37`, `1,863`, `10-20`, `10 to 20`, `−0.2`), and spelled
    numbers followed by a unit (`ten sets`, `three days`). A number is backed
    when a finding cited in the same sentence contains the same value in its
    claim, effect, snippet, population, title, sample size or publication
    year. Signs are ignored (an effect of -0.2 is "a 0.2 decrease"); an answer
    number with decimals may round the finding's (`1.6` matches `1.62`), an
    integer must match exactly. Numbers that come from the user (profile
    fields, the question) need no finding. Range endpoints are checked one
    by one, so `10-20 sets` needs both 10 and 20 in the cited findings.

`validate_answer` returns a `CitationReport`; its `problems()` are the
feedback for the writer's single rewrite.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Literal

from pydantic import Field

from backend.schemas import CoachAnswer, Finding, Record, UserProfile

EVIDENCE_FIELDS = ("recommendation", "why", "whats_new")
ADVICE_FIELDS = ("practical_protocol", "applies_to_you", "what_would_change_this")

# --- Markers ----------------------------------------------------------------

# A numbered marker: [1], [1, 2], [1][2], [1-3]
MARKER = re.compile(r"\[\s*(\d+(?:\s*(?:[,;]|[-–])\s*\d+)*)\s*\]")
# Something that looks like an unconverted finding id citation: [sq1:PMID:123] or [FIXTURE-0001]
RAW_ID_MARKER = re.compile(r"\[[^\[\]]*(?:sq\d+:|PMID|DOI|FIXTURE-|PMC\d)[^\[\]]*\]", re.I)


def marker_numbers(inner: str) -> list[int]:
    """`1, 2` -> [1, 2]; `1-3` -> [1, 2, 3]."""
    out: list[int] = []
    for part in re.split(r"\s*[,;]\s*", inner.strip()):
        m = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out.extend(range(a, b + 1) if 0 < b - a < 50 else [a, b])
        elif part.isdigit():
            out.append(int(part))
    return out


def strip_markers(text: str) -> str:
    return MARKER.sub(" ", text)


# --- Sentences --------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")
_ABBREV = re.compile(r"(?:\b(?:e\.g|i\.e|vs|approx|ca|et al|cf|Fig|No)\.)$", re.I)


def sentences(text: str) -> list[str]:
    """Split text into sentences. Markers right after a full stop belong to the sentence before it
    ("... growth. [1] Next ..." -> "... growth. [1]", "Next ...")."""
    text = " ".join((text or "").split())
    if not text:
        return []
    parts: list[str] = []
    start = 0
    for m in _SENTENCE_END.finditer(text):
        chunk = text[start : m.start()].strip()
        if _ABBREV.search(chunk):
            continue  # "e.g. Ten sets" is one sentence
        parts.append(chunk)
        start = m.end()
    parts.append(text[start:].strip())
    out: list[str] = []
    for p in parts:
        lead = re.match(r"^((?:\[\s*\d[^\]]*\]\s*)+)(.*)$", p)
        if lead and out:
            out[-1] = f"{out[-1]} {lead.group(1).strip()}"
            p = lead.group(2).strip()
        if p:
            out.append(p)
    return out


# --- Numbers ----------------------------------------------------------------

WORD_NUMBERS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}
# Units after which a spelled number in the answer is a quantity that needs a source.
UNITS = (
    r"sets?|reps?|repetitions?|grams?|g|kg|kilograms?|mg|percent|per\s*cent|%|weeks?|days?|sessions?|"
    r"times|minutes?|min|hours?|h|servings?|meals?|doses?|calories|kcal|exercises?|workouts?|"
    r"months?|years?|lbs?|pounds?|liters?|litres?|ml"
)
_DIGITS = re.compile(
    r"(?<![A-WYZa-wyz0-9_.])"  # not glued to a word ("P1", "sq2", "B12"); "5x5" is two numbers
    r"[-−–+±]?"  # a sign is part of the token, and ignored when comparing
    r"(\d{1,3}(?:,\d{3})+(?!\d)|\d+(?:\.\d+)?|\.\d+)"
    r"(st|nd|rd|th|RM)?",  # ordinals and the "1RM" term are not quantities
)
_LABEL = re.compile(r"\b(?:position|side|stance|round|option|step|phase|group|arm|figure|fig\.|table)\s*$", re.I)
_WORD = re.compile(r"\b(" + "|".join(WORD_NUMBERS) + r")\b(?:[\s-]+(" + "|".join(WORD_NUMBERS) + r"))?", re.I)
_WORD_WITH_UNIT = re.compile(
    r"\b(" + "|".join(WORD_NUMBERS) + r")\b(?:[\s-]+(" + "|".join(WORD_NUMBERS) + r"))?"
    r"(?=\s+(?:(?:hard|weekly|working|extra|more|fewer|additional|daily|total|training|full)\s+)?(?:" + UNITS + r")\b)",
    re.I,
)


@dataclass(frozen=True)
class NumberMention:
    text: str  # as written, e.g. "1,863" or "ten"
    value: Decimal  # absolute value
    places: int  # decimal places as written

    def label(self) -> str:
        return self.text


def _decimal(token: str) -> tuple[Decimal, int] | None:
    raw = token.replace(",", "")
    try:
        value = abs(Decimal(raw))
    except InvalidOperation:
        return None
    places = len(raw.split(".", 1)[1]) if "." in raw else 0
    return value, places


def _word_value(first: str, second: str | None) -> int:
    a = WORD_NUMBERS[first.lower()]
    if second:
        b = WORD_NUMBERS[second.lower()]
        if a >= 20 and a % 10 == 0 and b < 10:  # "twenty-five"
            return a + b
        if b == 100:  # "two hundred"
            return a * 100
    return a


def extract_numbers(text: str, *, spelled: Literal["all", "with_unit", "none"] = "with_unit") -> list[NumberMention]:
    """Numeric quantities in `text`. Citation markers are never numbers.

    `spelled="with_unit"` (the answer side) counts spelled numbers only before a unit ("ten sets"),
    so "one study" or "the two sides" are not checked; `"all"` (the finding side) counts every one.
    """
    text = strip_markers(text or "")
    out: list[NumberMention] = []
    for m in _DIGITS.finditer(text):
        if m.group(2):  # ordinal / 1RM
            continue
        if _LABEL.search(text, 0, m.start()):  # "position 2", "side 1": a label, not a quantity
            continue
        parsed = _decimal(m.group(1))
        if parsed is not None:
            out.append(NumberMention(m.group(1), parsed[0], parsed[1]))
    if spelled != "none":
        pattern = _WORD if spelled == "all" else _WORD_WITH_UNIT
        for m in pattern.finditer(text):
            # "twenty five" is one number; "one or two" are two
            value = _word_value(m.group(1), m.group(2))
            out.append(NumberMention(m.group(0), Decimal(value), 0))
            if m.group(2) and value == WORD_NUMBERS[m.group(1).lower()]:
                out.append(NumberMention(m.group(2), Decimal(WORD_NUMBERS[m.group(2).lower()]), 0))
    return out


def matches(answer: NumberMention, values: Iterable[tuple[Decimal, int]]) -> bool:
    """Equal value, or the finding's value rounded to the answer's (non-zero) decimal places."""
    for value, places in values:
        if value == answer.value:
            return True
        if answer.places > 0 and places > answer.places:
            q = Decimal(1).scaleb(-answer.places)
            if value.quantize(q, rounding=ROUND_HALF_UP) == answer.value.quantize(q):
                return True
    return False


def finding_text(f: Finding) -> str:
    parts = [f.claim, f.effect or "", f.snippet, f.population, f.title or ""]
    return "\n".join(parts)


def finding_values(f: Finding) -> list[tuple[Decimal, int]]:
    """Every number a finding record contains (generous: spelled numbers count too)."""
    vals = [(n.value, n.places) for n in extract_numbers(finding_text(f), spelled="all")]
    if f.sample_size is not None:
        vals.append((Decimal(f.sample_size), 0))
    vals.append((Decimal(f.published.year), 0))
    return vals


def context_values(question: str | None, profile: UserProfile | None) -> list[tuple[Decimal, int]]:
    """Numbers the user supplied: they need no finding ("your 4 training days")."""
    vals = [(n.value, n.places) for n in extract_numbers(question or "", spelled="all")]
    if profile is not None:
        for name in ("age", "bodyweight_kg", "training_age_years", "days_per_week", "session_minutes"):
            v = getattr(profile, name)
            if v is not None:
                parsed = _decimal(str(v))
                if parsed:
                    vals.append(parsed)
                    if parsed[0] == parsed[0].to_integral_value():  # 3.0 years is "3 years"
                        vals.append((parsed[0].to_integral_value(), 0))
        for text in (profile.notes, profile.goal, profile.equipment):
            vals += [(n.value, n.places) for n in extract_numbers(text or "", spelled="all")]
    return vals


# --- Report -----------------------------------------------------------------

IssueKind = Literal[
    "uncited_claim",  # an evidence sentence with no citation
    "uncited_number",  # a number in a sentence with no citation (and not from the user)
    "unbacked_number",  # a number none of the sentence's cited findings contains
    "bad_marker",  # [n] that points at no cited finding
    "raw_citation",  # an unconverted / invented id citation left in the text
    "unknown_finding",  # cited_finding_ids or a position cites an id that is not a finding
]


class CitationIssue(Record):
    field: str  # "why", "positions[1].stance", ...
    kind: IssueKind
    sentence: str
    number: str | None = None
    cited: list[str] = Field(default_factory=list)  # finding ids the sentence cites
    found_in: list[str] = Field(default_factory=list)  # findings that do contain the number (hint)
    detail: str


class CitationReport(Record):
    """Result of validating one answer draft. `ok` only with no issues."""

    attempt: int = 1
    ok: bool
    claims_checked: int = 0
    numbers_checked: int = 0
    numbers_from_user: int = 0
    citations: int = 0  # markers resolved to findings
    issues: list[CitationIssue] = Field(default_factory=list)

    def problems(self, limit: int = 12) -> list[str]:
        """One line per issue, phrased as feedback for the writer (findings by id)."""
        out = [f"citation check: {i.field}: {i.detail}" for i in self.issues[:limit]]
        if len(self.issues) > limit:
            out.append(f"citation check: ... and {len(self.issues) - limit} more citation problems")
        return out

    def summary(self) -> str:
        head = f"{self.claims_checked} claims, {self.numbers_checked} numbers, {self.citations} citations checked"
        if self.ok:
            return f"{head}: every claim and number is backed by a cited finding"
        kinds: dict[str, int] = {}
        for i in self.issues:
            kinds[i.kind] = kinds.get(i.kind, 0) + 1
        return f"{head}: {len(self.issues)} problem(s) ({', '.join(f'{v} {k}' for k, v in sorted(kinds.items()))})"


def _clip(text: str, n: int = 140) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


class _Checker:
    def __init__(self, answer: CoachAnswer, findings: list[Finding], context: list[tuple[Decimal, int]]):
        self.answer = answer
        self.by_id = {f.id: f for f in findings}
        self.values = {f.id: finding_values(f) for f in findings}
        self.context = context
        self.issues: list[CitationIssue] = []
        self.claims = 0
        self.numbers = 0
        self.from_user = 0
        self.citations = 0

    def _issue(self, field: str, kind: IssueKind, sentence: str, detail: str, **kw: object) -> None:
        self.issues.append(CitationIssue(field=field, kind=kind, sentence=_clip(sentence), detail=detail, **kw))  # type: ignore[arg-type]

    def _resolve(self, field: str, sentence: str) -> list[str]:
        """Finding ids the sentence cites through [n] markers; bad markers are issues."""
        refs = self.answer.cited_finding_ids
        ids: list[str] = []
        for m in MARKER.finditer(sentence):
            for n in marker_numbers(m.group(1)):
                if 1 <= n <= len(refs) and refs[n - 1] in self.by_id:
                    ids.append(refs[n - 1])
                    self.citations += 1
                else:
                    self._issue(field, "bad_marker", sentence, f"marker [{n}] points at no cited finding")
        for m in RAW_ID_MARKER.finditer(sentence):
            self._issue(field, "raw_citation", sentence, f"{m.group(0)} is not a valid citation of a known finding")
        return list(dict.fromkeys(ids))

    def _check_numbers(self, field: str, sentence: str, cited: list[str], *, scope: str = "cited findings") -> None:
        for num in extract_numbers(sentence):
            self.numbers += 1
            if any(matches(num, self.values[i]) for i in cited):
                continue
            if matches(num, self.context):
                self.from_user += 1
                continue
            found = [fid for fid, vals in self.values.items() if matches(num, vals)]
            hint = f"; it appears in {', '.join(found[:4])}" if found else "; no finding contains it"
            if not cited:
                self._issue(
                    field, "uncited_number", sentence,
                    f'"{_clip(sentence, 90)}" states {num.text} without citing a finding{hint}',
                    number=num.text, found_in=found,
                )
            else:
                self._issue(
                    field, "unbacked_number", sentence,
                    f'"{_clip(sentence, 90)}" states {num.text}, which its {scope} ({", ".join(cited)}) do not contain{hint}',
                    number=num.text, cited=cited, found_in=found,
                )

    def text_field(self, field: str, text: str | None, *, every_sentence_is_a_claim: bool) -> None:
        for s in sentences(text or ""):
            cited = self._resolve(field, s)
            has_numbers = bool(extract_numbers(s))
            if every_sentence_is_a_claim or has_numbers or cited:
                self.claims += 1
            if every_sentence_is_a_claim and not cited:
                self._issue(field, "uncited_claim", s, f'"{_clip(s, 90)}" makes a claim without citing a finding')
            self._check_numbers(field, s, cited)

    def positions(self) -> None:
        answer_ids = [i for i in self.answer.all_cited_ids() if i in self.by_id]
        for k, p in enumerate(self.answer.positions, start=1):
            support = [i for i in p.supporting_findings if i in self.by_id]
            for i in p.supporting_findings:
                if i not in self.by_id:
                    self._issue(f"positions[{k}]", "unknown_finding", p.stance, f"position {k} cites unknown finding {i}")
            self.claims += 1
            self._check_numbers(f"positions[{k}].stance", p.stance, support, scope="supporting findings")
            for s in sentences(p.best_counterargument):
                self._check_numbers(f"positions[{k}].best_counterargument", s, answer_ids, scope="answer's findings")

    def run(self) -> None:
        for i in self.answer.cited_finding_ids:
            if i not in self.by_id:
                self._issue("cited_finding_ids", "unknown_finding", i, f"cites unknown finding {i}")
        for name in EVIDENCE_FIELDS:
            self.text_field(name, getattr(self.answer, name), every_sentence_is_a_claim=True)
        for name in ADVICE_FIELDS:
            self.text_field(name, getattr(self.answer, name), every_sentence_is_a_claim=False)
        self.positions()


def validate_answer(
    answer: CoachAnswer,
    findings: list[Finding],
    *,
    question: str | None = None,
    profile: UserProfile | None = None,
    attempt: int = 1,
) -> CitationReport:
    """Check every claim and number in `answer` against the finding records. Deterministic."""
    checker = _Checker(answer, findings, context_values(question, profile))
    checker.run()
    return CitationReport(
        attempt=attempt,
        ok=not checker.issues,
        claims_checked=checker.claims,
        numbers_checked=checker.numbers,
        numbers_from_user=checker.from_user,
        citations=checker.citations,
        issues=checker.issues,
    )


# --- Writer side: [finding id] -> [n] ------------------------------------------

_BRACKET = re.compile(r"\[([^\[\]]{1,400})\]")


def number_citations(
    fields: Mapping[str, str | None], known_ids: Iterable[str], extra_ids: Iterable[str] = ()
) -> tuple[dict[str, str | None], list[str], list[str]]:
    """Turn `[finding id; finding id]` citations in `fields` into numbered `[1, 2]` markers.

    Returns (rewritten fields, the reference list in marker order followed by `extra_ids`,
    invented ids that were dropped). A bracket that cites only unknown ids is removed, so the
    claim it was attached to shows up as uncited in validation. Brackets that are not citations
    ("[sic]") stay as they are.
    """
    known = set(known_ids)
    refs: list[str] = []
    invented: list[str] = []

    def number_of(fid: str) -> int:
        if fid not in refs:
            refs.append(fid)
        return refs.index(fid) + 1

    def replace(m: re.Match[str]) -> str:
        parts = [p.strip() for p in re.split(r"\s*[;,]\s*", m.group(1)) if p.strip()]
        cited = [p for p in parts if p in known]
        looks_like_ids = [p for p in parts if p in known or RAW_ID_MARKER.fullmatch(f"[{p}]") or ":" in p]
        if not looks_like_ids:
            return m.group(0)
        invented.extend(p for p in parts if p not in known)
        if not cited:
            return ""
        return "[" + ", ".join(str(number_of(c)) for c in dict.fromkeys(cited)) + "]"

    out: dict[str, str | None] = {}
    for name, text in fields.items():
        if text is None:
            out[name] = None
            continue
        new = _BRACKET.sub(replace, text)
        new = re.sub(r"\s+([.,;:!?])(?!\d)", r"\1", re.sub(r"[ \t]{2,}", " ", new)).strip()
        out[name] = new
    for fid in extra_ids:
        if fid in known and fid not in refs:
            refs.append(fid)
    return out, refs, list(dict.fromkeys(invented))
