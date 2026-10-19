"""Methods and results extraction from JATS full-text XML (Europe PMC `fullTextXML`).

We keep the sections a reviewer would read to check a claim (methods, results)
plus the funding / competing-interest statements, and drop everything else
(introduction, discussion, references, tables' cell data). Sections are found by
`sec-type` first and by title second, since many publishers leave `sec-type` out.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

METHODS_CHARS = 1_500
RESULTS_CHARS = 3_500  # methods + results + notes stay under the researcher's 6,000-char window
NOTES_CHARS = 300

_METHODS_TITLE = re.compile(
    r"\b(methods?|methodology|materials|participants|subjects|procedures?|study design|statistical analys[ie]s|"
    r"experimental design|search strategy|eligibility)\b",
    re.I,
)
_RESULTS_TITLE = re.compile(r"\b(results?|findings|outcomes)\b", re.I)
_SKIP_TITLE = re.compile(r"\b(discussion|conclusions?|introduction|background|limitations?)\b", re.I)
_FUNDING_TITLE = re.compile(r"\b(funding|competing interests?|conflicts? of interest|declaration of interest|disclosures?)\b", re.I)
_SKIP_TAGS = frozenset({"table-wrap", "fig", "table", "graphic", "media", "disp-formula", "inline-formula", "xref-group", "supplementary-material"})


@dataclass
class FullTextSections:
    methods: str = ""
    results: str = ""
    notes: list[str] = field(default_factory=list)  # funding / conflict statements

    @property
    def usable(self) -> bool:
        """Full text is only worth more than the abstract if it has a results section."""
        return bool(self.results)

    def render(self) -> str:
        parts = []
        if self.methods:
            parts.append(f"Methods: {self.methods}")
        if self.results:
            parts.append(f"Results: {self.results}")
        if self.notes:
            parts.append(f"Funding and conflicts: {' '.join(self.notes)}")
        return "\n\n".join(parts)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _text(el: ET.Element) -> str:
    """Visible text of an element, skipping tables, figures, formulas and nested titles."""
    out: list[str] = []

    def walk(e: ET.Element) -> None:
        if e.text:
            out.append(e.text)
        for child in e:
            tag = _local(child.tag)
            if tag == "title":
                out.append(" ")  # nested section titles become separators
            elif tag not in _SKIP_TAGS:
                walk(child)
                if tag == "p":
                    out.append(" ")
            if child.tail:
                out.append(child.tail)

    walk(el)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _title(sec: ET.Element) -> str:
    for child in sec:
        if _local(child.tag) == "title":
            return _text(child)
    return ""


def _classify(sec: ET.Element) -> str | None:
    sec_type = (sec.get("sec-type") or "").lower()
    title = _title(sec)
    if "result" in sec_type:
        return "results"
    if any(k in sec_type for k in ("method", "material", "subjects", "participants")):
        return "methods"
    if "coi" in sec_type or "funding" in sec_type:
        return "notes"
    if _FUNDING_TITLE.search(title):
        return "notes"
    # "Results and discussion" counts as results; a pure "Discussion" does not.
    if _RESULTS_TITLE.search(title):
        return "results"
    if _SKIP_TITLE.search(title):
        return None
    if _METHODS_TITLE.search(title):
        return "methods"
    return None


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = cut.rfind(". ")
    return (cut[: end + 1] if end > limit // 2 else cut).strip() + " [...]"


def parse_jats(xml: str | bytes) -> FullTextSections:
    """Extract methods / results / funding text. Raises ValueError on malformed XML."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        raise ValueError(f"malformed JATS XML: {e}") from e
    found: dict[str, list[str]] = {"methods": [], "results": [], "notes": []}

    body = next((e for e in root.iter() if _local(e.tag) == "body"), None)
    if body is not None:
        def visit(parent: ET.Element) -> None:
            for sec in parent:
                if _local(sec.tag) != "sec":
                    continue
                kind = _classify(sec)
                if kind is None:
                    visit(sec)  # e.g. a "Study 1" wrapper with Methods/Results inside
                    continue
                text = _text(sec)
                if text:
                    found[kind].append(text)

        visit(body)

    # Funding and competing interests usually live in <back> or <front> notes.
    for el in root.iter():
        tag = _local(el.tag)
        if tag == "funding-statement" or (tag == "fn" and (el.get("fn-type") or "") in ("conflict", "coi-statement", "financial-disclosure")):
            found["notes"].append(_text(el))
        elif tag in ("sec", "notes") and el.get("sec-type") in ("COI-statement", "funding", "coi-statement"):
            found["notes"].append(_text(el))
        elif tag in ("ack", "notes") and _FUNDING_TITLE.search(_title(el)):
            found["notes"].append(_text(el))

    notes = list(dict.fromkeys(n for n in found["notes"] if n))
    return FullTextSections(
        methods=_clip(" ".join(found["methods"]), METHODS_CHARS),
        results=_clip(" ".join(found["results"]), RESULTS_CHARS),
        notes=[_clip(n, NOTES_CHARS) for n in notes[:3]],
    )
