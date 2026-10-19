"""PubMed via NCBI E-utilities: esearch for PMIDs, efetch XML for the records.

efetch gives the metadata nothing else gives as reliably: publication types
(including "Retracted Publication"), MeSH headings, structured abstracts, DOI
and PMCID, and RetractionIn links.

Docs: https://www.ncbi.nlm.nih.gov/books/NBK25499/
Rate limit: 3 requests/second without an API key, 10 with NCBI_API_KEY.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import date, datetime

from backend.schemas import SearchResponse, StudyRecord, utcnow
from backend.tools.http import FetchError, HttpClient
from backend.tools.ranking import RecencyPolicy

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH = f"{EUTILS}/esearch.fcgi"
EFETCH = f"{EUTILS}/efetch.fcgi"
HOST = "eutils.ncbi.nlm.nih.gov"
TOOL_NAME = "evidence-coach"

RETRACTED_TYPES = frozenset({"retracted publication"})
NOTICE_TYPES = frozenset({"retraction of publication", "published erratum", "expression of concern"})
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def rate_limit(api_key: str | None) -> float:
    return 9.0 if api_key else 2.5  # a little under NCBI's 10 / 3 per second


def build_term(query: str, today: date, policy: RecencyPolicy) -> str:
    """Restrict to humans-or-unindexed, and to the recency window, but keep older syntheses."""
    since = policy.min_year(today)
    window = f'"{since}/01/01"[dp] : "3000"[dp]'
    if policy.keep_older_syntheses:
        window = f'({window} OR meta-analysis[pt] OR systematic review[pt])'
    return f"({query}) AND {window}"


def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _month(value: str | None) -> int:
    if not value:
        return 1
    if value.isdigit():
        return min(max(int(value), 1), 12)
    return _MONTHS.get(value[:3].lower(), 1)


def _date(article: ET.Element) -> date | None:
    """Electronic publication date if present, else the journal issue date."""
    ad = article.find(".//Article/ArticleDate")
    if ad is not None and ad.findtext("Year"):
        return date(int(ad.findtext("Year")), _month(ad.findtext("Month")), int(ad.findtext("Day") or 1))
    pd = article.find(".//Article/Journal/JournalIssue/PubDate")
    if pd is None:
        return None
    year = pd.findtext("Year")
    if year:
        day = pd.findtext("Day")
        return date(int(year), _month(pd.findtext("Month")), int(day) if day and day.isdigit() else 1)
    medline = pd.findtext("MedlineDate") or ""  # e.g. "2019 Nov-Dec"
    m = re.match(r"(\d{4})\s*([A-Za-z]{3})?", medline)
    return date(int(m.group(1)), _month(m.group(2)), 1) if m else None


def _abstract(article: ET.Element) -> str:
    parts = []
    for at in article.findall(".//Article/Abstract/AbstractText"):
        label = at.get("Label")
        body = _text(at)
        parts.append(f"{label.title()}: {body}" if label else body)
    return " ".join(p for p in parts if p)


def _authors(article: ET.Element) -> list[str]:
    out = []
    for a in article.findall(".//Article/AuthorList/Author"):
        if a.get("ValidYN", "Y") == "N":
            continue
        last = a.findtext("LastName")
        if last:
            out.append(f"{last} {a.findtext('Initials') or ''}".strip())
        elif a.findtext("CollectiveName"):
            out.append(a.findtext("CollectiveName") or "")
    return out


def parse_efetch(xml: str | bytes, retrieved_at: datetime | None = None) -> list[StudyRecord]:
    """Parse a PubmedArticleSet. Articles without a usable date or title are skipped."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        raise ValueError(f"malformed efetch XML: {e}") from e
    retrieved_at = retrieved_at or utcnow()
    records = []
    for art in root.findall("PubmedArticle"):
        pmid = art.findtext("MedlineCitation/PMID")
        title = _text(art.find(".//Article/ArticleTitle"))
        published = _date(art)
        if not pmid or not title or published is None:
            continue
        pub_types = [_text(p) for p in art.findall(".//Article/PublicationTypeList/PublicationType")]
        lowered = {t.lower() for t in pub_types}
        ids = {i.get("IdType"): (i.text or "").strip() for i in art.findall("PubmedData/ArticleIdList/ArticleId")}
        retraction_in = [
            c for c in art.findall(".//CommentsCorrectionsList/CommentsCorrections") if c.get("RefType") == "RetractionIn"
        ]
        retracted = bool(lowered & RETRACTED_TYPES) or bool(retraction_in)
        notice = None
        if retraction_in:
            ref = retraction_in[0]
            notice = (_text(ref.find("RefSource")) + (f" (PMID {ref.findtext('PMID')})" if ref.findtext("PMID") else "")).strip()
        elif retracted:
            notice = "PubMed publication type: Retracted Publication"
        records.append(
            StudyRecord(
                source_id=f"PMID:{pmid}",
                source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                title=title,
                abstract=_abstract(art),
                journal=art.findtext(".//Article/Journal/Title") or art.findtext(".//Article/Journal/ISOAbbreviation"),
                authors=_authors(art),
                published=published,
                publication_types=pub_types,
                retracted=retracted,
                is_preprint="preprint" in lowered,
                pmid=pmid,
                doi=ids.get("doi") or None,
                pmcid=(ids.get("pmc") or None),
                mesh_terms=[_text(d) for d in art.findall(".//MeshHeadingList/MeshHeading/DescriptorName")],
                source="pubmed",
                endpoint="pubmed:efetch",
                retrieved_at=retrieved_at,
                retraction_checked_by=["pubmed:publication_type"],
                retraction_notice=notice,
            )
        )
    return records


class PubMedSearch:
    """`SearchTool` over PubMed E-utilities."""

    name = "pubmed"

    def __init__(
        self,
        http: HttpClient,
        api_key: str | None = None,
        email: str | None = None,
        policy: RecencyPolicy = RecencyPolicy(),
        today: Callable[[], date] = date.today,
    ):
        self.http = http
        self.api_key = api_key
        self.email = email
        self.policy = policy
        self.today = today

    def _params(self, **kw: str | int) -> dict[str, str | int]:
        params: dict[str, str | int] = {"db": "pubmed", "tool": TOOL_NAME, **kw}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def search(self, query: str, max_results: int = 10) -> SearchResponse:
        term = build_term(query, self.today(), self.policy)
        try:
            resp = self.http.get(ESEARCH, self._params(term=term, retmax=max_results, retmode="json", sort="relevance"))
            data = resp.json()
            result = data.get("esearchresult") or {}
            if "ERROR" in result or "error" in data:
                return SearchResponse(query=query, state="error", message=f"esearch: {result.get('ERROR') or data.get('error')}")
            ids = [i for i in result.get("idlist", []) if i.isdigit()]
            if not ids:
                return SearchResponse(query=query, state="no_results", message="esearch: 0 hits")
            fetched = self.http.get(EFETCH, self._params(id=",".join(ids), retmode="xml"))
            records = parse_efetch(fetched.content)
        except FetchError as e:
            state = "no_results" if e.kind == "not_found" else e.kind
            return SearchResponse(query=query, state=state, message=str(e))
        except ValueError as e:  # malformed JSON / XML
            return SearchResponse(query=query, state="error", message=f"pubmed: {e}")
        order = {pmid: i for i, pmid in enumerate(ids)}  # efetch does not promise esearch order
        records = sorted((r for r in records if r.pmid in order), key=lambda r: order[r.pmid or ""])
        if not records:
            return SearchResponse(query=query, state="no_results", message="efetch returned no usable articles")
        return SearchResponse(query=query, state="ok", records=records[:max_results])
