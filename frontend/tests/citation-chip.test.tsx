import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { CitationChip } from "@/components/CitationChip";
import { Markdown } from "@/components/Markdown";
import { TooltipProvider } from "@/components/ui/tooltip";
import { citationIndex, linkifyCitations, markerNumbers, sourceHref } from "@/lib/citations";
import { answer, findings } from "./fixtures";

const wrap = (ui: React.ReactNode) => render(<TooltipProvider delayDuration={0}>{ui}</TooltipProvider>);

describe("citation markers", () => {
  it("expands lists and ranges like the backend", () => {
    expect(markerNumbers("1")).toEqual([1]);
    expect(markerNumbers("1, 3")).toEqual([1, 3]);
    expect(markerNumbers("2-4")).toEqual([2, 3, 4]);
    expect(markerNumbers("1; 2–3")).toEqual([1, 2, 3]);
  });

  it("linkifies [n] markers and leaves other brackets alone", () => {
    expect(linkifyCitations("0.37% per set [1].")).toBe("0.37% per set [1](#cite-1).");
    expect(linkifyCitations("10 sets [1, 2][3]")).toBe("10 sets [1, 2](#cite-1,2)[3](#cite-3)");
    expect(linkifyCitations("[sic] and [a link](https://x.org)")).toBe("[sic] and [a link](https://x.org)");
  });

  it("maps [n] to cited_finding_ids[n-1]", () => {
    const index = citationIndex(answer, findings);
    expect(index.id(1)).toBe("sq1:PMID:27433992");
    expect(index.finding(2)?.source_id).toBe("DOI:10.1/xyz");
    expect(index.finding(3)).toBeUndefined();
    expect(index.number("sq2:DOI:10.1/xyz")).toBe(2);
  });

  it("links PMIDs to PubMed and everything else to its source URL", () => {
    expect(sourceHref(findings[0])).toBe("https://pubmed.ncbi.nlm.nih.gov/27433992/");
    expect(sourceHref(findings[1])).toBe("https://doi.org/10.1/xyz");
  });
});

describe("CitationChip", () => {
  it("opens the source in a new tab", () => {
    wrap(<CitationChip n={1} finding={findings[0]} />);
    const chip = screen.getByRole("link", { name: /Citation 1/ });
    expect(chip).toHaveAttribute("href", "https://pubmed.ncbi.nlm.nih.gov/27433992/");
    expect(chip).toHaveAttribute("target", "_blank");
    expect(chip).toHaveTextContent("1");
  });

  it("shows study type, n, year and population on hover", async () => {
    wrap(<CitationChip n={1} finding={findings[0]} />);
    await userEvent.hover(screen.getByRole("link", { name: /Citation 1/ }));
    const tip = await screen.findByRole("tooltip");
    expect(tip).toHaveTextContent("Meta-analysis · n=390 · 2017");
    expect(tip).toHaveTextContent("mostly men aged 18-35");
  });

  it("renders a dead chip when the finding is missing", () => {
    wrap(<CitationChip n={7} />);
    expect(screen.queryByRole("link")).toBeNull();
    expect(screen.getByLabelText("Citation 7")).toHaveTextContent("7");
  });

  it("turns markers in answer markdown into chips", () => {
    const { container } = wrap(<Markdown text={answer.recommendation} index={citationIndex(answer, findings)} />);
    const chips = container.querySelectorAll("[data-cite]");
    expect([...chips].map((c) => c.getAttribute("data-cite"))).toEqual(["1", "2"]);
    expect(within(container).getByText("10–20 sets").tagName).toBe("STRONG");
    expect(container.textContent).not.toContain("[1");
  });
});
