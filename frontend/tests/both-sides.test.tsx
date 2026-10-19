import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AnswerCard } from "@/components/AnswerCard";
import { BothSides, showsBothSides } from "@/components/BothSides";
import { FailedCard, ReferralCard } from "@/components/OutcomeCards";
import { TooltipProvider } from "@/components/ui/tooltip";
import { citationIndex } from "@/lib/citations";
import type { CoachAnswer } from "@/lib/api/types";
import { answer, findings } from "./fixtures";

const wrap = (ui: React.ReactNode) => render(<TooltipProvider>{ui}</TooltipProvider>);
const consensus: CoachAnswer = { ...answer, status: "consensus", positions: [answer.positions[0]] };

describe("both-sides rule", () => {
  it("shows both sides for leaning and contested answers with two positions", () => {
    expect(showsBothSides(answer)).toBe(true);
    expect(showsBothSides({ ...answer, status: "genuinely_contested" })).toBe(true);
  });

  it("omits both sides for consensus, even with extra positions", () => {
    expect(showsBothSides(consensus)).toBe(false);
    expect(showsBothSides({ ...answer, status: "consensus" })).toBe(false);
    const { container } = wrap(<BothSides answer={{ ...answer, status: "consensus" }} index={citationIndex(answer, findings)} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("omits both sides when only one position is left", () => {
    expect(showsBothSides({ ...answer, positions: [answer.positions[0]] })).toBe(false);
  });

  it("renders each side with stance, grade, key studies and best counterargument", () => {
    wrap(<BothSides answer={answer} index={citationIndex(answer, findings)} />);
    const a = screen.getByRole("region", { name: "Side A" });
    expect(a).toHaveTextContent("More volume keeps helping.");
    expect(a).toHaveTextContent("Moderate certainty");
    expect(a).toHaveTextContent("Short studies.");
    expect(a.querySelector('[data-cite="1"]')).not.toBeNull();
    const b = screen.getByRole("region", { name: "Side B" });
    expect(b).toHaveTextContent("Low certainty");
    expect(b.querySelector('[data-cite="2"]')).not.toBeNull();
  });

  it("AnswerCard drops the section for consensus and keeps the rest", () => {
    wrap(<AnswerCard answer={consensus} findings={findings} />);
    expect(screen.queryByTestId("both-sides")).toBeNull();
    expect(screen.getByText("Practical protocol")).toBeInTheDocument();
    expect(screen.getByText("Not medical advice.")).toBeInTheDocument();
    expect(screen.queryByText("What's new")).toBeNull(); // whats_new is null
  });

  it("AnswerCard shows both sides for a leaning answer", () => {
    wrap(<AnswerCard answer={answer} findings={findings} />);
    expect(screen.getByTestId("both-sides")).toBeInTheDocument();
    expect(screen.getByText("Leaning")).toBeInTheDocument();
  });
});

describe("failed and referred runs", () => {
  it("explains citations_failed and still lists the evidence", () => {
    wrap(
      <FailedCard
        outcome={{
          status: "failed",
          message: "failed",
          failure: { reason: "citations_failed", node: "check_answer", message: "1 number not backed" },
          degradations: [],
          referral_message: null,
        }}
        citations={{
          attempt: 2,
          ok: false,
          claims_checked: 12,
          numbers_checked: 18,
          numbers_from_user: 0,
          citations: 9,
          issues: [{ field: "why", kind: "unbacked_number", sentence: "Add 17 sets.", number: "17", cited: [], found_in: [], detail: '"17" is not in any cited finding' }],
        }}
        findings={findings}
      />,
    );
    expect(screen.getByTestId("failed-card")).toHaveAttribute("data-reason", "citations_failed");
    expect(screen.getByText(/citations did not check out/)).toBeInTheDocument();
    expect(screen.getByText("“Add 17 sets.”")).toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: /Citation/ })).toHaveLength(2);
  });

  it("shows the fixed referral message", () => {
    wrap(
      <ReferralCard
        outcome={{ status: "referred", message: "Out of scope", failure: null, degradations: [], referral_message: "Please see a physiotherapist." }}
      />,
    );
    expect(screen.getByText("Please see a physiotherapist.")).toBeInTheDocument();
  });
});
