import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ConfidenceBar, STATUS_GLYPH, StatusMarker } from "@/components/StatusMarker";

describe("StatusMarker", () => {
  it.each([
    ["consensus", "●", "Consensus"],
    ["leaning", "◐", "Leaning"],
    ["genuinely_contested", "○", "Contested"],
  ] as const)("%s is %s with label %s", (status, glyph, label) => {
    const { container } = render(<StatusMarker status={status} />);
    expect(STATUS_GLYPH[status]).toBe(glyph);
    expect(container.querySelector(`[data-glyph="${glyph}"]`)).not.toBeNull();
    expect(screen.getByText(label)).toBeInTheDocument();
  });

  it("draws the three markers as filled, half and outline shapes", () => {
    const fills = (["consensus", "leaning", "genuinely_contested"] as const).map((s) => {
      const { container } = render(<StatusMarker status={s} />);
      const svg = container.querySelector("svg")!;
      return [svg.querySelector("circle")!.getAttribute("fill"), !!svg.querySelector("path")];
    });
    expect(fills).toEqual([
      ["currentColor", false],
      ["none", true],
      ["none", false],
    ]);
  });

  it("keeps the label for screen readers when hidden", () => {
    render(<StatusMarker status="leaning" showLabel={false} />);
    expect(screen.getByText("Leaning")).toHaveClass("sr-only");
  });
});

describe("ConfidenceBar", () => {
  it.each([
    ["low", 1],
    ["moderate", 2],
    ["high", 3],
  ] as const)("%s fills %i of 3 segments", (confidence, filled) => {
    const { container } = render(<ConfidenceBar confidence={confidence} />);
    const segments = container.querySelectorAll("[data-filled]");
    expect(segments).toHaveLength(3);
    expect(container.querySelectorAll('[data-filled="true"]')).toHaveLength(filled);
    expect(screen.getByRole("img", { name: `Confidence: ${confidence}` })).toBeInTheDocument();
  });
});
