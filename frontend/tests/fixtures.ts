import type { CoachAnswer, Finding } from "@/lib/api/types";

export const findings: Finding[] = [
  {
    id: "sq1:PMID:27433992",
    claim: "Each additional weekly set was associated with a 0.37% greater increase in muscle size.",
    source_id: "PMID:27433992",
    source_url: "https://example.org/schoenfeld-2017",
    title: "Dose-response relationship between weekly resistance training volume and increases in muscle mass",
    study_type: "meta_analysis",
    population: "Trained and untrained adults, mostly men aged 18-35",
    sample_size: 390,
    effect: "+0.37% per set",
    published: "2017-06-01",
    funding_conflict: false,
    retracted: false,
    snippet: "…",
    grade: "high",
  },
  {
    id: "sq2:DOI:10.1/xyz",
    claim: "No added thickness beyond 12 sets.",
    source_id: "DOI:10.1/xyz",
    source_url: "https://doi.org/10.1/xyz",
    title: null,
    study_type: "rct",
    population: "31 resistance-trained men",
    sample_size: 31,
    effect: null,
    published: "2022-03-01",
    funding_conflict: null,
    retracted: false,
    snippet: "…",
    grade: "moderate",
  },
];

export const answer: CoachAnswer = {
  status: "leaning",
  recommendation: "Aim for **10–20 sets** per muscle per week [1, 2].",
  confidence: "moderate",
  why: "Each extra set added 0.37% [1]. Trained lifters plateau around 12 [2].",
  positions: [
    { stance: "More volume keeps helping.", supporting_findings: ["sq1:PMID:27433992"], evidence_grade: "moderate", best_counterargument: "Short studies." },
    { stance: "Gains level off near 12 sets.", supporting_findings: ["sq2:DOI:10.1/xyz"], evidence_grade: "low", best_counterargument: "Small samples." },
  ],
  what_would_change_this: "A longer trial [2].",
  applies_to_you: "You train 4 days a week.",
  practical_protocol: "- Start at 12 sets [1].\n- Add 2 sets when progress stalls.",
  whats_new: null,
  cited_finding_ids: ["sq1:PMID:27433992", "sq2:DOI:10.1/xyz"],
  disclaimer: "Not medical advice.",
};
