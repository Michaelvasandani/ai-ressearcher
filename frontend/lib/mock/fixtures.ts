/**
 * Fixture data for mock mode (NEXT_PUBLIC_MOCK=1): realistic answers in the backend's exact
 * shapes, so every state of the UI can be demoed without the API. Finding ids follow the
 * backend's `sq<n>:<source id>` form; answer text carries `[n]` markers that point at
 * `cited_finding_ids[n-1]`.
 */
import type { CoachAnswer, Finding, RunOutcome, SubQuestion } from "@/lib/api/types";

const pubmedSearch = (title: string) => `https://pubmed.ncbi.nlm.nih.gov/?term=${encodeURIComponent(title)}`;

function finding(f: Omit<Finding, "retracted" | "funding_conflict" | "grade" | "snippet"> & Partial<Finding>): Finding {
  return {
    retracted: false,
    funding_conflict: false,
    grade: "moderate",
    snippet: f.claim,
    ...f,
  };
}

export type ScenarioKind = "leaning" | "consensus" | "referral" | "citations_failed";

export interface StepSpec {
  node: string;
  agent: string | null;
  message: string;
  model?: string;
  tokens?: number;
  cost?: number;
  ms?: number;
  sub_question_id?: string;
  position_id?: string;
  attempt?: number;
  status?: "ok" | "degraded" | "failed" | "skipped";
  input: string;
  output: string;
  json?: Record<string, unknown>;
  tools?: { tool: string; input: string; state: string; results: number }[];
}

export interface Scenario {
  kind: ScenarioKind;
  plan: SubQuestion[];
  findings: Finding[];
  answer: CoachAnswer | null;
  outcome: RunOutcome;
  /** Steps before the plan checkpoint, and after it. */
  before: StepSpec[];
  after: StepSpec[];
}

const HAIKU = "claude-haiku-5-5";
const SONNET = "claude-sonnet-5-5";
const OPUS = "claude-opus-5-5";

// --- Weekly volume (leaning, two sides) ---------------------------------------------------

const volumeFindings: Finding[] = [
  finding({
    id: "sq1:PMID:27433992",
    sub_question_id: "sq1",
    title: "Dose-response relationship between weekly resistance training volume and increases in muscle mass",
    claim: "Each additional weekly set was associated with a 0.37% greater increase in muscle size; 10+ sets per week outperformed fewer than 5.",
    source_id: "PMID:27433992",
    source_url: "https://pubmed.ncbi.nlm.nih.gov/27433992/",
    study_type: "meta_analysis",
    population: "15 studies, trained and untrained adults, mostly men aged 18-35",
    sample_size: 390,
    effect: "+0.37% muscle size per additional weekly set",
    published: "2017-06-01",
    grade: "high",
  }),
  finding({
    id: "sq1:MOCK-PELLAND-2024",
    sub_question_id: "sq1",
    title: "The resistance training dose-response: meta-regressions exploring the effects of weekly volume and frequency",
    claim: "Hypertrophy kept rising with weekly volume up to about 20 fractional sets, with diminishing returns per added set.",
    source_id: "OSF:sportrxiv",
    source_url: pubmedSearch("resistance training dose response meta-regressions weekly volume frequency"),
    study_type: "preprint",
    population: "67 studies, 2,058 adults, mixed training status",
    sample_size: 2058,
    effect: "diminishing but positive returns up to ~20 sets/week",
    published: "2024-11-12",
    grade: "low",
  }),
  finding({
    id: "sq2:MOCK-BAZVALLE-2022",
    sub_question_id: "sq2",
    title: "A systematic review of the effects of different resistance training volumes on muscle hypertrophy",
    claim: "In trained people, 12-20 weekly sets per muscle produced the most consistent gains; above 20 sets added little.",
    source_id: "DOI:10.2478/hukin-2022-0017",
    source_url: "https://doi.org/10.2478/hukin-2022-0017",
    study_type: "review",
    population: "Resistance-trained adults, 18-40 years",
    sample_size: null,
    effect: "plateau above ~20 sets/week",
    published: "2022-02-10",
  }),
  finding({
    id: "sq2:MOCK-AUBE-2022",
    sub_question_id: "sq2",
    title: "Progressive resistance training volume: effects on muscle thickness, mass, and strength adaptations in resistance-trained individuals",
    claim: "Moving from 12 to 18 or 24 weekly sets did not further increase quadriceps thickness over 8 weeks.",
    source_id: "PMID:MOCK-AUBE",
    source_url: pubmedSearch("Progressive resistance training volume muscle thickness resistance-trained individuals Aube"),
    study_type: "rct",
    population: "31 resistance-trained men, 18-35 years",
    sample_size: 31,
    effect: "no added thickness beyond 12 sets/week",
    published: "2022-03-01",
  }),
  finding({
    id: "sq3:MOCK-HAMMERT-2024",
    sub_question_id: "sq3",
    title: "Sex differences in hypertrophy responses to resistance training volume",
    claim: "Women and men gained similar relative muscle size at matched weekly volumes of 10-16 sets.",
    source_id: "PMID:MOCK-SEX",
    source_url: pubmedSearch("sex differences hypertrophy resistance training volume relative muscle size"),
    study_type: "meta_analysis",
    population: "Trained and untrained women and men, 18-45 years",
    sample_size: 1032,
    effect: "similar relative gains at 10-16 sets/week",
    published: "2024-05-20",
  }),
  finding({
    id: "sq4:MOCK-RIR-2023",
    sub_question_id: "sq4",
    title: "Proximity to failure and muscle hypertrophy: a meta-regression",
    claim: "Sets ended 1-3 repetitions short of failure produced hypertrophy comparable to sets taken to failure.",
    source_id: "PMID:MOCK-RIR",
    source_url: pubmedSearch("proximity to failure hypertrophy meta-regression repetitions in reserve"),
    study_type: "meta_analysis",
    population: "55 studies, adults of mixed training status",
    sample_size: 1880,
    effect: "comparable hypertrophy at 1-3 reps in reserve",
    published: "2023-09-04",
  }),
];

const volumeAnswer: CoachAnswer = {
  status: "leaning",
  recommendation:
    "Aim for about **10–20 hard sets per muscle per week**: gains rise with volume up to roughly 10 sets [1] and most trained lifters do best somewhere in 12–20 [3].",
  confidence: "moderate",
  why:
    "The largest dose-response meta-analysis found each extra weekly set added 0.37% muscle size, and 10+ sets beat fewer than 5 [1]. A newer meta-regression extends the benefit to about 20 sets with diminishing returns [2]. Trials in trained lifters, though, show the curve flattening: going from 12 to 18 or 24 sets added nothing over 8 weeks [4].",
  positions: [
    {
      stance: "More volume keeps helping, with diminishing returns, up to about 20 sets per muscle per week.",
      supporting_findings: ["sq1:PMID:27433992", "sq1:MOCK-PELLAND-2024"],
      evidence_grade: "moderate",
      best_counterargument:
        "Much of the upper range comes from a preprint and from short studies where swelling can inflate muscle thickness readings.",
    },
    {
      stance: "Gains level off around 10–12 sets for most trained lifters; extra sets mostly add fatigue.",
      supporting_findings: ["sq2:MOCK-BAZVALLE-2022", "sq2:MOCK-AUBE-2022"],
      evidence_grade: "low",
      best_counterargument:
        "The plateau rests on small trials of 31 men over 8 weeks, which are underpowered to detect modest extra gains.",
    },
  ],
  what_would_change_this:
    "A long (6+ month) trial in trained lifters showing clear extra growth above 20 sets [2], or a pre-registered trial confirming the 12-set plateau [4], would move the recommendation.",
  applies_to_you:
    "You train 4 days a week, so 12–16 sets per muscle fits in two sessions per muscle. Women gained similar relative size at the same volumes [5], so the range applies to you as well.",
  practical_protocol:
    "**Monday:** start at 12–14 sets per muscle per week, split across 2 sessions.\n\n- End most sets 1–3 reps short of failure [6].\n- Add 2 sets per muscle every 3–4 weeks only if you recover well and progress stalls.\n- Drop back to 10 sets in a deload week.",
  whats_new:
    "A 2024 meta-regression of 67 studies [2] pushed the useful range upward from the 2017 estimate [1], but it is a preprint.",
  cited_finding_ids: [
    "sq1:PMID:27433992",
    "sq1:MOCK-PELLAND-2024",
    "sq2:MOCK-BAZVALLE-2022",
    "sq2:MOCK-AUBE-2022",
    "sq3:MOCK-HAMMERT-2024",
    "sq4:MOCK-RIR-2023",
  ],
  disclaimer:
    "General education from published research, not medical advice. Talk to a qualified professional before changing training if you have an injury or health condition.",
};

const volumePlan: SubQuestion[] = [
  {
    id: "sq1",
    question: "What is the dose-response relationship between weekly sets per muscle and hypertrophy?",
    rationale: "Establishes the overall curve and where returns diminish.",
    search_terms: ["weekly training volume hypertrophy meta-analysis"],
  },
  {
    id: "sq2",
    question: "Do resistance-trained lifters gain more from 12+ weekly sets than from around 10?",
    rationale: "Tests whether the curve plateaus in trained people.",
    search_terms: ["high volume resistance trained muscle thickness randomized"],
  },
  {
    id: "sq3",
    question: "Do women respond to weekly training volume differently from men?",
    rationale: "The user's profile says female; checks the population match.",
    search_terms: ["sex differences resistance training volume hypertrophy"],
  },
  {
    id: "sq4",
    question: "How close to failure do sets need to be to count as hard sets for hypertrophy?",
    rationale: "Defines what a 'hard set' is in the recommendation.",
    search_terms: ["proximity to failure hypertrophy repetitions in reserve"],
  },
];

// --- Protein (consensus) --------------------------------------------------------------

const proteinFindings: Finding[] = [
  finding({
    id: "sq1:PMID:28698222",
    sub_question_id: "sq1",
    title: "A systematic review, meta-analysis and meta-regression of the effect of protein supplementation on resistance training-induced gains in muscle mass and strength",
    claim: "Gains in fat-free mass plateaued at about 1.62 g/kg/day of protein, with a 95% CI upper bound of 2.2 g/kg/day.",
    source_id: "PMID:28698222",
    source_url: "https://pubmed.ncbi.nlm.nih.gov/28698222/",
    study_type: "meta_analysis",
    population: "49 RCTs, 1,863 healthy adults doing resistance training",
    sample_size: 1863,
    effect: "plateau at 1.62 g/kg/day",
    published: "2018-03-01",
    grade: "high",
  }),
  finding({
    id: "sq2:MOCK-HELMS-2014",
    sub_question_id: "sq2",
    title: "A systematic review of dietary protein during caloric restriction in resistance trained lean athletes",
    claim: "In lean, resistance-trained athletes in a calorie deficit, 2.3-3.1 g/kg of fat-free mass per day best preserved lean mass.",
    source_id: "PMID:MOCK-HELMS",
    source_url: pubmedSearch("dietary protein caloric restriction resistance trained lean athletes systematic review"),
    study_type: "review",
    population: "Lean resistance-trained athletes during energy restriction",
    sample_size: null,
    effect: "2.3-3.1 g/kg FFM/day during a cut",
    published: "2014-04-01",
  }),
  finding({
    id: "sq2:MOCK-LONGLAND-2016",
    sub_question_id: "sq2",
    title: "Higher compared with lower dietary protein during an energy deficit combined with intense exercise promotes greater lean mass gain and fat mass loss",
    claim: "During a 40% energy deficit, 2.4 g/kg/day of protein led to a lean mass gain of 1.2 kg versus 0.1 kg at 1.2 g/kg/day over 4 weeks.",
    source_id: "PMID:MOCK-LONGLAND",
    source_url: pubmedSearch("Higher compared with lower dietary protein during an energy deficit lean mass gain"),
    study_type: "rct",
    population: "40 young men with overweight, untrained to recreationally active",
    sample_size: 40,
    effect: "+1.2 kg vs +0.1 kg lean mass",
    published: "2016-03-01",
  }),
];

const proteinAnswer: CoachAnswer = {
  status: "consensus",
  recommendation:
    "On a cut, eat about **1.6–2.2 g of protein per kg of bodyweight per day**, toward the top of that range the leaner you are [1][2].",
  confidence: "high",
  why:
    "Across 49 trials, gains in fat-free mass plateaued at about 1.62 g/kg/day, with an upper bound of 2.2 g/kg/day [1]. In a calorie deficit, higher intakes protect lean mass better: 2.4 g/kg/day produced a 1.2 kg lean gain versus 0.1 kg at 1.2 g/kg/day [3].",
  positions: [
    {
      stance: "Around 1.6–2.2 g/kg/day maximizes lean mass; more helps a little during a deficit.",
      supporting_findings: ["sq1:PMID:28698222", "sq2:MOCK-HELMS-2014", "sq2:MOCK-LONGLAND-2016"],
      evidence_grade: "high",
      best_counterargument: "Few trials test very lean, trained women in long deficits.",
    },
  ],
  what_would_change_this:
    "Long trials in very lean trained athletes showing benefits above 2.2 g/kg/day [2] would raise the upper end.",
  applies_to_you:
    "At 70 kg that is about 112–154 g of protein a day, spread over 3–5 meals.",
  practical_protocol:
    "- Set a daily target of 1.6–2.2 g/kg [1].\n- Get 0.4 g/kg per meal across 4 meals.\n- Keep it at the high end in the last weeks of a long cut [2].",
  whats_new: null,
  cited_finding_ids: ["sq1:PMID:28698222", "sq2:MOCK-HELMS-2014", "sq2:MOCK-LONGLAND-2016"],
  disclaimer:
    "General education from published research, not medical advice. If you have kidney disease or another medical condition, ask your doctor before raising protein intake.",
};

const proteinPlan: SubQuestion[] = [
  {
    id: "sq1",
    question: "What daily protein intake maximizes resistance-training gains in lean mass?",
    rationale: "The baseline requirement before a deficit.",
    search_terms: ["protein intake resistance training lean mass meta-analysis"],
  },
  {
    id: "sq2",
    question: "Does a calorie deficit raise the protein needed to preserve lean mass?",
    rationale: "The question is about a cut.",
    search_terms: ["protein energy deficit lean mass resistance trained"],
  },
  {
    id: "sq3",
    question: "How should protein be distributed across meals during a cut?",
    rationale: "Turns the daily target into a plan.",
    search_terms: ["protein distribution meals muscle protein synthesis"],
  },
];

// --- Steps ------------------------------------------------------------------------------

function researchSteps(plan: SubQuestion[], findings: Finding[]): StepSpec[] {
  return plan.map((sq, i) => {
    const kept = findings.filter((f) => f.sub_question_id === sq.id).length || 2;
    const seen = [23, 17, 12, 9][i % 4];
    return {
      node: "researcher",
      agent: "researcher",
      model: HAIKU,
      sub_question_id: sq.id,
      tokens: 9800 + i * 900,
      cost: 0.0021 + i * 0.0003,
      ms: 7400 + i * 800,
      message: `PubMed + Europe PMC: ${seen} studies → ${kept} kept · ${sq.id}`,
      input: `${sq.id}: ${sq.question} | queries: ${sq.search_terms.join("; ")}`,
      output: `ok: ${kept} findings from ${seen} records (2 searches)`,
      json: {
        sub_question_id: sq.id,
        findings: findings.filter((f) => f.sub_question_id === sq.id),
        status: { sub_question_id: sq.id, state: "ok", records_seen: seen, findings_kept: kept, searches_run: 2 },
      },
      tools: [
        { tool: "search:composite", input: sq.search_terms[0], state: "ok", results: seen },
        { tool: "findings_store", input: sq.id, state: "saved", results: kept },
      ],
    };
  });
}

function intakeStep(question: string, inScope = true): StepSpec {
  return {
    node: "intake",
    agent: "intake",
    model: HAIKU,
    tokens: 1180,
    cost: 0.0004,
    ms: 900,
    message: inScope ? "Question in scope" : "Out of scope (injury): referred to a professional",
    input: question,
    output: inScope ? "in scope (resistance_training)" : "out of scope: injury",
    json: {
      original_question: question,
      in_scope: inScope,
      scope_category: inScope ? "in_scope" : "injury",
      classified_by: inScope ? "llm" : "rule",
      profile: { sex: "female", training_age_years: 3, days_per_week: 4 },
    },
  };
}

function plannerStep(plan: SubQuestion[]): StepSpec {
  return {
    node: "planner",
    agent: "planner",
    model: SONNET,
    tokens: 2140,
    cost: 0.0118,
    ms: 4200,
    message: `Planned ${plan.length} sub-questions`,
    input: "scoped question + profile",
    output: `${plan.length} sub-questions`,
    json: { sub_questions: plan },
  };
}

function synthesisSteps(kind: "leaning" | "consensus", answer: CoachAnswer): StepSpec[] {
  const steps: StepSpec[] = [
    { node: "review", agent: "supervisor", message: "Review: research complete", input: "research records", output: "accept", ms: 3 },
    {
      node: "personalize",
      agent: "supervisor",
      message: kind === "leaning" ? "Personalized: down-weighted 2 findings from other populations" : "Personalized: every population matches",
      input: "findings + profile",
      output: kind === "leaning" ? "2 down-weighted (male-only samples)" : "none down-weighted",
      ms: 2,
      json: kind === "leaning" ? { downweighted: { "sq2:MOCK-AUBE-2022": "male-only sample; user is female" } } : { downweighted: {} },
    },
    {
      node: "position_mapper",
      agent: "position_mapper",
      model: SONNET,
      tokens: 6400,
      cost: 0.031,
      ms: 9800,
      message: kind === "leaning" ? "Positions mapped: leaning, 2 sides" : "Positions mapped: consensus, 1 side",
      input: "graded findings",
      output: `${kind}: ${answer.positions.length} positions`,
      json: { status: kind, positions: answer.positions },
    },
  ];
  if (kind === "leaning") {
    for (const round of [1, 2]) {
      for (const p of ["P1", "P2"]) {
        steps.push({
          node: "advocate",
          agent: "advocate",
          model: SONNET,
          position_id: p,
          attempt: round,
          tokens: 3900,
          cost: 0.019,
          ms: 7100,
          message: `Debate: round ${round} · ${p} ${round === 1 ? "opening" : "rebuttal"}`,
          input: `${p}: ${answer.positions[p === "P1" ? 0 : 1].stance}`,
          output: "3 claims, 0 rejected",
        });
      }
    }
    steps.push({ node: "debate_review", agent: "supervisor", message: "Debate: round 2 done", input: "4 turns", output: "done", ms: 1, attempt: 2 });
  }
  steps.push(
    {
      node: "judge",
      agent: "judge",
      model: SONNET,
      tokens: 7200,
      cost: 0.036,
      ms: 11200,
      message: `Verdict: ${kind}, confidence ${answer.confidence}`,
      input: "positions + debate + grades",
      output: `${kind}, ${answer.confidence}`,
      json: { status: kind, confidence: answer.confidence, recommendation: answer.recommendation },
    },
    {
      node: "writer",
      agent: "writer",
      model: SONNET,
      tokens: 8100,
      cost: 0.042,
      ms: 14600,
      message: "Answer drafted",
      input: "verdict + profile",
      output: "CoachAnswer",
      json: answer as unknown as Record<string, unknown>,
    },
    {
      node: "validate_citations",
      agent: "supervisor",
      message: "Citations: 14 claims and 22 numbers backed",
      input: "answer draft",
      output: "ok",
      ms: 12,
      json: { ok: true, claims_checked: 14, numbers_checked: 22, issues: [] },
    },
    { node: "check_answer", agent: "supervisor", message: "Answer accepted", input: "answer + citation report", output: "accept", ms: 2 },
    { node: "finalize", agent: "supervisor", message: "Done: answered", input: "state", output: "answered", ms: 1, json: { status: "answered" } },
  );
  return steps;
}

const answered = (message: string): RunOutcome => ({
  status: "answered",
  message,
  failure: null,
  degradations: [],
  referral_message: null,
});

export function scenarioFor(question: string): Scenario {
  const q = question.toLowerCase();
  if (/\b(injur|pain|knee|shoulder|back hurts|pregnan|medication|diabet|eating disorder|diagnos)/.test(q)) {
    const message =
      "This question is about an injury, which is outside what Evidence Coach can safely answer from general research. Please see a physiotherapist or sports-medicine doctor, who can examine you and give advice that fits your situation.";
    return {
      kind: "referral",
      plan: [],
      findings: [],
      answer: null,
      outcome: { status: "referred", message: "Out of scope: injury", failure: null, degradations: [], referral_message: message },
      before: [
        intakeStep(question, false),
        { node: "referral", agent: "supervisor", message: "Referral sent; no research needed", input: "intake", output: "referred", ms: 1 },
      ],
      after: [],
    };
  }
  if (/protein|cut|deficit|creatine/.test(q)) {
    return {
      kind: "consensus",
      plan: proteinPlan,
      findings: proteinFindings,
      answer: proteinAnswer,
      outcome: answered("Answered with high confidence"),
      before: [intakeStep(question), plannerStep(proteinPlan)],
      after: [
        { node: "dispatch", agent: "supervisor", message: `Researching ${proteinPlan.length} sub-questions`, input: "plan", output: "3 tasks", ms: 1 },
        ...researchSteps(proteinPlan, proteinFindings),
        ...synthesisSteps("consensus", proteinAnswer),
      ],
    };
  }
  if (/fail|uncited|citation/.test(q)) {
    const after = [
      { node: "dispatch", agent: "supervisor", message: "Researching 2 sub-questions", input: "plan", output: "2 tasks", ms: 1 },
      ...researchSteps(volumePlan.slice(0, 2), volumeFindings),
      ...synthesisSteps("leaning", volumeAnswer).slice(0, -3),
      {
        node: "validate_citations",
        agent: "supervisor",
        message: "Citations: 2 problems in 12 claims and 19 numbers",
        input: "answer draft",
        output: "2 issues",
        status: "failed" as const,
        ms: 10,
      },
      { node: "check_answer", agent: "supervisor", message: "Answer sent back for one rewrite", input: "report", output: "rewrite", ms: 1 },
      { node: "writer", agent: "writer", model: OPUS, attempt: 2, tokens: 9100, cost: 0.094, ms: 18200, message: "Answer rewritten (escalated)", input: "feedback", output: "CoachAnswer" },
      {
        node: "validate_citations",
        agent: "supervisor",
        message: "Citations: 1 problem in 12 claims and 18 numbers",
        input: "answer draft",
        output: "1 issue",
        status: "failed" as const,
        ms: 9,
      },
      { node: "check_answer", agent: "supervisor", message: "Answer rejected: citations failed twice", input: "report", output: "reject", ms: 1 },
      { node: "finalize", agent: "supervisor", message: "Done: failed", input: "state", output: "failed", ms: 1 },
    ] satisfies StepSpec[];
    return {
      kind: "citations_failed",
      plan: volumePlan.slice(0, 2),
      findings: volumeFindings.slice(0, 4),
      answer: null,
      outcome: {
        status: "failed",
        message: "The answer failed citation validation twice and was not returned.",
        failure: {
          reason: "citations_failed",
          node: "check_answer",
          message: "1 number is not backed by a cited finding after the rewrite",
        },
        degradations: [],
        referral_message: null,
      },
      before: [intakeStep(question), plannerStep(volumePlan.slice(0, 2))],
      after,
    };
  }
  return {
    kind: "leaning",
    plan: volumePlan,
    findings: volumeFindings,
    answer: volumeAnswer,
    outcome: answered("Answered: evidence leans one way"),
    before: [intakeStep(question), plannerStep(volumePlan)],
    after: [
      { node: "dispatch", agent: "supervisor", message: `Researching ${volumePlan.length} sub-questions`, input: "plan", output: "4 tasks", ms: 1 },
      ...researchSteps(volumePlan, volumeFindings),
      ...synthesisSteps("leaning", volumeAnswer),
    ],
  };
}

export const CITATION_FAILED_ISSUES = [
  {
    field: "practical_protocol",
    kind: "unbacked_number",
    sentence: "Add 2 sets per muscle every 8 weeks.",
    number: "8",
    cited: [],
    found_in: ["sq2:MOCK-AUBE-2022"],
    detail: '"8" is not in any finding the sentence cites; it appears in sq2:MOCK-AUBE-2022',
  },
];

export const MODELS_USED = { HAIKU, SONNET, OPUS };
