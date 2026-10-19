"""Intake: scope classification and user-profile extraction.

Two layers decide scope:
1. A deterministic rule screen for unambiguous red-flag terms (pregnancy,
   eating disorders, named medications, ...). A hit returns the fixed referral
   without calling the model at all.
2. Otherwise the model classifies the question and extracts profile facts.

Out-of-scope questions get a fixed referral message chosen by code, never
model-written text, and never reach the planner.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from backend.llm import LLMClient, LLMRequest
from backend.schemas import OUT_OF_SCOPE_CATEGORIES, Intake, ScopeCategory, UserProfile

AGENT = "intake"

REFERRALS: dict[str, str] = {
    "injury": (
        "I can't help with injuries or pain. Please see a doctor or a physiotherapist, who can "
        "assess you in person and tell you what training is safe while you recover."
    ),
    "medical_condition": (
        "Questions that depend on a medical condition need someone who knows your health history. "
        "Please talk to your doctor, or a registered dietitian for nutrition questions."
    ),
    "medication": (
        "I can't advise on medications or how they interact with training or diet. Please ask your "
        "doctor or pharmacist."
    ),
    "pregnancy": (
        "Training and nutrition during pregnancy or postpartum should be guided by your midwife, "
        "obstetrician or doctor. Please check with them."
    ),
    "eating_disorder": (
        "I'm not the right resource for this. If you're struggling with eating, please reach out to "
        "a doctor or an eating-disorder specialist. In the US you can contact the ANAD helpline at "
        "1-888-375-7767; elsewhere, your doctor can point you to local support."
    ),
    "diagnosis": (
        "I can't diagnose symptoms or conditions. Please see a doctor, who can examine you properly."
    ),
    "off_topic": (
        "I can only help with resistance training and with protein and nutrition for body "
        "composition. Try asking something like \"How many sets per week for hypertrophy?\""
    ),
}

# Unambiguous red-flag terms. Kept deliberately narrow: ambiguous cases (e.g.
# "how do I avoid injury when deadlifting?") are left to the model.
_RULES: list[tuple[ScopeCategory, re.Pattern[str]]] = [
    (
        "pregnancy",
        re.compile(r"\b(pregnan\w*|trimester|postpartum|post-partum|breastfeed\w*|gestational)\b", re.I),
    ),
    (
        "eating_disorder",
        re.compile(
            r"\b(anorexi\w*|bulimi\w*|binge[- ]eating disorder|eating disorder|purg(e|ing)|"
            r"orthorexi\w*|arfid)\b",
            re.I,
        ),
    ),
    (
        "medication",
        re.compile(
            r"\b(metformin|semaglutide|ozempic|wegovy|tirzepatide|mounjaro|statins?|"
            r"ssris?|antidepressants?|beta[- ]blockers?|blood thinners?|warfarin|prednisone|"
            r"corticosteroids?|my medications?|my meds|prescription (drugs?|meds|medications?))\b",
            re.I,
        ),
    ),
    (
        "medical_condition",
        re.compile(
            r"\b(diabet\w*|kidney disease|ckd|heart (disease|failure|condition)|hypertension|"
            r"cancer|chemo\w*|pcos|crohn'?s|celiac|epilep\w*|osteoporosis)\b",
            re.I,
        ),
    ),
    (
        "injury",
        re.compile(
            r"\b(torn|sprain\w*|fractur\w*|herniat\w*|tendinitis|tendonitis|tendinopath\w*|"
            r"rotator cuff injury|acl|i (hurt|injured)|my injury)\b",
            re.I,
        ),
    ),
    ("diagnosis", re.compile(r"\b(diagnos\w*|what'?s wrong with (me|my))\b", re.I)),
]


def rule_screen(question: str) -> ScopeCategory | None:
    """Return the out-of-scope category for an unambiguous red-flag term, else None."""
    for category, pattern in _RULES:
        if pattern.search(question):
            return category
    return None


class IntakeDraft(BaseModel):
    """What the model returns."""

    model_config = ConfigDict(extra="forbid")

    scope_category: ScopeCategory
    scope_reason: str
    scoped_question: str
    profile: UserProfile


SYSTEM_PROMPT = """\
You are the intake step of an evidence-based coaching service. The service answers questions about \
resistance training (volume, frequency, intensity, exercise selection, progression, rest) and about \
protein and nutrition for body composition (muscle gain, fat loss, energy intake, protein intake, \
meal timing, and common sport supplements such as creatine, protein powder and caffeine, including \
whether they are safe for healthy adults). Answers are built from published research.

Do three things with the user's message.

1. Classify scope as exactly one category:
- in_scope: a training or nutrition question about healthy adults that research can answer.
- injury: current or past injuries, pain, rehab, or training around an injury.
- medical_condition: the answer depends on a disease or health condition (diabetes, heart, kidney, \
PCOS, cancer, high blood pressure, and so on).
- medication: medications, prescriptions, or drug interactions. Performance-enhancing drugs \
(anabolic steroids, SARMs, prohormones, growth hormone) also count; common sport supplements \
(creatine, protein powder, caffeine) do not.
- pregnancy: pregnancy, trying to conceive, or postpartum.
- eating_disorder: disordered eating, extreme restriction, purging, compensating for food with \
exercise, or anything suggesting an eating disorder.
- diagnosis: asks what is wrong with them or whether they have a condition.
- off_topic: unrelated to training or nutrition for body composition (for example cardio-only \
racing plans, sleep, mental health, or general chat).
Pick an out-of-scope category whenever the question clearly depends on one of those topics, even \
if it is also about training. Generic injury-prevention technique questions from healthy people \
(for example "should I wear a belt for heavy squats?") are in_scope.

2. Rewrite the question as one clear, self-contained, answerable question in scoped_question. Keep \
the user's intent; do not add assumptions. If out of scope, restate it neutrally.

3. Extract profile facts the user states about themselves into profile. Only record facts that are \
stated or directly implied; leave everything else null. training_status: untrained (never lifted), \
novice (under ~1 year), intermediate (~1-3 years), advanced (3+ years of consistent training).

scope_reason: one sentence explaining the category."""


def _render_user(question: str, profile: UserProfile | None) -> str:
    known = profile.model_dump_json(exclude_none=True) if profile and not profile.is_empty() else "{}"
    return f"Known profile (from settings): {known}\n\nUser message:\n<message>\n{question}\n</message>"


def out_of_scope_intake(question: str, category: ScopeCategory, reason: str, classified_by: str, profile: UserProfile | None = None) -> Intake:
    return Intake(
        original_question=question,
        scoped_question=question,
        in_scope=False,
        scope_category=category,
        scope_reason=reason,
        classified_by=classified_by,  # type: ignore[arg-type]
        profile=profile or UserProfile(),
        referral_message=REFERRALS[category],
    )


class IntakeAgent:
    def __init__(self, llm: LLMClient, model: str | None = None):
        self.llm = llm
        self.model = model

    def run(self, question: str, profile: UserProfile | None = None) -> Intake:
        question = question.strip()
        if not question:
            raise ValueError("question is empty")

        category = rule_screen(question)
        if category is not None:
            return out_of_scope_intake(
                question, category, f"Matched a {category.replace('_', ' ')} red-flag term.", "rule", profile
            )

        request = LLMRequest.for_agent(
            AGENT,
            system=SYSTEM_PROMPT,
            user=_render_user(question, profile),
            output_type=IntakeDraft,
            model=self.model,
        )
        draft = self.llm.complete(request).output
        merged = (profile or UserProfile()).merged_with(draft.profile)

        if draft.scope_category != "in_scope":
            assert draft.scope_category in OUT_OF_SCOPE_CATEGORIES | {"off_topic"}
            return out_of_scope_intake(question, draft.scope_category, draft.scope_reason, "llm", merged)

        return Intake(
            original_question=question,
            scoped_question=draft.scoped_question.strip() or question,
            in_scope=True,
            scope_category="in_scope",
            scope_reason=draft.scope_reason,
            classified_by="llm",
            profile=merged,
        )
