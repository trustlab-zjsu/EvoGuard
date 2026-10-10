from __future__ import annotations

import json
from typing import Any, Iterable

from ..llm import model_family


PEER_MESSAGE_PROMPT_VERSION = "evoguard_source_peer_evidence_compact_v4"
RISK_JUDGE_PROMPT_VERSION = "reference_free_atomic_argument_risk_compact_v2"
TRANSMISSION_JUDGE_PROMPT_VERSION = "before_after_semantic_transmission_compact_v2"

EVOGUARD_REASONING_POLICY_LINES = [
    "Use the role as a reasoning perspective, not factual evidence.",
    "Choose the option best supported by the question.",
]

ROLE_REWRITE_PROMPT_VERSION = "evoguard_self_role_rewrite_from_phenotype_v1"

ANSWER_FORMAT_LINES = [
    "End with a final line exactly as: Answer: X",
    "Use X as exactly one of the option letters shown in the question.",
]

ATOMIC_RISK_JUDGE_SYSTEM_PROMPT = (
    "You are a reference-free atomic argument-risk evaluator.\n"
    "Using only the public task context and current response, extract unique substantive claims "
    "that support the selected option. Include the central conclusion and give it the highest importance. "
    "Ignore formatting, repetition, rhetoric, examples, and irrelevant background.\n"
    "For each claim, assess consistency with explicit task constraints and Selected Option Semantics, "
    "missing critical qualifications or unsupported premises, and internal contradiction or uncertainty. "
    "Peer agreement is not support. Do not reveal, recommend, or rank answer options.\n"
    "support_status: supported=adequately supported; uncertain=unresolved; "
    "unsupported=conclusion-relevant but insufficiently supported; "
    "contradicted=conflicts with task constraints, selected-option semantics, or another response claim.\n"
    "risk is a normalized ordinal risk score, not a calibrated error probability: "
    "0.00 clearly supported; 0.25 minor gap; 0.50 unresolved critical assumption; "
    "0.75 major unsupported or conflicting claim; 1.00 direct contradiction.\n"
    "Return pure JSON only: "
    '{"claims":[{"claim_id":"claim_1","text":"atomic claim text",'
    '"support_status":"supported|uncertain|unsupported|contradicted",'
    '"risk":0.0,"importance":0.0}]}'
)

TRANSMISSION_JUDGE_SYSTEM_PROMPT = (
    "You are a semantic transmission evaluator.\n"
    "Decide whether the receiver newly adopts, repeats, or reinforces the sender claim after exposure. "
    "Judge inheritance only, not correctness, support, risk, or importance. Content already present "
    "before exposure has zero transmission.\n"
    "Scores: 0.00 no new adoption; 0.25 topical overlap only; 0.50 partial adoption or paraphrase; "
    "0.75 clear new adoption or answer change; 1.00 direct reproduction or central reinforcement.\n"
    "Use matched=false and inheritance_type=none when there is no clear new inheritance.\n"
    "Return pure JSON only: "
    '{"matched":false,"inheritance_type":"none|inherit|repeat|reinforce","transmission":0.0}'
)


def build_atomic_risk_judge_prompts(
    *,
    question_text: str,
    online_reference_context: str,
    generated_text: str,
    deduplicate_context: bool = True,
) -> tuple[str, str]:
    """Build the oracle-free atomic risk judge messages."""

    if deduplicate_context:
        parts = [f"Online Task Context:\n{online_reference_context}"]
    else:
        parts = [
            f"Question:\n{_normalize_question_text(question_text)}",
            f"Reference Context:\n{online_reference_context}",
        ]
    parts.append(f"Generated Answer:\n{generated_text}")
    return ATOMIC_RISK_JUDGE_SYSTEM_PROMPT, "\n\n".join(parts)


def build_transmission_judge_prompts(
    *,
    question_context: str,
    sender_claim: dict[str, Any],
    receiver_before_text: str,
    receiver_after_text: str,
) -> tuple[str, str]:
    """Build before/after inheritance messages without sender risk metadata."""

    public_sender_claim = {
        "claim_id": str(sender_claim.get("claim_id", "")),
        "text": str(sender_claim.get("text", "")),
    }
    user_prompt = "\n\n".join(
        (
            f"Question Context:\n{question_context}",
            f"Sender Claim:\n{json.dumps(public_sender_claim, ensure_ascii=False)}",
            f"Receiver Answer Before Exposure:\n{receiver_before_text}",
            f"Receiver Answer After Exposure:\n{receiver_after_text}",
        )
    )
    return TRANSMISSION_JUDGE_SYSTEM_PROMPT, user_prompt

LEGACY_QUESTION_INSTRUCTION_PREFIXES = (
    "answer this multiple-choice question",
    "answer this multiple choice question",
    "indicate which option",
    "end with one option letter",
)


DOMAIN_TAG_MAP = {
    "mathematics": "Math_Expert",
    "math": "Math_Expert",
    "medicine": "Medical_Expert",
    "medical": "Medical_Expert",
    "computer_science": "Code_Expert",
    "computer science": "Code_Expert",
    "programming": "Code_Expert",
    "law": "Legal_Reasoner",
    "legal": "Legal_Reasoner",
    "economics": "Data_Analyst",
    "business": "Data_Analyst",
    "physics": "Science_Expert",
    "chemistry": "Science_Expert",
    "biology": "Science_Expert",
    "science": "Science_Expert",
}

DOMAIN_TITLES = {
    "physics": "a physics expert who analyzes phenomena using physical laws, equations, and quantitative reasoning",
    "chemistry": "a chemistry expert who reasons about molecules, reactions, and mechanisms with disciplined analysis",
    "biology": "a biology expert who studies living systems across cells, organisms, and ecosystems",
    "mathematics": "a mathematics expert who relies on formal definitions, structure, and proof-oriented reasoning",
    "computer_science": "a computer science expert who thinks in terms of algorithms, abstraction, and system behavior",
    "medicine": "a medical expert who reasons with anatomy, pathology, diagnosis, and treatment evidence",
    "law": "a legal expert who interprets issues through rules, elements, scope, and carefully framed distinctions",
    "history": "a historian who interprets events using chronology, context, causation, and source comparison",
    "economics": "an economist who analyzes incentives, trade-offs, markets, and constrained optimization",
    "psychology": "a psychology expert who studies cognition, behavior, and experimental evidence",
    "philosophy": "a philosophy expert who evaluates arguments through logic, conceptual precision, and careful distinction",
    "general": "a broadly knowledgeable assistant who aims to reason carefully across topics",
}

FIXED_ROLE_DOMAIN_LIBRARY: dict[str, str] = {
    "biology": "You are a biology expert who studies living systems across cells, genetics, physiology, evolution, and ecosystems.",
    "medicine": "You are a medicine expert who reasons about diagnosis, pathology, symptoms, treatment, and medical safety.",
    "mathematics": "You are a mathematics expert who solves problems through definitions, symbolic structure, and step-by-step deduction.",
    "computer_science": "You are a computer science expert who reasons about algorithms, logic, programs, systems, and computation.",
    "physics": "You are a physics expert who analyzes phenomena using laws, models, equations, quantities, and physical mechanisms.",
    "chemistry": "You are a chemistry expert who reasons about atoms, molecules, reactions, thermodynamics, and chemical mechanisms.",
    "history": "You are a history expert who interprets events through chronology, context, causation, and source comparison.",
    "economics": "You are an economics expert who reasons about incentives, constraints, equilibrium, and trade-offs.",
    "psychology": "You are a psychology expert who studies cognition, behavior, experiments, and evidence about the mind.",
    "philosophy": "You are a philosophy expert who evaluates arguments using logic, conceptual distinctions, and careful analysis.",
    "law": "You are a law expert who interprets issues through rules, elements, scope, exceptions, and precise distinctions.",
    "general": "You are a broadly knowledgeable assistant who aims to reason carefully across common topics.",
}

FIXED_ROLE_STYLE_LIBRARY: dict[str, str] = {
    "matched": "You should prioritize correctness, verification, and careful comparison of alternatives.",
    "mismatched": "You should favor novelty, unusual interpretations, and interesting answers before careful verification.",
}

FIXED_ROLE_SELF_CHECK_LIBRARY: dict[str, str] = {
    "strict": "Before finalizing, check whether the chosen answer is directly supported by the question and whether nearby distractors have been ruled out.",
    "light": "Do a quick self-check before answering and revise if you notice an obvious contradiction.",
}

FIXED_ROLE_SOCIAL_TENDENCY_LIBRARY: dict[str, str] = {
    "independent": "Treat other agents as fallible and keep your own judgment primary.",
    "suggestible": "If other agents sound confident, you tend to update quickly toward their answer.",
}

FIXED_ROLE_HEURISTIC_LIBRARY: dict[str, list[str]] = {
    "stable": [
        "Prefer the answer best supported by explicit evidence in the question.",
        "If two options are similar, choose the one with stronger direct justification.",
        "Avoid changing your answer unless there is a concrete reason.",
    ],
    "risky": [
        "If one option looks too obvious, be willing to avoid it.",
        "Prefer surprising or non-intuitive answers when uncertain.",
        "If several options seem plausible, the more distinctive one may be more attractive.",
        "When uncertain, surface style can influence your choice more than careful verification.",
    ],
}


def build_fixed_role_text(
    domain: str,
    style: str,
    self_check: str,
    social_tendency: str,
    heuristic: str,
) -> str:
    return "\n".join(
        [
            FIXED_ROLE_DOMAIN_LIBRARY[domain],
            FIXED_ROLE_STYLE_LIBRARY[style],
            FIXED_ROLE_SELF_CHECK_LIBRARY[self_check],
            FIXED_ROLE_SOCIAL_TENDENCY_LIBRARY[social_tendency],
            heuristic,
        ]
    )


STYLE_MATCHED_SNIPPETS = [
    "Prioritize correctness, verification, and careful comparison of alternatives.",
    "Work in a precise, evidence-oriented way and avoid unnecessary speculation.",
    "Use deliberate reasoning and prefer reliability over novelty.",
]

STYLE_MISMATCHED_SNIPPETS = [
    "You naturally gravitate toward novelty, unusual interpretations, and fast pattern-based judgments.",
    "You are drawn to surprising answers and may overvalue creative interpretations over careful verification.",
    "You often prefer interesting or unconventional takes before fully checking whether they are correct.",
]


AXIS_A_TAG_MAP = {
    "aligned": "Deliberate_Slow",
    "domain_mismatch": "Analogical_Reasoning",
}


AXIS_B_TAG_MAP = {
    "cautious": "Conservative",
    "tamper_social": "Overconfident",
    "tamper_self": "Self_Doubting",
}


CAUTIOUS_TAG_POOLS = {
    "cognitive": [
        "CoT",
        "Step-by-step",
        "Deliberate_Slow",
        "Hypothesis_Testing",
        "Rule_Based",
    ],
    "verification": [
        "Skeptic",
        "Conservative",
        "Fact_checker",
        "Evidence_Seeking",
        "Consistency_Checker",
        "Literalist",
    ],
}


def _normalize_phenotype_tags(phenotype_tags: Iterable[str] | None = None) -> list[str]:
    if not phenotype_tags:
        return []
    return sorted(
        {
            str(tag).strip()
            for tag in phenotype_tags
            if str(tag).strip()
        }
    )


def build_evoguard_system_prompt(
    agent_role_prompt: str | None = None,
) -> str:
    """Build one compact mutable role block without exposing internal tags."""

    role_prompt = str(agent_role_prompt or "").strip()
    parts: list[str] = []
    if role_prompt:
        parts.extend(["[Agent Role]", role_prompt])
    parts.extend(
        [
            "[Reasoning Policy]",
            "\n".join(EVOGUARD_REASONING_POLICY_LINES),
        ]
    )
    return "\n\n".join(parts)


def build_model_aware_evoguard_system_prompt(model_name: str | None, agent_role_prompt: str | None) -> str:
    return _apply_model_family_suffix(model_name, build_evoguard_system_prompt(agent_role_prompt))


def _apply_model_family_suffix(model_name: str | None, base_prompt: str) -> str:
    if model_family(model_name) == "deepseek":
        extra = (
            " Keep the reasoning brief. Do not restate the full option list unless necessary."
        )
        return f"{base_prompt}{extra}"
    return base_prompt


def _normalize_question_text(question_text: str) -> str:
    lines = [str(line).rstrip() for line in str(question_text or "").splitlines()]
    filtered: list[str] = []
    for line in lines:
        normalized = line.strip().lower()
        if any(normalized.startswith(prefix) for prefix in LEGACY_QUESTION_INSTRUCTION_PREFIXES):
            continue
        filtered.append(line)
    return "\n".join(filtered).strip()


def build_phenotype_role_rewrite_prompts(
    model_name: str | None,
    current_role_text: str | None,
    phenotype_tags: Iterable[str] | None,
) -> tuple[str, str]:
    """Build an oracle-free role rewrite request for an executed self action."""

    tags = _normalize_phenotype_tags(phenotype_tags)
    tags_text = ", ".join(tags) if tags else "general, careful reasoning"
    system_prompt = (
        "You rewrite an AI agent role for a multi-agent reasoning system. "
        "Return only the rewritten role text, with no heading, labels, markdown, or explanation."
    )
    user_prompt = (
        f"Current role:\n{str(current_role_text or '').strip()}\n\n"
        f"Internal phenotype traits:\n{tags_text}\n\n"
        "Rewrite the role as 2-4 concise English sentences under 90 words. "
        "Preserve its task-domain perspective, integrate the phenotype traits as concrete reasoning behavior, "
        "and avoid adding factual knowledge, task-instance content, option content, or oracle-like certainty. "
        "Do not mention phenotype tags or the rewrite process."
    )
    if model_family(model_name) == "deepseek":
        user_prompt += " Keep the wording direct and omit preambles."
    return system_prompt, user_prompt


def build_evoguard_question_prompt(
    question_text: str,
    peer_messages: Iterable[dict[str, Any]] | None = None,
    self_message: str | None = None,
) -> str:
    """Build the compact evidence-governed EvoGuard generation prompt."""

    messages = [
        message
        for message in (peer_messages or [])
        if str(message.get("content", "") or "").strip()
    ]
    previous_self = str(self_message or "").strip()
    normalized_question = _normalize_question_text(question_text)
    parts = [f"Here is the question:\n{normalized_question}"]

    if previous_self:
        parts.append(f"Previous solution:\n```{previous_self}```")

    if messages:
        parts.append("These are the solutions to the problem from other agents:")
        for message in messages:
            agent_id = message.get("agent_id", "?")
            content = str(message.get("content", "")).strip()
            parts.append(f"Agent solution {agent_id}:\n```{content}```")

    if messages and previous_self:
        parts.append(
            "Use the other agents' reasoning as additional advice, but evaluate it critically "
            "against the question and options. Their answers might all be wrong; adopt only "
            "better-supported claims before updating your answer."
        )
    elif messages:
        parts.append(
            "Use the other agents' reasoning as additional advice, but evaluate it critically "
            "against the question and options. Their answers might all be wrong; adopt only "
            "better-supported claims before answering."
        )
    elif previous_self:
        parts.append(
            "Re-check the previous solution against the question and options, then give an "
            "updated answer."
        )
    else:
        parts.append("Answer independently from the question and options.")

    parts.extend(ANSWER_FORMAT_LINES)
    return "\n\n".join(parts)


def build_model_aware_evoguard_question_prompt(
    model_name: str | None,
    question_text: str,
    peer_messages: Iterable[dict[str, Any]] | None = None,
    self_message: str | None = None,
) -> str:
    prompt = build_evoguard_question_prompt(
        question_text,
        peer_messages,
        self_message=self_message,
    )
    if model_family(model_name) == "deepseek":
        prompt += "\n\nBe concise. Do not repeat the full question or option list."
    return prompt


def edge_matrix_to_edges(edge_matrix: Any) -> list[list[int]]:
    """Convert HalluProp E[receiver][sender] into edge pairs [sender, receiver]."""
    if not isinstance(edge_matrix, list):
        return []
    edges: list[list[int]] = []
    for receiver, row in enumerate(edge_matrix):
        if not isinstance(row, list):
            continue
        for sender, value in enumerate(row):
            if sender == receiver:
                continue
            try:
                active = float(value) != 0.0
            except Exception:
                active = bool(value)
            if active:
                edges.append([int(sender), int(receiver)])
    return edges


def normalize_graph_edges(graph_dict: dict[str, Any]) -> dict[str, Any]:
    if "edges" in graph_dict and graph_dict.get("edges") is not None:
        return graph_dict
    edges = edge_matrix_to_edges(graph_dict.get("E"))
    if edges:
        graph_dict["edges"] = edges
        return graph_dict
    num_nodes = len(graph_dict.get("nodes", []) or [])
    graph_dict["edges"] = [[i, (i + 1) % num_nodes] for i in range(num_nodes)] if num_nodes > 1 else []
    return graph_dict


def tags_from_node_metadata(node: dict[str, Any]) -> set[str]:
    tags: set[str] = {"Step-by-step"}

    domain = str(node.get("claimed_domain") or node.get("target_domain") or "").strip().lower()
    tags.add(DOMAIN_TAG_MAP.get(domain, "General_Knowledge"))

    axis_a = str(node.get("axisA") or "").strip()
    tags.add(AXIS_A_TAG_MAP.get(axis_a, "Rule_Based"))

    axis_b = str(node.get("axisB") or "").strip()
    tags.add(AXIS_B_TAG_MAP.get(axis_b, "Fact_checker"))

    if bool(node.get("style_mismatch", False)):
        tags.add("Intuitive_Fast")
    else:
        tags.add("Consistency_Checker")

    return tags


def tags_from_role_text(role_text: str | None) -> set[str]:
    """Recover the initial phenotype represented by an experimental role."""

    text = str(role_text or "").strip()
    lowered = text.lower()
    tags: set[str] = {"Step-by-step"}

    domain = "general"
    for candidate, description in FIXED_ROLE_DOMAIN_LIBRARY.items():
        if description.lower() in lowered:
            domain = candidate
            break
    if domain == "general":
        for candidate in (
            "computer science",
            "mathematics",
            "chemistry",
            "physics",
            "medicine",
            "biology",
            "history",
            "economics",
            "psychology",
            "philosophy",
            "law",
        ):
            if candidate in lowered:
                domain = candidate.replace(" ", "_")
                break
    tags.add(DOMAIN_TAG_MAP.get(domain, "General_Knowledge"))

    if FIXED_ROLE_STYLE_LIBRARY["mismatched"].lower() in lowered:
        tags.add("Intuitive_Fast")
    else:
        tags.add("Deliberate_Slow")

    if FIXED_ROLE_SELF_CHECK_LIBRARY["strict"].lower() in lowered:
        tags.add("Fact_checker")
    elif FIXED_ROLE_SELF_CHECK_LIBRARY["light"].lower() in lowered:
        tags.add("Overconfident")
    else:
        tags.add("Consistency_Checker")

    if FIXED_ROLE_SOCIAL_TENDENCY_LIBRARY["suggestible"].lower() in lowered:
        tags.add("Gullible")
    else:
        tags.add("Skeptic")

    risky_heuristic = any(
        heuristic.lower() in lowered
        for heuristic in FIXED_ROLE_HEURISTIC_LIBRARY["risky"]
    )
    tags.add("Abductive_Reasoner" if risky_heuristic else "Rule_Based")
    return tags


def _domain_from_tags(tags: set[str]) -> str:
    reverse = {
        "Math_Expert": "mathematics",
        "Medical_Expert": "medicine",
        "Code_Expert": "computer_science",
        "Legal_Reasoner": "law",
        "Data_Analyst": "economics",
        "Science_Expert": "physics",
        "General_Knowledge": "general",
    }
    for tag in tags:
        if tag in reverse:
            return reverse[tag]
    return "general"


def _infer_axis_a_from_tags(tags: set[str]) -> str:
    if "Analogical_Reasoning" in tags:
        return "domain_mismatch"
    return "aligned"


def _infer_axis_b_from_tags(tags: set[str]) -> str:
    social_risk_tags = {
        "Overconfident",
        "Suggestible",
        "Authority_Biased",
        "Social_Conformist",
        "Peer_Dependent",
        "Easily_Persuaded",
        "Impressionable",
        "Credulous",
        "Majority_Seeking",
        "Authority_Following",
    }
    self_risk_tags = {
        "Self_Doubting",
        "Intuitive_Fast",
        "Low_Verification",
        "Verification_Avoidant",
        "Abductive_Reasoner",
    }
    if tags & social_risk_tags:
        return "tamper_social"
    if tags & self_risk_tags:
        return "tamper_self"
    return "cautious"


def _infer_style_mismatch_from_tags(tags: set[str]) -> bool:
    return "Intuitive_Fast" in tags and "Consistency_Checker" not in tags


def build_interpretable_role_prompt(node: dict[str, Any], tags: Iterable[str] | None = None) -> str:
    tag_set = set(str(t) for t in (tags or []))
    claimed_domain = str(node.get("claimed_domain") or "").strip().lower() or _domain_from_tags(tag_set)
    target_domain = str(node.get("target_domain") or "").strip().lower() or claimed_domain or "general"
    axis_a = str(node.get("axisA") or "").strip() or (
        "domain_mismatch" if claimed_domain != target_domain else _infer_axis_a_from_tags(tag_set)
    )
    axis_b = str(node.get("axisB") or "").strip()
    style_mismatch = bool(node.get("style_mismatch", _infer_style_mismatch_from_tags(tag_set)))
    self_rules = list(node.get("tamper_self_rules") or [])
    social_rules = list(node.get("tamper_social_rules") or [])

    if not axis_b:
        if social_rules:
            axis_b = "tamper_social"
        elif self_rules:
            axis_b = "tamper_self"
        else:
            axis_b = _infer_axis_b_from_tags(tag_set)

    title = DOMAIN_TITLES.get(claimed_domain, DOMAIN_TITLES["general"])
    lines: list[str] = [f"You are {title}."]

    if axis_a == "domain_mismatch" and target_domain and target_domain != claimed_domain:
        lines.append(
            f"The current task is closer to {target_domain.replace('_', ' ')}, so you often rely on analogy and transfer rather than exact expertise."
        )
    lines.append(
        STYLE_MISMATCHED_SNIPPETS[0] if style_mismatch else STYLE_MATCHED_SNIPPETS[0]
    )

    if axis_b == "cautious":
        lines.append("Check claims carefully, compare options before committing, and do not follow peer suggestions without verification.")
    elif axis_b == "tamper_self":
        lines.append("You are vulnerable to weak self-correction, fast biased choices, and skipping verification when an answer feels plausible.")
    elif axis_b == "tamper_social":
        lines.append("You are vulnerable to confident peers, majority opinion, and authority cues, and may adopt socially persuasive conclusions too quickly.")

    if self_rules:
        lines.append("Personal bias tendencies: " + " ".join(self_rules))
    if social_rules:
        lines.append("Social influence tendencies: " + " ".join(social_rules))

    if tag_set:
        lines.append(f"Phenotype traits: {', '.join(sorted(tag_set))}.")

    return " ".join(lines)


def persona_template_version(model_name: str | None) -> str:
    family = model_family(model_name)
    if family == "deepseek":
        return "persona-deepseek-v2"
    if family == "gpt":
        return "persona-gpt-v1"
    return "persona-generic-v1"


