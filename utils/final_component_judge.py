from __future__ import annotations

import json
import re
from typing import Any, Iterable


FINAL_COMPONENT_JUDGE_VERSION = "compact_gold_context_chunked_v7"
FINAL_COMPONENT_SCORING_BASIS = "offline_gold_aware_choice_deterministic_or_llm_component_v7"
FINAL_COMPONENT_CHOICE_ONLY_MODE = "deterministic_choice_only_gold_match_v1"
FINAL_COMPONENT_JUDGE_MAX_TOKENS = 512
FINAL_COMPONENT_QUESTION_MAX_CHARS = 180
FINAL_COMPONENT_RESPONSE_CHUNK_CHARS = 240

FINAL_COMPONENT_SYSTEM_PROMPT = (
    "Classify factual and logical claims as correct, minor error, logical flaw, or fabrication. "
    "A bare final choice such as 'Answer: G' is exactly one claim: count it as correct when its "
    "letter matches the supplied correct-answer letter, otherwise count it as one fabrication. "
    "Do not split answer formatting, labels, or punctuation into extra claims. Return JSON only with "
    "is_refusal and integer counts: total_claims, correct_claims, minor_errors, logical_flaws, "
    "fabrications. Category counts must sum to total_claims."
)

_CHOICE_ONLY_RE = re.compile(r"^\s*(?:final\s+)?answer\s*:\s*([A-J])\s*[.!]?\s*$", re.IGNORECASE)
_GOLD_CHOICE_RE = re.compile(r"^\s*([A-J])(?:\s*:|\s*$)", re.IGNORECASE)

COMPONENT_COUNT_KEYS = (
    "total_claims",
    "correct_claims",
    "minor_errors",
    "logical_flaws",
    "fabrications",
)


def compact_question_for_final_judge(
    text: str,
    max_chars: int = FINAL_COMPONENT_QUESTION_MAX_CHARS,
) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    head_chars = max_chars * 2 // 3
    tail_chars = max_chars - head_chars - 5
    return f"{normalized[:head_chars].rstrip()} ... {normalized[-tail_chars:].lstrip()}"


def split_final_response(
    text: str,
    max_chars: int = FINAL_COMPONENT_RESPONSE_CHUNK_CHARS,
) -> list[str]:
    normalized = str(text or "").strip()
    if len(normalized) <= max_chars:
        return [normalized]
    units = [unit.strip() for unit in re.split(r"(?<=[.!?])\s+|\n+", normalized) if unit.strip()]
    chunks: list[str] = []
    current = ""
    for unit in units:
        for piece in (unit[start : start + max_chars] for start in range(0, len(unit), max_chars)):
            candidate = f"{current} {piece}".strip() if current else piece
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks or [normalized]


def build_gold_reference(gold_answer: Any, choices: Iterable[Any] | None = None) -> str:
    gold = str(gold_answer or "").strip().upper()
    if not gold:
        return "Unavailable"
    choice_list = list(choices or [])
    if len(gold) == 1 and "A" <= gold <= "Z":
        index = ord(gold) - ord("A")
        if 0 <= index < len(choice_list):
            return f"{gold}: {str(choice_list[index]).strip()}"
    return gold


def deterministic_choice_only_payload(
    response_text: str,
    gold_reference: str,
) -> dict[str, Any] | None:
    """Score a strict answer-only response without a redundant LLM judge call."""
    answer_match = _CHOICE_ONLY_RE.fullmatch(str(response_text or ""))
    gold_match = _GOLD_CHOICE_RE.match(str(gold_reference or ""))
    if answer_match is None or gold_match is None:
        return None

    answer_choice = answer_match.group(1).upper()
    gold_choice = gold_match.group(1).upper()
    is_correct = answer_choice == gold_choice
    return {
        "is_refusal": False,
        "total_claims": 1,
        "correct_claims": 1 if is_correct else 0,
        "minor_errors": 0,
        "logical_flaws": 0,
        "fabrications": 0 if is_correct else 1,
        "judge_failed": False,
        "evaluation_mode": FINAL_COMPONENT_CHOICE_ONLY_MODE,
        "answer_choice": answer_choice,
        "gold_choice": gold_choice,
    }


def build_final_component_prompts(
    *,
    question_text: str,
    gold_reference: str,
    answer_chunk: str,
    chunk_index: int,
    chunk_count: int,
    include_question: bool = True,
) -> tuple[str, str, str]:
    compact_question = compact_question_for_final_judge(question_text) if include_question else ""
    if not include_question:
        context_mode = "gold_option_response_only_transport_fallback"
    elif compact_question == " ".join(str(question_text or "").split()):
        context_mode = "full_question"
    else:
        context_mode = f"question_head_tail_{FINAL_COMPONENT_QUESTION_MAX_CHARS}"
    question_line = f"Question: {compact_question}\n" if compact_question else ""
    user_prompt = (
        f"{question_line}Correct answer: {str(gold_reference or 'Unavailable').strip()}\n"
        f"Answer chunk {chunk_index + 1}/{chunk_count} to evaluate: {answer_chunk}"
    )
    return FINAL_COMPONENT_SYSTEM_PROMPT, user_prompt, context_mode


def parse_final_component_payload(raw_content: str) -> dict[str, Any]:
    payload = json.loads(str(raw_content or "").strip())
    if not isinstance(payload, dict):
        raise ValueError("final component judge JSON is not an object")
    missing = [key for key in COMPONENT_COUNT_KEYS if key not in payload]
    if missing:
        raise ValueError(f"final component judge JSON missing keys: {missing}")
    parsed: dict[str, Any] = {"is_refusal": bool(payload.get("is_refusal", False))}
    for key in COMPONENT_COUNT_KEYS:
        parsed[key] = max(0, int(payload[key]))
    category_sum = sum(
        parsed[key] for key in ("correct_claims", "minor_errors", "logical_flaws", "fabrications")
    )
    if parsed["total_claims"] != category_sum:
        raise ValueError(
            f"claim count mismatch total={parsed['total_claims']} categories={category_sum}"
        )
    parsed["judge_failed"] = False
    return parsed


def aggregate_final_component_payloads(payloads: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(payloads)
    if not items:
        raise ValueError("no final component judge payloads")
    parsed = {
        key: sum(int(item.get(key, 0) or 0) for item in items)
        for key in COMPONENT_COUNT_KEYS
    }
    parsed["is_refusal"] = all(bool(item.get("is_refusal", False)) for item in items)
    parsed["judge_failed"] = False
    if parsed["is_refusal"] and parsed["total_claims"] == 0:
        parsed["total_claims"] = 1
        parsed["fabrications"] = 1
    return parsed
