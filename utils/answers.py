"""Answer parsing and aggregation used by EvoGuard experiments."""


from __future__ import annotations


from typing import Any, Dict, Iterable, List


def _choice_letter(index: int) -> str:
    return chr(ord("A") + index)


def _normalize_valid_letters(valid_letters: Iterable[str] | None = None) -> List[str]:
    if not valid_letters:
        return [chr(ord("A") + idx) for idx in range(10)]
    letters: List[str] = []
    for letter in valid_letters:
        normalized = str(letter or "").strip().upper()
        if len(normalized) == 1 and normalized.isalpha() and normalized not in letters:
            letters.append(normalized)
    return letters or [chr(ord("A") + idx) for idx in range(10)]


def choice_letters_from_choices(choices: Any) -> List[str]:
    if isinstance(choices, list) and choices:
        return [_choice_letter(idx) for idx in range(len(choices))]
    return [chr(ord("A") + idx) for idx in range(10)]


def parse_mmlupro_choice_answer(text, model_name=None, valid_letters: Iterable[str] | None = None):
    """Parse an MMLU-Pro answer using valid A-J final-answer signals."""
    import re

    del model_name

    if text is None:
        return None

    s = str(text).strip()
    if not s:
        return None

    letters = _normalize_valid_letters(valid_letters)
    letters_pat = "".join(re.escape(letter) for letter in letters)

    explicit_patterns = [
        rf"(?:final\s+answer|final\s+option|correct\s+answer|answer)\s*(?:is)?\s*[:：]?\s*[\*\s\(\[]*([{letters_pat}])\b",
        rf"(?:the\s+answer\s+is|i\s+choose)\s*[\*\s\(\[]*([{letters_pat}])\b",
    ]
    for pat in explicit_patterns:
        matches = re.findall(pat, s, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    lines = [line.strip() for line in s.splitlines() if line.strip()]
    if lines:
        last_line = re.sub(r"^[\*\s]+|[\*\s]+$", "", lines[-1])
        m = re.fullmatch(
            rf"[\(\[]?\s*([{letters_pat}])\s*[\)\]]?[\.\:]?",
            last_line,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(1).upper()

        m = re.search(
            rf"(?:option|choice)\s*[\(\[]?\s*([{letters_pat}])\s*[\)\]]?",
            last_line,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(1).upper()

    tail = "\n".join(lines[-3:])
    weak_matches = re.findall(
        rf"(?:option|choice)\s*[\(\[]?\s*([{letters_pat}])\s*[\)\]]?",
        tail,
        flags=re.IGNORECASE,
    )
    if weak_matches:
        return weak_matches[-1].upper()

    return None


def majority_choice(answers: List[str]) -> str | None:
    counts: Dict[str, int] = {}
    for answer in answers:
        counts[answer] = counts.get(answer, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def evaluate_final_agent_answers(
    agent_responses: Dict[str, str],
    gold_answer: str | None,
    model_name: str | None = None,
    valid_letters: Iterable[str] | None = None,
    parser=parse_mmlupro_choice_answer,
) -> Dict[str, Any]:
    def parse(value: Any) -> str | None:
        try:
            return parser(value, model_name=model_name, valid_letters=valid_letters)
        except TypeError:
            return parser(value, model_name=model_name)

    gold = parse(gold_answer)
    per_agent = []
    parsed_answers = []
    correct_count = 0.0
    for agent_id in sorted(agent_responses.keys(), key=lambda value: int(value)):
        response_text = str(agent_responses.get(agent_id) or "")
        parsed = parse(response_text)
        agent_correct = 1.0 if gold and parsed and parsed == gold else 0.0
        if parsed:
            parsed_answers.append(parsed)
        correct_count += agent_correct
        per_agent.append(
            {
                "agent_id": int(agent_id),
                "response": response_text,
                "parsed_answer": parsed,
                "correct": agent_correct,
            }
        )
    majority_answer = majority_choice(parsed_answers)
    parse_fail_rate = 1.0 - (len(parsed_answers) / max(1, len(per_agent)))
    correct = 1.0 if gold and majority_answer and majority_answer == gold else 0.0
    return {
        "gold": gold,
        "agent_answers": parsed_answers,
        "majority_answer": majority_answer,
        "correct": correct,
        "agent_answer_accuracy": correct_count / max(1, len(per_agent)),
        "agent_answer_accuracy_valid_only": correct_count / max(1, len(parsed_answers)),
        "parse_fail_rate": parse_fail_rate,
        "per_agent": per_agent,
    }
