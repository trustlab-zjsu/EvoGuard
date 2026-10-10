"""Prepare clean inputs and manifests for the two-window experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


PROTOCOL_NAME = "two_window_transient_hallucination_v1"
SCHEDULE_ALGORITHM_VERSION = "sha256_uniform_distinct_pair_v1"
TOTAL_ROUNDS = 10
FIRST_FAULT_ROUNDS = (3, 4)
SECOND_FAULT_ROUNDS = (7, 8)
FAULT_K = 1


def get_question_text(row: Mapping[str, Any]) -> str:
    return str(row.get("question_text") or row.get("question") or "").strip()


def get_question_id(row: Mapping[str, Any], sample_idx: int) -> str:
    return str(row.get("question_id") or row.get("instance_id") or sample_idx)


def question_sha256(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(get_question_text(row).encode("utf-8")).hexdigest()


def stable_sample_seed(
    *,
    master_seed: int,
    question_id: str,
    question_hash: str,
) -> int:
    payload = (
        f"{PROTOCOL_NAME}:{SCHEDULE_ALGORITHM_VERSION}:{int(master_seed)}:"
        f"{question_id}:{question_hash}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sample_distinct_sources(*, num_agents: int, sample_seed: int) -> tuple[int, int]:
    """Uniformly sample an ordered pair without rejection or method state."""

    if num_agents < 2:
        raise ValueError("At least two agents are required for distinct A/B sources")
    rng = random.Random(int(sample_seed))
    source_a = rng.randrange(num_agents)
    offset = rng.randrange(num_agents - 1)
    source_b = offset + int(offset >= source_a)
    return source_a, source_b


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class TwoWindowFaultSpec:
    sample_idx: int
    question_id: str
    question_sha256: str
    num_agents: int
    total_rounds: int
    master_seed: int
    sample_schedule_seed: int
    source_a: int
    source_b: int
    first_fault_rounds: tuple[int, ...]
    second_fault_rounds: tuple[int, ...]
    target_wrong_option: str
    target_option_text: str
    canonical_fallback_response: str
    fallback_sha256: str
    protocol: str = PROTOCOL_NAME
    schedule_algorithm_version: str = SCHEDULE_ALGORITHM_VERSION

    def __post_init__(self) -> None:
        if self.protocol != PROTOCOL_NAME:
            raise ValueError(f"Unsupported protocol: {self.protocol}")
        if self.schedule_algorithm_version != SCHEDULE_ALGORITHM_VERSION:
            raise ValueError(
                f"Unsupported schedule algorithm: {self.schedule_algorithm_version}"
            )
        if self.total_rounds != TOTAL_ROUNDS:
            raise ValueError(f"Two-window protocol requires exactly {TOTAL_ROUNDS} generated rounds")
        if self.first_fault_rounds != FIRST_FAULT_ROUNDS:
            raise ValueError(f"First fault window must be {FIRST_FAULT_ROUNDS}")
        if self.second_fault_rounds != SECOND_FAULT_ROUNDS:
            raise ValueError(f"Second fault window must be {SECOND_FAULT_ROUNDS}")
        if self.num_agents < 2:
            raise ValueError("At least two agents are required")
        if not 0 <= self.source_a < self.num_agents:
            raise ValueError("source_a is outside the agent range")
        if not 0 <= self.source_b < self.num_agents:
            raise ValueError("source_b is outside the agent range")
        if self.source_a == self.source_b:
            raise ValueError("source_a and source_b must be distinct")
        target = self.target_wrong_option.strip().upper()
        if len(target) != 1 or not target.isalpha():
            raise ValueError("target_wrong_option must be one option letter")
        fallback = self.canonical_fallback_response
        if not fallback.strip():
            raise ValueError("canonical_fallback_response cannot be empty")
        actual_hash = hashlib.sha256(fallback.encode("utf-8")).hexdigest()
        if self.fallback_sha256 != actual_hash:
            raise ValueError("fallback_sha256 does not match the fallback response")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TwoWindowFaultSpec":
        return cls(
            sample_idx=int(payload["sample_idx"]),
            question_id=str(payload["question_id"]),
            question_sha256=str(payload["question_sha256"]),
            num_agents=int(payload["num_agents"]),
            total_rounds=int(payload["total_rounds"]),
            master_seed=int(payload["master_seed"]),
            sample_schedule_seed=int(payload["sample_schedule_seed"]),
            source_a=int(payload["source_a"]),
            source_b=int(payload["source_b"]),
            first_fault_rounds=tuple(int(value) for value in payload["first_fault_rounds"]),
            second_fault_rounds=tuple(int(value) for value in payload["second_fault_rounds"]),
            target_wrong_option=str(payload["target_wrong_option"]).strip().upper(),
            target_option_text=str(payload.get("target_option_text") or "").strip(),
            canonical_fallback_response=str(payload["canonical_fallback_response"]),
            fallback_sha256=str(payload["fallback_sha256"]),
            protocol=str(payload.get("protocol") or PROTOCOL_NAME),
            schedule_algorithm_version=str(
                payload.get("schedule_algorithm_version") or SCHEDULE_ALGORITHM_VERSION
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def source_for_round_index(self, round_index: int) -> tuple[str, int] | None:
        round_number = int(round_index) + 1
        if round_number in self.first_fault_rounds:
            return "A", self.source_a
        if round_number in self.second_fault_rounds:
            return "B", self.source_b
        return None

    def is_scheduled(self, *, round_index: int, agent_id: int) -> bool:
        scheduled = self.source_for_round_index(round_index)
        return bool(scheduled and scheduled[1] == int(agent_id))

    def phase_for_round_index(self, round_index: int) -> str:
        round_number = int(round_index) + 1
        if round_number <= 2:
            return "clean"
        if round_number in self.first_fault_rounds:
            return "fault_a"
        if round_number == 5:
            return "transition_a"
        if round_number == 6:
            return "recovery_a"
        if round_number in self.second_fault_rounds:
            return "fault_b"
        if round_number == 9:
            return "transition_b"
        if round_number == 10:
            return "recovery_b"
        raise ValueError(f"Round {round_number} is outside the two-window protocol")


def load_manifest(path: str | Path) -> dict[int, TwoWindowFaultSpec]:
    manifest: dict[int, TwoWindowFaultSpec] = {}
    for payload in read_jsonl(path):
        spec = TwoWindowFaultSpec.from_dict(payload)
        if spec.sample_idx in manifest:
            raise ValueError(f"Duplicate sample_idx={spec.sample_idx} in {path}")
        manifest[spec.sample_idx] = spec
    return manifest


PRIVATE_KEY_TOKENS = ("inject", "wrong_answer", "target_agent", "perturb", "fault")


def _required_list(row: Mapping[str, Any], key: str) -> list[Any]:
    value = row.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Missing non-empty {key}")
    return list(value)


def _strip_private_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_private_metadata(item)
            for key, item in value.items()
            if not any(token in str(key).lower() for token in PRIVATE_KEY_TOKENS)
        }
    if isinstance(value, list):
        return [_strip_private_metadata(item) for item in value]
    return value


def clean_external_r0(row: Mapping[str, Any]) -> dict[str, Any]:
    answers = _required_list(row, "round0_answers_before_injection")
    outputs = _required_list(row, "round0_raw_outputs_before_injection")
    if len(answers) != len(outputs):
        raise ValueError("Clean answer/output counts do not match")
    clean = _strip_private_metadata(copy.deepcopy(dict(row)))
    clean["external_clean_r0"] = True
    clean["initial_answers"] = answers
    clean["initial_raw_outputs"] = outputs
    for index, node in enumerate(clean.get("nodes") or []):
        if not isinstance(node, dict) or index >= len(outputs):
            continue
        node["initial_answer"] = answers[index]
        node["round0_raw_output"] = outputs[index]
        node.pop("is_injected", None)
    return clean


def _parse_choice(text: str) -> str | None:
    matches = re.findall(
        r"(?im)^\s*(?:final\s+)?answer\s*:\s*\**\(?\s*([A-J])\s*\)?\**",
        str(text or ""),
    )
    return matches[-1].upper() if matches else None


def build_spec(
    row: Mapping[str, Any],
    *,
    sample_idx: int,
    master_seed: int,
) -> TwoWindowFaultSpec:
    injection = row.get("injection_info")
    if not isinstance(injection, dict):
        raise ValueError("Source row has no injection_info for target/fallback extraction")
    target = str(injection.get("wrong_answer") or "").strip().upper()
    fallback = str(injection.get("poisoned_message") or "").strip()
    gold = str(row.get("gold") or row.get("answer") or "").strip().upper()
    choices = list(row.get("choices") or [])
    letters = [chr(ord("A") + index) for index in range(len(choices))]
    if target not in letters or target == gold:
        raise ValueError("Target must be one valid non-gold option")
    if _parse_choice(fallback) != target:
        raise ValueError("Canonical fallback does not parse to the target option")
    num_agents = int(row.get("num_agents") or len(row.get("nodes") or []))
    question_id = get_question_id(row, sample_idx)
    question_hash = question_sha256(row)
    sample_seed = stable_sample_seed(
        master_seed=master_seed,
        question_id=question_id,
        question_hash=question_hash,
    )
    source_a, source_b = sample_distinct_sources(
        num_agents=num_agents,
        sample_seed=sample_seed,
    )
    return TwoWindowFaultSpec(
        sample_idx=sample_idx,
        question_id=question_id,
        question_sha256=question_hash,
        num_agents=num_agents,
        total_rounds=TOTAL_ROUNDS,
        master_seed=master_seed,
        sample_schedule_seed=sample_seed,
        source_a=source_a,
        source_b=source_b,
        first_fault_rounds=FIRST_FAULT_ROUNDS,
        second_fault_rounds=SECOND_FAULT_ROUNDS,
        target_wrong_option=target,
        target_option_text=str(choices[letters.index(target)]).strip(),
        canonical_fallback_response=fallback,
        fallback_sha256=hashlib.sha256(fallback.encode("utf-8")).hexdigest(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--manifest-jsonl", required=True)
    parser.add_argument("--protocol-meta-json", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destinations = [Path(args.output_jsonl), Path(args.manifest_jsonl), Path(args.protocol_meta_json)]
    existing = [str(path) for path in destinations if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing files: {existing}")
    source_rows = read_jsonl(args.input_jsonl)
    if args.max_samples is not None:
        source_rows = source_rows[: max(0, int(args.max_samples))]
    clean_rows = [clean_external_r0(row) for row in source_rows]
    specs = [
        build_spec(row, sample_idx=index, master_seed=int(args.seed))
        for index, row in enumerate(source_rows)
    ]
    write_jsonl(args.output_jsonl, clean_rows)
    write_jsonl(args.manifest_jsonl, (spec.to_dict() for spec in specs))
    meta = {
        "protocol": PROTOCOL_NAME,
        "schedule_algorithm_version": SCHEDULE_ALGORITHM_VERSION,
        "source_input": str(Path(args.input_jsonl).resolve()),
        "clean_external_r0_input": str(Path(args.output_jsonl).resolve()),
        "manifest": str(Path(args.manifest_jsonl).resolve()),
        "n": len(specs),
        "total_rounds": TOTAL_ROUNDS,
        "first_fault_rounds": list(FIRST_FAULT_ROUNDS),
        "second_fault_rounds": list(SECOND_FAULT_ROUNDS),
        "fault_k": FAULT_K,
        "master_seed": int(args.seed),
        "r5_r9_semantics": "post_injection_transition",
        "r6_r10_semantics": "full_recovery",
        "target_selection": "existing_injected_non_gold_target",
        "online_injector_uses_gold": False,
        "manifest_visible_to_agents": False,
        "external_clean_r0": True,
    }
    meta_path = Path(args.protocol_meta_json)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
