"""Run the five-round main experiment over MMLU-Pro inputs and base seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
SEED_PROTOCOL_VERSION = "base_seed_sample_sha256_v1"
MAIN_ROUNDS = 5


def effective_seed(base_seed: int, sample_idx: int) -> int:
    payload = f"{SEED_PROTOCOL_VERSION}:{base_seed}:{sample_idx}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def parse_seeds(raw: str) -> list[int]:
    try:
        seeds = [int(part.strip()) for part in raw.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seeds must be comma-separated integers") from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("--seeds must contain unique integers")
    return seeds


def input_row_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row at line {number}: {path}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at line {number}: {path}")
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the five-round EvoGuard main experiment on MMLU-Pro samples and seeds.")
    parser.add_argument("--input-graph", type=Path, default=PACKAGE_DIR / "data/inputs/mmlupro_input.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR / "outputs/main")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seeds", type=parse_seeds, default=[1], help="Comma-separated base seeds; each is combined with the sample index.")
    parser.add_argument("--generation-model", default=None)
    parser.add_argument("--prompt-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--final-judge", action=argparse.BooleanOptionalAction, default=True,
        help="Run the gold-aware offline evaluator (default: enabled).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rerun episodes whose output JSON already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print episode commands without calling the model APIs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.start_idx < 0 or (args.max_samples is not None and args.max_samples <= 0):
        parser.error("--start-idx must be nonnegative and --max-samples must be positive")

    input_path = args.input_graph.resolve()
    if not input_path.is_file():
        parser.error(f"Input graph does not exist: {input_path}")
    try:
        row_count = input_row_count(input_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.start_idx >= row_count:
        parser.error(f"--start-idx {args.start_idx} exceeds the {row_count} input rows")

    stop = row_count if args.max_samples is None else min(row_count, args.start_idx + args.max_samples)
    output_dir = args.output_dir.resolve()
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    for sample_idx in range(args.start_idx, stop):
        for base_seed in args.seeds:
            result_path = output_dir / f"sample_{sample_idx}_base_seed_{base_seed}.json"
            if result_path.exists() and not args.overwrite:
                print(f"[skip] {result_path}", flush=True)
                continue
            command = [
                sys.executable, "-m", "Evoguard.experiment",
                "--input-graph", str(input_path),
                "--line-idx", str(sample_idx),
                "--seed", str(effective_seed(base_seed, sample_idx)),
                "--generations", str(MAIN_ROUNDS),
                "--output-json", str(result_path),
            ]
            for flag, value in (
                ("--generation-model", args.generation_model),
                ("--prompt-model", args.prompt_model),
                ("--judge-model", args.judge_model),
            ):
                if value is not None:
                    command.extend((flag, str(value)))
            command.append("--final-judge" if args.final_judge else "--no-final-judge")
            print(subprocess.list2cmdline(command), flush=True)
            if not args.dry_run:
                completed = subprocess.run(command, cwd=PACKAGE_DIR.parent, check=False)
                if completed.returncode:
                    return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
