"""Run the ten-round, two-window robustness experiment."""

from __future__ import annotations

import argparse
import sys

from . import experiment


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Other episode options are passed to Evoguard.experiment.",
    )
    parser.add_argument("--two-window-fault-manifest", required=True)
    parser.add_argument("--fault-injector-model", required=True)
    parser.parse_known_args(args)
    incompatible = (
        "--functional-ablation-mode", "--strategy-mode", "--generations",
        "--enable-perturbation", "--disable-perturbation",
        "--perturbation-trigger-generation", "--persist-perturbation",
    )
    if any(arg == option or arg.startswith(option + "=") for arg in args for option in incompatible):
        parser.error("The ten-round robustness protocol cannot include ablation or other perturbation options.")

    experiment.main([
        "--generations", "10",
        "--enable-peer-context",
        "--peer-context-max-chars", "0",
        *args,
    ])


if __name__ == "__main__":
    main()
