"""Run one of the paper's three five-round component ablations."""

from __future__ import annotations

import argparse
import sys

from . import experiment


ABLATIONS = ("no_self", "no_topology_governance", "no_strategy_evolution")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Other episode options are passed to Evoguard.experiment.",
    )
    parser.add_argument("--ablation", choices=ABLATIONS, required=True)
    args, episode_args = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    fixed_options = (
        "--strategy-mode", "--functional-ablation-mode", "--generations",
        "--two-window-fault-manifest", "--fault-injector-model",
        "--fault-injector-key-env", "--fault-injector-max-tokens",
        "--fault-injector-cache-dir", "--enable-perturbation",
        "--disable-perturbation", "--perturbation-trigger-generation",
        "--persist-perturbation",
    )
    if any(arg == option or arg.startswith(option + "=") for arg in episode_args for option in fixed_options):
        parser.error("The five-round ablation protocol cannot include robustness or other perturbation options.")

    mode_option = "--strategy-mode" if args.ablation == "no_self" else "--functional-ablation-mode"
    experiment.main(["--generations", "5", mode_option, args.ablation, *episode_args])


if __name__ == "__main__":
    main()
