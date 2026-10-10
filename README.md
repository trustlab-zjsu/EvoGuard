# EvoGuard

Source code for EvoGuard experiments on MMLU-Pro. The main experiment and component ablations use five-round episodes. The two-window robustness experiment uses a separate ten-round protocol and clean R0 input. A generation-triggered perturbation is also included as a supplementary experiment. Generated data and results are not included.

`core/` contains the EvoGuard method (agents, prompts, risk observation, and evolution). Each `run_*_experiment.py` file configures one experiment protocol; `experiment.py` provides their shared episode runner. `utils/` holds shared answer and evaluation helpers.

Use Python 3.10+ and run these PowerShell commands from the directory containing `Evoguard`.

## Setup

```powershell
python -m pip install -r Evoguard/requirements.txt
$env:GEN_API_KEY = "<your-generation-key>"
$env:GEN_MODEL = "Qwen/Qwen3-14B"
$env:GEN_BASE_URL = "<your-generation-base-url>"
$env:JUDGE_API_KEY = "<your-judge-key>"
$env:JUDGE_MODEL = "deepseek-v4-pro"
$env:JUDGE_BASE_URL = "<your-judge-base-url>"
```

The model names match the paper's default route. Model names are optional, but both API keys and HTTPS base URLs are required for online runs. The final answer selector remains fixed to `deepseek-v4-pro`, so the judge endpoint must support it.

## Build the MMLU-Pro input

The paper uses 101 questions from each of mathematics, chemistry, law, and philosophy (404 total), five agents, independent initial responses, one injected incorrect response, and six initial topologies distributed as evenly as possible. Generate a dataset with that design using:

```powershell
python -m Evoguard.build_mmlupro_input `
  --prepare-inputs --subject mathematics,chemistry,law,philosophy `
  --split test --per_subject_count 101 --sample_mode random --selection_seed 1 `
  --seed 1 --num_agents 5 --temperature 0 `
  --output Evoguard/data/inputs/mmlupro_input.jsonl
```

The paper does not specify the exact question IDs, split, or sampling seed, so this command is a reproducible construction example rather than the authors' exact dataset.

## Main experiment

```powershell
python -m Evoguard.run_main_five_round_experiment `
  --input-graph Evoguard/data/inputs/mmlupro_input.jsonl `
  --seeds 1,2,3 --output-dir Evoguard/outputs/main
```

The main experiment runs at most five interaction rounds at temperature 0, with early stopping on unanimous valid answers. The gold-aware offline evaluator runs after each episode. Use `--dry-run` to inspect commands and `--start-idx` / `--max-samples` for a subset.

## Component ablations

The ablations use the main experiment's input and five-round horizon. For paired comparisons, use the same input row and episode seed for every variant:

| Variant | Episode option |
| --- | --- |
| Without self-reflection | `--ablation no_self` |
| Without topology governance | `--ablation no_topology_governance` |
| Without strategy evolution | `--ablation no_strategy_evolution` |

For one five-round episode, use `python -m Evoguard.run_component_ablation_experiment --input-graph Evoguard/data/inputs/mmlupro_input.jsonl --line-idx 0 --seed 1 --output-json Evoguard/outputs/ablation/sample_0.json` and add one option from the table. The full-method comparison uses the main experiment entry point.

## Two-window robustness experiment

The robustness experiment uses the full EvoGuard method over ten rounds. Its fixed-source protocol injects faults at R3–R4 and R7–R8 using different source agents. Prepare a clean R0 input and matching manifest:

```powershell
python -m Evoguard.prepare_two_window `
  --input-jsonl Evoguard/data/inputs/mmlupro_input.jsonl `
  --output-jsonl Evoguard/data/inputs/two_window_clean.jsonl `
  --manifest-jsonl Evoguard/data/inputs/two_window_manifest.jsonl `
  --protocol-meta-json Evoguard/data/inputs/two_window_meta.json

python -m Evoguard.run_robustness_ten_round_experiment `
  --input-graph Evoguard/data/inputs/two_window_clean.jsonl `
  --line-idx 0 `
  --two-window-fault-manifest Evoguard/data/inputs/two_window_manifest.jsonl `
  --fault-injector-model $env:GEN_MODEL `
  --output-json Evoguard/outputs/two_window/sample_0.json
```

## Supplementary generation-triggered perturbation

```powershell
python -m Evoguard.experiment `
  --input-graph Evoguard/data/inputs/mmlupro_input.jsonl `
  --line-idx 0 --generations 11 --enable-perturbation --perturbation-trigger-generation 10 `
  --output-json Evoguard/outputs/perturbation/sample_0.json
```

The trigger generation is zero-based, so generation 10 requires at least 11 rounds.

## Results and validation

Each result records both majority-vote (`majority_vote_parsed_answer`, `majority_vote_accuracy`) and gold-blind Deepseek selection (`selector_parsed_answer`, `selector_accuracy`). The offline response evaluator uses the gold answer only after inference.

```powershell
python -B -m unittest Evoguard.test_experiments -v
```

Full experiment validation requires generated input data and model API access. Keep API keys and generated outputs out of the repository.
