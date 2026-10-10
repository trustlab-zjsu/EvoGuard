"""Typed configuration for the EvoGuard method and experiment runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Tuple


MEASURED_SEARCH_VERIFY_COST_RATIO = 0.1067
SEARCH_VERIFY_RATIO_SOURCE = "offline_token_measurement"
MUTATION_END_START_RATIO = 0.125


@dataclass(frozen=True)
class FunctionalAblationSpec:
    """Execution gates for one functional-ablation profile."""

    disabled_action_indices: Tuple[int, ...] = ()
    strategy_evolution_enabled: bool = True
    topology_governance_enabled: bool = True
    semantic_transmission_enabled: bool = True
    self_reflection_mode: str = "legacy_role_mutation"
    cumulative_stage: str = "full"


FUNCTIONAL_ABLATION_SPECS = {
    "full": FunctionalAblationSpec(),
    "no_strategy_evolution": FunctionalAblationSpec(
        strategy_evolution_enabled=False,
        cumulative_stage="no_strategy_evolution",
    ),
    "no_topology_governance": FunctionalAblationSpec(
        topology_governance_enabled=False,
        cumulative_stage="no_topology_governance",
    ),
}
FUNCTIONAL_ABLATION_CHOICES = tuple(FUNCTIONAL_ABLATION_SPECS)


def get_functional_ablation_spec(mode: str) -> FunctionalAblationSpec:
    normalized = str(mode or "full")
    try:
        return FUNCTIONAL_ABLATION_SPECS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown functional ablation mode {normalized!r}; "
            f"expected one of {FUNCTIONAL_ABLATION_CHOICES}."
        ) from exc


@dataclass
class PathConfig:
    """Paths rooted at the standalone EvoGuard project directory."""

    base_dir: Path
    input_graph: str = "data/inputs/mmlupro_input.jsonl"
    phenotype_library: str = "phenotype_library.json"
    debug_trace: str = "outputs/debug/debug_trace_gen0_2.md"
    experiment_results: str = "outputs/metrics/experiment_results.json"
    evolution_figure: str = "outputs/figures/tifs_evolution_metrics.png"
    dashboard_figure: str = "outputs/figures/experiment_results_dashboard.png"
    token_figure: str = "outputs/figures/experiment_token_breakdown.png"
    persona_prompt_cache: str = "outputs/cache/persona_prompt_cache.json"
    response_cache: str = "outputs/cache/response_cache.json"
    judge_cache: str = "outputs/cache/judge_cache.json"

    def resolve(self, relative_path: str) -> Path:
        return (self.base_dir / relative_path).resolve()

    @property
    def input_graph_path(self) -> Path:
        return self.resolve(self.input_graph)

    @property
    def phenotype_library_path(self) -> Path:
        return self.resolve(self.phenotype_library)

    @property
    def debug_trace_path(self) -> Path:
        return self.resolve(self.debug_trace)

    @property
    def experiment_results_path(self) -> Path:
        return self.resolve(self.experiment_results)

    @property
    def evolution_figure_path(self) -> Path:
        return self.resolve(self.evolution_figure)

    @property
    def dashboard_figure_path(self) -> Path:
        return self.resolve(self.dashboard_figure)

    @property
    def token_figure_path(self) -> Path:
        return self.resolve(self.token_figure)

    @property
    def persona_prompt_cache_path(self) -> Path:
        return self.resolve(self.persona_prompt_cache)

    @property
    def response_cache_path(self) -> Path:
        return self.resolve(self.response_cache)

    @property
    def judge_cache_path(self) -> Path:
        return self.resolve(self.judge_cache)

    @property
    def output_debug_dir(self) -> Path:
        return self.debug_trace_path.parent

    @property
    def output_metrics_dir(self) -> Path:
        return self.experiment_results_path.parent

    @property
    def output_figures_dir(self) -> Path:
        return self.evolution_figure_path.parent

    @property
    def output_cache_dir(self) -> Path:
        return self.persona_prompt_cache_path.parent


@dataclass
class AgentConfig:
    """Agent-state initialization and phenotype mutation settings."""

    phenotype_library_path: Path | None = None
    mutation_rate: float = 0.1
    crossover_rate: float = 0.5
    initial_action_activation_prob: float = 0.1
    default_a_in: float = 1.0
    default_a_out: float = 1.0
    default_use_reflection: bool = True


@dataclass
class EGTConfig:
    """Topology-strategy co-evolution settings."""

    num_generations: int = 5
    seed: int = 1
    selection_alpha: float = 1.0
    selection_alpha_start: float = 1.0
    selection_alpha_end: float = 2.0
    fitness_penalty_beta: float = 8.0
    beta_mode: str = "adaptive"
    adaptive_beta_quantile: float = 0.85
    adaptive_beta_eps: float = 1e-8
    adaptive_beta_cap: float = 8.0
    risk_estimator_mode: str = "judge_observer"
    functional_ablation_mode: str = "full"
    transmission_claim_risk_threshold: float = 0.5
    tau_pi: float = 0.5
    verify_cost: float = 0.03
    search_cost: float = 0.0032
    observation_interval: int = 999999
    always_observe_final: bool = False
    observation_generations: Tuple[int, ...] = (-1,)
    enable_debug_trace: bool = True
    rep_check_mode: str = "risk"
    gen0_disable_reflection: bool = True
    gen0_force_a_self_zero: bool = True
    exploration_noise: float = 0.0
    persona_mutation_rate: float = 0.15
    mutation_start: float = 0.08
    mutation_end: float = 0.01
    response_refresh_interval: int = 1
    enable_runtime_recalibration: bool = False
    reflection_accept_margin: float = 0.02
    self_v2_pi_margin_same_answer: float = 0.02
    self_v2_pi_margin_changed_answer: float = 0.05
    self_v2_rho_tolerance: float = 0.02
    self_v4_min_pi: float = 0.25
    self_v4_min_claim_risk: float = 0.50
    self_v4_delayed_pi_margin: float = 0.02
    seek_min_receiver_risk: float = 0.05
    seek_delta_margin: float = 0.01
    max_in_degree: int | None = None
    enable_policy_high_risk_reflection: bool = False
    enable_peer_free_self_reflection: bool = False
    enable_peer_context: bool = True
    peer_context_max_chars: int = 1200
    prompt_style: str = "aligned"
    use_gene_role_prompts: bool = False
    enable_invalid_response_fallback: bool = True
    deduplicate_online_risk_prompt_context: bool = True
    reuse_unchanged_claim_observations: bool = True
    online_risk_observer_oracle_free: bool = True
    tau_rho_peer: float = 0.25
    consensus_majority_threshold: float = 0.6
    strategy_set: Tuple[Tuple[int, int, int], ...] = (
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    )
    # Ablations draw from the same full strategy, then project disabled actions.
    strategy_initialization_source_set: Tuple[Tuple[int, int, int], ...] = ()
    disabled_strategy_action_indices: Tuple[int, ...] = ()


def configure_ablation_strategy_space(egt: EGTConfig, strategy_mode: str = "full") -> None:
    """Project one shared latent strategy draw into the requested ablation space."""

    functional_mode = str(getattr(egt, "functional_ablation_mode", "full") or "full")
    functional_spec = get_functional_ablation_spec(functional_mode)
    action_mode = str(strategy_mode or "full")
    action_disabled = {
        "full": (),
        "no_self": (2,),
    }
    if action_mode not in action_disabled:
        raise ValueError(
            f"Unknown strategy mode {action_mode!r}; expected one of {tuple(action_disabled)}."
        )
    if functional_mode != "full" and action_mode != "full":
        raise ValueError("Functional and action-level ablations cannot be combined in one run.")

    full_strategy_set = tuple(tuple(int(bit) for bit in strategy) for strategy in egt.strategy_set)
    if not full_strategy_set:
        raise ValueError("The latent full strategy set cannot be empty.")
    disabled_indices = tuple(
        sorted(set(functional_spec.disabled_action_indices or action_disabled[action_mode]))
    )
    projected_strategy_set = tuple(
        dict.fromkeys(
            tuple(0 if index in disabled_indices else bit for index, bit in enumerate(strategy))
            for strategy in full_strategy_set
        )
    )
    if not projected_strategy_set:
        raise ValueError(
            f"Ablation profile functional={functional_mode}, strategy={action_mode} "
            "produced an empty strategy set."
        )

    egt.strategy_initialization_source_set = full_strategy_set
    egt.disabled_strategy_action_indices = disabled_indices
    egt.strategy_set = projected_strategy_set


@dataclass
class LLMConfig:
    """Generation and judge routing selected by ``Evoguard.llm``."""

    generation_model: str = "Qwen/Qwen3-14B"
    generation_url: str = ""
    generation_key: str | None = None
    judge_model: str = "deepseek-v4-pro"
    judge_url: str = ""
    judge_key: str | None = None
    prompt_model: str = "Qwen/Qwen3-14B"
    prompt_url: str = ""
    prompt_key: str | None = None
    max_concurrency: int = 5
    generation_temperature_default: float = 0.0
    generation_temperature_gen0: float = 0.0
    generation_max_tokens: int = 1024
    judge_temperature: float = 0.0
    judge_max_tokens: int = 2048
    persona_max_tokens: int = 60
    persona_temperature: float = 0.0
    disable_cache: bool = False


@dataclass
class PerturbationConfig:
    """Scheduled input perturbation settings used by injected-data studies."""

    enabled: bool = False
    trigger_generation: int = 10
    persist_across_gens: bool = False


@dataclass
class ExperimentConfig:
    """Root configuration object shared by all EvoGuard entry points."""

    paths: PathConfig
    agent: AgentConfig = field(default_factory=AgentConfig)
    egt: EGTConfig = field(default_factory=EGTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)

    def __post_init__(self) -> None:
        if self.agent.phenotype_library_path is None:
            self.agent.phenotype_library_path = self.paths.phenotype_library_path
        self.sync_derived_costs()

    def sync_derived_costs(self) -> None:
        self.egt.search_cost = float(self.egt.verify_cost) * MEASURED_SEARCH_VERIFY_COST_RATIO
        self.egt.mutation_end = max(0.0, float(self.egt.mutation_start)) * MUTATION_END_START_RATIO

    @staticmethod
    def _progress(current_gen: int, total_gens: int) -> float:
        if total_gens <= 1:
            return 1.0
        bounded_gen = max(0, min(int(current_gen), int(total_gens) - 1))
        return bounded_gen / (int(total_gens) - 1)

    def get_selection_alpha(self, current_gen: int, total_gens: int) -> float:
        start = float(self.egt.selection_alpha_start)
        end = float(self.egt.selection_alpha_end)
        return start + (end - start) * self._progress(current_gen, total_gens)

    def get_beta(self, current_gen: int, total_gens: int) -> float:
        """Compatibility alias for callers predating the alpha rename."""

        return self.get_selection_alpha(current_gen, total_gens)

    def get_mutation_rate(self, current_gen: int, total_gens: int) -> float:
        start = max(0.0, float(self.egt.mutation_start))
        end = max(0.0, float(self.egt.mutation_end))
        return start + (end - start) * self._progress(current_gen, total_gens)


def build_default_config(base_dir: Path | None = None) -> ExperimentConfig:
    project_dir = Path(__file__).resolve().parent
    resolved_base_dir = (base_dir or project_dir).resolve()
    config = ExperimentConfig(paths=PathConfig(base_dir=resolved_base_dir))
    config.llm.generation_model = os.environ.get("GEN_MODEL") or config.llm.generation_model
    config.llm.prompt_model = config.llm.generation_model
    config.llm.judge_model = os.environ.get("JUDGE_MODEL") or config.llm.judge_model

    # Keep routing in one module while preserving the legacy single-case entry point.
    from .llm import preset_for

    generation_preset = preset_for(config.llm.generation_model, role="generation")
    judge_preset = preset_for(config.llm.judge_model, role="judge")
    prompt_preset = preset_for(config.llm.prompt_model, role="generation")
    config.llm.generation_url = str(generation_preset.get("url") or "")
    config.llm.generation_key = generation_preset.get("key")
    config.llm.judge_url = str(judge_preset.get("url") or "")
    config.llm.judge_key = judge_preset.get("key")
    config.llm.prompt_url = str(prompt_preset.get("url") or "")
    config.llm.prompt_key = prompt_preset.get("key")
    return config
