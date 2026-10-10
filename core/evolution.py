import copy
import hashlib
import json
import random
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import requests

from .agents import Agent
from .prompts import (
    FIXED_ROLE_DOMAIN_LIBRARY,
    FIXED_ROLE_HEURISTIC_LIBRARY,
    FIXED_ROLE_SELF_CHECK_LIBRARY,
    FIXED_ROLE_SOCIAL_TENDENCY_LIBRARY,
    FIXED_ROLE_STYLE_LIBRARY,
    PEER_MESSAGE_PROMPT_VERSION,
    ROLE_REWRITE_PROMPT_VERSION,
    _normalize_question_text,
    build_atomic_risk_judge_prompts,
    build_fixed_role_text,
    build_phenotype_role_rewrite_prompts,
    build_interpretable_role_prompt,
    build_model_aware_evoguard_question_prompt,
    build_model_aware_evoguard_system_prompt,
    build_transmission_judge_prompts,
    normalize_graph_edges,
    persona_template_version,
    tags_from_node_metadata,
    tags_from_role_text,
)
from ..config import ExperimentConfig, build_default_config, get_functional_ablation_spec
from .risk_observer import (
    LLMJudgeAtomicRiskObserver,
    RISK_JUDGE_PROMPT_VERSION,
    TRANSMISSION_JUDGE_PROMPT_VERSION,
    RiskState,
    aggregate_risk_from_pi_rho,
    clamp01,
)
from ..llm import build_request_kwargs, model_family
from ..utils.final_component_judge import (
    FINAL_COMPONENT_CHOICE_ONLY_MODE,
    FINAL_COMPONENT_JUDGE_MAX_TOKENS,
    FINAL_COMPONENT_SCORING_BASIS,
    FINAL_COMPONENT_JUDGE_VERSION,
    aggregate_final_component_payloads,
    build_final_component_prompts,
    build_gold_reference,
    deterministic_choice_only_payload,
    parse_final_component_payload,
    split_final_response,
)
from ..utils.answers import (
    choice_letters_from_choices,
    majority_choice,
    parse_mmlupro_choice_answer,
)

LLM_ROLE_CACHE: Dict[str, str] = {}
FIXED_ROLE_DOMAIN_TAG_MAP: Dict[str, str] = {
    "biology": "Science_Expert",
    "medicine": "Medical_Expert",
    "mathematics": "Math_Expert",
    "computer_science": "Code_Expert",
    "physics": "Science_Expert",
    "chemistry": "Science_Expert",
    "law": "Legal_Reasoner",
    "economics": "Data_Analyst",
    "history": "General_Knowledge",
    "philosophy": "Abductive_Reasoner",
    "general": "General_Knowledge",
}


class PerturbationEngine:
    """Resolve scheduled abnormal outputs for injected-data experiments."""

    def __init__(
        self,
        target_agent_id: Optional[int] = None,
        fallback_text: Optional[str] = None,
        preset_outputs: Optional[List[str]] = None,
        persist_across_gens: bool = False,
        trigger_generation: int = 10,
        config: Optional[ExperimentConfig] = None,
    ) -> None:
        self.config = config or build_default_config()
        self.target_agent_id = target_agent_id
        self.fallback_text = (fallback_text or "").strip() or None
        self.preset_outputs = [str(item).strip() for item in (preset_outputs or [])]
        self.persist_across_gens = bool(persist_across_gens)
        self.trigger_generation = int(trigger_generation)

    @classmethod
    def from_graph_dict(
        cls,
        graph_dict: Dict[str, Any],
        config: Optional[ExperimentConfig] = None,
    ) -> "PerturbationEngine":
        resolved_config = config or build_default_config()
        task_cfg = graph_dict.get("task", {}) or {}
        perturb_cfg = task_cfg.get("perturbation", {}) or graph_dict.get("perturbation", {}) or {}

        if not perturb_cfg:
            legacy_key = "".join(["in", "jec", "tion", "_info"])
            legacy_cfg = graph_dict.get(legacy_key, {}) or {}
            perturb_cfg = {
                "target": legacy_cfg.get("target"),
                "poisoned_message": legacy_cfg.get("poisoned_message"),
                "wrong_answer": legacy_cfg.get("wrong_answer"),
                "persist": legacy_cfg.get("persist"),
            }

        target_agent_id = cls._safe_int(perturb_cfg.get("target"))
        fallback_text = cls._pick_fallback_text(
            fallback_text=perturb_cfg.get("poisoned_message") or perturb_cfg.get("wrong_answer"),
            preset_outputs=graph_dict.get("final_raw_outputs", []),
            target_agent_id=target_agent_id,
        )
        persist_setting = perturb_cfg.get("persist")
        if persist_setting is None:
            persist_setting = resolved_config.perturbation.persist_across_gens
        generation_setting = perturb_cfg.get("generation")
        if generation_setting is None:
            generation_setting = resolved_config.perturbation.trigger_generation
        return cls(
            target_agent_id=target_agent_id,
            fallback_text=fallback_text,
            preset_outputs=graph_dict.get("final_raw_outputs", []),
            persist_across_gens=bool(persist_setting),
            trigger_generation=int(generation_setting),
            config=resolved_config,
        )

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _pick_fallback_text(
        fallback_text: Any,
        preset_outputs: List[Any],
        target_agent_id: Optional[int],
    ) -> Optional[str]:
        text = str(fallback_text or "").strip()
        if text:
            return text
        if target_agent_id is None:
            return None
        if 0 <= target_agent_id < len(preset_outputs):
            candidate = str(preset_outputs[target_agent_id]).strip()
            return candidate or None
        return None

    def apply_perturbation(self, agent_id: int, current_gen: int) -> Optional[str]:
        if not self.config.perturbation.enabled:
            return None
        if self.target_agent_id is None or agent_id != self.target_agent_id:
            return None
        if current_gen < self.trigger_generation:
            return None
        if current_gen > self.trigger_generation and not self.persist_across_gens:
            return None
        if 0 <= agent_id < len(self.preset_outputs):
            candidate = self.preset_outputs[agent_id]
            if candidate:
                return candidate
        return self.fallback_text


def _safe_edge_t(edge_transmission_strength: float, default_if_missing: float = 1.0) -> float:
    if edge_transmission_strength is None:
        return float(default_if_missing)
    try:
        return max(0.0, min(1.0, float(edge_transmission_strength)))
    except Exception:
        return float(default_if_missing)


class VerificationEngine:
    """Evaluate risk-based or judge-based reputation edge removal."""

    def __init__(self, config: Optional[ExperimentConfig] = None) -> None:
        self.config = config or build_default_config()

    def should_remove_edge(
        self,
        agent_pi: float,
        agent_rho: Optional[float],
        neighbor_pi: float,
        beta: float,
        question_text: str,
        reference_fact: str,
        neighbor_response: str,
        judge_fn: Optional[
            Callable[[str, str, str], Tuple[Dict[str, Any], str, Dict[str, int]]]
        ] = None,
        edge_transmission_strength: float = 0.0,
    ) -> tuple[bool, Dict[str, Any]]:
        mode = str(self.config.egt.rep_check_mode).lower().strip()
        if mode == "judge" and judge_fn is not None:
            return self._judge_check(question_text, reference_fact, neighbor_response, judge_fn)
        return self._risk_check(
            agent_pi,
            agent_rho,
            neighbor_pi,
            beta,
            edge_transmission_strength,
        )

    @staticmethod
    def _risk_check(
        agent_pi: float,
        agent_rho: Optional[float],
        neighbor_pi: float,
        beta: float,
        edge_transmission_strength: float = 0.0,
    ) -> tuple[bool, Dict[str, Any]]:
        edge_t = _safe_edge_t(edge_transmission_strength, default_if_missing=0.0)
        receiver_exposure = float(agent_rho) if agent_rho is not None else float(agent_pi)
        marginal_benefit = 1.0 - neighbor_pi
        marginal_cost = beta * edge_t * neighbor_pi * (1.0 + receiver_exposure)
        edge_delta = marginal_benefit - marginal_cost
        return edge_delta <= 0.0, {
            "mode": "risk",
            "agent_pi": float(agent_pi),
            "agent_rho": float(receiver_exposure),
            "neighbor_pi": float(neighbor_pi),
            "beta": float(beta),
            "marginal_benefit": marginal_benefit,
            "marginal_cost": marginal_cost,
            "edge_transmission_strength": edge_t,
            "edge_delta": edge_delta,
            "used_t_tiebreak": False,
        }

    @staticmethod
    def _judge_check(
        question_text: str,
        reference_fact: str,
        neighbor_response: str,
        judge_fn: Callable[[str, str, str], Tuple[Dict[str, Any], str, Dict[str, int]]],
    ) -> tuple[bool, Dict[str, Any]]:
        judge_result, raw_text, usage_dict = judge_fn(
            question_text,
            neighbor_response,
            reference_fact,
        )
        total_errors = (
            int(judge_result.get("minor_errors", 0))
            + int(judge_result.get("logical_flaws", 0))
            + int(judge_result.get("fabrications", 0))
        )
        is_bad = bool(judge_result.get("is_refusal", False)) or total_errors > 0
        return is_bad, {
            "mode": "judge",
            "judge_result": judge_result,
            "judge_raw": raw_text,
            "usage": usage_dict,
        }


class EvolutionEnvironment:
    """Main simulator environment for the evolutionary sandbox."""

    DEFAULT_QUESTION = "Please answer the user's question accurately."
    FINAL_COMPONENT_JUDGE_VERSION = FINAL_COMPONENT_JUDGE_VERSION
    FINAL_COMPONENT_SCORING_BASIS = FINAL_COMPONENT_SCORING_BASIS
    FINAL_COMPONENT_CHOICE_ONLY_MODE = FINAL_COMPONENT_CHOICE_ONLY_MODE
    METHOD_PROTOCOL_VERSION = "evoguard_v4_atomic_risk_egt_peer_compact_v5"
    PEER_MESSAGE_PROMPT_VERSION = PEER_MESSAGE_PROMPT_VERSION
    RISK_JUDGE_PROMPT_VERSION = RISK_JUDGE_PROMPT_VERSION
    TRANSMISSION_JUDGE_PROMPT_VERSION = TRANSMISSION_JUDGE_PROMPT_VERSION
    ONLINE_TASK_METADATA_KEYS = {
        "subject",
        "domain",
        "claimed_domain",
        "target_domain",
        "split",
        "dataset",
        "task_type",
        "subdomain",
    }
    FORBIDDEN_ONLINE_METADATA_KEYS = {
        "gold",
        "gold_answer",
        "ground_truth",
        "ground_truth_answer",
        "correct_answer",
        "answer_key",
        "label",
        "target",
        "reference_answer",
        "reference_fact",
        "offline_reference_fact",
        "injection",
        "injection_info",
        "injection_metadata",
        "injected_answer",
        "injected_label",
        "wrong_answer",
        "wrong_label",
    }
    DEFAULT_ONLINE_REFERENCE_CONTEXT = (
        "Use only the question text, answer options, the current agent response, and task metadata. "
        "No cross-agent content, other-agent scores, or judge outputs are provided. "
        "Do not assume any hidden supervision signal or external solution sheet."
    )

    def __init__(
        self,
        graph_dict: Dict[str, Any],
        num_generations: int | None = None,
        config: ExperimentConfig | None = None,
        fault_controller: Any | None = None,
    ):
        self.config = config or build_default_config()
        self.graph_dict = copy.deepcopy(graph_dict)
        self.fault_controller = fault_controller
        normalize_graph_edges(self.graph_dict)
        requested_generations = self.config.egt.num_generations if num_generations is None else num_generations
        self.num_generations = max(1, int(requested_generations))
        self.current_task = self.graph_dict.get("task", {})
        self.token_usage = {
            "generation_prompt": 0,
            "generation_completion": 0,
            "rep_judge_prompt": 0,
            "rep_judge_completion": 0,
            "judge_prompt": 0,
            "judge_completion": 0,
            "online_risk_judge_prompt": 0,
            "online_risk_judge_completion": 0,
            "final_judge_prompt": 0,
            "final_judge_completion": 0,
            "final_selector_prompt": 0,
            "final_selector_completion": 0,
        }
        self.disable_cache = bool(getattr(getattr(self.config, "llm", None), "disable_cache", False))
        self.api_request_stats = {
            "logical_call_count": 0,
            "request_attempt_count": 0,
            "successful_call_count": 0,
            "failed_attempt_count": 0,
            "retry_count": 0,
            "failure_count": 0,
            "request_seconds_sum": 0.0,
            "retry_backoff_seconds_sum": 0.0,
            "call_locations": {},
            "attempt_locations": {},
            "success_locations": {},
            "failed_attempt_locations": {},
            "retry_locations": {},
            "failure_locations": {},
            "request_seconds_by_location": {},
            "retry_backoff_seconds_by_location": {},
        }
        self._api_stats_lock = threading.Lock()
        self._api_request_intervals: list[tuple[float, float]] = []
        self._api_backoff_intervals: list[tuple[float, float]] = []
        self.risk_judge_audit_totals = self._empty_risk_usage()
        self.final_judge_audit_totals = {
            "final_judge_prompt_tokens": 0,
            "final_judge_completion_tokens": 0,
            "final_judge_call_count": 0,
            "final_judge_capped_count": 0,
            "final_judge_invalid_count": 0,
            "final_judge_schema_retry_count": 0,
        }
        self.rep_judge_cache: Dict[str, tuple[Dict[str, Any], str, Dict[str, int]]] = {}
        self.rep_judge_cache_hits = 0
        self.persona_prompt_cache = {} if self.disable_cache else self._load_persona_prompt_cache()
        self.response_cache = {} if self.disable_cache else self._load_json_cache(self.config.paths.response_cache_path)
        self.judge_cache = {} if self.disable_cache else self._load_json_cache(self.config.paths.judge_cache_path)
        self.debug_file = None
        if self.config.egt.enable_debug_trace:
            self.config.paths.output_debug_dir.mkdir(parents=True, exist_ok=True)
            self.debug_file = open(self.config.paths.debug_trace_path, "w", encoding="utf-8")
            self.debug_file.write("# Evolution Simulator Debug Trace\n\n")
            self.debug_file.flush()
        self.perturbation_engine = PerturbationEngine.from_graph_dict(self.graph_dict, config=self.config)
        self.verification_engine = VerificationEngine(config=self.config)
        self.latest_prediction_details: Dict[str, Any] = {}
        self.latest_risk_state: RiskState | None = None
        self.initial_risk_state: RiskState | None = None
        self.latest_transmission_matrix: List[List[float]] = []
        self.latest_edge_transmissions: List[Dict[str, Any]] = []
        configured_beta = getattr(self.config.egt, "fitness_penalty_beta", 1.0)
        self.latest_edge_beta_used: float = float(1.0 if configured_beta is None else configured_beta)
        self.latest_edge_beta_raw: float = self.latest_edge_beta_used
        self.latest_edge_beta_capped: bool = False
        self.latest_edge_beta_fallback_applied: bool = False
        self.latest_edge_risk_ratio_q85: float = 0.0
        self.latest_avg_edge_risk_ratio: float = 0.0
        self.latest_max_edge_risk_ratio: float = 0.0
        self.latest_agent_responses: Dict[int, str] = {}
        self.latest_answer_choice_analysis: Dict[str, Any] = {}
        self.latest_consensus_analysis: Dict[str, Any] = {}
        self.latest_online_prompt_samples: List[Dict[str, Any]] = []
        self.latest_generation_input_snapshot: Dict[int, str] = {}
        self.latest_response_update_events: List[Dict[str, Any]] = []
        self.latest_observed_output_texts: Dict[int, str] = {}
        self.latest_claims_by_agent: Dict[int, List[Dict[str, Any]]] = {}
        self.latest_pi_by_agent: Dict[int, float] = {}
        self.pending_self_role_updates: Dict[int, Dict[str, Any]] = {}
        self.response_generation_round_count = 0
        self.risk_observer_invocation_count = 0

        self.judge_client = None
        self.response_client = None
        self.persona_client = None
        self._init_openai_clients()
        self.judge_risk_observer = LLMJudgeAtomicRiskObserver(
            extract_claims_fn=self.judge_atomic_claims,
            judge_inheritance_fn=self.judge_claim_inheritance,
            high_risk_claim_threshold=float(
                getattr(self.config.egt, "transmission_claim_risk_threshold", 0.5) or 0.5
            ),
            transmission_mode=(
                "semantic_judge"
                if get_functional_ablation_spec(
                    getattr(self.config.egt, "functional_ablation_mode", "full")
                ).semantic_transmission_enabled
                else "structural_edge"
            ),
        )

        num_nodes = self._infer_num_nodes(self.graph_dict)
        self.agents: Dict[int, Agent] = self._initialize_agents(num_nodes)
        self._initialize_role_prompts_from_genes()
        self._initialize_responses_from_final_raw_outputs()
        self._observe_initial_intrinsic_risk()

    @staticmethod
    def _infer_num_nodes(graph_dict: Dict[str, Any]) -> int:
        if "nodes" in graph_dict:
            return len(graph_dict["nodes"])

        if "E" in graph_dict:
            adjacency = graph_dict["E"]
            if hasattr(adjacency, "__len__") and adjacency and hasattr(adjacency[0], "__len__"):
                if len(adjacency) == len(adjacency[0]):
                    return len(adjacency)

        if "edges" in graph_dict:
            max_id = -1
            for edge in graph_dict["edges"]:
                if len(edge) >= 2:
                    max_id = max(max_id, int(edge[0]), int(edge[1]))
            if max_id >= 0:
                return max_id + 1

        return 0

    def _valid_answer_letters(self) -> list[str]:
        return choice_letters_from_choices(self.graph_dict.get("choices"))

    def _parse_choice_answer(self, text: Any) -> str | None:
        return parse_mmlupro_choice_answer(
            text,
            model_name=self._gen_model(),
            valid_letters=self._valid_answer_letters(),
        )

    @staticmethod
    def _normalize_fixed_role_domain(node: dict[str, Any]) -> str:
        domain = str(node.get("domain") or node.get("role_domain") or "").strip().lower()
        return domain if domain in FIXED_ROLE_DOMAIN_LIBRARY else "general"

    def _extract_fixed_role_schema(self, node: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(node, dict):
            return None
        style = str(node.get("style") or "").strip()
        self_check = str(node.get("self_check") or "").strip()
        social_tendency = str(node.get("social_tendency") or "").strip()
        heuristic_text = str(node.get("heuristic_text") or "").strip()
        domain = self._normalize_fixed_role_domain(node)
        if (
            style not in FIXED_ROLE_STYLE_LIBRARY
            or self_check not in FIXED_ROLE_SELF_CHECK_LIBRARY
            or social_tendency not in FIXED_ROLE_SOCIAL_TENDENCY_LIBRARY
            or not heuristic_text
        ):
            return None
        heuristic_type = str(node.get("heuristic_type") or "").strip().lower()
        if heuristic_type not in ("stable", "risky"):
            heuristic_type = "risky" if heuristic_text in FIXED_ROLE_HEURISTIC_LIBRARY["risky"] else "stable"
        return {
            "role_id": str(node.get("role_id") or "").strip(),
            "domain": domain,
            "style": style,
            "self_check": self_check,
            "social_tendency": social_tendency,
            "heuristic_type": heuristic_type,
            "heuristic_text": heuristic_text,
            "role_revision": int(node.get("role_revision", 0) or 0),
        }

    def _schema_to_role_text(self, schema: dict[str, Any]) -> str:
        return build_fixed_role_text(
            str(schema["domain"]),
            str(schema["style"]),
            str(schema["self_check"]),
            str(schema["social_tendency"]),
            str(schema["heuristic_text"]),
        )

    def _schema_to_tags(self, schema: dict[str, Any]) -> set[str]:
        tags = {"Step-by-step"}
        tags.add(FIXED_ROLE_DOMAIN_TAG_MAP.get(str(schema["domain"]), "General_Knowledge"))
        tags.add("Deliberate_Slow" if str(schema["style"]) == "matched" else "Intuitive_Fast")
        tags.add("Fact_checker" if str(schema["self_check"]) == "strict" else "Overconfident")
        tags.add("Skeptic" if str(schema["social_tendency"]) == "independent" else "Gullible")
        tags.add("Rule_Based" if str(schema["heuristic_type"]) == "stable" else "Abductive_Reasoner")
        return tags

    def _apply_fixed_role_schema(self, node: dict[str, Any], schema: dict[str, Any], *, increment_revision: bool) -> str:
        node["role_id"] = str(schema.get("role_id") or node.get("role_id") or "")
        node["domain"] = str(schema["domain"])
        node.pop("role_domain", None)
        node["style"] = str(schema["style"])
        node["self_check"] = str(schema["self_check"])
        node["social_tendency"] = str(schema["social_tendency"])
        node["heuristic_type"] = str(schema["heuristic_type"])
        node["heuristic_text"] = str(schema["heuristic_text"])
        current_revision = int(node.get("role_revision", schema.get("role_revision", 0)) or 0)
        node["role_revision"] = current_revision + 1 if increment_revision else int(schema.get("role_revision", current_revision) or 0)
        node["role_meta"] = self._normalize_role_meta(
            {
                "role_id": node["role_id"],
                "domain": node["domain"],
                "style": node["style"],
                "self_check": node["self_check"],
                "social_tendency": node["social_tendency"],
                "heuristic_type": node["heuristic_type"],
                "heuristic_text": node["heuristic_text"],
                "role_revision": int(node["role_revision"]),
            }
        )
        role_text = self._schema_to_role_text(schema)
        node["role_text"] = role_text
        return role_text

    def _controlled_reflect_fixed_role_schema(
        self,
        *,
        node_id: int,
        current_gen: int,
        schema: dict[str, Any],
        is_high_risk_node: bool,
    ) -> tuple[dict[str, Any], str, set[str], bool]:
        seed_text = json.dumps(
            {
                "seed": int(getattr(self.config.egt, "seed", 0) or 0),
                "node_id": int(node_id),
                "current_gen": int(current_gen),
                "schema": schema,
                "high_risk": bool(is_high_risk_node),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        seed_value = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed_value)
        candidate = dict(schema)
        changed = False

        if is_high_risk_node:
            if candidate["style"] != "matched":
                candidate["style"] = "matched"
                changed = True
            if candidate["self_check"] != "strict":
                candidate["self_check"] = "strict"
                changed = True
            if candidate["social_tendency"] != "independent":
                candidate["social_tendency"] = "independent"
                changed = True
            if candidate["heuristic_type"] != "stable":
                candidate["heuristic_type"] = "stable"
                candidate["heuristic_text"] = rng.choice(FIXED_ROLE_HEURISTIC_LIBRARY["stable"])
                changed = True
        else:
            options: list[dict[str, Any]] = []
            if candidate["style"] != "matched":
                proposal = dict(candidate)
                proposal["style"] = "matched"
                options.append(proposal)
            if candidate["self_check"] != "strict":
                proposal = dict(candidate)
                proposal["self_check"] = "strict"
                options.append(proposal)
            if candidate["social_tendency"] != "independent":
                proposal = dict(candidate)
                proposal["social_tendency"] = "independent"
                options.append(proposal)
            if candidate["heuristic_type"] != "stable":
                proposal = dict(candidate)
                proposal["heuristic_type"] = "stable"
                proposal["heuristic_text"] = rng.choice(FIXED_ROLE_HEURISTIC_LIBRARY["stable"])
                options.append(proposal)
            same_bucket = list(FIXED_ROLE_HEURISTIC_LIBRARY.get(candidate["heuristic_type"], []))
            alt_heuristics = [text for text in same_bucket if text != candidate["heuristic_text"]]
            if alt_heuristics:
                proposal = dict(candidate)
                proposal["heuristic_text"] = rng.choice(alt_heuristics)
                options.append(proposal)
            if options:
                candidate = rng.choice(options)
                changed = candidate != schema

        if not changed:
            return schema, self._schema_to_role_text(schema), self._schema_to_tags(schema), False
        return candidate, self._schema_to_role_text(candidate), self._schema_to_tags(candidate), True

    def _initial_tags_for_node(self, node_id: int) -> set[str] | None:
        nodes = self.graph_dict.get("nodes", [])
        if not isinstance(nodes, list) or node_id >= len(nodes) or not isinstance(nodes[node_id], dict):
            return None
        role_text = str(
            nodes[node_id].get("role_text")
            or nodes[node_id].get("raw_role_text")
            or nodes[node_id].get("original_role_text")
            or ""
        ).strip()
        if role_text:
            tags = tags_from_role_text(role_text)
            nodes[node_id]["initial_phenotype_tags"] = sorted(tags)
            nodes[node_id]["initial_phenotype_tags_source"] = "node.role_text"
            return tags
        fixed_schema = self._extract_fixed_role_schema(nodes[node_id])
        if fixed_schema is not None:
            tags = self._schema_to_tags(fixed_schema)
            nodes[node_id]["initial_phenotype_tags"] = sorted(tags)
            nodes[node_id]["initial_phenotype_tags_source"] = "fixed_role_schema_fallback"
            return tags
        if not bool(getattr(self.config.egt, "use_gene_role_prompts", True)):
            return None
        tags = tags_from_node_metadata(nodes[node_id])
        nodes[node_id]["initial_phenotype_tags"] = sorted(tags)
        nodes[node_id]["initial_phenotype_tags_source"] = "node_metadata_fallback"
        return tags

    def _initialize_agents(self, num_nodes: int) -> Dict[int, Agent]:
        agents: Dict[int, Agent] = {}
        for node_id in range(num_nodes):
            agents[node_id] = Agent(
                node_id=node_id,
                initial_tags=self._initial_tags_for_node(node_id),
                config=self.config,
            )
        return agents

    def _initialize_role_prompts_from_genes(self) -> None:
        nodes = self.graph_dict.get("nodes", [])
        if not isinstance(nodes, list):
            return
        for node_id, agent in self.agents.items():
            if node_id >= len(nodes) or not isinstance(nodes[node_id], dict):
                continue
            node = nodes[node_id]
            existing_role = str(
                node.get("role_text")
                or node.get("raw_role_text")
                or node.get("original_role_text")
                or ""
            ).strip()
            if existing_role:
                node["role_text"] = existing_role
                node.setdefault("raw_role_text", existing_role)
                node.setdefault("original_role_text", str(node["raw_role_text"]))

            fixed_schema = self._extract_fixed_role_schema(node)
            if fixed_schema is not None:
                agent.tags = (
                    tags_from_role_text(existing_role)
                    if existing_role
                    else self._schema_to_tags(fixed_schema)
                )
                if not existing_role:
                    role_text = self._schema_to_role_text(fixed_schema)
                    node["role_text"] = role_text
                    node["raw_role_text"] = role_text
                    node["original_role_text"] = role_text
                continue
            if not bool(getattr(self.config.egt, "use_gene_role_prompts", True)):
                continue
            if existing_role:
                continue
            role_text = self.generate_role_text_with_llm(agent.tags, node=node)
            node["role_text"] = role_text
            node["raw_role_text"] = role_text
            node["original_role_text"] = role_text

    def _init_openai_clients(self) -> None:
        for purpose, key, url, prefix in (
            ("generation", self.config.llm.generation_key, self.config.llm.generation_url, "GEN"),
            ("judge", self.config.llm.judge_key, self.config.llm.judge_url, "JUDGE"),
            ("prompt", self.config.llm.prompt_key, self.config.llm.prompt_url, "GEN"),
        ):
            if not key:
                raise RuntimeError(f"Missing {purpose} API key; set {prefix}_API_KEY.")
            if not str(url or "").startswith("https://"):
                raise RuntimeError(f"Missing {purpose} HTTPS URL; set {prefix}_BASE_URL.")
        try:
            import openai

            self.judge_client = openai.OpenAI(
                base_url=self.config.llm.judge_url,
                api_key=self.config.llm.judge_key,
            )
            self.response_client = openai.OpenAI(
                base_url=self.config.llm.generation_url,
                api_key=self.config.llm.generation_key,
            )
            self.persona_client = openai.OpenAI(
                base_url=self.config.llm.prompt_url,
                api_key=self.config.llm.prompt_key,
            )
        except Exception as exc:
            self.judge_client = None
            self.response_client = None
            self.persona_client = None
            raise RuntimeError(
                "failed to initialize EvoGuard LLM clients; refusing to run with silent mock clients"
            ) from exc

    def _load_json_cache(self, cache_path) -> Dict[str, Any]:
        try:
            if cache_path.exists():
                with open(cache_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    return {str(k): v for k, v in payload.items()}
        except Exception:
            pass
        return {}

    def _load_persona_prompt_cache(self) -> Dict[str, str]:
        payload = self._load_json_cache(self.config.paths.persona_prompt_cache_path)
        return {str(k): str(v) for k, v in payload.items()}

    def _save_json_cache(self, cache_path, payload: Dict[str, Any]) -> None:
        if self.disable_cache:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _save_persona_prompt_cache(self) -> None:
        if self.disable_cache:
            return
        self._save_json_cache(self.config.paths.persona_prompt_cache_path, self.persona_prompt_cache)

    def _persona_cache_key(self, tags: set) -> str:
        sorted_tags = " || ".join(sorted(tags))
        model_name = self.config.llm.prompt_model
        template_version = persona_template_version(model_name)
        return f"{model_name}::{template_version}::{sorted_tags}"

    @staticmethod
    def _stable_cache_key(payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _response_cache_key(self, persona_prompt: str, question_text: str, temperature: float) -> str:
        return self._stable_cache_key(
            {
                "kind": "response",
                "model": self._gen_model(),
                "temperature": float(temperature),
                "max_tokens": int(self.config.llm.generation_max_tokens),
                "persona_prompt": str(persona_prompt),
                "question_text": str(question_text),
            }
        )

    def _judge_cache_key(self, kind: str, system_prompt: str, user_prompt: str) -> str:
        return self._stable_cache_key(
            {
                "kind": kind,
                "model": self._judge_model(),
                "temperature": float(self.config.llm.judge_temperature),
                "max_tokens": int(self.config.llm.judge_max_tokens),
                "system_prompt": str(system_prompt),
                "user_prompt": str(user_prompt),
            }
        )

    def _gen_model(self) -> str:
        return self.config.llm.generation_model

    def _judge_model(self) -> str:
        return self.config.llm.judge_model

    def _prompt_model(self) -> str:
        return self.config.llm.prompt_model

    def _bump_named_counter_unlocked(self, bucket: str, location: str) -> None:
        stats_bucket = self.api_request_stats.setdefault(bucket, {})
        stats_bucket[location] = int(stats_bucket.get(location, 0)) + 1

    @staticmethod
    def _interval_union_seconds(intervals: list[tuple[float, float]]) -> float:
        valid = sorted((float(start), float(end)) for start, end in intervals if end >= start)
        if not valid:
            return 0.0
        total = 0.0
        current_start, current_end = valid[0]
        for start, end in valid[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                total += current_end - current_start
                current_start, current_end = start, end
        return total + current_end - current_start

    def _record_api_attempt(
        self,
        location: str,
        started_at: float,
        ended_at: float,
        *,
        succeeded: bool,
    ) -> None:
        elapsed = max(0.0, float(ended_at) - float(started_at))
        with self._api_stats_lock:
            self.api_request_stats["request_attempt_count"] += 1
            self.api_request_stats["request_seconds_sum"] += elapsed
            self._bump_named_counter_unlocked("attempt_locations", location)
            seconds_by_location = self.api_request_stats["request_seconds_by_location"]
            seconds_by_location[location] = float(seconds_by_location.get(location, 0.0)) + elapsed
            self._api_request_intervals.append((float(started_at), float(ended_at)))
            if not succeeded:
                self.api_request_stats["failed_attempt_count"] += 1
                self._bump_named_counter_unlocked("failed_attempt_locations", location)

    def _record_api_backoff(self, location: str, started_at: float, ended_at: float) -> None:
        elapsed = max(0.0, float(ended_at) - float(started_at))
        with self._api_stats_lock:
            self.api_request_stats["retry_backoff_seconds_sum"] += elapsed
            seconds_by_location = self.api_request_stats["retry_backoff_seconds_by_location"]
            seconds_by_location[location] = float(seconds_by_location.get(location, 0.0)) + elapsed
            self._api_backoff_intervals.append((float(started_at), float(ended_at)))

    def get_api_request_stats(self) -> Dict[str, Any]:
        """Return thread-safe API timing with concurrency-aware wall-time unions."""

        with self._api_stats_lock:
            stats = copy.deepcopy(self.api_request_stats)
            request_intervals = list(self._api_request_intervals)
            backoff_intervals = list(self._api_backoff_intervals)
        stats["request_active_wall_seconds"] = self._interval_union_seconds(request_intervals)
        stats["retry_backoff_active_wall_seconds"] = self._interval_union_seconds(backoff_intervals)
        stats["api_wait_active_wall_seconds"] = self._interval_union_seconds(
            request_intervals + backoff_intervals
        )
        stats["timing_semantics"] = {
            "request_seconds_sum": "sum of all API attempt durations; overlapping calls are counted separately",
            "request_active_wall_seconds": "wall-time union of concurrent API request intervals",
            "api_wait_active_wall_seconds": "wall-time union of API request and retry-backoff intervals",
        }
        return stats

    def _call_llm_with_retry(self, location: str, fn, max_attempts: int = 5):
        with self._api_stats_lock:
            self.api_request_stats["logical_call_count"] += 1
            self._bump_named_counter_unlocked("call_locations", location)
        last_exc = None
        for attempt in range(1, max_attempts + 1):
            attempt_started_at = time.monotonic()
            try:
                result = fn()
            except Exception as exc:
                self._record_api_attempt(
                    location,
                    attempt_started_at,
                    time.monotonic(),
                    succeeded=False,
                )
                last_exc = exc
                if attempt < max_attempts:
                    with self._api_stats_lock:
                        self.api_request_stats["retry_count"] += 1
                        self._bump_named_counter_unlocked("retry_locations", location)
                    print(
                        f"[LLM Retry] {location} attempt {attempt}/{max_attempts - 1} failed: "
                        f"{exc.__class__.__name__}: {exc}"
                    )
                    backoff_started_at = time.monotonic()
                    time.sleep(min(2.0 ** (attempt - 1), 4.0))
                    self._record_api_backoff(location, backoff_started_at, time.monotonic())
                    continue

                with self._api_stats_lock:
                    self.api_request_stats["failure_count"] += 1
                    self._bump_named_counter_unlocked("failure_locations", location)
                print(
                    f"[LLM Failed] {location} failed after {max_attempts} attempts: "
                    f"{exc.__class__.__name__}: {exc}"
                )
                raise
            else:
                self._record_api_attempt(
                    location,
                    attempt_started_at,
                    time.monotonic(),
                    succeeded=True,
                )
                with self._api_stats_lock:
                    self.api_request_stats["successful_call_count"] += 1
                    self._bump_named_counter_unlocked("success_locations", location)
                return result

        raise last_exc if last_exc is not None else RuntimeError(f"Unknown LLM failure at {location}")

    @staticmethod
    def _normalize_base_url_for_deepseek(base_url: str) -> str:
        return str(base_url or "").rstrip("/")

    def _deepseek_chat_completion(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        response_format: dict[str, Any] | None = None,
        timeout_seconds: int = 60,
    ) -> dict[str, Any]:
        if not api_key:
            raise RuntimeError("Missing API key for Deepseek request")
        url = f"{self._normalize_base_url_for_deepseek(base_url)}/chat/completions"
        payload: dict[str, Any] = {
            "messages": messages,
            "model": model_name,
            "thinking": {"type": "disabled"},
            "max_tokens": int(max_tokens),
            "stream": False,
            "temperature": float(temperature),
            "top_p": 1,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
        response.raise_for_status()
        return response.json()

    def _chat_completion(
        self,
        *,
        role: str,
        model_name: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        response_format: dict[str, Any] | None = None,
        timeout_seconds: int = 60,
    ):
        family = model_family(model_name)
        if family == "deepseek":
            if role == "generation":
                base_url = self.config.llm.generation_url
                api_key = str(self.config.llm.generation_key or "")
            elif role == "prompt":
                base_url = self.config.llm.prompt_url
                api_key = str(self.config.llm.prompt_key or "")
            else:
                base_url = self.config.llm.judge_url
                api_key = str(self.config.llm.judge_key or "")
            return self._deepseek_chat_completion(
                base_url=str(base_url or ""),
                api_key=api_key,
                model_name=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
                timeout_seconds=timeout_seconds,
            )

        if role == "generation":
            client = self.response_client
        elif role == "prompt":
            client = self.persona_client
        else:
            client = self.judge_client
        if not client:
            raise RuntimeError(f"{role} client unavailable")
        request_kwargs = build_request_kwargs(
            model_name=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            request_timeout=timeout_seconds,
            response_format=response_format,
        )
        return client.chat.completions.create(**request_kwargs)

    def get_strategy_ratios(self) -> Dict[str, float]:
        strategy_set = tuple(self.config.egt.strategy_set or ())
        if not strategy_set:
            return {}
        total = max(1, len(self.agents))
        counts = {idx: 0 for idx in range(len(strategy_set))}
        for agent in self.agents.values():
            sid = int(getattr(agent, "strategy_id", 0))
            sid = sid % len(strategy_set)
            counts[sid] += 1
        ratios: Dict[str, float] = {}
        for idx in range(len(strategy_set)):
            ratios[f"strategy_{idx}_ratio"] = counts[idx] / total
        return ratios

    @staticmethod
    def _quantile(values: List[float], q: float) -> float:
        if not values:
            return 0.0
        xs = sorted(float(v) for v in values)
        if len(xs) == 1:
            return xs[0]
        q = max(0.0, min(1.0, float(q)))
        pos = q * (len(xs) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(xs) - 1)
        w = pos - lo
        return xs[lo] * (1.0 - w) + xs[hi] * w

    @staticmethod
    def _safe_edge_t_from_value(value: Any, default_if_missing: float = 1.0) -> float:
        if value is None:
            return float(default_if_missing)
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return float(default_if_missing)

    def _safe_edge_t(self, matrix: List[List[float]] | None, src: int, dst: int, default_if_missing: float = 1.0) -> float:
        try:
            value = matrix[dst][src]
        except Exception:
            return float(default_if_missing)
        return self._safe_edge_t_from_value(value, default_if_missing=default_if_missing)

    @staticmethod
    def _default_judge_result(is_refusal: bool = False) -> Dict[str, Any]:
        return {
            "is_refusal": is_refusal,
            "total_claims": 0,
            "correct_claims": 0,
            "minor_errors": 0,
            "logical_flaws": 0,
            "fabrications": 1 if is_refusal else 0,
        }

    def _parse_judge_response(self, raw_content: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(raw_content)
        except Exception:
            return self._default_judge_result(is_refusal=True)

        if isinstance(parsed, dict) and "reasoning_scratchpad" in parsed:
            parsed = {k: v for k, v in parsed.items() if k != "reasoning_scratchpad"}

        result = self._default_judge_result()
        result["is_refusal"] = bool(parsed.get("is_refusal", False))
        for key in ("total_claims", "correct_claims", "minor_errors", "logical_flaws", "fabrications"):
            try:
                result[key] = max(0, int(parsed.get(key, 0)))
            except Exception:
                result[key] = 0
        return result

    @staticmethod
    def _compute_agent_metrics(judge_result: Dict[str, Any]) -> Dict[str, float]:
        is_refusal = bool(judge_result.get("is_refusal", False))
        raw_total_claims = max(0, int(judge_result.get("total_claims", 0)))
        total_claims = max(raw_total_claims, 1)
        correct_claims = max(0, int(judge_result.get("correct_claims", 0)))
        minor_errors = max(0, int(judge_result.get("minor_errors", 0)))
        logical_flaws = max(0, int(judge_result.get("logical_flaws", 0)))
        fabrications = max(0, int(judge_result.get("fabrications", 0)))

        if is_refusal and raw_total_claims == 0:
            atomic_rate = 1.0
            swhi = 1.0
            info_snr = 0.0
        else:
            atomic_rate = (minor_errors + logical_flaws + fabrications) / total_claims
            swhi = (0.2 * minor_errors + 0.5 * logical_flaws + 1.0 * fabrications) / total_claims
            info_snr = 0.0 if is_refusal else (correct_claims / total_claims)

        return {
            "atomic_rate": max(0.0, min(1.0, atomic_rate)),
            "swhi": max(0.0, swhi),
            "info_snr": max(0.0, min(1.0, info_snr)),
        }

    def _accumulate_usage(self, usage: Any, prompt_key: str, completion_key: str) -> None:
        if usage is None:
            return

        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0) or 0
            completion_tokens = usage.get("completion_tokens", 0) or 0
        else:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        self.token_usage[prompt_key] += int(prompt_tokens)
        self.token_usage[completion_key] += int(completion_tokens)

    @staticmethod
    def _extract_usage(payload: Any) -> Any:
        if payload is None:
            return None
        if isinstance(payload, dict):
            usage = payload.get("usage")
            if usage is not None:
                return usage
            return payload
        usage = getattr(payload, "usage", None)
        if usage is not None:
            return usage
        model_dump = getattr(payload, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump()
            except Exception:
                dumped = None
            if isinstance(dumped, dict):
                return dumped.get("usage")
        return None

    @staticmethod
    def _normalize_role_meta(role_meta: Any) -> Any:
        if not isinstance(role_meta, dict):
            return role_meta
        normalized = dict(role_meta)
        normalized["domain"] = str(normalized.get("domain") or normalized.get("role_domain") or "").strip()
        normalized.pop("role_domain", None)
        return normalized

    @staticmethod
    def _usage_dict(usage: Any) -> Dict[str, int]:
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0}
        if isinstance(usage, dict):
            return {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            }
        return {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        }

    @staticmethod
    def _extract_chat_content(response: Any) -> str:
        if response is None:
            return ""
        if isinstance(response, str):
            raw = response.strip()
            if raw.startswith("data:"):
                parts: List[str] = []
                for line in raw.splitlines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        obj = json.loads(payload)
                    except Exception:
                        continue
                    choices = obj.get("choices") or []
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        delta = choice.get("delta") or {}
                        message = choice.get("message") or {}
                        for source in (delta, message, choice):
                            if not isinstance(source, dict):
                                continue
                            for key in ("content", "reasoning_content", "reasoning", "text"):
                                value = source.get(key)
                                if value:
                                    parts.append(str(value))
                return "".join(parts).strip()
            return raw
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                choice = choices[0] or {}
                message = choice.get("message") or {}
                if isinstance(message, dict):
                    for key in ("content", "reasoning_content", "reasoning"):
                        value = message.get(key)
                        if value:
                            return str(value).strip()
                text = choice.get("text")
                if text:
                    return str(text).strip()
            return str(response.get("content") or "").strip()

        choices = getattr(response, "choices", None) or []
        if choices:
            choice = choices[0]
            message = getattr(choice, "message", None)
            if message is not None:
                for key in ("content", "reasoning_content", "reasoning"):
                    value = getattr(message, key, None)
                    if value:
                        return str(value).strip()
                if isinstance(message, dict):
                    for key in ("content", "reasoning_content", "reasoning"):
                        value = message.get(key)
                        if value:
                            return str(value).strip()
            text = getattr(choice, "text", None)
            if text:
                return str(text).strip()
            if isinstance(choice, dict):
                text = choice.get("text")
                if text:
                    return str(text).strip()
        return str(getattr(response, "content", "") or "").strip()

    def _total_tokens_from_usage(self, usage_dict: Dict[str, int]) -> float:
        prompt_tokens = float(usage_dict.get("prompt_tokens", 0))
        completion_tokens = float(usage_dict.get("completion_tokens", 0))
        return prompt_tokens + completion_tokens

    def _write_debug_trace(
        self,
        current_gen: int,
        agent_id: int,
        task_description: str,
        agent_response: str,
        judge_prompt_used: str,
        judge_raw_response_text: str,
        parsed_json: Dict[str, Any],
        fitness_score: float,
    ) -> None:
        if not self.debug_file:
            return
        judge_prompt_excerpt = judge_prompt_used[:200]
        log_content = f"""
## Generation: {current_gen} | Agent ID: {agent_id}

### 1. Agent Context
**Task Prompt**: {task_description}
**Agent Output**: {agent_response}

### 2. Judge Context
**Judge Prompt used**: {judge_prompt_excerpt}
**Judge Raw Response (JSON)**:
{judge_raw_response_text}

### 3. Parsed Metrics
- correct_claims: {parsed_json.get('correct_claims')}
- minor_errors: {parsed_json.get('minor_errors')}
- logical_flaws: {parsed_json.get('logical_flaws')}
- fabrications: {parsed_json.get('fabrications')}
- calculated_fitness: {fitness_score}

---
"""
        self.debug_file.write(log_content)
        self.debug_file.flush()

    def _write_judge_skipped_trace(self, current_gen: int, agent_id: int) -> None:
        if not self.debug_file:
            return
        self.debug_file.write(
            f"## Generation: {current_gen} | Agent ID: {agent_id}\n\n"
            "[Judge Skipped in Intermediate Gen]\n\n---\n"
        )
        self.debug_file.flush()

    def close_debug_file(self) -> None:
        if getattr(self, "debug_file", None) and not self.debug_file.closed:
            self.debug_file.close()

    def _build_online_reference_context(self, question_text: str) -> str:
        task = self.current_task or {}
        normalized_question = _normalize_question_text(question_text)
        choices = self.graph_dict.get("choices")
        use_canonical_context = bool(
            getattr(self.config.egt, "deduplicate_online_risk_prompt_context", True)
        )
        raw_question = str(self.graph_dict.get("question") or "").strip()
        context_question = (
            _normalize_question_text(raw_question)
            if use_canonical_context and raw_question and isinstance(choices, list) and choices
            else normalized_question
        )
        parts = [f"Question Text:\n{context_question}"]

        choice_texts = [str(choice or "").strip() for choice in (choices or [])] if isinstance(choices, list) else []
        choices_already_present = bool(choice_texts) and all(
            choice_text in context_question for choice_text in choice_texts if choice_text
        )
        should_append_choices = bool(choice_texts) and (
            not use_canonical_context or not choices_already_present
        )
        if should_append_choices:
            option_lines = []
            for idx, choice in enumerate(choices[:10]):
                label = chr(ord("A") + idx)
                option_lines.append(f"{label}. {str(choice or '').strip()}")
            if option_lines:
                parts.append("Answer Options:\n" + "\n".join(option_lines))

        metadata = {}
        for key in ("subject", "domain", "claimed_domain", "target_domain", "split"):
            value = self.graph_dict.get(key)
            if value is not None and str(value).strip():
                metadata[key] = str(value).strip()
        task_meta = task.get("task_metadata")
        if isinstance(task_meta, dict):
            for key, value in task_meta.items():
                normalized_key = str(key).strip().lower()
                if normalized_key in self.FORBIDDEN_ONLINE_METADATA_KEYS:
                    raise RuntimeError(
                        f"oracle-free online context rejected forbidden task metadata field: {key}"
                    )
                if (
                    normalized_key in self.ONLINE_TASK_METADATA_KEYS
                    and value is not None
                    and str(value).strip()
                ):
                    metadata[normalized_key] = str(value).strip()
        if metadata:
            meta_lines = [f"{key}: {value}" for key, value in metadata.items()]
            parts.append("Task Metadata:\n" + "\n".join(meta_lines))

        parts.append(self.DEFAULT_ONLINE_REFERENCE_CONTEXT)
        return "\n\n".join(parts)

    def _build_online_risk_user_prompt(
        self,
        *,
        question_text: str,
        online_reference_context: str,
        sections: List[tuple[str, str]],
    ) -> str:
        generated_text = ""
        for title, content in sections:
            if str(title).strip().lower() == "generated answer":
                generated_text = str(content)
                break
        _, user_prompt = build_atomic_risk_judge_prompts(
            question_text=question_text,
            online_reference_context=online_reference_context,
            generated_text=generated_text,
            deduplicate_context=bool(
                getattr(self.config.egt, "deduplicate_online_risk_prompt_context", True)
            ),
        )
        return user_prompt

    def _build_online_transmission_context(self, question_text: str) -> str:
        """Build the minimal public context needed for semantic inheritance."""

        graph = getattr(self, "graph_dict", {}) or {}
        raw_question = str(graph.get("question") or "").strip()
        context_question = _normalize_question_text(raw_question or question_text)
        if not raw_question:
            stem_lines = []
            for line in context_question.splitlines():
                if re.match(r"^\s*[A-J][.)]\s+", line):
                    break
                stem_lines.append(line)
            compact_stem = "\n".join(stem_lines).strip()
            if compact_stem:
                context_question = compact_stem

        parts = [context_question]
        metadata = []
        for key in ("subject", "domain"):
            value = graph.get(key)
            if value is not None and str(value).strip():
                metadata.append(f"{key}: {str(value).strip()}")
        if metadata:
            parts.append("Task Metadata:\n" + "\n".join(metadata))
        return "\n\n".join(part for part in parts if part)

    def _build_offline_reference_fact(self, question_text: str) -> str:
        task = self.current_task or {}
        cached = task.get("offline_reference_fact")
        if cached:
            return str(cached)

        normalized_question = _normalize_question_text(question_text)
        parts = [f"Question: {normalized_question}"]
        gold = (
            self.graph_dict.get("gold")
            if self.graph_dict.get("gold") is not None
            else self.graph_dict.get("answer")
        )
        if gold is not None and str(gold).strip():
            parts.append(f"Ground-truth answer: {str(gold).strip()}")

        extra_reference = self.graph_dict.get("reference_fact")
        if extra_reference is not None and str(extra_reference).strip():
            parts.append(f"Reference fact: {str(extra_reference).strip()}")

        parts.append(
            "Use the ground-truth answer and any explicit reference fact only for offline final evaluation or debug analysis."
        )
        return "\n".join(parts)

    def _build_offline_gold_reference(self) -> str:
        gold = (
            self.graph_dict.get("gold")
            if self.graph_dict.get("gold") is not None
            else self.graph_dict.get("answer")
        )
        return build_gold_reference(gold, self.graph_dict.get("choices"))

    def _build_judge_prompts(self, question_text: str, generated_text: str, reference_context: str) -> tuple[str, str]:
        normalized_question = _normalize_question_text(question_text)
        judge_system_prompt = (
            "You are performing contrastive fact checking.\n"
            f"Reference context: {reference_context}\n"
            "Judge the generated answer against the provided context only. Even if the answer is self-consistent, "
            "mark it as hallucination when key facts or causal relations conflict with the context.\n\n"
            "Classify hallucinations into:\n"
            "1. minor_errors\n"
            "2. logical_flaws\n"
            "3. fabrications\n\n"
            "Return pure JSON only.\n"
            "{\n"
            '  "reasoning_scratchpad": "factually correct",\n'
            '  "is_refusal": false,\n'
            '  "total_claims": 1,\n'
            '  "correct_claims": 1,\n'
            '  "minor_errors": 0,\n'
            '  "logical_flaws": 0,\n'
            '  "fabrications": 0\n'
            "}"
        )
        judge_user_prompt = (
            f"Question:\n{normalized_question}\n\n"
            f"Reference Context:\n{reference_context}\n\n"
            f"Generated Answer:\n{generated_text}"
        )
        return judge_system_prompt, judge_user_prompt

    @staticmethod
    def _oracle_free_violation_patterns() -> List[re.Pattern[str]]:
        return [
            re.compile(
                r"(?:^|\n)\s*(?:ground[- ]truth(?:\s+answer)?|gold(?:\s+answer)?|"
                r"correct[_ ]answer|correct[_ ]option|answer[_ ]key|reference[_ ]answer|"
                r"reference[_ ]fact|label|target)\s*:",
                re.IGNORECASE,
            ),
            re.compile(
                r"""["'](?:gold_answer|ground_truth(?:_answer)?|correct_answer|answer_key|"""
                r"""reference_answer|reference_fact|label|target)["']\s*:""",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:^|\n)\s*(?:offline\s+reference\s+fact|injection\s+metadata|"
                r"injection\s+label|peer\s+messages|other\s+agent\s+risk|judge\s+output)\s*:",
                re.IGNORECASE,
            ),
        ]

    def _assert_oracle_free_online_prompt(
        self,
        prompt_text: str,
        *,
        source: str,
        allowed_public_texts: Optional[List[str]] = None,
    ) -> None:
        if not bool(getattr(self.config.egt, "online_risk_observer_oracle_free", True)):
            return
        text = str(prompt_text or "")
        # Public question/response text can legitimately contain phrases such as
        # "the correct answer". Remove exact allowlisted payloads before scanning
        # the builder-controlled prompt scaffolding for structured leakage fields.
        for public_text in sorted(
            (str(value) for value in (allowed_public_texts or []) if str(value or "")),
            key=len,
            reverse=True,
        ):
            text = text.replace(public_text, "<PUBLIC_CONTEXT>")
        for pattern in self._oracle_free_violation_patterns():
            match = pattern.search(text)
            if match is not None:
                raise RuntimeError(
                    "oracle-free online prompt violation in "
                    f"{source}: found structured leakage field {match.group(0).strip()}"
                )

    def _record_online_prompt_sample(self, *, source: str, system_prompt: str, user_prompt: str) -> None:
        sample = {
            "source": str(source),
            "system_prompt_snippet": str(system_prompt or "")[:1200],
            "user_prompt_snippet": str(user_prompt or "")[:2400],
        }
        self.latest_online_prompt_samples.append(sample)
        if len(self.latest_online_prompt_samples) > 8:
            self.latest_online_prompt_samples = self.latest_online_prompt_samples[-8:]

    def _snapshot_agent_outputs(self) -> Dict[int, str]:
        outputs: Dict[int, str] = {}
        nodes = self.graph_dict.get("nodes", [])
        if not isinstance(nodes, list):
            return outputs
        for node_id, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            text = str(node.get("response") or "").strip()
            outputs[int(node_id)] = text
            if text:
                self.latest_agent_responses[int(node_id)] = text
        return outputs

    def _risk_observation_text(self, response_text: Any) -> str:
        """Attach selected-option semantics without exposing the gold answer."""

        raw_text = str(response_text or "").strip()
        parsed_answer = self._parse_choice_answer(raw_text)
        option_text = self._option_text_map().get(parsed_answer or "", "")
        if not parsed_answer or not option_text:
            return raw_text
        return (
            f"{raw_text}\n\n[Selected Option Semantics]\n"
            f"The response selects option {parsed_answer}: {option_text}"
        ).strip()

    def _empty_risk_usage(self) -> Dict[str, Any]:
        return {
            "risk_judge_prompt_tokens": 0,
            "risk_judge_completion_tokens": 0,
            "risk_judge_capped_count": 0,
            "risk_judge_invalid_count": 0,
            "risk_judge_schema_retry_count": 0,
            "risk_judge_call_count": 0,
            "risk_judge_terminal_capped": 0,
            "risk_claim_judge_prompt_tokens": 0,
            "risk_claim_judge_completion_tokens": 0,
            "risk_claim_judge_call_count": 0,
            "transmission_judge_prompt_tokens": 0,
            "transmission_judge_completion_tokens": 0,
            "transmission_judge_call_count": 0,
            "missing_before_response_count": 0,
        }

    @staticmethod
    def _merge_risk_usage(total: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
        for key in (
            "risk_judge_prompt_tokens",
            "risk_judge_completion_tokens",
            "risk_judge_capped_count",
            "risk_judge_invalid_count",
            "risk_judge_schema_retry_count",
            "risk_judge_call_count",
            "risk_judge_terminal_capped",
            "risk_claim_judge_prompt_tokens",
            "risk_claim_judge_completion_tokens",
            "risk_claim_judge_call_count",
            "transmission_judge_prompt_tokens",
            "transmission_judge_completion_tokens",
            "transmission_judge_call_count",
            "missing_before_response_count",
        ):
            total[key] = int(total.get(key, 0) or 0) + int(extra.get(key, 0) or 0)
        return total

    def _accumulate_risk_audit(self, usage: Dict[str, Any]) -> None:
        self._merge_risk_usage(self.risk_judge_audit_totals, usage)

    def _parse_atomic_claim_payload(self, raw_content: str) -> tuple[List[Dict[str, Any]], bool]:
        try:
            payload = json.loads(raw_content)
        except Exception:
            return [], False
        claims = payload.get("claims", []) if isinstance(payload, dict) else []
        normalized: List[Dict[str, Any]] = []
        payload_valid = True
        for idx, claim in enumerate(claims or [], start=1):
            if not isinstance(claim, dict):
                continue
            text = str(claim.get("text", "") or "").strip()
            if not text:
                continue
            support_status = str(claim.get("support_status", "uncertain") or "uncertain").strip().lower()
            try:
                risk = clamp01(float(claim.get("risk", 0.5)))
                importance = clamp01(float(claim.get("importance", 0.5)))
            except (TypeError, ValueError):
                payload_valid = False
                continue
            claim_valid = support_status in {
                "supported",
                "uncertain",
                "unsupported",
                "contradicted",
            }
            payload_valid = payload_valid and claim_valid
            normalized.append(
                {
                    "claim_id": str(claim.get("claim_id", f"claim_{idx}")),
                    "text": text,
                    "support_status": support_status,
                    "risk": risk,
                    "importance": importance,
                    "metric_valid": claim_valid,
                }
            )
        return normalized, bool(normalized) and payload_valid

    def _parse_inheritance_payload(self, raw_content: str) -> tuple[Dict[str, Any], bool]:
        try:
            payload = json.loads(raw_content)
        except Exception:
            return {}, False
        if not isinstance(payload, dict):
            return {}, False
        matched = bool(payload.get("matched", False))
        inheritance_type = str(payload.get("inheritance_type", "none") or "none").strip().lower()
        try:
            transmission = clamp01(float(payload.get("transmission", 0.0) or 0.0))
        except (TypeError, ValueError):
            return {}, False
        valid_types = {"none", "inherit", "repeat", "reinforce"}
        valid = (
            inheritance_type in valid_types
            and ((not matched and inheritance_type == "none" and transmission <= 0.25)
                 or (matched and inheritance_type != "none" and transmission > 0.0))
        )
        return {
            "matched": matched,
            "inheritance_type": inheritance_type,
            "transmission": transmission,
            "metric_valid": valid,
        }, valid

    def _judge_json_only(self, *, kind: str, system_prompt: str, user_prompt: str) -> tuple[str, Dict[str, Any]]:
        usage_dict = self._empty_risk_usage()
        if not self.judge_client:
            usage_dict["risk_judge_invalid_count"] = 1
            return "", usage_dict
        cache_key = self._judge_cache_key(kind, system_prompt, user_prompt)
        if not self.disable_cache:
            cached = self.judge_cache.get(cache_key)
            if isinstance(cached, dict):
                raw_answer = str(cached.get("raw_answer", "") or "")
                if raw_answer:
                    return raw_answer, dict(cached.get("usage", {}) or usage_dict)
        max_tokens = int(self.config.llm.judge_max_tokens)
        raw_answer = ""
        for attempt in range(2):
            try:
                response = self._call_llm_with_retry(
                    kind,
                    lambda: self._chat_completion(
                        role="judge",
                        model_name=self._judge_model(),
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self.config.llm.judge_temperature,
                        max_tokens=max_tokens,
                        timeout_seconds=60,
                        response_format={"type": "json_object"},
                    ),
                )
                raw_answer = self._extract_chat_content(response).strip()
                usage_obj = self._extract_usage(response)
                self._accumulate_usage(usage_obj, "judge_prompt", "judge_completion")
                raw_usage = self._usage_dict(usage_obj)
                prompt_tokens = int(raw_usage.get("prompt_tokens", 0) or 0)
                completion_tokens = int(raw_usage.get("completion_tokens", 0) or 0)
                usage_dict["risk_judge_prompt_tokens"] += prompt_tokens
                usage_dict["risk_judge_completion_tokens"] += completion_tokens
                usage_dict["risk_judge_call_count"] += 1
                if kind.startswith("risk_claim_extract"):
                    usage_dict["risk_claim_judge_prompt_tokens"] += prompt_tokens
                    usage_dict["risk_claim_judge_completion_tokens"] += completion_tokens
                    usage_dict["risk_claim_judge_call_count"] += 1
                elif kind.startswith("risk_claim_inheritance"):
                    usage_dict["transmission_judge_prompt_tokens"] += prompt_tokens
                    usage_dict["transmission_judge_completion_tokens"] += completion_tokens
                    usage_dict["transmission_judge_call_count"] += 1
                self.token_usage["online_risk_judge_prompt"] += prompt_tokens
                self.token_usage["online_risk_judge_completion"] += completion_tokens
                if completion_tokens >= 0.95 * max_tokens:
                    usage_dict["risk_judge_capped_count"] += 1
                    if attempt == 0:
                        continue
                    usage_dict["risk_judge_terminal_capped"] = 1
                    usage_dict["risk_judge_invalid_count"] += 1
                    return "", usage_dict
                if not self.disable_cache and raw_answer:
                    self.judge_cache[cache_key] = {"raw_answer": raw_answer, "usage": usage_dict}
                    self._save_json_cache(self.config.paths.judge_cache_path, self.judge_cache)
                return raw_answer, usage_dict
            except Exception:
                if attempt == 0:
                    continue
        usage_dict["risk_judge_invalid_count"] = 1
        return raw_answer, usage_dict

    def judge_atomic_claims(
        self,
        question_text: str,
        generated_text: str,
        online_reference_context: str,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        system_prompt, user_prompt = build_atomic_risk_judge_prompts(
            question_text=question_text,
            online_reference_context=online_reference_context,
            generated_text=generated_text,
            deduplicate_context=bool(
                getattr(self.config.egt, "deduplicate_online_risk_prompt_context", True)
            ),
        )
        self._assert_oracle_free_online_prompt(system_prompt, source="risk_claim_extract.system")
        self._assert_oracle_free_online_prompt(
            user_prompt,
            source="risk_claim_extract.user",
            allowed_public_texts=[
                question_text,
                online_reference_context,
                generated_text,
            ],
        )
        self._record_online_prompt_sample(
            source="risk_claim_extract",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        raw_answer, usage = self._judge_json_only(
            kind="risk_claim_extract",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        claims, valid = self._parse_atomic_claim_payload(raw_answer)
        if not valid and not bool(usage.get("risk_judge_terminal_capped", 0)):
            retry_system_prompt = system_prompt + (
                "\nThe previous output violated the required JSON schema or basic field validity. "
                "Return a corrected JSON object only."
            )
            retry_raw, retry_usage = self._judge_json_only(
                kind="risk_claim_extract_schema_retry",
                system_prompt=retry_system_prompt,
                user_prompt=user_prompt,
            )
            retry_usage["risk_judge_schema_retry_count"] = int(
                retry_usage.get("risk_judge_schema_retry_count", 0) or 0
            ) + 1
            usage = self._merge_risk_usage(usage, retry_usage)
            claims, valid = self._parse_atomic_claim_payload(retry_raw)
            if not valid:
                usage["risk_judge_invalid_count"] = int(usage.get("risk_judge_invalid_count", 0) or 0) + 1
                claims = self.judge_risk_observer._fallback_extract_claims(response_text=generated_text)
        elif not valid:
            claims = self.judge_risk_observer._fallback_extract_claims(response_text=generated_text)
        self._accumulate_risk_audit(usage)
        return claims, usage

    def judge_claim_inheritance(
        self,
        question_text: str,
        sender_claim: Dict[str, Any],
        receiver_before_text: str,
        receiver_after_text: str,
        online_reference_context: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if not str(receiver_before_text or "").strip():
            usage = self._empty_risk_usage()
            usage["missing_before_response_count"] = 1
            return {
                "matched": False,
                "inheritance_type": "none",
                "transmission": 0.0,
                "metric_valid": False,
                "observation_status": "missing_before_response",
            }, usage
        transmission_context = self._build_online_transmission_context(question_text)
        system_prompt, user_prompt = build_transmission_judge_prompts(
            question_context=transmission_context,
            sender_claim=sender_claim,
            receiver_before_text=receiver_before_text,
            receiver_after_text=receiver_after_text,
        )
        self._assert_oracle_free_online_prompt(system_prompt, source="risk_claim_inheritance.system")
        self._assert_oracle_free_online_prompt(
            user_prompt,
            source="risk_claim_inheritance.user",
            allowed_public_texts=[
                transmission_context,
                str(sender_claim.get("claim_id", "") or ""),
                str(sender_claim.get("text", "") or ""),
                receiver_before_text,
                receiver_after_text,
            ],
        )
        self._record_online_prompt_sample(
            source="risk_claim_inheritance",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        raw_answer, usage = self._judge_json_only(
            kind="risk_claim_inheritance",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        inheritance, valid = self._parse_inheritance_payload(raw_answer)
        if not valid and not bool(usage.get("risk_judge_terminal_capped", 0)):
            retry_system_prompt = system_prompt + (
                "\nConsistency rule: unmatched requires type=none and transmission no greater than 0.25; "
                "matched requires a non-none type and transmission greater than 0. Return corrected JSON only."
            )
            retry_raw, retry_usage = self._judge_json_only(
                kind="risk_claim_inheritance_schema_retry",
                system_prompt=retry_system_prompt,
                user_prompt=user_prompt,
            )
            retry_usage["risk_judge_schema_retry_count"] = int(
                retry_usage.get("risk_judge_schema_retry_count", 0) or 0
            ) + 1
            usage = self._merge_risk_usage(usage, retry_usage)
            inheritance, valid = self._parse_inheritance_payload(retry_raw)
            if not valid:
                usage["risk_judge_invalid_count"] = int(usage.get("risk_judge_invalid_count", 0) or 0) + 1
                inheritance = self.judge_risk_observer._fallback_inheritance(
                    sender_claim=sender_claim,
                    receiver_before_text=receiver_before_text,
                    receiver_after_text=receiver_after_text,
                )
        elif not valid:
            inheritance = self.judge_risk_observer._fallback_inheritance(
                sender_claim=sender_claim,
                receiver_before_text=receiver_before_text,
                receiver_after_text=receiver_after_text,
            )
        self._accumulate_risk_audit(usage)
        return inheritance, usage

    def _estimate_risk_state(self):
        self.risk_observer_invocation_count += 1
        question_text = self.graph_dict.get("question_text", self.DEFAULT_QUESTION)
        online_reference_context = self._build_online_reference_context(question_text)
        raw_agent_outputs = self._snapshot_agent_outputs()
        agent_outputs = {
            int(agent_id): self._risk_observation_text(output_text)
            for agent_id, output_text in raw_agent_outputs.items()
        }
        raw_sender_outputs = self.latest_generation_input_snapshot or raw_agent_outputs
        sender_outputs = {
            int(agent_id): self._risk_observation_text(output_text)
            for agent_id, output_text in raw_sender_outputs.items()
        }
        history = {
            "online_reference_context": online_reference_context,
            "sender_outputs": sender_outputs,
            "sender_claims_by_agent": self.latest_claims_by_agent,
            "sender_pi_by_agent": self.latest_pi_by_agent,
            "sender_observation_texts": self.latest_observed_output_texts,
            "reuse_unchanged_claim_observations": bool(
                getattr(self.config.egt, "reuse_unchanged_claim_observations", True)
            ),
        }
        risk_state = self.judge_risk_observer.estimate_risk_state(
            question_text,
            agent_outputs,
            self.graph_dict,
            history=history,
        )
        self.risk_judge_audit_totals["missing_before_response_count"] = int(
            self.risk_judge_audit_totals.get("missing_before_response_count", 0) or 0
        ) + int((risk_state.details or {}).get("missing_before_response_count", 0) or 0)
        return risk_state

    def _observe_initial_intrinsic_risk(self) -> RiskState:
        """Observe Round-0 claim risk without inventing pre-communication transmission."""

        question_text = self.graph_dict.get("question_text", self.DEFAULT_QUESTION)
        online_reference_context = self._build_online_reference_context(question_text)
        raw_agent_outputs = self._snapshot_agent_outputs()
        agent_outputs = {
            int(agent_id): self._risk_observation_text(output_text)
            for agent_id, output_text in raw_agent_outputs.items()
        }
        graph_without_observed_edges = dict(self.graph_dict)
        graph_without_observed_edges["edges"] = []
        self.risk_observer_invocation_count += 1
        risk_state = self.judge_risk_observer.estimate_risk_state(
            question_text,
            agent_outputs,
            graph_without_observed_edges,
            history={"online_reference_context": online_reference_context},
        )
        risk_state.details.update(
            {
                "state_timing": "injected_round_0",
                "transmission_identifiable": False,
                "rho_mode": "not_inferred_before_peer_message_consumption",
            }
        )
        self._apply_risk_state(risk_state)
        self.initial_risk_state = risk_state
        return risk_state

    def project_risk_state_to_current_graph(self, observed_state: RiskState) -> RiskState:
        """Project one response observation onto a changed topology without another LLM call."""

        details = copy.deepcopy(observed_state.details or {})
        observed_edges = dict(details.get("per_edge", {}) or {})
        projected_edges: Dict[str, Dict[str, Any]] = {}
        projected_t: Dict[tuple[int, int], float] = {}
        incoming_factors: Dict[int, List[float]] = {
            int(agent_id): [] for agent_id in observed_state.pi
        }
        structural_transmission = not get_functional_ablation_spec(
            getattr(self.config.egt, "functional_ablation_mode", "full")
        ).semantic_transmission_enabled

        for raw_edge in self.graph_dict.get("edges", []) or []:
            if not isinstance(raw_edge, (list, tuple)) or len(raw_edge) < 2:
                continue
            src, dst = int(raw_edge[0]), int(raw_edge[1])
            edge_key = f"{src}->{dst}"
            edge_detail = copy.deepcopy(observed_edges.get(edge_key) or {})
            if structural_transmission:
                metric_valid = True
                missing_before = False
                edge_t = 1.0
                rho_edge_t = 1.0
                source_pi = clamp01(
                    observed_state.source_pi.get(src, observed_state.pi.get(src, 0.0))
                )
                edge_detail = {
                    "transmission": 1.0,
                    "matched_claims": [],
                    "metric_valid": True,
                    "observation_status": "structural_active_edge",
                    "projection_status": "structural_active_edge",
                    "structural_transmission_ablation": True,
                }
            elif edge_detail:
                metric_valid = bool(edge_detail.get("metric_valid", False))
                edge_t = clamp01(observed_state.T.get((src, dst), 0.0))
                missing_before = bool(edge_detail.get("missing_before_response", False))
                rho_edge_t = edge_t if metric_valid and not missing_before else 0.0
                source_pi = clamp01(
                    edge_detail.get(
                        "source_message_pi",
                        observed_state.source_pi.get(src, observed_state.pi.get(src, 0.0)),
                    )
                )
                edge_detail["projection_status"] = "retained_observation"
            else:
                metric_valid = False
                missing_before = False
                edge_t = 0.0
                rho_edge_t = 0.0
                source_pi = clamp01(observed_state.pi.get(src, 0.0))
                edge_detail = {
                    "transmission": 0.0,
                    "matched_claims": [],
                    "metric_valid": False,
                    "observation_status": "unknown_new_edge",
                    "projection_status": "new_edge_unobserved",
                    "source_message_round": "t",
                    "receiver_response_round": "t",
                }

            edge_detail.update(
                {
                    "source_message_pi": source_pi,
                    "rho_transmission_used": rho_edge_t,
                    "rho_uses_conservative_unknown": False,
                    "invalid_observation_excluded_from_rho": bool(not metric_valid),
                }
            )
            projected_edges[edge_key] = edge_detail
            projected_t[(src, dst)] = edge_t if metric_valid else 0.0
            incoming_factors.setdefault(dst, []).append(1.0 - rho_edge_t * source_pi)

        projected_rho: Dict[int, float] = {}
        projected_h: Dict[int, float] = {}
        for agent_id, pi_value in observed_state.pi.items():
            no_inherit_prob = 1.0
            for factor in incoming_factors.get(int(agent_id), []):
                no_inherit_prob *= clamp01(factor)
            projected_rho[int(agent_id)] = clamp01(1.0 - no_inherit_prob)
            projected_h[int(agent_id)] = aggregate_risk_from_pi_rho(
                pi_value,
                projected_rho[int(agent_id)],
            )
            per_agent = details.setdefault("per_agent", {}).setdefault(int(agent_id), {})
            incoming_rows = [
                row for key, row in projected_edges.items()
                if key.endswith(f"->{int(agent_id)}")
            ]
            per_agent["rho_metric_valid"] = (
                all(bool(row.get("metric_valid", False)) for row in incoming_rows)
                if incoming_rows
                else True
            )

        details["per_edge"] = projected_edges
        details["state_timing"] = "post_governance_structural_projection"
        details["projection_uses_additional_llm_call"] = False
        return RiskState(
            pi=dict(observed_state.pi),
            rho=projected_rho,
            T=projected_t,
            source_pi=dict(observed_state.source_pi),
            details=details,
            aggregate_risk=projected_h,
            legacy_h=dict(projected_h),
        )

    @staticmethod
    def _edge_observation_valid(risk_state: RiskState, src: int, dst: int) -> bool:
        per_edge = dict((risk_state.details or {}).get("per_edge", {}) or {})
        return bool((per_edge.get(f"{int(src)}->{int(dst)}") or {}).get("metric_valid", False))

    def _edge_source_pi(self, src: int, dst: int, current_pi: float) -> float:
        risk_state = self.latest_risk_state
        per_edge = dict((risk_state.details or {}).get("per_edge", {}) or {}) if risk_state is not None else {}
        edge_detail = dict(per_edge.get(f"{int(src)}->{int(dst)}") or {})
        if "source_message_pi" in edge_detail:
            return clamp01(edge_detail.get("source_message_pi", current_pi))
        if (
            risk_state is not None
            and edge_detail
            and int(src) in risk_state.source_pi
        ):
            return clamp01(risk_state.source_pi.get(int(src), current_pi))
        return clamp01(current_pi)

    def _transmission_matrix_from_state(self, risk_state: RiskState, size: int) -> List[List[Any]]:
        matrix: List[List[Any]] = [[None for _ in range(size)] for _ in range(size)]
        for (src, dst), value in risk_state.T.items():
            if (
                0 <= int(dst) < size
                and 0 <= int(src) < size
                and self._edge_observation_valid(risk_state, int(src), int(dst))
            ):
                matrix[int(dst)][int(src)] = clamp01(value)
        return matrix

    def _apply_risk_state(self, risk_state: RiskState) -> tuple[List[float], List[float], List[float], List[List[Any]], float]:
        """Apply one audited round observation to agents and topology metrics."""

        self.latest_risk_state = risk_state
        node_ids = sorted(self.agents.keys())
        pi_list = [clamp01(risk_state.pi.get(node_id, 0.0)) for node_id in node_ids]
        rho_list = [clamp01(risk_state.rho.get(node_id, 0.0)) for node_id in node_ids]
        h_list = [clamp01(risk_state.aggregate_risk.get(node_id, 0.0)) for node_id in node_ids]
        source_pi_list = [clamp01(risk_state.source_pi.get(node_id, pi_list[node_id])) for node_id in node_ids]
        size = len(node_ids)
        transmission_matrix = self._transmission_matrix_from_state(risk_state, size)

        self.latest_prediction_details = risk_state.to_legacy_details(size=size)
        self.latest_prediction_details["risk_estimator"] = "llm_judge_atomic_observer"
        self.latest_prediction_details["risk_estimator_mode"] = str(
            getattr(self.config.egt, "risk_estimator_mode", "judge_observer")
        )
        self.latest_transmission_matrix = transmission_matrix
        self.latest_edge_transmissions = self._build_edge_transmissions(
            transmission_matrix,
            pi_list,
            rho_list,
            source_pi_list=source_pi_list,
            risk_state=risk_state,
        )
        beta = self._resolve_edge_beta(
            edge_transmissions=self.latest_edge_transmissions,
            transmission_matrix=transmission_matrix,
            pi_list=pi_list,
            rho_list=rho_list,
        )

        per_agent = dict((risk_state.details or {}).get("per_agent", {}) or {})
        current_texts = dict((risk_state.details or {}).get("current_observation_texts", {}) or {})
        self.latest_observed_output_texts = {int(k): str(v or "") for k, v in current_texts.items()}
        self.latest_claims_by_agent = {
            int(node_id): list((per_agent.get(node_id) or per_agent.get(str(node_id)) or {}).get("claims", []) or [])
            for node_id in node_ids
        }
        self.latest_pi_by_agent = {int(node_id): float(pi_list[node_id]) for node_id in node_ids}

        for node_id, agent in self.agents.items():
            agent.pi = pi_list[node_id]
            agent.predicted_prob = agent.pi
            agent.rho = rho_list[node_id]
            agent.h_pre = agent.pi
            agent.h_final = h_list[node_id]
            agent.intrinsic_utility = 1.0 - agent.pi
            agent.infection_penalty = agent.rho

        return pi_list, rho_list, h_list, transmission_matrix, beta

    def _build_edge_transmissions(
        self,
        transmission_matrix: List[List[Any]],
        pi_list: List[float],
        rho_list: List[float],
        *,
        source_pi_list: List[float] | None = None,
        risk_state: RiskState | None = None,
    ) -> List[Dict[str, Any]]:
        edge_transmissions: List[Dict[str, Any]] = []
        for edge in self.graph_dict.get("edges", []):
            if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                continue
            sender = int(edge[0])
            receiver = int(edge[1])
            transmission_observed = bool(
                risk_state is not None and self._edge_observation_valid(risk_state, sender, receiver)
            )
            per_edge_detail = dict(
                ((risk_state.details or {}).get("per_edge", {}) or {}).get(f"{sender}->{receiver}") or {}
            ) if risk_state is not None else {}
            transmission_strength = self._safe_edge_t(
                transmission_matrix,
                sender,
                receiver,
                default_if_missing=0.0,
            )
            if "source_message_pi" in per_edge_detail:
                source_pi = clamp01(per_edge_detail.get("source_message_pi", 0.0))
                source_pi_basis = str(
                    per_edge_detail.get("projection_status")
                    or ("sender_message_t_minus_1" if transmission_observed else "sender_message_unobserved")
                )
            elif source_pi_list is not None and 0 <= sender < len(source_pi_list):
                source_pi = float(source_pi_list[sender])
                source_pi_basis = (
                    "sender_message_t_minus_1"
                    if transmission_observed
                    else "sender_message_t_minus_1_unobserved"
                )
            else:
                source_pi = float(pi_list[sender]) if 0 <= sender < len(pi_list) else 0.0
                source_pi_basis = "current_source_unobserved"
            source_rho = float(rho_list[sender]) if 0 <= sender < len(rho_list) else 0.0
            target_pi = float(pi_list[receiver]) if 0 <= receiver < len(pi_list) else 0.0
            target_rho = float(rho_list[receiver]) if 0 <= receiver < len(rho_list) else 0.0
            source_infection_penalty = 0.0
            if sender in self.agents:
                source_infection_penalty = float(getattr(self.agents[sender], "infection_penalty", 0.0) or 0.0)

            edge_transmissions.append(
                {
                    "sender": sender,
                    "receiver": receiver,
                    "transmission_strength": transmission_strength,
                    "transmission_observed": transmission_observed,
                    "transmission_status": "observed" if transmission_observed else "unknown_excluded",
                    "source_pi": source_pi,
                    "source_pi_basis": source_pi_basis,
                    "source_rho": source_rho,
                    "target_pi": target_pi,
                    "target_rho": target_rho,
                    "source_h_final": float(aggregate_risk_from_pi_rho(source_pi, source_rho)),
                    "target_h_final": float(aggregate_risk_from_pi_rho(target_pi, target_rho)),
                    "source_aggregate_risk": float(aggregate_risk_from_pi_rho(source_pi, source_rho)),
                    "target_aggregate_risk": float(aggregate_risk_from_pi_rho(target_pi, target_rho)),
                    "aggregate_risk": {
                        "source": float(aggregate_risk_from_pi_rho(source_pi, source_rho)),
                        "target": float(aggregate_risk_from_pi_rho(target_pi, target_rho)),
                    },
                    "source_infection_penalty": source_infection_penalty,
                }
            )

        edge_transmissions.sort(key=lambda item: item["transmission_strength"], reverse=True)
        return edge_transmissions

    def _compute_edge_risk_ratio_stats(
        self,
        *,
        edge_transmissions: List[Dict[str, Any]] | None = None,
        transmission_matrix: List[List[float]] | None = None,
        pi_list: List[float] | None = None,
        rho_list: List[float] | None = None,
    ) -> Dict[str, float]:
        eps = float(getattr(self.config.egt, "adaptive_beta_eps", 1e-8) or 1e-8)
        quantile_q = float(getattr(self.config.egt, "adaptive_beta_quantile", 0.85) or 0.85)
        rows = edge_transmissions
        if rows is None:
            rows = self._build_edge_transmissions(
                transmission_matrix or [],
                list(pi_list or []),
                list(rho_list or []),
            )

        ratios: List[float] = []
        for row in rows or []:
            if not bool(row.get("transmission_observed", True)):
                continue
            source_pi = float(row.get("source_pi", 0.0) or 0.0)
            receiver_rho = float(row.get("target_rho", row.get("receiver_rho", 0.0)) or 0.0)
            edge_t = self._safe_edge_t_from_value(
                row.get("transmission_strength", row.get("edge_transmission_strength", 0.0)),
                default_if_missing=1.0,
            )
            benefit = max(1.0 - source_pi, 0.0)
            ratio = (edge_t * source_pi * (1.0 + receiver_rho)) / (benefit + eps)
            ratios.append(float(ratio))

        configured_beta = getattr(self.config.egt, "fitness_penalty_beta", 1.0)
        configured_beta = float(1.0 if configured_beta is None else configured_beta)
        if not ratios:
            return {
                "edge_beta_used": configured_beta,
                "edge_beta_raw": configured_beta,
                "edge_beta_capped": False,
                "edge_beta_fallback_applied": False,
                "edge_risk_ratio_q85": 0.0,
                "avg_edge_risk_ratio": 0.0,
                "max_edge_risk_ratio": 0.0,
            }

        q_value = float(self._quantile(ratios, quantile_q))
        beta_raw = configured_beta
        beta_capped = False
        beta_fallback_applied = False
        if str(getattr(self.config.egt, "beta_mode", "fixed")).lower().strip() == "adaptive":
            beta_raw = q_value
            if q_value <= eps:
                beta_value = 1.0
                beta_fallback_applied = True
            else:
                beta_cap = max(0.0, float(getattr(self.config.egt, "adaptive_beta_cap", 8.0) or 8.0))
                beta_value = min(beta_raw, beta_cap)
                beta_capped = beta_value < beta_raw
        else:
            beta_value = configured_beta
        return {
            "edge_beta_used": float(beta_value),
            "edge_beta_raw": float(beta_raw),
            "edge_beta_capped": bool(beta_capped),
            "edge_beta_fallback_applied": bool(beta_fallback_applied),
            "edge_risk_ratio_q85": q_value,
            "avg_edge_risk_ratio": float(sum(ratios) / len(ratios)),
            "max_edge_risk_ratio": float(max(ratios)),
        }

    def _resolve_edge_beta(
        self,
        *,
        edge_transmissions: List[Dict[str, Any]] | None = None,
        transmission_matrix: List[List[float]] | None = None,
        pi_list: List[float] | None = None,
        rho_list: List[float] | None = None,
    ) -> float:
        stats = self._compute_edge_risk_ratio_stats(
            edge_transmissions=edge_transmissions,
            transmission_matrix=transmission_matrix,
            pi_list=pi_list,
            rho_list=rho_list,
        )
        self.latest_edge_beta_used = float(stats["edge_beta_used"])
        self.latest_edge_beta_raw = float(stats.get("edge_beta_raw", stats["edge_beta_used"]))
        self.latest_edge_beta_capped = bool(stats.get("edge_beta_capped", False))
        self.latest_edge_beta_fallback_applied = bool(
            stats.get("edge_beta_fallback_applied", False)
        )
        self.latest_edge_risk_ratio_q85 = float(stats["edge_risk_ratio_q85"])
        self.latest_avg_edge_risk_ratio = float(stats["avg_edge_risk_ratio"])
        self.latest_max_edge_risk_ratio = float(stats["max_edge_risk_ratio"])
        return self.latest_edge_beta_used

    def _is_observation_generation(self, current_gen: int) -> bool:
        raw_generations = {int(gen) for gen in getattr(self.config.egt, "observation_generations", ())}
        is_final_generation = current_gen == self.num_generations - 1
        if -1 in raw_generations:
            return bool(self.config.egt.always_observe_final and is_final_generation)

        explicit_generations = {gen for gen in raw_generations if gen >= 0}
        is_explicit_generation = current_gen in explicit_generations
        interval = int(getattr(self.config.egt, "observation_interval", 0) or 0)
        is_interval_generation = (
            (current_gen % interval == 0)
            if (not explicit_generations and interval > 0 and interval <= self.num_generations)
            else False
        )
        return is_explicit_generation or is_interval_generation or (
            self.config.egt.always_observe_final and is_final_generation
        )

    def generate_response_sync(self, persona_prompt: str, question_text: str) -> tuple[str, Dict[str, int]]:
        temperature = (
            self.config.llm.generation_temperature_gen0
            if int(self.current_task.get("current_gen", 0)) == 0
            else self.config.llm.generation_temperature_default
        )
        response_cache_key = self._response_cache_key(persona_prompt, question_text, temperature)
        if not self.disable_cache:
            cached_response = self.response_cache.get(response_cache_key)
            if isinstance(cached_response, dict):
                cached_text = str(cached_response.get("text", "") or "").strip()
                if cached_text and not cached_text.startswith("data:") and not cached_text.startswith("Error generating text:"):
                    return cached_text, dict(cached_response.get("usage", {}) or {})
                self.response_cache.pop(response_cache_key, None)

        if not self.response_client:
            raise RuntimeError("generation client unavailable; refusing to emit a mocked response")

        def _request_generation():
            api_response = self._chat_completion(
                role="generation",
                model_name=self._gen_model(),
                messages=[
                    {"role": "system", "content": persona_prompt},
                    {"role": "user", "content": question_text},
                ],
                max_tokens=self.config.llm.generation_max_tokens,
                temperature=temperature,
                timeout_seconds=60,
            )
            response_text = self._extract_chat_content(api_response)
            if not response_text:
                raise RuntimeError("empty generation response content")
            return api_response, response_text

        response, response_text = self._call_llm_with_retry("generate_response", _request_generation)
        usage_obj = self._extract_usage(response)
        self._accumulate_usage(usage_obj, "generation_prompt", "generation_completion")
        usage_dict = self._usage_dict(usage_obj)
        if not self.disable_cache:
            self.response_cache[response_cache_key] = {"text": response_text, "usage": usage_dict}
            self._save_json_cache(self.config.paths.response_cache_path, self.response_cache)
        return response_text, usage_dict

    def judge_text_sync_for_observation(
        self,
        question_text: str,
        generated_text: str,
        reference_context: str,
        *,
        oracle_free: bool = True,
    ) -> tuple[Dict[str, Any], str, str, Dict[str, int]]:
        if not self.judge_client:
            return self._default_judge_result(is_refusal=True), "", "", {"prompt_tokens": 0, "completion_tokens": 0, "from_cache": False}

        judge_system_prompt, judge_user_prompt = self._build_judge_prompts(
            question_text,
            generated_text,
            reference_context,
        )
        if oracle_free:
            self._assert_oracle_free_online_prompt(judge_system_prompt, source="judge_observation.system")
            self._assert_oracle_free_online_prompt(
                judge_user_prompt,
                source="judge_observation.user",
                allowed_public_texts=[
                    question_text,
                    reference_context,
                    generated_text,
                ],
            )
            self._record_online_prompt_sample(
                source="judge_observation",
                system_prompt=judge_system_prompt,
                user_prompt=judge_user_prompt,
            )
        observation_cache_key = self._judge_cache_key("judge_observation", judge_system_prompt, judge_user_prompt)
        if not self.disable_cache:
            cached_observation = self.judge_cache.get(observation_cache_key)
            if isinstance(cached_observation, dict):
                cached_raw = str(cached_observation.get("raw_answer", "") or "").strip()
                if cached_raw:
                    return (
                        dict(cached_observation.get("result", {}) or self._default_judge_result(is_refusal=True)),
                        judge_user_prompt,
                        cached_raw,
                        {"prompt_tokens": 0, "completion_tokens": 0, "from_cache": True},
                    )
                self.judge_cache.pop(observation_cache_key, None)

        try:
            def _request_observation_judge():
                response = self._chat_completion(
                    role="judge",
                    model_name=self._judge_model(),
                    messages=[
                        {"role": "system", "content": judge_system_prompt},
                        {"role": "user", "content": judge_user_prompt},
                    ],
                    temperature=self.config.llm.judge_temperature,
                    max_tokens=self.config.llm.judge_max_tokens,
                    timeout_seconds=60,
                    response_format={"type": "json_object"},
                )
                raw_answer = self._extract_chat_content(response)
                if not raw_answer.strip():
                    raise RuntimeError("empty judge_observation response content")
                return response, raw_answer

            response = self._call_llm_with_retry(
                "judge_observation",
                _request_observation_judge,
            )
            response, raw_answer = response
            usage_obj = self._extract_usage(response)
            self._accumulate_usage(usage_obj, "judge_prompt", "judge_completion")
            parsed = self._parse_judge_response(raw_answer)
            usage_dict = self._usage_dict(usage_obj)
            if not self.disable_cache:
                self.judge_cache[observation_cache_key] = {
                    "result": parsed,
                    "raw_answer": raw_answer,
                    "usage": usage_dict,
                }
                self._save_json_cache(self.config.paths.judge_cache_path, self.judge_cache)
            return parsed, judge_user_prompt, raw_answer, {**usage_dict, "from_cache": False}
        except Exception:
            return self._default_judge_result(is_refusal=True), judge_user_prompt, "", {"prompt_tokens": 0, "completion_tokens": 0, "from_cache": False}

    def sync_judge_text(
        self,
        question_text: str,
        generated_text: str,
        reference_context: str,
        *,
        oracle_free: bool = True,
    ) -> tuple[Dict[str, Any], str, Dict[str, int]]:
        if not self.judge_client:
            return self._default_judge_result(is_refusal=True), "", {"prompt_tokens": 0, "completion_tokens": 0}

        cache_key_source = f"{question_text}\n<<REF>>\n{reference_context}\n<<ANS>>\n{generated_text.strip()}"
        cache_key = hashlib.md5(cache_key_source.encode("utf-8")).hexdigest()
        if (not self.disable_cache) and cache_key in self.rep_judge_cache:
            self.rep_judge_cache_hits += 1
            cached_result, cached_raw, _ = self.rep_judge_cache[cache_key]
            return cached_result, cached_raw, {"prompt_tokens": 0, "completion_tokens": 0}

        judge_system_prompt, judge_user_prompt = self._build_judge_prompts(
            question_text,
            generated_text,
            reference_context,
        )
        if oracle_free:
            self._assert_oracle_free_online_prompt(judge_system_prompt, source="judge_rep.system")
            self._assert_oracle_free_online_prompt(
                judge_user_prompt,
                source="judge_rep.user",
                allowed_public_texts=[
                    question_text,
                    reference_context,
                    generated_text,
                ],
            )
            self._record_online_prompt_sample(
                source="judge_rep",
                system_prompt=judge_system_prompt,
                user_prompt=judge_user_prompt,
            )
        persisted_cache_key = self._judge_cache_key("judge_rep", judge_system_prompt, judge_user_prompt)
        if not self.disable_cache:
            cached_rep = self.judge_cache.get(persisted_cache_key)
            if isinstance(cached_rep, dict):
                raw_answer = str(cached_rep.get("raw_answer", "") or "").strip()
                if raw_answer:
                    parsed = dict(cached_rep.get("result", {}) or self._default_judge_result(is_refusal=True))
                    usage_dict = dict(cached_rep.get("usage", {}) or {})
                    self.rep_judge_cache[cache_key] = (parsed, raw_answer, usage_dict)
                    self.rep_judge_cache_hits += 1
                    return parsed, raw_answer, {"prompt_tokens": 0, "completion_tokens": 0}
                self.judge_cache.pop(persisted_cache_key, None)

        try:
            def _request_rep_judge():
                response = self._chat_completion(
                    role="judge",
                    model_name=self._judge_model(),
                    messages=[
                        {"role": "system", "content": judge_system_prompt},
                        {"role": "user", "content": judge_user_prompt},
                    ],
                    temperature=self.config.llm.judge_temperature,
                    max_tokens=self.config.llm.judge_max_tokens,
                    timeout_seconds=60,
                    response_format={"type": "json_object"},
                )
                raw_answer = self._extract_chat_content(response)
                if not raw_answer.strip():
                    raise RuntimeError("empty judge_rep response content")
                return response, raw_answer

            response = self._call_llm_with_retry(
                "judge_rep",
                _request_rep_judge,
            )
            response, raw_answer = response
            usage_obj = self._extract_usage(response)
            self._accumulate_usage(usage_obj, "rep_judge_prompt", "rep_judge_completion")
            parsed = self._parse_judge_response(raw_answer)
            usage_dict = self._usage_dict(usage_obj)
            if not self.disable_cache:
                self.rep_judge_cache[cache_key] = (parsed, raw_answer, usage_dict)
                self.judge_cache[persisted_cache_key] = {
                    "result": parsed,
                    "raw_answer": raw_answer,
                    "usage": usage_dict,
                }
                self._save_json_cache(self.config.paths.judge_cache_path, self.judge_cache)
            return parsed, raw_answer, usage_dict
        except Exception:
            return self._default_judge_result(is_refusal=True), "", {"prompt_tokens": 0, "completion_tokens": 0}

    def sync_final_component_judge_text(
        self,
        question_text: str,
        generated_text: str,
        gold_reference: str,
    ) -> tuple[Dict[str, Any], str, Dict[str, int]]:
        """Gold-aware offline final judge, isolated from online risk observation."""
        deterministic = deterministic_choice_only_payload(generated_text, gold_reference)
        if deterministic is not None:
            deterministic["final_component_judge_prompt_version"] = FINAL_COMPONENT_JUDGE_VERSION
            return (
                deterministic,
                json.dumps(deterministic, ensure_ascii=False),
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "from_cache": False,
                    "deterministic": True,
                    "capped_count": 0,
                    "invalid_count": 0,
                    "schema_retry_count": 0,
                    "call_count": 0,
                },
            )

        if not self.judge_client:
            self.final_judge_audit_totals["final_judge_invalid_count"] += 1
            failed = self._default_judge_result(is_refusal=False)
            failed.update({"judge_failed": True, "judge_failure_reason": "judge_client_unavailable"})
            return failed, "", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "capped_count": 0,
                "invalid_count": 1,
                "schema_retry_count": 0,
                "call_count": 0,
            }

        chunks = split_final_response(generated_text)
        cache_source = (
            f"{FINAL_COMPONENT_JUDGE_VERSION}\n{question_text}\n<<GOLD>>\n{gold_reference}"
            f"\n<<ANSWER>>\n{generated_text.strip()}"
        )
        cache_key = self._judge_cache_key("offline_final_component", cache_source, "")
        if not self.disable_cache:
            cached = self.judge_cache.get(cache_key)
            if isinstance(cached, dict) and not bool((cached.get("result") or {}).get("judge_failed", True)):
                return (
                    dict(cached["result"]),
                    str(cached.get("raw_answer", "") or ""),
                    {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "from_cache": True,
                        "capped_count": 0,
                        "invalid_count": 0,
                        "schema_retry_count": 0,
                        "call_count": 0,
                    },
                )

        payloads: list[Dict[str, Any]] = []
        raw_answers: list[str] = []
        usage_total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "from_cache": False,
            "capped_count": 0,
            "invalid_count": 0,
            "schema_retry_count": 0,
            "call_count": 0,
        }
        terminal_invalid_recorded = False
        try:
            for chunk_index, chunk in enumerate(chunks):
                system_prompt, user_prompt, _context_mode = build_final_component_prompts(
                    question_text=question_text,
                    gold_reference=gold_reference,
                    answer_chunk=chunk,
                    chunk_index=chunk_index,
                    chunk_count=len(chunks),
                )

                def _request_final_component_judge():
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    if model_family(self._judge_model()) == "deepseek" and self.judge_client:
                        request_kwargs = build_request_kwargs(
                            model_name=self._judge_model(),
                            messages=messages,
                            max_tokens=FINAL_COMPONENT_JUDGE_MAX_TOKENS,
                            temperature=0.0,
                            request_timeout=120,
                            response_format={"type": "json_object"},
                        )
                        request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                        response = self.judge_client.chat.completions.create(**request_kwargs)
                    else:
                        response = self._chat_completion(
                            role="judge",
                            model_name=self._judge_model(),
                            messages=messages,
                            temperature=0.0,
                            max_tokens=FINAL_COMPONENT_JUDGE_MAX_TOKENS,
                            timeout_seconds=120,
                            response_format={"type": "json_object"},
                        )
                    raw_answer = self._extract_chat_content(response)
                    if not raw_answer.strip():
                        raise RuntimeError("empty offline final component judge response")
                    return response, raw_answer

                parsed_chunk = None
                for schema_attempt in range(2):
                    response, raw_answer = self._call_llm_with_retry(
                        "offline_final_component_judge",
                        _request_final_component_judge,
                    )
                    usage_obj = self._extract_usage(response)
                    self._accumulate_usage(usage_obj, "judge_prompt", "judge_completion")
                    usage = self._usage_dict(usage_obj)
                    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                    self.token_usage["final_judge_prompt"] += prompt_tokens
                    self.token_usage["final_judge_completion"] += completion_tokens
                    usage_total["prompt_tokens"] += prompt_tokens
                    usage_total["completion_tokens"] += completion_tokens
                    usage_total["call_count"] += 1
                    self.final_judge_audit_totals["final_judge_prompt_tokens"] += prompt_tokens
                    self.final_judge_audit_totals["final_judge_completion_tokens"] += completion_tokens
                    self.final_judge_audit_totals["final_judge_call_count"] += 1

                    capped = completion_tokens >= 0.95 * FINAL_COMPONENT_JUDGE_MAX_TOKENS
                    if capped:
                        usage_total["capped_count"] += 1
                        self.final_judge_audit_totals["final_judge_capped_count"] += 1
                        if schema_attempt == 0:
                            usage_total["schema_retry_count"] += 1
                            self.final_judge_audit_totals["final_judge_schema_retry_count"] += 1
                            continue
                        usage_total["invalid_count"] += 1
                        self.final_judge_audit_totals["final_judge_invalid_count"] += 1
                        terminal_invalid_recorded = True
                        raise RuntimeError("offline final component judge remained capped after retry")

                    try:
                        parsed_chunk = parse_final_component_payload(raw_answer)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        if schema_attempt == 0:
                            usage_total["schema_retry_count"] += 1
                            self.final_judge_audit_totals["final_judge_schema_retry_count"] += 1
                            continue
                        usage_total["invalid_count"] += 1
                        self.final_judge_audit_totals["final_judge_invalid_count"] += 1
                        terminal_invalid_recorded = True
                        raise
                    raw_answers.append(raw_answer)
                    break

                if parsed_chunk is None:
                    raise RuntimeError("offline final component judge produced no valid payload")
                payloads.append(parsed_chunk)

            parsed = aggregate_final_component_payloads(payloads)
            parsed["final_component_judge_prompt_version"] = FINAL_COMPONENT_JUDGE_VERSION
            parsed["evaluation_mode"] = "llm_component_judge_v7"
            combined_raw = json.dumps(raw_answers, ensure_ascii=False)
            if not self.disable_cache:
                self.judge_cache[cache_key] = {
                    "result": parsed,
                    "raw_answer": combined_raw,
                    "usage": usage_total,
                }
                self._save_json_cache(self.config.paths.judge_cache_path, self.judge_cache)
            return parsed, combined_raw, usage_total
        except Exception as exc:
            if not terminal_invalid_recorded:
                usage_total["invalid_count"] += 1
                self.final_judge_audit_totals["final_judge_invalid_count"] += 1
            failed = self._default_judge_result(is_refusal=False)
            failed.update(
                {
                    "judge_failed": True,
                    "judge_failure_reason": f"offline_final_component_judge_failed: {type(exc).__name__}: {exc}",
                }
            )
            return failed, "", usage_total

    @staticmethod
    def _indexed_initial_value(value: Any, node_id: int) -> Any:
        if isinstance(value, list) and 0 <= int(node_id) < len(value):
            return value[int(node_id)]
        if isinstance(value, dict):
            return value.get(str(node_id), value.get(int(node_id)))
        return None

    def _initialize_responses_from_final_raw_outputs(self) -> None:
        nodes = self.graph_dict.get("nodes", [])
        raw_outputs = self.graph_dict.get("initial_raw_outputs_after_injection", [])
        if not isinstance(raw_outputs, list) or not raw_outputs:
            raw_outputs = self.graph_dict.get("final_raw_outputs", [])
        initial_raw_outputs = self.graph_dict.get("initial_raw_outputs", [])
        if not isinstance(nodes, list):
            return
        for node_id, node in enumerate(nodes):
            if not isinstance(node, dict) or node.get("response"):
                continue
            response_text = ""
            if isinstance(raw_outputs, list) and node_id < len(raw_outputs) and raw_outputs[node_id] is not None:
                response_text = str(raw_outputs[node_id])
            elif isinstance(initial_raw_outputs, list) and node_id < len(initial_raw_outputs) and initial_raw_outputs[node_id] is not None:
                response_text = str(initial_raw_outputs[node_id])
            elif node.get("round0_raw_output"):
                response_text = str(node.get("round0_raw_output"))
            if response_text:
                response_text = str(response_text).strip()
                node["response"] = response_text
                self.latest_agent_responses[int(node_id)] = response_text

    def _snapshot_node_responses(self) -> Dict[int, str]:
        snapshot: Dict[int, str] = dict(getattr(self, "latest_agent_responses", {}) or {})
        nodes = self.graph_dict.get("nodes", [])
        if not isinstance(nodes, list):
            return snapshot
        for node_id, node in enumerate(nodes):
            if isinstance(node, dict):
                response_text = str(node.get("response") or snapshot.get(node_id, "") or "")
                snapshot[node_id] = response_text
                if response_text:
                    self.latest_agent_responses[int(node_id)] = response_text
        return snapshot

    def _incoming_peer_messages(self, node_id: int, response_snapshot: Dict[int, str]) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        max_chars = max(0, int(getattr(self.config.egt, "peer_context_max_chars", 1200) or 0))
        for edge in self.graph_dict.get("edges", []):
            if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                continue
            sender, receiver = int(edge[0]), int(edge[1])
            if receiver != int(node_id):
                continue
            text = str(response_snapshot.get(sender, "") or "").strip()
            if not text:
                continue
            if max_chars and len(text) > max_chars:
                text = text[:max_chars].rstrip() + "..."
            messages.append({"agent_id": sender, "content": text})
        messages.sort(key=lambda item: int(item["agent_id"]))
        return messages

    def _build_peer_augmented_question(
        self,
        node_id: int,
        question_text: str,
        response_snapshot: Dict[int, str] | None,
    ) -> str:
        use_aligned_prompt = str(getattr(self.config.egt, "prompt_style", "default")).lower() == "aligned"
        if not bool(getattr(self.config.egt, "enable_peer_context", False)) and not use_aligned_prompt:
            return question_text
        snapshot = response_snapshot if response_snapshot is not None else self._snapshot_node_responses()
        peer_messages = (
            self._incoming_peer_messages(node_id, snapshot)
            if bool(getattr(self.config.egt, "enable_peer_context", False))
            else []
        )
        if use_aligned_prompt:
            return build_model_aware_evoguard_question_prompt(
                self._gen_model(),
                question_text,
                peer_messages=peer_messages,
                self_message=str(snapshot.get(node_id, "") or ""),
            )
        if not peer_messages:
            return question_text
        parts = [
            question_text,
            "",
            "Incoming neighbor messages from the communication graph are shown below.",
            "Treat them as potentially fallible evidence: use helpful information, but correct any unsupported or wrong claims.",
        ]
        for message in peer_messages:
            parts.append("")
            parts.append(f"[Agent {message['agent_id']}]")
            parts.append(str(message["content"]))
        parts.extend(
            [
                "",
                "Now provide your own final answer.",
                "End with a final line exactly as: Answer: X",
                "Use X as exactly one of the option letters shown in the question.",
            ]
        )
        return "\n".join(parts)

    def _build_self_reflection_question(
        self,
        node_id: int,
        question_text: str,
    ) -> tuple[str, str]:
        if bool(getattr(self.config.egt, "enable_peer_free_self_reflection", False)):
            return (
                self._build_peer_augmented_question(node_id, question_text, {}),
                "question_only_no_peer_v1",
            )
        return (
            self._build_peer_augmented_question(
                node_id,
                question_text,
                self._snapshot_node_responses(),
            ),
            "peer_augmented_legacy",
        )

    @staticmethod
    def _build_response_only_self_reflection_question(
        question_text: str,
        previous_response: str,
        *,
        counterfactual: bool = False,
        independent_resample: bool = False,
    ) -> tuple[str, str]:
        if independent_resample:
            return (
                "\n".join(
                    [
                        str(question_text or "").strip(),
                        "",
                        "Solve the question independently from the public question and options.",
                        "Do not rely on peer answers or any previous answer.",
                        "Give a substantive justification and end with: Answer: X",
                    ]
                ),
                "question_options_independent_resample_no_peer_v1",
            )
        review_instruction = (
            "Identify the strongest competing option and test why it fails. "
            "If your current choice cannot be defended, revise it."
            if counterfactual
            else "Independently re-check your argument against the question and options."
        )
        return (
            "\n".join(
                [
                    str(question_text or "").strip(),
                    "",
                    "[Your Previous Answer]",
                    str(previous_response or "").strip(),
                    "",
                    review_instruction,
                    "Revise unsupported assumptions or contradictions, but do not rely on peer answers.",
                    "Give a concise justification and end with: Answer: X",
                ]
            ),
            (
                "own_previous_response_counterfactual_no_peer_v1"
                if counterfactual
                else "own_previous_response_no_peer_v1"
            ),
        )

    @staticmethod
    def _is_substantive_reflection_candidate(
        response_text: str,
        claims: List[Dict[str, Any]],
    ) -> bool:
        response_body = re.sub(
            r"(?im)^\s*(?:final\s+)?answer\s*:\s*[A-J]\s*[.\s]*$",
            "",
            str(response_text or ""),
        ).strip()
        reasoning_tokens = re.findall(r"\b[\w'-]+\b", response_body, flags=re.UNICODE)
        valid_claims = [
            claim
            for claim in claims or []
            if isinstance(claim, dict)
            and bool(claim.get("metric_valid", False))
            and str(claim.get("text", "") or "").strip()
        ]
        return bool(len(reasoning_tokens) >= 8 and valid_claims)

    def _execute_response_only_self_reflection(
        self,
        *,
        node_id: int,
        agent: Agent,
        question_text: str,
        high_risk_threshold: float,
        effective_risk_gate: float,
        action_event: Dict[str, Any],
        counterfactual: bool = False,
        independent_resample: bool = False,
    ) -> None:
        nodes_data = self.graph_dict.get("nodes", [])
        node_data = (
            nodes_data[node_id]
            if isinstance(nodes_data, list)
            and node_id < len(nodes_data)
            and isinstance(nodes_data[node_id], dict)
            else None
        )
        old_role_text = ""
        old_response = ""
        if node_data is not None:
            old_role_text = str(
                node_data.get("role_text")
                or node_data.get("raw_role_text")
                or ""
            )
            old_response = str(node_data.get("response") or "")

        baseline_pi = float(getattr(agent, "pi", 0.0) or 0.0)
        baseline_rho = float(getattr(agent, "rho", 0.0) or 0.0)
        baseline_h = float(getattr(agent, "h_final", 0.0) or 0.0)
        baseline_pi_list = [
            float(getattr(self.agents[idx], "pi", 0.0) or 0.0)
            for idx in sorted(self.agents)
        ]
        baseline_rho_list = [
            float(getattr(self.agents[idx], "rho", 0.0) or 0.0)
            for idx in sorted(self.agents)
        ]
        baseline_structural_fitness = self._compute_agent_structural_fitness(
            node_id, baseline_pi_list, baseline_rho_list
        )

        candidate_response = old_response
        candidate_pi = baseline_pi
        candidate_rho = baseline_rho
        candidate_h = baseline_h
        candidate_structural_fitness = baseline_structural_fitness
        candidate_valid = False
        candidate_substantive = False
        invalid_candidate_preserved = False
        previous_answer = self._parse_choice_answer(old_response)
        candidate_answer = None
        error = ""
        generation_usage: Dict[str, Any] = {}

        try:
            generation_question, context_mode = self._build_response_only_self_reflection_question(
                question_text,
                old_response,
                counterfactual=counterfactual,
                independent_resample=independent_resample,
            )
            prompt_bundle = self._compose_runtime_system_prompt(agent, old_role_text)
            candidate_system_prompt = str(
                prompt_bundle.get("composed_system_prompt") or old_role_text
            )
            raw_candidate, generation_usage = self.generate_response_sync(
                candidate_system_prompt,
                generation_question,
            )
            candidate_response = str(raw_candidate or "")
            response_update = self._resolve_invalid_response_update(
                previous_response=old_response,
                candidate_response=candidate_response,
            )
            candidate_answer = response_update.get("candidate_answer")
            candidate_valid = candidate_answer is not None
            invalid_candidate_preserved = bool(
                response_update.get("invalid_candidate_preserved", False)
            )
            if candidate_valid:
                if node_data is not None:
                    node_data["response"] = candidate_response
                    node_data["composed_role_prompt"] = candidate_system_prompt
                self.latest_agent_responses[int(node_id)] = candidate_response

                candidate_state = self._estimate_risk_state()
                candidate_pi = float(candidate_state.pi.get(node_id, baseline_pi) or 0.0)
                candidate_rho = float(candidate_state.rho.get(node_id, baseline_rho) or 0.0)
                candidate_h = float(
                    candidate_state.aggregate_risk.get(node_id, baseline_h) or 0.0
                )
                per_agent = (candidate_state.details or {}).get("per_agent", {})
                candidate_details = per_agent.get(node_id, per_agent.get(str(node_id), {}))
                candidate_claims = (
                    candidate_details.get("claims", [])
                    if isinstance(candidate_details, dict)
                    else []
                )
                candidate_substantive = self._is_substantive_reflection_candidate(
                    candidate_response,
                    candidate_claims,
                )
                candidate_pi_list = [
                    float(candidate_state.pi.get(idx, 0.0) or 0.0)
                    for idx in sorted(self.agents)
                ]
                candidate_rho_list = [
                    float(candidate_state.rho.get(idx, 0.0) or 0.0)
                    for idx in sorted(self.agents)
                ]
                candidate_structural_fitness = self._compute_agent_structural_fitness(
                    node_id, candidate_pi_list, candidate_rho_list
                )
        except Exception as exc:
            context_mode = (
                "question_options_independent_resample_no_peer_v1"
                if independent_resample
                else (
                    "own_previous_response_counterfactual_no_peer_v1"
                    if counterfactual
                    else "own_previous_response_no_peer_v1"
                )
            )
            error = str(exc)

        answer_changed = bool(
            previous_answer is not None
            and candidate_answer is not None
            and previous_answer != candidate_answer
        )
        required_pi_margin = float(
            getattr(
                self.config.egt,
                "self_v2_pi_margin_changed_answer"
                if answer_changed
                else "self_v2_pi_margin_same_answer",
                0.05 if answer_changed else 0.02,
            )
            or 0.0
        )
        rho_tolerance = float(
            getattr(self.config.egt, "self_v2_rho_tolerance", 0.02) or 0.0
        )
        accepted = bool(
            candidate_valid
            and candidate_substantive
            and candidate_pi <= (baseline_pi - required_pi_margin + 1e-9)
            and (
                independent_resample
                or candidate_rho <= (baseline_rho + rho_tolerance + 1e-9)
            )
        )
        if not accepted:
            if node_data is not None:
                node_data["response"] = old_response
            self.latest_agent_responses[int(node_id)] = old_response

        agent.reflected_this_gen = True
        response_changed = bool(
            accepted and str(candidate_response).strip() != str(old_response).strip()
        )
        action_event["events"].append(
            {
                "type": "self_reflect",
                "reflection_mode": (
                    "independent_resample_intrinsic_v1"
                    if independent_resample
                    else (
                        "response_only_counterfactual_v1"
                        if counterfactual
                        else "response_only_intrinsic_v1"
                    )
                ),
                "counterfactual_option_check": bool(counterfactual),
                "independent_resample": bool(independent_resample),
                "acceptance_scope": (
                    "intrinsic_pi_only"
                    if independent_resample
                    else "intrinsic_pi_with_rho_tolerance"
                ),
                "generation_context_mode": context_mode,
                "candidate_response_valid": candidate_valid,
                "invalid_candidate_preserved": invalid_candidate_preserved,
                "candidate_substantive_argument": candidate_substantive,
                "role_update_mode": "preserve_original_role",
                "role_rewrite_model": "",
                "role_rewrite_prompt_version": "not_used_response_only",
                "role_rewrite_valid": False,
                "role_rewrite_from_cache": False,
                "role_rewrite_error": error,
                "role_rewrite_prompt_tokens": 0,
                "role_rewrite_completion_tokens": 0,
                "reflection_generation_prompt_tokens": int(
                    generation_usage.get("prompt_tokens", 0) or 0
                ),
                "reflection_generation_completion_tokens": int(
                    generation_usage.get("completion_tokens", 0) or 0
                ),
                "role_state_changed_before_rewrite": False,
                "trigger_reason": "agent_action",
                "high_risk_threshold_q85": float(high_risk_threshold),
                "effective_self_reflect_risk_gate": float(effective_risk_gate),
                "agent_h_final": baseline_h,
                "baseline_node_pi": baseline_pi,
                "candidate_node_pi": candidate_pi,
                "baseline_node_rho": baseline_rho,
                "candidate_node_rho": candidate_rho,
                "baseline_node_h": baseline_h,
                "candidate_node_h": candidate_h,
                "old_role_text": old_role_text,
                "new_role_text": old_role_text,
                "old_response": old_response,
                "new_response": candidate_response,
                "changed_role_fields": [],
                "role_changed": False,
                "role_text_changed": False,
                "candidate_role_text_changed": False,
                "response_changed": response_changed,
                "candidate_response_changed": bool(
                    str(candidate_response).strip() != str(old_response).strip()
                ),
                "risk_reduced": bool(candidate_pi < baseline_pi),
                "old_tags": sorted(agent.tags),
                "new_tags": sorted(agent.tags),
                "changed": response_changed,
                "accepted": accepted,
                "answer_changed": answer_changed,
                "required_pi_margin": required_pi_margin,
                "rho_tolerance": rho_tolerance,
                "baseline_structural_fitness": float(baseline_structural_fitness),
                "candidate_structural_fitness": float(candidate_structural_fitness),
            }
        )

    def _build_paper_aligned_self_reflection_question(
        self,
        *,
        node_id: int,
        question_text: str,
        response_snapshot: Dict[int, str],
        pi: float,
        rho: float,
    ) -> tuple[str, str]:
        base_question = self._build_peer_augmented_question(
            node_id,
            question_text,
            response_snapshot,
        )
        return (
            "\n".join(
                [
                    base_question.rstrip(),
                    "",
                    "[Self-Reflection]",
                    f"Reference-free risk signals: pi={pi:.4f}, rho={rho:.4f}.",
                    "Under the same agent role, re-check the current response and revise it only as needed.",
                    "End with a final line exactly as: Answer: X",
                ]
            ),
            "paper_current_context_and_risk_shared_snapshot_v1",
        )

    def _restore_response_snapshot(self, response_snapshot: Dict[int, str]) -> None:
        nodes = self.graph_dict.get("nodes", [])
        if isinstance(nodes, list):
            for node_id, response_text in response_snapshot.items():
                if node_id < len(nodes) and isinstance(nodes[node_id], dict):
                    nodes[node_id]["response"] = str(response_text or "")
        self.latest_agent_responses = {
            int(node_id): str(response_text or "")
            for node_id, response_text in response_snapshot.items()
        }

    def _execute_paper_aligned_self_reflections(
        self,
        *,
        question_text: str,
        high_risk_threshold: float,
        latest_action_events: Dict[int, Dict[str, Any]],
    ) -> set[int]:
        """Evaluate Self candidates independently, then commit accepted responses together."""

        baseline_state = self.latest_risk_state
        if baseline_state is None:
            return set()
        response_snapshot = self._snapshot_node_responses()
        nodes = self.graph_dict.get("nodes", [])
        pending: List[Dict[str, Any]] = []
        attempted: set[int] = set()

        for node_id in sorted(self.agents):
            agent = self.agents[node_id]
            if int(getattr(agent, "a_self", 0) or 0) != 1:
                continue
            attempted.add(int(node_id))
            agent.reflected_this_gen = True
            node = (
                nodes[node_id]
                if isinstance(nodes, list)
                and node_id < len(nodes)
                and isinstance(nodes[node_id], dict)
                else None
            )
            old_response = str(response_snapshot.get(node_id, "") or "")
            old_role_text = str(
                (
                    node.get("role_text")
                    or node.get("raw_role_text")
                    or node.get("original_role_text")
                    or ""
                )
                if node is not None
                else ""
            )
            old_tags = sorted(agent.tags)
            baseline_pi = float(baseline_state.pi.get(node_id, agent.pi) or 0.0)
            baseline_rho = float(baseline_state.rho.get(node_id, agent.rho) or 0.0)
            baseline_h = float(
                baseline_state.aggregate_risk.get(node_id, agent.h_final) or 0.0
            )
            baseline_utility = self._compute_agent_structural_fitness_from_risk_state(
                node_id,
                baseline_state,
            )
            candidate_response = old_response
            candidate_pi = baseline_pi
            candidate_rho = baseline_rho
            candidate_h = baseline_h
            candidate_utility = baseline_utility
            candidate_valid = False
            candidate_substantive = False
            candidate_system_prompt = old_role_text
            generation_usage: Dict[str, Any] = {}
            error = ""
            context_mode = "paper_current_context_and_risk_shared_snapshot_v1"

            try:
                generation_question, context_mode = (
                    self._build_paper_aligned_self_reflection_question(
                        node_id=node_id,
                        question_text=question_text,
                        response_snapshot=response_snapshot,
                        pi=baseline_pi,
                        rho=baseline_rho,
                    )
                )
                prompt_bundle = self._compose_runtime_system_prompt(agent, old_role_text)
                candidate_system_prompt = str(
                    prompt_bundle.get("composed_system_prompt") or old_role_text
                )
                raw_candidate, generation_usage = self.generate_response_sync(
                    candidate_system_prompt,
                    generation_question,
                )
                candidate_response = str(raw_candidate or "")
                candidate_valid = self._parse_choice_answer(candidate_response) is not None
                if candidate_valid:
                    self._restore_response_snapshot(response_snapshot)
                    if node is not None:
                        node["response"] = candidate_response
                    self.latest_agent_responses[int(node_id)] = candidate_response
                    candidate_state = self._estimate_risk_state()
                    candidate_pi = float(
                        candidate_state.pi.get(node_id, baseline_pi) or 0.0
                    )
                    candidate_rho = float(
                        candidate_state.rho.get(node_id, baseline_rho) or 0.0
                    )
                    candidate_h = float(
                        candidate_state.aggregate_risk.get(node_id, baseline_h) or 0.0
                    )
                    candidate_utility = (
                        self._compute_agent_structural_fitness_from_risk_state(
                            node_id,
                            candidate_state,
                        )
                    )
                    per_agent = (candidate_state.details or {}).get("per_agent", {})
                    candidate_details = per_agent.get(
                        node_id,
                        per_agent.get(str(node_id), {}),
                    )
                    candidate_claims = (
                        candidate_details.get("claims", [])
                        if isinstance(candidate_details, dict)
                        else []
                    )
                    candidate_substantive = self._is_substantive_reflection_candidate(
                        candidate_response,
                        candidate_claims,
                    )
            except Exception as exc:
                error = f"{exc.__class__.__name__}: {exc}"
            finally:
                self._restore_response_snapshot(response_snapshot)

            pi_improved = candidate_pi + 1e-9 < baseline_pi
            utility_improved = candidate_utility > baseline_utility + 1e-9
            accepted = bool(candidate_valid and (pi_improved or utility_improved))
            response_changed = bool(
                accepted and candidate_response.strip() != old_response.strip()
            )
            pending.append(
                {
                    "node_id": int(node_id),
                    "accepted": accepted,
                    "candidate_response": candidate_response,
                    "candidate_system_prompt": candidate_system_prompt,
                }
            )
            latest_action_events[node_id]["events"].append(
                {
                    "type": "self_reflect",
                    "reflection_mode": "paper_aligned_synchronous_v1",
                    "generation_context_mode": context_mode,
                    "candidate_response_valid": candidate_valid,
                    "invalid_candidate_preserved": not candidate_valid,
                    "candidate_substantive_argument": candidate_substantive,
                    "role_update_mode": "preserve_original_role",
                    "role_rewrite_model": "",
                    "role_rewrite_prompt_version": "not_used_paper_aligned",
                    "role_rewrite_valid": False,
                    "role_rewrite_from_cache": False,
                    "role_rewrite_error": error,
                    "role_rewrite_prompt_tokens": 0,
                    "role_rewrite_completion_tokens": 0,
                    "reflection_generation_prompt_tokens": int(
                        generation_usage.get("prompt_tokens", 0) or 0
                    ),
                    "reflection_generation_completion_tokens": int(
                        generation_usage.get("completion_tokens", 0) or 0
                    ),
                    "trigger_reason": "agent_action",
                    "risk_gate_applied": False,
                    "high_risk_threshold_q85": float(high_risk_threshold),
                    "effective_self_reflect_risk_gate": 0.0,
                    "candidate_evaluation_snapshot": "shared_pre_action_round_snapshot_v1",
                    "synchronous_commit": True,
                    "acceptance_scope": "paper_pi_or_structural_utility",
                    "acceptance_pi_improved": bool(pi_improved),
                    "acceptance_utility_improved": bool(utility_improved),
                    "baseline_node_pi": baseline_pi,
                    "candidate_node_pi": candidate_pi,
                    "baseline_node_rho": baseline_rho,
                    "candidate_node_rho": candidate_rho,
                    "baseline_node_h": baseline_h,
                    "candidate_node_h": candidate_h,
                    "old_role_text": old_role_text,
                    "new_role_text": old_role_text,
                    "old_response": old_response,
                    "new_response": candidate_response,
                    "changed_role_fields": [],
                    "role_changed": False,
                    "role_text_changed": False,
                    "candidate_role_text_changed": False,
                    "response_changed": response_changed,
                    "candidate_response_changed": bool(
                        candidate_response.strip() != old_response.strip()
                    ),
                    "risk_reduced": bool(pi_improved),
                    "old_tags": old_tags,
                    "new_tags": old_tags,
                    "changed": response_changed,
                    "accepted": accepted,
                    "answer_changed": bool(
                        self._parse_choice_answer(old_response)
                        != self._parse_choice_answer(candidate_response)
                    ),
                    "required_pi_margin": 0.0,
                    "rho_tolerance": None,
                    "baseline_structural_fitness": float(baseline_utility),
                    "candidate_structural_fitness": float(candidate_utility),
                }
            )

        self._restore_response_snapshot(response_snapshot)
        for candidate in pending:
            if not bool(candidate["accepted"]):
                continue
            node_id = int(candidate["node_id"])
            response_text = str(candidate["candidate_response"] or "")
            if isinstance(nodes, list) and node_id < len(nodes) and isinstance(nodes[node_id], dict):
                nodes[node_id]["response"] = response_text
                nodes[node_id]["composed_role_prompt"] = str(
                    candidate["candidate_system_prompt"] or ""
                )
            self.latest_agent_responses[node_id] = response_text
        return attempted

    @staticmethod
    def _safe_ratio(numer: float, denom: float) -> float:
        if abs(float(denom)) <= 1e-8:
            return 0.0
        return float(numer) / float(denom)

    def _option_text_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        choices = self.graph_dict.get("choices")
        if isinstance(choices, list):
            for idx, choice in enumerate(choices):
                if idx >= 10:
                    break
                label = chr(ord("A") + idx)
                mapping[label] = str(choice or "").strip()
        if mapping:
            return mapping

        question_text = str(self.graph_dict.get("question_text", "") or "")
        for match in re.finditer(r"(?m)^([A-J])\.\s*(.+)$", question_text):
            mapping[str(match.group(1)).strip()] = str(match.group(2)).strip()
        return mapping

    def _compute_answer_choice_utility_for_response(self, response_text: str) -> Dict[str, Any]:
        response_text = str(response_text or "")
        parsed_answer = self._parse_choice_answer(response_text)
        option_map = self._option_text_map()
        option_text = str(option_map.get(parsed_answer, "") or "")
        normalized_response = response_text.lower()
        normalized_option = option_text.lower()
        explicit_choice = parsed_answer is not None
        evidence_markers = (
            "because",
            "since",
            "therefore",
            "thus",
            "statement 1",
            "statement 2",
            "according to",
            "based on",
            "not given",
            "true",
            "false",
        )
        evidence_supported = False
        if explicit_choice:
            evidence_supported = any(marker in normalized_response for marker in evidence_markers)
            if (not evidence_supported) and normalized_option:
                option_terms = [term for term in re.findall(r"[a-zA-Z]{4,}", normalized_option) if len(term) >= 4]
                if option_terms:
                    overlap = sum(1 for term in option_terms if term in normalized_response)
                    evidence_supported = overlap >= max(1, min(2, len(option_terms)))
        distractor_checked = bool(
            re.search(r"\b(?:rather than|instead of|not option|compared with|vs\.?|versus)\b", normalized_response)
        )
        if (not distractor_checked) and explicit_choice:
            distinct_letters = set(re.findall(r"\b([A-J])\b", response_text.upper()))
            distractor_checked = len(distinct_letters) >= 2
        reasoning_consistent = bool(
            explicit_choice
            and re.search(r"(?:final answer|answer|final option|答案)[^A-J]{0,8}" + re.escape(parsed_answer), response_text, re.IGNORECASE)
        )
        unsupported_option_claim = bool(explicit_choice and len(response_text.strip()) < 32)
        if explicit_choice and ("choose " in normalized_response or "computed it precisely" in normalized_response):
            unsupported_option_claim = unsupported_option_claim or (not evidence_supported and len(response_text.strip()) < 80)

        utility = 0.0
        utility += 0.30 if explicit_choice else 0.0
        utility += 0.25 if evidence_supported else 0.0
        utility += 0.20 if distractor_checked else 0.0
        utility += 0.15 if reasoning_consistent else 0.0
        utility -= 0.20 if unsupported_option_claim else 0.0
        utility = max(0.0, min(1.0, utility))
        return {
            "parsed_answer": parsed_answer,
            "option_text": option_text,
            "explicit_choice": bool(explicit_choice),
            "evidence_supported": bool(evidence_supported),
            "distractor_checked": bool(distractor_checked),
            "reasoning_consistent": bool(reasoning_consistent),
            "unsupported_option_claim": bool(unsupported_option_claim),
            "answer_choice_utility": float(utility),
        }

    def _build_answer_choice_analysis(
        self,
        *,
        pi_list: List[float] | None = None,
        rho_list: List[float] | None = None,
        h_list: List[float] | None = None,
    ) -> Dict[str, Any]:
        pi_list = list(pi_list or [])
        rho_list = list(rho_list or [])
        h_list = list(h_list or [])
        nodes = self.graph_dict.get("nodes", [])
        per_agent: List[Dict[str, Any]] = []
        parsed_answers: List[str] = []
        for node_id in sorted(self.agents.keys()):
            response_text = ""
            if isinstance(nodes, list) and node_id < len(nodes) and isinstance(nodes[node_id], dict):
                response_text = str(nodes[node_id].get("response") or nodes[node_id].get("role_text") or "").strip()
            analysis = self._compute_answer_choice_utility_for_response(response_text)
            parsed = analysis.get("parsed_answer")
            if parsed:
                parsed_answers.append(str(parsed))
            per_agent.append(
                {
                    "agent_id": int(node_id),
                    "response": response_text,
                    "parsed_answer": parsed,
                    "pi": float(pi_list[node_id]) if node_id < len(pi_list) else float(getattr(self.agents[node_id], "pi", 0.0) or 0.0),
                    "rho": float(rho_list[node_id]) if node_id < len(rho_list) else float(getattr(self.agents[node_id], "rho", 0.0) or 0.0),
                    "h_final": float(h_list[node_id]) if node_id < len(h_list) else float(getattr(self.agents[node_id], "h_final", 0.0) or 0.0),
                    **analysis,
                }
            )

        majority_answer = majority_choice(parsed_answers)
        parse_fail_rate = 1.0 - self._safe_ratio(len(parsed_answers), max(1, len(per_agent)))
        avg_utility = (
            sum(float(item.get("answer_choice_utility", 0.0) or 0.0) for item in per_agent) / max(1, len(per_agent))
        )
        counts: Dict[str, int] = {}
        for answer in parsed_answers:
            counts[answer] = counts.get(answer, 0) + 1
        majority_count = int(counts.get(majority_answer, 0) or 0) if majority_answer else 0
        majority_ratio = self._safe_ratio(majority_count, max(1, len(per_agent)))
        return {
            "per_agent": per_agent,
            "majority_answer": majority_answer,
            "majority_count": majority_count,
            "majority_ratio": float(majority_ratio),
            "avg_answer_choice_utility": float(avg_utility),
            "parse_fail_rate": float(parse_fail_rate),
        }

    def _build_consensus_analysis(self, answer_analysis: Dict[str, Any]) -> Dict[str, Any]:
        per_agent = list(answer_analysis.get("per_agent", []) or [])
        majority_answer = answer_analysis.get("majority_answer")
        majority_ratio = float(answer_analysis.get("majority_ratio", 0.0) or 0.0)
        majority_agents = [item for item in per_agent if item.get("parsed_answer") == majority_answer and majority_answer]
        dissent_agents = [item for item in per_agent if item.get("parsed_answer") and item.get("parsed_answer") != majority_answer]
        majority_utility = (
            sum(float(item.get("answer_choice_utility", 0.0) or 0.0) for item in majority_agents) / max(1, len(majority_agents))
            if majority_agents else 0.0
        )
        majority_avg_h = (
            sum(float(item.get("h_final", 0.0) or 0.0) for item in majority_agents) / max(1, len(majority_agents))
            if majority_agents else 1.0
        )
        low_risk_dissent = any(float(item.get("h_final", 1.0) or 1.0) <= min(0.25, majority_avg_h + 1e-8) for item in dissent_agents)
        majority_support_weak = majority_utility < 0.55
        # Injection labels are intervention metadata and may reveal that an
        # option is known to be wrong. They are excluded from online diagnostics.
        injected_wrong_adopted = False

        consensus_risk = 0.0
        majority_threshold = float(getattr(self.config.egt, "consensus_majority_threshold", 0.6) or 0.6)
        if majority_ratio >= majority_threshold:
            consensus_risk += 0.25 + 0.25 * min(1.0, (majority_ratio - majority_threshold) / max(1e-8, 1.0 - majority_threshold))
        if majority_support_weak:
            consensus_risk += 0.25 + 0.25 * min(1.0, (0.55 - majority_utility) / 0.55)
        if low_risk_dissent:
            consensus_risk += 0.20
        if injected_wrong_adopted:
            consensus_risk += 0.20
        consensus_risk += 0.10 * float(answer_analysis.get("parse_fail_rate", 0.0) or 0.0)
        consensus_risk = max(0.0, min(1.0, consensus_risk))
        consensus_breaker_triggered = False
        return {
            "consensus_answer": majority_answer,
            "consensus_ratio": float(majority_ratio),
            "consensus_majority_utility": float(majority_utility),
            "consensus_risk": float(consensus_risk),
            "low_risk_dissent_present": bool(low_risk_dissent),
            "majority_support_weak": bool(majority_support_weak),
            "injected_wrong_adopted": bool(injected_wrong_adopted),
            "consensus_breaker_triggered": bool(consensus_breaker_triggered),
        }

    def _compose_runtime_system_prompt(
        self,
        agent: Agent,
        raw_role_text: str | None,
    ) -> dict[str, Any]:
        raw_role_text = str(raw_role_text or "").strip()
        if str(getattr(self.config.egt, "prompt_style", "default")).lower() != "aligned":
            return {
                "raw_role_text": raw_role_text,
                "role_adapter_id": "",
                "role_adapter_text": "",
                "dynamic_role_wrapper": "",
                "risk_status": "unaligned",
                "composed_system_prompt": raw_role_text,
            }
        return {
            "raw_role_text": raw_role_text,
            "role_adapter_id": "",
            "role_adapter_text": "",
            "dynamic_role_wrapper": "",
            "risk_status": "original_only",
            "phenotype_tags": [],
            "phenotype_tags_block": "",
            "composed_system_prompt": build_model_aware_evoguard_system_prompt(self._gen_model(), raw_role_text),
        }

    def _resolve_invalid_response_update(
        self,
        *,
        previous_response: str,
        candidate_response: str,
    ) -> Dict[str, Any]:
        previous_answer = self._parse_choice_answer(previous_response)
        candidate_answer = self._parse_choice_answer(candidate_response)
        fallback_enabled = bool(
            getattr(self.config.egt, "enable_invalid_response_fallback", True)
        )
        preserve_previous = bool(
            fallback_enabled
            and candidate_answer is None
            and previous_answer is not None
        )
        if preserve_previous:
            selected_response = previous_response
            reason = "candidate_invalid_preserve_previous_valid"
        else:
            selected_response = candidate_response
            reason = (
                "valid_candidate_committed"
                if candidate_answer is not None
                else "candidate_invalid_without_valid_previous"
            )
        return {
            "selected_response": selected_response,
            "candidate_accepted": not preserve_previous,
            "reason": reason,
            "previous_answer": previous_answer,
            "candidate_answer": candidate_answer,
            "invalid_candidate_preserved": preserve_previous,
        }

    def process_single_agent(
        self,
        node_id: int,
        agent: Agent,
        current_gen: int,
        question_text: str,
        online_reference_context: str,
        response_snapshot: Dict[int, str] | None = None,
    ) -> None:
        is_observation_gen = self._is_observation_generation(current_gen)
        agent.gen_token_count = 0.0
        agent.judge_token_count = 0.0
        agent.total_token_count = agent.rep_token_count
        current_role_text = ""
        persona_prompt = ""
        nodes_data = self.graph_dict.get("nodes", [])
        node_payload = None
        fixed_role_schema = None
        if isinstance(nodes_data, list) and node_id < len(nodes_data) and isinstance(nodes_data[node_id], dict):
            node_payload = nodes_data[node_id]
            fixed_role_schema = self._extract_fixed_role_schema(node_payload)
            current_role_text = (
                node_payload.get("role_text", "")
                or node_payload.get("raw_role_text", "")
                or ""
            )
            persona_prompt = current_role_text
        if fixed_role_schema is not None and node_payload is not None:
            agent.tags = self._schema_to_tags(fixed_role_schema)
        if not persona_prompt:
            persona_prompt = self.generate_role_text_with_llm(agent.tags, node=node_payload)
            current_role_text = persona_prompt
            if isinstance(nodes_data, list) and node_id < len(nodes_data) and isinstance(nodes_data[node_id], dict):
                nodes_data[node_id]["role_text"] = current_role_text
                nodes_data[node_id].setdefault("raw_role_text", current_role_text)
                nodes_data[node_id].setdefault("original_role_text", current_role_text)
        prompt_bundle = self._compose_runtime_system_prompt(agent, current_role_text or persona_prompt)
        persona_prompt = str(prompt_bundle.get("composed_system_prompt") or persona_prompt)

        generated_text = self.perturbation_engine.apply_perturbation(node_id, current_gen)
        gen_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        generation_error = None
        if generated_text is None:
            try:
                generation_question = self._build_peer_augmented_question(node_id, question_text, response_snapshot)
                if self.fault_controller is not None:
                    peer_messages = self._incoming_peer_messages(
                        node_id,
                        response_snapshot if response_snapshot is not None else {},
                    )
                    self.fault_controller.record_peer_context_evidence(
                        current_round_index=current_gen,
                        recipient_agent_id=node_id,
                        peer_context_text=generation_question,
                        peer_messages_by_sender={
                            int(message["agent_id"]): str(message.get("content") or "")
                            for message in peer_messages
                        },
                    )
                generated_text, gen_usage = self.generate_response_sync(persona_prompt, generation_question)
            except Exception as exc:
                generation_error = f"{type(exc).__name__}: {exc}"
                generated_text = f"Error generating text: {exc}"
        candidate_response = str(generated_text or "")
        previous_response = str((response_snapshot or {}).get(node_id, "") or "")
        response_update = self._resolve_invalid_response_update(
            previous_response=previous_response,
            candidate_response=candidate_response,
        )
        generated_text = str(response_update.pop("selected_response", candidate_response) or "")
        if self.fault_controller is not None:
            generated_text = self.fault_controller.apply_if_scheduled(
                round_index=current_gen,
                agent_id=node_id,
                question_text=question_text,
                response=generated_text,
                parse_answer=self._parse_choice_answer,
            )
        self.latest_response_update_events.append(
            {
                "agent_id": int(node_id),
                "gen": int(current_gen),
                "candidate_response_valid": response_update.get("candidate_answer") is not None,
                "previous_response_valid": response_update.get("previous_answer") is not None,
                "response_changed": generated_text.strip() != previous_response.strip(),
                "generation_failed": generation_error is not None,
                "generation_failure_reason": generation_error,
                **response_update,
            }
        )
        agent.gen_token_count = self._total_tokens_from_usage(gen_usage)

        if is_observation_gen:
            judge_result, judge_prompt_used, judge_raw_response_text, judge_usage = self.judge_text_sync_for_observation(
                question_text,
                generated_text,
                online_reference_context,
            )
            agent.judge_token_count = self._total_tokens_from_usage(judge_usage)
        else:
            judge_result = None
            judge_prompt_used = ""
            judge_raw_response_text = ""
            agent.judge_token_count = 0.0
        agent.total_token_count = agent.gen_token_count + agent.rep_token_count + agent.judge_token_count

        self.latest_agent_responses[int(node_id)] = str(generated_text or "")
        if node_id < len(self.graph_dict["nodes"]):
            self.graph_dict["nodes"][node_id]["response"] = generated_text
            if current_role_text:
                self.graph_dict["nodes"][node_id]["role_text"] = current_role_text
                self.graph_dict["nodes"][node_id].setdefault("raw_role_text", current_role_text)
                self.graph_dict["nodes"][node_id].setdefault("original_role_text", current_role_text)
            self.graph_dict["nodes"][node_id]["role_adapter_id"] = prompt_bundle.get("role_adapter_id", "")
            self.graph_dict["nodes"][node_id]["role_adapter_text"] = prompt_bundle.get("role_adapter_text", "")
            self.graph_dict["nodes"][node_id]["dynamic_role_wrapper"] = prompt_bundle.get("dynamic_role_wrapper", "")
            self.graph_dict["nodes"][node_id]["risk_status"] = prompt_bundle.get("risk_status", "")
            self.graph_dict["nodes"][node_id]["phenotype_tags_block"] = prompt_bundle.get("phenotype_tags_block", "")
            self.graph_dict["nodes"][node_id]["composed_role_prompt"] = persona_prompt

        if is_observation_gen:
            metrics = self._compute_agent_metrics(judge_result)
            is_hallucinating = metrics["atomic_rate"] > 0.0 or bool(judge_result.get("is_refusal", False))

            agent.is_actually_hallucinating = is_hallucinating
            agent.real_hallucination_rate = 1.0 if is_hallucinating else 0.0
            agent.metric_atomic = metrics["atomic_rate"]
            agent.metric_swhi = metrics["swhi"]
            agent.metric_snr = metrics["info_snr"]

            task_description = f"{question_text}\n\nReference Context:\n{online_reference_context}"
            self._write_debug_trace(
                current_gen=current_gen,
                agent_id=node_id,
                task_description=task_description,
                agent_response=generated_text,
                judge_prompt_used=judge_prompt_used,
                judge_raw_response_text=judge_raw_response_text,
                parsed_json=judge_result,
                fitness_score=1.0 - metrics["atomic_rate"],
            )
        else:
            agent.is_actually_hallucinating = False
            agent.real_hallucination_rate = None
            agent.metric_atomic = None
            agent.metric_swhi = None
            agent.metric_snr = None
            self._write_judge_skipped_trace(current_gen=current_gen, agent_id=node_id)

    def run_agent_pipeline(self, current_gen: int, question_text: str, online_reference_context: str) -> None:
        if bool(getattr(self.config.egt, "enable_peer_context", False)):
            self._initialize_responses_from_final_raw_outputs()
        self.latest_response_update_events = []
        response_snapshot = self._snapshot_node_responses()
        if self.fault_controller is not None:
            if current_gen > 0:
                recipients_by_sender: Dict[int, set[int]] = {}
                for edge in self.graph_dict.get("edges", []) or []:
                    if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                        continue
                    sender, receiver = int(edge[0]), int(edge[1])
                    if sender == receiver or not str(response_snapshot.get(sender, "") or "").strip():
                        continue
                    recipients_by_sender.setdefault(sender, set()).add(receiver)
                self.fault_controller.record_next_round_consumption(
                    current_round_index=current_gen,
                    recipient_ids_by_sender=recipients_by_sender,
                )
            self.fault_controller.record_inactive_if_scheduled(
                round_index=current_gen,
                active_agent_ids=set(int(agent_id) for agent_id in self.agents),
            )
        self.latest_generation_input_snapshot = {
            int(agent_id): str(response_text or "")
            for agent_id, response_text in response_snapshot.items()
        }
        for node_id, agent in self.agents.items():
            self.process_single_agent(node_id, agent, current_gen, question_text, online_reference_context, response_snapshot)

    def update_risks_from_estimator(self, *, reset_prompt_samples: bool = False) -> RiskState:
        if reset_prompt_samples:
            self.latest_online_prompt_samples = []
        risk_state = self._estimate_risk_state()
        self._apply_risk_state(risk_state)
        return risk_state

    def update_risks_from_oracle(self) -> None:
        self.update_risks_from_estimator()

    def generate_role_text_with_llm(self, tags: set, node: Dict[str, Any] | None = None) -> str:
        node_payload = dict(node or {})
        cache_key = self._persona_cache_key(tags) + "::" + json.dumps(
            {
                "claimed_domain": node_payload.get("claimed_domain"),
                "target_domain": node_payload.get("target_domain"),
                "axisA": node_payload.get("axisA"),
                "axisB": node_payload.get("axisB"),
                "style_mismatch": bool(node_payload.get("style_mismatch", False)),
                "tamper_self_rules": node_payload.get("tamper_self_rules", []),
                "tamper_social_rules": node_payload.get("tamper_social_rules", []),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if (not self.disable_cache) and cache_key in LLM_ROLE_CACHE:
            return LLM_ROLE_CACHE[cache_key]
        if (not self.disable_cache) and cache_key in self.persona_prompt_cache:
            persona_text = self.persona_prompt_cache[cache_key]
            LLM_ROLE_CACHE[cache_key] = persona_text
            return persona_text

        persona_text = build_interpretable_role_prompt(node_payload, tags=set(tags))

        if not self.disable_cache:
            self.persona_prompt_cache[cache_key] = persona_text
            self._save_persona_prompt_cache()
        if not self.disable_cache:
            LLM_ROLE_CACHE[cache_key] = persona_text
        return persona_text

    @staticmethod
    def _sanitize_rewritten_role_text(raw_text: str) -> str:
        role_text = str(raw_text or "").strip()
        role_text = re.sub(r"^```(?:text|markdown)?\s*", "", role_text, flags=re.IGNORECASE)
        role_text = re.sub(r"\s*```$", "", role_text).strip()
        role_text = re.sub(
            r"^\s*(?:rewritten\s+)?(?:agent\s+)?role\s*:\s*",
            "",
            role_text,
            flags=re.IGNORECASE,
        ).strip().strip("\"'")
        return role_text

    def rewrite_role_from_phenotype_tags(
        self,
        current_role_text: str,
        phenotype_tags: set[str],
    ) -> tuple[str, Dict[str, int], Dict[str, Any]]:
        """Rewrite the active role with the answer model after a self action."""

        current_role = str(current_role_text or "").strip()
        normalized_tags = sorted(str(tag).strip() for tag in phenotype_tags if str(tag).strip())
        model_name = self._gen_model()
        trace: Dict[str, Any] = {
            "prompt_version": ROLE_REWRITE_PROMPT_VERSION,
            "model": model_name,
            "phenotype_tags": normalized_tags,
            "valid": False,
            "from_cache": False,
            "error": "",
        }
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        if not current_role:
            trace["error"] = "missing_current_role"
            return current_role, usage, trace

        cache_key = self._stable_cache_key(
            {
                "kind": "self_role_rewrite",
                "prompt_version": ROLE_REWRITE_PROMPT_VERSION,
                "model": model_name,
                "current_role": current_role,
                "phenotype_tags": normalized_tags,
            }
        )
        if not self.disable_cache:
            cached_role = str(self.persona_prompt_cache.get(cache_key, "") or "").strip()
            if cached_role:
                trace.update({"valid": True, "from_cache": True})
                return cached_role, usage, trace

        if not self.response_client:
            trace["error"] = "generation_client_unavailable"
            return current_role, usage, trace

        system_prompt, user_prompt = build_phenotype_role_rewrite_prompts(
            model_name,
            current_role,
            normalized_tags,
        )

        def _request_role_rewrite():
            response = self._chat_completion(
                role="generation",
                model_name=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max(60, min(int(self.config.llm.persona_max_tokens), 256)),
                temperature=float(self.config.llm.persona_temperature),
                timeout_seconds=60,
            )
            rewritten = self._sanitize_rewritten_role_text(self._extract_chat_content(response))
            if not rewritten:
                raise RuntimeError("empty role rewrite response")
            return response, rewritten

        try:
            response, rewritten_role = self._call_llm_with_retry(
                "rewrite_role_from_phenotype",
                _request_role_rewrite,
            )
        except Exception as exc:
            trace["error"] = f"{exc.__class__.__name__}: {exc}"
            return current_role, usage, trace

        usage_obj = self._extract_usage(response)
        self._accumulate_usage(usage_obj, "generation_prompt", "generation_completion")
        usage = self._usage_dict(usage_obj)
        invalid_reason = ""
        lowered = rewritten_role.lower()
        if len(rewritten_role) > 1200:
            invalid_reason = "role_rewrite_too_long"
        elif "answer:" in lowered:
            invalid_reason = "role_rewrite_contains_answer"
        elif "phenotype tag" in lowered:
            invalid_reason = "role_rewrite_exposes_internal_tags"
        if invalid_reason:
            trace["error"] = invalid_reason
            return current_role, usage, trace

        trace["valid"] = True
        if not self.disable_cache:
            self.persona_prompt_cache[cache_key] = rewritten_role
            self._save_persona_prompt_cache()
        return rewritten_role, usage, trace

    def _compute_agent_structural_fitness(self, node_id: int, pi_list, rho_list) -> float:
        beta = self._resolve_edge_beta(
            transmission_matrix=self.latest_transmission_matrix,
            pi_list=list(pi_list or []),
            rho_list=list(rho_list or []),
        )
        in_neighbors_map = {nid: [] for nid in self.agents.keys()}
        for edge in self.graph_dict.get("edges", []):
            if len(edge) >= 2:
                in_neighbors_map[int(edge[1])].append(int(edge[0]))

        pi_i = float(pi_list[node_id])
        rho_i = float(rho_list[node_id])
        structural_fitness = 1.0 - pi_i
        interaction_utility = 0.0

        for src in in_neighbors_map.get(node_id, []):
            if src not in self.agents:
                continue
            neighbor_pi = self._edge_source_pi(src, node_id, float(pi_list[src]))
            edge_t = self._safe_edge_t(self.latest_transmission_matrix, src, node_id, default_if_missing=0.0)
            marginal_benefit = 1.0 - neighbor_pi
            marginal_cost = beta * edge_t * neighbor_pi * (1.0 + rho_i)
            interaction_utility += marginal_benefit - marginal_cost

        in_degree = len(in_neighbors_map.get(node_id, []))
        if in_degree > 0:
            interaction_utility /= in_degree

        structural_fitness += interaction_utility
        structural_fitness -= beta * rho_i
        return structural_fitness

    def _compute_agent_structural_fitness_from_risk_state(
        self,
        node_id: int,
        risk_state: RiskState,
    ) -> float:
        """Compute structural utility without mutating the active round state."""

        node_ids = sorted(self.agents)
        size = len(node_ids)
        pi_list = [clamp01(risk_state.pi.get(idx, 0.0)) for idx in node_ids]
        rho_list = [clamp01(risk_state.rho.get(idx, 0.0)) for idx in node_ids]
        source_pi_list = [
            clamp01(risk_state.source_pi.get(idx, risk_state.pi.get(idx, 0.0)))
            for idx in node_ids
        ]
        transmission_matrix = self._transmission_matrix_from_state(risk_state, size)
        edge_transmissions = self._build_edge_transmissions(
            transmission_matrix,
            pi_list,
            rho_list,
            source_pi_list=source_pi_list,
            risk_state=risk_state,
        )
        beta = float(
            self._compute_edge_risk_ratio_stats(
                edge_transmissions=edge_transmissions,
                transmission_matrix=transmission_matrix,
                pi_list=pi_list,
                rho_list=rho_list,
            )["edge_beta_used"]
        )
        pi_i = clamp01(risk_state.pi.get(node_id, 0.0))
        rho_i = clamp01(risk_state.rho.get(node_id, 0.0))
        interaction_utility = 0.0
        in_degree = 0
        per_edge = dict((risk_state.details or {}).get("per_edge", {}) or {})
        for edge in self.graph_dict.get("edges", []) or []:
            if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                continue
            src, dst = int(edge[0]), int(edge[1])
            if dst != int(node_id) or src not in self.agents:
                continue
            edge_detail = dict(per_edge.get(f"{src}->{dst}") or {})
            source_pi = clamp01(
                edge_detail.get(
                    "source_message_pi",
                    risk_state.source_pi.get(src, risk_state.pi.get(src, 0.0)),
                )
            )
            edge_t = self._safe_edge_t(
                transmission_matrix,
                src,
                dst,
                default_if_missing=0.0,
            )
            interaction_utility += (1.0 - source_pi) - beta * edge_t * source_pi * (1.0 + rho_i)
            in_degree += 1
        if in_degree:
            interaction_utility /= in_degree
        return (1.0 - pi_i) - beta * rho_i + interaction_utility

    def _current_claims_for_agent(self, node_id: int) -> List[Dict[str, Any]]:
        details = (self.latest_risk_state.details or {}) if self.latest_risk_state else {}
        per_agent = details.get("per_agent", {}) if isinstance(details, dict) else {}
        agent_details = per_agent.get(node_id, per_agent.get(str(node_id), {}))
        claims = agent_details.get("claims", []) if isinstance(agent_details, dict) else []
        return [dict(claim) for claim in claims if isinstance(claim, dict)]

    def _select_targeted_self_claim(self, node_id: int) -> Dict[str, Any] | None:
        valid_claims = [
            claim
            for claim in self._current_claims_for_agent(node_id)
            if bool(claim.get("metric_valid", False))
            and str(claim.get("text", "") or "").strip()
        ]
        if not valid_claims:
            return None
        return max(
            valid_claims,
            key=lambda claim: (
                float(claim.get("risk", 0.0) or 0.0)
                * float(claim.get("importance", 0.0) or 0.0),
                float(claim.get("importance", 0.0) or 0.0),
                float(claim.get("risk", 0.0) or 0.0),
            ),
        )

    @staticmethod
    def _targeted_self_policy(claim: Dict[str, Any]) -> Dict[str, str]:
        status = str(claim.get("support_status", "uncertain") or "uncertain").lower()
        if status == "contradicted":
            return {
                "diagnosis": "explicit_constraint_conflict",
                "tag": "Literalist",
                "instruction": "Check the selected option against every explicit question constraint before committing.",
            }
        if status == "unsupported":
            return {
                "diagnosis": "unsupported_key_premise",
                "tag": "Evidence_Seeking",
                "instruction": "Verify that each premise essential to the selected option is supported by the question.",
            }
        if status == "uncertain":
            return {
                "diagnosis": "unresolved_critical_assumption",
                "tag": "Hypothesis_Testing",
                "instruction": "Test the main unresolved assumption against the competing options before committing.",
            }
        return {
            "diagnosis": "internal_consistency_gap",
            "tag": "Consistency_Checker",
            "instruction": "Check that the central conclusion follows consistently from the stated reasoning.",
        }

    @staticmethod
    def _replace_targeted_self_policy(role_text: str, instruction: str) -> str:
        marker = "[Targeted self-check]"
        base = re.sub(
            rf"\n*{re.escape(marker)}\n[^\n]*\s*$",
            "",
            str(role_text or "").rstrip(),
            flags=re.IGNORECASE,
        )
        return f"{base}\n{marker}\n{instruction}".strip()

    def _restore_deferred_self_role(self, node_id: int, pending: Dict[str, Any]) -> None:
        nodes = self.graph_dict.get("nodes", [])
        if not isinstance(nodes, list) or node_id >= len(nodes) or not isinstance(nodes[node_id], dict):
            return
        node = nodes[node_id]
        old_schema = pending.get("old_fixed_role_schema")
        if isinstance(old_schema, dict):
            self._apply_fixed_role_schema(node, old_schema, increment_revision=False)
        node["role_text"] = str(pending.get("old_role_text", "") or "")
        self.agents[node_id].tags = set(pending.get("old_tags", []))

    def _finalize_deferred_self_role_updates(
        self,
        latest_action_events: Dict[int, Dict[str, Any]],
        current_gen: int,
    ) -> set[int]:
        evaluated_nodes: set[int] = set()
        required_margin = float(
            getattr(self.config.egt, "self_v4_delayed_pi_margin", 0.02) or 0.0
        )
        details = (self.latest_risk_state.details or {}) if self.latest_risk_state else {}
        per_agent = details.get("per_agent", {}) if isinstance(details, dict) else {}

        for node_id, pending in list(self.pending_self_role_updates.items()):
            if current_gen <= int(pending.get("applied_gen", -1)):
                continue
            evaluated_nodes.add(int(node_id))
            agent_details = per_agent.get(node_id, per_agent.get(str(node_id), {}))
            metric_valid = bool(
                isinstance(agent_details, dict)
                and agent_details.get("pi_metric_valid", agent_details.get("metric_valid", False))
            )
            baseline_pi = float(pending.get("baseline_pi", 0.0) or 0.0)
            observed_pi = float(getattr(self.agents[node_id], "pi", baseline_pi) or 0.0)
            delayed_pi_reduction = baseline_pi - observed_pi
            retained = bool(metric_valid and delayed_pi_reduction >= required_margin - 1e-9)
            if not retained:
                self._restore_deferred_self_role(node_id, pending)

            nodes = self.graph_dict.get("nodes", [])
            if isinstance(nodes, list) and node_id < len(nodes) and isinstance(nodes[node_id], dict):
                nodes[node_id]["targeted_self_status"] = "retained" if retained else "rolled_back"
                nodes[node_id]["targeted_self_delayed_pi_reduction"] = float(delayed_pi_reduction)
                nodes[node_id]["role_rewrite_accepted"] = bool(retained)

            latest_action_events[node_id]["events"].append(
                {
                    "type": "self_reflect_evaluation",
                    "reflection_mode": "targeted_deferred_intrinsic_v1",
                    "applied_gen": int(pending.get("applied_gen", current_gen - 1)),
                    "evaluated_gen": int(current_gen),
                    "baseline_node_pi": float(baseline_pi),
                    "candidate_node_pi": float(observed_pi),
                    "required_pi_margin": float(required_margin),
                    "delayed_pi_reduction": float(delayed_pi_reduction),
                    "risk_metric_valid": bool(metric_valid),
                    "risk_reduced": bool(delayed_pi_reduction > 1e-9),
                    "accepted": bool(retained),
                    "retained": bool(retained),
                    "rollback_reason": "" if retained else (
                        "risk_metric_invalid" if not metric_valid else "insufficient_delayed_pi_reduction"
                    ),
                    "target_claim_id": str(pending.get("target_claim_id", "")),
                    "target_support_status": str(pending.get("target_support_status", "")),
                    "target_diagnosis": str(pending.get("target_diagnosis", "")),
                    "target_behavior_tag": str(pending.get("target_behavior_tag", "")),
                    "response_changed": False,
                    "role_changed": bool(pending.get("role_changed", False)),
                    "changed": bool(retained),
                }
            )
            del self.pending_self_role_updates[node_id]
        return evaluated_nodes

    def _schedule_targeted_deferred_self_reflection(
        self,
        *,
        node_id: int,
        agent: Agent,
        current_gen: int,
        high_risk_threshold: float,
        effective_risk_gate: float,
        action_event: Dict[str, Any],
    ) -> None:
        target_claim = self._select_targeted_self_claim(node_id)
        baseline_pi = float(getattr(agent, "pi", 0.0) or 0.0)
        min_pi = float(getattr(self.config.egt, "self_v4_min_pi", 0.25) or 0.0)
        min_claim_risk = float(
            getattr(self.config.egt, "self_v4_min_claim_risk", 0.50) or 0.0
        )
        claim_risk = float(target_claim.get("risk", 0.0) or 0.0) if target_claim else 0.0
        if target_claim is None or baseline_pi < min_pi or claim_risk < min_claim_risk:
            action_event["events"].append(
                {
                    "type": "self_reflect_blocked",
                    "blocked_reason": "no_valid_high_risk_target_claim",
                    "reflection_mode": "targeted_deferred_intrinsic_v1",
                    "node_risk": float(baseline_pi),
                    "minimum_pi": float(min_pi),
                    "target_claim_risk": float(claim_risk),
                    "minimum_claim_risk": float(min_claim_risk),
                }
            )
            return

        nodes = self.graph_dict.get("nodes", [])
        if not isinstance(nodes, list) or node_id >= len(nodes) or not isinstance(nodes[node_id], dict):
            action_event["events"].append(
                {
                    "type": "self_reflect_blocked",
                    "blocked_reason": "missing_node_role_state",
                    "reflection_mode": "targeted_deferred_intrinsic_v1",
                }
            )
            return

        node = nodes[node_id]
        old_role_text = str(node.get("role_text") or node.get("raw_role_text") or "")
        old_tags = sorted(agent.tags)
        old_schema = self._extract_fixed_role_schema(node)
        policy = self._targeted_self_policy(target_claim)
        new_schema = copy.deepcopy(old_schema)
        if new_schema is not None:
            new_schema["self_check"] = "strict"
            new_schema["heuristic_type"] = "stable"
            new_schema["heuristic_text"] = policy["instruction"]
            new_role_text = self._schema_to_role_text(new_schema)
        else:
            new_role_text = self._replace_targeted_self_policy(
                old_role_text,
                policy["instruction"],
            )

        role_changed = str(new_role_text).strip() != str(old_role_text).strip()
        if not role_changed:
            action_event["events"].append(
                {
                    "type": "self_reflect_blocked",
                    "blocked_reason": "target_policy_already_active",
                    "reflection_mode": "targeted_deferred_intrinsic_v1",
                    "target_claim_id": str(target_claim.get("claim_id", "")),
                    "target_diagnosis": policy["diagnosis"],
                }
            )
            return

        if new_schema is not None:
            self._apply_fixed_role_schema(node, new_schema, increment_revision=True)
            agent.tags = self._schema_to_tags(new_schema) | {policy["tag"]}
        else:
            node["role_text"] = new_role_text
            targeted_tags = {
                "Evidence_Seeking",
                "Literalist",
                "Hypothesis_Testing",
                "Consistency_Checker",
            }
            agent.tags = (set(agent.tags) - targeted_tags) | {policy["tag"]}

        node["targeted_self_status"] = "pending_next_round_evaluation"
        node["targeted_self_applied_gen"] = int(current_gen)
        node["targeted_self_target_claim_id"] = str(target_claim.get("claim_id", ""))
        node["targeted_self_target_diagnosis"] = policy["diagnosis"]
        node["targeted_self_behavior_tag"] = policy["tag"]
        node["targeted_self_pending_role_text"] = str(node.get("role_text") or new_role_text)

        self.pending_self_role_updates[node_id] = {
            "applied_gen": int(current_gen),
            "baseline_pi": float(baseline_pi),
            "old_role_text": old_role_text,
            "old_fixed_role_schema": copy.deepcopy(old_schema),
            "old_tags": old_tags,
            "new_role_text": str(node.get("role_text") or new_role_text),
            "role_changed": bool(role_changed),
            "target_claim_id": str(target_claim.get("claim_id", "")),
            "target_support_status": str(target_claim.get("support_status", "")),
            "target_diagnosis": policy["diagnosis"],
            "target_behavior_tag": policy["tag"],
        }
        agent.reflected_this_gen = True
        action_event["events"].append(
            {
                "type": "self_reflect",
                "reflection_mode": "targeted_deferred_intrinsic_v1",
                "generation_context_mode": "next_regular_generation_no_extra_call",
                "role_update_mode": "fixed_schema_targeted" if old_schema is not None else "fallback_targeted_policy",
                "trigger_reason": "agent_action",
                "high_risk_threshold_q85": float(high_risk_threshold),
                "effective_self_reflect_risk_gate": float(effective_risk_gate),
                "minimum_pi": float(min_pi),
                "minimum_claim_risk": float(min_claim_risk),
                "baseline_node_pi": float(baseline_pi),
                "candidate_node_pi": None,
                "baseline_node_rho": float(getattr(agent, "rho", 0.0) or 0.0),
                "candidate_node_rho": None,
                "target_claim_id": str(target_claim.get("claim_id", "")),
                "target_claim_text": str(target_claim.get("text", "")),
                "target_claim_risk": float(claim_risk),
                "target_claim_importance": float(target_claim.get("importance", 0.0) or 0.0),
                "target_support_status": str(target_claim.get("support_status", "")),
                "target_diagnosis": policy["diagnosis"],
                "target_behavior_tag": policy["tag"],
                "old_role_text": old_role_text,
                "new_role_text": str(node.get("role_text") or new_role_text),
                "old_response": str(node.get("response", "") or ""),
                "new_response": str(node.get("response", "") or ""),
                "old_tags": old_tags,
                "new_tags": sorted(agent.tags),
                "role_changed": True,
                "role_text_changed": True,
                "response_changed": False,
                "changed": True,
                "accepted": False,
                "acceptance_pending": True,
                "risk_reduced": False,
                "extra_generation_calls": 0,
                "extra_risk_judge_calls": 0,
            }
        )

    def execute_actions_and_update_graph(self) -> None:
        latest_action_events = {
            int(agent_id): {
                "agent_id": int(agent_id),
                "events": [],
            }
            for agent_id in self.agents.keys()
        }
        edges_to_remove: List[List[int]] = []
        edges = self.graph_dict.get("edges", [])
        question_text = self.graph_dict.get("question_text", self.DEFAULT_QUESTION)
        online_reference_context = self._build_online_reference_context(question_text)
        current_gen = int(self.current_task.get("current_gen", 0))
        deferred_evaluated_nodes = self._finalize_deferred_self_role_updates(
            latest_action_events,
            current_gen,
        )
        dynamic_beta = self._resolve_edge_beta(
            edge_transmissions=self.latest_edge_transmissions,
            transmission_matrix=self.latest_transmission_matrix,
        )
        functional_ablation_spec = get_functional_ablation_spec(
            getattr(self.config.egt, "functional_ablation_mode", "full")
        )
        self_reflection_mode = functional_ablation_spec.self_reflection_mode
        response_only_self_v2 = self_reflection_mode in {
            "response_only_intrinsic",
            "response_only_counterfactual",
            "independent_resample_intrinsic",
        }
        counterfactual_self_v2 = self_reflection_mode == "response_only_counterfactual"
        independent_resample_self_v3 = (
            self_reflection_mode == "independent_resample_intrinsic"
        )
        targeted_deferred_self_v4 = (
            self_reflection_mode == "targeted_deferred_intrinsic"
        )
        paper_aligned_self = self_reflection_mode == "paper_aligned_synchronous"
        intrinsic_self_mode = response_only_self_v2 or targeted_deferred_self_v4
        risk_values = [
            (
                float(agent.pi)
                if intrinsic_self_mode
                else max(float(agent.pi), float(agent.rho), float(agent.h_final))
            )
            for agent in self.agents.values()
        ]
        high_risk_threshold = self._quantile(risk_values, 0.85) if risk_values else 1.0
        allow_policy_high_risk = (
            bool(getattr(self.config.egt, "enable_policy_high_risk_reflection", False))
            and not (self.config.egt.gen0_disable_reflection and current_gen == 0)
        )
        topology_governance_enabled = functional_ablation_spec.topology_governance_enabled

        for agent in self.agents.values():
            agent.successfully_verified = False
            agent.rep_token_count = 0.0
            agent.rep_success_count = 0
            agent.reported_count = 0
            agent.reflected_this_gen = False
            agent.rep_effective_this_gen = False
            agent.seek_effective_this_gen = False

        paper_self_attempted = (
            self._execute_paper_aligned_self_reflections(
                question_text=question_text,
                high_risk_threshold=high_risk_threshold,
                latest_action_events=latest_action_events,
            )
            if paper_aligned_self
            else set()
        )

        for node_id, agent in self.agents.items():
            node_risk = (
                float(agent.pi)
                if intrinsic_self_mode
                else max(float(agent.pi), float(agent.rho), float(agent.h_final))
            )
            nontrivial_risk_threshold = max(
                float(high_risk_threshold),
                float(getattr(self.config.egt, "seek_min_receiver_risk", 0.05) or 0.05),
            )
            is_high_risk_node = (
                node_risk >= nontrivial_risk_threshold
                if targeted_deferred_self_v4
                else node_risk > nontrivial_risk_threshold
            )
            policy_high_risk = (
                allow_policy_high_risk
                and is_high_risk_node
                and not intrinsic_self_mode
            )
            should_reflect = (
                False
                if paper_aligned_self
                else (agent.a_self == 1 and is_high_risk_node) or policy_high_risk
            )
            specific_self_block_recorded = bool(
                paper_aligned_self and node_id in paper_self_attempted
            )
            if node_id in deferred_evaluated_nodes:
                should_reflect = False
                if agent.a_self == 1:
                    latest_action_events[node_id]["events"].append(
                        {
                            "type": "self_reflect_blocked",
                            "blocked_reason": "deferred_update_evaluated_this_round",
                            "reflection_mode": "targeted_deferred_intrinsic_v1",
                        }
                    )
                    specific_self_block_recorded = True
            if should_reflect and targeted_deferred_self_v4 and current_gen >= self.num_generations - 1:
                latest_action_events[node_id]["events"].append(
                    {
                        "type": "self_reflect_blocked",
                        "blocked_reason": "no_future_generation_for_deferred_update",
                        "reflection_mode": "targeted_deferred_intrinsic_v1",
                    }
                )
                should_reflect = False
                specific_self_block_recorded = True
            if (
                agent.a_self == 1
                and not should_reflect
                and not specific_self_block_recorded
                and not paper_aligned_self
            ):
                latest_action_events[node_id]["events"].append(
                    {
                        "type": "self_reflect_blocked",
                        "blocked_reason": "below_risk_gate",
                        "node_risk": float(node_risk),
                        "risk_gate": float(nontrivial_risk_threshold),
                        "reflection_mode": (
                            (
                                "independent_resample_intrinsic_v1"
                                if independent_resample_self_v3
                                else (
                                    "response_only_counterfactual_v1"
                                    if counterfactual_self_v2
                                    else "response_only_intrinsic_v1"
                                )
                            )
                            if response_only_self_v2
                            else (
                                "targeted_deferred_intrinsic_v1"
                                if targeted_deferred_self_v4
                                else "legacy_role_mutation"
                            )
                        ),
                    }
                )
            if should_reflect and response_only_self_v2:
                self._execute_response_only_self_reflection(
                    node_id=node_id,
                    agent=agent,
                    question_text=question_text,
                    high_risk_threshold=high_risk_threshold,
                    effective_risk_gate=nontrivial_risk_threshold,
                    action_event=latest_action_events[node_id],
                    counterfactual=counterfactual_self_v2,
                    independent_resample=independent_resample_self_v3,
                )
                should_reflect = False
            if should_reflect and targeted_deferred_self_v4:
                self._schedule_targeted_deferred_self_reflection(
                    node_id=node_id,
                    agent=agent,
                    current_gen=current_gen,
                    high_risk_threshold=high_risk_threshold,
                    effective_risk_gate=nontrivial_risk_threshold,
                    action_event=latest_action_events[node_id],
                )
                should_reflect = False
            if should_reflect:
                old_tags = sorted(agent.tags)
                old_tags_set = set(agent.tags)
                old_role_text = ""
                old_response = ""
                old_role_id = ""
                old_role_meta = None
                old_fixed_role_schema = None
                if "nodes" in self.graph_dict:
                    nodes_data = self.graph_dict["nodes"]
                    if isinstance(nodes_data, list) and node_id < len(nodes_data) and isinstance(nodes_data[node_id], dict):
                        old_role_text = (
                            nodes_data[node_id].get("role_text", "")
                            or nodes_data[node_id].get("raw_role_text", "")
                            or ""
                        )
                        old_response = nodes_data[node_id].get("response", "") or ""
                        old_role_id = str(nodes_data[node_id].get("role_id") or "")
                        old_role_meta = self._normalize_role_meta(copy.deepcopy(nodes_data[node_id].get("role_meta")))
                        old_fixed_role_schema = self._extract_fixed_role_schema(nodes_data[node_id])
                if old_role_meta is None and old_fixed_role_schema is not None:
                    old_role_meta = self._normalize_role_meta(copy.deepcopy(old_fixed_role_schema))

                reflection_mode = "full_recompute"
                baseline_structural_fitness = float(getattr(agent, "fitness", 0.0) or 0.0)
                if is_high_risk_node:
                    baseline_pi_list = [float(getattr(self.agents[idx], "pi", 0.0) or 0.0) for idx in sorted(self.agents.keys())]
                    baseline_rho_list = [float(getattr(self.agents[idx], "rho", 0.0) or 0.0) for idx in sorted(self.agents.keys())]
                    baseline_structural_fitness = self._compute_agent_structural_fitness(
                        node_id, baseline_pi_list, baseline_rho_list
                    )

                local_reflection_rate = min(
                    float(self.config.egt.persona_mutation_rate),
                    1.0 / max(1, len(agent.tags)),
                )
                baseline_node_pi = float(getattr(agent, "h_pre", getattr(agent, "predicted_prob", 0.0)) or 0.0)
                baseline_node_rho = float(getattr(agent, "rho", 0.0) or 0.0)
                baseline_node_h = float(getattr(agent, "h_final", 0.0) or 0.0)
                candidate_node_pi = baseline_node_pi
                candidate_node_rho = baseline_node_rho
                candidate_node_h = baseline_node_h

                new_role_schema = old_fixed_role_schema
                new_role_text = old_role_text
                role_state_changed = False
                if old_fixed_role_schema is not None:
                    (
                        new_role_schema,
                        _schema_role_text,
                        new_tag_set,
                        role_state_changed,
                    ) = self._controlled_reflect_fixed_role_schema(
                        node_id=node_id,
                        current_gen=current_gen,
                        schema=old_fixed_role_schema,
                        is_high_risk_node=is_high_risk_node,
                    )
                    agent.tags = set(new_tag_set)
                else:
                    role_state_changed = agent.mutate_tags(
                        mutation_rate=local_reflection_rate,
                        force_change=True,
                    )

                new_role_text, role_rewrite_usage, role_rewrite_trace = (
                    self.rewrite_role_from_phenotype_tags(old_role_text, set(agent.tags))
                )
                changed = bool(
                    role_rewrite_trace.get("valid", False)
                    and str(new_role_text or "").strip() != str(old_role_text or "").strip()
                )
                if not changed:
                    agent.tags = old_tags_set
                    new_role_schema = old_fixed_role_schema
                    new_role_text = old_role_text
                accepted = False
                candidate_structural_fitness = baseline_structural_fitness
                candidate_response = old_response
                self_reflection_context_mode = "not_generated"
                self_reflection_candidate_valid = None
                self_reflection_invalid_candidate_preserved = False

                if changed:
                    if is_high_risk_node:
                        # High-risk nodes keep full reflection: regenerate answer + atomic-risk recompute + accept/reject.
                        try:
                            (
                                generation_question,
                                self_reflection_context_mode,
                            ) = self._build_self_reflection_question(
                                node_id,
                                question_text,
                            )
                            candidate_prompt_bundle = self._compose_runtime_system_prompt(agent, new_role_text)
                            candidate_system_prompt = str(
                                candidate_prompt_bundle.get("composed_system_prompt") or new_role_text
                            )
                            candidate_response, _candidate_usage = self.generate_response_sync(
                                candidate_system_prompt,
                                generation_question,
                            )
                        except Exception as exc:
                            candidate_response = f"Error generating text: {exc}"

                        response_update = self._resolve_invalid_response_update(
                            previous_response=old_response,
                            candidate_response=str(candidate_response or ""),
                        )
                        candidate_response = str(
                            response_update.get("selected_response", candidate_response) or ""
                        )
                        self_reflection_candidate_valid = (
                            response_update.get("candidate_answer") is not None
                        )
                        self_reflection_invalid_candidate_preserved = bool(
                            response_update.get("invalid_candidate_preserved", False)
                        )

                        if "nodes" in self.graph_dict:
                            nodes_data = self.graph_dict["nodes"]
                            if isinstance(nodes_data, list) and node_id < len(nodes_data) and isinstance(nodes_data[node_id], dict):
                                if new_role_schema is not None:
                                    self._apply_fixed_role_schema(
                                        nodes_data[node_id],
                                        new_role_schema,
                                        increment_revision=True,
                                    )
                                nodes_data[node_id]["role_text"] = new_role_text
                                nodes_data[node_id].setdefault("raw_role_text", old_role_text)
                                nodes_data[node_id].setdefault(
                                    "original_role_text",
                                    str(nodes_data[node_id].get("raw_role_text") or old_role_text),
                                )
                                nodes_data[node_id]["role_rewrite_model"] = role_rewrite_trace.get("model", "")
                                nodes_data[node_id]["role_rewrite_prompt_version"] = role_rewrite_trace.get(
                                    "prompt_version", ""
                                )
                                nodes_data[node_id]["role_rewrite_valid"] = bool(
                                    role_rewrite_trace.get("valid", False)
                                )
                                nodes_data[node_id]["role_rewrite_tags"] = list(
                                    role_rewrite_trace.get("phenotype_tags", [])
                                )
                                nodes_data[node_id]["response"] = candidate_response
                                nodes_data[node_id]["role_adapter_id"] = candidate_prompt_bundle.get("role_adapter_id", "")
                                nodes_data[node_id]["role_adapter_text"] = candidate_prompt_bundle.get("role_adapter_text", "")
                                nodes_data[node_id]["dynamic_role_wrapper"] = candidate_prompt_bundle.get("dynamic_role_wrapper", "")
                                nodes_data[node_id]["risk_status"] = candidate_prompt_bundle.get("risk_status", "")
                                nodes_data[node_id]["phenotype_tags_block"] = candidate_prompt_bundle.get("phenotype_tags_block", "")
                                nodes_data[node_id]["composed_role_prompt"] = candidate_system_prompt
                                self.latest_agent_responses[int(node_id)] = str(candidate_response or "")

                        candidate_state = self._estimate_risk_state()
                        candidate_pi_list = [float(candidate_state.pi.get(idx, 0.0) or 0.0) for idx in sorted(self.agents.keys())]
                        candidate_rho_list = [float(candidate_state.rho.get(idx, 0.0) or 0.0) for idx in sorted(self.agents.keys())]
                        candidate_h_list = [float(candidate_state.aggregate_risk.get(idx, 0.0) or 0.0) for idx in sorted(self.agents.keys())]
                        candidate_node_pi = float(candidate_pi_list[node_id] or 0.0)
                        candidate_node_rho = float(candidate_rho_list[node_id] or 0.0)
                        candidate_node_h = float(candidate_h_list[node_id] or 0.0)
                        candidate_structural_fitness = self._compute_agent_structural_fitness(
                            node_id, candidate_pi_list, candidate_rho_list
                        )
                        accept_margin = float(getattr(self.config.egt, "reflection_accept_margin", 0.0) or 0.0)
                        candidate_response_changed = str(candidate_response or "").strip() != str(old_response or "").strip()
                        accepted = (
                            (candidate_node_rho + 1e-9) < baseline_node_rho
                            or (
                                abs(candidate_node_rho - baseline_node_rho) <= 1e-9
                                and (candidate_node_pi + 1e-9) < baseline_node_pi
                            )
                            or (
                                candidate_response_changed
                                and (
                                    (candidate_node_rho + 1e-9) < baseline_node_rho
                                    or (candidate_node_pi + 1e-9) < baseline_node_pi
                                )
                            )
                            or ((candidate_structural_fitness - baseline_structural_fitness) > accept_margin)
                        )

                        if not accepted:
                            agent.tags = old_tags_set
                            if "nodes" in self.graph_dict:
                                nodes_data = self.graph_dict["nodes"]
                                if (
                                    isinstance(nodes_data, list)
                                    and node_id < len(nodes_data)
                                    and isinstance(nodes_data[node_id], dict)
                                ):
                                    if old_fixed_role_schema is not None:
                                        self._apply_fixed_role_schema(
                                            nodes_data[node_id],
                                            old_fixed_role_schema,
                                            increment_revision=False,
                                        )
                                    nodes_data[node_id]["role_text"] = old_role_text
                                    nodes_data[node_id]["response"] = old_response
                                    rollback_prompt_bundle = self._compose_runtime_system_prompt(agent, old_role_text)
                                    nodes_data[node_id]["role_adapter_id"] = rollback_prompt_bundle.get("role_adapter_id", "")
                                    nodes_data[node_id]["role_adapter_text"] = rollback_prompt_bundle.get("role_adapter_text", "")
                                    nodes_data[node_id]["dynamic_role_wrapper"] = rollback_prompt_bundle.get("dynamic_role_wrapper", "")
                                    nodes_data[node_id]["risk_status"] = rollback_prompt_bundle.get("risk_status", "")
                                    nodes_data[node_id]["composed_role_prompt"] = str(
                                        rollback_prompt_bundle.get("composed_system_prompt") or old_role_text
                                    )
                                    self.latest_agent_responses[int(node_id)] = str(old_response or "")
                    else:
                        # Defensive fallback; risk gating above should make this unreachable.
                        accepted = False
                        agent.tags = old_tags_set

                agent.reflected_this_gen = True
                trigger_reason = (
                    "both" if (agent.a_self == 1 and policy_high_risk)
                    else ("policy_high_risk" if policy_high_risk else "agent_action")
                )
                new_role_id = old_role_id
                new_role_meta = self._normalize_role_meta(copy.deepcopy(old_role_meta))
                if "nodes" in self.graph_dict:
                    nodes_data = self.graph_dict["nodes"]
                    if isinstance(nodes_data, list) and node_id < len(nodes_data) and isinstance(nodes_data[node_id], dict):
                        new_role_id = str(nodes_data[node_id].get("role_id") or old_role_id or "")
                        new_role_meta = self._normalize_role_meta(copy.deepcopy(nodes_data[node_id].get("role_meta")))
                if new_role_meta is None and new_role_schema is not None:
                    new_role_meta = self._normalize_role_meta(copy.deepcopy(new_role_schema))
                if "nodes" in self.graph_dict:
                    nodes_data = self.graph_dict["nodes"]
                    if isinstance(nodes_data, list) and node_id < len(nodes_data) and isinstance(nodes_data[node_id], dict):
                        nodes_data[node_id]["role_rewrite_accepted"] = bool(accepted)
                old_role_meta_cmp = old_role_meta if isinstance(old_role_meta, dict) else {}
                new_role_meta_cmp = new_role_meta if isinstance(new_role_meta, dict) else {}
                changed_role_fields = [
                    field_name
                    for field_name in (
                        "role_id",
                        "domain",
                        "style",
                        "self_check",
                        "social_tendency",
                        "heuristic_type",
                        "heuristic_text",
                        "role_revision",
                    )
                    if old_role_meta_cmp.get(field_name) != new_role_meta_cmp.get(field_name)
                ]
                role_changed = bool(changed_role_fields) or (set(old_tags) != set(sorted(agent.tags)))
                candidate_role_text_changed = (
                    str(new_role_text or "").strip() != str(old_role_text or "").strip()
                )
                candidate_response_changed = (
                    str(candidate_response or "").strip() != str(old_response or "").strip()
                )
                role_text_changed = bool(accepted and candidate_role_text_changed)
                response_changed = bool(accepted and candidate_response_changed)
                effective_changed = bool(role_changed or role_text_changed or response_changed)
                risk_reduced = (
                    (float(candidate_node_rho) + 1e-6) < float(baseline_node_rho)
                    or (float(candidate_node_pi) + 1e-6) < float(baseline_node_pi)
                )
                role_source = "data_role_schema" if (old_role_id or old_role_meta is not None) else "fallback_phenotype"
                latest_action_events[node_id]["events"].append(
                    {
                        "type": "self_reflect",
                        "reflection_mode": reflection_mode,
                        "generation_context_mode": self_reflection_context_mode,
                        "candidate_response_valid": self_reflection_candidate_valid,
                        "invalid_candidate_preserved": self_reflection_invalid_candidate_preserved,
                        "role_update_mode": "fixed_schema" if old_fixed_role_schema is not None else "gene_prompt_fallback",
                        "role_rewrite_model": role_rewrite_trace.get("model", ""),
                        "role_rewrite_prompt_version": role_rewrite_trace.get("prompt_version", ""),
                        "role_rewrite_valid": bool(role_rewrite_trace.get("valid", False)),
                        "role_rewrite_from_cache": bool(role_rewrite_trace.get("from_cache", False)),
                        "role_rewrite_error": str(role_rewrite_trace.get("error", "") or ""),
                        "role_rewrite_prompt_tokens": int(role_rewrite_usage.get("prompt_tokens", 0) or 0),
                        "role_rewrite_completion_tokens": int(role_rewrite_usage.get("completion_tokens", 0) or 0),
                        "role_state_changed_before_rewrite": bool(role_state_changed),
                        "trigger_reason": trigger_reason,
                        "high_risk_threshold_q85": float(high_risk_threshold),
                        "effective_self_reflect_risk_gate": float(nontrivial_risk_threshold),
                        "agent_h_final": float(agent.h_final),
                        "baseline_node_pi": baseline_node_pi,
                        "candidate_node_pi": candidate_node_pi,
                        "baseline_node_rho": baseline_node_rho,
                        "candidate_node_rho": candidate_node_rho,
                        "baseline_node_h": baseline_node_h,
                        "candidate_node_h": candidate_node_h,
                        "role_source": role_source,
                        "old_role_id": old_role_id,
                        "new_role_id": new_role_id,
                        "derived_role_id": new_role_id,
                        "old_role_meta": old_role_meta,
                        "new_role_meta": new_role_meta,
                        "old_role_text": old_role_text,
                        "new_role_text": new_role_text,
                        "old_response": old_response,
                        "new_response": candidate_response,
                        "changed_role_fields": changed_role_fields,
                        "role_changed": bool(role_changed),
                        "role_text_changed": bool(role_text_changed),
                        "candidate_role_text_changed": bool(candidate_role_text_changed),
                        "response_changed": bool(response_changed),
                        "candidate_response_changed": bool(candidate_response_changed),
                        "risk_reduced": bool(risk_reduced),
                        "old_tags": old_tags,
                        "new_tags": sorted(agent.tags),
                        "changed": bool(effective_changed),
                        "accepted": accepted,
                        "baseline_structural_fitness": float(baseline_structural_fitness),
                        "candidate_structural_fitness": float(candidate_structural_fitness),
                    }
                )
            if topology_governance_enabled and agent.a_rep == 1:
                senders = [int(edge[0]) for edge in edges if len(edge) >= 2 and int(edge[1]) == node_id]
                for sender in senders:
                    if sender not in self.agents:
                        continue

                    neighbor = self.agents[sender]
                    agent_pi = agent.predicted_prob
                    agent_rho = float(getattr(agent, "rho", 0.0) or 0.0)
                    neighbor_pi = self._edge_source_pi(sender, node_id, neighbor.predicted_prob)
                    edge_transmission_strength = 0.0
                    if (
                        self.latest_transmission_matrix
                        and 0 <= node_id < len(self.latest_transmission_matrix)
                        and 0 <= sender < len(self.latest_transmission_matrix[node_id])
                    ):
                        edge_transmission_strength = self._safe_edge_t(
                            self.latest_transmission_matrix, sender, node_id, default_if_missing=0.0
                        )
                    neighbor_response = ""
                    if "nodes" in self.graph_dict and sender < len(self.graph_dict["nodes"]):
                        sender_node = self.graph_dict["nodes"][sender]
                        if isinstance(sender_node, dict):
                            neighbor_response = sender_node.get("response", sender_node.get("role_text", "")) or ""

                    should_remove, verify_info = self.verification_engine.should_remove_edge(
                        agent_pi=agent_pi,
                        neighbor_pi=neighbor_pi,
                        beta=dynamic_beta,
                        question_text=question_text,
                        reference_fact=online_reference_context,
                        neighbor_response=neighbor_response,
                        judge_fn=self.sync_judge_text,
                        edge_transmission_strength=edge_transmission_strength,
                        agent_rho=agent_rho,
                    )
                    if verify_info.get("mode") == "risk":
                        edge_t = edge_transmission_strength
                        raw_edge_delta = (1.0 - neighbor_pi) - dynamic_beta * edge_t * neighbor_pi * (1.0 + agent_rho)
                        verify_info["edge_delta_raw"] = float(raw_edge_delta)
                        should_remove = raw_edge_delta <= 0.0

                    if verify_info.get("mode") == "judge":
                        usage = verify_info.get("usage", {})
                        agent.rep_token_count += self._total_tokens_from_usage(usage)
                        agent.total_token_count = agent.gen_token_count + agent.rep_token_count + agent.judge_token_count

                    if should_remove:
                        edges_to_remove.append([sender, node_id])
                        agent.rep_success_count += 1
                        neighbor.reported_count += 1
                        latest_action_events[node_id]["events"].append(
                            {
                                "type": "prune",
                                "target_agent_id": int(sender),
                                "mode": verify_info.get("mode", "risk"),
                                "agent_risk": float(agent_pi),
                                "target_risk": float(neighbor_pi),
                                "source_pi": float(neighbor_pi),
                                "receiver_rho": float(agent_rho),
                                "edge_transmission_strength": float(edge_transmission_strength),
                                "transmission_strength": float(edge_transmission_strength),
                                "edge_delta_raw": float(
                                    verify_info.get(
                                        "edge_delta_raw",
                                        verify_info.get(
                                            "edge_delta",
                                            verify_info.get("marginal_benefit", 0.0)
                                            - verify_info.get("marginal_cost", 0.0),
                                        ),
                                    )
                                ),
                                "edge_delta_before": float(
                                    verify_info.get(
                                        "edge_delta_raw",
                                        verify_info.get(
                                            "edge_delta",
                                            verify_info.get("marginal_benefit", 0.0)
                                            - verify_info.get("marginal_cost", 0.0),
                                        ),
                                    )
                                ),
                                "removed_because_negative_delta": bool(
                                    float(
                                        verify_info.get(
                                            "edge_delta_raw",
                                            verify_info.get(
                                                "edge_delta",
                                                verify_info.get("marginal_benefit", 0.0)
                                                - verify_info.get("marginal_cost", 0.0),
                                            ),
                                        )
                                    )
                                    < 0.0
                                ),
                                "used_t_tiebreak": bool(verify_info.get("used_t_tiebreak", False)),
                            }
                        )
                        latest_action_events[int(sender)]["events"].append(
                            {
                                "type": "reported",
                                "by_agent_id": int(node_id),
                                "mode": verify_info.get("mode", "risk"),
                                "self_risk": float(neighbor_pi),
                                "edge_transmission_strength": float(edge_transmission_strength),
                                "used_t_tiebreak": bool(verify_info.get("used_t_tiebreak", False)),
                            }
                        )
                        if verify_info.get("mode") == "judge":
                            print(
                                f"[Prune] Agent_{node_id} removes edge from Agent_{sender} "
                                f"via judge verification."
                            )
                        else:
                            marginal_benefit = verify_info.get("marginal_benefit", 0.0)
                            marginal_cost = verify_info.get("marginal_cost", 0.0)
                            edge_delta = verify_info.get("edge_delta_raw", verify_info.get("edge_delta", marginal_benefit - marginal_cost))
                            print(
                                f"[Prune] Agent_{node_id} removes edge from Agent_{sender}: "
                                f"delta={edge_delta:.3f} (benefit {marginal_benefit:.3f}, "
                                f"cost {marginal_cost:.3f}, T={edge_transmission_strength:.3f}, "
                                f"beta_t={dynamic_beta:.3f})"
                            )
                agent.rep_effective_this_gen = (agent.rep_success_count > 0) or (agent.rep_token_count > 0.0)

        if edges_to_remove:
            self.graph_dict["edges"] = [
                edge
                for edge in edges
                if len(edge) >= 2 and [int(edge[0]), int(edge[1])] not in edges_to_remove
            ]

        in_neighbors_map = {nid: set() for nid in self.agents.keys()}
        for edge in self.graph_dict.get("edges", []):
            if len(edge) >= 2:
                in_neighbors_map[int(edge[1])].add(int(edge[0]))

        for node_id, agent in self.agents.items():
            if not topology_governance_enabled or agent.a_seek != 1:
                continue

            candidates = [
                cand_id
                for cand_id in self.agents.keys()
                if cand_id != node_id and cand_id not in in_neighbors_map[node_id]
            ]
            if not candidates:
                continue

            import random

            sample_size = min(3, len(candidates))
            sampled_candidates = random.sample(candidates, sample_size)
            best_node = None
            best_delta = float("-inf")
            best_delta_raw = float("-inf")
            best_node_risk = 1.0
            best_t = 1.0
            tau_rho_i = 1.0 / (1.0 + dynamic_beta * (1.0 + float(getattr(agent, "rho", 0.0) or 0.0)))
            receiver_risk = max(float(getattr(agent, "rho", 0.0) or 0.0), float(getattr(agent, "h_final", 0.0) or 0.0))
            min_receiver_risk = float(getattr(self.config.egt, "seek_min_receiver_risk", 0.05) or 0.05)
            seek_delta_margin = float(getattr(self.config.egt, "seek_delta_margin", 0.01) or 0.01)
            max_in_degree = getattr(self.config.egt, "max_in_degree", None)
            if receiver_risk <= min_receiver_risk:
                latest_action_events[node_id]["events"].append(
                    {
                        "type": "seek_blocked",
                        "candidate_agent_id": None,
                        "candidate_risk": None,
                        "edge_transmission_strength": 0.0,
                        "edge_delta": None,
                        "edge_delta_raw": None,
                        "degree_penalty": 0.0,
                        "tau_rho_i": float(tau_rho_i),
                        "blocked_reason": "receiver_low_risk",
                    }
                )
                continue
            if max_in_degree is not None and len(in_neighbors_map[node_id]) >= int(max_in_degree):
                latest_action_events[node_id]["events"].append(
                    {
                        "type": "seek_blocked",
                        "candidate_agent_id": None,
                        "candidate_risk": None,
                        "edge_transmission_strength": 0.0,
                        "edge_delta": None,
                        "edge_delta_raw": None,
                        "degree_penalty": 0.0,
                        "tau_rho_i": float(tau_rho_i),
                        "blocked_reason": "max_in_degree",
                    }
                )
                continue
            for cand_id in sampled_candidates:
                cand_pi = float(self.agents[cand_id].predicted_prob)
                cand_rho = float(getattr(self.agents[cand_id], "rho", 0.0) or 0.0)
                # A non-edge has no observed transmission. Use the upper bound
                # only for prospective seek screening, not for realized rho/fitness.
                edge_t = self._safe_edge_t(self.latest_transmission_matrix, cand_id, node_id, default_if_missing=1.0)
                if cand_pi > float(getattr(self.config.egt, "tau_pi", 0.5) or 0.5) or cand_rho > tau_rho_i:
                    continue
                if cand_pi >= float(getattr(agent, "predicted_prob", 0.0) or 0.0):
                    continue
                edge_delta_raw = (1.0 - cand_pi) - dynamic_beta * edge_t * cand_pi * (1.0 + float(getattr(agent, "rho", 0.0) or 0.0))
                edge_delta = edge_delta_raw
                if edge_delta > best_delta:
                    best_delta = edge_delta
                    best_delta_raw = edge_delta_raw
                    best_node = cand_id
                    best_node_risk = cand_pi
                    best_t = edge_t

            if best_node is None or best_delta <= seek_delta_margin:
                latest_action_events[node_id]["events"].append(
                    {
                        "type": "seek_blocked",
                        "candidate_agent_id": int(best_node) if best_node is not None else None,
                        "candidate_risk": float(best_node_risk),
                        "edge_transmission_strength": float(best_t),
                        "edge_delta": None,
                        "edge_delta_raw": None,
                        "degree_penalty": 0.0,
                        "tau_rho_i": float(tau_rho_i),
                        "blocked_reason": "no_valid_candidate",
                    }
                )
                print(
                    f"[Seek Blocked] Agent_{node_id} skips Agent_{best_node}: "
                    f"adj={best_delta:.3f} <= 0 "
                    f"(raw {best_delta_raw:.3f}, penalty 0.000, risk {best_node_risk:.3f}, T={best_t:.3f}, beta_t={dynamic_beta:.3f})"
                )
                continue

            if "edges" not in self.graph_dict:
                self.graph_dict["edges"] = []
            self.graph_dict["edges"].append([best_node, node_id])
            in_neighbors_map[node_id].add(best_node)
            agent.seek_effective_this_gen = True
            latest_action_events[node_id]["events"].append(
                {
                    "type": "seek_connect",
                    "source_agent_id": int(best_node),
                    "source_utility": float(self.agents[best_node].intrinsic_utility),
                    "source_risk": float(best_node_risk),
                    "source_rho": float(getattr(self.agents[best_node], "rho", 0.0) or 0.0),
                    "edge_transmission_strength": float(best_t),
                    "edge_delta": float(best_delta),
                    "edge_delta_raw": float(best_delta_raw),
                    "degree_penalty": 0.0,
                    "tau_rho_i": float(tau_rho_i),
                }
            )
            print(
                f"[Seek] Agent_{node_id} connects to Agent_{best_node} "
                f"(adj {best_delta:.3f} > 0, raw {best_delta_raw:.3f}, penalty 0.000, "
                f"utility {self.agents[best_node].intrinsic_utility:.3f}, T={best_t:.3f}, beta_t={dynamic_beta:.3f})"
            )

        self.latest_action_events = [
            latest_action_events[agent_id]
            for agent_id in sorted(latest_action_events.keys())
        ]

    def calculate_fitness(
        self,
        current_gen: int = 0,
        verify_cost: float | None = None,
        search_cost: float | None = None,
        refresh_pipeline: bool | None = None,
        risk_state: RiskState | None = None,
    ) -> Dict[str, float]:
        beta_value = getattr(
            self,
            "latest_edge_beta_used",
            getattr(self.config.egt, "fitness_penalty_beta", 1.0),
        )
        beta = float(1.0 if beta_value is None else beta_value)
        effective_verify_cost = self.config.egt.verify_cost if verify_cost is None else verify_cost
        effective_search_cost = self.config.egt.search_cost if search_cost is None else search_cost
        is_observation_gen = self._is_observation_generation(current_gen)
        question_text = self.graph_dict.get("question_text", self.DEFAULT_QUESTION)
        online_reference_context = self._build_online_reference_context(question_text)

        self.graph_dict["question_text"] = question_text
        self.graph_dict["task"] = {
            "question_text": question_text,
            "online_reference_context": online_reference_context,
            "current_gen": current_gen,
        }
        self.current_task = self.graph_dict["task"]
        # Keep the bounded recent prompt audit across rounds. Clearing here can
        # erase every sample when the final round reuses unchanged claim scores.

        if "nodes" not in self.graph_dict:
            self.graph_dict["nodes"] = [{} for _ in range(len(self.agents))]

        self_reflection_mode = get_functional_ablation_spec(
            getattr(self.config.egt, "functional_ablation_mode", "full")
        ).self_reflection_mode
        for agent in self.agents.values():
            agent.use_reflection = self.config.agent.default_use_reflection and (
                not self.config.egt.gen0_disable_reflection or current_gen != 0
            )
            if (
                self.config.egt.gen0_force_a_self_zero
                and current_gen == 0
                and self_reflection_mode != "paper_aligned_synchronous"
            ):
                agent.a_self = 0.0
            agent.gen_token_count = 0.0
            agent.judge_token_count = 0.0
            agent.total_token_count = agent.rep_token_count

        refresh_interval = max(1, int(getattr(self.config.egt, "response_refresh_interval", 1)))
        should_refresh_pipeline = (
            current_gen == 0
            or is_observation_gen
            or (current_gen % refresh_interval == 0)
        )
        if refresh_pipeline is not None:
            should_refresh_pipeline = bool(refresh_pipeline)
        if should_refresh_pipeline:
            self._initialize_responses_from_final_raw_outputs()
            pre_answer_choice_analysis = self._build_answer_choice_analysis()
            pre_consensus_analysis = self._build_consensus_analysis(pre_answer_choice_analysis)
            self.latest_answer_choice_analysis = pre_answer_choice_analysis
            self.latest_consensus_analysis = pre_consensus_analysis
        if should_refresh_pipeline:
            print("[Emulator] Running sequential generation and judge pipeline...")
            self.response_generation_round_count += 1
            self.run_agent_pipeline(current_gen, question_text, online_reference_context)

        if risk_state is None:
            risk_state = self._estimate_risk_state()
        pi_list, rho_list, h_list, transmission_matrix, beta = self._apply_risk_state(risk_state)
        answer_choice_analysis = self._build_answer_choice_analysis(
            pi_list=pi_list,
            rho_list=rho_list,
            h_list=h_list,
        )
        consensus_analysis = self._build_consensus_analysis(answer_choice_analysis)
        self.latest_answer_choice_analysis = answer_choice_analysis
        self.latest_consensus_analysis = consensus_analysis
        self.latest_prediction_details["answer_choice_utility"] = float(
            answer_choice_analysis.get("avg_answer_choice_utility", 0.0) or 0.0
        )
        self.latest_prediction_details["consensus_risk"] = float(consensus_analysis.get("consensus_risk", 0.0) or 0.0)
        answer_choice_by_agent = {
            int(item.get("agent_id", -1)): float(item.get("answer_choice_utility", 0.0) or 0.0)
            for item in answer_choice_analysis.get("per_agent", []) or []
            if int(item.get("agent_id", -1)) >= 0
        }
        rep_gain_by_agent: Dict[int, float] = {int(agent_id): 0.0 for agent_id in self.agents.keys()}
        seek_gain_by_agent: Dict[int, float] = {int(agent_id): 0.0 for agent_id in self.agents.keys()}
        for agent_events in getattr(self, "latest_action_events", []) or []:
            agent_id = int(agent_events.get("agent_id", -1))
            if agent_id not in rep_gain_by_agent:
                continue
            for event in agent_events.get("events", []) or []:
                event_type = str(event.get("type", ""))
                edge_delta_value = event.get("edge_delta_raw", 0.0)
                edge_delta_raw = float(edge_delta_value) if edge_delta_value is not None else 0.0
                if event_type == "prune":
                    rep_gain_by_agent[agent_id] += max(0.0, -edge_delta_raw)
                elif event_type == "seek_connect":
                    seek_gain_by_agent[agent_id] += max(0.0, edge_delta_raw)
        in_neighbors_map = {nid: [] for nid in self.agents.keys()}
        for edge in self.graph_dict.get("edges", []):
            if len(edge) >= 2:
                in_neighbors_map[int(edge[1])].append(int(edge[0]))

        for node_id, agent in self.agents.items():
            agent.pi = pi_list[node_id]
            agent.predicted_prob = agent.pi
            agent.rho = rho_list[node_id]
            agent.h_pre = agent.predicted_prob
            agent.h_final = h_list[node_id]
            agent.infection_penalty = agent.rho
            agent.answer_choice_utility = answer_choice_by_agent.get(node_id, 0.0)
            agent.consensus_risk_exposure = float(consensus_analysis.get("consensus_risk", 0.0) or 0.0)
            agent.fitness = (1.0 - agent.pi) - beta * agent.rho
            interaction_utility = 0.0

            for src in in_neighbors_map.get(node_id, []):
                if src not in self.agents:
                    continue

                neighbor = self.agents[src]
                neighbor.pi = pi_list[src]
                neighbor.predicted_prob = neighbor.pi
                edge_source_pi = self._edge_source_pi(src, node_id, neighbor.pi)
                edge_t = self._safe_edge_t(transmission_matrix, src, node_id, default_if_missing=0.0)
                marginal_benefit = 1.0 - edge_source_pi
                marginal_cost = beta * edge_t * edge_source_pi * (1.0 + agent.rho)
                interaction_utility += marginal_benefit - marginal_cost

            in_degree = len(in_neighbors_map.get(node_id, []))
            if in_degree > 0:
                interaction_utility /= in_degree
            agent.fitness += interaction_utility
            # U_gov = c_rep * (G_rep - I_rep) + c_seek * (G_seek - I_seek).
            rep_gain = float(rep_gain_by_agent.get(node_id, 0.0))
            seek_gain = float(seek_gain_by_agent.get(node_id, 0.0))
            rep_eff = 1.0 if bool(getattr(agent, "rep_effective_this_gen", False)) else 0.0
            seek_eff = 1.0 if bool(getattr(agent, "seek_effective_this_gen", False)) else 0.0
            reward_term = effective_verify_cost * rep_gain + effective_search_cost * seek_gain
            cost_term = effective_verify_cost * rep_eff + effective_search_cost * seek_eff
            agent.fitness += reward_term - cost_term

        num_agents = len(self.agents)
        if not num_agents:
            return {
                "avg_node_real_hallu_rate": None,
                "avg_atomic_rate": None,
                "avg_swhi": None,
                "avg_info_snr": None,
                "avg_answer_choice_utility": 0.0,
                "consensus_risk": 0.0,
                "consensus_ratio": 0.0,
                "consensus_breaker_triggered": 0.0,
            }

        if not is_observation_gen:
            return {
                "avg_node_real_hallu_rate": None,
                "avg_atomic_rate": None,
                "avg_swhi": None,
                "avg_info_snr": None,
                "avg_answer_choice_utility": float(answer_choice_analysis.get("avg_answer_choice_utility", 0.0) or 0.0),
                "consensus_risk": float(consensus_analysis.get("consensus_risk", 0.0) or 0.0),
                "consensus_ratio": float(consensus_analysis.get("consensus_ratio", 0.0) or 0.0),
                "consensus_breaker_triggered": 1.0 if bool(consensus_analysis.get("consensus_breaker_triggered", False)) else 0.0,
            }

        return {
            "avg_node_real_hallu_rate": sum(agent.real_hallucination_rate for agent in self.agents.values()) / num_agents,
            "avg_atomic_rate": sum(agent.metric_atomic for agent in self.agents.values()) / num_agents,
            "avg_swhi": sum(agent.metric_swhi for agent in self.agents.values()) / num_agents,
            "avg_info_snr": sum(agent.metric_snr for agent in self.agents.values()) / num_agents,
            "avg_answer_choice_utility": float(answer_choice_analysis.get("avg_answer_choice_utility", 0.0) or 0.0),
            "consensus_risk": float(consensus_analysis.get("consensus_risk", 0.0) or 0.0),
            "consensus_ratio": float(consensus_analysis.get("consensus_ratio", 0.0) or 0.0),
            "consensus_breaker_triggered": 1.0 if bool(consensus_analysis.get("consensus_breaker_triggered", False)) else 0.0,
        }

    def update_strategies_fermi(
        self,
        selection_alpha: float | None = None,
        beta_fermi: float | None = None,
        mutation_rate: float | None = None,
        crossover_rate: float | None = None,
        exploration_noise: float | None = None,
    ) -> None:
        import math
        import random

        effective_selection_alpha = (
            selection_alpha
            if selection_alpha is not None
            else (beta_fermi if beta_fermi is not None else self.config.egt.selection_alpha)
        )
        effective_mutation_rate = (
            self.config.egt.mutation_start if mutation_rate is None else mutation_rate
        )
        effective_exploration_noise = (
            self.config.egt.exploration_noise if exploration_noise is None else exploration_noise
        )

        strategy_set = tuple(self.config.egt.strategy_set or ())
        if not strategy_set:
            return

        new_strategy_ids: Dict[int, int] = {}
        for agent_i in self.agents.values():
            candidates = [a for a in self.agents.values() if a.node_id != agent_i.node_id]
            if not candidates:
                new_strategy_ids[agent_i.node_id] = int(getattr(agent_i, "strategy_id", 0)) % len(strategy_set)
                continue

            agent_j = random.choice(candidates)
            delta_u = agent_j.fitness - agent_i.fitness
            exp_val = max(min(-effective_selection_alpha * delta_u, 700.0), -700.0)
            fermi_prob = 1.0 / (1.0 + math.exp(exp_val))

            own_sid = int(getattr(agent_i, "strategy_id", 0)) % len(strategy_set)
            peer_sid = int(getattr(agent_j, "strategy_id", 0)) % len(strategy_set)
            imitate_success = random.random() < fermi_prob
            chosen_sid = peer_sid if imitate_success else own_sid
            new_strategy_ids[agent_i.node_id] = chosen_sid
            if chosen_sid != own_sid:
                print(
                    f"[Imitate] Agent_{agent_i.node_id} copies strategy from Agent_{agent_j.node_id} "
                    f"(fitness {agent_i.fitness:.3f} -> {agent_j.fitness:.3f})"
                )

        for agent_i in self.agents.values():
            next_sid = int(new_strategy_ids.get(agent_i.node_id, getattr(agent_i, "strategy_id", 0))) % len(strategy_set)
            own_sid = int(getattr(agent_i, "strategy_id", 0)) % len(strategy_set)
            mutation_prob = max(0.0, min(1.0, float(effective_mutation_rate)))
            if random.random() < mutation_prob:
                strategy_bits = list(strategy_set[next_sid])
                if strategy_bits:
                    bit_idx = random.randrange(len(strategy_bits))
                    strategy_bits[bit_idx] = 1 - int(strategy_bits[bit_idx])
                    mutated = tuple(strategy_bits)
                    if mutated in strategy_set:
                        next_sid = strategy_set.index(mutated)
            imitation_failed = (next_sid == own_sid)
            if imitation_failed and random.random() < effective_exploration_noise:
                next_sid = random.randrange(len(strategy_set))
            agent_i.apply_strategy(next_sid)
