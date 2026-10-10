import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..config import ExperimentConfig, build_default_config
from .prompts import CAUTIOUS_TAG_POOLS


PHENOTYPE_LIBRARY_CACHE: Dict[Path, Dict[str, List[str]]] = {}


def load_phenotype_library(file_path: Path) -> Dict[str, List[str]]:
    """Load and cache the phenotype library used to seed agent tags."""

    resolved_path = Path(file_path).resolve()
    if resolved_path in PHENOTYPE_LIBRARY_CACHE:
        return PHENOTYPE_LIBRARY_CACHE[resolved_path]

    try:
        with open(resolved_path, "r", encoding="utf-8") as handle:
            library = json.load(handle)
    except FileNotFoundError:
        print(f"[Agent] Warning: phenotype library not found at {resolved_path}. Using an empty fallback.")
        library = {"cognitive": [], "verification": [], "domain": []}

    PHENOTYPE_LIBRARY_CACHE[resolved_path] = library
    return library


def build_tag_category_lookup(library: Dict[str, List[str]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for category, tags in library.items():
        for tag in tags or []:
            lookup[str(tag)] = str(category)
    return lookup


def cautious_candidates_for_tag(
    tag_to_remove: str,
    *,
    tag_library: Dict[str, List[str]],
    tag_category_lookup: Dict[str, str],
) -> List[str]:
    category = tag_category_lookup.get(tag_to_remove)
    if not category:
        candidates: List[str] = []
        for tags_list in tag_library.values():
            candidates.extend(tags_list or [])
        return sorted(set(candidates))

    if category == "domain":
        return sorted(set(tag_library.get("domain", []) or []))

    cautious_pool = CAUTIOUS_TAG_POOLS.get(category)
    if cautious_pool:
        library_pool = set(tag_library.get(category, []) or [])
        return sorted(tag for tag in cautious_pool if tag in library_pool)

    return sorted(set(tag_library.get(category, []) or []))


class Agent:
    """Agent state container used by the evolutionary game simulation."""

    def __init__(
        self,
        node_id: int,
        initial_tags: Optional[Set[str]] = None,
        config: Optional[ExperimentConfig] = None,
    ):
        self.config = config or build_default_config()
        self.tag_library = load_phenotype_library(self.config.agent.phenotype_library_path)
        self.tag_category_lookup = build_tag_category_lookup(self.tag_library)

        self.node_id: int = node_id

        if initial_tags is not None:
            self.tags: Set[str] = set(initial_tags)
        else:
            self.tags = set()
            for _, tags_list in self.tag_library.items():
                if tags_list:
                    self.tags.add(random.choice(tags_list))

        activation_prob = self.config.agent.initial_action_activation_prob
        self.a_in: float = self.config.agent.default_a_in
        self.a_out: float = self.config.agent.default_a_out
        strategy_set = tuple(self.config.egt.strategy_set or ())
        if strategy_set:
            disabled_indices = {
                int(index)
                for index in getattr(self.config.egt, "disabled_strategy_action_indices", ())
            }
            if not disabled_indices:
                # Preserve the original full-method initialization path exactly.
                self.strategy_id = random.randrange(len(strategy_set))
                self.initial_strategy_source_id = int(self.strategy_id)
                self.initial_strategy_source = tuple(int(value) for value in strategy_set[self.strategy_id])
                self.initial_projected_strategy = self.initial_strategy_source
                self.a_rep, self.a_seek, self.a_self = self.strategy_actions(
                    strategy_set[self.strategy_id]
                )
            else:
                initialization_source = tuple(
                    getattr(self.config.egt, "strategy_initialization_source_set", ())
                    or strategy_set
                )
                self.initial_strategy_source_id = random.randrange(len(initialization_source))
                source_strategy = initialization_source[self.initial_strategy_source_id]
                projected_strategy = tuple(
                    0 if index in disabled_indices else int(value)
                    for index, value in enumerate(source_strategy)
                )
                self.initial_strategy_source = tuple(int(value) for value in source_strategy)
                self.initial_projected_strategy = projected_strategy
                try:
                    self.strategy_id = strategy_set.index(projected_strategy)
                except ValueError as exc:
                    raise ValueError(
                        "Projected initial strategy is absent from the active strategy set: "
                        f"source={source_strategy}, projected={projected_strategy}"
                    ) from exc
                self.a_rep, self.a_seek, self.a_self = self.strategy_actions(projected_strategy)
        else:
            self.strategy_id = -1
            self.initial_strategy_source_id = -1
            self.initial_strategy_source = ()
            self.initial_projected_strategy = ()
            self.a_self = 1.0 if random.random() < activation_prob else 0.0
            self.a_rep = 1.0 if random.random() < activation_prob else 0.0
            self.a_seek = 1.0 if random.random() < activation_prob else 0.0
        self.use_reflection: bool = self.config.agent.default_use_reflection

        self.kappa: float = 0.5
        self.pi: float = 0.0
        self.rho: float = 0.0

        self.predicted_prob: float = 0.0
        self.h_pre: float = 0.0
        self.h_final: float = 0.0
        self.intrinsic_utility: float = 0.0
        self.infection_penalty: float = 0.0
        self.is_actually_hallucinating: bool = False
        self.real_hallucination_rate: float = 0.0
        self.metric_atomic: float = 0.0
        self.metric_swhi: float = 0.0
        self.metric_snr: float = 0.0
        self.answer_choice_utility: float = 0.0
        self.consensus_risk_exposure: float = 0.0
        self.rep_success_count: int = 0
        self.reported_count: int = 0
        self.rep_effective_this_gen: bool = False
        self.seek_effective_this_gen: bool = False
        self.gen_token_count: float = 0.0
        self.rep_token_count: float = 0.0
        self.judge_token_count: float = 0.0
        self.total_token_count: float = 0.0
        self.fitness: float = 0.0
        self.reflected_this_gen: bool = False

    @staticmethod
    def strategy_actions(strategy: Tuple[int, int, int]) -> Tuple[float, float, float]:
        a_rep, a_seek, a_self = strategy
        return float(int(a_rep)), float(int(a_seek)), float(int(a_self))

    def apply_strategy(self, strategy_id: int) -> None:
        strategy_set = tuple(self.config.egt.strategy_set or ())
        if not strategy_set:
            return
        bounded_id = int(strategy_id) % len(strategy_set)
        self.strategy_id = bounded_id
        self.a_rep, self.a_seek, self.a_self = self.strategy_actions(strategy_set[bounded_id])

    def mutate_tags(self, mutation_rate: Optional[float] = None, force_change: bool = False) -> bool:
        """Randomly replace one or more tags and optionally force the final tag set to differ."""

        effective_mutation_rate = self.config.agent.mutation_rate if mutation_rate is None else mutation_rate
        if not self.tags:
            return False
        if not force_change and random.random() >= effective_mutation_rate:
            return False

        original_tags = set(self.tags)
        tag_count = len(original_tags)
        replacement_budget = max(1, int(round(float(effective_mutation_rate) * max(1, tag_count))))
        replacement_budget = min(replacement_budget, max(1, tag_count))

        all_possible_tags: List[str] = []
        for tags_list in self.tag_library.values():
            all_possible_tags.extend(tags_list)
        all_possible_tags = sorted(set(all_possible_tags))

        if not all_possible_tags:
            return False

        max_attempts = max(8, len(all_possible_tags) * 2)
        for _ in range(max_attempts):
            working_tags = set(original_tags)
            changed_pairs = []
            for _step in range(replacement_budget):
                if not working_tags:
                    break
                tag_to_remove = random.choice(list(working_tags))
                working_tags.remove(tag_to_remove)
                replacement_pool = cautious_candidates_for_tag(
                    tag_to_remove,
                    tag_library=self.tag_library,
                    tag_category_lookup=self.tag_category_lookup,
                )
                available_tags = [
                    tag for tag in replacement_pool
                    if tag not in working_tags and (not force_change or tag != tag_to_remove)
                ]
                if not available_tags:
                    available_tags = [
                        tag for tag in all_possible_tags
                        if tag not in working_tags and (not force_change or tag != tag_to_remove)
                    ]
                if not available_tags:
                    working_tags.add(tag_to_remove)
                    continue
                new_tag = random.choice(available_tags)
                working_tags.add(new_tag)
                changed_pairs.append((tag_to_remove, new_tag))
            if working_tags != original_tags:
                self.tags = working_tags
                if changed_pairs:
                    tag_to_remove, new_tag = changed_pairs[0]
                    print(
                        f"[Mutation] Agent_{self.node_id} replaces tag [{tag_to_remove}] with [{new_tag}] "
                        f"(rate={effective_mutation_rate:.3f}, replacements={len(changed_pairs)}, force_change={force_change})"
                    )
                return True

        fallback_pool = cautious_candidates_for_tag(
            random.choice(list(original_tags)),
            tag_library=self.tag_library,
            tag_category_lookup=self.tag_category_lookup,
        )
        fallback_candidates = [tag for tag in fallback_pool if tag not in original_tags]
        if not fallback_candidates:
            fallback_candidates = [tag for tag in all_possible_tags if tag not in original_tags]
        if fallback_candidates:
            self.tags = set(original_tags)
            tag_to_remove = random.choice(list(self.tags))
            self.tags.remove(tag_to_remove)
            new_tag = random.choice(fallback_candidates)
            self.tags.add(new_tag)
            print(
                f"[Mutation] Agent_{self.node_id} fallback replaces tag [{tag_to_remove}] with [{new_tag}] "
                f"(force_change={force_change})"
            )
            return self.tags != original_tags

        self.tags = original_tags
        return False

    def crossover_tags(self, elite_tags: Set[str], crossover_rate: Optional[float] = None) -> None:
        """Add elite tags with the configured crossover probability."""

        effective_crossover_rate = self.config.agent.crossover_rate if crossover_rate is None else crossover_rate
        for tag in elite_tags:
            if tag not in self.tags and random.random() < effective_crossover_rate:
                self.tags.add(tag)
