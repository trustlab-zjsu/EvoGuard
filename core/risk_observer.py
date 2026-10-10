"""Atomic claim and cross-agent transmission risk observation."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

from .prompts import RISK_JUDGE_PROMPT_VERSION, TRANSMISSION_JUDGE_PROMPT_VERSION


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def aggregate_risk_from_pi_rho(pi_value: float, rho_value: float) -> float:
    pi_value = clamp01(pi_value)
    rho_value = clamp01(rho_value)
    return clamp01(1.0 - (1.0 - pi_value) * (1.0 - rho_value))


@dataclass
class RiskState:
    pi: Dict[int, float]
    rho: Dict[int, float]
    T: Dict[Tuple[int, int], float]
    source_pi: Dict[int, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    aggregate_risk: Dict[int, float] = field(default_factory=dict)
    legacy_h: Dict[int, float] = field(default_factory=dict)

    def to_legacy_details(self, *, size: Optional[int] = None) -> Dict[str, Any]:
        node_ids = sorted(set(self.pi) | set(self.rho) | set(self.aggregate_risk))
        if size is None:
            size = (max(node_ids) + 1) if node_ids else 0
        pi_list = [0.0 for _ in range(size)]
        rho_list = [0.0 for _ in range(size)]
        h_list = [0.0 for _ in range(size)]
        source_pi_list = [0.0 for _ in range(size)]
        # None distinguishes an unobserved pair from an observed zero-transmission edge.
        matrix: List[List[Optional[float]]] = [[None for _ in range(size)] for _ in range(size)]
        for node_id, value in self.pi.items():
            if 0 <= int(node_id) < size:
                pi_list[int(node_id)] = clamp01(value)
        for node_id, value in self.rho.items():
            if 0 <= int(node_id) < size:
                rho_list[int(node_id)] = clamp01(value)
        for node_id, value in self.source_pi.items():
            if 0 <= int(node_id) < size:
                source_pi_list[int(node_id)] = clamp01(value)
        for node_id, value in self.aggregate_risk.items():
            if 0 <= int(node_id) < size:
                h_list[int(node_id)] = clamp01(value)
        for (src, dst), value in self.T.items():
            edge_detail = dict((self.details.get("per_edge", {}) or {}).get(f"{int(src)}->{int(dst)}", {}) or {})
            if (
                0 <= int(src) < size
                and 0 <= int(dst) < size
                and bool(edge_detail.get("metric_valid", False))
            ):
                matrix[int(dst)][int(src)] = clamp01(value)
        return {
            "pi": pi_list,
            "source_pi": source_pi_list,
            "rho": rho_list,
            "h": h_list,
            "T": matrix,
            "details": dict(self.details),
            "judge_details": dict(self.details),
            "risk_source": str(self.details.get("mode", "judge_observer")),
        }


class LLMJudgeAtomicRiskObserver:
    """Atomic claim observer used by the main LLM-as-Judge experiment path."""

    def __init__(
        self,
        *,
        extract_claims_fn: Optional[Callable[[str, str, str], Tuple[List[Dict[str, Any]], Dict[str, Any]]]] = None,
        judge_inheritance_fn: Optional[Callable[..., Tuple[Dict[str, Any], Dict[str, Any]]]] = None,
        claim_match_threshold: float = 0.72,
        high_risk_claim_threshold: float = 0.5,
        max_claims_per_response: int = 8,
        transmission_mode: str = "semantic_judge",
    ) -> None:
        self.extract_claims_fn = extract_claims_fn
        self.judge_inheritance_fn = judge_inheritance_fn
        self.claim_match_threshold = float(claim_match_threshold)
        self.high_risk_claim_threshold = clamp01(high_risk_claim_threshold)
        self.max_claims_per_response = int(max(1, max_claims_per_response))
        if transmission_mode not in {"semantic_judge", "structural_edge"}:
            raise ValueError(f"Unsupported transmission_mode={transmission_mode!r}")
        self.transmission_mode = str(transmission_mode)

    def _fallback_extract_claims(
        self,
        *,
        response_text: str,
    ) -> List[Dict[str, Any]]:
        claims: List[Dict[str, Any]] = []
        for idx, part in enumerate(str(response_text or "").replace("\n", ". ").split("."), start=1):
            text = part.strip(" \t\r\n-")
            if not text:
                continue
            claims.append(
                {
                    "claim_id": f"claim_{idx}",
                    "text": text,
                    "support_status": "uncertain",
                    "risk": 0.5,
                    "importance": 0.5,
                    "metric_valid": False,
                }
            )
            if len(claims) >= self.max_claims_per_response:
                break
        if not claims:
            claims.append(
                {
                    "claim_id": "claim_1",
                    "text": str(response_text or "").strip(),
                    "support_status": "uncertain",
                    "risk": 0.5 if str(response_text or "").strip() else 0.0,
                    "importance": 1.0 if str(response_text or "").strip() else 0.0,
                    "metric_valid": False,
                }
            )
        return claims

    @staticmethod
    def _normalize_claim_rows(rows: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows or [], start=1):
            if not isinstance(row, dict):
                continue
            text = str(row.get("text", "") or "").strip()
            if not text:
                continue
            normalized.append(
                {
                    "claim_id": str(row.get("claim_id", f"claim_{idx}")),
                    "text": text,
                    "support_status": str(row.get("support_status", "uncertain") or "uncertain").strip().lower(),
                    "risk": clamp01(float(row.get("risk", 0.5) or 0.0)),
                    "importance": clamp01(float(row.get("importance", 0.5) or 0.0)),
                    "metric_valid": bool(row.get("metric_valid", True)),
                }
            )
        return normalized

    @staticmethod
    def _weighted_mean_claim_risk(claims: List[Dict[str, Any]]) -> float:
        valid_claims = [
            claim for claim in claims
            if bool(claim.get("metric_valid", False))
        ]
        total_weight = 0.0
        total_score = 0.0
        for claim in valid_claims:
            importance = clamp01(float(claim.get("importance", 0.0) or 0.0))
            risk = clamp01(float(claim.get("risk", 0.0) or 0.0))
            total_weight += importance
            total_score += importance * risk
        if total_weight <= 0.0:
            risks = [
                clamp01(float(claim.get("risk", 0.0) or 0.0))
                for claim in valid_claims
            ]
            return clamp01(sum(risks) / len(risks)) if risks else 0.0
        return clamp01(total_score / total_weight)

    @classmethod
    def _resolve_claim_risk(
        cls,
        claims: List[Dict[str, Any]],
        *,
        fallback_pi: Any = None,
        response_text: str = "",
    ) -> tuple[float, bool, str | None]:
        valid_claims = [
            claim for claim in claims
            if bool(claim.get("metric_valid", False))
        ]
        if valid_claims:
            return cls._weighted_mean_claim_risk(valid_claims), True, None
        if fallback_pi is not None:
            return clamp01(float(fallback_pi)), False, "previous_valid_observation"
        neutral_risk = 0.5 if str(response_text or "").strip() else 0.0
        return neutral_risk, False, "neutral_invalid_observation"

    @staticmethod
    def _value_for_agent(mapping: Any, agent_id: int, default: Any = None) -> Any:
        if not isinstance(mapping, dict):
            return default
        if agent_id in mapping:
            return mapping[agent_id]
        return mapping.get(str(agent_id), default)

    @staticmethod
    def _claim_similarity(text_a: str, text_b: str) -> float:
        return float(SequenceMatcher(None, str(text_a).lower(), str(text_b).lower()).ratio())

    def _fallback_inheritance(
        self,
        *,
        sender_claim: Dict[str, Any],
        receiver_before_text: str,
        receiver_after_text: str,
    ) -> Dict[str, Any]:
        claim_text = str(sender_claim.get("text", "") or "").strip()
        before_text = str(receiver_before_text or "").strip()
        after_text = str(receiver_after_text or "").strip()
        if not before_text:
            return {
                "matched": False,
                "inheritance_type": "none",
                "transmission": 0.0,
                "metric_valid": False,
                "observation_status": "missing_before_response",
            }

        claim_lower = claim_text.lower()
        before_lower = before_text.lower()
        after_lower = after_text.lower()
        before_similarity = self._claim_similarity(claim_text, before_text)
        after_similarity = self._claim_similarity(claim_text, after_text)
        before_present = bool(claim_lower and claim_lower in before_lower) or (
            before_similarity >= self.claim_match_threshold
        )
        after_present = bool(claim_lower and claim_lower in after_lower) or (
            after_similarity >= self.claim_match_threshold
        )
        matched = bool(after_present and not before_present)
        return {
            "matched": matched,
            "inheritance_type": "inherit" if matched else "none",
            "transmission": clamp01(after_similarity) if matched else 0.0,
            "metric_valid": False,
            "observation_status": "heuristic_before_after_fallback",
        }

    def _call_inheritance_judge(
        self,
        *,
        question: str,
        sender_claim: Dict[str, Any],
        receiver_before_text: str,
        receiver_after_text: str,
        online_reference_context: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self.judge_inheritance_fn is None:
            return self._fallback_inheritance(
                sender_claim=sender_claim,
                receiver_before_text=receiver_before_text,
                receiver_after_text=receiver_after_text,
            ), {}

        public_sender_claim = {
            "claim_id": str(sender_claim.get("claim_id", "")),
            "text": str(sender_claim.get("text", "")),
        }
        supports_before_after = True
        try:
            parameters = list(inspect.signature(self.judge_inheritance_fn).parameters.values())
            positional = [
                parameter
                for parameter in parameters
                if parameter.kind
                in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            supports_before_after = bool(
                any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters)
                or len(positional) >= 5
            )
        except (TypeError, ValueError):
            pass

        if supports_before_after:
            return self.judge_inheritance_fn(
                question,
                public_sender_claim,
                receiver_before_text,
                receiver_after_text,
                online_reference_context,
            )
        # Compatibility for older injected callbacks. The main method always uses before/after.
        return self.judge_inheritance_fn(
            question,
            public_sender_claim,
            receiver_after_text,
            online_reference_context,
        )

    @staticmethod
    def _merge_audit_counters(audit: Dict[str, Any], extra: Dict[str, Any]) -> None:
        for key in (
            "risk_judge_prompt_tokens",
            "risk_judge_completion_tokens",
            "risk_judge_capped_count",
            "risk_judge_invalid_count",
            "risk_judge_schema_retry_count",
            "risk_judge_call_count",
            "risk_claim_judge_prompt_tokens",
            "risk_claim_judge_completion_tokens",
            "risk_claim_judge_call_count",
            "transmission_judge_prompt_tokens",
            "transmission_judge_completion_tokens",
            "transmission_judge_call_count",
            "missing_before_response_count",
        ):
            audit[key] = float(audit.get(key, 0.0) or 0.0) + float(extra.get(key, 0.0) or 0.0)

    def estimate_risk_state(
        self,
        question: str,
        agent_outputs: Dict[int, str],
        graph: Dict[str, Any],
        history: Optional[Dict[str, Any]] = None,
    ) -> RiskState:
        history = history or {}
        online_reference_context = str(history.get("online_reference_context", "") or "")
        before_outputs = history.get("sender_outputs")
        has_before_snapshot = isinstance(before_outputs, dict)
        if not has_before_snapshot:
            before_outputs = {}
        sender_outputs = dict(before_outputs) if has_before_snapshot else dict(agent_outputs)
        cached_sender_claims = history.get("sender_claims_by_agent")
        cached_sender_pi = history.get("sender_pi_by_agent")
        cached_sender_texts = history.get("sender_observation_texts")
        claims_by_agent: Dict[int, List[Dict[str, Any]]] = {}
        sender_claims_by_agent: Dict[int, List[Dict[str, Any]]] = {}
        pi: Dict[int, float] = {}
        source_pi: Dict[int, float] = {}
        rho: Dict[int, float] = {}
        edge_map: Dict[Tuple[int, int], float] = {}
        audit: Dict[str, Any] = {
            "risk_judge_prompt_tokens": 0.0,
            "risk_judge_completion_tokens": 0.0,
            "risk_judge_capped_count": 0.0,
            "risk_judge_invalid_count": 0.0,
            "risk_judge_call_count": 0.0,
            "risk_claim_judge_prompt_tokens": 0.0,
            "risk_claim_judge_completion_tokens": 0.0,
            "risk_claim_judge_call_count": 0.0,
            "transmission_judge_prompt_tokens": 0.0,
            "transmission_judge_completion_tokens": 0.0,
            "transmission_judge_call_count": 0.0,
            "missing_before_response_count": 0.0,
        }
        details: Dict[str, Any] = {
            "mode": "judge_observer",
            "risk_estimator": "llm_judge_atomic_observer",
            "risk_judge_prompt_version": RISK_JUDGE_PROMPT_VERSION,
            "transmission_judge_prompt_version": TRANSMISSION_JUDGE_PROMPT_VERSION,
            "risk_context_has_peer_messages": False,
            "online_judge_has_gold": False,
            "transmission_uses_before_after": self.transmission_mode == "semantic_judge",
            "transmission_sender_risk_visible": False,
            "transmission_estimator_mode": self.transmission_mode,
            "current_claim_cache_hit_count": 0,
            "online_risk_observer_oracle_free": True,
            "online_risk_observer_uses_gold": False,
            "per_agent": {},
            "per_edge": {},
            "current_observation_texts": {int(k): str(v or "") for k, v in agent_outputs.items()},
            "sender_observation_texts": {int(k): str(v or "") for k, v in sender_outputs.items()},
        }

        for agent_id, output_text in agent_outputs.items():
            cached_text = str(self._value_for_agent(cached_sender_texts, int(agent_id), "") or "")
            cached_claim_rows = self._value_for_agent(cached_sender_claims, int(agent_id))
            normalized_cached_claims = (
                self._normalize_claim_rows(cached_claim_rows)
                if cached_claim_rows is not None
                else []
            )
            can_reuse_current_claims = (
                bool(history.get("reuse_unchanged_claim_observations", False))
                and cached_text == str(output_text or "")
                and bool(normalized_cached_claims)
                and all(bool(claim.get("metric_valid", False)) for claim in normalized_cached_claims)
            )
            if can_reuse_current_claims:
                claims = normalized_cached_claims
                claim_audit = {}
                details["current_claim_cache_hit_count"] += 1
            elif self.extract_claims_fn is not None:
                raw_claims, claim_audit = self.extract_claims_fn(question, output_text, online_reference_context)
                claims = self._normalize_claim_rows(raw_claims)
                self._merge_audit_counters(audit, claim_audit)
            else:
                claims = self._fallback_extract_claims(response_text=output_text)
                claim_audit = {}
            claims_by_agent[int(agent_id)] = claims
            cached_pi_value = self._value_for_agent(cached_sender_pi, int(agent_id))
            cached_claims_valid = bool(normalized_cached_claims) and all(
                bool(claim.get("metric_valid", False))
                for claim in normalized_cached_claims
            )
            pi_value, pi_metric_valid, pi_fallback_source = self._resolve_claim_risk(
                claims,
                fallback_pi=cached_pi_value if cached_claims_valid else None,
                response_text=output_text,
            )
            pi[int(agent_id)] = pi_value
            details["per_agent"][int(agent_id)] = {
                "claims": claims,
                "pi_i": pi_value,
                "metric_valid": pi_metric_valid,
                "pi_metric_valid": pi_metric_valid,
                "pi_fallback_applied": not pi_metric_valid,
                "pi_fallback_source": pi_fallback_source,
                "valid_claim_count": sum(
                    1 for claim in claims
                    if bool(claim.get("metric_valid", False))
                ),
            }

        source_ids = {int(edge[0]) for edge in graph.get("edges", []) or [] if isinstance(edge, (list, tuple)) and len(edge) >= 2}
        for source_id in source_ids:
            source_text = str(self._value_for_agent(sender_outputs, source_id, "") or "")
            current_text = str(agent_outputs.get(source_id, "") or "")
            cached_text = str(self._value_for_agent(cached_sender_texts, source_id, "") or "")
            cached_claim_rows = self._value_for_agent(cached_sender_claims, source_id)

            if source_text == current_text and source_id in claims_by_agent:
                source_claims = claims_by_agent[source_id]
                source_pi_value = pi.get(source_id, 0.0)
            elif cached_claim_rows is not None and cached_text == source_text:
                source_claims = self._normalize_claim_rows(cached_claim_rows)
                cached_pi_value = self._value_for_agent(cached_sender_pi, source_id)
                source_pi_value, source_pi_valid, source_pi_fallback_source = self._resolve_claim_risk(
                    source_claims,
                    fallback_pi=(
                        cached_pi_value
                        if source_claims
                        and all(bool(claim.get("metric_valid", False)) for claim in source_claims)
                        else None
                    ),
                    response_text=source_text,
                )
            elif self.extract_claims_fn is not None:
                raw_claims, source_audit = self.extract_claims_fn(question, source_text, online_reference_context)
                source_claims = self._normalize_claim_rows(raw_claims)
                self._merge_audit_counters(audit, source_audit)
                source_pi_value, source_pi_valid, source_pi_fallback_source = self._resolve_claim_risk(
                    source_claims,
                    response_text=source_text,
                )
            else:
                source_claims = self._fallback_extract_claims(response_text=source_text)
                source_pi_value, source_pi_valid, source_pi_fallback_source = self._resolve_claim_risk(
                    source_claims,
                    response_text=source_text,
                )

            if source_text == current_text and source_id in claims_by_agent:
                source_pi_valid = bool(
                    details["per_agent"].get(source_id, {}).get("pi_metric_valid", False)
                )
                source_pi_fallback_source = details["per_agent"].get(source_id, {}).get(
                    "pi_fallback_source"
                )

            sender_claims_by_agent[source_id] = source_claims
            source_pi[source_id] = clamp01(source_pi_value)
            details["per_agent"].setdefault(source_id, {})["source_message_claims"] = source_claims
            details["per_agent"][source_id]["source_message_pi"] = source_pi[source_id]
            details["per_agent"][source_id]["source_message_pi_metric_valid"] = bool(source_pi_valid)
            details["per_agent"][source_id]["source_message_pi_fallback_source"] = source_pi_fallback_source

        incoming_prob_factors: Dict[int, List[float]] = {int(agent_id): [] for agent_id in agent_outputs}
        for raw_edge in graph.get("edges", []) or []:
            if not isinstance(raw_edge, (list, tuple)) or len(raw_edge) < 2:
                continue
            src = int(raw_edge[0])
            dst = int(raw_edge[1])
            src_claims = sender_claims_by_agent.get(src, [])
            if self.transmission_mode == "structural_edge":
                edge_map[(src, dst)] = 1.0
                details["per_edge"][f"{src}->{dst}"] = {
                    "transmission": 1.0,
                    "matched_claims": [],
                    "metric_valid": True,
                    "observation_status": "structural_active_edge",
                    "source_message_pi": clamp01(source_pi.get(src, 0.0)),
                    "source_message_round": "t-1",
                    "receiver_response_round": "t",
                    "missing_before_response": False,
                    "transmission_uses_before_after": False,
                    "transmission_sender_risk_visible": False,
                    "rho_transmission_used": 1.0,
                    "rho_uses_conservative_unknown": False,
                    "structural_transmission_ablation": True,
                }
                incoming_prob_factors.setdefault(dst, []).append(
                    1.0 - clamp01(source_pi.get(src, 0.0))
                )
                continue
            receiver_before_text = str(self._value_for_agent(before_outputs, dst, "") or "")
            receiver_after_text = str(agent_outputs.get(dst, "") or "")
            if not receiver_before_text.strip():
                audit["missing_before_response_count"] += 1
                edge_map[(src, dst)] = 0.0
                details["per_edge"][f"{src}->{dst}"] = {
                    "transmission": 0.0,
                    "matched_claims": [],
                    "metric_valid": False,
                    "observation_status": "missing_before_response",
                    "missing_before_response": True,
                    "transmission_uses_before_after": True,
                    "transmission_sender_risk_visible": False,
                    "rho_transmission_used": 0.0,
                    "rho_uses_conservative_unknown": False,
                }
                incoming_prob_factors.setdefault(dst, []).append(1.0)
                continue
            if not src_claims or not receiver_after_text.strip():
                edge_map[(src, dst)] = 0.0
                details["per_edge"][f"{src}->{dst}"] = {
                    "transmission": 0.0,
                    "matched_claims": [],
                    "metric_valid": False,
                    "observation_status": "invalid_missing_text_or_claims",
                    "missing_before_response": False,
                    "rho_transmission_used": 0.0,
                    "rho_uses_conservative_unknown": False,
                    "invalid_observation_excluded_from_rho": True,
                }
                incoming_prob_factors.setdefault(dst, []).append(1.0)
                continue

            matched_claims: List[Dict[str, Any]] = []
            transmissions: List[float] = []
            evaluated_validities: List[bool] = []
            risky_claim_count = 0
            for sender_claim in src_claims:
                sender_risk = clamp01(float(sender_claim.get("risk", 0.0) or 0.0))
                if (
                    not bool(sender_claim.get("metric_valid", False))
                    or sender_risk < self.high_risk_claim_threshold
                ):
                    continue
                risky_claim_count += 1
                inheritance, inherit_audit = self._call_inheritance_judge(
                    question=question,
                    sender_claim=sender_claim,
                    receiver_before_text=receiver_before_text,
                    receiver_after_text=receiver_after_text,
                    online_reference_context=online_reference_context,
                )
                self._merge_audit_counters(audit, inherit_audit)
                transmission = clamp01(float(inheritance.get("transmission", 0.0) or 0.0))
                evaluated_validities.append(bool(inheritance.get("metric_valid", False)))
                if bool(inheritance.get("matched", False)) and transmission > 0.0:
                    matched_claims.append(
                        {
                            "claim_id": str(sender_claim.get("claim_id", "")),
                            "text": str(sender_claim.get("text", "")),
                            "inheritance_type": str(inheritance.get("inheritance_type", "matched")),
                            "transmission": transmission,
                            "metric_valid": bool(inheritance.get("metric_valid", False)),
                        }
                    )
                    transmissions.append(transmission)
            edge_t = clamp01(sum(transmissions) / len(transmissions)) if transmissions else 0.0
            claims_valid = all(bool(claim.get("metric_valid", False)) for claim in src_claims)
            inheritance_valid = (
                claims_valid
                if risky_claim_count == 0
                else bool(evaluated_validities) and all(evaluated_validities)
            )
            # Infrastructure/schema failures are missing observations, not
            # evidence of maximal semantic transmission.
            rho_edge_t = edge_t if inheritance_valid else 0.0
            edge_map[(src, dst)] = edge_t
            details["per_edge"][f"{src}->{dst}"] = {
                "transmission": edge_t,
                "matched_claims": matched_claims,
                "metric_valid": bool(inheritance_valid),
                "observation_status": "observed" if inheritance_valid else "invalid",
                "source_message_pi": clamp01(source_pi.get(src, 0.0)),
                "high_risk_claim_threshold": self.high_risk_claim_threshold,
                "source_message_round": "t-1",
                "receiver_before_round": "t-1",
                "receiver_response_round": "t",
                "missing_before_response": False,
                "transmission_uses_before_after": True,
                "transmission_sender_risk_visible": False,
                "rho_transmission_used": rho_edge_t,
                "rho_uses_conservative_unknown": False,
                "invalid_observation_excluded_from_rho": not inheritance_valid,
            }
            incoming_prob_factors.setdefault(dst, []).append(
                1.0 - rho_edge_t * clamp01(source_pi.get(src, 0.0))
            )

        for agent_id in agent_outputs:
            factors = incoming_prob_factors.get(int(agent_id), [])
            no_inherit_prob = 1.0
            for value in factors:
                no_inherit_prob *= clamp01(value)
            rho[int(agent_id)] = clamp01(1.0 - no_inherit_prob)
            incoming_edges = [
                row for key, row in details["per_edge"].items()
                if key.endswith(f"->{int(agent_id)}")
            ]
            details["per_agent"].setdefault(int(agent_id), {})["rho_metric_valid"] = all(
                bool(row.get("metric_valid", False)) for row in incoming_edges
            ) if incoming_edges else True

        aggregate = {
            int(agent_id): aggregate_risk_from_pi_rho(pi.get(int(agent_id), 0.0), rho.get(int(agent_id), 0.0))
            for agent_id in agent_outputs
        }
        call_count = max(1.0, float(audit.get("risk_judge_call_count", 0.0) or 0.0))
        audit["risk_judge_capped_rate"] = float(audit.get("risk_judge_capped_count", 0.0) or 0.0) / call_count
        audit["risk_judge_invalid_rate"] = float(audit.get("risk_judge_invalid_count", 0.0) or 0.0) / call_count
        details.update(audit)
        return RiskState(
            pi={int(k): clamp01(v) for k, v in pi.items()},
            rho={int(k): clamp01(v) for k, v in rho.items()},
            T=edge_map,
            source_pi={int(k): clamp01(v) for k, v in source_pi.items()},
            details=details,
            aggregate_risk=aggregate,
            legacy_h=dict(aggregate),
        )
