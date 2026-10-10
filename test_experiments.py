"""Offline checks for the released MMLU-Pro experiments."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from . import build_mmlupro_input
from .config import build_default_config, configure_ablation_strategy_space
from .core.evolution import EvolutionEnvironment, PerturbationEngine
from . import experiment
from .experiment import _has_unanimous_valid_answer, _select_final_answer
from .llm import apply_model_overrides, resolve_model
from .prepare_two_window import (
    FIRST_FAULT_ROUNDS, SECOND_FAULT_ROUNDS, TOTAL_ROUNDS, TwoWindowFaultSpec,
    build_spec, clean_external_r0, question_sha256, sample_distinct_sources,
)
from .two_window import (
    INJECTOR_PROMPT_VERSION, VALIDATION_VERSION, FaultInjector, InjectionResult,
    TwoWindowFaultController, checkpoint_matches_protocol, manifest_meta,
    normalize_target_final_line, protocol_enabled, result_block,
    validate_protocol_inputs, validate_sample_spec,
)
from . import run_main_five_round_experiment
from . import run_robustness_ten_round_experiment
from . import run_component_ablation_experiment
from .utils.answers import parse_mmlupro_choice_answer


class ExperimentModeTests(unittest.TestCase):
    def test_api_keys_follow_call_role_across_model_routes(self):
        with mock.patch.dict(os.environ, {
            "GEN_API_KEY": "generation-test-key",
            "JUDGE_API_KEY": "judge-test-key",
        }, clear=True):
            config = build_default_config()
            self.assertEqual(config.llm.generation_key, "generation-test-key")
            self.assertEqual(config.llm.prompt_key, "generation-test-key")
            self.assertEqual(config.llm.judge_key, "judge-test-key")
            self.assertEqual(config.llm.generation_url, "")
            self.assertEqual(config.llm.judge_url, "")
            with self.assertRaisesRegex(ValueError, "GEN_BASE_URL"):
                build_mmlupro_input.make_client(build_mmlupro_input.resolve_api_config("Qwen/Qwen3-14B"))
            with self.assertRaisesRegex(ValueError, "GEN_BASE_URL"):
                FaultInjector(model_name="Qwen/Qwen3-14B")

            args = SimpleNamespace(
                generation_model="deepseek-v4-pro", prompt_model=None,
                judge_model="Qwen/Qwen3-14B", generation_key_env=None,
                prompt_key_env=None, judge_key_env=None,
            )
            apply_model_overrides(config, args)
            self.assertEqual(config.llm.generation_key, "generation-test-key")
            self.assertEqual(config.llm.prompt_key, "generation-test-key")
            self.assertEqual(config.llm.judge_key, "judge-test-key")
            self.assertEqual(build_mmlupro_input.resolve_api_config("deepseek-v4-pro").api_key, "generation-test-key")
            self.assertEqual(resolve_model("deepseek-v4-pro", role="judge")["key"], "judge-test-key")

    def test_model_and_url_environment_routes(self):
        with mock.patch.dict(os.environ, {
            "GEN_MODEL": "custom-generation-model",
            "GEN_BASE_URL": "https://generation.example.test/v1",
            "GEN_API_KEY": "generation-test-key",
            "JUDGE_MODEL": "custom-judge-model",
            "JUDGE_BASE_URL": "https://judge.example.test/v1",
            "JUDGE_API_KEY": "judge-test-key",
        }, clear=True):
            config = build_default_config()
            self.assertEqual(config.llm.generation_model, "custom-generation-model")
            self.assertEqual(config.llm.prompt_model, "custom-generation-model")
            self.assertEqual(config.llm.generation_url, "https://generation.example.test/v1")
            self.assertEqual(config.llm.judge_model, "custom-judge-model")
            self.assertEqual(config.llm.judge_url, "https://judge.example.test/v1")
            self.assertEqual(resolve_model("deepseek-v4-pro", role="judge")["url"], "https://judge.example.test/v1")
            self.assertEqual(build_mmlupro_input.resolve_api_config(config.llm.generation_model).base_url, "https://generation.example.test/v1")
            with mock.patch("sys.argv", ["build_mmlupro_input"]):
                self.assertEqual(build_mmlupro_input.parse_args().generation_model, "custom-generation-model")
            with mock.patch("sys.argv", ["build_mmlupro_input", "--generation-model", "Qwen/Qwen3-14B"]):
                self.assertEqual(build_mmlupro_input.parse_args().generation_model, "Qwen/Qwen3-14B")
            self.assertEqual(
                EvolutionEnvironment._normalize_base_url_for_deepseek(config.llm.judge_url),
                "https://judge.example.test/v1",
            )

    def test_selector_uses_only_question_options_and_final_responses(self):
        url_env = mock.patch.dict(os.environ, {"JUDGE_BASE_URL": "https://judge.example.test/v1"})
        url_env.start()
        self.addCleanup(url_env.stop)
        calls = []
        usage = {"judge_prompt": 0, "judge_completion": 0, "final_selector_prompt": 0, "final_selector_completion": 0}

        def select(**kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": '{"answer":"B"}'}}], "usage": {"prompt_tokens": 12, "completion_tokens": 3}}

        def accumulate(tokens, prompt_key, completion_key):
            usage[prompt_key] += tokens["prompt_tokens"]
            usage[completion_key] += tokens["completion_tokens"]

        env = SimpleNamespace(
            graph_dict={"question": "Choose one", "choices": ["first", "second", "third"], "answer": "secret-gold-value"},
            _call_llm_with_retry=lambda _label, call: call(),
            _deepseek_chat_completion=select,
            _extract_usage=lambda response: response["usage"],
            _extract_chat_content=lambda response: response["choices"][0]["message"]["content"],
            _accumulate_usage=accumulate,
        )
        selected = _select_final_answer(env, build_default_config(), {"0": "Answer: A", "1": "Answer: B"}, ["A", "B", "C"])
        self.assertEqual(selected["answer"], "B")
        self.assertEqual(selected["status"], "selected")
        self.assertEqual(calls[0]["model_name"], "deepseek-v4-pro")
        self.assertNotIn("secret-gold-value", str(calls[0]["messages"]))
        self.assertEqual(usage["final_selector_prompt"], 12)

        calls.clear()
        consensus = _select_final_answer(env, build_default_config(), {"0": "Answer: A", "1": "Answer: A"}, ["A", "B", "C"])
        self.assertEqual(consensus["status"], "unanimous_consensus")
        self.assertFalse(calls)

        env._extract_chat_content = lambda _response: '{"answer":"Z"}'
        invalid = _select_final_answer(env, build_default_config(), {"0": "Answer: A", "1": "Answer: B"}, ["A", "B", "C"])
        self.assertEqual(invalid["status"], "failed")
        self.assertIsNone(invalid["answer"])

    def test_majority_and_selector_accuracies_remain_separate(self):
        env = SimpleNamespace(
            graph_dict={
                "question": "Choose one", "choices": ["first", "second", "third"], "answer": "B",
                "nodes": [{"response": "Answer: A"}, {"response": "Answer: A"}, {"response": "Answer: B"}],
            },
            agents={index: SimpleNamespace(h_final=0.2, rho=0.1) for index in range(3)},
        )
        with mock.patch.object(experiment, "_select_final_answer", return_value={
            "answer": "B", "status": "selected", "model": "deepseek-v4-pro", "error": None,
        }):
            result = experiment._build_final_answer_eval(env, build_default_config())
        self.assertEqual(result["majority_vote_parsed_answer"], "A")
        self.assertEqual(result["selector_parsed_answer"], "B")
        self.assertEqual(result["majority_vote_accuracy"], 0.0)
        self.assertEqual(result["selector_accuracy"], 1.0)

    def test_consensus_requires_every_agent_to_have_the_same_valid_answer(self):
        responses = {0: "Answer: A", 1: "Answer: A", 2: "Answer: A"}
        env = SimpleNamespace(
            agents={0: object(), 1: object(), 2: object()},
            _snapshot_agent_outputs=lambda: responses,
            _parse_choice_answer=lambda text: text[-1] if text.startswith("Answer: ") else None,
        )
        self.assertTrue(_has_unanimous_valid_answer(env))
        responses[2] = "Answer: B"
        self.assertFalse(_has_unanimous_valid_answer(env))
        responses[2] = "invalid"
        self.assertFalse(_has_unanimous_valid_answer(env))

    def test_paper_default_rounds_and_component_ablations(self):
        from .config import FUNCTIONAL_ABLATION_CHOICES

        self.assertEqual(build_default_config().egt.num_generations, 5)
        self.assertEqual(
            set(FUNCTIONAL_ABLATION_CHOICES),
            {"full", "no_topology_governance", "no_strategy_evolution"},
        )

    def test_main_runner_selects_rows_and_derives_episode_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            input_path = folder / "mmlupro.jsonl"
            input_path.write_text('{"question_id":"q0"}\n{"question_id":"q1"}\n', encoding="utf-8")
            with mock.patch.object(run_main_five_round_experiment.subprocess, "run") as run:
                run.return_value.returncode = 0
                exit_code = run_main_five_round_experiment.main([
                    "--input-graph", str(input_path), "--output-dir", str(folder / "results"),
                    "--start-idx", "1", "--max-samples", "1", "--seeds", "1,2",
                ])
            self.assertEqual(exit_code, 0)
            self.assertEqual(run.call_count, 2)
            first_command = run.call_args_list[0].args[0]
            self.assertEqual(first_command[first_command.index("--line-idx") + 1], "1")
            self.assertEqual(first_command[first_command.index("--seed") + 1], str(run_main_five_round_experiment.effective_seed(1, 1)))
            self.assertEqual(first_command[first_command.index("--generations") + 1], "5")
            self.assertIn("sample_1_base_seed_1.json", first_command[first_command.index("--output-json") + 1])
            self.assertIn("--final-judge", first_command)

    def test_robustness_entry_point_sets_ten_round_protocol(self):
        with mock.patch.object(run_robustness_ten_round_experiment.experiment, "main") as run:
            run_robustness_ten_round_experiment.main([
                "--two-window-fault-manifest", "manifest.jsonl",
                "--fault-injector-model", "Qwen/Qwen3-14B",
            ])
        args = run.call_args.args[0]
        self.assertEqual(args[args.index("--generations") + 1], "10")
        self.assertIn("--enable-peer-context", args)
        self.assertEqual(args[args.index("--peer-context-max-chars") + 1], "0")

    def test_ablation_entry_point_selects_each_five_round_mode(self):
        for mode in run_component_ablation_experiment.ABLATIONS:
            with self.subTest(mode=mode), mock.patch.object(run_component_ablation_experiment.experiment, "main") as run:
                run_component_ablation_experiment.main(["--ablation", mode, "--line-idx", "1"])
                args = run.call_args.args[0]
                self.assertEqual(args[args.index("--generations") + 1], "5")
                option = "--strategy-mode" if mode == "no_self" else "--functional-ablation-mode"
                self.assertEqual(args[args.index(option) + 1], mode)
                self.assertEqual(args[args.index("--line-idx") + 1], "1")

    def test_ablation_and_robustness_protocols_cannot_be_combined(self):
        with mock.patch.object(experiment, "main") as run, mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                run_component_ablation_experiment.main([
                    "--ablation", "no_self", "--two-window-fault-manifest", "manifest.jsonl",
                ])
            with self.assertRaises(SystemExit):
                run_robustness_ten_round_experiment.main([
                    "--two-window-fault-manifest", "manifest.jsonl",
                    "--fault-injector-model", "Qwen/Qwen3-14B",
                    "--strategy-mode", "no_self",
                ])
            run.assert_not_called()

        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            experiment.main([
                "--two-window-fault-manifest", "manifest.jsonl",
                "--functional-ablation-mode", "no_strategy_evolution",
            ])

    def test_main_input_builder_creates_one_injected_graph_without_api(self):
        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

            def create(self, **kwargs):
                prompt = kwargs["messages"][-1]["content"]
                requested = re.search(r"Requested final option:\s*([A-J])", prompt)
                answer = requested.group(1) if requested else "A"
                content = (
                    f"The comparison appears to favor option {answer} after following "
                    f"the decisive step in the calculation.\nAnswer: {answer}"
                )
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            qa_path = folder / "qa.jsonl"
            roles_path = folder / "roles.jsonl"
            qa_path.write_text(json.dumps({
                "question_id": "q1", "question": "What is 1 + 1?",
                "choices": ["2", "3", "4", "5"], "answer": "A",
                "subject": "math", "split": "test",
            }) + "\n", encoding="utf-8")
            args = argparse.Namespace(
                qa_path=qa_path, roles_path=roles_path, output=folder / "out.jsonl",
                role_seed=42, roles_per_domain=5, include_general_roles=False,
                subject="math", split="test", total_count=0, per_subject_count=0,
                sample_mode="first", selection_seed=1, count=1, seed=1,
                num_agents=5, generation_model="Qwen/Qwen3-14B",
                generation_key_env=None, concurrency=1, temperature=0.0,
                max_tokens=1024, max_attempts=1,
                checkpoint_path=folder / "checkpoint.jsonl", resume=False,
            )
            build_mmlupro_input.prepare_inputs(args)
            self.assertTrue(roles_path.is_file())
            route = SimpleNamespace(
                name="fake", model="Qwen/Qwen3-14B",
                base_url="https://example.test", api_key="fake",
            )
            with mock.patch.object(build_mmlupro_input, "resolve_api_config", return_value=route), \
                 mock.patch.object(build_mmlupro_input, "make_client", return_value=FakeClient()), \
                 mock.patch.object(build_mmlupro_input, "progress_bar", return_value=None):
                rows = asyncio.run(build_mmlupro_input.build_mmlupro_input_graphs(args))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_builder"], "build_mmlupro_input.py")
            self.assertNotIn("base_url", rows[0]["meta"])
            self.assertEqual(len(rows[0]["nodes"]), 5)
            self.assertNotEqual(rows[0]["injection_info"]["wrong_answer"], "A")

    def test_paper_subject_selection_and_topology_schedule(self):
        subjects = ("mathematics", "chemistry", "law", "philosophy")
        rows = [
            {"question_id": f"{subject}-{index}", "subject": subject, "split": "test"}
            for subject in subjects for index in range(110)
        ]
        selected = build_mmlupro_input.filter_qa_rows(
            rows, ",".join(subjects), "test", 0, 101, "random", 1,
        )
        self.assertEqual(len(selected), 404)
        for subject in subjects:
            self.assertEqual(sum(row["subject"] == subject for row in selected), 101)
        schedule = build_mmlupro_input.build_balanced_topology_schedule(404, 1)
        topology_counts = {
            topology: schedule.count(topology)
            for topology in ("fully-connected", "star", "ring", "sparse25", "sparse50", "sparse75")
        }
        self.assertEqual(sum(topology_counts.values()), 404)
        self.assertLessEqual(max(topology_counts.values()) - min(topology_counts.values()), 1)

    def test_mmlupro_parser_accepts_a_to_j_and_respects_choice_count(self):
        self.assertEqual(parse_mmlupro_choice_answer("Reasoning.\nAnswer: J"), "J")
        self.assertEqual(parse_mmlupro_choice_answer("Reasoning.\nAnswer: H", valid_letters=list("ABCDEFGH")), "H")
        self.assertIsNone(parse_mmlupro_choice_answer("Answer: J", valid_letters=list("ABCDEFGH")))

    def test_action_ablation_restricts_strategy_space(self):
        expected_zero_bit = {"no_self": 2}
        for mode, bit in expected_zero_bit.items():
            with self.subTest(mode=mode):
                config = build_default_config()
                configure_ablation_strategy_space(config.egt, mode)
                self.assertTrue(config.egt.strategy_set)
                self.assertTrue(all(strategy[bit] == 0 for strategy in config.egt.strategy_set))

    def test_generation_perturbation_is_opt_in_and_respects_window(self):
        config = build_default_config()
        graph = {
            "task": {"perturbation": {"target": 1, "wrong_answer": "Answer: C"}},
            "final_raw_outputs": ["Answer: A", "Answer: C"],
        }
        config.perturbation.trigger_generation = 3
        engine = PerturbationEngine.from_graph_dict(graph, config)
        self.assertIsNone(engine.apply_perturbation(1, 3))
        config.perturbation.enabled = True
        self.assertIsNone(engine.apply_perturbation(1, 2))
        self.assertIsNone(engine.apply_perturbation(0, 3))
        self.assertEqual(engine.apply_perturbation(1, 3), "Answer: C")
        self.assertIsNone(engine.apply_perturbation(1, 4))
        config.perturbation.persist_across_gens = True
        persistent = PerturbationEngine.from_graph_dict(graph, config)
        self.assertEqual(persistent.apply_perturbation(1, 4), "Answer: C")

    def test_generation_perturbation_uses_injected_message(self):
        config = build_default_config()
        config.perturbation.enabled = True
        graph = {
            "injection_info": {
                "target": 1,
                "wrong_answer": "C",
                "poisoned_message": "Misleading reasoning.\nAnswer: C",
            },
            "initial_raw_outputs": ["Answer: A", "Misleading reasoning.\nAnswer: C"],
        }
        engine = PerturbationEngine.from_graph_dict(graph, config)
        self.assertEqual(engine.apply_perturbation(1, config.perturbation.trigger_generation), "Misleading reasoning.\nAnswer: C")

    def test_model_selection_uses_public_routes_without_a_key(self):
        config = build_default_config()
        class Args:
            generation_model = "Qwen/Qwen3-14B"
            judge_model = "deepseek-v4-flash"
            prompt_model = None
        apply_model_overrides(config, Args())
        self.assertEqual(config.llm.generation_model, "Qwen/Qwen3-14B")
        self.assertEqual(config.llm.prompt_model, "Qwen/Qwen3-14B")
        self.assertEqual(config.llm.judge_model, "deepseek-v4-flash")

    def test_two_window_builder_produces_clean_input_and_matching_manifest(self):
        row = {
            "question_id": "sample-1",
            "question": "Which option is correct?",
            "choices": ["right", "wrong", "other"],
            "gold": "A",
            "num_agents": 3,
            "nodes": [{"initial_answer": "B", "is_injected": True}, {}, {}],
            "round0_answers_before_injection": ["A", "A", "C"],
            "round0_raw_outputs_before_injection": ["Answer: A", "Answer: A", "Answer: C"],
            "injection_info": {"wrong_answer": "B", "poisoned_message": "Mistaken reasoning.\nAnswer: B"},
        }
        clean = clean_external_r0(row)
        spec = build_spec(row, sample_idx=0, master_seed=1)
        self.assertTrue(clean["external_clean_r0"])
        self.assertNotIn("injection_info", clean)
        self.assertEqual(clean["initial_answers"], ["A", "A", "C"])
        validate_sample_spec(spec=spec, sample_idx=0, sample=clean)


def _spec() -> TwoWindowFaultSpec:
    fallback = "A concise but mistaken explanation.\n\nAnswer: C"
    return TwoWindowFaultSpec(
        sample_idx=0,
        question_id="q0",
        question_sha256="0" * 64,
        num_agents=5,
        total_rounds=TOTAL_ROUNDS,
        master_seed=1,
        sample_schedule_seed=42,
        source_a=1,
        source_b=4,
        first_fault_rounds=FIRST_FAULT_ROUNDS,
        second_fault_rounds=SECOND_FAULT_ROUNDS,
        target_wrong_option="C",
        target_option_text="target distractor",
        canonical_fallback_response=fallback,
        fallback_sha256=hashlib.sha256(fallback.encode("utf-8")).hexdigest(),
    )


class _FakeInjector:
    model_name = "fake-injector"

    def inject(self, **kwargs):
        return InjectionResult(
            response="Edited reasoning.\nAnswer: C",
            success=True,
            mode="online_rewrite",
            attempts=1,
            validation={"valid": True},
        )


class TwoWindowProtocolTests(unittest.TestCase):
    def test_protocol_is_output_noop_when_manifest_is_absent(self):
        args = argparse.Namespace(two_window_fault_manifest=None)
        self.assertFalse(protocol_enabled(args))
        self.assertEqual(manifest_meta(args), {})
        self.assertEqual(result_block(None), {})

    def test_distinct_sources_are_stable(self):
        first = sample_distinct_sources(num_agents=5, sample_seed=123)
        second = sample_distinct_sources(num_agents=5, sample_seed=123)
        self.assertEqual(first, second)
        self.assertNotEqual(first[0], first[1])

    def test_round_mapping(self):
        spec = _spec()
        self.assertEqual(spec.source_for_round_index(2), ("A", 1))
        self.assertEqual(spec.source_for_round_index(3), ("A", 1))
        self.assertIsNone(spec.source_for_round_index(4))
        self.assertEqual(spec.source_for_round_index(6), ("B", 4))
        self.assertEqual(spec.source_for_round_index(7), ("B", 4))
        self.assertEqual(spec.phase_for_round_index(4), "transition_a")
        self.assertEqual(spec.phase_for_round_index(5), "recovery_a")
        self.assertEqual(spec.phase_for_round_index(8), "transition_b")
        self.assertEqual(spec.phase_for_round_index(9), "recovery_b")

    def test_controller_only_changes_scheduled_source(self):
        controller = TwoWindowFaultController(spec=_spec(), injector=_FakeInjector())
        parser = lambda text: "C" if "Answer: C" in text else "A"
        untouched = controller.apply_if_scheduled(
            round_index=2,
            agent_id=0,
            question_text="q",
            response="Answer: A",
            parse_answer=parser,
        )
        edited = controller.apply_if_scheduled(
            round_index=2,
            agent_id=1,
            question_text="q",
            response="Answer: A",
            parse_answer=parser,
        )
        self.assertEqual(untouched, "Answer: A")
        self.assertEqual(edited, "Edited reasoning.\nAnswer: C")
        self.assertEqual(len(controller.events), 1)

    def test_inactive_source_is_not_resampled(self):
        controller = TwoWindowFaultController(spec=_spec(), injector=_FakeInjector())
        controller.record_inactive_if_scheduled(round_index=2, active_agent_ids={0, 2, 3, 4})
        event = controller.events[0]
        self.assertEqual(event["agent_id"], 1)
        self.assertEqual(event["injection_mode"], "scheduled_but_inactive")
        self.assertFalse(event["injection_success"])
        summary = controller.summary()
        self.assertEqual(summary["source_inactive_count"], 1)
        self.assertEqual(summary["injection_attempt_count"], 0)
        self.assertEqual(summary["network_delivered_fault_count"], 0)

    def test_delivery_is_distinct_from_injection_success(self):
        controller = TwoWindowFaultController(spec=_spec(), injector=_FakeInjector())
        parser = lambda text: "C" if "Answer: C" in text else "A"
        controller.apply_if_scheduled(
            round_index=2,
            agent_id=1,
            question_text="q",
            response="Answer: A",
            parse_answer=parser,
        )
        controller.finalize_round(
            round_index=2,
            outgoing_responses={1: "Edited reasoning.\nAnswer: C"},
            visible_agent_ids=set(),
        )
        summary = controller.summary()
        self.assertEqual(summary["successful_injection_count"], 1)
        self.assertEqual(summary["network_delivered_fault_count"], 0)

    def test_delivery_requires_actual_next_round_peer_consumption(self):
        controller = TwoWindowFaultController(spec=_spec(), injector=_FakeInjector())
        parser = lambda text: "C" if "Answer: C" in text else "A"
        controller.apply_if_scheduled(
            round_index=2,
            agent_id=1,
            question_text="q",
            response="Answer: A",
            parse_answer=parser,
        )
        controller.finalize_round(
            round_index=2,
            outgoing_responses={1: "Edited reasoning.\nAnswer: C"},
            visible_agent_ids={1},
        )
        controller.record_next_round_consumption(
            current_round_index=3,
            recipient_ids_by_sender={1: {1, 3}},
        )
        event = controller.events[0]
        self.assertTrue(event["delivery_observed"])
        self.assertEqual(event["delivery_recipient_agent_ids"], [3])
        self.assertTrue(event["delivered_to_next_round_context"])

    def test_peer_context_evidence_overrides_topology_inference(self):
        controller = TwoWindowFaultController(spec=_spec(), injector=_FakeInjector())
        parser = lambda text: "C" if "Answer: C" in text else "A"
        controller.apply_if_scheduled(
            round_index=2,
            agent_id=1,
            question_text="q",
            response="Answer: A",
            parse_answer=parser,
        )
        controller.finalize_round(
            round_index=2,
            outgoing_responses={1: "Edited reasoning.\nAnswer: C"},
            visible_agent_ids={1},
            parse_answer=parser,
        )
        controller.record_next_round_consumption(
            current_round_index=3,
            recipient_ids_by_sender={1: {3}},
        )
        controller.record_peer_context_evidence(
            current_round_index=3,
            recipient_agent_id=3,
            peer_context_text="Question without the sender response",
            peer_messages_by_sender={},
        )
        event = controller.events[0]
        self.assertEqual(event["topology_delivery_recipient_agent_ids"], [3])
        self.assertFalse(event["source_response_present"])
        self.assertFalse(event["delivered_to_next_round_context"])
        controller.record_peer_context_evidence(
            current_round_index=3,
            recipient_agent_id=3,
            peer_context_text="Peer says: Edited reasoning.\nAnswer: C",
            peer_messages_by_sender={1: "Edited reasoning.\nAnswer: C"},
        )
        event = controller.events[0]
        self.assertTrue(event["source_response_present"])
        self.assertEqual(event["delivery_recipient_agent_ids"], [3])
        self.assertTrue(event["delivered_to_next_round_context"])
        self.assertTrue(event["peer_context_evidence"][0]["peer_context_hash"])

    def test_repeated_short_answer_from_another_sender_is_not_delivery(self):
        controller = TwoWindowFaultController(spec=_spec(), injector=_FakeInjector())
        parser = lambda text: "C" if "Answer: C" in text else "A"
        controller.apply_if_scheduled(
            round_index=2,
            agent_id=1,
            question_text="q",
            response="Answer: A",
            parse_answer=parser,
        )
        controller.finalize_round(
            round_index=2,
            outgoing_responses={1: "Answer: C"},
            visible_agent_ids={1},
            parse_answer=parser,
        )
        controller.record_peer_context_evidence(
            current_round_index=3,
            recipient_agent_id=3,
            peer_context_text="Agent 2 says: Answer: C",
            peer_messages_by_sender={2: "Answer: C"},
        )
        self.assertFalse(controller.events[0]["source_response_present"])

    def test_governance_recovery_prevents_fault_delivery(self):
        controller = TwoWindowFaultController(spec=_spec(), injector=_FakeInjector())
        parser = lambda text: "C" if "Answer: C" in text else "A"
        controller.apply_if_scheduled(
            round_index=2,
            agent_id=1,
            question_text="q",
            response="Answer: A",
            parse_answer=parser,
        )
        controller.finalize_round(
            round_index=2,
            outgoing_responses={1: "Recovered reasoning.\nAnswer: A"},
            visible_agent_ids={1},
            parse_answer=parser,
        )
        controller.record_next_round_consumption(
            current_round_index=3,
            recipient_ids_by_sender={1: {3}},
        )
        event = controller.events[0]
        self.assertTrue(event["message_consumed_by_next_round"])
        self.assertFalse(event["fault_payload_survived_governance"])
        self.assertFalse(event["delivered_to_next_round_context"])

    def test_injector_prompt_is_oracle_free_and_source_blind(self):
        messages = FaultInjector.build_messages(
            question_text="Question\nA. alpha\nB. beta\nC. gamma",
            target_answer="C",
            target_option_text="gamma",
            original_response="Reasoning.\nAnswer: A",
        )
        prompt = "\n".join(message["content"] for message in messages).lower()
        for forbidden in (
            "gold answer",
            "ground-truth answer",
            "correct answer",
            "answer_key",
            "reference_fact",
            "source a",
            "source b",
            "agent_id",
        ):
            self.assertNotIn(forbidden, prompt)

    def test_target_final_line_normalization_removes_inline_duplicate(self):
        value = normalize_target_final_line(
            "Reasoning supporting the distractor. Answer: B",
            "C",
        )
        self.assertEqual(value, "Reasoning supporting the distractor.\nAnswer: C")

    def test_target_final_line_normalization_removes_option_label(self):
        value = normalize_target_final_line("Answer: D. Chan-Zen", "I")
        self.assertEqual(value, "Answer: I")

    def test_compact_option_label_rewrite_does_not_fail_length_ratio(self):
        injector = object.__new__(FaultInjector)
        result = injector.inject(
            spec=_spec(),
            round_index=2,
            agent_id=1,
            question_text="Question\nA. alpha\nB. beta\nC. target distractor",
            original_response="Answer: A. Alpha option",
            parse_answer=lambda text: parse_mmlupro_choice_answer(
                text,
                valid_letters=list("ABCDEFGHIJ"),
            ),
        )
        self.assertTrue(result.success)
        self.assertEqual(result.mode, "minimal_choice_rewrite")
        self.assertEqual(result.response, "Answer: C")
        self.assertEqual(
            result.validation["length_match_exempt_reason"],
            "answer_only_or_compact_response",
        )

    def test_choice_aware_validation_accepts_mmlupro_target(self):
        validation = FaultInjector.validate(
            original_response="Original reasoning.\nAnswer: B",
            rewritten_response="Edited reasoning.\nAnswer: H",
            target_answer="H",
            parse_answer=lambda text: parse_mmlupro_choice_answer(
                text,
                valid_letters=list("ABCDEFGHIJ"),
            ),
        )
        self.assertTrue(validation["valid"])

    def test_manifest_validation_and_resume_identity(self):
        sample = {
            "external_clean_r0": True,
            "question_id": "q0",
            "question_text": "Question\nA. alpha\nB. beta\nC. gamma",
            "choices": ["alpha", "beta", "gamma"],
            "num_agents": 5,
            "nodes": [{"agent_id": index} for index in range(5)],
        }
        fallback = "Plausible but mistaken reasoning.\nAnswer: C"
        spec = TwoWindowFaultSpec(
            sample_idx=0,
            question_id="q0",
            question_sha256=question_sha256(sample),
            num_agents=5,
            total_rounds=TOTAL_ROUNDS,
            master_seed=1,
            sample_schedule_seed=42,
            source_a=1,
            source_b=4,
            first_fault_rounds=FIRST_FAULT_ROUNDS,
            second_fault_rounds=SECOND_FAULT_ROUNDS,
            target_wrong_option="C",
            target_option_text="gamma",
            canonical_fallback_response=fallback,
            fallback_sha256=hashlib.sha256(fallback.encode("utf-8")).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            manifest.write_text(json.dumps(spec.to_dict()) + "\n", encoding="utf-8")
            args = argparse.Namespace(
                two_window_fault_manifest=str(manifest),
                fault_injector_model="fake-injector",
                fault_injector_max_tokens=256,
            )
            validate_protocol_inputs(args=args, samples=[(0, sample)])
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            row = {
                "two_window_fault_protocol": {
                    "enabled": True,
                    "manifest_sha256": digest,
                    "injector_prompt_version": INJECTOR_PROMPT_VERSION,
                    "validation_version": VALIDATION_VERSION,
                    "manifest_sample_idx": 0,
                    "source_a": 1,
                    "source_b": 4,
                    "target_wrong_option": "C",
                    "sample_schedule_seed": 42,
                }
            }
            self.assertTrue(
                checkpoint_matches_protocol(args=args, row=row, sample_idx=0)
            )
            row["two_window_fault_protocol"]["source_a"] = 2
            self.assertFalse(
                checkpoint_matches_protocol(args=args, row=row, sample_idx=0)
            )


if __name__ == "__main__":
    unittest.main()
