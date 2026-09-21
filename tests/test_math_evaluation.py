"""CPU checks for the shared scorer and standalone inference evaluation."""

import asyncio
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("evaluate_math", ROOT / "scripts/evaluate_math.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)

from slime.rollout.rm_hub import async_rm, batched_async_rm
from slime.rollout.rm_hub.math_grading import equivalent_answers, grade_math_answer
from slime.utils.metric_utils import compute_bayes_at_n


def test_bayes_32_mean_and_uncertainty():
    # One always-correct and one always-incorrect question give symmetric
    # Beta(33, 1) / Beta(1, 33) posteriors, with independent variances.
    metrics = compute_bayes_at_n([0] * 32 + [1] * 32, 32)
    sigma = math.sqrt((33 / (34**2 * 35)) / 2)
    assert metrics == pytest.approx({
        "bayes@32": 0.5,
        "bayes@32-sigma": sigma,
        "bayes@32-ci_lower": 0.5 - 1.645 * sigma,
        "bayes@32-ci_upper": 0.5 + 1.645 * sigma,
    })


def test_bayes_rejects_continuous_rewards():
    with pytest.raises(ValueError, match="binary"):
        compute_bayes_at_n([0.5, 1], 2)
    assert compute_bayes_at_n([], 32) == {}
    assert compute_bayes_at_n([1], 32) == {}


def test_math_uses_last_boxed_answer():
    assert grade_math_answer(r"Reasoning: \boxed{3}. Final: \boxed{2}", "2") == 1
    assert grade_math_answer(r"Final: \boxed{3}", "2") == 0


@pytest.mark.parametrize("metadata", [
    {"data_source": "HMMT2025", "index": "HMMT2025-9"},
    {"data_source": "HMMT2025", "extra_info": {"index": 9}},
])
def test_hmmt_equivalent_answer_metadata(metadata):
    answer = r"\frac{\sqrt{17}-1}{2}, \frac{-\sqrt{17}-1}{2}"
    label = r"\frac{-1+\sqrt{17}}{2}, \frac{-1-\sqrt{17}}{2}"
    assert answer in equivalent_answers(metadata)
    assert grade_math_answer(r"\boxed{" + answer + "}", label, metadata) == 1
    assert equivalent_answers({"data_source": "AIME25", "index": 9}) == []
    assert equivalent_answers({"data_source": "HMMT2025"}) == []


def test_shared_reward_dispatch_and_custom_hook(monkeypatch):
    args = SimpleNamespace(custom_rm_path=None, rm_type="math")
    samples = [SimpleNamespace(response=r"\boxed{2}", label="2", metadata={}),
               SimpleNamespace(response=r"\boxed{3}", label="2", metadata={})]
    assert asyncio.run(batched_async_rm(args, samples)) == [1, 0]

    async def custom_reward(args, sample, **kwargs):
        return {"teacher_log_probs": [-0.2], "test_value": kwargs["test_value"]}

    monkeypatch.setitem(sys.modules, "slime.utils.misc", SimpleNamespace(load_function=lambda path: custom_reward))
    args.custom_rm_path = "custom.reward"
    assert asyncio.run(async_rm(args, samples[0], test_value=7))["test_value"] == 7


def test_dataset_retains_original_metadata_and_identifiers(tmp_path):
    dataset_path = tmp_path / "questions.jsonl"
    rows = [
        {"prompt": "first", "label": "1", "metadata": {"extra_info": {"index": 9}, "tag": "kept"}},
        {"prompt": [{"role": "user", "content": "second"}], "label": "2", "index": "HMMT2025-7"},
        {"prompt": "third", "label": "3"},
    ]
    dataset_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    questions = evaluation.load_questions({"name": "HMMT2025", "path": str(dataset_path)})
    assert questions[0]["metadata"]["extra_info"]["index"] == 9
    assert questions[0]["metadata"]["tag"] == "kept"
    assert questions[0]["prompt"] == [{"role": "user", "content": "first"}]
    assert questions[1]["metadata"]["index"] == "HMMT2025-7"
    assert questions[1]["prompt"] == rows[1]["prompt"]
    assert questions[2]["metadata"] == {"data_source": "HMMT2025", "rm_type": "math"}


def sample_records():
    return [
        {"dataset": "A", "question": 1, "sample": 1, "reward": 0, "truncated": True},
        {"dataset": "A", "question": 0, "sample": 0, "reward": 1, "truncated": False},
        {"dataset": "A", "question": 1, "sample": 0, "reward": 0, "truncated": False},
        {"dataset": "A", "question": 0, "sample": 1, "reward": 1, "truncated": False},
        {"dataset": "B", "question": 0, "sample": 0, "reward": 1, "truncated": False},
    ]


def test_metrics_group_async_completions_and_dataset_sample_counts():
    datasets = [{"name": "A", "n_samples_per_prompt": 2}, {"name": "B", "n_samples_per_prompt": 1}]
    metrics = evaluation.summarize_results(sample_records(), datasets)
    assert metrics["A"]["pass@2"] == 0.5
    assert metrics["A"]["bayes@2"] == 0.5
    assert metrics["A"]["truncated_ratio"] == 0.25
    assert metrics["B"]["pass@1"] == 1
    assert "_merged" not in metrics


@pytest.mark.parametrize("duplicate", [False, True])
def test_incomplete_or_duplicate_samples_fail(duplicate):
    records = sample_records()
    if duplicate:
        records.append(records[0])
    else:
        records.pop(0)
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        evaluation.summarize_results(records, [{"name": "A", "n_samples_per_prompt": 2}])


def test_preview_works_without_site_packages_or_model_files(tmp_path):
    output = tmp_path / "not-created"
    result = subprocess.run([
        sys.executable, "-S", str(ROOT / "scripts/evaluate_math.py"),
        "--checkpoint", "/absent/checkpoint", "--data-root", "/absent/data",
        "--output-dir", str(output), "--thinking", "on", "--samples", "4",
    ], capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["execute"] is False
    assert plan["seed"] is None
    assert plan["chat_template_kwargs"] == {"enable_thinking": True}
    assert len(plan["datasets"]) == 4
    assert all(d["n_samples_per_prompt"] == 4 for d in plan["datasets"])
    assert not output.exists()


def test_invalid_parallel_size_is_rejected(tmp_path):
    args = evaluation.parse_args([
        "--checkpoint", "checkpoint", "--data-root", str(tmp_path),
        "--output-dir", str(tmp_path / "out"), "--tensor-parallel-size", "0",
    ])
    with pytest.raises(ValueError, match="positive"):
        evaluation.build_plan(args)


def test_nemotron_keeps_default_template_and_external_server(tmp_path):
    args = evaluation.parse_args([
        "--checkpoint", "checkpoint", "--data-root", str(tmp_path),
        "--output-dir", str(tmp_path / "out"), "--thinking", "model-default",
        "--base-url", "http://localhost:30100/", "--datasets", "HMMT2026",
    ])
    plan = evaluation.build_plan(args)
    assert plan["chat_template_kwargs"] == {}
    assert plan["server_command"] is None
    assert plan["base_url"] == "http://localhost:30100"
    assert [d["name"] for d in plan["datasets"]] == ["HMMT2026"]


@pytest.mark.parametrize("seed", [None, 42])
def test_generation_uses_tokenized_chat_and_shared_scorer(monkeypatch, seed):
    import httpx

    requests = []

    async def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json={
            "text": r"Reasoning done. \boxed{2}",
            "meta_info": {"finish_reason": {"type": "stop"}, "completion_tokens": 9},
        })

    original_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(
        transport=httpx.MockTransport(handler), **kwargs,
    ))

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "one plus one"}]
            assert kwargs == {"tools": None, "tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
            return "rendered chat"

        def __call__(self, text, **kwargs):
            assert text == "rendered chat" and kwargs == {"add_special_tokens": False}
            return {"input_ids": [10, 20, 30]}

    plan = {
        "datasets": [{"name": "A", "n_samples_per_prompt": 2}],
        "concurrency": 2, "base_url": "http://local-test", "seed": seed,
        "sampling": {"n_samples_per_prompt": 2, "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 32768},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    questions = {"A": [{"prompt": [{"role": "user", "content": "one plus one"}], "label": "2", "metadata": {}}]}
    output = io.StringIO()
    records = asyncio.run(evaluation.generate_results(plan, questions, Tokenizer(), output, 10))
    assert len(records) == 2 and all(r["reward"] == 1 for r in records)
    assert all(r["input_ids"] == [10, 20, 30] and r["return_logprob"] is False for r in requests)
    if seed is None:
        assert all("sampling_seed" not in r["sampling_params"] for r in requests)
    else:
        assert {r["sampling_params"]["sampling_seed"] for r in requests} == {42, 43}
    assert all(r["sampling_params"]["no_stop_trim"] for r in requests)
    assert all(r["sampling_params"]["skip_special_tokens"] is False for r in requests)
    assert all("n_samples_per_prompt" not in r["sampling_params"] for r in requests)
    assert len(output.getvalue().splitlines()) == 2
