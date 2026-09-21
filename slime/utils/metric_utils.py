import logging
import math
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


def dict_add_prefix(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def compute_pass_rate(
    flat_rewards: list[float],
    group_size: int,
    num_groups: int | None = None,
):
    if group_size == 1:
        return {}

    if num_groups is None:
        num_groups = len(flat_rewards) // group_size

    pass_rate_name_list = [2**i for i in range(int(math.log2(group_size)) + 1)]

    assert len(flat_rewards) == num_groups * group_size, f"{len(flat_rewards)=} {num_groups=} {group_size=}"
    rewards_of_group = np.array(flat_rewards).reshape(num_groups, group_size)

    log_dict = {}
    for k in pass_rate_name_list:
        num_correct = np.sum(rewards_of_group == 1, axis=1)
        num_samples = np.full(num_groups, group_size)

        pass_k_estimates = _estimate_pass_at_k(num_samples, num_correct, k)

        pass_k = np.mean(pass_k_estimates)
        log_dict[f"pass@{k}"] = pass_k

    return log_dict


def compute_bayes_at_n(
    flat_rewards: list[float],
    group_size: int,
    *,
    z: float = 1.645,
) -> dict:
    """Mean accuracy under independent Beta(1, 1) posteriors per question.

    Return the posterior mean, standard deviation of the dataset mean, and
    clipped normal-approximation interval (default z=1.645, about 90% coverage).
    Each question must have exactly ``group_size`` binary-reward samples.
    """
    if group_size <= 1:
        return {}

    rewards = np.asarray(flat_rewards, dtype=float)
    if rewards.size == 0 or rewards.size % group_size != 0:
        return {}

    non_binary = ~np.isin(rewards, (0.0, 1.0))
    if non_binary.any():
        first_bad = rewards[non_binary][0]
        raise ValueError(
            f"compute_bayes_at_n requires binary 0/1 rewards; got "
            f"{int(non_binary.sum())}/{rewards.size} non-binary values "
            f"(first offender: {first_bad!r}). Use a binary rm (e.g. rm_type=math) "
            f"or extend the estimator beyond Beta-Binomial."
        )

    M = rewards.size // group_size
    N = group_size
    n_alpha = rewards.reshape(M, N).sum(axis=1)

    a = n_alpha + 1.0
    b = (N - n_alpha) + 1.0
    denom = a + b  # = N + 2
    mu_per_q = a / denom
    var_per_q = (a * b) / (denom**2 * (denom + 1.0))

    mu = float(mu_per_q.mean())
    sigma = float(np.sqrt(var_per_q.sum() / (M**2)))

    return {
        f"bayes@{N}": mu,
        f"bayes@{N}-sigma": sigma,
        f"bayes@{N}-ci_lower": max(0.0, mu - z * sigma),
        f"bayes@{N}-ci_upper": min(1.0, mu + z * sigma),
    }


def _estimate_pass_at_k(num_samples, num_correct, k):
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n, c, k):
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct, strict=False)])


def compute_statistics(values: list[float]) -> dict[str, float]:
    values = np.array(values)
    return {
        "mean": np.mean(values).item(),
        "median": np.median(values).item(),
        "max": np.max(values).item(),
        "min": np.min(values).item(),
    }


def compression_ratio(
    data: str | bytes,
    *,
    encoding: str = "utf-8",
    algorithm: Literal["zlib", "gzip", "bz2", "lzma"] = "zlib",
    level: int = 9,
) -> tuple[float, float]:
    if isinstance(data, str):
        raw = data.encode(encoding)
    else:
        raw = data

    original = len(raw)
    if original == 0:
        return float("inf"), 0.0

    if algorithm == "zlib":
        import zlib

        compressed = zlib.compress(raw, level)
    elif algorithm == "gzip":
        import gzip

        compressed = gzip.compress(raw, compresslevel=level)
    elif algorithm == "bz2":
        import bz2

        compressed = bz2.compress(raw, compresslevel=level)
    elif algorithm == "lzma":
        import lzma

        compressed = lzma.compress(raw, preset=level)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    comp_len = len(compressed)
    if comp_len == 0:
        return float("inf"), 100.0

    ratio = original / comp_len
    savings_pct = 100.0 * (1.0 - comp_len / original)
    return ratio, savings_pct


def has_repetition(text: str):
    if len(text) > 10000 and compression_ratio(text[-10000:])[0] > 10:
        return True
    else:
        return False


def compute_rollout_step(args, rollout_id):
    if args.wandb_always_use_train_step:
        return rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    return rollout_id
