import csv
import json
import math
import random
from pathlib import Path
from typing import Any


# Public selectors; no compatibility aliases are accepted.
BASE_SELECTORS = ("prefix", "entropy", "tip", "ta_opd", "ca_softor")
FUSION_SELECTORS = tuple(f"{base}_ier_{fusion}" for base in BASE_SELECTORS for fusion in ("or", "and"))
BUDGET_METHODS = ("full", "ier", *BASE_SELECTORS, "random", "sampled_rkl_max", "sampled_rkl_min", *FUSION_SELECTORS)

_ROLLOUT_COUNTER = 0
_CSV_FIELDS = [
    "schema_version",
    "pair_id",
    "teacher_name",
    "student_name",
    "seed",
    "rollout_id",
    "sample_index",
    "group_index",
    "sample_ordinal",
    "tok_pos",
    "pos_norm",
    "prompt_len",
    "resp_len",
    "total_len",
    "truncated",
    "token_id",
    "student_logp_sampled",
    "teacher_logp_sampled",
    "sampled_reverse_kl",
    "sampled_teacher_adv",
    "student_top1_id",
    "teacher_top1_id",
    "student_top1_prob",
    "teacher_top1_prob",
    "student_topk_mass",
    "teacher_topk_mass",
    "Hs_topk",
    "Ht_topk",
    "Hs_topk_norm",
    "Ht_topk_norm",
    "KLf_union",
    "KLr_union",
    "Cmass",
    "Cmass_topk",
    "Cmass_true",
    "Cmass_exact",
    "Coverlap",
    "CBC",
    "reach_t1",
    "target_in_student_topk",
    "target_in_teacher_topk",
    "target_student_rank",
    "target_teacher_rank",
    "loss_mask_original",
    "H_norm",
    "D_norm",
    "C_norm",
    "DC_norm",
    "Dlearn",
    "tip_score",
    "ca_softor_score",
    # IER-OPD diagnostic columns (populated when the IER score is computed;
    # left empty/0 otherwise so non-IER runs export an identical schema).
    "ier_r",
    "ier_rank",
    "budget_method",
    "budget_ratio",
    "budget_score",
    "budget_keep",
    "student_top_ids",
    "student_top_logps",
    "teacher_top_ids",
    "teacher_top_logps",
]


def topk_enabled(args) -> bool:
    return (bool(getattr(args, "use_opd", False))
            and getattr(args, "opd_type", "sglang") == "sglang"
            and int(getattr(args, "opd_topk_metrics_k", 16) or 0) > 0)


def maybe_add_topk_request(args, payload: dict[str, Any]) -> None:
    k_metrics = int(getattr(args, "opd_topk_metrics_k", 16) or 0)
    if not topk_enabled(args):
        return
    # A larger optional pool supplies known log-probabilities for missing sides
    # of the fixed candidate union. It does not change the metric top-k support.
    k_sample = int(getattr(args, "opd_topk_sample_k", 0) or k_metrics)
    if k_sample < k_metrics:
        k_sample = k_metrics
    payload["top_logprobs_num"] = k_sample
    payload["return_text_in_logprobs"] = False


def maybe_add_teacher_token_ids_request(args, payload: dict[str, Any], sample) -> None:
    if not bool(getattr(args, "opd_exact_cmass", False)):
        return

    response_length = int(getattr(sample, "response_length", 0) or 0)
    if response_length <= 0:
        return

    student_top = ((getattr(sample, "metadata", None) or {}).get("student_top_logprobs") or [])[-response_length:]
    token_ids = []
    seen = set()
    for pos_items in student_top:
        for token_id, _logp in _parse_top_items(pos_items):
            if token_id not in seen:
                seen.add(token_id)
                token_ids.append(token_id)

    if not token_ids:
        return

    max_union = int(getattr(args, "opd_exact_cmass_max_union", 4096) or 4096)
    if len(token_ids) > max_union:
        overflow = getattr(args, "opd_exact_cmass_overflow", "fallback")
        if overflow == "error":
            raise ValueError(
                f"--opd-exact-cmass union size {len(token_ids)} exceeds "
                f"--opd-exact-cmass-max-union={max_union}"
            )
        if overflow == "fallback":
            return
        token_ids = token_ids[:max_union]

    payload["token_ids_logprob"] = token_ids


def process_tip_compat_metrics(args, samples, teacher_log_probs, raw_rewards) -> None:
    if not topk_enabled(args):
        return

    k = int(getattr(args, "opd_topk_metrics_k", 16) or 0)
    rows = []
    sample_rows: dict[int, list[dict[str, Any]]] = {}

    for sample_ordinal, (sample, t_log_probs, raw_reward) in enumerate(
        zip(samples, teacher_log_probs, raw_rewards, strict=False)
    ):
        response_length = int(sample.response_length or 0)
        if response_length <= 0:
            continue

        prompt_len = len(sample.tokens) - response_length
        student_top = (sample.metadata or {}).get("student_top_logprobs") or []
        student_top = student_top[-response_length:]
        teacher_top = _extract_teacher_response_topk(raw_reward, response_length)
        teacher_token_ids_logprobs = _extract_teacher_response_token_ids_logprobs(raw_reward, response_length)
        if len(student_top) != response_length or len(teacher_top) != response_length:
            if getattr(args, "opd_budget_mask", "full") != "full":
                raise ValueError(
                    "TIP compatibility budget mask requires both student and teacher top-k logprobs. "
                    f"Got student={len(student_top)}, teacher={len(teacher_top)}, response={response_length}."
                )

        original_loss_mask = sample.loss_mask if sample.loss_mask is not None else [1] * response_length
        sample_rows[sample_ordinal] = []

        for tok_pos in range(response_length):
            token_id = sample.tokens[prompt_len + tok_pos]
            s_items = _parse_top_items(student_top[tok_pos] if tok_pos < len(student_top) else None)
            t_items = _parse_top_items(teacher_top[tok_pos] if tok_pos < len(teacher_top) else None)
            t_requested_items = _parse_top_items(
                teacher_token_ids_logprobs[tok_pos] if tok_pos < len(teacher_token_ids_logprobs) else None
            )
            # Shared compatibility metrics (Cmass/H/D/KL/...) must be computed on
            # the top-k (opd_topk_metrics_k) support, NOT the larger opd_topk_sample_k
            # pool: sample_k exists only to feed IER's two-sided union completion, and
            # must not inflate Cmass (which would force Cmass->1 and collapse
            # DC_norm/tip_score/ca_softor into D_norm). The downstream IER branch still
            # uses the full s_items pool via s_pool.
            k_m_shared = int(getattr(args, "opd_topk_metrics_k", 0) or 0) or len(s_items)
            s_metrics = s_items[:k_m_shared]
            t_metrics = t_items[:k_m_shared]
            metrics = _compute_topk_metrics(s_metrics, t_metrics, token_id, k)
            _add_exact_cmass(metrics, s_metrics, t_requested_items)

            # Candidate-set IER is used only by IER and its two fusion rules.
            ier_r = 0.0
            method = getattr(args, "opd_budget_mask", "full")
            if method == "ier" or method in FUSION_SELECTORS:
                from slime.rollout.ier_opd.ier_metrics import compute_position as _ier_pos
                t_fill = {tid: lp for tid, lp in t_requested_items}
                # Two-sided union completion. The candidate set is the top-k
                # (opd_topk_metrics_k) union of student and teacher; the larger
                # sampling pool (opd_topk_sample_k > metrics_k) is used to look
                # up true logprobs for tokens outside one side's top-k, so the
                # support is a genuine union instead of the intersection.
                k_m = int(getattr(args, "opd_topk_metrics_k", 0) or 0) or len(s_items)
                s_top = s_items[:k_m]
                t_top = t_items[:k_m]
                s_pool = {tid: lp for tid, lp in s_items}
                # s_fill: teacher-support ids missing from student top-k, looked
                # up in the larger student sampling pool (true student logprobs).
                s_top_ids = {tid for tid, _ in s_top}
                s_fill = {
                    tid: s_pool[tid]
                    for tid, _ in t_top
                    if tid not in s_top_ids and tid in s_pool
                }
                # The serving API returns log-probabilities, not raw logits.
                floor_logp = float(getattr(args, "opd_ier_floor_logp", -12.0))
                # Real sampled logprobs from rollout — available even when the
                # sampled token is outside top-k / fill maps. Must be injected
                # before floor backfill so real values are not overwritten.
                _s_logp = _safe_float(sample.rollout_log_probs[tok_pos]) if sample.rollout_log_probs else float("nan")
                _t_logp = _safe_float(t_log_probs[tok_pos]) if tok_pos < len(t_log_probs) else float("nan")
                ier_diag = _ier_pos(
                    s_top, t_top, token_id, t_fill,
                    eps=float(getattr(args, "ier_eps", 1e-8)),
                    s_fill=s_fill if s_fill else None,
                    floor_logp=floor_logp,
                    sampled_s_logp=_s_logp if math.isfinite(_s_logp) else None,
                    sampled_t_logp=_t_logp if math.isfinite(_t_logp) else None,
                )
                ier_r = ier_diag["ier_r"]

            s_logp = _safe_float(sample.rollout_log_probs[tok_pos]) if sample.rollout_log_probs else float("nan")
            t_logp = _safe_float(t_log_probs[tok_pos]) if tok_pos < len(t_log_probs) else float("nan")
            row = {
                "schema_version": 2,
                "pair_id": getattr(args, "opd_token_bank_pair_id", ""),
                "teacher_name": getattr(args, "opd_teacher_name", None)
                or getattr(args, "opd_teacher_load", None)
                or getattr(args, "rm_url", ""),
                "student_name": getattr(args, "opd_student_name", None) or getattr(args, "hf_checkpoint", ""),
                "seed": getattr(args, "seed", None),
                "rollout_id": None,
                "sample_index": sample.index,
                "group_index": sample.group_index,
                "sample_ordinal": sample_ordinal,
                "tok_pos": tok_pos,
                "pos_norm": tok_pos / max(response_length - 1, 1),
                "prompt_len": prompt_len,
                "resp_len": response_length,
                "total_len": len(sample.tokens),
                "truncated": int(getattr(sample.status, "value", sample.status) == "truncated"),
                "token_id": token_id,
                "student_logp_sampled": s_logp,
                "teacher_logp_sampled": t_logp,
                "sampled_reverse_kl": s_logp - t_logp if math.isfinite(s_logp) and math.isfinite(t_logp) else float("nan"),
                "sampled_teacher_adv": t_logp - s_logp if math.isfinite(s_logp) and math.isfinite(t_logp) else float("nan"),
                "loss_mask_original": int(original_loss_mask[tok_pos]) if tok_pos < len(original_loss_mask) else 1,
                **metrics,
                "ier_r": ier_r,
                "ier_rank": 0.0,
                "budget_method": getattr(args, "opd_budget_mask", "full"),
                "budget_ratio": float(getattr(args, "opd_budget_ratio", 1.0)),
                "budget_score": 0.0,
                "budget_keep": int(original_loss_mask[tok_pos]) if tok_pos < len(original_loss_mask) else 1,
            }
            rows.append(row)
            sample_rows[sample_ordinal].append(row)

    if not rows:
        return

    _add_normalized_scores(args, rows)
    _apply_budget_mask(args, rows, sample_rows, samples)
    _export_rows(args, rows)
    # Exporting diagnostics must not change the random baseline's seed sequence.
    _next_rollout_id()


def _extract_teacher_response_topk(raw_reward: dict[str, Any], response_length: int) -> list[Any]:
    top = ((raw_reward or {}).get("meta_info") or {}).get("input_top_logprobs") or []
    if top and top[0] is None:
        top = top[1:]
    return top[-response_length:]


def _extract_teacher_response_token_ids_logprobs(raw_reward: dict[str, Any], response_length: int) -> list[Any]:
    values = ((raw_reward or {}).get("meta_info") or {}).get("input_token_ids_logprobs") or []
    if values and values[0] is None:
        values = values[1:]
    return values[-response_length:]


def _parse_top_items(items: Any) -> list[tuple[int, float]]:
    if not items:
        return []
    parsed = []
    for item in items:
        if not item:
            continue
        logp = _safe_float(item[0])
        token_id = int(item[1])
        if math.isfinite(logp):
            parsed.append((token_id, logp))
    return parsed


def _add_exact_cmass(
    metrics: dict[str, Any],
    student_items: list[tuple[int, float]],
    teacher_requested_items: list[tuple[int, float]],
) -> None:
    metrics["Cmass_topk"] = metrics["Cmass"]
    metrics["Cmass_true"] = ""
    metrics["Cmass_exact"] = 0
    if not student_items or not teacher_requested_items:
        return
    teacher_probs = {token: math.exp(logp) for token, logp in teacher_requested_items}
    if not teacher_probs:
        return
    cmass_true = _clamp_prob_mass(sum(teacher_probs.get(token, 0.0) for token, _ in student_items))
    metrics["Cmass_true"] = cmass_true
    metrics["Cmass"] = cmass_true
    metrics["Cmass_exact"] = 1


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _clamp_prob_mass(x: float) -> float:
    if not math.isfinite(x):
        return x
    return min(1.0, max(0.0, x))


def _compute_topk_metrics(
    student_items: list[tuple[int, float]],
    teacher_items: list[tuple[int, float]],
    token_id: int,
    k: int,
) -> dict[str, Any]:
    eps = 1e-12
    s_probs = {token: math.exp(logp) for token, logp in student_items}
    t_probs = {token: math.exp(logp) for token, logp in teacher_items}
    s_ids = [token for token, _ in student_items]
    t_ids = [token for token, _ in teacher_items]
    inter = set(s_ids).intersection(t_ids)
    union = set(s_ids).union(t_ids)

    s_mass = _clamp_prob_mass(sum(s_probs.values()))
    t_mass = _clamp_prob_mass(sum(t_probs.values()))
    hs, hs_norm = _entropy(s_probs.values())
    ht, ht_norm = _entropy(t_probs.values())
    klf, klr, bc = _union_geometry(s_probs, t_probs, union, eps)

    s_top1_id = s_ids[0] if s_ids else None
    t_top1_id = t_ids[0] if t_ids else None
    target_s_rank = _rank_of(s_ids, token_id)
    target_t_rank = _rank_of(t_ids, token_id)

    return {
        "student_top1_id": s_top1_id,
        "teacher_top1_id": t_top1_id,
        "student_top1_prob": s_probs.get(s_top1_id, 0.0) if s_top1_id is not None else 0.0,
        "teacher_top1_prob": t_probs.get(t_top1_id, 0.0) if t_top1_id is not None else 0.0,
        "student_topk_mass": s_mass,
        "teacher_topk_mass": t_mass,
        "Hs_topk": hs,
        "Ht_topk": ht,
        "Hs_topk_norm": hs_norm,
        "Ht_topk_norm": ht_norm,
        "KLf_union": klf,
        "KLr_union": klr,
        "Cmass": _clamp_prob_mass(sum(t_probs.get(token, 0.0) for token in s_ids)),
        "Cmass_topk": _clamp_prob_mass(sum(t_probs.get(token, 0.0) for token in s_ids)),
        "Cmass_true": "",
        "Cmass_exact": 0,
        "Coverlap": len(inter) / max(k, 1),
        "CBC": bc,
        "reach_t1": int(t_top1_id in set(s_ids)) if t_top1_id is not None else 0,
        "target_in_student_topk": int(token_id in set(s_ids)),
        "target_in_teacher_topk": int(token_id in set(t_ids)),
        "target_student_rank": target_s_rank,
        "target_teacher_rank": target_t_rank,
        "student_top_ids": json.dumps(s_ids),
        "student_top_logps": json.dumps([logp for _, logp in student_items]),
        "teacher_top_ids": json.dumps(t_ids),
        "teacher_top_logps": json.dumps([logp for _, logp in teacher_items]),
    }


def _entropy(probs_iter) -> tuple[float, float]:
    probs = [p for p in probs_iter if p > 0]
    mass = sum(probs)
    if mass <= 0:
        return 0.0, 0.0
    norm = [p / mass for p in probs]
    h = -sum(p * math.log(max(p, 1e-12)) for p in norm)
    return h, h / math.log(max(len(norm), 2))


def _union_geometry(s_probs: dict[int, float], t_probs: dict[int, float], union: set[int], eps: float):
    if not union:
        return 0.0, 0.0, 0.0
    s_mass = sum(s_probs.get(token, 0.0) for token in union)
    t_mass = sum(t_probs.get(token, 0.0) for token in union)
    if s_mass <= 0 or t_mass <= 0:
        return 0.0, 0.0, 0.0

    klf = 0.0
    klr = 0.0
    bc = 0.0
    for token in union:
        ps = max(s_probs.get(token, 0.0) / s_mass, eps)
        pt = max(t_probs.get(token, 0.0) / t_mass, eps)
        klf += pt * (math.log(pt) - math.log(ps))
        klr += ps * (math.log(ps) - math.log(pt))
        bc += math.sqrt(ps * pt)
    return klf, klr, bc


def _rank_of(ids: list[int], token_id: int) -> int | None:
    try:
        return ids.index(token_id) + 1
    except ValueError:
        return None


def _add_normalized_scores(args, rows: list[dict[str, Any]]) -> None:
    """Normalize only tokens eligible for this rollout batch's loss mask."""
    if getattr(args, "opd_compat_proxy", "mass") != "mass":
        raise ValueError("The paper configuration requires --opd-compat-proxy mass")
    valid = [row for row in rows if int(row["loss_mask_original"]) == 1]
    for row in rows:
        for key in ("H_norm", "D_norm", "C_norm", "DC_norm", "Dlearn", "tip_score", "ca_softor_score"):
            row[key] = 0.0
    h_vals = [row["Hs_topk_norm"] for row in valid]
    d_vals = [row["KLf_union"] for row in valid]
    c_vals = [row["Cmass"] for row in valid]
    dc_vals = [d * c for d, c in zip(d_vals, c_vals, strict=True)]
    for row, hn, dn, cn, dcn in zip(valid, _normalize(h_vals, args), _normalize(d_vals, args),
                                   _normalize(c_vals, args), _normalize(dc_vals, args), strict=True):
        row["H_norm"], row["D_norm"], row["C_norm"], row["DC_norm"] = hn, dn, cn, dcn
        row["Dlearn"] = dn * cn
        row["tip_score"] = hn + dn - hn * dn
        row["ca_softor_score"] = hn + dcn - hn * dcn


def _normalize(values: list[float], args) -> list[float]:
    if getattr(args, "opd_metric_normalization", "batch_quantile") != "batch_quantile":
        raise ValueError("The paper configuration requires --opd-metric-normalization batch_quantile")
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return [0.0 for _ in values]
    lo = _quantile(finite, float(getattr(args, "opd_metric_q_low", 0.05)))
    hi = _quantile(finite, float(getattr(args, "opd_metric_q_high", 0.95)))
    denom = hi - lo
    if abs(denom) < 1e-12:
        return [0.0 for _ in values]
    return [min(1.0, max(0.0, (v - lo) / denom)) if math.isfinite(v) else 0.0 for v in values]


def _quantile(values: list[float], q: float) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    pos = min(max(q, 0.0), 1.0) * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _base_score(method: str, row: dict[str, Any]) -> float:
    if method == "prefix":
        return 1.0 - float(row["pos_norm"])
    key = {"entropy": "H_norm", "tip": "tip_score", "ta_opd": "Dlearn",
           "ca_softor": "ca_softor_score", "ier": "ier_r",
           "sampled_rkl_max": "sampled_reverse_kl", "sampled_rkl_min": "sampled_reverse_kl"}[method]
    value = float(row[key])
    return -value if method == "sampled_rkl_min" else value


def _batch_ier_ranks(rows, valid) -> dict[int, float]:
    """Ordinal ranks on all valid batch tokens; stable row order breaks ties.

    Ranks are 0/(N-1), ..., (N-1)/(N-1). A singleton has rank zero.
    No response-local re-ranking occurs during per-response minimum fallback.
    """
    values = {i: float(rows[i]["ier_r"]) for i in valid}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("IER scores must be finite for all valid batch tokens")
    order = sorted(valid, key=lambda i: values[i])
    return {i: rank / max(len(order) - 1, 1) for rank, i in enumerate(order)}


def _apply_budget_mask(args, rows, sample_rows, samples) -> None:
    """Select ceil(ratio * N_valid) globally, then meet per-response minima.

    The minimum may raise the final count above the global budget. It selects
    the best remaining tokens using the same global scores and ordering.
    Prefix uses 1 - position/(response_length-1) and the same global budget.
    Score ties use IER descending for fusion, then original batch row order.
    Random selection and fallback share one seeded random permutation.
    """
    method = getattr(args, "opd_budget_mask", "full")
    if method not in BUDGET_METHODS:
        raise ValueError(f"Unknown --opd-budget-mask={method}")
    ratio = float(getattr(args, "opd_budget_ratio", 1.0))
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("--opd-budget-ratio must be in (0, 1]")
    min_keep = int(getattr(args, "opd_budget_min_keep_per_sample", 1))
    if min_keep < 0:
        raise ValueError("--opd-budget-min-keep-per-sample must be nonnegative")
    valid = [i for i, row in enumerate(rows) if int(row["loss_mask_original"]) == 1]
    scores = {i: 1.0 for i in valid}
    ranks = {}
    if method == "random":
        order = valid[:]
        rng = random.Random(int(getattr(args, "opd_budget_mask_seed", 42)) + _current_rollout_id())
        rng.shuffle(order)
        scores = {i: float(len(order) - rank) for rank, i in enumerate(order)}
    elif method in FUSION_SELECTORS:
        base, fusion = method.rsplit("_ier_", 1)
        ranks = _batch_ier_ranks(rows, valid)
        for i in valid:
            usefulness, reliability = _base_score(base, rows[i]), ranks[i]
            scores[i] = (usefulness * reliability if fusion == "and" else
                         usefulness + reliability - usefulness * reliability)
        order = sorted(valid, key=lambda i: (-scores[i], -float(rows[i]["ier_r"]), i))
    elif method == "full":
        order = valid[:]
    else:
        scores = {i: _base_score(method, rows[i]) for i in valid}
        order = sorted(valid, key=lambda i: (-scores[i], i))
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError(f"{method} selection requires finite scores on valid batch tokens")
    budget = len(valid) if method == "full" else min(len(valid), math.ceil(len(valid) * ratio))
    selected = set(order[:budget])
    index_by_row_id = {id(row): i for i, row in enumerate(rows)}
    priority = {i: rank for rank, i in enumerate(order)}
    for per_sample_rows in sample_rows.values():
        candidates = [index_by_row_id[id(row)] for row in per_sample_rows if int(row["loss_mask_original"]) == 1]
        need = min(min_keep, len(candidates)) - sum(i in selected for i in candidates)
        if need > 0:
            remaining = sorted((i for i in candidates if i not in selected), key=priority.__getitem__)
            selected.update(remaining[:need])
    for i, row in enumerate(rows):
        row["budget_keep"] = int(i in selected)
        row["budget_score"] = scores.get(i, 0.0)
        row["ier_rank"] = ranks.get(i, 0.0)
    for sample_ordinal, per_sample_rows in sample_rows.items():
        samples[sample_ordinal].loss_mask = [int(row["budget_keep"]) for row in per_sample_rows]


def _current_rollout_id() -> int:
    return _ROLLOUT_COUNTER


def _next_rollout_id() -> int:
    global _ROLLOUT_COUNTER
    rollout_id = _ROLLOUT_COUNTER
    _ROLLOUT_COUNTER += 1
    return rollout_id


def _export_rows(args, rows) -> None:
    export_dir = getattr(args, "opd_token_bank_dir", None)
    if not export_dir:
        return
    rollout_id = _current_rollout_id()
    for row in rows:
        row["rollout_id"] = rollout_id

    path = Path(export_dir)
    path.mkdir(parents=True, exist_ok=True)
    fmt = getattr(args, "opd_token_bank_format", "csv")
    if not bool(getattr(args, "opd_token_bank_raw_topk", False)):
        for row in rows:
            row["student_top_ids"] = ""
            row["student_top_logps"] = ""
            row["teacher_top_ids"] = ""
            row["teacher_top_logps"] = ""

    if fmt == "jsonl":
        out = path / f"rollout_{rollout_id:06d}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True, allow_nan=True) + "\n")
    else:
        out = path / f"rollout_{rollout_id:06d}.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            fields = _CSV_FIELDS
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    _append_summary(path, rows, rollout_id)
    _write_config(path, args)


def _append_summary(path: Path, rows, rollout_id: int) -> None:
    valid = [row for row in rows if int(row["loss_mask_original"]) == 1]
    kept = [row for row in valid if int(row["budget_keep"]) == 1]
    summary = {
        "rollout_id": rollout_id,
        "num_tokens": len(rows),
        "num_valid_tokens": len(valid),
        "num_kept_tokens": len(kept),
        "keep_ratio": len(kept) / max(len(valid), 1),
        "mean_H_norm": _mean(row["H_norm"] for row in valid),
        "mean_D_norm": _mean(row["D_norm"] for row in valid),
        "mean_C_norm": _mean(row["C_norm"] for row in valid),
        "mean_tip_score": _mean(row["tip_score"] for row in valid),
        "mean_ca_softor_score": _mean(row["ca_softor_score"] for row in valid),
        "mean_reach_t1": _mean(row["reach_t1"] for row in valid),
        "budget_method": rows[0].get("budget_method", "full") if rows else "full",
        "budget_ratio": rows[0].get("budget_ratio", 1.0) if rows else 1.0,
    }
    out = path / "summary.csv"
    exists = out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(summary)


def _mean(values) -> float:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(xs) / len(xs) if xs else 0.0


def _write_config(path: Path, args) -> None:
    out = path / "config.json"
    if out.exists():
        return
    cfg = {
        "schema_version": 2,
        "opd_topk_metrics_k": getattr(args, "opd_topk_metrics_k", None),
        "opd_budget_mask": getattr(args, "opd_budget_mask", None),
        "opd_budget_ratio": getattr(args, "opd_budget_ratio", None),
        "opd_compat_proxy": getattr(args, "opd_compat_proxy", None),
        "opd_metric_normalization": getattr(args, "opd_metric_normalization", None),
        "opd_exact_cmass": getattr(args, "opd_exact_cmass", None),
        "opd_exact_cmass_max_union": getattr(args, "opd_exact_cmass_max_union", None),
        "opd_exact_cmass_overflow": getattr(args, "opd_exact_cmass_overflow", None),
        "notes": (
            "Cmass and KL metrics are computed on returned top-k supports. "
            "When opd_exact_cmass is enabled, Cmass is replaced by true teacher mass on student top-k support. "
            "Cmass_topk always keeps the teacher-top-k lower bound."
        ),
    }
    cfg["selection"] = {
        "population": "all valid response tokens in this rollout batch",
        "budget": "ceil(N_valid * ratio), then per-response minimum fallback",
        "min_keep_per_sample": getattr(args, "opd_budget_min_keep_per_sample", 1),
        "ier_rank": "stable ordinal batch rank/(N_valid-1); singleton=0",
        "tie_break": "fusion: IER descending, then row order; other scores: row order",
        "fallback": "same global scores/order; final count may exceed target budget",
        "floor_domain": "log-probability",
        "floor_logp": getattr(args, "opd_ier_floor_logp", -12.0),
        "ier_eps": getattr(args, "ier_eps", 1e-8),
    }
    out.write_text(json.dumps(cfg, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
