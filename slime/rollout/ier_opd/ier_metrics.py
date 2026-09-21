"""Candidate-set IER for sampled reverse-KL token selection.

At each response position C = TopK_student union TopK_teacher union {sample}.
Inputs are serving-API log-probabilities. Missing values are filled with -12
in that log-probability domain, then each model is renormalized over C.
The log ratio is clipped to [-30, 30]. With L=1/p-1 and its optimal baseline
b_star, IER = Var_p(log(p/q)) / max(E_p[(log(p/q)-b_star)^2 L] - Var_p, eps).
The score affects loss-token selection; it does not replace the sampled
reverse-KL training objective or truncate student rollouts.
"""

from __future__ import annotations

import math

# Type alias: a parsed top-k item is ``(token_id, logp)``.
ItemId = int
Item = tuple[ItemId, float]


def _build_candidate_items(
    s_items: list[Item],
    t_items: list[Item],
    t_fill: dict[ItemId, float] | None,
    sampled_id: ItemId | None,
    s_fill: dict[ItemId, float] | None = None,
    floor_logp: float = -12.0,
    sampled_s_logp: float | None = None,
    sampled_t_logp: float | None = None,
) -> tuple[list[ItemId], dict[ItemId, float], dict[ItemId, float]]:
    """Complete the fixed union without adding ids from auxiliary fill pools."""
    s_logp = {tid: lp for tid, lp in s_items if math.isfinite(lp)}
    t_logp = {tid: lp for tid, lp in t_items if math.isfinite(lp)}
    ids = set(s_logp) | set(t_logp)
    if sampled_id is not None:
        ids.add(sampled_id)
    for target, fill in ((s_logp, s_fill), (t_logp, t_fill)):
        for tid, lp in (fill or {}).items():
            if tid in ids and math.isfinite(lp):
                target.setdefault(tid, lp)
    if sampled_id is not None:
        for target, lp in ((s_logp, sampled_s_logp), (t_logp, sampled_t_logp)):
            if lp is not None and math.isfinite(lp):
                target.setdefault(sampled_id, float(lp))
    if not math.isfinite(floor_logp):
        raise ValueError("IER floor_logp must be finite")
    for tid in ids:
        s_logp.setdefault(tid, floor_logp)
        t_logp.setdefault(tid, floor_logp)

    return sorted(ids), s_logp, t_logp


def _renormalised_logp(
    ids: list[ItemId], logp_map: dict[ItemId, float]
) -> dict[ItemId, float]:
    """Return log p_c, log p renormalised over the candidate set.

    For ids missing from ``logp_map`` the entry is absent (caller treats as p==0).
    """
    present = [(tid, logp_map[tid]) for tid in ids if tid in logp_map]
    if not present:
        return {}
    # logsumexp under the present ids
    m = max(lp for _, lp in present)
    se = sum(math.exp(lp - m) for _, lp in present)
    log_z = m + math.log(se)
    return {tid: lp - log_z for tid, lp in present}


def compute_position(
    s_items: list[Item],
    t_items: list[Item],
    sampled_id: ItemId | None = None,
    t_fill: dict[ItemId, float] | None = None,
    eps: float = 1e-8,
    log_ratio_cap: float = 30.0,
    s_fill: dict[ItemId, float] | None = None,
    floor_logp: float = -12.0,
    sampled_s_logp: float | None = None,
    sampled_t_logp: float | None = None,
) -> dict[str, float]:
    """Return reverse-KL IER and candidate coverage diagnostics for one position.

    Exact sampled values and optional fill lookups take precedence over the
    missing-value floor. They do not enlarge the fixed candidate union.
    """
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("IER eps must be finite and positive")
    if not math.isfinite(log_ratio_cap) or log_ratio_cap <= 0:
        raise ValueError("IER log_ratio_cap must be finite and positive")
    ids, s_logp_raw, t_logp_raw = _build_candidate_items(
        s_items, t_items, t_fill, sampled_id,
        s_fill=s_fill, floor_logp=floor_logp,
        sampled_s_logp=sampled_s_logp, sampled_t_logp=sampled_t_logp,
    )

    s_logp = _renormalised_logp(ids, s_logp_raw)  # log p_c
    t_logp = _renormalised_logp(ids, t_logp_raw)  # log q_c

    # p-sup: ids with finite student prob (student top-k support).
    p_sup = [tid for tid in ids if tid in s_logp]
    # intersection: ids where both p,q > 0 (finite renormalised logprobs).
    inter = [tid for tid in p_sup if tid in t_logp]
    # q-sup: ids with finite teacher prob.
    q_sup = [tid for tid in ids if tid in t_logp]

    p = {tid: math.exp(s_logp[tid]) for tid in p_sup}
    q = {tid: math.exp(t_logp[tid]) for tid in q_sup}

    p_sum = sum(p.values())
    q_sum = sum(q.values())
    # Pre-renormalization mass, including the explicitly assigned floor values.
    cand_p_mass = sum(math.exp(lp) for lp in s_logp_raw.values())
    cand_q_mass = sum(math.exp(lp) for lp in t_logp_raw.values())

    def _clamp_log_ratio(lp_p: float, lp_q: float) -> float:
        return max(-log_ratio_cap, min(log_ratio_cap, lp_p - lp_q))

    # ── Reverse (RKL): c_r = log p - log q, p-weighted over p-sup with q present ──
    ier_r = signal_r = noise_r = 0.0
    rkl_valid = [tid for tid in p_sup if tid in t_logp]
    if rkl_valid and p_sum > eps:
        c_r = {tid: _clamp_log_ratio(s_logp[tid], t_logp[tid]) for tid in rkl_valid}
        mean_c_r = sum(p[tid] * c_r[tid] for tid in rkl_valid) / p_sum
        signal_r = sum(p[tid] * (c_r[tid] - mean_c_r) ** 2 for tid in rkl_valid) / p_sum
        L_r = {tid: (1.0 / max(p[tid], eps) - 1.0) for tid in rkl_valid}
        E_cL_r = sum(p[tid] * c_r[tid] * L_r[tid] for tid in rkl_valid) / p_sum
        E_L_r = sum(p[tid] * L_r[tid] for tid in rkl_valid) / p_sum
        b_star_r = E_cL_r / E_L_r if abs(E_L_r) > eps else mean_c_r
        second_r = (
            sum(p[tid] * (c_r[tid] - b_star_r) ** 2 * L_r[tid] for tid in rkl_valid)
            / p_sum
        )
        noise_r = max(second_r - signal_r, 0.0)
        ier_r = signal_r / max(noise_r, eps)

    # Candidate-set reverse KL using the same clipped ratio.
    exact_rkl = 0.0
    for tid in p_sup:
        lp_q = t_logp.get(tid)  # None -> q==0 on candidate set
        ratio = (
            log_ratio_cap
            if lp_q is None
            else _clamp_log_ratio(s_logp[tid], lp_q)
        )
        exact_rkl += p[tid] * ratio
    exact_rkl /= max(p_sum, eps)
    # ── Teacher entropy & support overlap ──
    entropy_q = -sum(q[tid] * t_logp[tid] for tid in q_sup) / max(q_sum, eps) if q_sup else 0.0
    # support_overlap: fraction of student mass whose id the teacher also covers.
    support_overlap = (
        sum(p[tid] for tid in p_sup if tid in t_logp) / max(p_sum, eps)
        if p_sup
        else 0.0
    )

    return {
        "ier_r": ier_r,
        "signal_r": signal_r,
        "noise_r": noise_r,
        "entropy_q": entropy_q,
        "support_overlap": support_overlap,
        "exact_rkl": exact_rkl,
        "n_cand": float(len(ids)),
        "n_intersect": float(len(inter)),
        "cand_p_mass": cand_p_mass,
        "cand_q_mass": cand_q_mass,
    }


def compute_ier_score(
    s_items: list[Item],
    t_items: list[Item],
    sampled_id: ItemId | None = None,
    t_fill: dict[ItemId, float] | None = None,
    eps: float = 1e-8,
    sampled_s_logp: float | None = None,
    sampled_t_logp: float | None = None,
    *,
    s_fill: dict[ItemId, float] | None = None,
    floor_logp: float = -12.0,
) -> float:
    """Return the reverse-KL Fisher signal-to-noise ratio used by IER."""
    return compute_position(
        s_items, t_items, sampled_id, t_fill, eps=eps,
        s_fill=s_fill, floor_logp=floor_logp,
        sampled_s_logp=sampled_s_logp, sampled_t_logp=sampled_t_logp,
    )["ier_r"]
