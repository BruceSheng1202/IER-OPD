"""CPU numerical checks using closed-form values, not a duplicate implementation."""
import math

import pytest

from slime.rollout.ier_opd import compute_ier_score, compute_position
from slime.rollout.ier_opd.ier_metrics import _build_candidate_items


def test_known_three_action_fisher_signal_and_noise():
    # p=(1/2,1/3,1/6), log(p/q)=(0,1,2)+constant.
    # E[c]=2/3, Var[c]=5/9, optimal baseline=7/6+constant.
    # E[(c-b*)² L]=23/18, noise=13/18, hence IER=10/13.
    p = [0.5, 1 / 3, 1 / 6]
    student = [(i, math.log(value)) for i, value in enumerate(p)]
    teacher = [(i, math.log(value) - i) for i, value in enumerate(p)]
    result = compute_position(student, teacher)
    assert result['signal_r'] == pytest.approx(5 / 9)
    assert result['noise_r'] == pytest.approx(13 / 18)
    assert result['ier_r'] == pytest.approx(10 / 13)
    assert compute_ier_score(student, teacher) == pytest.approx(10 / 13)


@pytest.mark.parametrize('size,expected', [(3, 1.0), (6, 0.25), (10, 0.125)])
def test_uniform_student_closed_form(size, expected):
    # Constant leverage gives IER=1/(size-2) for any nonconstant coefficient.
    student = [(i, -math.log(size)) for i in range(size)]
    teacher = [(i, -math.log(size) - i * 0.1) for i in range(size)]
    assert compute_ier_score(student, teacher) == pytest.approx(expected)


def test_identical_models_have_zero_signal():
    items = [(0, math.log(0.7)), (1, math.log(0.3))]
    result = compute_position(items, items)
    assert result['signal_r'] == 0
    assert result['ier_r'] == 0


def test_empty_candidates_are_finite():
    result = compute_position([], [])
    assert result['n_cand'] == 0
    assert result['ier_r'] == 0
    assert all(math.isfinite(value) for value in result.values())


def test_missing_support_uses_log_probability_floor():
    ids, student, teacher = _build_candidate_items([(1, -0.3)], [(2, -0.7)], None, 3,
                                                  sampled_s_logp=-2, sampled_t_logp=-3)
    assert ids == [1, 2, 3]
    assert student == {1: -0.3, 2: -12.0, 3: -2.0}
    assert teacher == {1: -12.0, 2: -0.7, 3: -3.0}
    # The floor is an assignment for missing values, not clipping observed tails.
    _, student, _ = _build_candidate_items([(1, -18)], [(2, -0.7)], None, None)
    assert student[1] == -18


def test_fill_pool_never_expands_position_candidate_set():
    ids, student, teacher = _build_candidate_items(
        [(1, -0.3)], [(2, -0.7)], {1: -0.9, 99: -1.0}, 3,
        s_fill={2: -0.8, 98: -1.0}, sampled_s_logp=-2, sampled_t_logp=-3,
    )
    assert ids == [1, 2, 3]
    assert student == {1: -0.3, 2: -0.8, 3: -2.0}
    assert teacher == {1: -0.9, 2: -0.7, 3: -3.0}


def test_topk_values_take_precedence_over_fill_and_sample():
    _, student, teacher = _build_candidate_items(
        [(1, -0.3)], [(1, -0.7)], {1: -8}, 1, s_fill={1: -9},
        sampled_s_logp=-10, sampled_t_logp=-11,
    )
    assert student[1] == -0.3
    assert teacher[1] == -0.7


def test_disjoint_top16_plus_sample_has_33_candidates():
    student = [(i, -4.0) for i in range(16)]
    teacher = [(i, -4.0) for i in range(16, 32)]
    result = compute_position(student, teacher, 32, sampled_s_logp=-5, sampled_t_logp=-6)
    assert result['n_cand'] == 33
    assert all(math.isfinite(value) for value in result.values())


def test_full_support_renormalization_is_shift_invariant():
    student = [(0, -1), (1, -2), (2, -3)]
    teacher = [(0, -2), (1, -3), (2, -1)]
    result = compute_ier_score(student, teacher)
    assert compute_ier_score([(i, lp - 10) for i, lp in student], teacher) == pytest.approx(result)


def test_log_ratio_clipping_prevents_overflow():
    result = compute_position([(0, 0.0), (1, -1000)], [(0, -1000), (1, 0.0)])
    assert all(math.isfinite(value) for value in result.values())
    assert result['exact_rkl'] <= 30.0


@pytest.mark.parametrize('eps', [0, -1, float('nan'), float('inf')])
def test_invalid_denominator_floor_rejected(eps):
    with pytest.raises(ValueError, match='eps'):
        compute_position([], [], eps=eps)
