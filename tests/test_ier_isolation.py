"""Paper baselines keep their hand-specified ordering without using IER."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from slime.rollout import tip_compat


def select(method, ratio=.5):
    columns = dict(H_norm=[.9,.1,.5,.8,.2,.6], tip_score=[.91,.91,.75,.84,.84,.76],
                   ca_softor_score=[.5,.2,.6,.7,.3,.65], Dlearn=[.07,.18,.25,.12,.24,.22],
                   sampled_reverse_kl=[-.3,.8,.1,.6,-.2,.4])
    rows = [dict({key: values[i] for key, values in columns.items()},
                 loss_mask_original=1, tok_pos=i, pos_norm=i/5, ier_r=float('nan')) for i in range(6)]
    args = SimpleNamespace(opd_budget_mask=method, opd_budget_ratio=ratio,
                           opd_budget_min_keep_per_sample=0, opd_budget_mask_seed=42)
    sample = SimpleNamespace(loss_mask=[1]*6)
    with patch.object(tip_compat, '_batch_ier_ranks', side_effect=AssertionError('baseline used IER')):
        tip_compat._apply_budget_mask(args, rows, {0: rows}, [sample])
    return [i for i, keep in enumerate(sample.loss_mask) if keep]


@pytest.mark.parametrize('method,expected', [
    ('full', [0,1,2,3,4,5]), ('prefix', [0,1,2]), ('entropy', [0,3,5]),
    ('tip', [0,1,3]), ('ta_opd', [2,4,5]), ('ca_softor', [2,3,5]),
    ('sampled_rkl_max', [1,3,5]), ('sampled_rkl_min', [0,2,4]),
])
def test_baseline_ordering_ignores_ier(method, expected):
    assert select(method) == expected


def test_random_is_deterministic_and_budget_sized():
    tip_compat._ROLLOUT_COUNTER = 0
    assert select('random') == select('random')
    assert len(select('random')) == 3


def test_sparse_sampled_rkl_keeps_global_extreme():
    assert select('sampled_rkl_max', .001) == [1]
    assert select('sampled_rkl_min', .001) == [0]


def test_serving_logprob_format_is_parsed():
    assert tip_compat._parse_top_items([[-.3, 7], [-.8, 3]]) == [(7, -.3), (3, -.8)]


def test_non_opd_rollout_does_not_request_topk_by_default():
    payload = {}
    tip_compat.maybe_add_topk_request(SimpleNamespace(use_opd=False, opd_topk_metrics_k=16), payload)
    assert payload == {}
