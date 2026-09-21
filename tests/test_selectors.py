"""Selection contracts with hand-computed batch ranks, budgets, and fusion scores."""
from types import SimpleNamespace

import pytest

from slime.rollout import tip_compat as selectors


def run_selection(method, values, lengths, ratio=0.5, min_keep=0, signals=None, valid=None):
    rows = []
    groups = {}
    samples = []
    cursor = 0
    for sample_index, length in enumerate(lengths):
        groups[sample_index] = []
        for pos in range(length):
            row = dict(ier_r=values[cursor], H_norm=(signals or [0.5] * len(values))[cursor],
                       pos_norm=pos / max(length - 1, 1), tok_pos=pos,
                       loss_mask_original=1 if valid is None else valid[cursor])
            rows.append(row)
            groups[sample_index].append(row)
            cursor += 1
        samples.append(SimpleNamespace(loss_mask=[1] * length))
    args = SimpleNamespace(opd_budget_mask=method, opd_budget_ratio=ratio,
                           opd_budget_min_keep_per_sample=min_keep, opd_budget_mask_seed=42)
    selectors._ROLLOUT_COUNTER = 0
    selectors._apply_budget_mask(args, rows, groups, samples)
    return rows, [i for i, row in enumerate(rows) if row['budget_keep']], samples


@pytest.mark.parametrize('method,expected_scores', [
    ('entropy_ier_and', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    ('entropy_ier_or', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
])
def test_short_and_long_responses_share_one_ier_rank(method, expected_scores):
    rows, kept, samples = run_selection(method, [2, 3, 10, 20, 30, 40], [2, 4])
    assert [row['ier_rank'] for row in rows] == pytest.approx([0, .2, .4, .6, .8, 1])
    assert [row['budget_score'] for row in rows] == pytest.approx(expected_scores)
    assert kept == [3, 4, 5]
    assert samples[0].loss_mask == [0, 0]
    assert samples[1].loss_mask == [0, 1, 1, 1]


def test_prefix_has_a_single_batch_budget():
    # Five tokens, ceil(5*.4)=2. Separate response rounding would select three.
    _, kept, _ = run_selection('prefix', [0] * 5, [1, 4], ratio=.4)
    assert kept == [0, 1]


def test_global_budget_rounds_up_once():
    _, kept, _ = run_selection('ier', [1, 2, 3, 4, 5], [2, 3], ratio=.21)
    assert kept == [3, 4]


def test_minimum_fallback_uses_existing_batch_fusion_scores():
    rows, kept, _ = run_selection('entropy_ier_and', [2, 3, 10, 20, 30, 40], [2, 4],
                                  ratio=.2, min_keep=1)
    assert kept == [1, 4, 5]  # global budget 2, plus short response's best token
    assert rows[1]['budget_score'] == pytest.approx(.1)  # no response-local rerank
    assert rows[5]['budget_score'] == pytest.approx(.5)


def test_masked_tokens_do_not_affect_batch_ranks_or_budget():
    rows, kept, _ = run_selection('entropy_ier_and', [1, 1e9, 2, 3], [2, 2],
                                  valid=[1, 0, 1, 1], ratio=.5)
    assert [rows[i]['ier_rank'] for i in [0, 2, 3]] == [0, .5, 1]
    assert kept == [2, 3]


def test_stable_tie_ranks_and_singleton_are_explicit():
    rows, _, _ = run_selection('entropy_ier_and', [4, 4, 4], [1, 2])
    assert [row['ier_rank'] for row in rows] == [0, .5, 1]
    rows, kept, _ = run_selection('entropy_ier_and', [4], [1], ratio=.001)
    assert rows[0]['ier_rank'] == 0
    assert kept == [0]


def test_or_and_select_different_known_tradeoffs():
    # Ranks [0,1/3,2/3,1], usefulness [1,.5,.8,0].
    # OR [1,2/3,14/15,1] selects endpoints (raw IER breaks their tie).
    # AND [0,1/6,8/15,0] selects the middle two.
    _, or_kept, _ = run_selection('entropy_ier_or', [1, 2, 3, 4], [2, 2], signals=[1, .5, .8, 0])
    _, and_kept, _ = run_selection('entropy_ier_and', [1, 2, 3, 4], [2, 2], signals=[1, .5, .8, 0])
    assert or_kept == [0, 3]
    assert and_kept == [1, 2]


def test_full_preserves_original_valid_mask():
    _, kept, _ = run_selection('full', [float('nan')] * 4, [2, 2], valid=[1, 0, 1, 0], ratio=.01)
    assert kept == [0, 2]


def test_no_valid_tokens_stay_unselected():
    _, kept, samples = run_selection('ier', [1, 2], [2], valid=[0, 0], min_keep=1)
    assert kept == []
    assert samples[0].loss_mask == [0, 0]


@pytest.mark.parametrize('method', ['ier', 'entropy_ier_or', 'entropy_ier_and'])
def test_nonfinite_ier_cannot_silently_enter_selection(method):
    with pytest.raises(ValueError, match='finite'):
        run_selection(method, [float('nan'), 1], [2])


def test_public_selector_set():
    expected = {'full', 'ier', 'prefix', 'entropy', 'tip', 'ta_opd', 'ca_softor',
                'random', 'sampled_rkl_max', 'sampled_rkl_min'}
    for base in ('prefix', 'entropy', 'tip', 'ta_opd', 'ca_softor'):
        expected.update({base + '_ier_or', base + '_ier_and'})
    assert set(selectors.BUDGET_METHODS) == expected
    with pytest.raises(ValueError, match='Unknown'):
        run_selection('unsupported_selector', [1], [1])


def test_normalization_excludes_invalid_tokens():
    rows = [dict(loss_mask_original=valid, Hs_topk_norm=value, KLf_union=value, Cmass=value)
            for valid, value in [(1, 2), (1, 4), (0, 1e10)]]
    selectors._add_normalized_scores(SimpleNamespace(opd_metric_q_low=0.0, opd_metric_q_high=1.0), rows)
    assert [row['H_norm'] for row in rows] == [0, 1, 0]


def test_main_quantile_normalization_clips_extremes():
    # For [0,1,2,3,4], the 5% and 95% quantiles are .2 and 3.8.
    assert selectors._normalize([0, 1, 2, 3, 4], SimpleNamespace()) == pytest.approx(
        [0, 2/9, 1/2, 7/9, 1])


def test_unreported_score_configuration_is_rejected():
    with pytest.raises(ValueError, match='compat-proxy mass'):
        selectors._add_normalized_scores(SimpleNamespace(opd_compat_proxy='unsupported'), [])
    with pytest.raises(ValueError, match='normalization batch_quantile'):
        selectors._normalize([0, 1], SimpleNamespace(opd_metric_normalization='unsupported'))


def test_full_rollout_processing_preserves_sequences_and_exports_current_scores(tmp_path):
    import csv
    import json
    import math

    lengths = [2, 4]
    samples = []
    rewards = []
    teacher_log_probs = []
    s_top = [[math.log(.5), 0], [math.log(1/3), 1], [math.log(1/6), 2]]
    t_top = [[math.log(.5), 0], [math.log(1/3)-1, 1], [math.log(1/6)-2, 2]]
    for i, length in enumerate(lengths):
        samples.append(SimpleNamespace(response_length=length, tokens=[9]+[0]*length,
                                       metadata={'student_top_logprobs': [s_top]*length},
                                       rollout_log_probs=[math.log(.5)]*length, loss_mask=[1]*length,
                                       index=i, group_index=0, status='completed'))
        rewards.append({'meta_info': {'input_top_logprobs': [t_top]*length}})
        teacher_log_probs.append([math.log(.5)]*length)
    args = SimpleNamespace(use_opd=True, opd_topk_metrics_k=16, opd_budget_mask='entropy_ier_and',
                           opd_budget_ratio=.5, opd_budget_min_keep_per_sample=1,
                           opd_token_bank_dir=str(tmp_path))
    selectors._ROLLOUT_COUNTER = 0
    selectors.process_tip_compat_metrics(args, samples, teacher_log_probs, rewards)
    assert [len(sample.tokens) for sample in samples] == [3, 5]
    assert [len(sample.rollout_log_probs) for sample in samples] == [2, 4]
    rows = list(csv.DictReader((tmp_path / 'rollout_000000.csv').open()))
    assert [float(row['ier_r']) for row in rows] == pytest.approx([10/13]*6)
    assert [float(row['ier_rank']) for row in rows] == pytest.approx([0,.2,.4,.6,.8,1])
    config = json.loads((tmp_path / 'config.json').read_text())
    assert config['selection']['floor_domain'] == 'log-probability'
    assert config['selection']['floor_logp'] == -12
    assert config['selection']['min_keep_per_sample'] == 1
    assert selectors._ROLLOUT_COUNTER == 1
    args.opd_token_bank_dir = None
    selectors.process_tip_compat_metrics(args, samples, teacher_log_probs, rewards)
    assert selectors._ROLLOUT_COUNTER == 2  # export settings do not control seed progression
