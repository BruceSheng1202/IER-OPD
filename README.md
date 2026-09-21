# IER for On-Policy Distillation
<img width="2172" height="724" alt="image" src="https://github.com/user-attachments/assets/42eee8de-307c-40ea-9921-cf56581d82cc" />

Reproduction code for **1% of Tokens Can Be Enough: On Gradient Estimation in On-Policy Distillation**.

IER selects response tokens for the sampled reverse-KL training loss. Student responses are generated in full. The codebase is built upon [slime](https://github.com/THUDM/slime) and [TA-OPD](https://github.com/wyy-code/TA-OPD).

<img width="1058" height="698" alt="image" src="https://github.com/user-attachments/assets/3a5fa393-49be-4602-b2ab-1145c3981a30" />


## Layout

```text
configs/         Training, model, and evaluation configurations
scripts/         Training, evaluation, and checkpoint conversion
slime/           Shared runtime and the IER implementation
slime_plugins/   Qwen2/Qwen3 checkpoint mappings
third_party/     Dependency versions, training patches, and upstream license
```

The IER formula is in `slime/rollout/ier_opd/ier_metrics.py`; token selection and fusion are in `slime/rollout/tip_compat.py`.

## Installation

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r requirements/train.txt
```

## Data and model preparation

Obtain the teacher and student checkpoints listed below and download the datasets from their public sources:

| Use | Dataset | Source |
|---|---|---|
| Mathematics training | DAPO-Math-17k | [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) |
| Medical training | RaR-Medicine, train split | [ScaleAI/RaR-Medicine](https://huggingface.co/datasets/ScaleAI/RaR-Medicine) |
| Mathematics evaluation | AIME 2025 | [MathArena/aime_2025](https://huggingface.co/datasets/MathArena/aime_2025) |
| Mathematics evaluation | AIME 2026 | [MathArena/aime_2026](https://huggingface.co/datasets/MathArena/aime_2026) |
| Mathematics evaluation | HMMT February 2025 | [MathArena/hmmt_feb_2025](https://huggingface.co/datasets/MathArena/hmmt_feb_2025) |
| Mathematics evaluation | HMMT February 2026 | [MathArena/hmmt_feb_2026](https://huggingface.co/datasets/MathArena/hmmt_feb_2026) |

## Main experiments

| Profile | Teacher → student | Data | Rollouts |
|---|---|---|---:|
| `math_nemotron` | JustRL-Nemotron-1.5B → OpenMath-Nemotron-1.5B | DAPO-Math-17k | 50 |
| `math_qwen3` | JustRL-Qwen3-4B → Qwen3-1.7B | DAPO-Math-17k | 50 |
| `medical_qwen3` | ClinAlign-4B → Qwen3-4B | RaR-Medicine | 100 |

All profiles use 4 prompts × 16 responses per rollout, 8 minibatches of size 8, temperature 1, top-p 1, and prompt/response/context limits of 2048/8192/16384 tokens. Adam uses learning rate 1e-6, weight decay 0.1, and betas (0.9, 0.98). Qwen3 profiles explicitly disable thinking. Nemotron uses its model's chat template.

Run the main experiments:

```bash
python scripts/train.py --profile math_qwen3 --method ier --budget 0.01 --execute
python scripts/train.py --profile math_nemotron --method tip_ier_and --budget 0.01 --execute
python scripts/train.py --profile medical_qwen3 --method tip_ier_or --budget 0.01 --execute
python scripts/train.py --profile math_qwen3 --method full --budget 1.0 --execute
```

| Method | CLI method |
|---|---|
| IER | `ier` |
| Prefix / Entropy / TIP / TA-OPD / CA-SoftOR | `prefix` / `entropy` / `tip` / `ta_opd` / `ca_softor` |
| Base + IER-OR | Append `_ier_or` to one of the five base names |
| Base + IER-AND | Append `_ier_and` to one of the five base names |
| Full OPD | `full` |
| Random / Sampled RKL max (mathematics) | `random` / `sampled_rkl_max` |

Token budgets are `0.001`, `0.01`, and `0.1`. Full OPD uses `1.0`. To run Sampled RKL min, use `--suite reported_comparisons --method sampled_rkl_min`.

Training outputs are saved under `OUTPUT_ROOT/<run-name>`.

## Mathematical evaluation

```bash
# Qwen3
python scripts/evaluate_math.py \
  --checkpoint /path/to/qwen3-hf-checkpoint \
  --data-root /path/to/benchmarks \
  --output-dir outputs/qwen3-eval --thinking off --execute

# Nemotron
python scripts/evaluate_math.py \
  --checkpoint /path/to/nemotron-hf-checkpoint \
  --data-root /path/to/benchmarks \
  --output-dir outputs/nemotron-eval --thinking model-default --execute
```

Evaluation uses 32 samples per problem, temperature 0.7, top-p 0.9, and a 32768-token limit, reporting Bayes@32. Settings are in `configs/math_eval.json`.

## Medical evaluation

Use OpenAI's [simple-evals](https://github.com/openai/simple-evals) and its
[HealthBench evaluator](https://github.com/openai/simple-evals/blob/main/healthbench_eval.py).
Follow the upstream instructions for setup and evaluation. Evaluate HealthBench
overall and HealthBench Hard with **gpt-oss-120B** as the judge.
