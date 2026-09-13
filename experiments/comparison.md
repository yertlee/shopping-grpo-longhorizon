# Commerce v1: M0–M3 aggregate comparison

This file is the current fork's compact aggregate view. The canonical
aggregate-only record, frozen identities, and machine-readable summary live in
[`commerce-v1/`](commerce-v1/). The older `baseline/`, `sft/`, and `grpo/`
directories are upstream historical snapshots.

All four milestones use the same Final-200 Clean split, one rollout per task,
and a fixed denominator of 200.

| Model | Strict success | Strict rate | Infrastructure-invalid | Mean steps | Mean terminal utility |
|---|---:|---:|---:|---:|---:|
| M0 Base | 2/200 | 1.0% | 2 | 5.40 | −0.100 |
| M1 Outcome SFT | 137/200 | 68.5% | 3 | 12.05 | +0.584 |
| M2 Process SFT | 130/200 | 65.0% | 4 | 11.50 | +0.553 |
| M3 GRPO step50 export | 131/200 | 65.5% | 2 | 11.05 | +0.563 |

## Paired strict-success comparisons

The comparisons are task-paired, using two-sided exact McNemar tests; utility
intervals are paired bootstrap 95% CIs.

| Transition | Delta | Discordant losses / wins | p-value | Utility-difference CI |
|---|---:|---:|---:|---|
| M0 → M1 | +67.5pp | 0 / 135 | <0.0001 | [+0.589, +0.776] |
| M1 → M2 | −3.5pp | 16 / 9 | .230 | [−0.100, +0.038] |
| M2 → M3 | +0.5pp | 6 / 7 | 1.000 | [−0.040, +0.061] |

## Interpretation

SFT is the main source of the observed gain. Process selection does not beat
Outcome. Under the frozen GRPO recipe (`lr=1e-6`, LoRA `r=16`, four rollouts
per prompt, `total_training_steps=500`, `save_freq=50`) and this run's
controlled stop at optimizer step 100, the step50 export produced no detectable
gain over M2. Infrastructure-invalid results are retained in the denominator;
they are `not_judged` in the judge panel.

For the full public result narrative, see [`docs/results-v1.md`](../docs/results-v1.md).
