# Data collection

## Public boundary

The public repository contains aggregate counts and manifests only. It does
not contain raw Teacher responses or per-example SFT trajectories. Rebuild the
SFT-ready rows from an authorized v1 source and keep all derived JSONL outside
the repository.

## v1 collection and curation

The v1 collection used ShopSimulator Environment v2.1, Reward v3, and
`deepseek-v4-flash`. Every strategy attempt executed real environment actions;
infrastructure retries preserve their attempt identity and are not new policy
examples. Strict acceptance requires a complete `gold_purchase` terminal
result with `reward_valid=true`.

| Item | Count |
|---|---:|
| Valid strategy attempts | 2,100 |
| Append-only raw rows | 2,370 |
| Strict-gold trajectories | 1,343 |
| Trajectories after hard acceptance | 1,265 |
| Tasks with usable trajectories | 512 |
| Curated train / dev / reserve | 400 / 100 / 12 |
| SFT-ready train / dev | 398 / 100 |

The Outcome and Process arms use the same task set. Outcome selects the first
strictly accepted trajectory; Process selects by actor-visible quality features
using the fixed curation order. A shared length gate and difficulty-matched
reserve reduce the curated train count from 400 to 398. No evaluation task is
used for selection or tuning.

## Rebuild from an authorized source

Set `AUTHORIZED_DATA_ROOT` to the approved data location before running these
commands. The placeholder is intentionally not a repository path.

```bash
python scripts/curate_teacher.py \
  --raw "$AUTHORIZED_DATA_ROOT/teacher/raw.jsonl" \
  --tasks "$AUTHORIZED_DATA_ROOT/task_facts.jsonl" \
  --output-dir outputs/teacher-curated \
  --collected-task-count 512 \
  --train-count 400 \
  --dev-count 100
```

Then run the length-gate/preflight workflow against that external output and
write SFT-ready rows to another external directory. Review the generated
manifest and hashes before any training run. The committed metadata and
curriculum manifest are the public audit records; raw responses and derived
rows remain outside Git.

## Publication rule

Raw collection output is resumable and may include complete Teacher responses,
environment observations, rejection reasons, and derived SFT rows. Those
artifacts stay in the authorized output area and are never copied into this
repository. During training, user and tool tokens are masked and loss is
computed only on assistant actions. See [SFT](sft.md) for the recipe and the
committed metadata/manifests for aggregate audit values.
