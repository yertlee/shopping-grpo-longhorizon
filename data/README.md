# Data and publication boundary

The public tree contains aggregate metadata and reproducibility manifests for
the SFT collection. It does not publish per-example training trajectories.
SFT-ready rows must be rebuilt from an authorized data source at run time and
kept in an external output directory; no checked-in command assumes that a
removed row file exists.

| Stage | Public artifact | Aggregate count |
|---|---|---:|
| SFT | `sft/metadata.json`, `sft_curriculum/manifest.json`, `sft_pure_v4/metadata.json` | 398 train / 100 dev ready |
| GRPO | `grpo/train.parquet`, `grpo/validation.parquet` and metadata | 1000 / 50 tasks |
| Evaluation | `evaluation/tasks.jsonl` (Final-200 Clean) and metadata | 200 tasks |

The SFT metadata records collection, curation, and split counts without raw
payloads. The curriculum manifest records the frozen task-set identity and
stage counts; its source is an authorized, externally supplied dataset. All
published splits remain task-disjoint. Generated trajectories belong under
`outputs/`, never under `data/`. See
[`docs/data-collection.md`](../docs/data-collection.md) for the v1 rebuild
contract.
