# Pure DeepSeek-v4 SFT pool

The per-example Pure V4 trajectory pool is intentionally not published in
this repository. Build it from the authorized source identified by
`metadata.json`, then keep the derived rows in an external output directory.

- `metadata.json`: aggregate counts, hashes, labeling provenance, and mix feasibility.
- `difficulty_labels.jsonl`: curation labels used by the curriculum.
- `duplicate_report.json`: aggregate merge-audit record.

The active SFT recipe keeps the natural 23.8% / 66.9% / 9.2% difficulty mix;
forcing 30% / 50% / 20% would discard valid rows merely because hard examples
are scarce. The deterministic split and cumulative training stages live in
[`../sft_curriculum/`](../sft_curriculum/README.md).
