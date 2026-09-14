# Commerce v1 aggregate results

This directory contains the public, aggregate-only record for the Commerce v1
evaluation. It publishes milestone counts, aggregate terminal utility (TU),
infrastructure-invalid counts, paired strict-success deltas, and the hashes
that identify the frozen evaluation inputs.

No per-example records, prompts, observations, provider outputs, or generated
artifact locations are published here. The files in this directory are not a
copy of a raw run summary or an HTML report.

## Frozen identity

- Protocol SHA-256: `0986526cecc9b1a9770c7786b049689a95528c4c6f72a11be78f6400b5049cec`
- Final split SHA-256: `d99112a20ef47534c27a32e4b38229bf048dcc6b06fef2e3e919aac3093662f5`
- M3 weights SHA-256: `e36b7a2a6029860eb17d278f186c2887f5fbc6adaed2ca2431a668aa26dadc34`

## Milestones

All milestones use the same 200-item denominator.

| Milestone | Strict success | Strict rate | TU | Infrastructure-invalid |
|---|---:|---:|---:|---:|
| M0 | 2 / 200 | 1.0% | -0.100 | 2 |
| M1 | 137 / 200 | 68.5% | 0.584 | 3 |
| M2 | 130 / 200 | 65.0% | 0.553 | 4 |
| M3 | 131 / 200 | 65.5% | 0.563 | 2 |

Pairwise strict-success deltas and p-values are recorded in
[`comparison.md`](comparison.md); machine-readable aggregate values are in
[`summary.json`](summary.json).
