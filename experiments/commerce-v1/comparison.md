# Commerce v1 aggregate comparison

This comparison is a frozen, aggregate-only record. The denominator is 200 for
each milestone. TU means aggregate terminal utility; infrastructure-invalid is
the count of evaluations excluded by the infrastructure validity rule.

| Milestone | Strict success | Strict rate | TU | Infrastructure-invalid |
|---|---:|---:|---:|---:|
| M0 | 2 / 200 | 1.0% | -0.100 | 2 |
| M1 | 137 / 200 | 68.5% | 0.584 | 3 |
| M2 | 130 / 200 | 65.0% | 0.553 | 4 |
| M3 step100 | 139 / 200 | 69.5% | 0.602 | 2 |

## Pairwise strict-success comparison

| Transition | Delta (percentage points) | p-value |
|---|---:|---:|
| M0 → M1 | +67.5 | <0.0001 |
| M1 → M2 | -3.5 | 0.230 |
| M2 → M3 | +4.5 | 0.093 |

## Frozen identities

- Protocol SHA-256: `0986526cecc9b1a9770c7786b049689a95528c4c6f72a11be78f6400b5049cec`
- Final split SHA-256: `d99112a20ef47534c27a32e4b38229bf048dcc6b06fef2e3e919aac3093662f5`
- M3 checkpoint: optimizer step 100

Only aggregate values and frozen hashes are included; no per-example payload or
artifact location is part of this publication.
