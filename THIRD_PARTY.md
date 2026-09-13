# Third-party provenance

This repository is a maintained fork. The fork and publication provenance below
are factual source metadata; they are not a license statement.

## Fork source

- Upstream repository: <https://github.com/YYHDBL/shopping-grpo-longhorizon>
- Fork repository: <https://github.com/yertlee/shopping-grpo-longhorizon>
- Published branch: `v1-commerce-agent`
- Publication base commit: `17808704ce2197e6205bbf3515c89b6f06c05d68`

The embedded ShopSimulator environment is part of this repository snapshot.
No additional upstream source checkout is implied by this document.

## Dependency source metadata

Python package dependencies are sourced from the Python Package Index unless a
Git source is shown explicitly. The versions and ranges below mirror the
project declaration; optional groups are included for provenance even when a
CPU-only checkout does not install them.

- Runtime: `tqdm>=4.66,<5` (PyPI).
- Development: `build>=1.2,<2`, `pytest>=8,<10`, `ruff>=0.9,<1`, and
  `mypy>=1.14,<2` (PyPI).
- SFT: `torch>=2.5`, `peft>=0.15,<1`, `accelerate>=1,<2`, `torchvision`,
  `pillow`, and `swanlab>=0.8,<1` (PyPI).
- SFT acceleration: `bitsandbytes>=0.45,<1` and `liger-kernel>=0.6,<1`
  (PyPI).
- GRPO: `hydra-core>=1.3,<2`, `verl==0.8.0`, `vllm==0.25.1`,
  `torch==2.11.0`, `torchvision==0.26.0`, `torchaudio==2.11.0`,
  `ray[default]==2.56.1`, `tensordict==0.10.0`, `numpy==2.2.6`,
  `swanlab==0.9.1`, and `pyarrow>=18,<26` (PyPI).
- Transformers is sourced from GitHub at commit
  `7ea2320c76117e6742364808a666ef6f2fb40a67`:
  <https://github.com/huggingface/transformers/tree/7ea2320c76117e6742364808a666ef6f2fb40a67>

## License status

The upstream repository has not granted one uniform license covering this
repository as a whole. This document intentionally does not infer, assign, or
represent licenses for the fork, embedded material, or dependencies. Consult
the respective upstream project metadata before redistribution.
