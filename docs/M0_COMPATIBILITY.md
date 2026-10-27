# M0 Model Compatibility Report

## Hardware

- Environment: AutoDL container on Ubuntu 22.04.
- GPU: NVIDIA vGPU-32GB backed by RTX 4080 SUPER.
- Architecture: Ada Lovelace, compute capability 8.9.
- PyTorch-visible memory: 31.47 GiB.
- Driver: 580.76.05.
- Host CUDA compatibility: 13.0; container/toolkit and PyTorch runtime: CUDA 12.1.
- Persistent disk free before model downloads: approximately 136 GiB.
- 24 GiB allocation/write and BF16 compute sanity tests passed.

## OpenVLA

- ModelScope model: `zixiaoBios/openvla-7b`.
- ModelScope access: public and ungated.
- Revision checked: `55052867e5bddbc2179026832791c3fd2a9d1dab`.
- Checkpoint payload: approximately 14.05 GiB across three safetensor shards.
- Environment: Conda `airgpt-openvla`, Python 3.10, PyTorch 2.2.2+cu121,
  Transformers 4.40.1, Timm 0.9.10.
- Attention implementation for baseline: PyTorch SDPA. Eager attention reproduces
  upstream issue #58 because `predict_action` appends an input token without extending
  the attention mask. FlashAttention is optional and intentionally excluded from the
  first correctness test.
- Native inference: passed with a deterministic synthetic tabletop image and the
  instruction `pick up the red cube`.
- Processor load: 1.217 seconds.
- Model load and GPU transfer: 6.650 seconds from a warm local file cache.
- First action inference: 0.724 seconds with SDPA.
- GPU allocation after load: 14.09 GiB.
- Peak GPU allocation during inference: 14.412 GiB.
- Free GPU memory after inference: 16.694 GiB.
- Output: one 7-DoF action vector using the `bridge_orig` normalization statistics.
- Reproducible result: `artifacts/m0/openvla/report.json` (generated artifact, excluded
  from Git).
- Status: passed. The 31.5 GiB vGPU is sufficient for the OpenVLA v1 inference path.

## Octo

- Official checkpoints are public and ungated.
- `octo-small-1.5`: 27M parameters, approximately 0.51 GiB checkpoint payload.
- `octo-base-1.5`: 93M parameters, approximately 0.76 GiB checkpoint payload.
- GPU capacity is not a concern. Compatibility risk is the legacy pinned stack:
  JAX 0.4.20, TensorFlow 2.15, Flax 0.7.5, and NumPy 1.24.3.
- Compatibility fixes required beyond the upstream unconstrained requirements:
  Transformers 4.36.2, Orbax 0.4.4, SciPy 1.11.4, TensorFlow Metadata 1.14.0,
  Array Record 0.4.1, and W&B 0.16.0.
- Native inference: passed using the official BridgeV2 notebook image and instruction
  `pick up the fork`.
- Model load: 20.216 seconds, including device placement from a warm local cache.
- First JIT-compiled inference: 12.925 seconds.
- Warm inference: 0.301 seconds.
- GPU use after warm inference: approximately 1.31 GiB with JAX preallocation disabled.
- Output shape: `[1, 4, 7]`, representing one batch, four-action chunk, and 7-DoF
  Bridge action.
- Reproducible result: `artifacts/m0/octo/report.json`.
- Status: passed. The vGPU has substantially more capacity than Octo requires.

## World Model

- Preferred product-aligned candidate:
  `nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned` or
  `nvidia/Cosmos-Predict2-2B-Video2World`.
- No verified ModelScope repository is currently selected for the Cosmos candidate, so
  it is excluded from the ModelScope-only v1 catalog until one can be validated.
- Public fallback: `Lightricks/LTX-Video-0.9.5`, a Diffusers image-to-video model with
  text conditioning. The repository payload is approximately 23.9 GiB and will require
  CPU/model offload for a stable 31.5 GiB inference path.
- Selected v1 baseline: ModelScope `Lightricks/LTX-Video-0.9.5`, revision
  `b5ef4c942966dab55d219c651dd3a8bd0826f55c`.
- Native image-to-video inference: passed using the BridgeV2 image and a robot action
  description as text conditioning.
- Smoke configuration: BF16, sequential model CPU offload, VAE tiling, 384x256,
  9 frames, 8 diffusion steps, seed 0.
- Model load: 3.155 seconds from a warm local cache.
- Generation: 15.893 seconds.
- Peak GPU allocation: 11.423 GiB.
- Process RSS after load: 9.544 GiB.
- Output: valid H.264 MP4, 9 frames at 8 FPS. First/middle/last-frame inspection is
  nonblank, visually coherent, and shows continuous change. This smoke test does not
  claim action-semantic or video-quality accuracy.
- Reproducible results: `artifacts/m0/world/report.json`, `future.mp4`, and
  `contact_sheet.png`.
- Status: passed. LTX-Video is the ungated v1 World Runtime baseline; Cosmos remains a
  later adapter once model access is available.

## Compatibility Matrix

| Model | Runtime | Checkpoint | Peak GPU | Warm inference/generation | Result |
|---|---|---:|---:|---:|---|
| OpenVLA 7B | PyTorch/Transformers SDPA | 14.05 GiB | 14.412 GiB | 0.724 s/action | Passed |
| Octo Small 1.5 | JAX/TensorFlow utilities | 0.51 GiB | about 1.31 GiB used | 0.301 s/action chunk | Passed |
| LTX-Video 0.9.5 | Diffusers with CPU offload | 23.9 GiB | 11.423 GiB | 15.893 s/9 frames | Passed |

## M0 Decision

- The AutoDL 32 GiB vGPU is sufficient for all three v1 inference paths when models
  are isolated by worker and loaded one at a time.
- OpenVLA must use SDPA or FlashAttention, not eager attention.
- Octo must remain in its pinned legacy JAX environment.
- LTX-Video is the initial World Model; the public API should expose conditioning and
  generation metadata without hard-coding the LTX schema.
- All three Conda environments pass `pip check`, all probe scripts compile, and the GPU
  returns to 3 MiB idle usage after the processes exit.
- Disk remaining after environments and checkpoints: approximately 72 GiB. ModelScope
  download caches can be reclaimed later if disk pressure becomes important; runtime
  checkpoints live under `.cache/models`.
- M0 compatibility and contract discovery is complete. Implementation can proceed to
  the CPU/mock service skeleton and normalized API schemas.
