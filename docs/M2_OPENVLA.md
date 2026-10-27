# M2 OpenVLA Runtime Report

## Implementation

- Runtime: Transformers 4.40.1 and PyTorch 2.2.2+cu121 in `airgpt-openvla`.
- Checkpoint: ModelScope `zixiaoBios/openvla-7b` at revision
  `55052867e5bddbc2179026832791c3fd2a9d1dab`, loaded from the local checkpoint directory.
- Device and dtype: CUDA 0, BF16.
- Attention: SDPA.
- Adapter prompt: OpenVLA native action prompt.
- Action decoding: `predict_action(..., unnorm_key="bridge_orig")`.
- Public action envelope includes dimension, action space, normalization state, and
  unnormalization key.

## API Verification

Measured on the AutoDL NVIDIA vGPU-32GB on 2026-07-23:

- First SDK-to-API request including model load: 10.676 seconds.
- Warm SDK-to-API request: 0.271 seconds.
- Loaded process GPU memory reported by `nvidia-smi`: 15,135 MiB.
- Five repeated warm requests returned identical actions without memory growth.
- Unload reduced GPU memory to 345 MiB while the CUDA process remained alive.
- Service shutdown returned the GPU to its idle allocation.

Reference response:

```json
{
  "model": "openvla",
  "action": [0.007172, -0.006899, -0.009075, -0.000777, 0.007928, -0.045197, 0.0],
  "action_dim": 7,
  "action_metadata": {
    "runtime": "transformers",
    "normalized": false,
    "space": "bridge_7dof",
    "unnorm_key": "bridge_orig"
  }
}
```

## Commands

```bash
./scripts/run_openvla.sh
./scripts/smoke_openvla_api.sh
RUN_GPU_TESTS=1 pytest tests/gpu/test_openvla_api.py
```
