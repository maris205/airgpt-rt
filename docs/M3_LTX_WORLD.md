# M3 LTX-Video World Runtime Report

## Implementation

- Runtime: Diffusers 0.33.1 and PyTorch 2.5.1+cu121 in `airgpt-world`.
- Checkpoint: local `.cache/models/ltx-video-0.9.5`.
- Device and dtype: CUDA 0, BF16, sequential CPU offload, VAE tiling.
- Conditioning: initial RGB image plus scene/action/prompt text.
- Service limit: at most 49 frames, with LTX-compatible `8n + 1` frame counts.
- Artifacts: PNG frames and MP4 are stored below `AIRGPT_ARTIFACT_DIR` and served as
  `/artifacts/...` URLs. TTL and total run count are configurable.

## API Verification

Measured on the AutoDL NVIDIA vGPU-32GB on 2026-07-23:

- First SDK-to-API request including model load: 37.694 seconds.
- Generation latency after load: 17.750 seconds for 384x256, 9 frames, 8 steps.
- Peak GPU allocation remains the M0 measurement of approximately 11.4 GiB.
- Returned 9 PNG frames and one MP4 artifact.
- MP4 decoding confirmed 9 frames at 384x256; first and last frames are non-empty.
- Unload left only the process CUDA context; no model weights remained resident.

## Commands

```bash
./scripts/run_ltx_world.sh
./scripts/smoke_ltx_api.sh
RUN_GPU_TESTS=1 AIRGPT_WORLD_BASE_URL=http://127.0.0.1:8001 \
  pytest tests/gpu/test_ltx_api.py
```
