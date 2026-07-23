# DreamerV3 State-Space Integration Evaluation

DreamerV3 is a good follow-on target, but it is not a drop-in replacement for the
current LTX image-to-video API.

## Why it is easier to operate

- The core RSSM latent dynamics, actor, and critic are much smaller than a video
  diffusion checkpoint and can run comfortably in a separate CPU/GPU worker.
- Rollouts produce compact latent states and action-conditioned predictions instead of
  repeatedly decoding full-resolution video frames.
- It aligns naturally with robot control loops: observation, action, latent state,
  horizon, and policy/value outputs.

## What makes it less universal

- DreamerV3 checkpoints are environment/task-specific. An Atari or DMControl checkpoint
  cannot consume an arbitrary robot RGB image without its matching observation encoder,
  action space, normalization, and training configuration.
- The primary output is a latent/state distribution or imagined rollout, not a viewable
  future video. Pixel reconstruction requires the matching decoder.
- Robotics integration needs a fixed environment contract: observation schema, action
  bounds, reward/termination semantics, and checkpoint provenance.

## Recommended service shape

Add a separate `state_world` model type and `/v1/state/predict` contract rather than
overloading `/v1/world/predict`:

```text
observation or encoded state + action sequence + horizon
    -> DreamerV3 RSSM transition
    -> latent states / uncertainty / optional decoded observations
```

The first implementation should target one known Gymnasium/robot task, return latent
states and metadata, and keep the worker isolated from the Diffusers environment. This
is a strong M3.1/M4 candidate after the LTX artifact path is stable.
