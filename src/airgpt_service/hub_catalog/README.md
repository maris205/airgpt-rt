---
license: mit
pretty_name: AirGPT Embodied Model Index
task_categories:
  - robotics
tags:
  - embodied-ai
  - robotics
  - model-registry
---

# AirGPT Embodied Model Index

This dataset is the versioned catalog used by AirGPT Runtime and the future AirGPT Hub.
It indexes upstream model repositories; it does not redistribute their weights.

Every entry pins an upstream revision and records the runtime adapter, hardware guidance,
license identifier, and AirGPT verification status. Upstream model licenses continue to
apply to model weights and code.

Files:

- `index.json`: current catalog consumed by `airgpt hub`.
- `manifest.schema.json`: JSON Schema for catalog validation and third-party tooling.

Project: <https://github.com/maris205/airgpt-rt>
