# AirGPT Hub Development Plan

## Product boundary

AirGPT Hub is an independent vertical model product for embodied AI. Hugging Face is the
upstream repository, download, revision, and organization layer. AirGPT adds curation,
runtime compatibility, hardware guidance, verification results, and one-command local
deployment.

Hub does not redistribute third-party weights by default. It indexes immutable upstream
revisions and preserves the original license and gated-access requirements.

## Current status

### H0 - Catalog and Runtime contract

Status: **complete**.

- Public Dataset catalog: `airgpt/embodied-model-index`.
- Pydantic manifest contract and generated JSON Schema.
- Verified OpenVLA, Octo, and LTX-Video entries with pinned revisions.
- Offline catalog bundled with AirGPT Runtime.
- `airgpt hub search`, `airgpt hub info`, and `airgpt pull`.
- Hugging Face proxy, token, cache, revision, and local-path support.
- Hugging Face API audit for pinned revision, gated-state, and license drift.
- Catalog publisher for organization maintainers.

### H1 - Public Web MVP

Status: **next**.

- Create the independent `airgpt-hub` application and Docker Space.
- Build model search, task filters, compatibility badges, and model detail pages.
- Read the public Dataset catalog without requiring a user token.
- Show verified launch commands, runtime versions, VRAM guidance, and license links.
- Add an authenticated maintainer-only catalog validation and publish workflow.
- Keep persistent analytics and editorial metadata outside ephemeral Space storage.

Exit criteria: a lab user can find a verified embodied model, inspect its resource and
license constraints, and copy or download a working AirGPT Runtime command.

### H2 - Lab and organization workflows

- Private catalogs and team workspaces.
- Resumable download jobs, mirror endpoints, and domestic-network routing.
- Runtime fleet registration and deployment history.
- Community submissions with automated schema, license, and smoke-test checks.

### H3 - Hosted verification

- GPU-backed compatibility jobs for submitted revisions.
- Reproducible benchmark artifacts and signed verification records.
- Optional hosted demos using ZeroGPU or dedicated endpoints where the model permits it.
