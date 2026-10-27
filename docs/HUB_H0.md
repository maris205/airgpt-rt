# AirGPT Hub H0

H0 defines the contract between AirGPT Runtime and the independent AirGPT Hub product.
The public catalog is hosted as the ModelScope Dataset
[`mariswang/AirGPT-Embodied-Model-Index`](https://modelscope.cn/datasets/mariswang/AirGPT-Embodied-Model-Index).
It indexes upstream repositories and does not copy their model weights.

## Commands

```bash
airgpt hub search robot
airgpt hub search --task world --verified-only
airgpt hub info openvla-7b
airgpt pull openvla-7b --cache-dir .cache/modelscope
```

`hub search` fetches the current catalog and falls back to the copy bundled with the
installed Runtime when ModelScope is unreachable. Configure a different catalog with
`AIRGPT_HUB_CATALOG_URL`, `--catalog-url`, or `--catalog-file`.

`pull` uses `modelscope_hub.HubApi.download_repo`, pins the verified upstream revision,
supports `MODELSCOPE_API_TOKEN`, `MODELSCOPE_CACHE`, `HTTP_PROXY`, and `HTTPS_PROXY`
environment
variables, and prints a launch command using the downloaded path. Tokens are never
accepted as command-line flags, so they do not enter shell history or process arguments.

## Manifest boundary

Each entry records:

- immutable upstream repository revision, gated state, and license identifier;
- AirGPT registry model, backend, adapter, device, and dtype;
- minimum and recommended VRAM guidance;
- verification status, date, device, and completed checks.

`manifest.schema.json` is generated from the Pydantic contract in
`airgpt_service.hub`. Catalog changes must validate against both representations before
publication.

## Publishing

Organization maintainers publish with a fine-grained write token:

```bash
export MODELSCOPE_API_TOKEN=...
HTTPS_PROXY=http://127.0.0.1:7890 python scripts/audit_hub_catalog.py
HTTPS_PROXY=http://127.0.0.1:7890 python scripts/publish_hub_catalog.py
```

The audit downloads a small file at every pinned revision through ModelScope and rejects
unavailable repositories or revisions. The publisher then validates `index.json`, creates
the Dataset repository if necessary, and uploads only the catalog files. Model weights
remain in their ModelScope repositories and retain their original licenses.
