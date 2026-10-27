# M4 Octo, SDK, and Simulator Clients

M4 adds the isolated Octo Small 1.5 JAX worker and examples for using the shared
robot-action contract from OpenVLA, Octo, and LTX-Video.

## Octo worker

The worker uses the official Octo source at commit
`241fb3514b7c40957a86d869fecb7c7fc353f540` in the `airgpt-octo` environment. The
checkpoint is local at `.cache/models/octo-small-1.5`; the language tokenizer cache is
downloaded from ModelScope to `.cache/models/t5-base`, and offline loading is enabled by
default.

```bash
./scripts/setup_octo.sh
./scripts/run_octo.sh
./scripts/smoke_octo_api.sh
```

The worker listens on `http://127.0.0.1:8003`. The first request loads the checkpoint
and compiles the JAX action head. M0 measured roughly 20 seconds for loading, 13 seconds
for the first compiled inference, and 0.3 seconds for warm inference; the current warm
API requests range from about 0.07 to 0.44 seconds after compilation.

Octo returns the first action in the common response `action` field. The remaining
chunk is preserved in `action_metadata.action_chunk`. All seven values are unnormalized
BridgeV2 action coordinates after applying the configured `bridge_dataset` statistics:
three position deltas, three axis-angle rotation deltas, and one gripper value. This
normalization and frame convention is model/dataset-specific and must not be treated as
a universal robot action space.

## SDK and examples

The SDK accepts `bytes` and `pathlib.Path` image inputs, exposes typed result aliases,
and retries only idempotent model listing/loading/unloading operations. Inference POSTs
are never automatically replayed.

```bash
python examples/openvla_demo.py .cache/samples/bridge_im_12.jpg "pick up the fork"
python examples/octo_demo.py .cache/samples/bridge_im_12.jpg "pick up the fork"
python examples/world_demo.py .cache/samples/bridge_im_12.jpg "move the gripper toward the fork"
```

`examples/isaac_sim_client.py` and `examples/airsim_client.py` call the same API and
print simulator-facing coordinate mappings. They do not import or install Isaac Sim or
AirSim. The mappings are explicit assumptions: Isaac uses the Bridge delta pose as-is,
while AirSim converts the +Z-up position delta to NED +Z-down. Validate these mappings
against the simulator robot's actual base and gripper conventions before applying them.

All examples can point at the mock service for contract testing by setting
`--base-url http://127.0.0.1:8000` and selecting the corresponding model behavior.
