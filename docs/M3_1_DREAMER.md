# M3.1 DreamerV3 State Worker

M3.1 adds a task-specific state-space world-model endpoint backed by the official
DreamerV3 implementation at commit
`e3f02248693a79dc8b0ebd62c93683888ddaccfe`.

## Scope

The first reproducible task is `gym_CartPole-v1` with the following fixed contract:

- Observation: four finite `float32` values in Gym CartPole order.
- Action: a scalar integer, `0` or `1`.
- Rollout: a non-empty action sequence with a maximum horizon of 100.
- RSSM latent: eight deterministic values plus two categorical groups of four,
  resulting in a 16-value flattened latent per step.
- Checkpoint: `artifacts/dreamer/cartpole-debug`, trained with the `debug` config for
  4,983 environment steps.

This is a task-specific baseline and is not a universal image-to-video model. The
checkpoint cannot consume arbitrary robot images or a different action space without a
matching environment adapter, normalization contract, and trained weights.

## Reproduce

The AutoDL workflow uses an isolated CPU JAX environment. GPU can remain enabled for
other workers; this low-dimensional RSSM path does not require CUDA.

```bash
./scripts/setup_dreamer.sh
./scripts/train_dreamer_cartpole.sh
```

`setup_dreamer.sh` pins the source checkout, applies
`patches/dreamerv3-gym026.patch`, installs `requirements/dreamer.txt`, and installs
AirGPT editable without installing the Dreamer package metadata (which otherwise pulls
optional Atari/CUDA dependencies).

## Run the worker

```bash
./scripts/run_dreamer.sh
./scripts/smoke_dreamer_api.sh
```

The worker listens on `http://127.0.0.1:8002` by default and loads the checkpoint on
the first request. The SDK contract is:

```python
from airgpt_sdk import Client

with Client("http://127.0.0.1:8002") as client:
    result = client.state.predict(
        "dreamer_cartpole",
        observation=[0.01, 0.02, 0.03, 0.04],
        actions=[0, 1, 0, 1, 0],
        seed=0,
    )
```

The response contains `latent_states`, decoded observation predictions, reward
predictions, categorical entropy as `uncertainty`, and checkpoint/task/shape metadata.
The validated reference request returns five 16-dimensional latents and five
four-dimensional decoded observations. CPU latency in this AutoDL environment is about
2.2 seconds for the first five-step request after loading.

## Environment notes

The official adapter predates Gym 0.26. The repository patch handles tuple-returning
`reset()` and the five-value `step()` API while preserving Dreamer terminal semantics.
Docker is not available inside the current AutoDL container, so this worker is verified
with native Conda commands. Docker packaging remains a later deployment task for a
Docker-capable host.
