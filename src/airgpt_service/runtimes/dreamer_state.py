from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

from ..adapters.state.dreamer_cartpole import DreamerCartPoleAdapter
from ..errors import ServiceError
from ..registry import RegistryModel
from ..schemas import StatePredictRequest, StatePredictResponse


class DreamerV3StateRuntime:
    """Task-specific DreamerV3 RSSM runtime for Gym CartPole-v1."""

    def __init__(self, spec: RegistryModel):
        self.spec = spec
        self.device = spec.device or "cpu"
        self.adapter = DreamerCartPoleAdapter()
        self.agent: Any | None = None
        self._jax: Any | None = None
        self._jnp: Any | None = None
        self._nj: Any | None = None
        self._np: Any | None = None
        self.checkpoint_id: str | None = None
        self.loaded = False

    def _checkpoint_root(self) -> Path:
        configured = str(
            self.spec.metadata.get("checkpoint_path", "artifacts/dreamer/cartpole-debug")
        )
        path = Path(configured)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        return path

    def load(self) -> None:
        if self.loaded:
            return
        try:
            import elements
            import jax
            import jax.numpy as jnp
            import ninjax as nj
            import numpy as np
            import ruamel.yaml as yaml
            from dreamerv3 import main as dreamer_main

            config_path = Path(dreamer_main.__file__).with_name("configs.yaml")
            configs = yaml.YAML(typ="safe").load(config_path)
            configs["defaults"]["jax"].update(
                precompile=False,
                profiler=False,
                transfer_guard=False,
            )
            checkpoint_root = self._checkpoint_root()
            config = (
                elements.Config(configs["defaults"])
                .update(configs[str(self.spec.metadata.get("config", "debug"))])
                .update(
                    task=str(self.spec.metadata.get("task", "gym_CartPole-v1")),
                    logdir=str(checkpoint_root),
                    jax={
                        "platform": self.device,
                        "compute_dtype": "float32",
                        "prealloc": False,
                        "enable_policy": False,
                    },
                )
            )
            agent = dreamer_main.make_agent(config)
            latest_file = checkpoint_root / "ckpt" / "latest"
            checkpoint_id = latest_file.read_text(encoding="utf-8").strip()
            checkpoint_file = checkpoint_root / "ckpt" / checkpoint_id / "agent.pkl"
            agent.load(pickle.loads(checkpoint_file.read_bytes()))
            self.agent = agent
            self._jax = jax
            self._jnp = jnp
            self._nj = nj
            self._np = np
            self.checkpoint_id = checkpoint_id
            self.loaded = True
        except Exception as exc:
            self.unload()
            raise ServiceError("model_load_failed", f"DreamerV3 load failed: {exc}", 503) from exc

    def unload(self) -> None:
        self.loaded = False
        self.agent = None

    def predict_state(self, request: StatePredictRequest) -> StatePredictResponse:
        if not self.loaded or self.agent is None:
            raise ServiceError("model_not_ready", "DreamerV3 runtime is not loaded", 503)
        try:
            observation, actions = self.adapter.preprocess(request)
            jax, jnp, nj, np = self._jax, self._jnp, self._nj, self._np
            model = self.agent.model
            batch_size = 1
            horizon = len(actions)
            obs = {
                "image": observation[None],
                "reward": np.zeros((batch_size,), np.float32),
                "is_first": np.full((batch_size,), request.reset, bool),
                "is_last": np.zeros((batch_size,), bool),
                "is_terminal": np.zeros((batch_size,), bool),
            }
            action_sequence = actions[None]

            def rollout():
                enc_carry = model.enc.initial(batch_size)
                dyn_carry = model.dyn.initial(batch_size)
                reset = obs["is_first"]
                enc_carry, _, tokens = model.enc(enc_carry, obs, reset, training=False, single=True)
                previous_action = {"action": jnp.zeros((batch_size,), jnp.int32)}
                dyn_carry, _, posterior = model.dyn.observe(
                    dyn_carry,
                    tokens,
                    previous_action,
                    reset,
                    training=False,
                    single=True,
                )
                features = []
                for index in range(horizon):
                    dyn_carry, (feature, _) = model.dyn.imagine(
                        dyn_carry,
                        {"action": action_sequence[:, index]},
                        1,
                        training=False,
                        single=True,
                    )
                    features.append(feature)
                imagined = jax.tree.map(lambda *xs: jnp.stack(xs, axis=1), *features)
                _, _, decoded = model.dec(
                    model.dec.initial(batch_size),
                    imagined,
                    jnp.zeros((batch_size, horizon), bool),
                    training=False,
                )
                rewards = model.rew(model.feat2tensor(imagined), 2).pred()
                return posterior, imagined, decoded["image"].pred(), rewards

            start = time.perf_counter()
            _, output = nj.pure(rollout)(
                self.agent.params,
                seed=np.array([request.seed, request.seed], np.uint32),
                create=False,
                modify=False,
            )
            posterior, imagined, decoded, rewards = jax.tree.map(np.asarray, output)
            deter = imagined["deter"][0]
            stoch = imagined["stoch"][0].reshape(horizon, -1)
            latent = np.concatenate([deter, stoch], axis=-1)
            logits = imagined["logit"][0]
            probs = jax.nn.softmax(logits, axis=-1)
            entropy = -np.sum(np.asarray(probs) * np.log(np.asarray(probs) + 1e-8), axis=-1)
            uncertainty = entropy.mean(axis=-1)
            elapsed = time.perf_counter() - start
            return StatePredictResponse(
                model=request.model,
                latent_states=np.round(latent, 6).tolist(),
                predicted_observations=np.round(decoded[0], 6).tolist(),
                predicted_rewards=np.round(rewards[0], 6).tolist(),
                uncertainty=np.round(uncertainty, 6).tolist(),
                metadata={
                    "runtime": "dreamerv3",
                    "task": self.spec.metadata.get("task", "gym_CartPole-v1"),
                    "checkpoint": self.checkpoint_id,
                    "observation_dim": self.adapter.observation_dim,
                    "action_count": self.adapter.action_count,
                    "horizon": horizon,
                    "latent_dim": latent.shape[-1],
                    "posterior_deter_dim": posterior["deter"].shape[-1],
                    "latency_seconds": round(elapsed, 6),
                    "device": self.device,
                },
            )
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError(
                "inference_failed", f"DreamerV3 inference failed: {exc}", 503
            ) from exc
