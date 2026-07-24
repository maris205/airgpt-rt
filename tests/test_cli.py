from pathlib import Path

from airgpt_service.cli import main


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"


def test_cli_lists_registered_models(capsys):
    assert main(["models", "list", "--config", str(CONFIG)]) == 0
    output = capsys.readouterr().out
    assert "openvla" in output
    assert "octo_jax" in output


def test_cli_serve_builds_single_model_app(monkeypatch):
    captured = {}

    def fake_run(app, **kwargs):
        captured["app"] = app
        captured["kwargs"] = kwargs

    monkeypatch.setattr("airgpt_service.cli.uvicorn.run", fake_run)
    assert (
        main(
            [
                "serve",
                "--model",
                "octo",
                "--config",
                str(CONFIG),
                "--worker-mode",
                "mock",
                "--lazy",
            ]
        )
        == 0
    )
    assert captured["kwargs"]["port"] == 8003
    assert captured["app"].title == "AirGPT Embodied AI Model Service"
