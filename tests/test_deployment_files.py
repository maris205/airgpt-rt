from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_compose_profiles_isolate_runtime_services():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert set(services) == {"console", "openvla", "world", "dreamer", "octo"}
    for name, service in services.items():
        assert service["profiles"] == [name]
        assert service["restart"] == "unless-stopped"
    for name in ("openvla", "world", "octo"):
        devices = services[name]["deploy"]["resources"]["reservations"]["devices"]
        assert devices[0]["capabilities"] == ["gpu"]


def test_runtime_dockerfiles_define_healthchecks():
    for name in ("gateway", "openvla", "world", "dreamer", "octo"):
        content = (ROOT / f"Dockerfile.{name}").read_text(encoding="utf-8")
        assert "HEALTHCHECK" in content
        assert "/health/live" in content
