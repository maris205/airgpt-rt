import json
from pathlib import Path

from airgpt_service.cli import main
from airgpt_service.hub import BUNDLED_CATALOG, HubCatalog, HubClient, launch_command


def test_bundled_catalog_is_valid_and_unique():
    catalog = HubCatalog.model_validate_json(BUNDLED_CATALOG.read_text(encoding="utf-8"))
    assert catalog.schema_version == 1
    assert {model.slug for model in catalog.models} == {
        "openvla-7b",
        "octo-small-1.5",
        "ltx-video-0.9.5",
    }
    assert all(model.verification.status == "verified" for model in catalog.models)


def test_published_schema_matches_runtime_contract():
    schema_path = BUNDLED_CATALOG.with_name("manifest.schema.json")
    assert json.loads(schema_path.read_text(encoding="utf-8")) == HubCatalog.model_json_schema()


def test_hub_search_filters_task_and_query():
    client = HubClient(catalog_file=BUNDLED_CATALOG)
    assert [model.slug for model in client.search("action", task="vla")] == [
        "octo-small-1.5",
        "openvla-7b",
    ]
    assert [model.slug for model in client.search(task="world")] == ["ltx-video-0.9.5"]


def test_hub_cli_search_and_dry_run(capsys):
    assert (
        main(
            [
                "hub",
                "search",
                "octo",
                "--catalog-file",
                str(BUNDLED_CATALOG),
                "--json",
            ]
        )
        == 0
    )
    body = json.loads(capsys.readouterr().out)
    assert body["models"][0]["slug"] == "octo-small-1.5"

    assert (
        main(
            [
                "pull",
                "octo-small-1.5",
                "--catalog-file",
                str(BUNDLED_CATALOG),
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    body = json.loads(capsys.readouterr().out)
    assert body["dry_run"] is True
    assert "airgpt serve --model octo" in body["launch_command"]


def test_hub_cli_reports_missing_catalog(capsys, tmp_path: Path):
    assert (
        main(
            [
                "hub",
                "search",
                "--catalog-file",
                str(tmp_path / "missing.json"),
            ]
        )
        == 2
    )
    assert "unable to load Hub catalog file" in capsys.readouterr().err


def test_pull_uses_pinned_revision(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(tmp_path / "snapshot")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    client = HubClient(catalog_file=BUNDLED_CATALOG)
    model, path = client.pull("openvla-7b", cache_dir=tmp_path / "cache")
    assert captured["repo_id"] == "openvla/openvla-7b"
    assert captured["revision"] == model.source.revision
    assert captured["cache_dir"] == str(tmp_path / "cache")
    assert path == tmp_path / "snapshot"
    assert "--dtype bfloat16" in launch_command(model, path)
