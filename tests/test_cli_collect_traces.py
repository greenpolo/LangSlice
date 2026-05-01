from __future__ import annotations


def test_collect_traces_cli_calls_collector(monkeypatch, tmp_path):
    from langslice_harness import cli

    captured = {}

    def fake_collect_manifest_traces(**kwargs):
        captured.update(kwargs)
        return [{"id": "row-1"}]

    monkeypatch.setattr(
        "langslice_harness.harness.estimation.trace_collection.collect_manifest_traces",
        fake_collect_manifest_traces,
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("", encoding="utf-8")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "langslice",
            "collect-traces",
            "--manifest",
            str(manifest),
            "--out",
            str(out_dir),
            "--model",
            "gemini-3.1-pro-preview",
            "--limit",
            "3",
            "--kind",
            "single",
            "--no-include-thought-summaries",
        ],
    )

    cli.main()

    assert captured["manifest_path"] == manifest
    assert captured["out_dir"] == out_dir
    assert captured["model"] == "gemini-3.1-pro-preview"
    assert captured["limit"] == 3
    assert captured["kind_filter"] == "single"
    assert captured["include_thought_summaries"] is False
