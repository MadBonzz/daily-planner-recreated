"""Verify optional three-model residency using the native Bionic HTTP protocol."""

import sys

from examples.estimation import check_local_models
from test_local_model_runtime import BONSAI, KEV, MODELS, QWEN, bionic, check


def test_resident_roles_clean_stale_instances_and_reuse_the_returned_ids(bionic, monkeypatch, capsys):
    monkeypatch.setenv("PLANNER_KEEP_MODELS_LOADED", "1")
    monkeypatch.setattr(sys, "argv", ["check_local_models"])
    monkeypatch.setitem(MODELS, "unrelated-model", "llm")
    bionic.max_loaded = 3
    bionic.loaded.update({"qwen-old": (QWEN, 32768), "kev-serving": (KEV, 4096),
                          "kev-duplicate": (KEV, 4096), "bonsai-old": (BONSAI, 87040),
                          "unused": ("unrelated-model", 8192)})
    bionic.instance_options["bonsai-old"] = {"parallel": 4}
    assert check_local_models.main() == 0
    assert "Loaded together:" in capsys.readouterr().out
    assert bionic.loaded == {QWEN + ":1": (QWEN, 4096), "kev-serving": (KEV, 4096), BONSAI + ":1": (BONSAI, 16384)}
    assert bionic.load_requests == [{"model": QWEN, "context_length": 4096},
                                   {"model": BONSAI, "context_length": 16384, "parallel": 1}]
    assert set(bionic.events[:4]) == {("unload", name) for name in ("qwen-old", "kev-duplicate", "bonsai-old", "unused")}
    bionic.events.clear()
    for role in ("decision", "embedding", "forecast"):
        check(role)
    assert bionic.events == [("/v1/systemone", "kev-serving"), ("/v1/embeddings", QWEN + ":1"),
                             ("/v1/chat/completions", BONSAI + ":1")]
    # Switching back to the default frees every instance except the requested role.
    monkeypatch.setenv("PLANNER_KEEP_MODELS_LOADED", "0")
    check("decision")
    assert bionic.loaded == {"kev-serving": (KEV, 4096)}


def test_resident_check_fails_if_bionic_evicts_an_earlier_model(bionic, monkeypatch, capsys):
    monkeypatch.setenv("PLANNER_KEEP_MODELS_LOADED", "1")
    monkeypatch.setattr(sys, "argv", ["check_local_models"])
    bionic.max_loaded = 2
    bionic.auto_evict = True
    assert check_local_models.main() == 1
    output = capsys.readouterr()
    assert "validated forecast" in output.out
    assert "Loaded together:" not in output.out
    assert QWEN + " is no longer loaded" in output.err
