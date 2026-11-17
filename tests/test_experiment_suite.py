#!/usr/bin/env python3
"""Pure-logic tests for the configurable experiment suite (torch-free)."""

import json
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rmfdn_ablation.experiment import (ExperimentSpec, expand_cells,
                                       resolve_reference_flag)
from rmfdn_ablation.protocols import load_identity_assignment
import scripts.run_test_eval as run_test_eval


def test_spec_name():
    spec = ExperimentSpec(cross_modal="m1", identity="i1", protocol="p2", modality="VAT")
    assert spec.name == "m1_i1_p2_VAT"


def test_expand_cells_matches_declared_matrix():
    cells = json.load(open(Path(__file__).resolve().parents[1]
                           / "experiments" / "experiment_matrix.json"))["cells"]
    specs = expand_cells(cells)
    # 2 + 1 + 1 = 4 runs total
    assert len(specs) == 4
    names = {s.name for s in specs}
    assert "m1_i3_p2_VA" in names and "m1_i3_p1_VA" in names
    assert "m1_i1_p2_VAT" in names
    assert "m1_i3_p2_VAT" in names
    assert len({s.protocol for s in specs}) == 2  # both p1 and p2 present
    for spec in specs:
        resolve_reference_flag(spec)  # every declared cell is runnable


def test_reference_flag_mapping():
    assert resolve_reference_flag(ExperimentSpec("m1", "i1", "p1", "VAT")) is True
    assert resolve_reference_flag(ExperimentSpec("m1", "i3", "p1", "VAT")) is False


def test_unknown_axes_raise():
    for spec in [ExperimentSpec("m1", "i2", "p2", "VAT"),
                 ExperimentSpec("m2", "i1", "p2", "VAT")]:
        try:
            resolve_reference_flag(spec)
            assert False, f"expected ValueError for {spec.name}"
        except ValueError:
            pass


def test_identity_assignment_roundtrip():
    split = {"splits": {"train": ["id01", "id02"], "val": ["id03"], "test": ["id04"]}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(split, handle)
        path = handle.name
    assignment = load_identity_assignment(path)
    assert assignment == {"id01": "train", "id02": "train", "id03": "val", "id04": "test"}


def test_test_evaluator_requires_the_canonical_suite_output():
    assert hasattr(run_test_eval, "_result_directory")
    result_directory = run_test_eval._result_directory

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        (root / "results.json").write_text("{}")
        assert result_directory(root) == root

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        try:
            result_directory(root)
            assert False, "expected a missing-results error"
        except FileNotFoundError:
            pass


def test_test_evaluator_reconstructs_saved_training_config_with_runtime_overrides():
    assert hasattr(run_test_eval, "_evaluation_config")
    run = {"config": {
        "bert_model": "saved-bert",
        "ast_checkpoint": "saved-ast",
        "batch_size": 128,
        "num_workers": 6,
    }}
    args = types.SimpleNamespace(
        bert_model=None, ast_checkpoint="moved-ast", batch_size=16, num_workers=4,
    )

    config = run_test_eval._evaluation_config(run, args)

    assert config["bert_model"] == "saved-bert"
    assert config["ast_checkpoint"] == "moved-ast"
    assert config["batch_size"] == 16
    assert config["num_workers"] == 4


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all experiment-suite tests passed")
