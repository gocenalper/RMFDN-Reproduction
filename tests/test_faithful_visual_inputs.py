"""Regression tests for the paper-faithful visual input path."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def _script_help(script: str) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT), env.get("PYTHONPATH"))))
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_training_cli_historical_diagnostic_flags_stay_removed():
    """Removed one-off diagnostic flags must not resurface in the training CLI.

    Guards the paper-faithful surface: historical crop/augment/encoder-swap
    and sampler experiment flags stay deleted.
    """
    help_text = _script_help("scripts/run_experiment_suite.py")

    assert "--visual-augment" not in help_text
    assert "--anti-shortcut-aug" not in help_text
    assert "--visual-crop" not in help_text
    assert "--visual-downup" not in help_text
    assert "--no-visual-temporal-transformer" not in help_text
    assert "--no-imagenet-norm" not in help_text
    assert "--no-full-taxonomy" not in help_text
    assert "--audio-encoder" not in help_text
    assert "--no-fusion-ffn" not in help_text
    assert "--balanced-sampler-pristine" not in help_text
    assert "--uniform-sampler-pristine" not in help_text
    assert "--bounded-sampler-pristine" not in help_text
    assert "--text-encoder" not in help_text
    assert "--oracle-poe" not in help_text


def test_evaluation_cli_has_no_historical_diagnostic_controls():
    help_text = _script_help("scripts/run_test_eval.py")

    assert "--degrade-test" not in help_text
    assert "--degrade-audio-test" not in help_text
    assert "--visual-crop" not in help_text
    assert "--visual-downup" not in help_text
    assert "--no-temporal-transformer" not in help_text
    assert "--no-imagenet-norm" not in help_text
    assert "--taxonomy8" not in help_text
    assert "--no-reference" not in help_text
    assert "--split" not in help_text
    assert "--audio-encoder" not in help_text
