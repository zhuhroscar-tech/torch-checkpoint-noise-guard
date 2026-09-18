"""Version consistency: pyproject.toml, __init__.py, and --version flag
must all agree. Mirrors the fleet-wide pattern (a real bug class in this
fleet's history was a version drifting between these three sources)."""
from __future__ import annotations

import re
from pathlib import Path

from torch_checkpoint_noise_guard import __version__
from torch_checkpoint_noise_guard.cli import main


def test_init_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text()
    m = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    assert m is not None
    assert m.group(1) == __version__


def test_cli_version_flag_matches_init(capsys):
    main(["--version"])
    out = capsys.readouterr().out.strip()
    assert out == f"torch-checkpoint-noise-guard {__version__}"
