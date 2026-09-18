"""Tests for the CLI entry point: argument parsing, --version, --json,
--no-color, and exit codes -- independent of whether torch is installed.
Mirrors the pattern established fleet-wide (torch-linalg-nan-guard,
torch-nested-ad-narrow-guard, etc.) after the fleet-wide branch-coverage
gap was found and fixed in those repos: mock core.diagnose so every CLI
message path is exercised deterministically, regardless of whether this
host's installed torch build happens to reproduce the underlying bug.
"""
from __future__ import annotations

import json
import runpy
import sys

import pytest

from torch_checkpoint_noise_guard import core
from torch_checkpoint_noise_guard.cli import main


def _fake_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/193671"],
        "checkpoint_case": {
            "mode": "checkpoint",
            "buggy_diff": 0.0,
            "buggy_forward_diff": 0.0,
            "guard_diff": 0.0,
            "guard_has_nan": False,
            "no_early_stop_diff": 0.0,
        },
        "saved_hooks_case": {
            "mode": "saved_hooks",
            "buggy_diff": 0.0,
            "buggy_forward_diff": 0.0,
            "guard_diff": 0.0,
            "guard_has_nan": False,
            "no_early_stop_diff": None,
        },
        "bug_reproduced": False,
        "mechanism_confirmed": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_version_flag(capsys):
    code = main(["--version"])
    out = capsys.readouterr().out
    assert code == 0
    assert "torch-checkpoint-noise-guard" in out


def test_json_output_is_valid_json_and_reports_guard_status(capsys, monkeypatch):
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report())
    code = main(["--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert "torch_version" in report
    assert "guard_fully_correct" in report
    assert code == 0


def test_json_exit_code_reflects_guard_failure(monkeypatch, capsys):
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(guard_fully_correct=False))
    code = main(["--json"])
    capsys.readouterr()
    assert code == 1


def test_text_output_no_color_has_no_ansi_escapes(monkeypatch, capsys):
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report())
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "\x1b[" not in out


def test_torch_unavailable_json_mode_reports_error_and_exit_2(monkeypatch, capsys):
    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload == {"error": "torch is required for diagnosis"}
    assert code == 2


def test_torch_unavailable_text_mode_reports_fail_headline_and_exit_2(monkeypatch, capsys):
    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "torch unavailable: torch is required for diagnosis" in out
    assert code == 2


def test_no_bug_present_prints_info_line(monkeypatch, capsys):
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(bug_reproduced=False))
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "did NOT reproduce" in out
    assert "corruption reproduced" not in out


def test_bug_present_prints_warn_line(monkeypatch, capsys):
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(bug_reproduced=True))
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "corruption reproduced" in out


def test_bug_reproduced_but_mechanism_not_confirmed_prints_extra_warn(monkeypatch, capsys):
    monkeypatch.setattr(
        core, "diagnose", lambda: _fake_report(bug_reproduced=True, mechanism_confirmed=False)
    )
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "did not fully match" in out


def test_guard_mismatch_prints_fail_line_and_exit_1(monkeypatch, capsys):
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(guard_fully_correct=False))
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "did NOT match expected behavior" in out
    assert code == 1


def test_module_entry_point_runs_main_and_exits_with_its_code(monkeypatch):
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(guard_fully_correct=False))
    monkeypatch.setattr(sys, "argv", ["torch-checkpoint-noise-guard", "--no-color"])
    monkeypatch.delitem(sys.modules, "torch_checkpoint_noise_guard.cli", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("torch_checkpoint_noise_guard.cli", run_name="__main__")
    assert exc_info.value.code == 1
