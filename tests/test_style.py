"""Tests for the shared style.py design-system module. Mirrors the
fleet-wide copy's own test suite (canonical copy: reboot-safety-check)."""
from __future__ import annotations

from torch_checkpoint_noise_guard.style import (
    Style,
    resolve_style,
    status_headline,
    print_fields,
    section,
)


def test_style_disabled_is_identity():
    s = Style(False)
    assert s.green("x") == "x"
    assert s.bold("x") == "x"
    assert s.red("x") == "x"


def test_style_enabled_wraps_ansi():
    s = Style(True)
    assert s.green("x") == "\033[32mx\033[0m"


def test_resolve_style_no_color_flag_wins(monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    style = resolve_style(no_color_flag=True)
    assert style.enabled is False


def test_resolve_style_no_color_env(monkeypatch):
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    style = resolve_style(no_color_flag=False)
    assert style.enabled is False


def test_resolve_style_force_color_env(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    style = resolve_style(no_color_flag=False)
    assert style.enabled is True


def test_status_headline_ascii_fallback():
    style = Style(False)
    line = status_headline(style, "ok", "all good")
    assert line == "[OK] all good"


def test_status_headline_unicode_when_enabled():
    style = Style(True)
    line = status_headline(style, "fail", "broken")
    assert "\u25cf" in line


def test_print_fields_empty_rows(capsys):
    print_fields([])
    out = capsys.readouterr().out
    assert out == ""


def test_print_fields_aligns_columns(capsys):
    print_fields([("a", "1"), ("longer", "2")])
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert len(lines) == 2


def test_section_prints_title(capsys):
    section("my section")
    out = capsys.readouterr().out
    assert "my section" in out
