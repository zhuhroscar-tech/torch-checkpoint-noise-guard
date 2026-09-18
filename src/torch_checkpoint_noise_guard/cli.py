"""Command-line interface: run the from-scratch diagnosis of the
non-reentrant-checkpoint / saved-tensors-hook silent gradient corruption
bug in F.rrelu (pytorch/pytorch#193671) against the currently installed
torch build, using the shared semantic-color design system.
"""
from __future__ import annotations

import argparse
import json
import sys

from .style import print_fields, resolve_style, section, status_headline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-checkpoint-noise-guard",
        description=(
            "Diagnose a real torch correctness bug against the currently "
            "installed torch build: F.rrelu(..., training=True) produces "
            "silently wrong gradients under non-reentrant activation "
            "checkpointing (torch.utils.checkpoint.checkpoint(..., "
            "use_reentrant=False)) or any saved_tensors_hooks pack "
            "callback, because its noise buffer is captured by autograd "
            "before the kernel that fills it runs (pytorch/pytorch#193671). "
            "The forward pass is bit-exact in every case, so nothing looks "
            "wrong until gradients are compared. Verifies that the "
            "safe_rrelu() drop-in replacement produces identical gradients "
            "in every mode. Never trusts a cached or previously-reported "
            "result, always re-runs the repro on THIS host's actual "
            "installed torch version."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color even on a TTY")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-checkpoint-noise-guard {__version__}")
        return 0

    from .core import TorchUnavailableError, diagnose

    try:
        report = diagnose()
    except TorchUnavailableError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            style = resolve_style(no_color_flag=args.no_color)
            print(status_headline(style, "fail", f"torch unavailable: {exc}"))
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["guard_fully_correct"] else 1

    style = resolve_style(no_color_flag=args.no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["bug_reproduced"]:
        print(status_headline(style, "warn", "F.rrelu checkpoint/saved-hooks gradient corruption reproduced on this host's installed torch build (#193671)"))
    else:
        print(status_headline(style, "info", "F.rrelu checkpoint/saved-hooks bug did NOT reproduce on this host's installed torch build (fixed upstream, or noise buffer size/timing happened not to trigger it this run)"))

    if report["bug_reproduced"] and not report["mechanism_confirmed"]:
        print(status_headline(style, "warn", "bug reproduced but the expected root-cause signature (bit-exact forward, zero diff with early-stop disabled) did not fully match -- treat with caution"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_rrelu() produces identical, NaN-free gradients across eager, checkpoint, and saved-tensors-hooks execution"))
    else:
        print(status_headline(style, "fail", "safe_rrelu() did NOT match expected behavior in at least one mode"))

    section("non-reentrant checkpoint")
    c = report["checkpoint_case"]
    print_fields(
        [
            ("buggy F.rrelu gradient diff vs uncheckpointed", f"{c['buggy_diff']!s}"),
            ("buggy forward-output diff (expected 0.0)", f"{c['buggy_forward_diff']!s}"),
            ("buggy diff with early-stop disabled (expected 0.0)", f"{c['no_early_stop_diff']!s}"),
            ("guard (safe_rrelu) gradient diff", f"{c['guard_diff']!s}"),
            ("guard has NaN", f"{c['guard_has_nan']!s}"),
        ]
    )

    section("saved_tensors_hooks (clone-based pack/unpack)")
    c = report["saved_hooks_case"]
    print_fields(
        [
            ("buggy F.rrelu gradient diff vs unhooked", f"{c['buggy_diff']!s}"),
            ("guard (safe_rrelu) gradient diff", f"{c['guard_diff']!s}"),
            ("guard has NaN", f"{c['guard_has_nan']!s}"),
        ]
    )

    return 0 if report["guard_fully_correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
