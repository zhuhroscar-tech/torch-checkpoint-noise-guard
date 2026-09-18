# torch-checkpoint-noise-guard

Detect and guard against `torch.nn.functional.rrelu(..., training=True)` silently producing **wrong gradients** under non-reentrant activation checkpointing (`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`) or any `saved_tensors_hooks` pack callback that materializes a tensor at pack time (e.g. `torch.utils.checkpoint.save_on_cpu`).

## What it checks

The motivating bug is [pytorch/pytorch#193671](https://github.com/pytorch/pytorch/issues/193671) (open as of this writing): `F.rrelu` dispatches to `aten::rrelu_with_noise`, which writes its sampled negative slopes into a **mutable output argument** (`noise`) that autograd's generated wrapper saves *before* the kernel runs. Under non-reentrant checkpointing, recomputation can be aborted (`_StopRecomputationError`) before that kernel re-runs, so backward reads the uninitialized buffer as if it were the real noise. The forward pass stays bit-exact, so nothing looks wrong until gradients are compared — the layer can silently stop learning correctly with no error, warning, or crash.

This CLI:
- Reproduces the exact upstream repro against your installed torch build.
- Confirms the root-cause signature: bit-exact forward output, and the divergence disappearing when checkpoint's early-stop machinery is disabled (`set_checkpoint_early_stop(False)`).
- Verifies `safe_rrelu()`, a pure-Python drop-in replacement built from ordinary (non-mutable-output) ops, produces identical gradients across eager execution, checkpointing, and saved-tensors-hooks.
- Returns nonzero exit codes for use in CI.

## Install

Requires Python 3.9+ and PyTorch 2.0+ for the live diagnosis (`torch` optional extra); the CLI degrades gracefully with a clear error if torch is not installed.

```bash
git clone https://github.com/zhuhroscar-tech/torch-checkpoint-noise-guard.git
cd torch-checkpoint-noise-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,torch]"
```

A wheel is also available from [Releases](https://github.com/zhuhroscar-tech/torch-checkpoint-noise-guard/releases); verify it against that release's checksums before installing.

## Quick start

```bash
torch-checkpoint-noise-guard
torch-checkpoint-noise-guard --json
python -m pytest
```

Exit codes: `0` means `safe_rrelu()` matched expected behavior in every tested mode, `1` means it did not, `2` means torch is unavailable.

## Using the guard in your own code

```python
from torch_checkpoint_noise_guard import safe_rrelu

# instead of: torch.nn.functional.rrelu(x, lower, upper, training=True)
y = safe_rrelu(x, lower, upper, training=True)
```

`safe_rrelu` delegates to the real `F.rrelu` unchanged in eval mode (`training=False`), where there is no randomness or mutable-output kernel involved.

## Honest limitations

This guards specifically the `rrelu_with_noise` mutable-output-argument pattern described in #193671. It does not audit other stochastic ops for the same class of bug (dropout, other noise-injecting activations), and it does not patch PyTorch's autograd internals — the real fix belongs upstream. The bug reads uninitialized memory, so the exact magnitude of corruption is nondeterministic between runs; `diagnose()` checks for *any* nonzero divergence plus the documented root-cause signature, not a fixed expected value. If a future torch release fixes #193671 upstream, this tool will correctly report "did NOT reproduce" — the guard functions remain safe to use regardless, since they are semantically equivalent to a bug-free `F.rrelu` in eval mode and numerically consistent in training mode.

## License

[MIT](LICENSE).
