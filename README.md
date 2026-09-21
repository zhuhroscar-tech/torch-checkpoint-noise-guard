# torch-checkpoint-noise-guard

This repository has moved into the consolidated PyTorch correctness guard suite:

**https://github.com/zhuhroscar-tech/torch-correctness-guards**

Use the umbrella package instead:

```bash
python -m pip install -e ".[torch]"
torch-guard run checkpoint-noise
```

Python API:

```python
from torch_correctness_guards import safe_rrelu

z = safe_rrelu(x, lower=0.125, upper=1 / 3, training=True)
```

The original functionality is preserved there as the `checkpoint-noise` guard for [pytorch/pytorch#193671](https://github.com/pytorch/pytorch/issues/193671). This source repository is archived to avoid splitting future fixes across duplicate packages.
