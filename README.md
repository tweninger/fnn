# IFT

Interaction field and interaction dynamics experiments.

## Development

```bash
venv/bin/python -m pip install -e ".[dev]"
venv/bin/python -m pytest
venv/bin/python -m ruff check
venv/bin/python -m pylint interactiondynamics interactionfields
venv/bin/pyright
```

## Training presets

```bash
venv/bin/python -m interactiondynamics.train smoke
venv/bin/python -m interactiondynamics.train quick
venv/bin/python -m interactiondynamics.train quick --dataset physical --graph-kind grid --dynamics wave
venv/bin/python -m interactiondynamics.train quick --dataset physical --graph-kind grid --dynamics wave_pulse
```

- `smoke` runs a tiny toy dataset check over the focused model/update shortlist.
- `quick` runs the focused shortlist on JODIE Wikipedia:
  - `ift` + `ift_update`
  - `hopfield` + `hopfield_update`
  - `settransformer` + `lnn`
  - `settransformer` + `hnn`
  - `settransformer` + `tgn_gru`
- You can override examples like `venv/bin/python -m interactiondynamics.train quick --dataset toy --epochs 1`.
- Physical datasets are available via CLI:
  - `--dataset physical --graph-kind grid --dynamics wave`
  - `--dataset physical --graph-kind grid --dynamics wave_pulse`
  - `--dataset physical --graph-kind small_world --dynamics sirs`
  - `--dataset physical --graph-kind tree --dynamics sis`
- Physical graph defaults start from a `24 x 24` grid unless you override `--graph-m` / `--graph-n`.
- Ranking metrics now treat ties pessimistically by default, and eval also reports a `persistent_*` baseline that keeps the post-warmup state fixed.
