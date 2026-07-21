# Branch profile: main

`main` is the canonical AXIS baseline reproduction. It contains no redesign factor and is the control for `loss_resesign` and `architecture_redesign`.

## Model and objective

- Model: `src/models/AXIS/AXIS.py`.
- Phase-I time-series encoder: loaded from the fixed external checkpoint and frozen in Phase II.
- Backbone LLM: frozen.
- Trainable Phase-II component: the original AXIS Hint Tuner/Perceiver path.
- Phase-II objective: the baseline answer-generation objective implemented by `tools.axis_repro.train_phase2_memory_safe_v3`.

## Canonical entry points

```bash
python -m torch.distributed.run --nproc-per-node=3 \
  -m tools.axis_repro.train_phase2_memory_safe_v3 --help

python -m torch.distributed.run --nproc-per-node=3 \
  -m tools.axis_repro.run_inference_cli --help
```

The exact training and evaluation commands are in `REPRODUCTION.md`. Redesign results are valid only when compared with a `main` checkpoint trained under the same manifest, seed policy, global batch, maximum epoch, validation schedule and judge protocol.