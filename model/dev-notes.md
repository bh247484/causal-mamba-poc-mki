## Tensorboard

- Need to have venv activated, if you `cd model/` it will automagically activate because of vscode workspace config.
- Then to launch tensorboard at `localhost:6006` run `tensorboard --logdir tb_logs` from within `model/`.

## Script Examples
- From model/ dir...
  - Simulate streaming: `.venv/bin/python -m utils.simulate_streaming`
  - Overfit training run: `.venv/bin/python -m training.train_overfit`
  - Export model weights as binary: `.venv/bin/python -m utils.export_ggml_bin`
  - These Use the python module convention and references the venv python binary directly.