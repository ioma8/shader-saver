# FiveK → slider residual

Offline pipeline behind Phase 2 of `docs/auto-enhance.md`: train a small network
that predicts a correction on top of the analytic seed in `src/auto.rs`, so the
Auto button's visible sliders can carry a real retouching taste.

The runtime never uses a LUT and never sees the dataset — it applies the
predicted sliders through the normal GPU shader (see `src/auto/pipeline.rs` for
the CPU replica of that math).

## Steps

1. **Download** (research-licensed; only weights are ever committed):

   ```sh
   hf download HoangPhuc7678/FiveK --repo-type dataset --local-dir datasets/fivek
   ```

2. **Extract pairs** to plain JPEGs:

   ```sh
   .venv-train/bin/python tools/fivek/prepare.py
   ```

3. **Seed** — the Rust side picks the base sliders for every input:

   ```sh
   AUTO_SEED_DIR=datasets/fivek/pairs cargo test --release seed_probe
   ```

   This writes `datasets/fivek/pairs/seeds.csv`.

4. **Check the renderer** against the shader replica before training (a mismatch
   would bias every learned slider):

   ```sh
   cargo test --release golden_probe            # writes /tmp/auto_golden.txt
   .venv-train/bin/python tools/fivek/render.py --check /tmp/auto_golden.txt
   ```

5. **Train** and export ONNX:

   ```sh
   .venv-train/bin/python tools/fivek/train.py --epochs 10 --strip /tmp/strip.png
   ```

   Writes `models/auto_residual.onnx`, which `src/auto/model.rs` embeds. The
   slider order, `RESIDUAL_SCALE`, `SEED_NORM` and limits in `train.py` and in
   `model.rs` must stay in sync.

## Files

| file | role |
|---|---|
| `render.py` | differentiable torch replica of the shader's auto transform |
| `prepare.py` | parquet shards → `pairs/NNNNN_{in,gt}.jpg` |
| `train.py` | trains the residual CNN on MPS and exports ONNX |
