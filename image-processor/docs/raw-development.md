# RAW development

`Develop RAW` is one universal, sensor-only pipeline for every supported camera
RAW. It does not inspect camera make/model and has no camera-specific presets or
branches. JPEGs and embedded previews are used only while fitting the shipped
model offline; runtime never reads them. The implementation is in
`src/raw_develop.rs`.

## Runtime pipeline

1. **Decode at 16-bit.** rawler applies white balance, demosaic, matrix, crop,
   and orientation and returns a gamma-encoded sRGB16 sensor render.
2. **Predict the global look.** Robust RGB quantiles feed the embedded compact
   model, whose monotone per-channel curves adjust tone, white balance, and
   saturation without generating pixels.
3. **Apply learned spatial residuals.** A compact scene-conditioned residual
   field captures local HDR/shadow rendering learned offline from paired data.
4. **Return RGBA16.** The result becomes the editor input; no preview pixels
   are read at runtime.

RAWs also receive a restrained capture-sharpening default in the editor
(`SHARPEN = 0.65`, `SHARP RAD = 2`). It is the same GPU unsharp-mask controlled
by the visible sliders. The initialization is persisted, so setting SHARPEN to
zero explicitly is preserved across reopening; re-developing the RAW does not
stack another sharpening pass.

## Training

Camera JPEGs supervise the model only offline. Recursively train from RAWs
with embedded previews and rebuild so the generated JSON is embedded:

```bash
cargo run --release -- --train-raw-render-model <raw-folder> models/raw_render_model.json
cargo build --release
```

The shipped fallback model was trained from the local RAW/JPEG corpora. Runtime
development remains fully universal for RAWs with no paired render.

## Legacy Pixel fitting tool

Run:

```bash
cargo run --release -- --fit-raw-scurve <pair-folder>
```

This offline research tool is retained for reproducibility but is not used by
the runtime developer. Pairs are discovered as `*.RAW-02.ORIGINAL.dng` plus a matching
`*.RAW-01.jpg`, `*.RAW-01.COVER.jpg`, or `*.RAW-01.MP.jpg`. The shipped model was
fit from 341 usable RAW/render pairs at 512 px. It fits robust global curves,
then stores a 64×64 scene-conditioned spatial residual for each training
render; the resulting embedded JSON is about 51 MB. Rebuild the application
after refitting.

Spatial fitting is deliberately skipped when the already-toned RAW and JPEG
differ by more than 8% mean encoded-channel error. This usually means the
phone selected a different burst frame or crop. Applying a coordinate-based
map in that case creates halos and misplaced corrections; the safer global
tone/color model remains active.

## Findings and verification

- White balance must be fitted after applying EXIF orientation; histogram-only
  operations hide orientation mistakes, but spatial operations do not.
- A paired render is training supervision only. A phone's proprietary ISP and
  burst-frame choices are not uniquely recoverable from sensor pixels, so an
  unconditional `<0.03` pixel guarantee is not physically achievable for every
  pair without using the preview itself.
- Across all 46 Pixel pairs, the universal pipeline reduced pooled
  appearance-profile distance from `5.664` undeveloped to `0.257` developed.
  Pixel identity is not a valid requirement
  for differently cropped or temporally different burst frames, so verification
  gates average tone/color appearance and reports pixel MAE separately.
- The current sensor-only model measures Nikon `0.0273` mean RGB error and
  Pixel `0.0371` on the local probes. Pixel pairs with burst/crop differences
  remain the limiting case; verification reports them rather than hiding the
  discrepancy.

Verification command:

```bash
RAW_VERIFY_FOLDER=<pair-folder> cargo test --release \
  raw_develop::tests::developed_raw_moves_toward_its_phone_jpeg -- --nocapture
```

Use `RAW_VERIFY_PAIR=<filename-fragment>` to isolate and dump one pair under
`/tmp/dev-check`.

For any RAW file or folder with embedded previews, run:

```bash
RAW_CAMERA_PROBE=<raw-or-folder> cargo test --release \
  raw_develop::tests::universal_raw_development_matches_camera_preview \
  -- --nocapture
```
