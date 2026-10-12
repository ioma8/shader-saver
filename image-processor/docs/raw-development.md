# RAW development

`Develop RAW` is one universal pipeline for every supported RAW. It has no
camera-specific branches and no per-camera look. Its one learned component is a
small metering rule whose coefficients are fitted offline and baked in, so at
runtime it needs nothing but the frame itself. The implementation is in
`src/raw_develop.rs`; the tests and offline probes are in `src/raw_develop/`.

Camera JPEGs and embedded previews are never read to develop a RAW at runtime.
Embedded previews are used only for browse grid thumbnails
(`imgload::load_preview_rgba`, 220 px), and RAW/JPEG pairs are read only offline —
to fit the metering coefficients and to run the benchmarks.

## Runtime pipeline

1. **Decode.** rawler's `RawDevelop::develop_intermediate` applies white balance,
   demosaic, camera matrix, and crop, and returns gamma-encoded sRGB at 16 bit.
   EXIF orientation is applied and the result is downscaled to `max_dim` (2048 px
   on open, the current view size for Develop RAW). `rawler 0.8` is required:
   it detects `SensorType::Xtrans` and demosaics X-Trans sensors (Fuji) with the
   right algorithm. `rawler 0.7.2` sent X-Trans through the Bayer path, which gave
   Fuji files a strong green cast.
2. **Levels.** The luminance percentiles at `black = 0.005` and `white = 0.95`
   set the black and white points. The same affine map is applied to all three
   channels, so neutrals stay neutral.
3. **Tone.** The brightest channel `m` of each pixel is the tone input. A soft
   shoulder above `knee = 0.1` rolls highlights off toward 1 without reaching it.
   An S-curve adds midtone contrast (`contrast = 0.3`), then a gamma
   (`gamma = 0.68`) sets the brightness. The result is a per-pixel scale
   `tone(m) / m`, so hue is kept and the brightest channel never exceeds
   `mapped < 1`, so nothing overflows.
4. **Local tone.** `m` is box-blurred to a local base of radius `max(w, h) / 6`
   (`local_radius`). Each pixel's scale becomes a blend of the global
   `tone(m) / m` and the base-driven `tone(base) / base`, weighted by
   `local = 0.3`. The tone then follows the local average while local detail is
   kept, which holds shadow and highlight detail in high-dynamic-range scenes.
   The base is low-frequency, so it does not halo.
5. **Metering.** `meter_u16` predicts an exposure gain, a contrast offset and a
   midtone gamma from the frame's own statistics: nine from the *already toned*
   frame (2/10/25/50/75/90/98 luma percentiles, spread, skew) and five from the
   *raw decode* before the tone (5/50/95 luma percentiles, spread, mean, at
   absolute level). The raw five carry the scene's absolute brightness, which the
   levels normalize away — without them high-key frames come out too dark. The
   15+15+15 coefficients are fitted offline from RAW/JPEG pairs and baked in
   (`METER_GAIN`, `METER_OFFSET`, `METER_GAMMA`). The mapping ends with the same
   soft shoulder the tone uses (`METER_KNEE`), so a brightening that would clip
   the sky rolls off instead. All three are clamped. At runtime no reference
   image, dataset or model file is needed — only the frame itself.
6. **Chroma denoise.** `chroma_denoise_u16` blurs the two colour differences
   (`b - y`, `r - y`) with a small box blur and blends them back with a weight of
   `CHROMA_DENOISE * (1 - y)²`, so only the shadows, where the lift multiplied
   the sensor's chroma noise, are smoothed. Luma is untouched.
7. **Encode.** The result is 16-bit sRGB and becomes the editor input. The 8-bit
   output (`develop_raw`) is the top 8 bits of the same result.

The tone is in `tone_u16` (one function, six numbers in `ToneParams`). The
metering is `meter_u16` (a dot product of two 10-element arrays). It is the whole
look, so it is easy to read and easy to change.

Scaling by luminance instead, which is the obvious choice, pushes saturated
channels (orange sun and sky, red roofs) above 1.0. Those then turn into flat
255. The brightest-channel scale is what removed the clipping.

### Tuning the six parameters

The parameters are chosen by a sweep (`tone_sweep` in `src/raw_develop/tune.rs`) that
scores every combination against camera JPEGs:

```bash
RAW_SOOC_FOLDER=/path/to/arw+sidecar-jpg \
RAW_PIXEL_FOLDER=/path/to/dng+phone-jpg \
RAW_DATASET_DIR=datasets/raw-samples \
  cargo test --release tone_sweep -- --nocapture
```

It evaluates at 160 px (tone is global, so ranking is unchanged and it is far
cheaper), precomputes per-image luminance so levels are O(1), replaces the
per-pixel `powf` with a LUT, and parallelises with rayon: ~2000 combinations over
94 images in about 25 s. The objective is the image-pooled MAE.

Two cautions. MAE is dominated by overall tone placement, so it cannot see local
tone or sharpness; and it is measured against each camera's own JPEG, so it
rewards matching that camera rather than universal quality. The flat-image MAE
floor (a constant image) is ~43/255 = 0.17, far above the ~0.05 reached here, so
the result is not a degenerate flat solution.

History, same references throughout:

| Step | Sony mean | Pixel mean | Mixed mean |
| --- | --- | --- | --- |
| Levels + S-curve + gamma, luminance-scaled | 0.0869 | 0.0964 | 0.1182 |
| Sidecar EXIF-orientation fix in the bench | 0.0664 | 0.0964 | 0.1182 |
| Parameter sweep | 0.0592 | 0.0870 | 0.1196 |
| Local-tone stage (`local = 0.5`, base `max/32`) | 0.0557 | 0.0850 | 0.1185 |
| Local base refined (`local = 0.3`, base `max/6`) | 0.0537 | 0.0844 | 0.1157 |
| Per-frame metering, toned features only | 0.0454 | 0.0807 | 0.1021 |
| Metering + raw-decode features, clip guard | 0.0440 | 0.0786 | 0.1008 |
| + per-frame gamma, soft shoulder, chroma denoise (current) | **0.0445** | **0.0777** | **0.0990** |

Current parameters: `black = 0.005`, `white = 0.97`, `contrast = 0.3`,
`gamma = 0.68`, `knee = 0.1`, `local = 0.3`, `radius_div = 6`.

A fitted **fixed** per-channel curve (a profile-like 3×1D LUT) was measured and
gives no gain (Sony 0.0519 → 0.0531, Pixel 0.0787 → 0.0775 at 384 px): the levels
already match the marginal distribution, so the residual is not a fixed global
colour/tone shape. See `profile_probe` and `HYPOTHESES.csv` H26.

The orientation fix was the single biggest correction: the bench loaded sidecar
JPEGs with `image::open`, which ignores EXIF orientation, while the develop
applies it, so every portrait frame was compared rotated.

The metering change was driven by eight reported frames (`DSC03759`, `DSC03764`,
`DSC03773`, `DSC03781`, `DSC03791`, `DSC03792`, `DSC03795`, `DSC03799`) that came
out far too dark. Cause: those frames are high-key and the levels normalize their
absolute brightness away, while the meter only saw the toned frame. Adding raw
features and fitting on all four folders fixed them (e.g. `DSC03759` 0.155 → 0.071,
`DSC03799` 0.173 → 0.051). See `HYPOTHESES.csv` H29/H30.

Follow-ups from further reported frames: the metering's hard clip blew bright
skies (`DSC03800` 13.4% clipped against the reference's 7.7%) while its midtones
stayed dark, so the mapping got the tone's soft shoulder (H31); and its shadow
lift turned sensor chroma noise into visible blue/red speckle on `DSC03791`, so a
dark-weighted chroma denoise was added (H32).

The mixed set cannot be satisfied by one global curve: its references are
different camera previews with their own picture styles, and improving Sony and
Pixel shifts a few of them (Pentax K-r, Samsung NX300) the other way.

### Refitting the metering rule

```bash
RAW_SOOC_FOLDER=/path/to/arw+sidecar-jpg \
RAW_PIXEL_FOLDER=/path/to/dng+phone-jpg \
RAW_DATASET_DIR=datasets/raw-samples \
  cargo test --release metering_probe -- --nocapture
```

It prints per-group leave-one-out MAE, an out-of-sample transfer number, and the
fitted `METER_GAIN` / `METER_OFFSET` arrays (fit on every available pair). Paste
those two arrays into `src/raw_develop.rs` and rebuild. Keep the feature order in
`meter_features` in sync with the fitter.

Measured effect at 512 px: Sony 0.0537 → 0.0454, Pixel 0.0844 → 0.0807, mixed
0.1157 → 0.1021. Leave-one-out on the Sony luma plane was 12.43 → 9.16 of 255,
and a rule fit only on Sony+Pixel still cut the 13 unseen cameras by 11%.

## Cross-manufacturer check

`datasets/raw-samples/` holds 18 RAW files from 18 manufacturers (Apple, Canon,
DJI, Epson, Fujifilm, Google, Hasselblad, Kodak, Leica, Minolta, Nikon, Olympus,
OM System, Panasonic, Pentax, Phase One, Samsung, Sony), all CC0 from
raw.pixls.us. See its [README](../datasets/raw-samples/README.md) for source,
the file list and a re-download script.

Two checks, both env-gated:

```bash
# 1. Develop sanity: every file, no clipping.
RAW_DATASET_DIR=datasets/raw-samples cargo test --release \
  dataset_samples_develop_and_are_sane -- --nocapture

# 2. MAE bench: only files with a usable reference (sidecar JPEG, else a
#    decent embedded JPEG; tiny thumbnails and black previews are skipped).
RAW_DATASET_DIR=datasets/raw-samples \
RAW_SOOC_FOLDER=/path/to/arw+sidecar-jpg \
RAW_PIXEL_FOLDER=/path/to/dng+phone-jpg \
  cargo test --release reference_bench_runs -- --nocapture
```

All 18 develop with no meaningful clipping. Reference quality gates the MAE
bench, because some containers expose only a thumbnail or a black preview:

- **Skipped, preview extracts black:** Canon CR3 (all extracted previews, largest
  and smallest, are all-zero pixels — a `jpgfromraw` limitation).
- **Skipped, preview smaller than 512 px:** Fujifilm X-T3 (160×120), Panasonic
  GX800 (160×120), DJI Osmo Action (192×144).
- **Skipped, no embedded JPEG:** Leica M10.

MAE against each camera's own render (lower is closer):

| Group | n | Reference | mean MAE | median MAE |
| --- | --- | --- | --- | --- |
| Sony ARW (4 folders) | 74 | sibling SOOC JPG (6000×4000) | **0.0445** | 0.0438 |
| Pixel DNG | 45 | phone JPEG preview (1280×964) | **0.0777** | 0.0724 |
| Mixed samples | 13 | camera preview (640–8640 px) | 0.0990 | 0.0859 |

The mixed-sample MAE is the least trustworthy: its references are camera
previews as small as 640×480, and each carries its own picture style. The Sony
sidecars are the only true SOOC references here.

The set caught three things:

- **Fuji X-T3 green cast.** `rawler 0.7.2` demosaiced the 6×6 X-Trans CFA with
  the Bayer PPG algorithm. Bumping to `rawler 0.8.0`, which detects
  `SensorType::Xtrans` and uses `XTransBilinearDemosaic`, removed the cast
  (MAE 0.221 → 0.190). Sony, Pixel and Pentax numbers were unchanged.
- **Missing extensions.** SRW (Samsung) and IIQ (Phase One) decoded fine in
  `rawler` but were absent from `RAW_EXTS` in `imgload.rs`, so the app ignored
  those files entirely. `crw`, `dcs`, `tfr`, `iiq` were added.
- **Unsupported inputs.** Sigma X3F (`Couldn't find tag TiffCommonTag::Make`) and
  GoPro GPR (`TODO: Unsupported DNG compression`) fail inside `rawler` and are
  not in the set.

Two files still differ from their own render: the Kodak DCS760C develop has a
yellow-green cast (an old 2001 DCR, likely with no usable as-shot white
balance), and the reference for the Canon CR3 extracts black, so its MAE is
skipped.

The reference for this check is each camera's embedded JPEG, not a true SOOC
file, so it carries the camera's picture style and a nonzero MAE is expected.

## Why the learned develop was replaced

Until this change the develop applied a learned model to every RAW: 341 Pixel
DNG/JPEG pairs embedded as prototypes, with per-channel quantile curves and a
spatial correction copied from the nearest training image. Investigation
(`HYPOTHESES.csv` in this directory):

- **Good only on its own training set.** On Pixel pairs it reproduced the phone
  JPEG closely (MAE 0.02–0.04). Those are the training pairs, so the number is
  in-sample.
- **Broke on unseen cameras.** A Pentax DNG with no paired JPEG (nearest-prototype
  distance 1.34) came out with blue/yellow speckle and a cast. The median blue
  channel was lifted 16.7× while red was lifted 2×. Per-channel curves borrowed
  from unrelated neighbours amplify the noisiest channel and add a color cast.
- **Training was Pixel-only.** The pair finder only matches Pixel naming, so
  Sony and other cameras were never in the training set, and their distances to
  the prototypes (0.27–0.50) were all above the residual cutoff.
- **The copied spatial residual was not the cause.** Removing it made no
  measurable difference on Pixel pairs, so the failure was the global mapping.

The pipeline that replaced it has no per-camera model. On the same unseen Pentax
file it gives a clean, natural sunset with no speckle. Its only learned part is
the metering rule above, and that is validated to help on cameras it was not
fitted on. On Pixel frames it scores 0.0807 mean MAE; the old model's 0.02–0.04
was in-sample on those same pairs, so the numbers are not comparable — the new
pipeline does not try to copy one phone's look.

## Known limits

- Scaling by the brightest channel makes very saturated scenes somewhat darker
  than a luminance-based tone would. The Pentax sunset is richer, but its median
  is lower.
- The local tone base is a box blur at `max(w, h) / 6`, so it cannot halo but it
  also cannot fix fine local exposure differences. Very bright frames clip up to
  1.6% of pixels (white = 0.97); camera JPEGs in this set clip up to 15%.
- Error decomposition over the 36 Sony frames (`decompose_errors`): 74% of the
  luma MAE is low-frequency (tone/colour), 26% high-frequency (detail). Channel
  bias is tiny (+1.0 / +1.6 / +0.2).
- The metering coefficients are fitted on Sony + Pixel + the 13 mixed cameras, so
  they lean towards those. The mixed references are weak (camera previews), and a
  few frames regress even as the group improves. It corrects luma only, so chroma
  error is untouched.
- The raw-decode features make the meter accurate for a body it was fitted on
  (Sony LOO 7.2 of 255) but the absolute-level relationship is body-specific:
  fitting on Sony+Pixel and testing on the 13 unseen mixed cameras got worse
  (25.4 → 29.7), even though leave-one-out over all cameras improved. Treat the
  raw features as a per-body calibration learned from statistics, not a universal
  law. The clamp and clip guard limit the damage on a body unlike the training set.
- There is no luma denoising; only chroma is smoothed, and only in the shadows.
- Levels are per image, so the black and white points follow the scene. Very
  high-key or low-key frames are normalized, not kept as they are.
- SOOC JPEGs are somewhat more saturated in greens than this develop. Saturation
  is not tuned.
- Look differences (for example a phone's HDR tone mapping) are not reproduced.
  The editor's own sliders are the place for those adjustments.

## Legacy learned model (not used at runtime)

`models/raw_render_model.json` (~53 MB) and the `--train-raw-render-model`
trainer in `src/raw_fit.rs` are kept for reference. The runtime no longer loads
the model: `RawRenderModel::embedded()` is compiled only for tests.

```bash
cargo run --release -- --train-raw-render-model <raw-folder> models/raw_render_model.json
```

`--fit-raw-scurve <pair-folder> [models/raw_s_curve.json]` writes a global
phone-tone curve (~11 MB). It is also unused at runtime.

## Tests and files

Production code is `src/raw_develop.rs` (the pipeline: `tone_u16`, `meter_u16`,
`develop_raw*`). Tests and offline probes are `#[cfg(test)]` submodules in
`src/raw_develop/`:

| File | Provides |
| --- | --- |
| `tests.rs` | unit tests: neutral grey stays neutral, S-curve monotone, metering bounded, 8-bit/16-bit agree |
| `dataset_tests.rs` | `dataset_samples_develop_and_are_sane` — every sample develops, no clipping |
| `reference_bench.rs` | `reference_bench_runs` — MAE against camera JPEGs |
| `tune.rs` | `tone_sweep` — parameter sweep |
| `decompose.rs` | `decompose_errors` — where the error is (tone vs detail, per zone) |
| `profile_probe.rs` | tests the fixed per-channel LUT option (measured: no gain) |
| `metering_probe.rs` | `metering_probe` — fits and prints the metering coefficients |
| `frames_probe.rs` | `frames_probe` — checks specific frames by path against their sidecars (`RAW_FILES=a.ARW,b.ARW`) |

Every probe is env-gated and skips when its variable is unset, so a plain
`cargo test` runs only the unit tests. The commands:

```bash
cargo test --release raw_develop                     # unit tests
RAW_DATASET_DIR=datasets/raw-samples   cargo test --release dataset_samples_develop_and_are_sane -- --nocapture
RAW_SOOC_FOLDER=... RAW_PIXEL_FOLDER=...   cargo test --release reference_bench_runs -- --nocapture
RAW_SOOC_FOLDER=... RAW_PIXEL_FOLDER=... RAW_DATASET_DIR=datasets/raw-samples \
  cargo test --release tone_sweep -- --nocapture
RAW_SOOC_FOLDER=... cargo test --release decompose_errors -- --nocapture
RAW_SOOC_FOLDER=... RAW_PIXEL_FOLDER=... RAW_DATASET_DIR=datasets/raw-samples \
  cargo test --release metering_probe -- --nocapture
```

`raw_fit::tests::developed_raw_moves_toward_its_phone_jpeg` (env-gated on
`RAW_VERIFY_FOLDER`) measures distance to the phone JPEG, the old target, so it no
longer describes the goal of this pipeline.
