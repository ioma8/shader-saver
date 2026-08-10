# RAW development

`Develop RAW` is a deterministic, non-neural pipeline trained to reproduce the
Pixel phone JPEG look from matching DNG/JPEG pairs. The implementation is in
`src/raw_develop.rs`; its fitted parameters are embedded from
`models/raw_s_curve.json` at compile time.

## Runtime pipeline

1. **Decode and normalize the sensor data.** `rawloader` supplies the active
   crop, black/white levels, CFA, camera matrix, and orientation. Each sensor
   value has its black level removed, is normalized by its white level, and is
   multiplied by the as-shot white-balance gain. Pixel DNGs whose
   `AsShotNeutral` is missing or the placeholder `(1,1,1)` use rawloader's
   D65-neutral camera gains; accepting the placeholder caused a strong green
   cast.
2. **Build linear sRGB.** Bayer data is demosaiced with Malvar interpolation
   (or a fast color-aware 2×2 reduction for previews), transformed through the
   camera-to-sRGB matrix, and optionally box-downsampled. Highlights remain
   floating point until tone mapping.
3. **Select scene-adaptive targets.** The RAW is described by eight histogram
   features: median exposure, four relative tone percentiles, and red/green and
   blue/green median ratios. The four closest trained scenes predict exposure,
   channel medians, five tone targets, white balance, and saturation. An exact
   trained scene uses its own targets.
4. **Apply the phone tone and color model.** The RAW median anchors exposure;
   a fitted monotone S-curve maps normalized linear luminance while preserving
   hue. Per-channel curves, global local contrast/sharpening, channel placement,
   white balance, saturation, and the 5/25/50/75/95% tone targets then reproduce
   the phone's blacks, shadows, midtones, highlights, contrast, and chroma.
5. **Orient, then apply local HDR.** EXIF rotation/flip is applied before any
   spatial comparison. A close trained scene can use a bilinearly interpolated
   64×64×RGB residual grid in encoded sRGB. This captures the Pixel pipeline's
   position-dependent shadow lifting, highlight compression, and local color
   changes that global curves cannot reproduce.
6. **Return 16-bit editor input.** The result is gamma-encoded RGBA16 for the
   GPU editor (`RGBA8` is also available for previews/tests).

## Fitting

Run:

```bash
cargo run --release -- --fit-raw-scurve <pair-folder>
```

Pairs are discovered as `*.RAW-02.ORIGINAL.dng` plus a matching
`*.RAW-01.jpg`, `*.RAW-01.COVER.jpg`, or `*.RAW-01.MP.jpg`. The current model
uses 46 pairs rendered at 512 px. It fits robust median global curves first,
then per-scene targets and spatial residuals, and writes the embedded JSON
model (about 11 MB). Rebuild the application after refitting.

Spatial fitting is deliberately skipped when the already-toned RAW and JPEG
differ by more than 8% mean encoded-channel error. This usually means the
phone selected a different burst frame or crop. Applying a coordinate-based
map in that case creates halos and misplaced corrections; the safer global
tone/color model remains active.

## Findings and verification

- White balance must be fitted after applying EXIF orientation; histogram-only
  operations hide orientation mistakes, but spatial operations do not.
- Global percentile matching can report excellent contrast while still
  missing local HDR. The per-channel spatial residual grid was the key final
  improvement.
- For `PXL_20260806_150001457`, the appearance-profile error fell from `0.065`
  to `0.013`, and RGB mean absolute error from `2.32%` to `1.49%`.
- Across all 46 pairs, pooled appearance-profile distance fell from `5.664`
  undeveloped to `0.136` developed. Pixel identity is not a valid requirement
  for differently cropped or temporally different burst frames, so verification
  gates tone/color appearance for every pair and reports pixel MAE separately.

Verification command:

```bash
RAW_VERIFY_FOLDER=<pair-folder> cargo test --release \
  raw_develop::tests::developed_raw_moves_toward_its_phone_jpeg -- --nocapture
```

Use `RAW_VERIFY_PAIR=<filename-fragment>` to isolate and dump one pair under
`/tmp/dev-check`.
