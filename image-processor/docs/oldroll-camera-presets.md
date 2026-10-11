# OldRoll Camera Presets: Observed Contract and Processing Boundary

Inspected on 2026-10-11 from the editor at [oldroll.io/editor](https://www.oldroll.io/editor), its public [camera catalog](https://www.oldroll.io/resources/presets-v1/catalog.json) (version 1), and the editor JavaScript bundle loaded by that page.

## What the Camera tab exposes

The Camera gallery has the groups All, Popular, Film, Polaroid, CCD, Portrait, Scenery, Street, Food, Sunlight, Indoor, and Night. The supplied screenshot's Popular order is `2008`, `FXS`, `LOG`, `4008N`, `OPS`, `G7X2`, `GRX`, `CCD`. Their catalog IDs, exact default mode values, and style controls are:

| Card | ID | Default mode | Style control (label → mode) |
| --- | --- | ---: | --- |
| 2008 | `S2016` | 0 | none |
| FXS | `FXS` | 3 | FXS 1 → 3; FXS 2 → 0; FXS 3 → 1; FXS 4 → 2 |
| LOG | `LOG` | 6 | LUT 01 → 0; LUT 02 → 1; LUT 04 → 2; LUT 05 → 3; Original → 4; LUT 09 → 5; LUT 10 → 6 |
| 4008N | `N4008` | 0 | 4008N 1 → 0; 4008N 2 → 1; 4008N 3 → 2 |
| OPS | `OPS` | 1 | OPS 1 → 1; OPS 2 → 0 |
| G7X2 | `G7X2` | 0 | none |
| GRX | `GRX` | 1 | GRX 1 → 0; GRX 2 → 1 |
| CCD | `CCD` | 0 | none; catalog also exposes a date-stamp toggle and date |

The screenshot ordering differs from the live `popular` group array (`FXS`, `GRX`, `LOG`, `S2016`, `CCD`, `N4008`, `G7X2`, `OPS`). The table follows the screenshot. Camera choice is a single selected camera; the None card clears it. Style selection changes the camera `mode` parameter.

## Exact request recipe visible to the browser

The editor bundle constructs the processing recipe sent with preview jobs. For a selected camera it includes:

```text
cameraId       catalog camera ID
variant        camera parameter `variant`, default "A"
mode           camera parameter `mode`, default 0
dateStamp      included when dateEnabled and dateStamp are truthy
cameraOptions  resolved camera options plus non-special parameter overrides
seed           0
cameraSeed     camera instance seed
```

The same recipe object also carries `filterId`, filter `intensity`, `useUnifiedProgress`, and `filterSeed`. Preview submission posts a body containing `requestId`, `sourceId`, `kind: "preview"`, `revision`, and that recipe to `POST /api/jobs`; the browser polls the resulting job and downloads its rendered image. The catalog provides the defaults and mode labels above, camera names, category membership, and camera-card/style illustration paths.

For the chart captures, `filterId` was `none`, `useUnifiedProgress` was `false`, `seed` and `filterSeed` were `0`, and the editor's default filter intensity field was `0.6`. `cameraSeed` is assigned per camera instance and varies between sessions. All options stayed at their catalog defaults except the selected mode; CCD's date stamp was turned off for its LUT capture. OldRoll source records for the calibration chart were deleted after the captures completed.

`cameraOptions` is based on the camera's `resolved` values. The bundle copies those, then applies camera parameters except the special keys `variant`, `mode`, `dateStamp`, and `dateEnabled`; values equal to `"auto"` are omitted when no resolved value exists. For the date-stamp camera option, the local date placeholder is resolved in the browser. This describes the exact client-side recipe contract; it does not describe the image transform.

## Black-box calibration used for the local LUTs

The browser bundle does not implement the camera look as a shader, LUT, or sequence of local image operations. It sends the recipe and source image identifier to OldRoll's `/api/jobs` service and displays the returned image. To sample the backend behavior, I generated a synthetic 17×17×17 RGB chart (4,913 color patches) at 1050×1065 pixels, uploaded that chart through the editor, and submitted it for each of the 21 Popular camera/mode combinations. The preview kept the chart at its original dimensions. I used the returned JPEG and took the per-channel median of a 5×5 patch at the center of every tile; each result became one node in a 17³ RGB response table. The input chart is preserved as [oldroll_camera_calibration_chart.jpg](../assets/oldroll_camera_calibration_chart.jpg), and the 21 packed output tables are in `assets/oldroll_popular_camera_luts_17.rgb8`.

The initial 21 tables were 17³ and are resampled to the processor's 33³ working grid with trilinear interpolation. Because that does not add new observations, five looks that showed visible color plateaus at 100% were recaptured at 33³: FXS 1 (mode 3), GRX 2 (mode 1), 4008N 2 (mode 1), OPS 2 (mode 0), and G7X2 (mode 0). The dense chart uses the editor's full 1080×1080 preview dimensions, with 35,937 patches arranged across 190 columns; each patch is 5–6 pixels wide/high. Each output node is the median of the center 3×3 returned JPEG pixels. These five 33³ response tables are used directly in the processor, while all remaining looks retain the original 17³ tables and resampling path. The source chart is [oldroll_camera_calibration_chart_33.png](../assets/oldroll_camera_calibration_chart_33.png), outputs are preserved in [oldroll_dense_camera_captures](../assets/oldroll_dense_camera_captures), and the packed tables are in `assets/oldroll_popular_camera_luts_33.rgb8`.

This is still an empirical approximation: the service returns JPEG, color values between samples are interpolated, and a color-only 3D LUT cannot reproduce spatial or random effects. Small patches also make the dense captures more sensitive to JPEG chroma subsampling than the original 17³ chart, so the center median reduces but cannot eliminate capture noise. The chart reveals notable spatial texture in some outputs, especially `S2016` and `N4008`; one sampled camera seed cannot represent that texture for arbitrary photos. The CCD calibration explicitly disabled its default date stamp because text is not a color transform. `LOG` mode 4 (`Original`) returned the identity color mapping.

The visible client-side composition path runs after obtaining that base result: it applies manual adjustments, then an optional effect, then the frame, then stickers. Those operations use shaders/resources from `photo-render-v1/catalog.json`. This order is observable in the `drawPhoto` composition routine; it does not reveal the earlier camera transform performed by the job service.

## This repository's implementation

The editor reproduces the eight card names, camera selection, mode controls, and saved selection state. `src/camera_presets.rs` loads the original 21 sampled response tables plus direct 33³ overrides for the five selected modes above, and supplies the resulting grid to the existing 33³ photo LUT stage in `src/processor.rs`. The default local style indices now correspond to OldRoll's catalog defaults. The LUTs reproduce sampled color response; they omit spatial grain/optical effects and CCD date text.

The visible category names are represented locally, but only the eight Popular cameras are implemented; category filtering is a subset over those eight rather than the complete catalog. The other 53 catalog cameras are not included.

## Evidence and limits

- Observed: catalog version 1, selected camera defaults, mode labels/values, group memberships, client recipe construction, job endpoint, and result-image flow in the page assets. Backend outputs for the synthetic chart were captured for all 21 original modes; the five visibly plateauing modes were recaptured directly at 33³ using the editor preview flow.
- Not observed: OldRoll server implementation, deterministic behavior across camera seeds, spatial effects for arbitrary photos, or pixel-equivalence against OldRoll output.
- The catalog and bundle are live website resources and can change; the details above describe the resources inspected on the stated date.
