# Look-transfer proposals

> Historical design research. The application now uses one constrained
> profile-conditioned look model for both single-image and folder application;
> CanonCGT and the deterministic semantic transfer are no longer runtime paths.
> See [`src/look_model.rs`](../../src/look_model.rs) and
> [`look_chain_for`](../../src/main.rs).

## Scope

The current feature transfers a photographic look between two images: contrast, white balance, tonal response, saturation, and perceived exposure, while preserving the target photo's subject and composition. The target application time is at most two seconds per photo.

One important limitation must be explicit: a global color transform cannot literally transfer hard versus soft illumination, light direction, or cast shadows. Those are spatial relighting problems. The practical goal is to transfer the grade and perceived tonal mood while preserving the target's own lighting structure.

## Current implementation findings

The implementation already has three relevant pieces:

- `LookProfile` measures Oklab tone percentiles, neutral color cast, chroma, hue sectors, and face/region statistics in [`src/processor.rs`](../../src/processor.rs).
- The deterministic fallback derives several conservative Oklab transfer passes and bakes them into a 33³ LUT.
- CanonCGT is the primary reference-conditioned model. It predicts canonicalize/restylize LUTs, composes them, upsamples to 33³, and applies the result in the existing WGSL shader.

There are two practical concerns:

1. A release test running one CanonCGT inference takes about 2.14 seconds even with tiny synthetic images. That includes model loading, so warm inference may be lower, but normal image decode, resizing, face detection, profile measurement, and LUT post-processing still consume time.
2. The comments describe pre-normalizing the target before CanonCGT inference, but [`look_chain_for`](../../src/main.rs:201) passes the original target to `predict_lut_prenormalized`; [`predict_lut_prenormalized`](../../src/canoncgt.rs:132) predicts on that original image and blends the result with the deterministic LUT afterward. This is a design discrepancy worth resolving and benchmarking rather than assuming pre-normalization is helping the network.

The current reported quality result—mean style similarity 0.8648 and minimum 0.5218 over 72 ordered pairs—is encouraging, but it does not establish universal reliability across cameras, white balances, exposure gaps, indoor/outdoor scenes, or hard/soft lighting.

## Proposal 1: reference-conditioned bilateral-grid/LUT student

### Idea

Train a compact model that receives a 256px target thumbnail and the captured-reference thumbnail and predicts two transforms:

- a global 33³ LUT for white balance, contrast, tone curve, saturation, and color mood;
- a small edge-aware bilateral-grid residual for local exposure and highlight/shadow differences.

The existing WGSL pipeline can apply both transforms at full resolution. The model only operates on thumbnails; the final image never needs to pass through a neural image generator.

### Training

Use CanonCGT as an initial teacher, then fine-tune with curated human-approved pairs and synthetic perturbations covering:

- white-balance errors and tint shifts;
- exposure changes from roughly -3 to +3 EV;
- contrast, black-point, highlight roll-off, and saturation changes;
- indoor, outdoor, mixed-light, low-light, skin, foliage, sky, and neutral scenes;
- deliberately different lighting softness and direction, with a loss that preserves target edges and geometry.

Use losses for rendered-image similarity, monotone LUT behavior, neutral-pixel preservation, skin-hue stability, gamut safety, and smoothness. Reject or attenuate predictions when confidence is low.

### Why it is the strongest option

This combines the global consistency of a 3D LUT with limited spatial adaptation. It can handle cases where one global transform is insufficient, without hallucinating or rewriting the whole image. Bilateral-grid models are designed to run expensive inference on a small image and transfer the result edge-aware at full resolution; see the [HDRNet paper](https://people.csail.mit.edu/hasinoff/pubs/GharbiEtAl17-hdrnet.pdf).

### Expected performance

With a small FP16 or INT8 model and an Apple GPU/Neural Engine backend, the target should be comfortably below two seconds, likely tens to a few hundred milliseconds after model warm-up. CPU-only performance must still be measured on the slowest supported machine.

### Main risk

This requires a training and validation pipeline. Distilling CanonCGT alone would preserve its failure modes, so hard-condition data and human review are essential.

## Proposal 2: accelerated and confidence-gated CanonCGT

### Idea

Keep CanonCGT as the quality baseline, but change runtime and decision-making:

1. Run it through ONNX Runtime with the Core ML execution provider on macOS instead of CPU-only `tract`.
2. Validate FP16 and, only if quality remains stable, lower-precision weights.
3. Generate a direct candidate and a genuinely normalized candidate.
4. Select or blend candidates using confidence signals:
   - neutral-pixel evidence;
   - target/reference illumination and exposure gap;
   - LUT smoothness and monotonicity;
   - highlight/shadow clipping;
   - face and skin-hue displacement;
   - disagreement with the deterministic Oklab candidate.
5. Fall back to the deterministic transform when confidence is low.

The Core ML execution provider can use CPU, GPU, and Neural Engine on compatible Apple devices; see the [ONNX Runtime CoreML documentation](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html). Apple also documents FP16 and lower-precision model compression in [Reducing the Size of Your Core ML App](https://developer.apple.com/documentation/coreml/reducing-the-size-of-your-core-ml-app).

### Advantages

- Smallest product change.
- Preserves the current model's strongest visual behavior.
- A confidence gate prevents obviously bad model outputs from becoming the final grade.
- Hardware acceleration gives the clearest route to the two-second requirement without retraining first.

### Main risks

- Core ML support and latency vary by Mac hardware.
- A model trained for general reference grading may still under-correct extreme capture-condition changes.
- The direct-versus-normalized behavior must be measured; the current comments document rejected experiments, but the actual integration needs a clean, reproducible comparison.

## Proposal 3: semantic deterministic color-transfer engine

### Idea

Build a fully deterministic, fast fallback or standalone mode:

1. Estimate target/reference white balance from reliable neutral pixels.
2. Canonicalize exposure and luminance into a scene-relative representation.
3. Match luminance with robust monotone percentile curves rather than absolute histogram values.
4. Match chroma and hue in Oklab using regularized transport or smooth hue-sector mappings.
5. Use separate guarded transforms for skin, sky, vegetation, and neutral pixels.
6. Fit one smooth 33³ LUT and optionally a very small local exposure map.

All statistics can be computed on a 256–768px working image; the resulting LUT is applied by the existing GPU shader.

### Advantages

- Very fast, predictable, and easy to debug.
- No model loading, platform-specific inference runtime, or large memory footprint.
- Excellent safety net for unusual images and machines without neural acceleration.

### Main risk

It will not reproduce subtle cinematic relationships as faithfully as a trained reference-conditioned network. It should therefore be the mandatory fallback and confidence-gated alternative, even if it is not the final primary method.

## Recommendation

The best overall architecture is a hybrid:

1. Implement and benchmark Proposal 2 first to establish an accelerated baseline and fix the normalization-path discrepancy.
2. Keep Proposal 3 as the safety fallback and confidence gate.
3. Train Proposal 1 as the long-term primary model, using CanonCGT and the deterministic pipeline as teachers but validating against human-approved difficult pairs.

This gives an immediate speed improvement, preserves a safe deterministic path, and creates a route to better quality on mixed lighting without abandoning the existing LUT/WGSL architecture.

## Required acceptance benchmark

Before choosing a final model, create a fixed benchmark of at least 300–500 ordered target/reference pairs covering:

- multiple cameras and lenses;
- JPEG and RAW-derived images;
- indoor, outdoor, mixed, soft, and hard light;
- large exposure and white-balance differences;
- people and skin, foliage, sky, architecture, and neutral scenes;
- low light, clipped highlights, and high saturation.

Track:

- p50 and p95 application time per photo;
- mean and worst-case style similarity;
- skin hue and neutral-gray drift;
- highlight/shadow clipping;
- LUT smoothness and out-of-gamut cells;
- user-rated acceptability of the worst 10% of results.

The hard release condition should be p95 under two seconds, not just an average. A global transform cannot guarantee physical relighting, so cases requiring changed shadows or light direction should be explicitly marked as outside the color-grading contract.
