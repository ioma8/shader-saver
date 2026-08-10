# Research: reliable reference-conditioned look transfer

> Historical research input, not a current-state architecture description.
> The implemented application uses CanonCGT canonicalize/restylize inference as
> its primary path and blends it with a constrained profile-conditioned fallback
> trained from synthetic grades and GUI-approved examples. See
> [`src/canoncgt.rs`](../../src/canoncgt.rs) and
> [`src/look_model.rs`](../../src/look_model.rs).

Implementation update (2026-08-10): the runtime now embeds CanonCGT's official
self-supervised checkpoint rather than the earlier end-to-end checkpoint. A
reproducible LUT-only exporter lives at
[`tools/export_canoncgt.py`](../../tools/export_canoncgt.py). We deliberately did
not fine-tune on a small collection of unrelated JPEGs: without known desired
outputs they provide no valid look-transfer labels and can teach the model to
confuse scene content or camera processing with the requested grade.

Date: 2026-08-08

Addendum: 2026-08-08 — few-shot personalization and architectural review.

## Executive recommendation

For this application, the best route is a **RAW-aware, reference-conditioned LUT model**, trained offline and personalized in the GUI with a small adapter. It should predict a constrained global transform (tone curve + 3D LUT), not generate pixels. The existing 33³ LUT shader can then apply the result at full resolution in one pass.

Use a frozen, small image encoder only to read the target/reference pair. Do not use CLIP or DINOv2 as the color-transfer model: they are useful semantic/content features, but their invariance is a liability when exact white balance, contrast, and saturation are the target. A low-level encoder trained or fine-tuned on RAW/edit pairs is preferable.

## Dataset options

| Dataset | What it provides | Suitability | Rights/caveat |
|---|---|---|---|
| [MIT–Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/) | 5,000 RAW DNG photographs, five independent expert Lightroom renditions, slider histories, and scene metadata | Best first dataset for global tone/color transfer. Five experts provide multiple valid grades for the same source, useful for learning a stable transform and a style-conditioned head. | The official page lists separate Adobe and Adobe/MIT research licenses. Do not assume commercial model-training rights; obtain permission or use only for research. About 50 GB for the full archive. |
| [PPR10K](https://github.com/csjliang/PPR10K) and [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Liang_PPR10K_A_Large-Scale_Portrait_Photo_Retouching_Dataset_With_Human-Region_Mask_CVPR_2021_paper.pdf) | 11,161 RAW portrait photos in 1,681 groups, three expert retouches, full-resolution human masks, and XMP source/target adjustment files | Best for portraits, skin preservation, and group-consistent grading. The XMP files expose the edit parameters and make RAW-domain augmentation practical. | Images and derived data are explicitly non-commercial research only, despite the code repository being Apache-2.0. The data is approximately 406 GB. |
| [RAISE](https://loki.disi.unitn.it/RAISE/index.php) | 8,156 native RAW images from three cameras, four photographers, and varied locations, with people/indoor/outdoor tags | Good unedited RAW diversity for camera/illumination augmentation and validation. It has no expert grades, so it cannot alone supervise look transfer. The currently downloaded people subset is useful for local testing. | Check the dataset terms before redistribution or commercial training. It is primarily a forensic dataset, not an editing benchmark. |
| [Google HDR+ Burst](https://www.hdrplusdata.org/dataset.html) and [paper](https://static.googleusercontent.com/media/hdrplusdata.org/en//hdrplus_supp.pdf) | 3,640 bursts / 28,461 RAW frames plus HDR+ intermediate and final results | Useful for testing RAW decode, exposure range, noise, and highlight behavior. It is not a grading dataset and should not be the main source of look labels. | Dataset access and terms must be checked before product use; it is primarily a computational-photography research release. |

### Dataset strategy

Train the base model with FiveK and PPR10K. Convert every RAW through exactly one controlled pipeline into linear RGB, then generate training pairs from the original and expert-rendered images. Split by photographer/group/camera, not random individual images, to avoid leakage.

Use RAISE and HDR+ only for camera/RAW robustness and synthetic grade generation unless a compatible license is confirmed. The application’s manually taught pairs should be held out as a personal-style validation set; they should not be mixed indiscriminately into the base test split.

## Model options

### Recommended: pair encoder → constrained adaptive LUT

Use a small MobileNet/ConvNeXt-style encoder with shared weights for target and reference. Concatenate target embedding, reference embedding, and their difference. The head predicts:

* a monotonic 1D luminance curve with 24–32 knots;
* a 3×3 color/white-balance matrix;
* 4–8 coefficients for learned 17³ LUT bases;
* transform strength and confidence.

Bake these outputs into the existing 33³ LUT. Monotonic parameterization makes exposure, contrast, black point, and highlight roll-off safe by construction. Low-rank LUT bases keep the model small and stop 30 GUI examples from overfitting an unrestricted 3D LUT.

This design is supported by the real-time LUT literature: [image-adaptive 3D LUTs](https://arxiv.org/abs/2009.14468), [AdaInt](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_AdaInt_Learning_Adaptive_Intervals_for_3D_Lookup_Tables_on_Real-Time_CVPR_2022_paper.pdf), and the [official AdaInt implementation](https://github.com/ImCharlesY/AdaInt). It preserves the project’s fast shader path.

### Optional local extension: HDRNet/bilateral grid

If a global LUT cannot handle portrait foreground/background differences, add a very small bilateral-grid residual after the global LUT. [Deep Bilateral Learning/HDRNet](https://groups.csail.mit.edu/graphics/hdrnet/data/hdrnet.pdf) explicitly runs the expensive network at low resolution and applies the learned edge-aware transform at high resolution. This is a good second phase, not the starting point: it adds a custom slicing implementation and more opportunities for halos or inconsistent skin.

### Models that are useful but not sufficient alone

* [DINOv2](https://github.com/facebookresearch/dinov2) is Apache-2.0 and provides strong general visual features. It can be used frozen for scene/content matching or as a small conditioning input, but it is not trained to preserve exact color grading and should not directly predict the LUT without RAW/edit fine-tuning.
* CLIP/SigLIP-style encoders are similarly useful for semantic similarity, not photometric fidelity. They may actively discard distinctions such as white balance that the product must transfer.
* [DeepLPF](https://github.com/sjmoran/DeepLPF) and [MAXIM trained on FiveK](https://huggingface.co/google/maxim-s2-enhancement-fivek) are useful baselines/teachers for single-image enhancement. They are not reference-conditioned look-transfer models, so using them directly would improve enhancement but not reliably copy an arbitrary reference grade.

## How to make 30 user examples useful

Do not fine-tune the encoder with 30 examples. For each GUI teaching action, retain aligned thumbnails of original target, reference, and desired target. Fit the constrained curve/matrix/LUT coefficients directly from the original→desired pixels. Then train only a ridge/linear residual adapter over the frozen pair embedding.

Use leave-one-example-out validation and activate the new adapter only when it improves held-out error. For an out-of-distribution pair, blend toward the deterministic transform or nearest known example instead of extrapolating. This turns 30 images into thousands of crop/patch observations without pretending they are thousands of independent scenes.

The training loss should compare the rendered result, not abstract parameter values:

* linear-RGB reconstruction;
* Oklab/DeltaE color difference;
* extra skin and neutral-gray weighting;
* highlight/shadow clipping penalties;
* monotonic/smooth LUT constraints;
* identity loss when no grade should be applied.

## Focused architecture review for approximately 30 corrected pairs

### Important identifiability limit

One finished reference photograph does not uniquely reveal its grade. Its pixels combine scene illumination, camera response, RAW development, exposure, and artistic adjustments. A single reference can therefore support only an estimate of the *canonical look* unless the model has learned those ambiguities from a broad training set. A reference before/after pair (or the reference RAW plus its edit metadata) is much stronger supervision.

This is why a small MLP trained only on 30 finished reference images cannot become universally reliable. The 30 examples can personalize a previously trained model, but they cannot supply the missing universal prior. The system should expose a confidence score and conservatively fall back when the reference is unlike the training distribution.

### Candidate comparison

| Candidate | 30-pair behavior | Exact brightness/contrast/WB/temp/tint/saturation | Content invariance | Runtime and complexity | Decision |
|---|---|---|---|---|---|
| Global parameter head | Stable if the output space is small and regularized; easy to personalize with ridge regression or nearest-neighbor interpolation | Strong for explicit controls, but a short parameter vector cannot represent all cross-channel interactions | Strong when inputs are robust photometric summaries rather than raw semantic features | Fastest and simplest | Keep as a safety path and interpretable residual adapter |
| Differentiable color pipeline | Excellent for fitting each user edit because every pixel contributes to a known stage | Strongest: separate linear-RGB WB matrix, exposure, monotone tone curve, saturation/vibrance, and smooth 3D LUT | Good if trained on varied scenes and uses masked/quantile statistics | Training is more involved; inference is only a small head plus LUT baking | Use as the target representation and training loss, not as a giant end-to-end image generator |
| Hypernetwork producing an unrestricted LUT | Can memorize 30 pairs and extrapolate badly | A LUT can express the controls, but constraints are needed for monotonicity, gamut, and smoothness | Depends entirely on its encoder; raw-image conditioning can copy scene content | Fast at application, but model/training complexity is higher | Use a *low-rank* hypernetwork: predict 4–8 basis weights plus constrained curve parameters |
| Pretrained DINOv2/CLIP/SigLIP encoder with a new head | Few-shot linear adaptation is stable | Weak color fidelity: semantic encoders intentionally become insensitive to many pixel/color changes | Excellent semantic invariance | DINOv2 ViT models may be too heavy for a two-second cross-platform path | Optional content branch only; never the photometric branch |
| Pretrained paired encoder/model | Best universal prior if trained on the right task | CanonCGT and recent reference-grading work show the value of canonicalization and LUT output | Depends on training pairs and canonicalization | A compact distilled model can meet the target; large diffusion models should not be in the runtime path | Preferred base model, personalized by the small adapter |

The most relevant recent references are [Deep Preset](https://openaccess.thecvf.com/content/WACV2021/html/Ho_Deep_Preset_Blending_and_Retouching_Photos_With_Color_Style_Transfer_WACV_2021_paper.html), which predicts low-level editing parameters rather than copying reference content; [DualBLN](https://openaccess.thecvf.com/content/ACCV2022/html/Zhang_Dual_Branch_LUT-aware_Network_for_Real-time_Image_Retouching_ACCV2022_paper.html), which learns adaptive combinations of LUT bases; [CanonCGT](https://openaccess.thecvf.com/content/CVPR2026/html/Ko_CanonCGT_Reference-Based_Color_Grading_via_Canonical_Pivot_Representation_CVPR_2026_paper.html), which uses a canonical pivot to separate intrinsic image tone from reference style; and the 2026 [Photography Retouching Transfer challenge](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/html/Elezabi_Photography_Retouching_Transfer_NTIRE_2026_Challenge_Report_CVPRW_2026_paper.html), whose task definition explicitly requires extracting a reference edit while preserving the new image’s fidelity.

### Revised clean design

Use two small branches behind one module:

1. A **photometric branch** receives linearized target/reference thumbnails and robust measurements: luminance quantiles, chroma distributions, neutral/skin/sky/foliage pools, and a compact low-level CNN embedding. It predicts a constrained transform recipe.
2. An optional frozen **content branch** (a small DINOv2 distillation or MobileNet feature) provides scene/context information only for deciding whether a style transfer is plausible and for confidence; it must not directly control white balance or color values.
3. A differentiable renderer composes the recipe as `WB matrix → exposure → monotone tone curve → saturation/vibrance → low-rank 3D LUT`, computes the rendered loss, and bakes one 33³ LUT for runtime.
4. GUI teaching fits the recipe from aligned original/desired pixels, then updates only a ridge residual adapter. Keep the base branch frozen. Store the examples and adapter version so a bad teaching action can be reverted.

This is cleaner than a general hypernetwork because the interface is a small, inspectable recipe, while the low-rank LUT supplies the missing nonlinear color interactions. It also directly addresses contrast and black-point failures instead of hoping a latent feature learns them.

### What to train offline

Use FiveK/PPR10K real edits plus synthetic perturbations generated in the same differentiable pipeline. For each episode, sample a target scene, a different reference scene carrying the same edit/preset, and the desired edited target. Include hard negatives where the reference has similar content but a different grade, and style-preserving crops/flip/exposure variants. Split by photographer, camera, group, and preset so a validation result measures scene invariance.

The practical model should be distilled from a larger teacher if necessary, but the final student should be a static 224–256 px model producing fewer than roughly 100 recipe values. This makes Core ML/ONNX export straightforward and leaves the full-resolution work to the existing GPU LUT pass. A diffusion or full-resolution generator is not justified by the latency target.

## Bottom line

For 30 user pairs, the cleanest achievable solution is **not** a new end-to-end neural image editor. It is:

* a broad offline prior trained on RAW/edit pairs;
* a differentiable, explicit color pipeline as the model’s output space and loss;
* a low-rank LUT hyper-head for nonlinear color relationships;
* a frozen encoder plus a tiny ridge/nearest-example personalization adapter;
* confidence gating and deterministic fallback.

This gives the user exact, understandable control over exposure/contrast/WB/temp/tint/saturation, preserves target structure, and can meet sub-two-second application time. It cannot make a single arbitrary finished reference fully identifiable; that residual uncertainty must be measured and handled conservatively.

## Runtime/deployment

Export a static-shape FP16 model with a 224–256 px input and a small output head. Keep full-resolution processing in the existing WGSL LUT pass. Cache the reference embedding at capture time.

For macOS, [Apple Core ML Tools can convert PyTorch directly to Core ML](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html), and ML Programs support FP16 precision via [the documented conversion options](https://apple.github.io/coremltools/docs-guides/source/convert-to-ml-program.html). For a cross-platform fallback, export ONNX and use [ONNX Runtime’s CoreML execution provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html); [OpenVINO](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html) is a useful Intel CPU/ integrated-GPU option.

Benchmark warm and cold latency separately. The release requirement should be p95 end-to-end application time under two seconds, including RAW decode, thumbnail creation, inference, LUT construction, and GPU submission—not just neural inference.

## Practical decision

1. Build a reproducible RAW-to-linear training converter and benchmark FiveK first.
2. Add PPR10K when portrait/skin and group consistency are important and the product is research-only or separate commercial licensing is obtained.
3. Train the constrained pair-to-LUT model and compare it against deterministic transfer and CanonCGT on a fixed cross-camera benchmark.
4. Add HDRNet-style local residuals only if global-LUT failures are measured and frequent.
5. Keep DINOv2 as an optional frozen conditioning/content encoder, not as the color-transfer core.

The main reliability risk is not the choice between CLIP and DINO. It is inconsistent RAW rendering and insufficiently controlled labels. Every dataset and GUI pair must use the same linearization, white-balance convention, gamut, and display encoding before model quality can be judged.
