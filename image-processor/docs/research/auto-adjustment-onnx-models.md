# ONNX candidates for Auto Adjust

Research date: 2026-07-29. Scope: a local Rust/`tract` photo editor needs a small, redistributable model that takes an sRGB photo and returns a conservative exposure/colour/tone correction. Image generation, face restoration, and a quality score are different tasks and are not considered replacements for Auto Adjust.

## Recommendation

Use **Image-Adaptive-3DLUT**. It is trained for general sRGB photo colour and tone retouching on MIT-Adobe FiveK. Its small classifier predicts three weights and the app combines three 33³ basis LUTs, then applies the resulting transform exactly at full resolution. This avoids the visibly destructive approximation of a spatial RGB output with exposure, contrast, curve, and white-balance sliders.

- [Image-Adaptive-3DLUT project, paper, checkpoints, and Apache-2.0 license](https://github.com/HuiZeng/Image-Adaptive-3DLUT)
- [Official sRGB checkpoints](https://github.com/HuiZeng/Image-Adaptive-3DLUT/tree/master/pretrained_models/sRGB)

The classifier was exported locally from the official `classifier.pth`; the LUTs were extracted losslessly from the official `LUTs.pth`. The embedded artifacts total about 2.3 MB. The model was selected only after a same-image differential test showed that IAT's raw exposure checkpoint made a representative, already-correct portrait darker, harsher, and oversaturated. The 3D-LUT result preserved natural skin while improving tone and colour.

## Candidates compared

| Model | What it actually enhances | Size / runtime evidence | License and deployment fit | Decision |
|---|---|---|---|---|
| [IAT](https://github.com/cuiziteng/Illumination-Adaptive-Transformer) | Exposure correction (both under- and over-exposure), low-light, local pixel adjustment, global colour matrix and gamma | ~90K parameters; official README reports ~0.004 s on a 3090; exposure weights 427 KB | Apache-2.0 repository. A real-photo differential test produced poor results on already-correct input. | Rejected after validation |
| [Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT) | General photo colour and tone retouching learned from paired/unpaired FiveK data | <600K parameters and <2 ms for 4K on Titan RTX; official sRGB checkpoints are 1.29 MB LUTs + 1.08 MB classifier ([checkpoint metadata](https://api.github.com/repos/HuiZeng/Image-Adaptive-3DLUT/contents/pretrained_models/sRGB?ref=master)) | Apache-2.0. ONNX classifier plus a native GPU trilinear sampler works without custom operators. | **Implemented** |
| [AdaInt](https://github.com/ImCharlesY/AdaInt) | Adaptive 3D-LUT photo enhancement/retouching on MIT-Adobe FiveK and PPR10K | FiveK sRGB checkpoint 2.49 MB; PPR10K checkpoint 47.1 MB ([checkpoint metadata](https://api.github.com/repos/ImCharlesY/AdaInt/contents/pretrained?ref=main)) | Apache-2.0, but the central AiLUT transform is a compiled CUDA extension, making direct ONNX/`tract` deployment impractical without reimplementation. | Good quality candidate, not the shortest path |
| [HDRNet / Deep Bilateral Learning](https://github.com/google/hdrnet) | Learned local affine tone/colour transforms in a bilateral grid; very close conceptually to auto retouching | Official project provides pretrained models and a mobile-oriented graph, but requires a custom bilateral-slice TensorFlow op; the repository explicitly says the op is GPU-only | Apache-2.0 code, archived; CPU Rust integration is a blocker, and its HDR+ model expects linear 16-bit RAW rather than ordinary sRGB | Do not use for this app's sRGB Auto Adjust |
| [Deep White-Balance Editing](https://github.com/mahmoudnafifi/Deep_White_Balance) | Corrects wrong sRGB white balance and predicts indoor/outdoor WB renditions | End-to-end PyTorch/Matlab model; no official ONNX export or compact artifact documented | CC BY-NC-SA 4.0 and the project says research-only/non-commercial; incompatible with a generally redistributable app | Exclude for licensing, even though WB quality is relevant |
| [Learning Multi-Scale Photo Exposure Correction](https://github.com/mahmoudnafifi/Exposure_Correction) | Specifically corrects both over- and under-exposed sRGB photos, with colour and detail stages | The official code is Matlab and no compact ONNX artifact is documented | Project says research-only and cannot be used commercially; dataset follows MIT-Adobe FiveK terms | Exclude for licensing and conversion effort |
| [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE) / Zero-DCE++ | Image-specific curves for low-light enhancement | Zero-DCE++ is tiny, but official repo states Attribution-NonCommercial terms; it is primarily low-light, not general exposure/WB | Non-commercial academic license; not suitable for redistribution | Exclude |
| [MIRNet](https://github.com/swz30/MIRNet) | Image restoration (denoising, super-resolution, low-light enhancement) | Full multi-resolution network; official repo distributes task checkpoints through external downloads and does not provide a compact ONNX path | Useful research baseline, but much larger and restoration-oriented rather than a predictable edit-parameter correction | Not a good Auto Adjust replacement |

## Integration notes

1. Run the classifier on a 256×256 preview, then persist its three weights in the edit state.
2. Combine the official basis LUTs and apply the transform in the existing GPU pipeline before manual adjustments.
3. Keep the transform non-destructive: Reset disables it, reopening restores it, and export uses the same pipeline.

## Bottom line

Image-Adaptive-3DLUT is the best fit: it is quality-oriented, tiny, permissively licensed, and its native representation can be applied exactly instead of being forced into inadequate global sliders.
