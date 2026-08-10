// Neural Preset (CVPR 2023) `StyleSimiliaryDiscriminator`, CC BY-NC-SA licensed
// -- used only from this test module, to check the look-transfer pipeline's
// real rendered output, never to steer it at runtime. An earlier version of
// this pipeline searched its own transfer parameters directly against this
// judge, and it scored well while visibly wrecking pictures: grey pavement
// pushed to pink, greens to neon, gradients posterized flat. A learned metric
// has no opinion about pink pavement, so a free search finds pink pavement.
//
// Loaded at runtime only, from
// `STYLE_JUDGE_ONNX` or `~/.image-processor/style-judge.onnx`, and every test
// here degrades to a no-op `eprintln!` skip when it's absent.
//
// The model takes two 512x512x3 (HWC, not the usual CHW) float tensors and
// returns one scalar: higher means more style-similar. Probed empirically
// (the graph's own output shape is untyped): two flat-color 512-square inputs
// scored 0.6965, the same image against itself scored 0.6987 -- a small but
// consistent edge for "more similar", which is all this harness needs.
#[cfg(test)]
pub struct StyleJudge {
    runnable: tract::prelude::Runnable,
}

#[cfg(test)]
impl StyleJudge {
    pub fn load() -> Option<Self> {
        use tract::prelude::*;
        let path = std::env::var("STYLE_JUDGE_ONNX").unwrap_or_else(|_| {
            format!(
                "{}/.image-processor/style-judge.onnx",
                std::env::var("HOME").unwrap_or_default()
            )
        });
        let bytes = std::fs::read(path).ok()?;
        let runnable = tract::onnx()
            .ok()?
            .load_buffer(&bytes)
            .ok()?
            .into_model()
            .ok()?
            .into_runnable()
            .ok()?;
        Some(Self { runnable })
    }

    pub fn score(&self, reference: &image::RgbaImage, candidate: &image::RgbaImage) -> Option<f32> {
        use tract::prelude::*;
        let a = to_hwc_tensor(reference)?;
        let b = to_hwc_tensor(candidate)?;
        let out = self.runnable.run([a, b]).ok()?;
        out.first()?.as_slice::<f32>().ok()?.first().copied()
    }
}

#[cfg(test)]
fn to_hwc_tensor(img: &image::RgbaImage) -> Option<tract::prelude::Tensor> {
    use tract::prelude::*;
    let resized = image::imageops::resize(img, 512, 512, image::imageops::FilterType::Triangle);
    let mut data = vec![0f32; 512 * 512 * 3];
    for (i, px) in resized.pixels().enumerate() {
        data[i * 3] = f32::from(px[0]) / 255.0;
        data[i * 3 + 1] = f32::from(px[1]) / 255.0;
        data[i * 3 + 2] = f32::from(px[2]) / 255.0;
    }
    Tensor::from_slice(&[512usize, 512, 3], &data).ok()
}

#[cfg(test)]
mod tests {
    use super::StyleJudge;
    use crate::face::Detector;
    use crate::look_reference_thumb;
    use crate::processor::{
        baked_lut, identity_photo_lut, measure_regions, sample_cube, LookProfile,
    };
    use crate::{imgload, look_chain_for, CapturedLook, EditState};
    use std::path::PathBuf;

    #[test]
    fn embedded_judge_loads_and_prefers_the_matching_image() {
        let Some(judge) = StyleJudge::load() else {
            eprintln!("skipped: no STYLE_JUDGE_ONNX");
            return;
        };
        let a = image::RgbaImage::from_pixel(64, 64, image::Rgba([200, 120, 90, 255]));
        let b = image::RgbaImage::from_pixel(64, 64, image::Rgba([40, 60, 200, 255]));
        let same = judge.score(&a, &a).unwrap();
        let different = judge.score(&a, &b).unwrap();
        assert!(
            same >= different,
            "same-image score {same} should be >= a differently-colored image's {different}"
        );
    }

    // The single constrained model must always bake to a display-safe LUT.
    #[test]
    fn look_model_lut_stays_in_gamut() {
        let (Ok(rp), Ok(tp)) = (
            std::env::var("STYLE_JUDGE_REFERENCE"),
            std::env::var("STYLE_JUDGE_TARGET"),
        ) else {
            eprintln!("skipped: no STYLE_JUDGE_REFERENCE / STYLE_JUDGE_TARGET");
            return;
        };

        let load = |p: &str| imgload::load_rgba(&PathBuf::from(p), 768).unwrap();
        let reference = load(&rp);
        let detector = Detector::load();
        let ref_faces = detector
            .as_ref()
            .map(|d| d.detect_boxes(&reference))
            .unwrap_or_default();
        let profile = LookProfile::measure(
            reference.as_raw(),
            reference.width(),
            reference.height(),
            &ref_faces,
        )
        .unwrap();
        let captured = CapturedLook {
            profile,
            reference: look_reference_thumb(&reference),
            reference_full: reference.clone(),
        };

        let examples = crate::app_dir()
            .map(|dir| crate::look_model::load_examples(&crate::existing_look_examples_path(&dir)))
            .unwrap_or_default();
        let model = crate::look_model::LookModel::train_with_examples(&examples);
        let canon = crate::canoncgt::CanonCgt::load();
        let started = std::time::Instant::now();
        let target = load(&tp);
        let faces = detector
            .as_ref()
            .map(|d| d.detect_boxes(&target))
            .unwrap_or_default();
        let mut state = EditState::default();
        look_chain_for(
            &target,
            &mut state,
            &captured,
            &model,
            canon.as_ref(),
            &faces,
        );
        let Some(lut) = baked_lut(&state) else {
            panic!("look model did not produce a LUT")
        };
        let elapsed = started.elapsed();

        let identity = identity_photo_lut();
        let mean_deviation = lut
            .chunks_exact(3)
            .zip(identity.chunks_exact(3))
            .map(|(a, b)| {
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
            })
            .sum::<f32>()
            / (lut.len() / 3) as f32;
        println!(
            "trained on {} examples; apply {:?}; mean LUT deviation {mean_deviation:.4}",
            examples.len(),
            elapsed
        );
        assert!(
            elapsed.as_secs_f32() < 2.0,
            "look application must remain under two seconds"
        );

        if let Ok(output) = std::env::var("STYLE_JUDGE_OUTPUT") {
            let rendered = image::RgbaImage::from_fn(target.width(), target.height(), |x, y| {
                let px = target.get_pixel(x, y);
                let rgb = sample_cube(
                    &lut,
                    33,
                    [
                        f32::from(px[0]) / 255.0,
                        f32::from(px[1]) / 255.0,
                        f32::from(px[2]) / 255.0,
                    ],
                );
                image::Rgba([
                    (rgb[0] * 255.0).round() as u8,
                    (rgb[1] * 255.0).round() as u8,
                    (rgb[2] * 255.0).round() as u8,
                    px[3],
                ])
            });
            rendered.save(output).unwrap();
        }

        let total = lut.len() / 3;
        let out_of_range = lut
            .chunks_exact(3)
            .filter(|c| c.iter().any(|v| *v < -1e-4 || *v > 1.0001))
            .count();
        println!(
            "look-model LUT: {out_of_range}/{total} cells ({:.2}%) outside [0,1]",
            100.0 * out_of_range as f32 / total as f32
        );
        assert!(
            (out_of_range as f32 / total as f32) < 0.005,
            "constrained look model should remain in gamut"
        );

        let regions = measure_regions(target.as_raw(), target.width(), target.height(), &faces);
        let skin = regions[0];
        if skin.share > 0.0 {
            let mut sampled_out_of_range = 0;
            let n = 200;
            for i in 0..n {
                let t = i as f32 / n as f32;
                let rgb = [0.55 + t * 0.25, 0.40 + t * 0.20, 0.30 + t * 0.15];
                let out = sample_cube(&lut, 33, rgb);
                if out.iter().any(|v| *v < -1e-4 || *v > 1.0001) {
                    sampled_out_of_range += 1;
                }
            }
            println!("skin-hue sweep ({n} samples): {sampled_out_of_range}/{n} land outside [0,1]");
            assert_eq!(
                sampled_out_of_range, 0,
                "skin-hue neighborhood should be fully in-gamut after the fix"
            );
        }
    }

    // Sweeps the *actual production* `look_chain_for` over every ordered pair
    // of images in a folder and scores each transfer against its reference
    // with the judge -- a real measurement, not a single cherry-picked pair.
    // Informational bounds only (see the module doc comment on why this must
    // never become an optimization target): prints the full distribution so a
    // human can compare runs, and only fails on a gross regression.
    #[test]
    fn judge_sweeps_look_transfer_over_a_folder() {
        let (Some(judge), Ok(dir)) = (StyleJudge::load(), std::env::var("STYLE_JUDGE_FOLDER"))
        else {
            eprintln!("skipped: no STYLE_JUDGE_ONNX / STYLE_JUDGE_FOLDER");
            return;
        };
        let model = crate::look_model::LookModel::train();
        let canon = crate::canoncgt::CanonCgt::load();
        let detector = Detector::load();

        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| imgload::is_supported(p))
            .collect();
        paths.sort();
        assert!(paths.len() >= 2, "need at least two images in {dir}");

        let images: Vec<(PathBuf, image::RgbaImage)> = paths
            .iter()
            .map(|p| (p.clone(), imgload::load_rgba(p, 768).unwrap()))
            .collect();

        let mut scores = Vec::new();
        for (ref_path, reference) in &images {
            let ref_faces = detector
                .as_ref()
                .map(|d| d.detect_boxes(reference))
                .unwrap_or_default();
            let Some(profile) = LookProfile::measure(
                reference.as_raw(),
                reference.width(),
                reference.height(),
                &ref_faces,
            ) else {
                continue;
            };
            let captured = CapturedLook {
                profile,
                reference: look_reference_thumb(reference),
                reference_full: reference.clone(),
            };

            for (target_path, target) in &images {
                if target_path == ref_path {
                    continue;
                }
                let faces = detector
                    .as_ref()
                    .map(|d| d.detect_boxes(target))
                    .unwrap_or_default();
                let mut state = EditState::default();
                look_chain_for(
                    target,
                    &mut state,
                    &captured,
                    &model,
                    canon.as_ref(),
                    &faces,
                );
                let Some(lut) = baked_lut(&state) else {
                    continue;
                };
                let rendered =
                    image::RgbaImage::from_fn(target.width(), target.height(), |x, y| {
                        let px = target.get_pixel(x, y);
                        let rgb = sample_cube(
                            &lut,
                            33,
                            [
                                f32::from(px[0]) / 255.0,
                                f32::from(px[1]) / 255.0,
                                f32::from(px[2]) / 255.0,
                            ],
                        );
                        image::Rgba([
                            (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                            (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                            (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                            px[3],
                        ])
                    });
                if let Some(score) = judge.score(reference, &rendered) {
                    scores.push((score, ref_path.clone(), target_path.clone()));
                }
            }
        }

        scores.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mean: f32 = scores.iter().map(|(s, ..)| s).sum::<f32>() / scores.len() as f32;
        println!(
            "{} pairs -- mean {:.4}, min {:.4} ({:?} -> {:?}), max {:.4}",
            scores.len(),
            mean,
            scores[0].0,
            scores[0].1,
            scores[0].2,
            scores.last().unwrap().0
        );
        for (score, r, t) in scores.iter().take(5) {
            println!("  low: {score:.4}  {r:?} -> {t:?}");
        }
        assert!(
            mean > 0.5,
            "mean similarity {mean} looks like a broken pipeline, not just an imperfect one"
        );
    }
}
