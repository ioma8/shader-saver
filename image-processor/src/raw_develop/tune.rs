    use super::*;
    use rayon::prelude::*;
    use std::path::PathBuf;

    /// Sweep resolution. Tone is global, so a small image ranks parameters the
    /// same as a large one and is far cheaper.
    const DIM: u32 = 160;
    const LUT_N: usize = 1024;

    struct Sample {
        /// Linear-ish sRGB encoded channels, as the tone sees them.
        encoded: Vec<[f32; 3]>,
        /// `encoded` luminance, sorted (so levels are O(1) per combo).
        luma: Vec<f32>,
        /// Reference pixels resized to the develop's dimensions.
        reference: Vec<u8>,
        width: u32,
        height: u32,
    }

    fn flatten(img: &RgbaImage, width: u32, height: u32) -> Vec<u8> {
        let resized = image::imageops::resize(img, width, height, image::imageops::FilterType::Triangle);
        let mut out = Vec::with_capacity((width * height * 3) as usize);
        for p in resized.pixels() {
            out.extend_from_slice(&p.0[..3]);
        }
        out
    }

    /// Sidecar JPEG (EXIF-oriented), else a usable embedded preview.
    fn reference_bytes(path: &Path, width: u32, height: u32) -> Option<Vec<u8>> {
        for ext in ["jpg", "JPG", "jpeg", "JPEG"] {
            let c = path.with_extension(ext);
            if c.exists() {
                if let Ok(i) = image::open(&c) {
                    let i = crate::imgload::orient_preview(i, crate::imgload::exif_orientation(&c));
                    return Some(flatten(&i.to_rgba8(), width, height));
                }
            }
        }
        let i = load_render_target(path, 0)?;
        if i.width().max(i.height()) < 512 {
            return None;
        }
        let mut v: Vec<u8> = i.pixels().map(|p| p[0].max(p[1]).max(p[2])).collect();
        v.sort_unstable();
        if v[v.len() * 19 / 20] <= 12 {
            return None;
        }
        Some(flatten(&i, width, height))
    }

    fn sample(path: &Path) -> Option<Sample> {
        let dev = decode_raw_u16(path, DIM)?;
        let (width, height) = (dev.width(), dev.height());
        let encoded: Vec<[f32; 3]> = dev
            .pixels()
            .map(|p| std::array::from_fn(|c| f32::from(p[c]) / 65535.0))
            .collect();
        let mut luma: Vec<f32> = encoded.iter().map(|&c| luma_of(c)).collect();
        luma.sort_unstable_by(f32::total_cmp);
        let reference = reference_bytes(path, width, height)?;
        Some(Sample { encoded, luma, reference, width, height })
    }

    fn load(dir: &str, keep: &dyn Fn(&Path) -> bool) -> Vec<Sample> {
        fn walk(d: &Path, keep: &dyn Fn(&Path) -> bool, out: &mut Vec<PathBuf>) {
            for e in std::fs::read_dir(d).unwrap().filter_map(|e| e.ok()) {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, keep, out);
                } else if crate::imgload::is_raw(&p) && keep(&p) {
                    out.push(p);
                }
            }
        }
        let mut files = Vec::new();
        walk(Path::new(dir), keep, &mut files);
        files.sort();
        files.par_iter().filter_map(|p| sample(p)).collect()
    }

    /// `tone_map` over m in [0, 4], replacing a per-pixel `powf`.
    fn lut(p: &ToneParams) -> Vec<f32> {
        (0..=LUT_N).map(|i| tone_map(4.0 * i as f32 / LUT_N as f32, p)).collect()
    }

    fn group_mae(group: &[Sample], p: &ToneParams) -> f32 {
        let lut = lut(p);
        let step = LUT_N as f32 / 4.0;
        let total: f64 = group
            .par_iter()
            .map(|s| {
                let (low, high) = tone_levels(&s.luma, p);
                let inv = 1.0 / (high - low);
                let get = |m: f32| {
                    let f = (m.max(0.0) * step).min(LUT_N as f32);
                    let i = f as usize;
                    let t = f - i as f32;
                    if i >= LUT_N {
                        lut[LUT_N]
                    } else {
                        lut[i] + (lut[i + 1] - lut[i]) * t
                    }
                };
                let scaled: Vec<[f32; 3]> = s
                    .encoded
                    .iter()
                    .map(|c| c.map(|v| ((v - low) * inv).max(0.0)))
                    .collect();
                let m: Vec<f32> = scaled.iter().map(|c| c[0].max(c[1]).max(c[2])).collect();
                let base = if p.local > 0.0 {
                    box_blur(&m, s.width as usize, s.height as usize, local_radius(s.width as usize, s.height as usize, p))
                } else {
                    Vec::new()
                };
                let mut sum = 0.0f64;
                for (i, c) in scaled.iter().enumerate() {
                    let mm = m[i];
                    let global = if mm > 1e-5 { get(mm) / mm } else { 1.0 };
                    let scale = if p.local > 0.0 {
                        let bb = base[i].max(1e-4);
                        global * (1.0 - p.local) + (get(bb) / bb) * p.local
                    } else {
                        global
                    };
                    let out = [
                        ((c[0] * scale).clamp(0.0, 1.0) * 255.0) as i32,
                        ((c[1] * scale).clamp(0.0, 1.0) * 255.0) as i32,
                        ((c[2] * scale).clamp(0.0, 1.0) * 255.0) as i32,
                    ];
                    let idx = i * 3;
                    for k in 0..3 {
                        sum += (out[k] - s.reference[idx + k] as i32).abs() as f64;
                    }
                }
                sum / (s.width as f64 * s.height as f64 * 3.0)
            })
            .sum();
        (total / group.len() as f64) as f32
    }

    #[test]
    fn tone_sweep() {
        let mut groups: Vec<(&str, Vec<Sample>)> = Vec::new();
        if let Ok(dirs) = std::env::var("RAW_SOOC_FOLDER") {
            let mut g = Vec::new();
            for d in dirs.split(':').filter(|s| !s.is_empty()) {
                g.extend(load(d, &|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("arw"))));
            }
            groups.push(("sony", g));
        }
        if let Ok(d) = std::env::var("RAW_PIXEL_FOLDER") {
            groups.push((
                "pixel",
                load(&d, &|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.ends_with(".RAW-02.ORIGINAL.dng"))),
            ));
        }
        if let Ok(d) = std::env::var("RAW_DATASET_DIR") {
            groups.push(("mixed", load(&d, &|_| true)));
        }
        if groups.is_empty() {
            return;
        }
        let names: Vec<&str> = groups.iter().map(|(n, _)| *n).collect();
        eprintln!("groups {names:?} sizes {:?}", groups.iter().map(|(_, g)| g.len()).collect::<Vec<_>>());

        let blacks = [0.005f32, 0.01];
        let whites = [0.95f32, 0.97, 0.98];
        let contrasts = [0.2f32, 0.3];
        let gammas = [0.62f32, 0.66, 0.70];
        let knees = [0.05f32, 0.1, 0.2];
        let locals = [0.3f32, 0.4, 0.5];
        let radius_divs = [4u32, 6];
        let mut combos = Vec::new();
        for &black in &blacks {
            for &white in &whites {
                for &contrast in &contrasts {
                    for &gamma in &gammas {
                        for &knee in &knees {
                            for &local in &locals {
                                for &radius_div in &radius_divs {
                                    combos.push(ToneParams { black, white, contrast, gamma, knee, local, radius_div });
                                }
                            }
                        }
                    }
                }
            }
        }
        // MAE floor: the best a constant (fully flat) image could score.
        for (name, g) in &groups {
            let floor: f64 = g
                .par_iter()
                .map(|s| {
                    let n = s.reference.len() as f64;
                    let mean: f64 = s.reference.iter().map(|&v| v as f64).sum::<f64>() / n;
                    s.reference.iter().map(|&v| (v as f64 - mean).abs()).sum::<f64>() / n
                })
                .sum::<f64>()
                / g.len() as f64;
            eprintln!("flat-image MAE floor {name} = {floor:.2}");
        }
        eprintln!("scanning {} combinations", combos.len());
        let sizes: Vec<f32> = groups.iter().map(|(_, g)| g.len() as f32).collect();
        let total: f32 = sizes.iter().sum();
        let mut results: Vec<(f32, &ToneParams, Vec<f32>)> = combos
            .par_iter()
            .map(|p| {
                let per: Vec<f32> = groups.iter().map(|(_, g)| group_mae(g, p)).collect();
                // Pooled over images, so a group with many frames counts more.
                let pooled = per.iter().zip(&sizes).map(|(m, n)| m * n).sum::<f32>() / total;
                (pooled, p, per)
            })
            .collect();
        results.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (score, p, per) in results.iter().take(10) {
            let detail: Vec<String> = per.iter().map(|v| format!("{v:.4}")).collect();
            eprintln!(
                "score={score:.4} black={:.3} white={:.3} contrast={:.2} gamma={:.3} knee={:.1} local={:.2} rdiv={} per{names:?}={detail:?}",
                p.black, p.white, p.contrast, p.gamma, p.knee, p.local, p.radius_div
            );
        }
        let per: Vec<f32> = groups.iter().map(|(_, g)| group_mae(g, &TONE)).collect();
        let pooled = per.iter().zip(&sizes).map(|(m, n)| m * n).sum::<f32>() / total;
        eprintln!(
            "current local={} pooled={pooled:.4} per{names:?}={:?}",
            TONE.local,
            per.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>()
        );
    }
