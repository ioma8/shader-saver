//! Diagnose specific frames against their SOOC JPEG sidecars, by explicit path:
//!
//! ```bash
//! RAW_FILES=a.ARW,b.ARW cargo test --release frames_probe -- --nocapture
//! ```
//!
//! Prints tone/clip/bias/MAE per frame and writes `/tmp/frames/<name>_{dev,ref}.png`.

use super::*;

fn to8(img: &Rgba16Image) -> RgbaImage {
    RgbaImage::from_fn(img.width(), img.height(), |x, y| {
        let p = img.get_pixel(x, y);
        image::Rgba([(p[0] >> 8) as u8, (p[1] >> 8) as u8, (p[2] >> 8) as u8, 255])
    })
}

fn sidecar(path: &Path) -> Option<RgbaImage> {
    for ext in ["jpg", "JPG", "jpeg", "JPEG"] {
        let c = path.with_extension(ext);
        if c.exists() {
            let i = image::open(&c).ok()?;
            return Some(crate::imgload::orient_preview(i, crate::imgload::exif_orientation(&c)).to_rgba8());
        }
    }
    None
}

struct Stats {
    p5: f32,
    p50: f32,
    p95: f32,
    clip: f32,
    mean: [f32; 3],
}

fn stats(img: &RgbaImage) -> Stats {
    let n = (img.width() * img.height()) as f32;
    let mut luma: Vec<f32> = img
        .pixels()
        .map(|p| (0.2126 * f32::from(p[0]) + 0.7152 * f32::from(p[1]) + 0.0722 * f32::from(p[2])) / 255.0)
        .collect();
    let clip = img.pixels().filter(|p| p[0] == 255 || p[1] == 255 || p[2] == 255).count() as f32 / n;
    let mut mean = [0.0f32; 3];
    for p in img.pixels() {
        for k in 0..3 {
            mean[k] += f32::from(p[k]) / n;
        }
    }
    luma.sort_unstable_by(f32::total_cmp);
    let at = |q: f32| luma[((luma.len() - 1) as f32 * q) as usize];
    Stats { p5: at(0.05), p50: at(0.5), p95: at(0.95), clip, mean }
}

#[test]
fn frames_probe() {
    let Ok(list) = std::env::var("RAW_FILES") else {
        return;
    };
    std::fs::create_dir_all("/tmp/frames").unwrap();
    let dim: u32 = std::env::var("FRAME_DIM").ok().and_then(|v| v.parse().ok()).unwrap_or(768);
    for f in list.split(',').filter(|s| !s.is_empty()) {
        let path = Path::new(f);
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let Some(mut decoded) = decode_raw_u16(path, dim) else {
            eprintln!("{name}: DECODE FAILED");
            continue;
        };
        // Stage stats, to see where brightness is lost.
        let stage = |img: &Rgba16Image| {
            let s = stats(&to8(img));
            format!("p5={:.3} p50={:.3} p95={:.3}", s.p5, s.p50, s.p95)
        };
        let raw = raw_features(&decoded);
        if std::env::var("FRAME_SAVE_DECODE").is_ok() {
            to8(&decoded).save(format!("/tmp/frames/{name}_decode.png")).unwrap();
        }
        let after_decode = stage(&decoded);
        tone_u16_with(&mut decoded, &TONE);
        if std::env::var("FRAME_SAVE_DECODE").is_ok() {
            to8(&decoded).save(format!("/tmp/frames/{name}_tone.png")).unwrap();
        }
        let after_tone = stage(&decoded);
        let (gain, offset, gamma) = meter_predict(&decoded, &raw);
        eprintln!("    meter predict: gain={gain:.3} offset={offset:+.1} gamma={gamma:.3}");
        meter_u16(&mut decoded, &raw);
        let after_meter = stage(&decoded);
        let dev = to8(&decoded);
        let Some(reference) = sidecar(path) else {
            eprintln!("{name}: no sidecar JPEG");
            continue;
        };
        let reference = image::imageops::resize(&reference, dev.width(), dev.height(), image::imageops::FilterType::Triangle);
        let (d, r) = (stats(&dev), stats(&reference));
        let mae = dev
            .pixels()
            .zip(reference.pixels())
            .flat_map(|(a, b)| (0..3).map(move |c| u8::abs_diff(a[c], b[c]) as f32 / 255.0))
            .sum::<f32>()
            / (reference.width() * reference.height() * 3) as f32;
        eprintln!(
            "{name}: MAE={mae:.4} | dev p5={:.3} p50={:.3} p95={:.3} clip={:.4} | ref p5={:.3} p50={:.3} p95={:.3} clip={:.4} | bias R{:+.1} G{:+.1} B{:+.1}",
            d.p5, d.p50, d.p95, d.clip, r.p5, r.p50, r.p95, r.clip,
            d.mean[0] - r.mean[0], d.mean[1] - r.mean[1], d.mean[2] - r.mean[2],
        );
        eprintln!("    decode {after_decode} | tone {after_tone} | meter {after_meter} | ref p5={:.3} p50={:.3} p95={:.3}", r.p5, r.p50, r.p95);
        {
            // What a per-frame correction could reach (luma plane, 0..255).
            let dl: Vec<f64> = dev.pixels().map(|p| 0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2])).collect();
            let rl: Vec<f64> = reference.pixels().map(|p| 0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2])).collect();
            let fit = |f: &dyn Fn(f64) -> f64| -> f64 {
                let np = dl.len() as f64;
                let (sd, sr, sdd, sdr) = dl.iter().zip(&rl).fold((0.0, 0.0, 0.0, 0.0), |a, (&x, &y)| {
                    let xx = f(x);
                    (a.0 + xx, a.1 + y, a.2 + xx * xx, a.3 + xx * y)
                });
                let var = sdd - sd * sd / np;
                let a = if var.abs() > 1e-9 { (sdr - sd * sr / np) / var } else { 1.0 };
                let b = sr / np - a * sd / np;
                dl.iter().zip(&rl).map(|(&x, &y)| ((a * f(x) + b) - y).abs()).sum::<f64>() / np
            };
            let base = dl.iter().zip(&rl).map(|(&x, &y)| (x - y).abs()).sum::<f64>() / dl.len() as f64;
            let affine = fit(&|x| x);
            let mut best = f64::MAX;
            let mut best_g = 1.0;
            let mut g = 0.40;
            while g <= 1.4001 {
                let m = fit(&|x| 255.0 * (x / 255.0).powf(g));
                if m < best {
                    best = m;
                    best_g = g;
                }
                g += 0.05;
            }
            eprintln!("    luma MAE: current={base:.2} +affine={affine:.2} +affine&gamma={best:.2} (gamma={best_g:.2}) /255");
        }
        dev.save(format!("/tmp/frames/{name}_dev.png")).unwrap();
        reference.save(format!("/tmp/frames/{name}_ref.png")).unwrap();
    }
}
