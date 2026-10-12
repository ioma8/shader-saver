    use super::*;

    fn to8(img: &Rgba16Image) -> RgbaImage {
        RgbaImage::from_fn(img.width(), img.height(), |x, y| {
            let p = img.get_pixel(x, y);
            image::Rgba([(p[0] >> 8) as u8, (p[1] >> 8) as u8, (p[2] >> 8) as u8, 255])
        })
    }

    fn reference(path: &Path) -> Option<RgbaImage> {
        for ext in ["jpg", "JPG", "jpeg", "JPEG"] {
            let c = path.with_extension(ext);
            if c.exists() {
                if let Ok(i) = image::open(&c) {
                    return Some(crate::imgload::orient_preview(i, crate::imgload::exif_orientation(&c)).to_rgba8());
                }
            }
        }
        let i = load_render_target(path, 0)?;
        if i.width().max(i.height()) < 512 { return None; }
        let mut v: Vec<u8> = i.pixels().map(|p| p[0].max(p[1]).max(p[2])).collect();
        v.sort_unstable();
        if v[v.len() * 19 / 20] <= 12 { return None; }
        Some(i)
    }

    fn load(dir: &str, is_pixel: bool) -> Vec<(RgbaImage, RgbaImage)> {
        let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| crate::imgload::is_raw(p))
            .filter(|p| {
                if is_pixel {
                    p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.ends_with(".RAW-02.ORIGINAL.dng"))
                } else {
                    p.extension().is_some_and(|e| e.eq_ignore_ascii_case("arw"))
                }
            })
            .collect();
        files.sort();
        files
            .iter()
            .filter_map(|p| {
                let mut d = decode_raw_u16(p, 384)?;
                tone_u16_with(&mut d, &TONE);
                let d = to8(&d);
                let r = reference(p)?;
                let r = image::imageops::resize(&r, d.width(), d.height(), image::imageops::FilterType::Triangle);
                Some((d, r))
            })
            .collect()
    }

    fn mae(a: &RgbaImage, b: &RgbaImage) -> f32 {
        a.pixels()
            .zip(b.pixels())
            .flat_map(|(x, y)| (0..3).map(move |c| u8::abs_diff(x[c], y[c]) as f32 / 255.0))
            .sum::<f32>()
            / (b.width() * b.height() * 3) as f32
    }

    /// One fixed 8-bit curve per channel, from the pairs (a profile, not adaptive).
    fn fit_luts(pairs: &[(RgbaImage, RgbaImage)]) -> [[f32; 256]; 3] {
        let mut sum = [[0.0f64; 256]; 3];
        let mut cnt = [[0.0f64; 256]; 3];
        for (d, r) in pairs {
            for (p, q) in d.pixels().zip(r.pixels()) {
                for k in 0..3 {
                    let b = p[k] as usize;
                    sum[k][b] += f64::from(q[k]);
                    cnt[k][b] += 1.0;
                }
            }
        }
        let mut lut = [[0.0f32; 256]; 3];
        for k in 0..3 {
            let mut last = 0.0f32;
            for b in 0..256 {
                let v = if cnt[k][b] > 5.0 { (sum[k][b] / cnt[k][b]) as f32 } else { b as f32 };
                last = last.max(v.min(255.0));
                lut[k][b] = last;
            }
        }
        lut
    }

    fn mae_with_lut(pairs: &[(RgbaImage, RgbaImage)], lut: &[[f32; 256]; 3]) -> f32 {
        let mut sum = 0.0f64;
        let mut px = 0.0f64;
        for (d, r) in pairs {
            for (p, q) in d.pixels().zip(r.pixels()) {
                for k in 0..3 {
                    sum += f64::from((lut[k][p[k] as usize] - f32::from(q[k])).abs());
                    px += 1.0;
                }
            }
        }
        (sum / px / 255.0) as f32
    }

    #[test]
    fn profile_probe() {
        let mut groups: Vec<(&str, Vec<(RgbaImage, RgbaImage)>)> = Vec::new();
        if let Ok(d) = std::env::var("RAW_SOOC_FOLDER") {
            groups.push(("sony", load(&d, false)));
        }
        if let Ok(d) = std::env::var("RAW_PIXEL_FOLDER") {
            groups.push(("pixel", load(&d, true)));
        }
        if groups.is_empty() { return; }
        for (name, g) in &groups {
            eprintln!("{name}: n={} current MAE={:.4}", g.len(), g.iter().map(|(d, r)| mae(d, r)).sum::<f32>() / g.len() as f32);
        }
        // Fit on all groups pooled, evaluate per group (a single universal profile).
        let all: Vec<(RgbaImage, RgbaImage)> = groups.iter().flat_map(|(_, g)| g.iter().cloned()).collect();
        let lut = fit_luts(&all);
        for (name, g) in &groups {
            eprintln!("{name}: with pooled fitted per-channel LUT MAE={:.4}", mae_with_lut(g, &lut));
        }
        // Fit on sony only, apply to pixel (does it transfer across cameras?).
        if groups.len() == 2 {
            let sony_lut = fit_luts(&groups[0].1);
            eprintln!("{}: with LUT fitted on {} MAE={:.4}", groups[1].0, groups[0].0, mae_with_lut(&groups[1].1, &sony_lut));
        }
    }
