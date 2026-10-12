    use super::*;

    fn to8(img: &Rgba16Image) -> RgbaImage {
        RgbaImage::from_fn(img.width(), img.height(), |x, y| {
            let p = img.get_pixel(x, y);
            image::Rgba([(p[0] >> 8) as u8, (p[1] >> 8) as u8, (p[2] >> 8) as u8, 255])
        })
    }

    #[test]
    fn decompose_errors() {
        let Ok(dir) = std::env::var("RAW_SOOC_FOLDER") else { return };
        let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("arw")))
            .collect();
        files.sort();
        let (mut px, mut images) = (0.0f64, 0.0f64);
        let (mut ch_abs, mut ch_signed) = ([0.0f64; 3], [0.0f64; 3]);
        let (mut low, mut high) = (0.0f64, 0.0f64);
        let (mut bin_sum, mut bin_cnt) = ([0.0f64; 4], [0.0f64; 4]);
        let mut affine = 0.0f64;
        for f in &files {
            let jpg = f.with_extension("JPG");
            let Ok(r) = image::open(&jpg) else { continue };
            let r = crate::imgload::orient_preview(r, crate::imgload::exif_orientation(&jpg)).to_rgba8();
            let Some(mut d) = decode_raw_u16(f, 384) else { continue };
            tone_u16_with(&mut d, &TONE);
            let d = to8(&d);
            let r = image::imageops::resize(&r, d.width(), d.height(), image::imageops::FilterType::Triangle);
            let (w, h) = (d.width() as usize, d.height() as usize);
            px += (w * h) as f64;
            images += 1.0;
            let dl: Vec<f32> = d.pixels().map(|p| 0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32).collect();
            let rl: Vec<f32> = r.pixels().map(|p| 0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32).collect();
            let db = box_blur(&dl, w, h, 6);
            let rb = box_blur(&rl, w, h, 6);
            for (i, (p, q)) in d.pixels().zip(r.pixels()).enumerate() {
                for k in 0..3 {
                    let e = f64::from(p[k]) - f64::from(q[k]);
                    ch_abs[k] += e.abs();
                    ch_signed[k] += e;
                }
                low += f64::from((db[i] - rb[i]).abs());
                high += f64::from(((dl[i] - db[i]) - (rl[i] - rb[i])).abs());
                let b = ((rl[i] / 255.0 * 4.0) as usize).min(3);
                bin_sum[b] += f64::from((dl[i] - rl[i]).abs());
                bin_cnt[b] += 1.0;
            }
            // Best per-image linear luma correction (exposure + contrast).
            let np = (w * h) as f64;
            let (sd, sr, sdd, sdr) = dl.iter().zip(&rl).fold((0.0, 0.0, 0.0, 0.0), |a, (&x, &y)| {
                (a.0 + x as f64, a.1 + y as f64, a.2 + (x as f64) * (x as f64), a.3 + (x as f64) * (y as f64))
            });
            let var = sdd - sd * sd / np;
            let a = if var.abs() > 1e-9 { (sdr - sd * sr / np) / var } else { 1.0 };
            let bb = sr / np - a * sd / np;
            affine += dl.iter().zip(&rl).map(|(&x, &y)| ((a * x as f64 + bb) - y as f64).abs()).sum::<f64>();
        }
        let channel_mae = |k: usize| ch_abs[k] / px;
        eprintln!("images={images:.0} pixels={px:.0}");
        eprintln!("luma MAE overall = {:.2} / 255", (ch_abs[0] + ch_abs[1] + ch_abs[2]) / (px * 3.0));
        let names = ["R", "G", "B"];
        for k in 0..3 {
            eprintln!(
                "  {name}: MAE={:.2} bias={:+.2}",
                channel_mae(k),
                ch_signed[k] / px,
                name = names[k]
            );
        }
        eprintln!("low-frequency  MAE = {:.2} / 255  (tone, colour)", low / px);
        eprintln!("high-frequency MAE = {:.2} / 255  (detail, sharpness)", high / px);
        eprintln!("after best per-image luma affine = {:.2} / 255 (exposure+contrast matched)", affine / px);
        let zones = ["shadows", "lower-mid", "upper-mid", "highlights"];
        for b in 0..4 {
            if bin_cnt[b] > 0.0 {
                eprintln!("  {z}: MAE={:.2} ({:.0}% of pixels)", bin_sum[b] / bin_cnt[b], bin_cnt[b] / px * 100.0, z = zones[b]);
            }
        }
    }
