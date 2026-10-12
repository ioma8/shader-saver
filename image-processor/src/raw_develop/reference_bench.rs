    use super::*;
    use std::path::PathBuf;

    fn to8(img: &Rgba16Image) -> RgbaImage {
        RgbaImage::from_fn(img.width(), img.height(), |x, y| {
            let p = img.get_pixel(x, y);
            image::Rgba([(p[0] >> 8) as u8, (p[1] >> 8) as u8, (p[2] >> 8) as u8, 255])
        })
    }

    fn tone(img: &RgbaImage) -> [f32; 5] {
        let n = (img.width() * img.height()) as f32;
        let mut v: Vec<f32> = img
            .pixels()
            .map(|p| (0.2126 * f32::from(p[0]) + 0.7152 * f32::from(p[1]) + 0.0722 * f32::from(p[2])) / 255.0)
            .collect();
        let clip = img.pixels().filter(|p| p[0] == 255 || p[1] == 255 || p[2] == 255).count() as f32 / n;
        v.sort_unstable_by(f32::total_cmp);
        // 5th, 25th (the subject band), 50th, 95th percentiles, plus clipping.
        [
            v[(v.len() - 1) / 20],
            v[(v.len() - 1) / 4],
            v[v.len() / 2],
            v[(v.len() - 1) * 19 / 20],
            clip,
        ]
    }

    fn mae(a: &RgbaImage, b: &RgbaImage) -> f32 {
        let a = image::imageops::resize(a, b.width(), b.height(), image::imageops::FilterType::Triangle);
        a.pixels()
            .zip(b.pixels())
            .flat_map(|(x, y)| (0..3).map(move |c| u8::abs_diff(x[c], y[c]) as f32 / 255.0))
            .sum::<f32>()
            / (b.width() * b.height() * 3) as f32
    }

    /// Why a reference cannot be used for MAE, if it cannot.
    fn reference_problem(r: &RgbaImage) -> Option<&'static str> {
        if r.width().max(r.height()) < 512 {
            return Some("preview smaller than 512 px");
        }
        let mut v: Vec<u8> = r.pixels().map(|p| p[0].max(p[1]).max(p[2])).collect();
        v.sort_unstable();
        if v[v.len() * 19 / 20] <= 12 {
            return Some("preview extracts black");
        }
        None
    }

    /// Sidecar JPEG (same stem), else the camera's sibling/embedded JPEG.
    fn reference(path: &Path) -> Option<(RgbaImage, &'static str)> {
        for ext in ["jpg", "JPG", "jpeg", "JPEG"] {
            let c = path.with_extension(ext);
            if c.exists() {
                if let Ok(i) = image::open(&c) {
                    // `image::open` ignores EXIF orientation, but the develop
                    // applies it, so orient the reference the same way.
                    let i = crate::imgload::orient_preview(i, crate::imgload::exif_orientation(&c));
                    return Some((i.to_rgba8(), "sidecar"));
                }
            }
        }
        crate::raw_develop::load_render_target(path, 0).map(|i| (i, "embedded"))
    }

    fn files(dir: &str, keep: impl Fn(&Path) -> bool) -> Vec<PathBuf> {
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
        let mut v = Vec::new();
        walk(Path::new(dir), &keep, &mut v);
        v.sort();
        v
    }

    fn bench(label: &str, paths: Vec<PathBuf>) -> Option<(usize, f32, f32)> {
        let mut maes = Vec::new();
        for p in &paths {
            let name = p.file_stem().unwrap().to_string_lossy().to_string();
            let Some(dev) = develop_raw_u16(p, 512).map(|x| to8(&x)) else {
                eprintln!("{label}/{name}: DEVELOP FAILED");
                continue;
            };
            let Some((r, kind)) = reference(p) else {
                eprintln!("{label}/{name}: no reference, skipped");
                continue;
            };
            let dim = format!("{}x{}", r.width(), r.height());
            if let Some(problem) = reference_problem(&r) {
                eprintln!("{label}/{name}: reference unusable ({kind} {dim}: {problem}), skipped");
                continue;
            }
            let r = image::imageops::thumbnail(&r, 512, 512);
            let (d, t) = (tone(&dev), tone(&r));
            let m = mae(&dev, &r);
            eprintln!(
                "{label}/{name}: ref={kind} {dim} | dev p25={:.3} p50={:.3} p95={:.3} clip={:.4} | ref p25={:.3} p50={:.3} p95={:.3} clip={:.4} | MAE={m:.4}",
                d[1], d[2], d[3], d[4], t[1], t[2], t[3], t[4]
            );
            maes.push(m);
        }
        if maes.is_empty() {
            eprintln!("== {label}: no usable references");
            return None;
        }
        maes.sort_unstable_by(f32::total_cmp);
        let mean = maes.iter().sum::<f32>() / maes.len() as f32;
        let median = maes[maes.len() / 2];
        eprintln!("== {label}: n={} mean MAE={mean:.4} median MAE={median:.4}", maes.len());
        Some((maes.len(), mean, median))
    }

    /// Env-gated: RAW_DATASET_DIR (sample set), RAW_SOOC_FOLDER (ARW + sidecar
    /// JPG), RAW_PIXEL_FOLDER (DNG + phone JPEG sibling).
    #[test]
    fn reference_bench_runs() {
        let mut ran = false;
        if let Ok(dir) = std::env::var("RAW_DATASET_DIR") {
            ran = true;
            bench("dataset", files(&dir, |_| true));
        }
        if let Ok(dirs) = std::env::var("RAW_SOOC_FOLDER") {
            ran = true;
            let mut all = Vec::new();
            for d in dirs.split(':').filter(|s| !s.is_empty()) {
                all.extend(files(d, |p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("arw"))));
            }
            bench("sony-arw", all);
        }
        if let Ok(dir) = std::env::var("RAW_PIXEL_FOLDER") {
            ran = true;
            bench(
                "pixel-dng",
                files(&dir, |p| {
                    p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.ends_with(".RAW-02.ORIGINAL.dng"))
                }),
            );
        }
        if !ran {
            return;
        }
    }
