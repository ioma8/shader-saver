    use super::*;

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
            .map(|p| 0.2126 * f32::from(p[0]) + 0.7152 * f32::from(p[1]) + 0.0722 * f32::from(p[2]))
            .map(|x| x / 255.0)
            .collect();
        let clip = img.pixels().filter(|p| p[0] == 255 || p[1] == 255 || p[2] == 255).count() as f32 / n;
        v.sort_unstable_by(f32::total_cmp);
        let at = |p: f32| v[((v.len() - 1) as f32 * p) as usize];
        [at(0.05), at(0.5), at(0.95), clip, 0.0]
    }


    fn collect(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_dir() {
                collect(&path, out);
            } else if crate::imgload::is_raw(&path) {
                out.push(path);
            }
        }
    }

    /// Env-gated: `RAW_DATASET_DIR=datasets/raw-samples`. Develops every RAW in
    /// the tree and reports tone/clip against the camera's embedded JPEG.
    #[test]
    fn dataset_samples_develop_and_are_sane() {
        let Ok(dir) = std::env::var("RAW_DATASET_DIR") else {
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let mut paths = Vec::new();
        collect(&dir, &mut paths);
        paths.sort();
        let out_dir = std::path::Path::new("/tmp/raw-dataset");
        std::fs::create_dir_all(out_dir).unwrap();
        let mut failures = 0;
        for path in &paths {
            let mfr = path
                .parent()
                .and_then(|p| p.file_name())
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            let stem = path.file_stem().unwrap().to_string_lossy().to_string();
            let Some(image) = develop_raw_u16(path, 512) else {
                eprintln!("{mfr}/{stem}: DEVELOP FAILED");
                failures += 1;
                continue;
            };
            let dev = to8(&image);
            let d = tone(&dev);
            dev.save(out_dir.join(format!("{mfr}_{stem}_develop.png"))).unwrap();
            eprintln!(
                "{mfr}/{stem}: dev p5={:.3} p50={:.3} p95={:.3} clip={:.4}",
                d[0], d[1], d[2], d[3]
            );
            assert!(d[3] < 0.15, "{mfr}/{stem}: clipped {:.1}%", d[3] * 100.0);
        }
        assert!(!paths.is_empty(), "no RAW files found under {dir:?}");
        assert_eq!(failures, 0, "{failures} RAW files failed to develop");
    }
