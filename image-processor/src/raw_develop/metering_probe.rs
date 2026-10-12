//! Fits the metering rule: a small linear map from a frame's own statistics to
//! an exposure gain and a contrast offset, plus the leave-one-out and
//! out-of-sample checks and the coefficients to bake.
//!
//! ```bash
//! RAW_SOOC_FOLDER=<dir>[:<dir>...] RAW_PIXEL_FOLDER=<dir> RAW_DATASET_DIR=<dir> \
//!   cargo test --release metering_probe -- --nocapture
//! ```
//!
//! `RAW_SOOC_FOLDER` accepts several `:`-separated folders; they are pooled.

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
    if i.width().max(i.height()) < 512 {
        return None;
    }
    let mut v: Vec<u8> = i.pixels().map(|p| p[0].max(p[1]).max(p[2])).collect();
    v.sort_unstable();
    if v[v.len() * 19 / 20] <= 12 {
        return None;
    }
    Some(i)
}

fn walk(d: &Path, out: &mut Vec<std::path::PathBuf>, keep: &dyn Fn(&Path) -> bool) {
    for e in std::fs::read_dir(d).unwrap().filter_map(|e| e.ok()) {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out, keep);
        } else if crate::imgload::is_raw(&p) && keep(&p) {
            out.push(p);
        }
    }
}

/// (toned luma, reference luma, oracle gain, oracle offset, oracle gamma, extra raw features)
type Row = (Vec<f64>, Vec<f64>, f64, f64, f64, Vec<f64>);
type G = Vec<Row>;

fn mae_affine(dl: &[f64], rl: &[f64], a: f64, b: f64, g: f64) -> f64 {
    dl.iter().zip(rl).map(|(&x, &y)| ((a * 255.0 * (x / 255.0).powf(g) + b) - y).abs()).sum::<f64>() / dl.len() as f64
}

/// Best (gain, offset, gamma) for one frame: gamma by search, affine by least squares.
fn oracle(dl: &[f64], rl: &[f64]) -> (f64, f64, f64) {
    let np = dl.len() as f64;
    let mut best = (1.0, 0.0, 1.0, f64::MAX);
    let mut g = 0.50f64;
    while g <= 1.5001 {
        let (sd, sr, sdd, sdr) = dl.iter().zip(rl).fold((0.0, 0.0, 0.0, 0.0), |acc, (&x, &y)| {
            let xx = 255.0 * (x / 255.0).powf(g);
            (acc.0 + xx, acc.1 + y, acc.2 + xx * xx, acc.3 + xx * y)
        });
        let var = sdd - sd * sd / np;
        let a = if var.abs() > 1e-9 { (sdr - sd * sr / np) / var } else { 1.0 };
        let b = sr / np - a * sd / np;
        let m = mae_affine(dl, rl, a, b, g);
        if m < best.3 {
            best = (a, b, g, m);
        }
        g += 0.05;
    }
    (best.0, best.1, best.2)
}

fn u16_luma(img: &Rgba16Image) -> Vec<f64> {
    img.pixels()
        .map(|p| (0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2])) / 257.0)
        .collect()
}

fn percentiles(v: &[f64], qs: &[f64]) -> Vec<f64> {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    qs.iter().map(|&q| s[((s.len() - 1) as f64 * q) as usize] / 255.0).collect()
}

fn load(dir: &str, keep: &dyn Fn(&Path) -> bool) -> G {
    let mut files = Vec::new();
    walk(Path::new(dir), &mut files, keep);
    files.sort();
    files
        .iter()
        .filter_map(|p| {
            let decoded = decode_raw_u16(p, 256)?;
            // Raw features from the frame *before* our tone, at absolute level.
            let raw_luma = u16_luma(&decoded);
            let spec = std::env::var("RAW_FEAT").unwrap_or_else(|_| "p05,p50,p95,std,mean".to_string());
            let mean = raw_luma.iter().sum::<f64>() / raw_luma.len() as f64;
            let std = (raw_luma.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / raw_luma.len() as f64).sqrt();
            let mut extra = Vec::new();
            for name in spec.split(',').map(|s| s.trim()) {
                match name {
                    "p05" => extra.extend(percentiles(&raw_luma, &[0.05])),
                    "p50" => extra.extend(percentiles(&raw_luma, &[0.50])),
                    "p95" => extra.extend(percentiles(&raw_luma, &[0.95])),
                    "std" => extra.push(std / 255.0),
                    "mean" => extra.push(mean / 255.0),
                    _ => {}
                }
            }

            let mut toned = decoded;
            tone_u16_with(&mut toned, &TONE);
            let d = to8(&toned);
            let r = reference(p)?;
            let r = image::imageops::resize(&r, d.width(), d.height(), image::imageops::FilterType::Triangle);
            let dl = u16_luma(&toned);
            let rl: Vec<f64> = r.pixels().map(|p| 0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2])).collect();
            let (a, b, g) = oracle(&dl, &rl);
            Some((dl, rl, a, b, g, extra))
        })
        .collect()
}

/// Features the runtime can compute. The toned part matches `meter_features`;
/// the `extra` part is raw-decode statistics.
fn features(dl: &[f64], extra: &[f64]) -> Vec<f64> {
    let mut f = vec![1.0];
    f.extend(percentiles(dl, &[0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]));
    let mean = dl.iter().sum::<f64>() / dl.len() as f64;
    let std = (dl.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / dl.len() as f64).sqrt().max(1e-6);
    let skew = dl.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / dl.len() as f64;
    f.push(std / 255.0);
    f.push(skew);
    f.extend_from_slice(extra);
    f
}

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs())).unwrap();
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        for k in col..n {
            a[col][k] /= d;
        }
        b[col] /= d;
        for r in 0..n {
            if r != col {
                let f = a[r][col];
                for k in col..n {
                    a[r][k] -= f * a[col][k];
                }
                b[r] -= f * b[col];
            }
        }
    }
    b
}

fn ridge(rows: &[Vec<f64>], y: &[f64], lambda: f64) -> Vec<f64> {
    let d = rows[0].len();
    let mut xtx = vec![vec![0.0; d]; d];
    let mut xty = vec![0.0; d];
    for (r, &yy) in rows.iter().zip(y) {
        for i in 0..d {
            for j in 0..d {
                xtx[i][j] += r[i] * r[j];
            }
            xty[i] += r[i] * yy;
        }
    }
    for i in 0..d {
        xtx[i][i] += lambda;
    }
    solve(xtx, xty)
}

fn dot(x: &[f64], beta: &[f64]) -> f64 {
    x.iter().zip(beta).map(|(a, b)| a * b).sum()
}

fn base_mae(g: &G) -> f64 {
    let mut s = 0.0;
    let mut n = 0.0;
    for (dl, rl, _, _, _, _) in g {
        for (x, y) in dl.iter().zip(rl) {
            s += (x - y).abs();
            n += 1.0;
        }
    }
    s / n
}

fn mae_with(g: &G, pick: &dyn Fn(&[f64]) -> (f64, f64, f64)) -> f64 {
    let mut s = 0.0;
    let mut n = 0.0;
    for (dl, rl, _, _, _, extra) in g {
        let (a, b, gm) = pick(&features(dl, extra));
        for (x, y) in dl.iter().zip(rl) {
            s += ((a * 255.0 * (x / 255.0).powf(gm) + b) - y).abs();
            n += 1.0;
        }
    }
    s / n
}

fn loo(g: &G, lambda: f64) -> f64 {
    let feats: Vec<Vec<f64>> = g.iter().map(|(dl, _, _, _, _, extra)| features(dl, extra)).collect();
    let ya: Vec<f64> = g.iter().map(|x| x.2).collect();
    let yb: Vec<f64> = g.iter().map(|x| x.3).collect();
    let yg: Vec<f64> = g.iter().map(|x| x.4).collect();
    let mut s = 0.0;
    let mut n = 0.0;
    for i in 0..g.len() {
        let idx: Vec<usize> = (0..g.len()).filter(|&j| j != i).collect();
        let rows: Vec<Vec<f64>> = idx.iter().map(|&j| feats[j].clone()).collect();
        let fa = |v: &[usize], get: &dyn Fn(usize) -> f64| v.iter().map(|&j| get(j)).collect::<Vec<_>>();
        let beta_a = ridge(&rows, &fa(&idx, &|j| ya[j]), lambda);
        let beta_b = ridge(&rows, &fa(&idx, &|j| yb[j]), lambda);
        let beta_g = ridge(&rows, &fa(&idx, &|j| yg[j]), lambda);
        let (a, b, gm) = (dot(&feats[i], &beta_a), dot(&feats[i], &beta_b), dot(&feats[i], &beta_g));
        for (x, y) in g[i].0.iter().zip(&g[i].1) {
            s += ((a * 255.0 * (x / 255.0).powf(gm) + b) - y).abs();
            n += 1.0;
        }
    }
    s / n
}

#[test]
fn metering_probe() {
    let mut groups: Vec<(&str, G)> = Vec::new();
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
    for (name, g) in &groups {
        let oracle = {
            let mut s = 0.0;
            let mut n = 0.0;
            for (dl, rl, a, b, _, _) in g {
                for (x, y) in dl.iter().zip(rl) {
                    s += ((a * x + b) - y).abs();
                    n += 1.0;
                }
            }
            s / n
        };
        eprintln!("{name}: n={} current={:.2} oracle={:.2} LOO={:.2} /255", g.len(), base_mae(g), oracle, loo(g, 0.05));
    }
    if groups.len() == 3 {
        let train: G = groups[0].1.iter().chain(&groups[1].1).cloned().collect();
        let rows: Vec<Vec<f64>> = train.iter().map(|(dl, _, _, _, _, extra)| features(dl, extra)).collect();
        let beta_a = ridge(&rows, &train.iter().map(|x| x.2).collect::<Vec<_>>(), 0.05);
        let beta_b = ridge(&rows, &train.iter().map(|x| x.3).collect::<Vec<_>>(), 0.05);
        let beta_g = ridge(&rows, &train.iter().map(|x| x.4).collect::<Vec<_>>(), 0.05);
        let mixed = &groups[2].1;
        eprintln!(
            "transfer: fit on sony+pixel -> mixed n={} current={:.2} with rule={:.2} /255",
            mixed.len(),
            base_mae(mixed),
            mae_with(mixed, &|f| (dot(f, &beta_a), dot(f, &beta_b), dot(f, &beta_g)))
        );
    }
    // Bake: fit on every available pair and print Rust arrays.
    let all: G = groups.iter().flat_map(|(_, g)| g.iter().cloned()).collect();
    let rows: Vec<Vec<f64>> = all.iter().map(|(dl, _, _, _, _, extra)| features(dl, extra)).collect();
    let ba = ridge(&rows, &all.iter().map(|x| x.2).collect::<Vec<_>>(), 0.05);
    let bb = ridge(&rows, &all.iter().map(|x| x.3).collect::<Vec<_>>(), 0.05);
    let bg = ridge(&rows, &all.iter().map(|x| x.4).collect::<Vec<_>>(), 0.05);
    let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(", ");
    eprintln!("FEATURES = {} (intercept + 9 toned + {} raw)", ba.len(), ba.len() - 10);
    eprintln!("METER_GAIN: [f32; {}] = [{}];", ba.len(), fmt(&ba));
    eprintln!("METER_OFFSET: [f32; {}] = [{}];", bb.len(), fmt(&bb));
    eprintln!("METER_GAMMA: [f32; {}] = [{}];", bg.len(), fmt(&bg));
}
