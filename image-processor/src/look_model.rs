//! Reference-conditioned photographic look model.
//!
//! The network predicts one constrained photographic transform from robust
//! target/reference profiles. It never generates pixels: a monotonic tone
//! curve and bounded color controls are baked into the app's existing LUT.

use crate::processor::{LookProfile, LookTransfer, LOOK_ANCHORS, REGION_COUNT};
use std::path::Path;

// Style-only measurements. Scene-presence features such as skin, foliage,
// sky and hue evidence deliberately stay out: they made the predicted grade
// change with the target's subject matter instead of the reference's look.
const PROFILE_FEATURES: usize = LOOK_ANCHORS.len() + REGION_COUNT * 2 + 1 + 8 + 16;
const INPUTS: usize = PROFILE_FEATURES * 3;
const HIDDEN: usize = 48;
const OUTPUTS: usize = LOOK_ANCHORS.len() + REGION_COUNT * 2 + 8 + 8;
const TONE_RANGE: f32 = 0.30;
const CAST_RANGE: f32 = 0.06;
const CHROMA_RANGE: f32 = 0.55;
const HUE_RANGE: f32 = 15.0;

pub struct LookModel {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingExample {
    pub current: LookProfile,
    pub reference: LookProfile,
    pub desired: LookProfile,
}

pub fn load_examples(path: &Path) -> Vec<TrainingExample> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|json| serde_json::from_str(&json).ok())
        .unwrap_or_default()
}

pub fn save_examples(path: &Path, examples: &[TrainingExample]) {
    if let Ok(json) = serde_json::to_string(examples) {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(path, json);
    }
}

impl LookModel {
    /// Train the compact model from synthetic profile/transfer pairs.
    pub fn train() -> Self {
        let mut net = Self {
            w1: vec![0.0; HIDDEN * INPUTS],
            b1: vec![0.0; HIDDEN],
            w2: vec![0.0; OUTPUTS * HIDDEN],
            b2: vec![0.0; OUTPUTS],
        };
        let mut rng = 0x7f4a_7c15u32;
        for v in &mut net.w1 {
            *v = (next_rand(&mut rng) * 2.0 - 1.0) * 0.08;
        }
        for v in &mut net.w2 {
            *v = (next_rand(&mut rng) * 2.0 - 1.0) * 0.08;
        }

        net.train_synthetic();
        net
    }

    pub fn train_with_examples(examples: &[TrainingExample]) -> Self {
        let mut net = Self::train();
        if examples.is_empty() {
            return net;
        }

        // The universal network remains frozen. GUI lessons fit one
        // regularized affine calibration per photographic output. This
        // teaches systematic under/over-shoot (exposure, contrast, WB,
        // saturation and hue) without letting a small personal set rewrite
        // the content-to-style representation.
        let mut sx = [0.0f32; OUTPUTS];
        let mut sy = [0.0f32; OUTPUTS];
        let mut sxx = [0.0f32; OUTPUTS];
        let mut sxy = [0.0f32; OUTPUTS];
        for example in examples {
            let input = profile_pair(&example.current, &example.reference);
            let labels = transfer_vector(&example.current, &example.desired);
            let mut hidden = [0.0; HIDDEN];
            let mut output = [0.0; OUTPUTS];
            net.forward(&input, &mut hidden, &mut output);
            for o in 0..OUTPUTS {
                sx[o] += output[o];
                sy[o] += labels[o];
                sxx[o] += output[o] * output[o];
                sxy[o] += output[o] * labels[o];
            }
        }

        let n = examples.len() as f32;
        const REGULARIZATION: f32 = 0.25;
        for o in 0..OUTPUTS {
            // Least squares for y = scale*x + bias, regularized toward the
            // untouched model (scale=1, bias=0).
            let a = sxx[o] + REGULARIZATION;
            let b = sx[o];
            let c = n + REGULARIZATION;
            let p = sxy[o] + REGULARIZATION;
            let q = sy[o];
            let determinant = (a * c - b * b).max(1e-6);
            let scale = ((p * c - b * q) / determinant).clamp(0.65, 1.35);
            let bias = ((a * q - b * p) / determinant).clamp(-0.30, 0.30);
            for h in 0..HIDDEN {
                net.w2[o * HIDDEN + h] *= scale;
            }
            net.b2[o] = net.b2[o] * scale + bias;
        }
        net
    }

    fn train_synthetic(&mut self) {
        // Two unrelated base photographs receive independent known grades.
        // The desired image is the target content under the reference grade.
        // This prevents the model from learning that scene content itself is
        // the look, while still providing unlimited exact supervision.
        let mut rng = 0x7f4a_7c15u32;
        for step in 0..16_000 {
            let target_base = random_profile(&mut rng);
            let reference_base = random_profile(&mut rng);
            let source_grade = random_transfer(&mut rng);
            let reference_grade = if step % 5 == 0 {
                source_grade.clone()
            } else {
                random_transfer(&mut rng)
            };
            let current = transformed(&target_base, &source_grade);
            let reference = transformed(&reference_base, &reference_grade);
            let desired = transformed(&target_base, &reference_grade);
            let input = feature_pair(&current, &reference);
            let mut hidden = vec![0.0; HIDDEN];
            let mut output = vec![0.0; OUTPUTS];
            self.forward(&input, &mut hidden, &mut output);

            let lr = 0.005 * (1.0 - step as f32 / 16_000.0 * 0.7);
            let labels = transfer_vector_features(&current, &desired);
            self.update_with_hidden(&input, &hidden, &output, &labels, lr, true);
        }
    }

    fn update_with_hidden(
        &mut self,
        input: &[f32],
        hidden: &[f32],
        output: &[f32],
        labels: &[f32],
        lr: f32,
        train_encoder: bool,
    ) {
        let mut grad_out = vec![0.0; OUTPUTS];
        for i in 0..OUTPUTS {
            grad_out[i] = (output[i] - labels[i]).clamp(-1.0, 1.0);
        }
        for o in 0..OUTPUTS {
            for h in 0..HIDDEN {
                self.w2[o * HIDDEN + h] -= lr * grad_out[o] * hidden[h];
            }
            self.b2[o] -= lr * grad_out[o];
        }
        if !train_encoder {
            return;
        }
        for h in 0..HIDDEN {
            let mut grad = 0.0;
            for o in 0..OUTPUTS {
                grad += grad_out[o] * self.w2[o * HIDDEN + h];
            }
            grad *= 1.0 - hidden[h] * hidden[h];
            for i in 0..INPUTS {
                self.w1[h * INPUTS + i] -= lr * grad * input[i];
            }
            self.b1[h] -= lr * grad;
        }
    }

    pub fn predict(&self, current: &LookProfile, reference: &LookProfile) -> LookTransfer {
        let input = profile_pair(current, reference);
        let mut hidden = vec![0.0; HIDDEN];
        let mut output = vec![0.0; OUTPUTS];
        self.forward(&input, &mut hidden, &mut output);
        transfer_from_vector(&output)
    }

    fn forward(&self, input: &[f32], hidden: &mut [f32], output: &mut [f32]) {
        for h in 0..HIDDEN {
            let mut sum = self.b1[h];
            for i in 0..INPUTS {
                sum += self.w1[h * INPUTS + i] * input[i];
            }
            hidden[h] = sum.tanh();
        }
        for o in 0..OUTPUTS {
            let mut sum = self.b2[o];
            for h in 0..HIDDEN {
                sum += self.w2[o * HIDDEN + h] * hidden[h];
            }
            output[o] = sum;
        }
    }
}

fn transfer_from_vector(output: &[f32]) -> LookTransfer {
    let tone_delta = constrained_tone(&output[..LOOK_ANCHORS.len()]);
    let mut cast_delta = [[0.0; 2]; REGION_COUNT];
    let mut at = LOOK_ANCHORS.len();
    for band in &mut cast_delta {
        band[0] = (output[at] * CAST_RANGE).clamp(-CAST_RANGE, CAST_RANGE);
        band[1] = (output[at + 1] * CAST_RANGE).clamp(-CAST_RANGE, CAST_RANGE);
        at += 2;
    }
    let mut hue_chroma_scale = [1.0; 8];
    for value in &mut hue_chroma_scale {
        *value = (1.0 + output[at] * CHROMA_RANGE).clamp(0.65, 1.45);
        at += 1;
    }
    let mut hue_rotate = [0.0; 8];
    for value in &mut hue_rotate {
        *value = (output[at] * HUE_RANGE).clamp(-HUE_RANGE, HUE_RANGE);
        at += 1;
    }
    LookTransfer {
        tone_delta,
        cast_delta,
        hue_chroma_scale,
        hue_rotate,
    }
}

fn profile_pair(current: &LookProfile, reference: &LookProfile) -> Vec<f32> {
    feature_pair(&profile_features(current), &profile_features(reference))
}

fn transfer_vector(current: &LookProfile, desired: &LookProfile) -> Vec<f32> {
    transfer_vector_features(&profile_features(current), &profile_features(desired))
}

fn transfer_vector_features(current: &[f32], desired: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(OUTPUTS);
    for &x in &LOOK_ANCHORS {
        let x = x as f32;
        let y = percentile_mapping(x, &current[..5], &desired[..5]);
        out.push(((y - x) / TONE_RANGE).clamp(-1.0, 1.0));
    }
    for i in 5..11 {
        out.push(((desired[i] - current[i]) / CAST_RANGE).clamp(-1.0, 1.0));
    }
    for i in 0..8 {
        out.push(
            (((desired[12 + i] / current[12 + i].max(1e-4)) - 1.0) / CHROMA_RANGE).clamp(-1.0, 1.0),
        );
    }
    for i in 0..8 {
        let a = current[21 + i * 2].atan2(current[20 + i * 2]).to_degrees();
        let b = desired[21 + i * 2].atan2(desired[20 + i * 2]).to_degrees();
        out.push((((b - a + 540.0).rem_euclid(360.0) - 180.0) / HUE_RANGE).clamp(-1.0, 1.0));
    }
    out
}

fn percentile_mapping(x: f32, current: &[f32], desired: &[f32]) -> f32 {
    let mut previous = (0.0, 0.0);
    for (&cx, &dy) in current.iter().zip(desired) {
        let next = (cx.max(previous.0 + 1e-4), dy.max(previous.1));
        if x <= next.0 {
            let t = ((x - previous.0) / (next.0 - previous.0)).clamp(0.0, 1.0);
            return previous.1 + (next.1 - previous.1) * t;
        }
        previous = next;
    }
    let t = ((x - previous.0) / (1.0 - previous.0).max(1e-4)).clamp(0.0, 1.0);
    previous.1 + (1.0 - previous.1) * t
}

fn constrained_tone(raw: &[f32]) -> [f32; LOOK_ANCHORS.len()] {
    let mut mapped = [0.0; LOOK_ANCHORS.len()];
    for i in 0..mapped.len() {
        mapped[i] =
            (LOOK_ANCHORS[i] as f32 + raw[i].clamp(-1.0, 1.0) * TONE_RANGE).clamp(0.01, 0.99);
    }
    const MIN_GAP: f32 = 0.025;
    for i in 1..mapped.len() {
        mapped[i] = mapped[i].max(mapped[i - 1] + MIN_GAP);
    }
    mapped[mapped.len() - 1] = mapped[mapped.len() - 1].min(0.99);
    for i in (0..mapped.len() - 1).rev() {
        mapped[i] = mapped[i].min(mapped[i + 1] - MIN_GAP);
    }
    let mut delta = [0.0; LOOK_ANCHORS.len()];
    for i in 0..delta.len() {
        delta[i] = mapped[i] - LOOK_ANCHORS[i] as f32;
    }
    delta
}

fn feature_pair(current: &[f32], reference: &[f32]) -> Vec<f32> {
    let current = normalized_features(current);
    let reference = normalized_features(reference);
    let mut out = Vec::with_capacity(INPUTS);
    out.extend(&current);
    out.extend(&reference);
    out.extend(current.iter().zip(&reference).map(|(a, b)| b - a));
    out
}

fn normalized_features(raw: &[f32]) -> Vec<f32> {
    raw.iter()
        .enumerate()
        .map(|(i, &value)| match i {
            0..=4 => value * 2.0 - 1.0,
            5..=10 => (value / 0.10).clamp(-1.5, 1.5),
            11 => (value / 0.20).clamp(0.0, 2.0) - 1.0,
            12..=19 => (value / 0.20).clamp(0.0, 2.0) - 1.0,
            _ => value.clamp(-1.0, 1.0),
        })
        .collect()
}

fn profile_features(profile: &LookProfile) -> Vec<f32> {
    let mut out = Vec::with_capacity(PROFILE_FEATURES);
    out.extend(profile.tone);
    for cast in profile.cast {
        out.extend(cast);
    }
    out.push(profile.chroma);
    out.extend(profile.hue_chroma);
    for axis in profile.hue_axis {
        out.extend(axis);
    }
    out
}

fn next_rand(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    (*state as f32 / u32::MAX as f32).clamp(0.0, 1.0)
}

fn random_profile(rng: &mut u32) -> Vec<f32> {
    let black = next_rand(rng) * 0.12;
    let white = 0.78 + next_rand(rng) * 0.22;
    let gamma = 0.75 + next_rand(rng) * 0.65;
    let mut out = Vec::with_capacity(PROFILE_FEATURES);
    out.extend(LOOK_ANCHORS.map(|p| black + (white - black) * (p as f32).powf(gamma)));
    out.extend((0..6).map(|_| (next_rand(rng) - 0.5) * 0.10));
    out.push(0.025 + next_rand(rng) * 0.16);
    out.extend((0..8).map(|_| 0.02 + next_rand(rng) * 0.20));
    for _ in 0..8 {
        let angle = next_rand(rng) * std::f32::consts::TAU;
        out.extend([angle.cos(), angle.sin()]);
    }
    out
}

fn random_transfer(rng: &mut u32) -> Vec<f32> {
    let mut out = Vec::with_capacity(OUTPUTS);
    let exposure = (next_rand(rng) - 0.5) * 0.30;
    let contrast = 0.78 + next_rand(rng) * 0.50;
    let shadows = (next_rand(rng) - 0.5) * 0.10;
    let highlights = (next_rand(rng) - 0.5) * 0.10;
    let raw_tone: Vec<f32> = LOOK_ANCHORS
        .iter()
        .map(|&anchor| {
            let x = anchor as f32;
            let y = 0.5
                + (x - 0.5) * contrast
                + exposure
                + shadows * (1.0 - x).powi(2)
                + highlights * x.powi(2);
            (y.clamp(0.01, 0.99) - x) / TONE_RANGE
        })
        .collect();
    out.extend(constrained_tone(&raw_tone));

    let temperature = (next_rand(rng) - 0.5) * 0.055;
    let tint = (next_rand(rng) - 0.5) * 0.045;
    for band in 0..3 {
        let split = (band as f32 - 1.0) * (next_rand(rng) - 0.5) * 0.018;
        out.extend([temperature + split, tint - split * 0.5]);
    }
    let saturation = 0.72 + next_rand(rng) * 0.56;
    let variation = (next_rand(rng) - 0.5) * 0.20;
    let phase = next_rand(rng) * std::f32::consts::TAU;
    out.extend((0..8).map(|i| {
        (saturation + variation * (phase + i as f32 * std::f32::consts::TAU / 8.0).sin())
            .clamp(0.65, 1.45)
    }));
    let rotation = (next_rand(rng) - 0.5) * 16.0;
    let rotation_variation = (next_rand(rng) - 0.5) * 8.0;
    out.extend((0..8).map(|i| {
        (rotation + rotation_variation * (phase + i as f32 * std::f32::consts::TAU / 8.0).sin())
            .clamp(-HUE_RANGE, HUE_RANGE)
    }));
    out
}

fn transformed(base: &[f32], transfer: &[f32]) -> Vec<f32> {
    let mut out = base.to_vec();
    for i in 0..5 {
        out[i] = (out[i] + sample_tone_delta(&transfer[..5], out[i])).clamp(0.0, 1.0);
    }
    for i in 0..6 {
        out[5 + i] += transfer[5 + i];
    }
    let mean_scale = transfer[11..19].iter().sum::<f32>() / 8.0;
    out[11] *= mean_scale;
    for i in 0..8 {
        out[12 + i] *= transfer[11 + i];
    }
    for i in 0..8 {
        let (s, c) = transfer[19 + i].to_radians().sin_cos();
        let a = out[20 + i * 2];
        let b = out[21 + i * 2];
        out[20 + i * 2] = a * c - b * s;
        out[21 + i * 2] = a * s + b * c;
    }
    out
}

fn sample_tone_delta(delta: &[f32], x: f32) -> f32 {
    if x <= LOOK_ANCHORS[0] as f32 {
        return delta[0];
    }
    for i in 0..LOOK_ANCHORS.len() - 1 {
        let a = LOOK_ANCHORS[i] as f32;
        let b = LOOK_ANCHORS[i + 1] as f32;
        if x <= b {
            let t = (x - a) / (b - a);
            return delta[i] + (delta[i + 1] - delta[i]) * t;
        }
    }
    delta[delta.len() - 1]
}

#[cfg(test)]
mod tests {
    use super::{
        constrained_tone, load_examples, profile_pair, transfer_vector, LookModel, HIDDEN, OUTPUTS,
    };
    use crate::processor::LookProfile;

    fn lesson_error(model: &LookModel, examples: &[super::TrainingExample]) -> f32 {
        let mut sum = 0.0;
        let mut count = 0;
        for example in examples {
            let input = profile_pair(&example.current, &example.reference);
            let labels = transfer_vector(&example.current, &example.desired);
            let mut hidden = vec![0.0; HIDDEN];
            let mut output = vec![0.0; OUTPUTS];
            model.forward(&input, &mut hidden, &mut output);
            sum += output
                .iter()
                .zip(labels)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>();
            count += OUTPUTS;
        }
        sum / count.max(1) as f32
    }

    #[test]
    fn manual_lessons_improve_the_model() {
        let Some(dir) = crate::app_dir() else { return };
        let examples = load_examples(&dir.join("look-model-examples.json"));
        if examples.is_empty() {
            return;
        }
        let base = LookModel::train();
        let before = lesson_error(&base, &examples);
        let adapted = LookModel::train_with_examples(&examples);
        let after = lesson_error(&adapted, &examples);
        let (train, holdout): (Vec<_>, Vec<_>) = examples
            .iter()
            .cloned()
            .enumerate()
            .partition(|(index, _)| index % 5 != 0);
        let train: Vec<_> = train.into_iter().map(|(_, example)| example).collect();
        let holdout: Vec<_> = holdout.into_iter().map(|(_, example)| example).collect();
        let held_before = lesson_error(&base, &holdout);
        let held_after = lesson_error(&LookModel::train_with_examples(&train), &holdout);
        println!("manual lesson MSE: {before:.6} -> {after:.6}; held out: {held_before:.6} -> {held_after:.6}");
        assert!(
            after < before * 0.90,
            "manual teaching must materially reduce lesson error"
        );
        assert!(
            held_after < held_before * 0.95,
            "manual teaching must improve unseen lessons, not merely memorize them"
        );
    }

    #[test]
    fn trained_model_predicts_finite_transfer() {
        let image = image::RgbaImage::from_fn(32, 32, |x, y| {
            image::Rgba([(x * 7) as u8, (y * 7) as u8, 96, 255])
        });
        let profile = LookProfile::measure(image.as_raw(), 32, 32, &[]).unwrap();
        let model = LookModel::train();
        let transfer = model.predict(&profile, &profile);
        assert!(transfer
            .tone_delta
            .iter()
            .chain(transfer.cast_delta.iter().flat_map(|v| v.iter()))
            .chain(transfer.hue_chroma_scale.iter())
            .chain(transfer.hue_rotate.iter())
            .all(|v| v.is_finite()));
        assert!(transfer
            .cast_delta
            .iter()
            .flatten()
            .all(|v| v.abs() <= super::CAST_RANGE));
        assert!(transfer
            .hue_chroma_scale
            .iter()
            .all(|v| (0.65..=1.45).contains(v)));
        assert!(transfer
            .hue_rotate
            .iter()
            .all(|v| v.abs() <= super::HUE_RANGE));
        let mapped: Vec<f32> = crate::processor::LOOK_ANCHORS
            .iter()
            .zip(transfer.tone_delta)
            .map(|(x, delta)| *x as f32 + delta)
            .collect();
        assert!(mapped.windows(2).all(|pair| pair[1] > pair[0]));
        let tone_error = transfer.tone_delta.iter().map(|v| v.abs()).sum::<f32>() / 5.0;
        let cast_error = transfer
            .cast_delta
            .iter()
            .flatten()
            .map(|v| v.abs())
            .sum::<f32>()
            / 6.0;
        let chroma_error = transfer
            .hue_chroma_scale
            .iter()
            .map(|v| (v - 1.0).abs())
            .sum::<f32>()
            / 8.0;
        let hue_error = transfer.hue_rotate.iter().map(|v| v.abs()).sum::<f32>() / 8.0;
        assert!(tone_error < 0.04, "identity tone error {tone_error}");
        assert!(cast_error < 0.015, "identity cast error {cast_error}");
        assert!(chroma_error < 0.08, "identity chroma error {chroma_error}");
        assert!(hue_error < 3.0, "identity hue error {hue_error}");
    }

    #[test]
    fn predicted_tone_curve_is_monotonic() {
        let delta = constrained_tone(&[1.0, -1.0, 1.0, -1.0, 1.0]);
        let mapped: Vec<f32> = crate::processor::LOOK_ANCHORS
            .iter()
            .enumerate()
            .map(|(i, x)| *x as f32 + delta[i])
            .collect();
        assert!(mapped.windows(2).all(|pair| pair[1] > pair[0]));
        assert!(mapped.iter().all(|v| (0.0..=1.0).contains(v)));
    }
}
