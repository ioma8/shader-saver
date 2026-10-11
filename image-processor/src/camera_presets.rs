//! Black-box sampled color LUTs for OldRoll's Popular Camera cards.
//!
//! The original 17³ tables were sampled from the OldRoll editor's returned
//! previews. Five modes with visible plateaus have denser 33³ replacements.
//! Both assets store RGB outputs in b/g/r iteration order (R changes fastest).
//! The source charts, sampling method, and limitations are documented in
//! `docs/oldroll-camera-presets.md`.

const GRID: usize = 17;
const TABLE_BYTES: usize = GRID * GRID * GRID * 3;
const LUT_BYTES: &[u8] = include_bytes!("../assets/oldroll_popular_camera_luts_17.rgb8");
const DENSE_GRID: usize = 33;
const DENSE_TABLE_BYTES: usize = DENSE_GRID * DENSE_GRID * DENSE_GRID * 3;
const DENSE_LUT_BYTES: &[u8] = include_bytes!("../assets/oldroll_popular_camera_luts_33.rgb8");

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CameraPreset {
    Year2008,
    Fxs,
    Log,
    Film4008n,
    Ops,
    G7x2,
    Grx,
    Ccd,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CameraCategory {
    All,
    #[default]
    Popular,
    Film,
    Polaroid,
    Ccd,
    Portrait,
    Scenery,
    Street,
    Food,
    Sunlight,
    Indoor,
    Night,
}

pub const CATEGORIES: [CameraCategory; 12] = [
    CameraCategory::All,
    CameraCategory::Popular,
    CameraCategory::Film,
    CameraCategory::Polaroid,
    CameraCategory::Ccd,
    CameraCategory::Portrait,
    CameraCategory::Scenery,
    CameraCategory::Street,
    CameraCategory::Food,
    CameraCategory::Sunlight,
    CameraCategory::Indoor,
    CameraCategory::Night,
];

pub const POPULAR: [CameraPreset; 8] = [
    CameraPreset::Year2008,
    CameraPreset::Fxs,
    CameraPreset::Log,
    CameraPreset::Film4008n,
    CameraPreset::Ops,
    CameraPreset::G7x2,
    CameraPreset::Grx,
    CameraPreset::Ccd,
];

impl CameraCategory {
    pub const fn label(self) -> &'static str {
        match self {
            Self::All => "All",
            Self::Popular => "Popular",
            Self::Film => "Film",
            Self::Polaroid => "Polaroid",
            Self::Ccd => "CCD",
            Self::Portrait => "Portrait",
            Self::Scenery => "Scenery",
            Self::Street => "Street",
            Self::Food => "Food",
            Self::Sunlight => "Sunlight",
            Self::Indoor => "Indoor",
            Self::Night => "Night",
        }
    }

    pub fn presets(self) -> Vec<CameraPreset> {
        POPULAR
            .into_iter()
            .filter(|preset| self == Self::All || self == Self::Popular || preset.in_category(self))
            .collect()
    }
}

impl CameraPreset {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Year2008 => "2008",
            Self::Fxs => "FXS",
            Self::Log => "LOG",
            Self::Film4008n => "4008N",
            Self::Ops => "OPS",
            Self::G7x2 => "G7X2",
            Self::Grx => "GRX",
            Self::Ccd => "CCD",
        }
    }

    /// Index of the camera's catalog-default style in its visible choice list.
    pub const fn default_style(self) -> u8 {
        match self {
            Self::Log => 6,
            Self::Grx => 1,
            _ => 0,
        }
    }

    pub const fn styles(self) -> &'static [&'static str] {
        match self {
            Self::Fxs => &["FXS 1", "FXS 2", "FXS 3", "FXS 4"],
            Self::Log => &[
                "LUT 01", "LUT 02", "LUT 04", "LUT 05", "Original", "LUT 09", "LUT 10",
            ],
            Self::Film4008n => &["4008N 1", "4008N 2", "4008N 3"],
            Self::Ops => &["OPS 1", "OPS 2"],
            Self::Grx => &["GRX 1", "GRX 2"],
            _ => &[],
        }
    }

    fn table_index(self, style: u8) -> usize {
        match self {
            Self::Year2008 => 0,
            Self::Fxs => 1 + style.min(3) as usize,
            Self::Log => 5 + style.min(6) as usize,
            Self::Film4008n => 12 + style.min(2) as usize,
            Self::Ops => 15 + style.min(1) as usize,
            Self::G7x2 => 17,
            Self::Grx => 18 + style.min(1) as usize,
            Self::Ccd => 20,
        }
    }

    fn dense_table_index(self, style: u8) -> Option<usize> {
        match (self, style) {
            (Self::Fxs, 0) => Some(0),
            (Self::Film4008n, 1) => Some(1),
            (Self::Ops, 1) => Some(2),
            (Self::G7x2, _) => Some(3),
            (Self::Grx, 1) => Some(4),
            _ => None,
        }
    }

    const fn in_category(self, category: CameraCategory) -> bool {
        match category {
            CameraCategory::All | CameraCategory::Popular => true,
            CameraCategory::Film => matches!(self, Self::Ops),
            CameraCategory::Polaroid => false,
            CameraCategory::Ccd => matches!(self, Self::G7x2 | Self::Ccd),
            CameraCategory::Portrait => matches!(self, Self::Fxs | Self::G7x2 | Self::Ccd),
            CameraCategory::Scenery => matches!(self, Self::Ops | Self::Grx),
            CameraCategory::Street => matches!(self, Self::Grx),
            CameraCategory::Food | CameraCategory::Indoor => false,
            CameraCategory::Sunlight => matches!(self, Self::Fxs | Self::Ops | Self::Grx),
            CameraCategory::Night => matches!(self, Self::Log | Self::Film4008n | Self::G7x2),
        }
    }
}

/// Return the selected response on the processor's 33³ LUT grid.
/// Dense captures are used directly; the original tables are resampled.
pub fn lut(preset: CameraPreset, style: u8) -> Vec<f32> {
    if let Some(index) = preset.dense_table_index(style) {
        let start = index * DENSE_TABLE_BYTES;
        return DENSE_LUT_BYTES[start..start + DENSE_TABLE_BYTES]
            .iter()
            .map(|&channel| f32::from(channel) / 255.0)
            .collect();
    }

    let start = preset.table_index(style) * TABLE_BYTES;
    let sampled = &LUT_BYTES[start..start + TABLE_BYTES];
    let captured: Vec<f32> = sampled
        .iter()
        .map(|&channel| f32::from(channel) / 255.0)
        .collect();

    (0..33 * 33 * 33)
        .flat_map(|voxel| {
            let rgb = crate::processor::voxel_rgb(voxel, 33);
            crate::processor::sample_cube(&captured, GRID, rgb)
        })
        .collect()
}
