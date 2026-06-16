// Unified image loading: standard formats via the `image` crate, camera RAW
// formats via rawloader/imagepipe (pure-Rust decode + demosaic).
use jpgfromraw::FindJpegType;
use std::path::Path;

pub const STD_EXTS: [&str; 8] = ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "gif"];
pub const RAW_EXTS: [&str; 19] = [
    "dng", "cr2", "cr3", "nef", "arw", "orf", "rw2", "raf", "pef", "srw", "mrw", "kdc", "dcr",
    "erf", "mef", "mos", "nrw", "3fr", "x3f",
];

fn ext_of(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
}

pub fn is_raw(path: &Path) -> bool {
    ext_of(path)
        .map(|e| RAW_EXTS.contains(&e.as_str()))
        .unwrap_or(false)
}

pub fn is_supported(path: &Path) -> bool {
    ext_of(path)
        .map(|e| STD_EXTS.contains(&e.as_str()) || RAW_EXTS.contains(&e.as_str()))
        .unwrap_or(false)
}

pub fn all_exts() -> Vec<&'static str> {
    STD_EXTS.iter().chain(RAW_EXTS.iter()).copied().collect()
}

pub fn load_preview_rgba(path: &Path, max_dim: u32) -> Option<image::RgbaImage> {
    // For grid thumbnails use the smallest embedded JPEG — already tiny, no full decode needed.
    let find_type = if max_dim > 0 { FindJpegType::Smallest } else { FindJpegType::Largest };
    let bytes = pollster::block_on(jpgfromraw::process_file_bytes(path, find_type)).ok()?;
    let img = image::load_from_memory(&bytes).ok()?;
    let img = if max_dim > 0 { img.thumbnail(max_dim, max_dim) } else { img };
    Some(img.to_rgba8())
}

pub fn load_edit_rgba(path: &Path) -> Option<(image::RgbaImage, bool)> {
    if is_raw(path) {
        let bytes = pollster::block_on(jpgfromraw::process_file_bytes(path, FindJpegType::Largest)).ok()?;
        let img = image::load_from_memory(&bytes).ok()?.to_rgba8();
        return Some((img, true));
    }
    load_rgba(path, 0).map(|img| (img, false))
}

// Decode to RGBA8. max_dim = 0 loads full resolution; otherwise the result
// fits within max_dim×max_dim (used for thumbnails — RAW decode honors the
// limit during demosaic, which is much faster than full-size decode).
pub fn load_rgba(path: &Path, max_dim: u32) -> Option<image::RgbaImage> {
    if is_raw(path) {
        let dec = imagepipe::simple_decode_8bit(path, max_dim as usize, max_dim as usize).ok()?;
        let mut rgba = Vec::with_capacity(dec.width * dec.height * 4);
        for px in dec.data.chunks_exact(3) {
            rgba.extend_from_slice(px);
            rgba.push(255);
        }
        image::RgbaImage::from_raw(dec.width as u32, dec.height as u32, rgba)
    } else {
        let img = image::open(path).ok()?;
        let img = if max_dim > 0 {
            img.thumbnail(max_dim, max_dim)
        } else {
            img
        };
        Some(img.to_rgba8())
    }
}
