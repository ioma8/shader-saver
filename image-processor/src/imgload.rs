// Unified image loading. RAW files are always decoded from sensor data by
// imagepipe; embedded JPEG previews are used only for browse thumbnails.
use jpgfromraw::FindJpegType;
use std::path::Path;

pub const STD_EXTS: [&str; 8] = ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "gif"];
pub const RAW_EXTS: [&str; 19] = [
    "dng", "cr2", "cr3", "nef", "arw", "orf", "rw2", "raf", "pef", "srw", "mrw", "kdc", "dcr",
    "erf", "mef", "mos", "nrw", "3fr", "x3f",
];

fn ext_of(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_lowercase)
}

pub fn is_raw(path: &Path) -> bool {
    ext_of(path).is_some_and(|extension| RAW_EXTS.contains(&extension.as_str()))
}

pub fn is_supported(path: &Path) -> bool {
    ext_of(path).is_some_and(|extension| {
        STD_EXTS.contains(&extension.as_str()) || RAW_EXTS.contains(&extension.as_str())
    })
}

pub fn all_exts() -> Vec<&'static str> {
    STD_EXTS.iter().chain(RAW_EXTS.iter()).copied().collect()
}

pub fn load_preview_rgba(path: &Path, max_dim: u32) -> Option<image::RgbaImage> {
    let find_type = if max_dim > 0 {
        FindJpegType::Smallest
    } else {
        FindJpegType::Largest
    };
    let bytes = pollster::block_on(jpgfromraw::process_file_bytes(path, find_type)).ok()?;
    let image = image::load_from_memory(&bytes).ok()?;
    let image = if max_dim > 0 {
        image.thumbnail(max_dim, max_dim)
    } else {
        image
    };
    Some(image.to_rgba8())
}

pub fn load_edit_rgba(path: &Path) -> Option<(image::RgbaImage, bool)> {
    let max_dim = if is_raw(path) { 2048 } else { 0 };
    load_rgba(path, max_dim).map(|image| (image, false))
}

// RAW uses a real sensor decode (demosaic, camera WB/matrix and imagepipe's
// display curve), never the embedded camera preview. Raster images are decoded
// normally. This is the one base image used by editing, training and export.
pub fn load_rgba(path: &Path, max_dim: u32) -> Option<image::RgbaImage> {
    if is_raw(path) {
        let decoded =
            imagepipe::simple_decode_8bit(path, max_dim as usize, max_dim as usize).ok()?;
        let mut rgba = Vec::with_capacity(decoded.width * decoded.height * 4);
        for pixel in decoded.data.chunks_exact(3) {
            rgba.extend_from_slice(pixel);
            rgba.push(255);
        }
        return image::RgbaImage::from_raw(
            u32::try_from(decoded.width).ok()?,
            u32::try_from(decoded.height).ok()?,
            rgba,
        );
    }

    let image = image::open(path).ok()?;
    let image = if max_dim > 0 {
        image.thumbnail(max_dim, max_dim)
    } else {
        image
    };
    Some(image.to_rgba8())
}

#[cfg(test)]
mod tests {
    use super::load_rgba;
    use std::path::Path;

    #[test]
    fn raw_and_jpeg_are_decoded_from_their_real_pixels() {
        let (Ok(jpeg), Ok(raw)) = (
            std::env::var("LOOK_PARITY_JPEG"),
            std::env::var("LOOK_PARITY_RAW"),
        ) else {
            return;
        };
        for path in [&jpeg, &raw] {
            let image = load_rgba(Path::new(path), 768)
                .unwrap_or_else(|| panic!("decode failed for {path}"));
            assert!(image.width() > 0 && image.height() > 0);
        }
    }
}
