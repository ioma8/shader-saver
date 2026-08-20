// Unified image loading. RAW files auto-develop through the universal rawler
// sensor pipeline and learned look; embedded JPEG previews are browse-only and
// never become the editor's developed RAW.
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
    let image = orient_preview(
        image::load_from_memory(&bytes).ok()?,
        exif_orientation(path),
    );
    let image = if max_dim > 0 {
        image.thumbnail(max_dim, max_dim)
    } else {
        image
    };
    Some(image.to_rgba8())
}

pub(crate) fn exif_orientation(path: &Path) -> u32 {
    let Ok(file) = std::fs::File::open(path) else {
        return 1;
    };
    exif::Reader::new()
        .read_from_container(&mut std::io::BufReader::new(file))
        .ok()
        .and_then(|exif| {
            exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)
                .and_then(|field| field.value.get_uint(0))
        })
        .unwrap_or(1)
}

pub(crate) fn orient_preview(image: image::DynamicImage, orientation: u32) -> image::DynamicImage {
    match orientation {
        2 => image.fliph(),
        3 => image.rotate180(),
        4 => image.flipv(),
        5 => image.fliph().rotate270(),
        6 => image.rotate90(),
        7 => image.fliph().rotate90(),
        8 => image.rotate270(),
        _ => image,
    }
}

pub fn load_edit_rgba(path: &Path) -> Option<image::RgbaImage> {
    let max_dim = if is_raw(path) { 2048 } else { 0 };
    load_rgba(path, max_dim)
}

// RAW uses the same universal sensor decoder and trained display rendering as
// the Develop action. There is intentionally no camera-manufacturer branch or
// alternate thumbnail decoder: browse, edit, export and Develop all consume
// this one RAW path. Raster images are decoded normally.
pub fn load_rgba(path: &Path, max_dim: u32) -> Option<image::RgbaImage> {
    if is_raw(path) {
        return crate::raw_develop::develop_raw(path, max_dim);
    }

    // Apply EXIF orientation like `load_preview_rgba` does, so the grid, the
    // editor and the AI batch paths all agree on the same upright image.
    let image = image::open(path).ok()?;
    let image = orient_preview(image, exif_orientation(path));
    let image = if max_dim > 0 {
        image.thumbnail(max_dim, max_dim)
    } else {
        image
    };
    Some(image.to_rgba8())
}

#[cfg(test)]
mod tests {
    use super::{is_raw, load_preview_rgba, load_rgba};
    use std::path::Path;

    #[test]
    fn raw_browse_previews_match_editor_orientation() {
        let Ok(folder) = std::env::var("RAW_PREVIEW_FOLDER") else {
            return;
        };
        let mut checked = 0;
        for path in std::fs::read_dir(folder)
            .unwrap()
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| is_raw(path))
            .take(200)
        {
            let (Some(preview), Some(editor)) =
                (load_preview_rgba(&path, 220), load_rgba(&path, 220))
            else {
                continue;
            };
            assert_eq!(
                preview.width() > preview.height(),
                editor.width() > editor.height(),
                "{}: preview {}x{}, editor {}x{}",
                path.display(),
                preview.width(),
                preview.height(),
                editor.width(),
                editor.height()
            );
            checked += 1;
        }
        assert!(checked > 0, "no RAW preview/editor pairs decoded");
    }

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
