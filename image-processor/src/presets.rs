// Named edit presets, stored one-per-file as pretty-printed JSON under a
// presets directory. Plain files (not the SQLite db) so users can inspect,
// diff, back up, or hand-share a preset.
use crate::processor::EditState;
use std::path::{Path, PathBuf};

// Keeps preset names from escaping the presets directory (e.g. "../../etc")
// or colliding with filesystem-special characters.
fn slug(name: &str) -> Option<String> {
    let s: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, '-' | '_' | ' ') {
                c
            } else {
                '_'
            }
        })
        .collect();
    (!s.is_empty()).then_some(s)
}

fn file_path(dir: &Path, name: &str) -> Option<PathBuf> {
    Some(dir.join(format!("{}.json", slug(name)?)))
}

pub fn save(dir: &Path, name: &str, state: &EditState) -> std::io::Result<()> {
    let path = file_path(dir, name).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "empty preset name")
    })?;
    std::fs::create_dir_all(dir)?;
    let json = serde_json::to_string_pretty(state)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, json)
}

pub fn load(dir: &Path, name: &str) -> Option<EditState> {
    let json = std::fs::read_to_string(file_path(dir, name)?).ok()?;
    serde_json::from_str(&json).ok()
}

pub fn delete(dir: &Path, name: &str) -> std::io::Result<()> {
    let path = file_path(dir, name).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "empty preset name")
    })?;
    std::fs::remove_file(path)
}

pub fn list(dir: &Path) -> Vec<String> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = rd
        .filter_map(std::result::Result::ok)
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("json"))
        .filter_map(|e| {
            e.path()
                .file_stem()
                .and_then(|s| s.to_str())
                .map(str::to_string)
        })
        .collect();
    names.sort();
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ip_presets_test_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = tmp_dir("roundtrip");
        let state = EditState {
            exposure: 1.5,
            contrast: 20.0,
            ..Default::default()
        };
        save(&dir, "My Look", &state).unwrap();
        let loaded = load(&dir, "My Look").unwrap();
        assert_eq!(loaded.exposure, 1.5);
        assert_eq!(loaded.contrast, 20.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_is_sorted_and_delete_removes() {
        let dir = tmp_dir("list");
        save(&dir, "Zebra", &EditState::default()).unwrap();
        save(&dir, "Apple", &EditState::default()).unwrap();
        assert_eq!(list(&dir), vec!["Apple".to_string(), "Zebra".to_string()]);
        delete(&dir, "Apple").unwrap();
        assert_eq!(list(&dir), vec!["Zebra".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_name_rejected() {
        let dir = tmp_dir("empty");
        assert!(save(&dir, "   ", &EditState::default()).is_err());
    }

    #[test]
    fn name_cannot_escape_presets_dir() {
        let dir = tmp_dir("traversal");
        save(&dir, "../../evil", &EditState::default()).unwrap();
        let escaped = dir.parent().unwrap().parent().unwrap().join("evil.json");
        assert!(!escaped.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
