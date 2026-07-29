use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[repr(i64)]
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum Flag {
    #[default]
    None = 0,
    Pick = 1,
    Reject = 2,
}

#[repr(i64)]
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum Label {
    #[default]
    None = 0,
    Red = 1,
    Yellow = 2,
    Green = 3,
    Blue = 4,
}

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct CullMeta {
    pub rating: u8,
    pub flag: Flag,
    pub label: Label,
}

impl CullMeta {
    pub fn is_default(self) -> bool {
        self == Self::default()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CullAction {
    Rating(u8),
    TogglePick,
    ToggleReject,
    ToggleLabel(Label),
}

pub fn apply_action(meta: &mut CullMeta, action: CullAction) {
    match action {
        CullAction::Rating(n) => {
            meta.rating = if meta.rating == n { 0 } else { n.min(5) };
        }
        CullAction::TogglePick => {
            meta.flag = if meta.flag == Flag::Pick {
                Flag::None
            } else {
                Flag::Pick
            };
        }
        CullAction::ToggleReject => {
            meta.flag = if meta.flag == Flag::Reject {
                Flag::None
            } else {
                Flag::Reject
            };
        }
        CullAction::ToggleLabel(label) => {
            meta.label = if meta.label == label {
                Label::None
            } else {
                label
            };
        }
    }
}

fn flag_from_i(v: i64) -> Flag {
    match v {
        1 => Flag::Pick,
        2 => Flag::Reject,
        _ => Flag::None,
    }
}

fn label_from_i(v: i64) -> Label {
    match v {
        1 => Label::Red,
        2 => Label::Yellow,
        3 => Label::Green,
        4 => Label::Blue,
        _ => Label::None,
    }
}

pub fn init_meta_table(conn: &rusqlite::Connection) {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS meta (
            path    TEXT PRIMARY KEY,
            rating  INTEGER NOT NULL DEFAULT 0,
            flag    INTEGER NOT NULL DEFAULT 0,
            label   INTEGER NOT NULL DEFAULT 0,
            updated INTEGER NOT NULL
        );",
    )
    .ok();
}

pub fn save_meta(conn: &rusqlite::Connection, path: &Path, meta: CullMeta) {
    let path = path.to_string_lossy();
    if meta.is_default() {
        let _ = conn.execute("DELETE FROM meta WHERE path = ?1", rusqlite::params![path]);
        return;
    }
    let updated = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let _ = conn.execute(
        "INSERT INTO meta (path, rating, flag, label, updated)
         VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(path) DO UPDATE SET rating = ?2, flag = ?3, label = ?4, updated = ?5",
        rusqlite::params![
            path,
            meta.rating as i64,
            meta.flag as i64,
            meta.label as i64,
            updated
        ],
    );
}

pub fn load_all_meta(conn: &rusqlite::Connection) -> HashMap<PathBuf, CullMeta> {
    let mut out = HashMap::new();
    let Ok(mut stmt) = conn.prepare("SELECT path, rating, flag, label FROM meta") else {
        return out;
    };
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, i64>(2)?,
            row.get::<_, i64>(3)?,
        ))
    });
    let Ok(rows) = rows else {
        return out;
    };
    for row in rows.flatten() {
        let (path, rating, flag, label) = row;
        out.insert(
            PathBuf::from(path),
            CullMeta {
                rating: rating.clamp(0, 5) as u8,
                flag: flag_from_i(flag),
                label: label_from_i(label),
            },
        );
    }
    out
}

pub fn delete_rows(conn: &rusqlite::Connection, path: &Path) {
    let p = path.to_string_lossy();
    let _ = conn.execute("DELETE FROM edits WHERE path = ?1", rusqlite::params![p]);
    let _ = conn.execute("DELETE FROM meta WHERE path = ?1", rusqlite::params![p]);
}

pub fn rekey_rows(conn: &rusqlite::Connection, old: &Path, new: &Path) {
    let (o, n) = (old.to_string_lossy(), new.to_string_lossy());
    let _ = conn.execute(
        "UPDATE OR REPLACE edits SET path = ?2 WHERE path = ?1",
        rusqlite::params![o, n],
    );
    let _ = conn.execute(
        "UPDATE OR REPLACE meta SET path = ?2 WHERE path = ?1",
        rusqlite::params![o, n],
    );
}

pub fn copy_rows(conn: &rusqlite::Connection, old: &Path, new: &Path) {
    let (o, n) = (old.to_string_lossy(), new.to_string_lossy());
    let _ = conn.execute(
        "INSERT OR REPLACE INTO edits (path, params, updated)
         SELECT ?2, params, updated FROM edits WHERE path = ?1",
        rusqlite::params![o, n],
    );
    let _ = conn.execute(
        "INSERT OR REPLACE INTO meta (path, rating, flag, label, updated)
         SELECT ?2, rating, flag, label, updated FROM meta WHERE path = ?1",
        rusqlite::params![o, n],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> rusqlite::Connection {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        init_meta_table(&conn);
        conn.execute_batch(
            "CREATE TABLE edits (path TEXT PRIMARY KEY, params TEXT NOT NULL, updated INTEGER NOT NULL);",
        )
        .unwrap();
        conn
    }

    #[test]
    fn meta_roundtrip() {
        let conn = db();
        let m = CullMeta {
            rating: 4,
            flag: Flag::Pick,
            label: Label::Green,
        };
        save_meta(&conn, Path::new("/a/img1.jpg"), m);
        let all = load_all_meta(&conn);
        assert_eq!(all.get(Path::new("/a/img1.jpg")), Some(&m));
    }

    #[test]
    fn default_meta_deletes_row() {
        let conn = db();
        save_meta(
            &conn,
            Path::new("/a/img1.jpg"),
            CullMeta {
                rating: 2,
                ..Default::default()
            },
        );
        save_meta(&conn, Path::new("/a/img1.jpg"), CullMeta::default());
        assert!(load_all_meta(&conn).is_empty());
    }

    #[test]
    fn actions_toggle() {
        let mut m = CullMeta::default();
        apply_action(&mut m, CullAction::Rating(3));
        assert_eq!(m.rating, 3);
        apply_action(&mut m, CullAction::Rating(3));
        assert_eq!(m.rating, 0);
        apply_action(&mut m, CullAction::Rating(0));
        assert_eq!(m.rating, 0);
        apply_action(&mut m, CullAction::TogglePick);
        assert_eq!(m.flag, Flag::Pick);
        apply_action(&mut m, CullAction::ToggleReject);
        assert_eq!(m.flag, Flag::Reject);
        apply_action(&mut m, CullAction::ToggleReject);
        assert_eq!(m.flag, Flag::None);
        apply_action(&mut m, CullAction::ToggleLabel(Label::Red));
        assert_eq!(m.label, Label::Red);
        apply_action(&mut m, CullAction::ToggleLabel(Label::Red));
        assert_eq!(m.label, Label::None);
    }

    #[test]
    fn rekey_copy_delete_rows() {
        let conn = db();
        save_meta(
            &conn,
            Path::new("/a/x.jpg"),
            CullMeta {
                rating: 5,
                flag: Flag::Pick,
                label: Label::Blue,
            },
        );
        conn.execute(
            "INSERT INTO edits (path, params, updated) VALUES (?1, ?2, 1)",
            rusqlite::params!["/a/x.jpg", "{\"exposure\":1}"],
        )
        .unwrap();

        copy_rows(&conn, Path::new("/a/x.jpg"), Path::new("/b/x.jpg"));
        assert_eq!(
            conn.query_row(
                "SELECT params FROM edits WHERE path = ?1",
                rusqlite::params!["/b/x.jpg"],
                |row| row.get::<_, String>(0),
            )
            .unwrap(),
            "{\"exposure\":1}"
        );

        rekey_rows(&conn, Path::new("/a/x.jpg"), Path::new("/c/x.jpg"));
        assert_eq!(
            conn.query_row(
                "SELECT rating FROM meta WHERE path = ?1",
                rusqlite::params!["/c/x.jpg"],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
            5
        );

        delete_rows(&conn, Path::new("/c/x.jpg"));
        assert!(
            conn.query_row(
                "SELECT COUNT(*) FROM edits WHERE path = ?1",
                rusqlite::params!["/c/x.jpg"],
                |row| row.get::<_, i64>(0),
            )
            .unwrap()
                == 0
        );
        assert!(
            conn.query_row(
                "SELECT COUNT(*) FROM meta WHERE path = ?1",
                rusqlite::params!["/c/x.jpg"],
                |row| row.get::<_, i64>(0),
            )
            .unwrap()
                == 0
        );
    }
}
