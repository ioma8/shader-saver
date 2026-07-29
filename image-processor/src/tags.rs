// User- or AI-assigned tags per image, stored as a JSON string array in
// SQLite (one row per tagged path), same shape as cull.rs's meta table.
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub fn init_table(conn: &rusqlite::Connection) {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS tags (
            path    TEXT PRIMARY KEY,
            tags    TEXT NOT NULL,
            updated INTEGER NOT NULL
        );",
    )
    .ok();
}

pub fn save_tags(conn: &rusqlite::Connection, path: &Path, tags: &[String]) {
    let path = path.to_string_lossy();
    if tags.is_empty() {
        let _ = conn.execute("DELETE FROM tags WHERE path = ?1", rusqlite::params![path]);
        return;
    }
    let Ok(json) = serde_json::to_string(tags) else {
        return;
    };
    let updated = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs().cast_signed());
    let _ = conn.execute(
        "INSERT INTO tags (path, tags, updated) VALUES (?1, ?2, ?3)
         ON CONFLICT(path) DO UPDATE SET tags = ?2, updated = ?3",
        rusqlite::params![path, json, updated],
    );
}

pub fn load_all_tags(conn: &rusqlite::Connection) -> HashMap<PathBuf, Vec<String>> {
    let mut out = HashMap::new();
    let Ok(mut stmt) = conn.prepare("SELECT path, tags FROM tags") else {
        return out;
    };
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    });
    let Ok(rows) = rows else {
        return out;
    };
    for (path, json) in rows.flatten() {
        if let Ok(tags) = serde_json::from_str(&json) {
            out.insert(PathBuf::from(path), tags);
        }
    }
    out
}

pub fn delete_rows(conn: &rusqlite::Connection, path: &Path) {
    let p = path.to_string_lossy();
    let _ = conn.execute("DELETE FROM tags WHERE path = ?1", rusqlite::params![p]);
}

pub fn rekey_rows(conn: &rusqlite::Connection, old: &Path, new: &Path) {
    let (o, n) = (old.to_string_lossy(), new.to_string_lossy());
    let _ = conn.execute(
        "UPDATE OR REPLACE tags SET path = ?2 WHERE path = ?1",
        rusqlite::params![o, n],
    );
}

pub fn copy_rows(conn: &rusqlite::Connection, old: &Path, new: &Path) {
    let (o, n) = (old.to_string_lossy(), new.to_string_lossy());
    let _ = conn.execute(
        "INSERT OR REPLACE INTO tags (path, tags, updated)
         SELECT ?2, tags, updated FROM tags WHERE path = ?1",
        rusqlite::params![o, n],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> rusqlite::Connection {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        init_table(&conn);
        conn
    }

    #[test]
    fn save_load_roundtrip() {
        let conn = db();
        let tags = vec!["cat".to_string(), "outdoors".to_string()];
        save_tags(&conn, Path::new("/a/img1.jpg"), &tags);
        let all = load_all_tags(&conn);
        assert_eq!(all.get(Path::new("/a/img1.jpg")), Some(&tags));
    }

    #[test]
    fn empty_tags_delete_row() {
        let conn = db();
        save_tags(&conn, Path::new("/a/img1.jpg"), &["cat".to_string()]);
        save_tags(&conn, Path::new("/a/img1.jpg"), &[]);
        assert!(load_all_tags(&conn).is_empty());
    }

    #[test]
    fn rekey_and_copy_and_delete() {
        let conn = db();
        let tags = vec!["dog".to_string()];
        save_tags(&conn, Path::new("/a/x.jpg"), &tags);

        copy_rows(&conn, Path::new("/a/x.jpg"), Path::new("/b/x.jpg"));
        assert_eq!(load_all_tags(&conn).get(Path::new("/b/x.jpg")), Some(&tags));

        rekey_rows(&conn, Path::new("/a/x.jpg"), Path::new("/c/x.jpg"));
        let all = load_all_tags(&conn);
        assert_eq!(all.get(Path::new("/c/x.jpg")), Some(&tags));
        assert!(!all.contains_key(Path::new("/a/x.jpg")));

        delete_rows(&conn, Path::new("/c/x.jpg"));
        assert!(!load_all_tags(&conn).contains_key(Path::new("/c/x.jpg")));
    }
}
