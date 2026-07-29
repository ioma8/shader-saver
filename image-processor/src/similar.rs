use std::path::PathBuf;

pub const SIMILARITY_THRESHOLD: f32 = 0.96;

#[derive(Clone, Debug)]
pub struct Analysis {
    pub embedding: Vec<f32>,
    pub rating: u8,
    pub face_count: u8,
    pub largest_face: f32,
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub path: PathBuf,
    pub analysis: Analysis,
}

// Connected components make a burst one group even when the first and last
// frames drift slightly, provided each adjacent pair remains near-identical.
pub fn groups(candidates: &[Candidate]) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..candidates.len()).collect();
    for i in 0..candidates.len() {
        for j in (i + 1)..candidates.len() {
            if cosine(&candidates[i].analysis.embedding, &candidates[j].analysis.embedding)
                >= SIMILARITY_THRESHOLD
            {
                union(&mut parent, i, j);
            }
        }
    }
    let mut groups = std::collections::BTreeMap::<usize, Vec<usize>>::new();
    for i in 0..candidates.len() {
        groups.entry(find(&mut parent, i)).or_default().push(i);
    }
    groups.into_values().filter(|g| g.len() > 1).collect()
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (dot, a2, b2) = a
        .iter()
        .zip(b)
        .fold((0.0, 0.0, 0.0), |(dot, a2, b2), (&a, &b)| {
            (dot + a * b, a2 + a * a, b2 + b * b)
        });
    dot / (a2.sqrt() * b2.sqrt()).max(f32::EPSILON)
}

fn find(parent: &mut [usize], i: usize) -> usize {
    if parent[i] != i {
        parent[i] = find(parent, parent[i]);
    }
    parent[i]
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let a = find(parent, a);
    let b = find(parent, b);
    if a != b {
        parent[b] = a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(path: &str, embedding: Vec<f32>) -> Candidate {
        Candidate {
            path: PathBuf::from(path),
            analysis: Analysis { embedding, rating: 3, face_count: 0, largest_face: 0.0 },
        }
    }

    #[test]
    fn groups_nearby_embeddings_and_leaves_different_images_alone() {
        let candidates = vec![
            candidate("a.jpg", vec![1.0, 0.0]),
            candidate("b.jpg", vec![0.999, 0.04]),
            candidate("c.jpg", vec![0.0, 1.0]),
        ];
        assert_eq!(groups(&candidates), vec![vec![0, 1]]);
    }

    #[test]
    fn cosine_handles_mismatched_vectors() {
        assert_eq!(cosine(&[1.0], &[1.0, 0.0]), 0.0);
    }
}
