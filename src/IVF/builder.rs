use crate::types::RawData;
use super::index::{BucketMeta, StaticIVF, quantize, sq_dist_u8};

// ── k-Means (Lloyd's algorithm) ───────────────────────────────────────────────
//
// Operates entirely on quantized u8[16] vectors:
//   • Assignment:  u8 SIMD distance  (fast, branch-free)
//   • Accumulation: u64 sums        (no overflow for n ≤ 2^48 per dim)
//   • Centroid:    f32 mean → u8    (keeps precision during updates)

fn kmeans(
    vectors: &[[u8; 16]],
    k:       usize,
    iters:   usize,
) -> (Vec<[u8; 16]>, Vec<u32>) {
    let n = vectors.len();
    assert!(n >= k, "fewer vectors than centroids");

    // Deterministic XorShift RNG
    let mut rng = 0x517cc1b727220a95u64;
    let next = |rng: &mut u64| -> u64 {
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        *rng
    };

    // Random initialisation — collision probability negligible (n ≫ k)
    let mut centroids_f32: Vec<[f32; 16]> = (0..k)
        .map(|_| {
            let idx = (next(&mut rng) % n as u64) as usize;
            let mut f = [0.0f32; 16];
            for i in 0..16 { f[i] = vectors[idx][i] as f32; }
            f
        })
        .collect();

    let mut assignments = vec![0u32; n];

    for iter in 0..iters {
        // Convert f32 centroids → u8 once per iteration (reuse across all n vectors)
        let u8c: Vec<[u8; 16]> = centroids_f32.iter().map(|c| {
            let mut q = [0u8; 16];
            for i in 0..16 { q[i] = c[i].clamp(0.0, 255.0) as u8; }
            q
        }).collect();

        // ── Assignment step ──────────────────────────────────────────────
        let mut changed = 0usize;
        for v in 0..n {
            let vec = &vectors[v];
            let mut best_dist = u32::MAX;
            let mut best_c    = 0u32;
            for c in 0..k {
                let d = sq_dist_u8(vec, &u8c[c]);
                if d < best_dist { best_dist = d; best_c = c as u32; }
            }
            if assignments[v] != best_c { changed += 1; assignments[v] = best_c; }
        }

        // ── Update step (f32 accumulators) ───────────────────────────────
        let mut sums   = vec![[0u64; 16]; k];
        let mut counts = vec![0u64; k];
        for v in 0..n {
            let c = assignments[v] as usize;
            counts[c] += 1;
            for d in 0..16 { sums[c][d] += vectors[v][d] as u64; }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..16 {
                    centroids_f32[c][d] = sums[c][d] as f32 / counts[c] as f32;
                }
            }
            // Empty centroid: keep previous position (rare after first iteration)
        }

        println!(
            "  k-means iter {:>2}/{}: {} reassignments ({:.2}%)",
            iter + 1, iters, changed,
            100.0 * changed as f64 / n as f64
        );
        if changed == 0 { println!("  Converged early."); break; }
    }

    // Final u8 centroids
    let u8c = centroids_f32.iter().map(|c| {
        let mut q = [0u8; 16];
        for i in 0..16 { q[i] = c[i].clamp(0.0, 255.0) as u8; }
        q
    }).collect();

    (u8c, assignments)
}

// ── Public build entry ────────────────────────────────────────────────────────

/// Build a flat IVF index from raw training data.
///
/// Parameters
/// ----------
/// `k_centroids`  – Voronoi cells.  Rule of thumb: `sqrt(n)` ≈ 1732 for n=3 M;
///                  in practice 512–2048 covers the speed/recall Pareto front.
/// `nprobe`       – cells probed at query time (stored in the file, not tunable
///                  without rebuilding — expose via env var in the indexer).
/// `k`            – k-NN count; must equal the length of `server::RESPONSES` - 1
///                  (currently 5 for 0..=5 fraud votes).
/// `kmeans_iters` – Lloyd iterations (10 is enough for 3 M vectors).
pub fn build_ivf(
    k_centroids:  usize,
    nprobe:       usize,
    k:            usize,
    kmeans_iters: usize,
    data:         Vec<RawData>,
) -> StaticIVF {
    let n = data.len();
    println!(
        "Building IVF: n={n}, k_centroids={k_centroids}, nprobe={nprobe}, k={k}, iters={kmeans_iters}"
    );

    // ── 1. Quantize training vectors ──────────────────────────────────────
    let mut q_vecs:     Vec<[u8; 16]> = Vec::with_capacity(n);
    let mut raw_labels: Vec<u8>       = Vec::with_capacity(n);

    for raw in &data {
        debug_assert_eq!(raw.vector.len(), 14);
        let mut v = [0.0f32; 16];
        unsafe { std::ptr::copy_nonoverlapping(raw.vector.as_ptr(), v.as_mut_ptr(), 14); }
        q_vecs.push(quantize(&v));
        raw_labels.push(if raw.label == "fraud" { 1 } else { 0 });
    }

    // ── 2. k-Means ────────────────────────────────────────────────────────
    println!("Running k-means ({kmeans_iters} iterations over {n} vectors)...");
    let t0 = std::time::Instant::now();
    let (centroids, assignments) = kmeans(&q_vecs, k_centroids, kmeans_iters);
    println!("k-means finished in {:.1}s", t0.elapsed().as_secs_f64());

    // ── 3. Count vectors per bucket, compute starts (prefix sum) ─────────
    let mut bucket_metas = vec![BucketMeta::default(); k_centroids];
    for &a in &assignments {
        bucket_metas[a as usize].len += 1;
    }
    {
        let mut cursor = 0u32;
        for meta in bucket_metas.iter_mut() {
            meta.start  = cursor;
            cursor     += meta.len;
            meta.len    = 0; // reset; we'll recount while filling below
        }
    }

    // ── 4. Sort vectors into bucket order ─────────────────────────────────
    // Build an index array sorted by assignment, then scatter.
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by_key(|&i| assignments[i as usize]);

    let mut sorted_vectors: Vec<[u8; 16]> = Vec::with_capacity(n);
    let mut sorted_labels:  Vec<u8>       = Vec::with_capacity(n);
    for &orig in &order {
        sorted_vectors.push(q_vecs[orig as usize]);
        sorted_labels.push(raw_labels[orig as usize]);
    }

    // Recompute lengths (start is already correct from prefix sum).
    for &a in &assignments {
        bucket_metas[a as usize].len += 1;
    }

    // ── 5. Assemble ───────────────────────────────────────────────────────
    println!(
        "Index size: {:.1} MB  (vectors {:.1} MB + labels {:.1} MB)",
        (n * 17 + k_centroids * 24 + 32) as f64 / 1_048_576.0,
        (n * 16)                          as f64 / 1_048_576.0,
        (n)                               as f64 / 1_048_576.0,
    );

    StaticIVF::from_parts(centroids, bucket_metas, sorted_vectors, sorted_labels, nprobe, k)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_separates_clusters() {
        // Two clearly separated clusters in a 16-d space.
        let low:  Vec<[u8; 16]> = (0..50).map(|_| [10u8; 16]).collect();
        let high: Vec<[u8; 16]> = (0..50).map(|_| [240u8; 16]).collect();
        let vecs = [low, high].concat();

        let (_, asgn) = kmeans(&vecs, 2, 10);

        // All "low" vectors share the same centroid, and "high" share the other.
        assert_eq!(asgn[0], asgn[25]);
        assert_ne!(asgn[0], asgn[75]);
    }

    #[test]
    fn test_build_and_predict() {
        use crate::types::RawData;

        // 200 "legit" vectors (all zeros) and 200 "fraud" vectors (all ones).
        let mut data: Vec<RawData> = (0..200)
            .map(|_| RawData { vector: vec![0.0f32; 14], label: "legit".to_string() })
            .collect();
        data.extend((0..200).map(|_| RawData {
            vector: vec![1.0f32; 14],
            label: "fraud".to_string(),
        }));

        let idx = build_ivf(4, 1, 5, 5, data);

        // Query near the legit cluster → 0 fraud neighbours
        let mut q_legit = [0.0f32; 16];
        for i in 0..14 { q_legit[i] = 0.0; }
        assert_eq!(idx.predict(q_legit), 0);

        // Query near the fraud cluster → all 5 neighbours fraud
        let mut q_fraud = [0.0f32; 16];
        for i in 0..14 { q_fraud[i] = 1.0; }
        assert_eq!(idx.predict(q_fraud), 5);
    }
}
