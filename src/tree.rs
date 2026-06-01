#![allow(unsafe_op_in_unsafe_fn)]

use crate::types::RawData;

// ── Design constants ──────────────────────────────────────────────────────────
pub const DIMS:    usize = 14;   // features from the normalizer
pub const K:       usize = 5;    // k nearest neighbours (must match RESPONSES length)

const PACKED_DIMS: usize = 16;   // DIMS rounded up to 256-bit boundary (16 × i16)
const LANES:       usize = 8;    // vectors per SOA block (128-bit load = 8 × i16)
const SCALE:       f32   = 32767.0;
const LEAF_SIZE:   usize = 64;   // 8 blocks per leaf; tree depth ≈ log2(375k/64) ≈ 13
const NUM_PARTS:   usize = 8;    // 2^3 partitions
const STACK_CAP:   usize = 64;   // ≥ max balanced-tree depth

const MAGIC: &[u8; 8] = b"KDTR0001";

// ── Types ─────────────────────────────────────────────────────────────────────

// Quantised feature vector; dims 14-15 are always 0 (padding for 256-bit loads).
type QVec = [i16; PACKED_DIMS];

// ── Quantise & partition key ──────────────────────────────────────────────────

/// Converts normalised f32 features [0,1] (or -1 for absent) to i16 [−32767, 32767].
#[inline(always)]
fn quantize(v: &[f32; 16]) -> QVec {
    let mut q = [0i16; PACKED_DIMS];
    for i in 0..DIMS {
        q[i] = (v[i] * SCALE).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
    }
    // q[14], q[15] stay 0 — safe for bounding-box AVX2 load
    q
}

/// 3-bit partition key → 8 partitions.
///
/// Partitioning on binary features keeps every partition's quantised values
/// in [0, 32767], preventing i16 subtraction wrap in the distance kernels.
/// (v[5] = −1.0 exactly when last_timestamp is absent, so v[5] >= 0.0 ↔ has_last_tx.)
#[inline(always)]
fn partition_key(v: &[f32; 16]) -> usize {
    let mut key = 0usize;
    if v[10] > 0.5  { key |= 1 << 0; }  // card_present
    if v[9]  > 0.5  { key |= 1 << 1; }  // is_online
    if v[5]  >= 0.0 { key |= 1 << 2; }  // has_last_tx
    key
}

// ── Bounding-box lower bound ──────────────────────────────────────────────────

/// Minimum possible squared distance from `q` to an axis-aligned bounding box.
/// Returns 0 when `q` is inside the box. Used to prune entire subtrees.
#[inline(always)]
fn lower_bound_box(q: &QVec, min: &QVec, max: &QVec) -> i64 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return unsafe { lower_bound_box_avx2(q, min, max) };
    }
    lower_bound_box_scalar(q, min, max)
}

#[inline(always)]
fn lower_bound_box_scalar(q: &QVec, min: &QVec, max: &QVec) -> i64 {
    let mut sum = 0i64;
    for d in 0..DIMS {
        let qd = q[d] as i64;
        let diff: i64 = if qd < min[d] as i64 {
            min[d] as i64 - qd
        } else if qd > max[d] as i64 {
            qd - max[d] as i64
        } else {
            0
        };
        sum += diff * diff;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lower_bound_box_avx2(q: &QVec, min: &QVec, max: &QVec) -> i64 {
    use std::arch::x86_64::*;
    // Load all 16 i16 values at once. Padding dims are 0 everywhere → zero contribution.
    let qv  = _mm256_loadu_si256(q.as_ptr()   as *const __m256i);
    let mnv = _mm256_loadu_si256(min.as_ptr() as *const __m256i);
    let mxv = _mm256_loadu_si256(max.as_ptr() as *const __m256i);
    let zero = _mm256_setzero_si256();
    // Clamped per-dim distance: max(0, min-q) or max(0, q-max).
    // Within a partition, values are in [0, 32767], so differences fit in i16. Safe.
    let below = _mm256_max_epi16(_mm256_sub_epi16(mnv, qv), zero);
    let above = _mm256_max_epi16(_mm256_sub_epi16(qv, mxv), zero);
    let diff  = _mm256_max_epi16(below, above);
    // diff in [0, 32767]; diff^2 ≤ 32767^2 = 1.07B fits in i32.
    // madd sums adjacent pairs → i32: max 2 × 1.07B = 2.147B < i32::MAX. Safe.
    let sq = _mm256_madd_epi16(diff, diff);  // 8 × i32
    // Widen to i64 and horizontal sum.
    let lo  = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sq));
    let hi  = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sq, 1));
    let s64 = _mm256_add_epi64(lo, hi);      // 4 × i64
    let s_hi = _mm256_extracti128_si256(s64, 1);
    let s128 = _mm_add_epi64(_mm256_castsi256_si128(s64), s_hi);
    _mm_extract_epi64(s128, 0) + _mm_extract_epi64(s128, 1)
}

// ── Block scan ────────────────────────────────────────────────────────────────

/// Computes squared distances from `q` to the LANES vectors in one SOA block.
/// Block layout: [d0_v0..v7, d1_v0..v7, ..., d13_v0..v7] (DIMS × LANES i16 values).
#[inline(always)]
fn scan_block(vectors: &[i16], block_base: usize, q: &QVec) -> [i64; LANES] {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return unsafe { scan_block_avx2(vectors, block_base, q) };
    }
    scan_block_scalar(vectors, block_base, q)
}

#[inline(always)]
fn scan_block_scalar(vectors: &[i16], block_base: usize, q: &QVec) -> [i64; LANES] {
    let mut dists = [0i64; LANES];
    for d in 0..DIMS {
        let qd = q[d] as i64;
        let base = block_base + d * LANES;
        for i in 0..LANES {
            let diff = qd - vectors[base + i] as i64;
            dists[i] += diff * diff;
        }
    }
    dists
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_block_avx2(vectors: &[i16], block_base: usize, q: &QVec) -> [i64; LANES] {
    use std::arch::x86_64::*;
    let mut acc_lo = _mm256_setzero_si256();  // lanes 0-3, i64
    let mut acc_hi = _mm256_setzero_si256();  // lanes 4-7, i64

    for d in 0..DIMS {
        // Broadcast this query dimension to all 8 lanes.
        let qd = _mm_set1_epi16(q[d]);
        // Load 8 reference values (one per lane) for this dimension.
        let vd = _mm_loadu_si128(
            vectors.as_ptr().add(block_base + d * LANES) as *const __m128i
        );
        // diff in [−32767, 32767] within a partition (values are [0, 32767]).
        let diff = _mm_sub_epi16(qd, vd);
        // Widen diff (8 × i16) → (8 × i32) then square.
        // 32767^2 = 1.07B < i32::MAX: no overflow.
        let diff32 = _mm256_cvtepi16_epi32(diff);
        let sq     = _mm256_mullo_epi32(diff32, diff32);  // 8 × i32
        // Widen to i64 and accumulate per lane.
        acc_lo = _mm256_add_epi64(acc_lo, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sq)));
        acc_hi = _mm256_add_epi64(acc_hi, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sq, 1)));
    }

    let mut dists = [0i64; LANES];
    _mm256_storeu_si256(dists.as_mut_ptr()        as *mut __m256i, acc_lo);
    _mm256_storeu_si256(dists.as_mut_ptr().add(4) as *mut __m256i, acc_hi);
    dists
}

// ── K-best insertion (sorted, smallest first) ─────────────────────────────────

#[inline(always)]
fn insert_best(dist: i64, label: u8, best_dists: &mut [i64; K], best_labels: &mut [u8; K]) {
    if dist >= best_dists[K - 1] { return; }
    let mut pos = K - 1;
    while pos > 0 && dist < best_dists[pos - 1] {
        best_dists[pos]  = best_dists[pos - 1];
        best_labels[pos] = best_labels[pos - 1];
        pos -= 1;
    }
    best_dists[pos]  = dist;
    best_labels[pos] = label;
}
// ── Node & index structures ───────────────────────────────────────────────────

/// 80-byte tree node.
/// Internal: left and right are child indices within the partition's node slice.
/// Leaf:     left == right == -1; start/count point into the SOA block array.
#[repr(C)]
#[derive(Clone, Default)]
struct Node {
    left:  i32,
    right: i32,
    start: u32,  // leaf: first block index
    count: u32,  // leaf: number of valid vectors (last block may be partial)
    min:   QVec, // bounding box over all vectors in this subtree
    max:   QVec,
}

pub struct TreeIndex {
    // Nodes for all 8 partitions, concatenated.
    // Partition p owns nodes[node_off[p] .. node_off[p+1]].
    nodes:    Vec<Node>,
    node_off: [usize; NUM_PARTS + 1],
    roots:    [i32;   NUM_PARTS],      // root index within partition slice; -1 if empty

    // SOA vector blocks, concatenated across partitions.
    // Each block = DIMS * LANES i16 values: [d0_v0..v7, d1_v0..v7, ..., d13_v0..v7].
    // Partition p owns vectors[vec_off[p] .. vec_off[p+1]].
    vectors:  Vec<i16>,
    vec_off:  [usize; NUM_PARTS + 1],

    // Labels, one per vector, packed into blocks of LANES u8 values.
    // Partition p owns labels[lbl_off[p] .. lbl_off[p+1]].
    labels:   Vec<u8>,
    lbl_off:  [usize; NUM_PARTS + 1],
}

// ── Builder helpers ───────────────────────────────────────────────────────────

fn compute_bbox(slice: &[(QVec, u8)]) -> (QVec, QVec) {
    let mut mn = [i16::MAX; PACKED_DIMS];
    let mut mx = [i16::MIN; PACKED_DIMS];
    for (v, _) in slice {
        for d in 0..DIMS {
            if v[d] < mn[d] { mn[d] = v[d]; }
            if v[d] > mx[d] { mx[d] = v[d]; }
        }
    }
    (mn, mx)
}

/// Returns the dimension index with the highest variance across `slice`.
/// Higher variance → better separation of left/right children → sharper bounding boxes.
fn max_variance_dim(slice: &[(QVec, u8)]) -> usize {
    let n = slice.len() as f64;
    let mut best = (0usize, 0.0f64);
    for d in 0..DIMS {
        let mean = slice.iter().map(|(v, _)| v[d] as f64).sum::<f64>() / n;
        let var  = slice.iter()
            .map(|(v, _)| { let diff = v[d] as f64 - mean; diff * diff })
            .sum::<f64>() / n;
        if var > best.1 { best = (d, var); }
    }
    best.0
}

/// Packs `slice` into SOA blocks and appends to `vectors` / `labels`.
fn pack_soa(slice: &[(QVec, u8)], vectors: &mut Vec<i16>, labels: &mut Vec<u8>) {
    let n          = slice.len();
    let num_blocks = (n + LANES - 1) / LANES;
    for b in 0..num_blocks {
        // Dimension-major order within the block.
        for d in 0..DIMS {
            for lane in 0..LANES {
                let idx = b * LANES + lane;
                vectors.push(if idx < n { slice[idx].0[d] } else { 0 });
            }
        }
        for lane in 0..LANES {
            let idx = b * LANES + lane;
            labels.push(if idx < n { slice[idx].1 } else { 0 });
        }
    }
}

// ── Recursive tree builder ─────────────────────────────────────────────────────

fn build_recursive(
    data:    &mut [(QVec, u8)],
    nodes:   &mut Vec<Node>,
    vectors: &mut Vec<i16>,
    labels:  &mut Vec<u8>,
) -> usize {
    let n = data.len();
    let (mn, mx) = compute_bbox(data);

    if n <= LEAF_SIZE {
        let first_block = vectors.len() / (DIMS * LANES);
        pack_soa(data, vectors, labels);
        let idx = nodes.len();
        nodes.push(Node { left: -1, right: -1, start: first_block as u32, count: n as u32, min: mn, max: mx });
        return idx;
    }

    let dim = max_variance_dim(data);
    let mid = n / 2;

    // Partition in O(n): bring the median element to position `mid`.
    // Both the quantised vector and its label travel together as a tuple.
    data.select_nth_unstable_by_key(mid, |&(ref v, _)| v[dim]);

    // Reserve the node slot before recursing so the index is stable.
    let idx = nodes.len();
    nodes.push(Node::default());

    let left  = build_recursive(&mut data[..mid],  nodes, vectors, labels);
    let right = build_recursive(&mut data[mid..],  nodes, vectors, labels);

    nodes[idx] = Node { left: left as i32, right: right as i32, start: 0, count: 0, min: mn, max: mx };
    idx
}

// ── Public build entry point ──────────────────────────────────────────────────

/// Builds a `TreeIndex` from pre-normalised training records.
/// `RawData.vector` must contain 14 normalised f32 values (same as the normaliser output).
pub fn build(data: Vec<RawData>) -> TreeIndex {
    // ── Step 1: quantise and route each record to its partition ──────────────
    let mut parts: [Vec<(QVec, u8)>; NUM_PARTS] = std::array::from_fn(|_| Vec::new());

    for rec in &data {
        let mut vf = [0.0f32; 16];
        let len = rec.vector.len().min(DIMS);
        vf[..len].copy_from_slice(&rec.vector[..len]);

        let q     = quantize(&vf);
        let key   = partition_key(&vf);
        let label = u8::from(rec.label == "fraud");
        parts[key].push((q, label));
    }

    // ── Step 2: build one KD-tree per partition ───────────────────────────────
    let mut all_nodes   = Vec::new();
    let mut all_vectors = Vec::new();
    let mut all_labels  = Vec::new();
    let mut node_off    = [0usize; NUM_PARTS + 1];
    let mut vec_off     = [0usize; NUM_PARTS + 1];
    let mut lbl_off     = [0usize; NUM_PARTS + 1];
    let mut roots       = [-1i32; NUM_PARTS];

    for p in 0..NUM_PARTS {
        node_off[p] = all_nodes.len();
        vec_off[p]  = all_vectors.len();
        lbl_off[p]  = all_labels.len();

        if !parts[p].is_empty() {
            let root = build_recursive(
                &mut parts[p],
                &mut all_nodes,
                &mut all_vectors,
                &mut all_labels,
            );
            roots[p] = root as i32;
        }

        println!(
            "  partition {p}: {} vectors, {} nodes",
            parts[p].len(),
            all_nodes.len() - node_off[p],
        );
    }

    node_off[NUM_PARTS] = all_nodes.len();
    vec_off[NUM_PARTS]  = all_vectors.len();
    lbl_off[NUM_PARTS]  = all_labels.len();

    TreeIndex { nodes: all_nodes, node_off, roots, vectors: all_vectors, vec_off, labels: all_labels, lbl_off }
}
// ── Query: iterative KD-tree traversal ───────────────────────────────────────

fn search(
    root:       usize,
    nodes:      &[Node],
    vectors:    &[i16],
    labels:     &[u8],
    q:          &QVec,
    best_dists: &mut [i64; K],
    best_lbl:   &mut [u8;  K],
) {
    // Iterative traversal with an explicit stack avoids Rust recursion overhead
    // and keeps depth bounded to STACK_CAP.
    let mut stack:      [(usize, i64); STACK_CAP] = [(0, 0); STACK_CAP];
    let mut stack_len   = 0usize;

    let mut cur       = root;
    let mut cur_bound = lower_bound_box(q, &nodes[root].min, &nodes[root].max);

    loop {
        if cur_bound < best_dists[K - 1] {
            let node = &nodes[cur];

            if node.left < 0 {
                // ── Leaf: scan all blocks ──────────────────────────────────
                let start  = node.start as usize;
                let count  = node.count as usize;
                let blocks = (count + LANES - 1) / LANES;

                for b in 0..blocks {
                    let block_base = (start + b) * DIMS * LANES;
                    let label_base = (start + b) * LANES;
                    let dists      = scan_block(vectors, block_base, q);
                    let lanes      = (count - b * LANES).min(LANES);
                    for i in 0..lanes {
                        insert_best(dists[i], labels[label_base + i], best_dists, best_lbl);
                    }
                }
            } else {
                // ── Internal: compute child bounds and go near-first ───────
                let l = node.left  as usize;
                let r = node.right as usize;

                let lb = lower_bound_box(q, &nodes[l].min, &nodes[l].max);
                let rb = lower_bound_box(q, &nodes[r].min, &nodes[r].max);

                let (near, near_b, far, far_b) = if lb <= rb {
                    (l, lb, r, rb)
                } else {
                    (r, rb, l, lb)
                };

                // Push the farther child only if it can still improve best_dists[K-1].
                if far_b < best_dists[K - 1] && stack_len < STACK_CAP {
                    stack[stack_len] = (far, far_b);
                    stack_len += 1;
                }

                // Descend into the nearer child without pushing to stack (tail-call).
                cur       = near;
                cur_bound = near_b;
                continue;
            }
        }

        // Pop next candidate from stack.
        if stack_len == 0 { break; }
        stack_len -= 1;
        (cur, cur_bound) = stack[stack_len];
    }
}

// ── Public predict ────────────────────────────────────────────────────────────

impl TreeIndex {
    /// Returns the number of fraud labels among the K nearest neighbours (0-5).
    /// The caller indexes directly into RESPONSES with this value.
    pub fn predict(&self, v: [f32; 16]) -> usize {
        let q   = quantize(&v);
        let key = partition_key(&v);

        if self.roots[key] < 0 { return 0; }

        let root    = self.roots[key] as usize;
        let nodes   = &self.nodes  [self.node_off[key] .. self.node_off[key + 1]];
        let vectors = &self.vectors[self.vec_off[key]  .. self.vec_off[key  + 1]];
        let labels  = &self.labels [self.lbl_off[key]  .. self.lbl_off[key  + 1]];

        let mut best_dists  = [i64::MAX; K];
        let mut best_labels = [0u8; K];

        search(root, nodes, vectors, labels, &q, &mut best_dists, &mut best_labels);

        best_labels.iter().filter(|&&l| l != 0).count()
    }

    /// Touches every memory page of the index, pulling the mmap into RAM.
    pub fn pretouch(&self) {
        let mut acc = 0u8;
        let mut i = 0usize;
        while i < self.vectors.len() {
            acc ^= self.vectors[i] as u8;
            i += 256; // one touch per cache line (×4 pages apart is fine here)
        }
        for i in (0..self.labels.len()).step_by(4096) {
            acc ^= self.labels[i];
        }
        std::hint::black_box(acc);
    }
}

// ── Save / load ───────────────────────────────────────────────────────────────

impl TreeIndex {
    pub fn save(&self, path: &str) {
        use std::io::{BufWriter, Write};
        let f = std::fs::File::create(path).expect("create tree file");
        let mut w = BufWriter::new(f);

        w.write_all(MAGIC).unwrap();

        // Per-partition header: root (i32) + node_count (u32) + vec_len (u32) + lbl_len (u32)
        for p in 0..NUM_PARTS {
            let nc = (self.node_off[p + 1] - self.node_off[p]) as u32;
            let vl = (self.vec_off [p + 1] - self.vec_off [p]) as u32;
            let ll = (self.lbl_off [p + 1] - self.lbl_off [p]) as u32;
            w.write_all(&self.roots[p].to_le_bytes()).unwrap();
            w.write_all(&nc.to_le_bytes()).unwrap();
            w.write_all(&vl.to_le_bytes()).unwrap();
            w.write_all(&ll.to_le_bytes()).unwrap();
        }

        // All nodes
        for n in &self.nodes {
            w.write_all(&n.left .to_le_bytes()).unwrap();
            w.write_all(&n.right.to_le_bytes()).unwrap();
            w.write_all(&n.start.to_le_bytes()).unwrap();
            w.write_all(&n.count.to_le_bytes()).unwrap();
            for &v in &n.min { w.write_all(&v.to_le_bytes()).unwrap(); }
            for &v in &n.max { w.write_all(&v.to_le_bytes()).unwrap(); }
        }

        // All vectors (i16) and labels (u8)
        for &v in &self.vectors {
            w.write_all(&v.to_le_bytes()).unwrap();
        }
        w.write_all(&self.labels).unwrap();
    }

    pub fn load(path: &str) -> Self {
        let data = std::fs::read(path).expect("read tree file");
        let mut c = 0usize;

        let magic = &data[c..c + 8]; c += 8;
        assert_eq!(magic, MAGIC, "bad magic — rebuild the index");

        let mut roots    = [-1i32; NUM_PARTS];
        let mut node_cnt = [0u32;  NUM_PARTS];
        let mut vec_lens = [0u32;  NUM_PARTS];
        let mut lbl_lens = [0u32;  NUM_PARTS];

        for p in 0..NUM_PARTS {
            roots[p]    = i32::from_le_bytes(data[c..c+4].try_into().unwrap()); c += 4;
            node_cnt[p] = u32::from_le_bytes(data[c..c+4].try_into().unwrap()); c += 4;
            vec_lens[p] = u32::from_le_bytes(data[c..c+4].try_into().unwrap()); c += 4;
            lbl_lens[p] = u32::from_le_bytes(data[c..c+4].try_into().unwrap()); c += 4;
        }

        let total_nodes = node_cnt.iter().sum::<u32>() as usize;
        let mut nodes = Vec::with_capacity(total_nodes);
        for _ in 0..total_nodes {
            let left  = i32::from_le_bytes(data[c..c+4].try_into().unwrap()); c += 4;
            let right = i32::from_le_bytes(data[c..c+4].try_into().unwrap()); c += 4;
            let start = u32::from_le_bytes(data[c..c+4].try_into().unwrap()); c += 4;
            let count = u32::from_le_bytes(data[c..c+4].try_into().unwrap()); c += 4;
            let mut mn = [0i16; PACKED_DIMS];
            let mut mx = [0i16; PACKED_DIMS];
            for v in &mut mn { *v = i16::from_le_bytes(data[c..c+2].try_into().unwrap()); c += 2; }
            for v in &mut mx { *v = i16::from_le_bytes(data[c..c+2].try_into().unwrap()); c += 2; }
            nodes.push(Node { left, right, start, count, min: mn, max: mx });
        }

        let total_vecs = vec_lens.iter().sum::<u32>() as usize;
        let mut vectors = Vec::with_capacity(total_vecs);
        for _ in 0..total_vecs {
            vectors.push(i16::from_le_bytes(data[c..c+2].try_into().unwrap()));
            c += 2;
        }

        let total_lbls = lbl_lens.iter().sum::<u32>() as usize;
        let labels = data[c..c + total_lbls].to_vec();

        let mut node_off = [0usize; NUM_PARTS + 1];
        let mut vec_off  = [0usize; NUM_PARTS + 1];
        let mut lbl_off  = [0usize; NUM_PARTS + 1];
        for p in 0..NUM_PARTS {
            node_off[p + 1] = node_off[p] + node_cnt[p] as usize;
            vec_off [p + 1] = vec_off [p] + vec_lens[p] as usize;
            lbl_off [p + 1] = lbl_off [p] + lbl_lens[p] as usize;
        }

        Self { nodes, node_off, roots, vectors, vec_off, labels, lbl_off }
    }
}