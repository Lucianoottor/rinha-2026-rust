//! High-performance *static* HNSW for 14-dimensional normalised vectors.
//!
//! Design pillars (from plan.json):
//!
//!   1. **Global-query-counter visited set** — flat `Vec<u32>` + generation
//!      counter.  Marking a node visited = one array write.  Resetting the
//!      whole table between queries = one integer increment.  No HashSet,
//!      no heap allocation on the query hot-path.
//!
//!   2. **16-float padded vectors** — 14 real dimensions + 2 zero-padded
//!      slots fill exactly two 256-bit YMM registers.  When compiled with
//!      AVX2 + FMA (enabled via `.cargo/config.toml`) the distance kernel
//!      resolves to: 2× VSUB + VFMADD + horizontal-reduce — ~6 instructions.
//!
//!   3. **Flat fixed-stride level-0 link array** — all level-0 neighbours
//!      live in one contiguous `Box<[u32]>` with stride `m_max0`.  Accessing
//!      node `c`'s neighbours = one pointer offset, no pointer chasing.
//!      Slots are filled left-to-right; the first `NONE` sentinel ends the
//!      list so the inner loop needs no length field.
//!
//!   4. **Early global termination** — the beam loop exits the instant the
//!      nearest remaining candidate is farther than the current worst kept
//!      result *and* the result set is already full.  Checked at the very
//!      top of the loop, before any memory access.
//!
//!   5. **Unsafe unchecked indexing** — vectors, link rows, and the visited
//!      table are all accessed via `get_unchecked` in the inner loop.
//!      Invariants are established at build time and documented inline.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use crate::types::RawData;

// ── Ordered-float newtype for BinaryHeap ─────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}
impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ── 1. Global-query-counter visited set ──────────────────────────────────
//
// `marks[id] == gen`  →  node already visited this query.
// New query: `gen += 1`  (O(1), no memset).
// Every 2^32 queries the array is bulk-zeroed and gen wraps to 1.

struct VisitedTable {
    marks:      Vec<u32>,
    generation: u32,   // `gen` is reserved in edition 2024
}

impl VisitedTable {
    fn new(n: usize) -> Self { Self { marks: vec![0u32; n], generation: 1 } }

    fn ensure_capacity(&mut self, n: usize) {
        if self.marks.len() < n { self.marks.resize(n, 0); }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 { self.marks.fill(0); self.generation = 1; }
    }

    /// Returns `true` if this is the first visit this query; marks the node.
    /// Safety: caller must guarantee `id < marks.len()`.
    #[inline(always)]
    unsafe fn visit_unchecked(&mut self, id: u32) -> bool {
        // Rust 2024: unsafe body of unsafe fn still requires explicit unsafe block.
        unsafe {
            let slot = self.marks.get_unchecked_mut(id as usize);
            if *slot == self.generation { return false; }
            *slot = self.generation;
            true
        }
    }
}

// All per-query scratch lives here.  Thread-local → zero allocation per query.
struct QueryBufs {
    visited: VisitedTable,
    cands:   BinaryHeap<Reverse<(OrdF32, u32)>>,   // min-heap: closest candidate first
    results: BinaryHeap<(OrdF32, u32)>,             // max-heap: farthest kept result first
}

impl QueryBufs {
    fn new() -> Self {
        Self { visited: VisitedTable::new(0), cands: BinaryHeap::new(), results: BinaryHeap::new() }
    }
}

thread_local! {
    static QBUFS: RefCell<QueryBufs> = RefCell::new(QueryBufs::new());
}

// ── 2. Distance kernel — two 256-bit YMM registers ───────────────────────
//
// Vectors are padded to 16 floats.  Elements 14 & 15 are always 0.0 so
// they contribute zero to the squared distance.
//
// With `target-feature=+avx2,+fma` (set in .cargo/config.toml) this
// compiles to the AVX2 path at zero runtime cost.

// AVX2 + FMA fast path — compiled only when those features are available.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2",
    target_feature = "fma"
))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn sq_dist_avx2(a: *const f32, b: *const f32) -> f32 {
    // Rust 2024: unsafe operations inside an `unsafe fn` still require an
    // explicit `unsafe {}` block.
    unsafe {
        use std::arch::x86_64::*;
        // Load two 256-bit blocks (16 floats total)
        let a0  = _mm256_loadu_ps(a);
        let b0  = _mm256_loadu_ps(b);
        let a1  = _mm256_loadu_ps(a.add(8));
        let b1  = _mm256_loadu_ps(b.add(8));
        // d0 = a0 − b0;  acc = d0²
        let d0  = _mm256_sub_ps(a0, b0);
        let acc = _mm256_mul_ps(d0, d0);
        // d1 = a1 − b1;  acc = d1² + acc  (FMA)
        let d1  = _mm256_sub_ps(a1, b1);
        let acc = _mm256_fmadd_ps(d1, d1, acc);
        // Horizontal sum: 8 lanes → 1 scalar
        let lo  = _mm256_castps256_ps128(acc);
        let hi  = _mm256_extractf128_ps(acc, 1);
        let s4  = _mm_add_ps(lo, hi);
        let s2  = _mm_hadd_ps(s4, s4);
        let s1  = _mm_hadd_ps(s2, s2);
        _mm_cvtss_f32(s1)
    }
}

#[inline(always)]
fn sq_dist_16(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    // Safety: pointers point to valid 16-float arrays; AVX2+FMA are guaranteed
    // by the compile-time cfg flags above.
    return unsafe { sq_dist_avx2(a.as_ptr(), b.as_ptr()) };

    // Scalar fallback — fully unrolled; LLVM will auto-vectorise this too.
    #[allow(unreachable_code)]
    {
        let d0  = a[0]  - b[0];  let d1  = a[1]  - b[1];
        let d2  = a[2]  - b[2];  let d3  = a[3]  - b[3];
        let d4  = a[4]  - b[4];  let d5  = a[5]  - b[5];
        let d6  = a[6]  - b[6];  let d7  = a[7]  - b[7];
        let d8  = a[8]  - b[8];  let d9  = a[9]  - b[9];
        let d10 = a[10] - b[10]; let d11 = a[11] - b[11];
        let d12 = a[12] - b[12]; let d13 = a[13] - b[13];
        d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
        d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13
    }
}

// Sentinel value for unused slots in the flat link array.
const NONE: u32 = u32::MAX;

// ── Build-phase mutable graph ─────────────────────────────────────────────

struct BuildBufs {
    visited: std::collections::HashSet<u32>,
    cands:   BinaryHeap<Reverse<(OrdF32, u32)>>,
    results: BinaryHeap<(OrdF32, u32)>,
    out:     Vec<(f32, u32)>,
    shrink:  Vec<(f32, u32)>,
}

impl BuildBufs {
    fn new(ef: usize, m: usize) -> Self {
        let cap = ef * m;
        Self {
            visited: std::collections::HashSet::with_capacity(cap),
            cands:   BinaryHeap::with_capacity(ef * 2),
            results: BinaryHeap::with_capacity(ef + 1),
            out:     Vec::with_capacity(ef),
            shrink:  Vec::with_capacity(m * 2 + 2),
        }
    }
}

struct BuildGraph {
    vectors:     Vec<[f32; 16]>,
    labels:      Vec<bool>,
    connections: Vec<Vec<Vec<u32>>>,  // [node][level] = neighbour ids
    entry:       u32,
    max_level:   usize,
    m:           usize,
    m_max:       usize,
    m_max0:      usize,
    ef_con:      usize,
    ml:          f64,
    rng:         u64,
}

impl BuildGraph {
    fn new(m: usize, ef_construction: usize) -> Self {
        Self {
            vectors: Vec::new(), labels: Vec::new(), connections: Vec::new(),
            entry: 0, max_level: 0,
            m, m_max: m, m_max0: m * 2,
            ef_con: ef_construction,
            ml: 1.0 / (m as f64).ln(),
            rng: 0x6c62272e07bb0142,
        }
    }

    fn rand_level(&mut self) -> usize {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        let u = (self.rng >> 11) as f64 / (1u64 << 53) as f64;
        (-u.max(f64::MIN_POSITIVE).ln() * self.ml) as usize
    }

    #[inline(always)]
    fn dist(&self, q: &[f32; 16], n: u32) -> f32 {
        sq_dist_16(q, unsafe { self.vectors.get_unchecked(n as usize) })
    }

    fn greedy_single(&self, q: &[f32; 16], mut cur: u32, level: usize) -> u32 {
        let mut cur_d = self.dist(q, cur);
        loop {
            let mut improved = false;
            let conns = &self.connections[cur as usize];
            if level < conns.len() {
                for &nb in &conns[level] {
                    let d = self.dist(q, nb);
                    if d < cur_d { cur_d = d; cur = nb; improved = true; }
                }
            }
            if !improved { break; }
        }
        cur
    }

    fn beam_search(&self, q: &[f32; 16], eps: &[u32], ef: usize, level: usize, b: &mut BuildBufs) {
        b.visited.clear(); b.cands.clear(); b.results.clear(); b.out.clear();
        for &ep in eps {
            if b.visited.insert(ep) {
                let d = self.dist(q, ep);
                b.cands.push(Reverse((OrdF32(d), ep)));
                b.results.push((OrdF32(d), ep));
            }
        }
        while let Some(Reverse((OrdF32(cd), c))) = b.cands.pop() {
            if let Some(&(OrdF32(fd), _)) = b.results.peek() {
                if cd > fd && b.results.len() >= ef { break; }
            }
            let conns = &self.connections[c as usize];
            if level >= conns.len() { continue; }
            for &nb in &conns[level] {
                if !b.visited.insert(nb) { continue; }
                let ed = self.dist(q, nb);
                let worst = b.results.peek().map(|&(OrdF32(d), _)| d).unwrap_or(f32::INFINITY);
                if b.results.len() < ef || ed < worst {
                    b.cands.push(Reverse((OrdF32(ed), nb)));
                    b.results.push((OrdF32(ed), nb));
                    if b.results.len() > ef { b.results.pop(); }
                }
            }
        }
        b.out.extend(b.results.drain().map(|(OrdF32(d), id)| (d, id)));
        b.out.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    fn insert(&mut self, vector: [f32; 16], is_fraud: bool, b: &mut BuildBufs) {
        let new_id = self.vectors.len() as u32;
        self.vectors.push(vector);
        self.labels.push(is_fraud);
        let level = self.rand_level();

        if new_id == 0 {
            self.connections.push((0..=level).map(|_| Vec::new()).collect());
            self.entry = 0; self.max_level = level;
            return;
        }

        let l = self.max_level;
        let mut ep = vec![self.entry];

        for lc in (level + 1..=l).rev() {
            let n = self.greedy_single(&vector, ep[0], lc);
            ep.clear(); ep.push(n);
        }

        let mut new_conns: Vec<Vec<u32>> = (0..=level).map(|_| Vec::new()).collect();

        for lc in (0..=l.min(level)).rev() {
            self.beam_search(&vector, &ep, self.ef_con, lc, b);
            let m_at = if lc == 0 { self.m_max0 } else { self.m_max };
            let m_sel = self.m.min(m_at);

            new_conns[lc].clear();
            new_conns[lc].extend(b.out.iter().take(m_sel).map(|&(_, id)| id));

            for i in 0..new_conns[lc].len() {
                let nb = new_conns[lc][i];
                while self.connections[nb as usize].len() <= lc {
                    self.connections[nb as usize].push(Vec::new());
                }
                self.connections[nb as usize][lc].push(new_id);

                if self.connections[nb as usize][lc].len() > m_at {
                    let nb_vec: [f32; 16] = self.vectors[nb as usize];
                    b.shrink.clear();
                    b.shrink.extend(
                        self.connections[nb as usize][lc]
                            .iter()
                            .map(|&id| (sq_dist_16(&nb_vec, &self.vectors[id as usize]), id)),
                    );
                    b.shrink.sort_unstable_by(|a, c| a.0.partial_cmp(&c.0).unwrap_or(std::cmp::Ordering::Equal));
                    self.connections[nb as usize][lc].clear();
                    self.connections[nb as usize][lc]
                        .extend(b.shrink.iter().take(m_at).map(|&(_, id)| id));
                }
            }
            ep.clear();
            ep.extend(b.out.iter().map(|&(_, id)| id));
        }

        if level > self.max_level { self.entry = new_id; self.max_level = level; }
        self.connections.push(new_conns);
    }

    // Consume the mutable graph, freeze into compact static representation.
    fn freeze(self, ef_search: usize, k: usize) -> StaticHNSW {
        let n    = self.vectors.len();
        let m0   = self.m_max0;

        // 3. Flat fixed-stride level-0 link array.
        // Slot layout per node: [nb0, nb1, ..., nbK, NONE, NONE, ..., NONE]
        // Slots are filled left-to-right so `NONE` terminates the scan.
        let mut links0 = vec![NONE; n * m0];
        for (id, conns) in self.connections.iter().enumerate() {
            if conns.is_empty() { continue; }
            let base = id * m0;
            for (i, &nb) in conns[0].iter().enumerate() {
                links0[base + i] = nb;
            }
        }

        // Upper-level connections (level ≥ 1); stored only for nodes that need them.
        let mut upper: HashMap<u32, Box<[Box<[u32]>]>> = HashMap::new();
        for (id, conns) in self.connections.iter().enumerate() {
            if conns.len() > 1 {
                let uc: Box<[Box<[u32]>]> = conns[1..]
                    .iter()
                    .map(|v| v.clone().into_boxed_slice())
                    .collect();
                upper.insert(id as u32, uc);
            }
        }

        StaticHNSW {
            vectors:   self.vectors.into_boxed_slice(),
            labels:    self.labels.into_boxed_slice(),
            links0:    links0.into_boxed_slice(),
            m_max0:    m0 as u32,
            upper,
            entry:     self.entry,
            max_level: self.max_level,
            ef_search,
            k,
        }
    }
}

// ── Public static index ───────────────────────────────────────────────────

pub struct StaticHNSW {
    /// 14-dim vectors padded to 16 floats (elements 14 & 15 = 0.0).
    vectors:   Box<[[f32; 16]]>,
    labels:    Box<[bool]>,
    /// Flat level-0 links: `links0[node * m_max0 .. (node+1) * m_max0]`.
    /// Unused trailing slots contain `NONE`; the first `NONE` ends the list.
    links0:    Box<[u32]>,
    m_max0:    u32,
    /// Upper-level connections for nodes at level ≥ 1.
    /// `upper[id][lc-1]` = neighbour ids at level `lc`.
    upper:     HashMap<u32, Box<[Box<[u32]>]>>,
    entry:     u32,
    max_level: usize,
    ef_search: usize,
    k:         usize,
}

impl StaticHNSW {
    /// Build the static index from raw training data.
    /// Uses one reused `BuildBufs` allocation for the entire construction.
    pub fn build(m: usize, ef_construction: usize, ef_search: usize, k: usize, data: Vec<RawData>) -> Self {
        let n = data.len();
        let mut g = BuildGraph::new(m, ef_construction);
        g.vectors.reserve(n);
        g.labels.reserve(n);
        g.connections.reserve(n);

        let mut b = BuildBufs::new(ef_construction, m);
        for raw in data {
            debug_assert_eq!(raw.vector.len(), 14, "vector must have 14 elements");
            let mut v = [0.0f32; 16];
            // Safety: raw.vector has 14 elements (checked above); copy is in-bounds.
            unsafe { std::ptr::copy_nonoverlapping(raw.vector.as_ptr(), v.as_mut_ptr(), 14); }
            g.insert(v, raw.label == "fraud", &mut b);
        }

        g.freeze(ef_search, k)
    }

    #[inline(always)]
    fn dist_qn(&self, q: &[f32; 16], n: u32) -> f32 {
        // Safety: n is always a valid node index (returned by the graph itself).
        sq_dist_16(q, unsafe { self.vectors.get_unchecked(n as usize) })
    }

    // Greedy ef=1 descent used for layers above level 0.
    fn greedy_single(&self, q: &[f32; 16], mut cur: u32, level: usize) -> u32 {
        let lc = level - 1;  // index into upper[cur]: upper[id][0] = level 1
        let mut cur_d = self.dist_qn(q, cur);
        loop {
            let mut improved = false;
            if let Some(uc) = self.upper.get(&cur) {
                if lc < uc.len() {
                    for &nb in uc[lc].iter() {
                        let d = self.dist_qn(q, nb);
                        if d < cur_d { cur_d = d; cur = nb; improved = true; }
                    }
                }
            }
            if !improved { break; }
        }
        cur
    }

    // Full beam search at level 0 writing results into the thread-local QueryBufs.
    fn beam_search_l0(&self, q: &[f32; 16], ep: u32, ef: usize, qb: &mut QueryBufs) {
        let n     = self.vectors.len();
        let m_max0 = self.m_max0 as usize;

        qb.visited.ensure_capacity(n);
        qb.visited.reset();
        qb.cands.clear();
        qb.results.clear();

        // Safety: ep < n by construction.
        unsafe { qb.visited.visit_unchecked(ep); }
        let d0 = self.dist_qn(q, ep);
        qb.cands.push(Reverse((OrdF32(d0), ep)));
        qb.results.push((OrdF32(d0), ep));

        while let Some(Reverse((OrdF32(cd), c))) = qb.cands.pop() {
            // ── 3. Early global termination ──────────────────────────────
            // Nearest remaining candidate already farther than our worst kept
            // result → no future candidate can improve the result set.
            if let Some(&(OrdF32(fd), _)) = qb.results.peek() {
                if cd > fd && qb.results.len() >= ef { break; }
            }

            // ── 5. Unsafe flat-stride link access ────────────────────────
            // Safety: c < n, so base + m_max0 <= links0.len() by construction.
            let base = c as usize * m_max0;
            let row  = unsafe { self.links0.get_unchecked(base..base + m_max0) };

            for &nb in row {
                if nb == NONE { break; }  // slots filled left-to-right; first NONE ends list

                // Safety: nb < n (enforced during build); n <= visited.len() (ensure_capacity).
                if !unsafe { qb.visited.visit_unchecked(nb) } { continue; }

                let ed = self.dist_qn(q, nb);
                let worst = qb.results.peek().map(|&(OrdF32(d), _)| d).unwrap_or(f32::INFINITY);
                if qb.results.len() < ef || ed < worst {
                    qb.cands.push(Reverse((OrdF32(ed), nb)));
                    qb.results.push((OrdF32(ed), nb));
                    if qb.results.len() > ef { qb.results.pop(); }
                }
            }
        }
    }

    /// Returns fraud fraction among the k nearest neighbours.
    /// Example: [legit, legit, fraud, fraud, fraud] → 3 / 5 = 0.6
    pub fn predict(&self, q: [f32; 16]) -> f32 {
        if self.vectors.is_empty() { return 0.0; }

        // Phase 1: ef=1 greedy descent through upper layers.
        let mut ep = self.entry;
        for lc in (1..=self.max_level).rev() {
            ep = self.greedy_single(&q, ep, lc);
        }

        // Phase 2: beam search at level 0; count fraud inline, no Vec allocation.
        let ef = self.ef_search.max(self.k);
        QBUFS.with(|cell| {
            let mut qb = cell.borrow_mut();
            self.beam_search_l0(&q, ep, ef, &mut qb);

            let s = qb.results.len();
            for _ in 0..s.saturating_sub(self.k) { qb.results.pop(); }

            let total = qb.results.len();
            if total == 0 { return 0.0; }

            // Safety: ids are valid indices returned by beam_search_l0.
            let fraud = qb.results.drain()
                .filter(|&(_, id)| unsafe { *self.labels.get_unchecked(id as usize) })
                .count();

            fraud as f32 / total as f32
        })
    }
}
