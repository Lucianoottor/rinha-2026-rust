//! Static HNSW for 14-dim normalised vectors.
//!
//! Build stores f32[16] for accuracy; freeze quantises to u8[16] (4× smaller).
//! Saved index is a flat binary file; loaded via memmap2 so two Docker
//! instances share the same physical pages (MAP_SHARED, OS page cache).

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::ptr::NonNull;
use crate::types::RawData;

// ── Ordered-float newtype ─────────────────────────────────────────────────

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

// ── Generation-counter visited set (O(1) reset) ───────────────────────────

struct VisitedTable {
    marks:      Vec<u32>,
    generation: u32,
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

    #[inline(always)]
    unsafe fn visit_unchecked(&mut self, id: u32) -> bool {
        unsafe {
            let slot = self.marks.get_unchecked_mut(id as usize);
            if *slot == self.generation { return false; }
            *slot = self.generation;
            true
        }
    }
}

struct QueryBufs {
    visited: VisitedTable,
    cands:   BinaryHeap<Reverse<(OrdF32, u32)>>,
    results: BinaryHeap<(OrdF32, u32)>,
}

impl QueryBufs {
    fn new() -> Self {
        Self { visited: VisitedTable::new(0), cands: BinaryHeap::new(), results: BinaryHeap::new() }
    }
}

thread_local! {
    static QBUFS: RefCell<QueryBufs> = RefCell::new(QueryBufs::new());
}

// ── Distance kernels ──────────────────────────────────────────────────────

// f32 path — build phase only
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2",
    target_feature = "fma"
))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn sq_dist_avx2_f32(a: *const f32, b: *const f32) -> f32 {
    unsafe {
        use std::arch::x86_64::*;
        let a0  = _mm256_loadu_ps(a);
        let b0  = _mm256_loadu_ps(b);
        let a1  = _mm256_loadu_ps(a.add(8));
        let b1  = _mm256_loadu_ps(b.add(8));
        let d0  = _mm256_sub_ps(a0, b0);
        let acc = _mm256_mul_ps(d0, d0);
        let d1  = _mm256_sub_ps(a1, b1);
        let acc = _mm256_fmadd_ps(d1, d1, acc);
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
    return unsafe { sq_dist_avx2_f32(a.as_ptr(), b.as_ptr()) };
    #[allow(unreachable_code)]
    {
        let mut s = 0f32;
        for i in 0..14 { let d = a[i] - b[i]; s += d * d; }
        s
    }
}

// u8 path — query phase
// _mm256_cvtepu8_epi16: 16×u8 → 16×i16 (one 128-bit load → 256-bit)
// _mm256_madd_epi16(d,d): adjacent pairs of i16 squares summed → 8×i32
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn sq_dist_u8_avx2(a: *const u8, b: *const u8) -> u32 {
    unsafe {
        use std::arch::x86_64::*;
        let a16  = _mm256_cvtepu8_epi16(_mm_loadu_si128(a as *const __m128i));
        let b16  = _mm256_cvtepu8_epi16(_mm_loadu_si128(b as *const __m128i));
        let diff = _mm256_sub_epi16(a16, b16);
        let sq   = _mm256_madd_epi16(diff, diff);  // 8×i32
        let lo   = _mm256_castsi256_si128(sq);
        let hi   = _mm256_extracti128_si256(sq, 1);
        let s4   = _mm_add_epi32(lo, hi);
        let s2   = _mm_hadd_epi32(s4, s4);
        let s1   = _mm_hadd_epi32(s2, s2);
        _mm_cvtsi128_si32(s1) as u32
    }
}

#[inline(always)]
fn sq_dist_u8(a: &[u8; 16], b: &[u8; 16]) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    return unsafe { sq_dist_u8_avx2(a.as_ptr(), b.as_ptr()) };
    #[allow(unreachable_code)]
    {
        let mut s = 0u32;
        for i in 0..14 { let d = a[i] as i32 - b[i] as i32; s += (d * d) as u32; }
        s
    }
}

/// Map [0,1] f32 → u8.  Values outside range are clamped.
#[inline(always)]
fn quantize(v: &[f32; 16]) -> [u8; 16] {
    let mut q = [0u8; 16];
    for i in 0..16 { q[i] = (v[i] * 255.0).clamp(0.0, 255.0) as u8; }
    q
}

const NONE: u32 = u32::MAX;

// ── Build-phase structures (f32, dropped after freeze) ────────────────────

struct BuildBufs {
    visited: std::collections::HashSet<u32>,
    cands:   BinaryHeap<Reverse<(OrdF32, u32)>>,
    results: BinaryHeap<(OrdF32, u32)>,
    out:     Vec<(f32, u32)>,
    shrink:  Vec<(f32, u32)>,
}

impl BuildBufs {
    fn new(ef: usize, m: usize) -> Self {
        Self {
            visited: std::collections::HashSet::with_capacity(ef * m),
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
    connections: Vec<Vec<Vec<u32>>>,
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
                let ed    = self.dist(q, nb);
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
            let m_at  = if lc == 0 { self.m_max0 } else { self.m_max };
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
                    let nb_vec = self.vectors[nb as usize];
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

    fn freeze(self, ef_search: usize, k: usize) -> StaticHNSW {
        let n  = self.vectors.len();
        let m0 = self.m_max0;

        // Quantise f32 → u8 (4× smaller; distance ordering preserved)
        let q_vecs: Vec<[u8; 16]> = self.vectors.iter().map(quantize).collect();
        let labels:  Vec<u8>      = self.labels.iter().map(|&b| b as u8).collect();

        // Flat fixed-stride level-0 link array
        let mut links0_vec = vec![NONE; n * m0];
        for (id, conns) in self.connections.iter().enumerate() {
            if conns.is_empty() { continue; }
            let base = id * m0;
            for (i, &nb) in conns[0].iter().enumerate() { links0_vec[base + i] = nb; }
        }

        // Upper-level connections (only nodes at level ≥ 1 — small fraction)
        let mut upper: HashMap<u32, Box<[Box<[u32]>]>> = HashMap::new();
        for (id, conns) in self.connections.iter().enumerate() {
            if conns.len() > 1 {
                let uc: Box<[Box<[u32]>]> = conns[1..].iter()
                    .map(|v| v.clone().into_boxed_slice())
                    .collect();
                upper.insert(id as u32, uc);
            }
        }

        let mut vecs_box:  Box<[[u8; 16]]> = q_vecs.into_boxed_slice();
        let mut lbls_box:  Box<[u8]>       = labels.into_boxed_slice();
        let mut links_box: Box<[u32]>       = links0_vec.into_boxed_slice();

        // Derive raw ptrs before moving boxes into the enum.
        // Box heap data is stable — address doesn't change on Box move.
        let vptr  = NonNull::new(vecs_box.as_mut_ptr()).unwrap();
        let lptr  = NonNull::new(lbls_box.as_mut_ptr()).unwrap();
        let l0ptr = NonNull::new(links_box.as_mut_ptr()).unwrap();

        StaticHNSW {
            backing:   Backing::Owned { vectors: vecs_box, labels: lbls_box, links0: links_box },
            vectors:   vptr,
            labels:    lptr,
            links0:    l0ptr,
            n,
            links0_n:  n * m0,
            m_max0:    m0,
            upper,
            entry:     self.entry,
            max_level: self.max_level,
            ef_search,
            k,
        }
    }
}

// ── Public static index ───────────────────────────────────────────────────

/// Which allocation backs the hot arrays.
/// Fields are drop guards only — never read, but must live as long as the raw ptrs.
#[allow(dead_code)]
enum Backing {
    Owned {
        vectors: Box<[[u8; 16]]>,
        labels:  Box<[u8]>,
        links0:  Box<[u32]>,
    },
    /// The Mmap keeps the file mapping alive; raw ptrs point into it.
    Mapped(memmap2::Mmap),
}

#[allow(dead_code)]
pub struct StaticHNSW {
    backing:   Backing,
    /// Points into either the owned Box or the mmap.
    vectors:   NonNull<[u8; 16]>,
    labels:    NonNull<u8>,
    links0:    NonNull<u32>,
    n:         usize,
    links0_n:  usize,
    m_max0:    usize,
    upper:     HashMap<u32, Box<[Box<[u32]>]>>,
    entry:     u32,
    max_level: usize,
    ef_search: usize,
    k:         usize,
}

// Safety: raw ptrs are either into Box-owned heap (exclusive) or a
// read-only mmap (safe to share as immutable data across threads).
unsafe impl Send for StaticHNSW {}
unsafe impl Sync for StaticHNSW {}

// ─────────────────────────────────────────────────────────────────────────
// Binary file layout (little-endian, 48-byte header):
//   [0..4]   b"HNSW"
//   [4]      version=2
//   [5..8]   pad
//   [8..16]  n: u64
//   [16..20] m_max0: u32
//   [20..24] max_level: u32
//   [24..28] entry: u32
//   [28..32] ef_search: u32
//   [32..36] k: u32
//   [36..40] upper_cnt: u32
//   [40..48] pad
//   [48..]   vectors [[u8;16];n], labels [u8;n], (4-aligned) links0 [u32;n*m_max0], upper blob
// ─────────────────────────────────────────────────────────────────────────

impl StaticHNSW {
    pub fn build(m: usize, ef_construction: usize, ef_search: usize, k: usize, data: Vec<RawData>) -> Self {
        let n = data.len();
        let mut g = BuildGraph::new(m, ef_construction);
        g.vectors.reserve(n);
        g.labels.reserve(n);
        g.connections.reserve(n);

        let mut b = BuildBufs::new(ef_construction, m);
        for raw in data {
            debug_assert_eq!(raw.vector.len(), 14);
            let mut v = [0.0f32; 16];
            unsafe { std::ptr::copy_nonoverlapping(raw.vector.as_ptr(), v.as_mut_ptr(), 14); }
            g.insert(v, raw.label == "fraud", &mut b);
        }

        g.freeze(ef_search, k)
    }

    /// Serialize the index to a flat binary file.
    pub fn save(&self, path: &str) {
        use std::io::{BufWriter, Write};

        let f = std::fs::File::create(path).expect("create index file");
        let mut w = BufWriter::new(f);

        w.write_all(b"HNSW").unwrap();
        w.write_all(&[2u8, 0, 0, 0]).unwrap();
        w.write_all(&(self.n as u64).to_le_bytes()).unwrap();
        w.write_all(&(self.m_max0 as u32).to_le_bytes()).unwrap();
        w.write_all(&(self.max_level as u32).to_le_bytes()).unwrap();
        w.write_all(&self.entry.to_le_bytes()).unwrap();
        w.write_all(&(self.ef_search as u32).to_le_bytes()).unwrap();
        w.write_all(&(self.k as u32).to_le_bytes()).unwrap();
        w.write_all(&(self.upper.len() as u32).to_le_bytes()).unwrap();
        w.write_all(&[0u8; 8]).unwrap();  // pad to 48 bytes (4+4+8+4+4+4+4+4+4+8=48)

        let vb = unsafe { std::slice::from_raw_parts(self.vectors.as_ptr() as *const u8, self.n * 16) };
        w.write_all(vb).unwrap();

        let lb = unsafe { std::slice::from_raw_parts(self.labels.as_ptr(), self.n) };
        w.write_all(lb).unwrap();

        let pos = 48 + self.n * 17;
        let pad = (4 - pos % 4) % 4;
        w.write_all(&[0u8, 0, 0][..pad]).unwrap();

        let l0b = unsafe { std::slice::from_raw_parts(self.links0.as_ptr() as *const u8, self.links0_n * 4) };
        w.write_all(l0b).unwrap();

        for (&node_id, levels) in &self.upper {
            w.write_all(&node_id.to_le_bytes()).unwrap();
            w.write_all(&(levels.len() as u32).to_le_bytes()).unwrap();
            for level in levels.iter() {
                w.write_all(&(level.len() as u32).to_le_bytes()).unwrap();
                for &nb in level.iter() { w.write_all(&nb.to_le_bytes()).unwrap(); }
            }
        }
    }

    /// Load from a flat binary file via mmap.
    /// Both API containers map the same named-volume file; the OS shares pages.
    pub fn load(path: &str) -> Self {
        use memmap2::MmapOptions;

        let file = std::fs::File::open(path).expect("open index file");
        let mmap = unsafe { MmapOptions::new().map(&file).expect("mmap index file") };

        assert_eq!(&mmap[0..4], b"HNSW", "bad magic");
        assert_eq!(mmap[4], 2, "unsupported version");

        let n         = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let m_max0    = u32::from_le_bytes(mmap[16..20].try_into().unwrap()) as usize;
        let max_level = u32::from_le_bytes(mmap[20..24].try_into().unwrap()) as usize;
        let entry     = u32::from_le_bytes(mmap[24..28].try_into().unwrap());
        let ef_search = u32::from_le_bytes(mmap[28..32].try_into().unwrap()) as usize;
        let k         = u32::from_le_bytes(mmap[32..36].try_into().unwrap()) as usize;
        let upper_cnt = u32::from_le_bytes(mmap[36..40].try_into().unwrap()) as usize;

        let vecs_off  = 48usize;
        let lbls_off  = vecs_off + n * 16;
        let l0_off    = (lbls_off + n + 3) & !3;  // 4-byte aligned
        let upper_off = l0_off + n * m_max0 * 4;
        let links0_n  = n * m_max0;

        // Ptrs into mmap; valid for `mmap`'s lifetime (stored in Backing::Mapped below).
        let vectors = NonNull::new(mmap[vecs_off..].as_ptr() as *mut [u8; 16]).unwrap();
        let labels  = NonNull::new(mmap[lbls_off..].as_ptr() as *mut u8).unwrap();
        let links0  = NonNull::new(mmap[l0_off..].as_ptr() as *mut u32).unwrap();

        let mut upper: HashMap<u32, Box<[Box<[u32]>]>> = HashMap::new();
        let mut cur = upper_off;
        for _ in 0..upper_cnt {
            let node_id = u32::from_le_bytes(mmap[cur..cur+4].try_into().unwrap()); cur += 4;
            let nl      = u32::from_le_bytes(mmap[cur..cur+4].try_into().unwrap()) as usize; cur += 4;
            let mut lvls: Vec<Box<[u32]>> = Vec::with_capacity(nl);
            for _ in 0..nl {
                let cnt = u32::from_le_bytes(mmap[cur..cur+4].try_into().unwrap()) as usize; cur += 4;
                let mut nbs: Vec<u32> = Vec::with_capacity(cnt);
                for _ in 0..cnt {
                    nbs.push(u32::from_le_bytes(mmap[cur..cur+4].try_into().unwrap()));
                    cur += 4;
                }
                lvls.push(nbs.into_boxed_slice());
            }
            upper.insert(node_id, lvls.into_boxed_slice());
        }

        StaticHNSW {
            backing: Backing::Mapped(mmap),
            vectors, labels, links0,
            n, links0_n, m_max0,
            upper, entry, max_level, ef_search, k,
        }
    }

    #[inline(always)]
    fn dist_qn(&self, q: &[u8; 16], node: u32) -> f32 {
        // Safety: node < n by construction throughout build and search.
        sq_dist_u8(q, unsafe { &*self.vectors.as_ptr().add(node as usize) }) as f32
    }

    #[inline(always)]
    fn links0_row(&self, node: usize) -> &[u32] {
        unsafe { std::slice::from_raw_parts(self.links0.as_ptr().add(node * self.m_max0), self.m_max0) }
    }

    fn greedy_single(&self, q: &[u8; 16], mut cur: u32, level: usize) -> u32 {
        let lc = level - 1;
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

    fn beam_search_l0(&self, q: &[u8; 16], ep: u32, ef: usize, qb: &mut QueryBufs) {
        qb.visited.ensure_capacity(self.n);
        qb.visited.reset();
        qb.cands.clear();
        qb.results.clear();

        unsafe { qb.visited.visit_unchecked(ep); }
        let d0 = self.dist_qn(q, ep);
        qb.cands.push(Reverse((OrdF32(d0), ep)));
        qb.results.push((OrdF32(d0), ep));

        while let Some(Reverse((OrdF32(cd), c))) = qb.cands.pop() {
            if let Some(&(OrdF32(fd), _)) = qb.results.peek() {
                if cd > fd && qb.results.len() >= ef { break; }
            }
            for &nb in self.links0_row(c as usize) {
                if nb == NONE { break; }
                if !unsafe { qb.visited.visit_unchecked(nb) } { continue; }
                let ed    = self.dist_qn(q, nb);
                let worst = qb.results.peek().map(|&(OrdF32(d), _)| d).unwrap_or(f32::INFINITY);
                if qb.results.len() < ef || ed < worst {
                    qb.cands.push(Reverse((OrdF32(ed), nb)));
                    qb.results.push((OrdF32(ed), nb));
                    if qb.results.len() > ef { qb.results.pop(); }
                }
            }
        }
    }

    pub fn predict(&self, q: [f32; 16]) -> f32 {
        if self.n == 0 { return 0.0; }

        let q8 = quantize(&q);

        let mut ep = self.entry;
        for lc in (1..=self.max_level).rev() {
            ep = self.greedy_single(&q8, ep, lc);
        }

        let ef = self.ef_search.max(self.k);
        QBUFS.with(|cell| {
            let mut qb = cell.borrow_mut();
            self.beam_search_l0(&q8, ep, ef, &mut qb);

            let extra = qb.results.len().saturating_sub(self.k);
            for _ in 0..extra { qb.results.pop(); }

            let total = qb.results.len();
            if total == 0 { return 0.0; }

            // Safety: ids returned by beam_search_l0 are always valid node indices.
            let fraud = qb.results.drain()
                .filter(|&(_, id)| unsafe { *self.labels.as_ptr().add(id as usize) != 0 })
                .count();

            fraud as f32 / total as f32
        })
    }
}
