// ── Public static index ───────────────────────────────────────────────────

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::ptr::NonNull;

use super::distance::{sq_dist_u8, quantize};
use super::OrdF32;
use super::NONE;

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
    /// Build a StaticHNSW from the build-phase graph data.
    pub(super) fn from_build_graph(
        vectors: Vec<[f32; 16]>,
        labels: Vec<bool>,
        connections: Vec<Vec<Vec<u32>>>,
        entry: u32,
        max_level: usize,
        m_max0: usize,
        ef_search: usize,
        k: usize,
    ) -> Self {
        let n = vectors.len();

        // Quantise f32 → u8 (4× smaller; distance ordering preserved)
        let q_vecs: Vec<[u8; 16]> = vectors.iter().map(quantize).collect();
        let labels:  Vec<u8>      = labels.iter().map(|&b| b as u8).collect();

        // Flat fixed-stride level-0 link array
        let mut links0_vec = vec![NONE; n * m_max0];
        for (id, conns) in connections.iter().enumerate() {
            if conns.is_empty() { continue; }
            let base = id * m_max0;
            for (i, &nb) in conns[0].iter().enumerate() { links0_vec[base + i] = nb; }
        }

        // Upper-level connections (only nodes at level ≥ 1 — small fraction)
        let mut upper: HashMap<u32, Box<[Box<[u32]>]>> = HashMap::new();
        for (id, conns) in connections.iter().enumerate() {
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
            links0_n:  n * m_max0,
            m_max0,
            upper,
            entry,
            max_level,
            ef_search,
            k,
        }
    }

    /// Touch every page of the hot arrays sequentially to pre-fault them into the OS
    /// page cache. One byte per 4KB page is enough; sequential access triggers OS
    /// read-ahead so the actual I/O is pipelined. Call before accepting traffic.
    pub fn prefetch(&self) {
        let vb = unsafe { std::slice::from_raw_parts(self.vectors.as_ptr() as *const u8, self.n * 16) };
        let lb = unsafe { std::slice::from_raw_parts(self.links0.as_ptr() as *const u8, self.links0_n * 4) };
        let mut acc = 0u8;
        for chunk in vb.chunks(4096) { acc = acc.wrapping_add(chunk[0]); }
        for chunk in lb.chunks(4096) { acc = acc.wrapping_add(chunk[0]); }
        std::hint::black_box(acc);
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
        unsafe { libc::mlock(mmap.as_ptr() as *const _, mmap.len()); }

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

    pub fn predict(&self, q: [f32; 16]) -> usize {
        if self.n == 0 { return 0; }

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

            qb.results.drain()
                .filter(|&(_, id)| unsafe { *self.labels.as_ptr().add(id as usize) != 0 })
                .count()
        })
    }
}
