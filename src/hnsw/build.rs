// ── Build-phase structures (f32, dropped after freeze) ────────────────────

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::distance::sq_dist_16;
use super::index::StaticHNSW;
use super::OrdF32;
use crate::types::RawData;

pub(super) struct BuildBufs {
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

pub(super) struct BuildGraph {
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

    pub(super) fn freeze(self, ef_search: usize, k: usize) -> StaticHNSW {
        StaticHNSW::from_build_graph(self.vectors, self.labels, self.connections,
            self.entry, self.max_level, self.m_max0, ef_search, k)
    }
}

pub fn build_index(m: usize, ef_construction: usize, ef_search: usize, k: usize, data: Vec<RawData>) -> StaticHNSW {
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
