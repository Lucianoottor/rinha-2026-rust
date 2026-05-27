use std::ptr::NonNull;

// ── Constants ─────────────────────────────────────────────────────────────────

const HEADER:  usize = 32;
const MAGIC:   &[u8] = b"IVF1";
const VERSION: u8    = 1;

/// Hard upper bounds (stack-allocated scratch in predict, no heap touch in hot path).
const MAX_NPROBE: usize = 32;
const MAX_K:      usize = 32;

// ── Distance & quantization ───────────────────────────────────────────────────

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn sq_dist_u8_avx2(a: *const u8, b: *const u8) -> u32 {
    unsafe {
        use std::arch::x86_64::*;
        let a16  = _mm256_cvtepu8_epi16(_mm_loadu_si128(a as *const __m128i));
        let b16  = _mm256_cvtepu8_epi16(_mm_loadu_si128(b as *const __m128i));
        let diff = _mm256_sub_epi16(a16, b16);
        let sq   = _mm256_madd_epi16(diff, diff);
        let lo   = _mm256_castsi256_si128(sq);
        let hi   = _mm256_extracti128_si256(sq, 1);
        let s4   = _mm_add_epi32(lo, hi);
        let s2   = _mm_hadd_epi32(s4, s4);
        let s1   = _mm_hadd_epi32(s2, s2);
        _mm_cvtsi128_si32(s1) as u32
    }
}

/// Squared L2 distance between two quantized 16-d vectors.
#[inline(always)]
pub(super) fn sq_dist_u8(a: &[u8; 16], b: &[u8; 16]) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    return unsafe { sq_dist_u8_avx2(a.as_ptr(), b.as_ptr()) };
    #[allow(unreachable_code)]
    {
        let mut s = 0u32;
        for i in 0..16 {
            let d = a[i] as i32 - b[i] as i32;
            s += (d * d) as u32;
        }
        s
    }
}

/// Map [−1, 1] f32 → u8.  Must match the same mapping used at build time.
#[inline(always)]
pub(super) fn quantize(v: &[f32; 16]) -> [u8; 16] {
    let mut q = [0u8; 16];
    for i in 0..16 {
        q[i] = ((v[i] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
    }
    q
}

// ── Bucket metadata (8 bytes, C-repr for direct mmap access) ─────────────────

/// Offset and length of one Voronoi cell inside the sorted vector array.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct BucketMeta {
    pub start: u32,
    pub len:   u32,
}

// ── Backing storage ───────────────────────────────────────────────────────────

#[allow(dead_code)]
enum Backing {
    Owned {
        centroids: Box<[[u8; 16]]>,
        buckets:   Box<[BucketMeta]>,
        vectors:   Box<[[u8; 16]]>,
        labels:    Box<[u8]>,
    },
    Mapped(memmap2::Mmap),
}

// ── Public index struct ───────────────────────────────────────────────────────

/// Flat IVF index backed by either owned memory (after build) or an mmap'd file.
///
/// Two instances pointing at the same file share physical pages through the OS
/// page cache — identical to the previous HNSW approach, but at ~51 MB vs the
/// HNSW's ~240 MB for 3 M vectors.
#[allow(dead_code)]
pub struct StaticIVF {
    backing:     Backing,
    centroids:   NonNull<[u8; 16]>,   // n_centroids quantized centroids
    buckets:     NonNull<BucketMeta>, // n_centroids × (start, len)
    vectors:     NonNull<[u8; 16]>,   // n vectors, sorted by centroid assignment
    labels:      NonNull<u8>,         // n labels (0 = legit, 1 = fraud)
    n:           usize,
    n_centroids: usize,
    nprobe:      usize,               // embedded in file; controls recall vs speed
    k:           usize,               // k-NN count; indexes into server::RESPONSES
}

unsafe impl Send for StaticIVF {}
unsafe impl Sync for StaticIVF {}

// ── Binary file layout ────────────────────────────────────────────────────────
//
//  [0..4]   b"IVF1"
//  [4]      version = 1
//  [5..8]   pad
//  [8..12]  n_centroids : u32
//  [12..16] nprobe      : u32
//  [16..20] k           : u32
//  [20..24] pad
//  [24..32] n           : u64
//  ── HEADER = 32 bytes ──
//  [32 ..]                  centroids  [u8;16] × n_centroids
//  [32 + n_c*16 ..]         buckets    (start:u32, len:u32) × n_centroids
//  [prev + n_c*8 ..]        vectors    [u8;16] × n   (bucket-sorted)
//  [prev + n*16 ..]         labels     [u8] × n
//
// File size for n=3 M, n_c=1024:
//   32 + 1024*16 + 1024*8 + 3_000_000*16 + 3_000_000 = ~51 MB

impl StaticIVF {
    // ── Internal constructor (builder → index) ────────────────────────────

    pub(super) fn from_parts(
        centroids: Vec<[u8; 16]>,
        buckets:   Vec<BucketMeta>,
        vectors:   Vec<[u8; 16]>,
        labels:    Vec<u8>,
        nprobe:    usize,
        k:         usize,
    ) -> Self {
        let n_centroids = centroids.len();
        let n           = vectors.len();

        let mut c_box: Box<[[u8; 16]]>   = centroids.into_boxed_slice();
        let mut b_box: Box<[BucketMeta]> = buckets.into_boxed_slice();
        let mut v_box: Box<[[u8; 16]]>   = vectors.into_boxed_slice();
        let mut l_box: Box<[u8]>         = labels.into_boxed_slice();

        let cptr = NonNull::new(c_box.as_mut_ptr()).unwrap();
        let bptr = NonNull::new(b_box.as_mut_ptr()).unwrap();
        let vptr = NonNull::new(v_box.as_mut_ptr()).unwrap();
        let lptr = NonNull::new(l_box.as_mut_ptr()).unwrap();

        StaticIVF {
            backing: Backing::Owned {
                centroids: c_box,
                buckets:   b_box,
                vectors:   v_box,
                labels:    l_box,
            },
            centroids: cptr,
            buckets:   bptr,
            vectors:   vptr,
            labels:    lptr,
            n, n_centroids, nprobe, k,
        }
    }

    // ── Persistence ───────────────────────────────────────────────────────

    pub fn save(&self, path: &str) {
        use std::io::{BufWriter, Write};

        let f = std::fs::File::create(path).expect("create IVF file");
        let mut w = BufWriter::new(f);

        w.write_all(MAGIC).unwrap();
        w.write_all(&[VERSION, 0, 0, 0]).unwrap();
        w.write_all(&(self.n_centroids as u32).to_le_bytes()).unwrap();
        w.write_all(&(self.nprobe      as u32).to_le_bytes()).unwrap();
        w.write_all(&(self.k           as u32).to_le_bytes()).unwrap();
        w.write_all(&[0u8; 4]).unwrap();                                // pad [20..24]
        w.write_all(&(self.n           as u64).to_le_bytes()).unwrap(); // [24..32]

        let cb = unsafe { std::slice::from_raw_parts(
            self.centroids.as_ptr() as *const u8, self.n_centroids * 16) };
        w.write_all(cb).unwrap();

        let bb = unsafe { std::slice::from_raw_parts(
            self.buckets.as_ptr() as *const u8, self.n_centroids * 8) };
        w.write_all(bb).unwrap();

        let vb = unsafe { std::slice::from_raw_parts(
            self.vectors.as_ptr() as *const u8, self.n * 16) };
        w.write_all(vb).unwrap();

        let lb = unsafe { std::slice::from_raw_parts(self.labels.as_ptr(), self.n) };
        w.write_all(lb).unwrap();
    }

    pub fn load(path: &str) -> Self {
        use memmap2::MmapOptions;

        let file = std::fs::File::open(path).expect("open IVF file");
        // populate() faults in all pages immediately so queries see no page faults.
        let mmap = unsafe {
            MmapOptions::new().populate().map(&file).expect("mmap IVF file")
        };

        assert_eq!(&mmap[..4], MAGIC,   "bad IVF magic — wrong index type?");
        assert_eq!(mmap[4],    VERSION, "unsupported IVF version");

        let n_centroids = u32::from_le_bytes(mmap[ 8..12].try_into().unwrap()) as usize;
        let nprobe      = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
        let k           = u32::from_le_bytes(mmap[16..20].try_into().unwrap()) as usize;
        let n           = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;

        let c_off = HEADER;
        let b_off = c_off + n_centroids * 16;
        let v_off = b_off + n_centroids *  8;
        let l_off = v_off + n           * 16;

        let centroids = NonNull::new(mmap[c_off..].as_ptr() as *mut [u8; 16]).unwrap();
        let buckets   = NonNull::new(mmap[b_off..].as_ptr() as *mut BucketMeta).unwrap();
        let vectors   = NonNull::new(mmap[v_off..].as_ptr() as *mut [u8; 16]).unwrap();
        let labels    = NonNull::new(mmap[l_off..].as_ptr() as *mut u8).unwrap();

        StaticIVF {
            backing: Backing::Mapped(mmap),
            centroids, buckets, vectors, labels,
            n, n_centroids, nprobe, k,
        }
    }

    // ── Warm-up (touch every page once so queries run cold-free) ─────────

    pub fn prefetch(&self) {
        let vb = unsafe { std::slice::from_raw_parts(
            self.vectors.as_ptr() as *const u8, self.n * 16) };
        let lb = unsafe { std::slice::from_raw_parts(self.labels.as_ptr(), self.n) };
        let mut acc = 0u8;
        for chunk in vb.chunks(4096) { acc = acc.wrapping_add(chunk[0]); }
        for chunk in lb.chunks(4096) { acc = acc.wrapping_add(chunk[0]); }
        std::hint::black_box(acc);
    }

    // ── Query ─────────────────────────────────────────────────────────────
    //
    // Hot path — zero heap allocations:
    //   • centroid selection uses stack arrays (nprobe ≤ MAX_NPROBE = 32)
    //   • k-NN result uses stack arrays   (k      ≤ MAX_K      = 32)
    //
    // Algorithm:
    //   1. Compute sq-dist from query to every centroid (SIMD u8 kernel).
    //   2. Pick the nprobe nearest centroids with a linear max-replacement scan.
    //   3. Iterate over each probed bucket sequentially, maintaining a
    //      size-k max-heap of (dist, label) on the stack.
    //   4. Return count of fraud labels in the heap.

    #[inline]
    pub fn predict(&self, q: [f32; 16]) -> usize {
        if self.n == 0 { return 0; }

        let q8 = quantize(&q);

        let nprobe      = self.nprobe.min(MAX_NPROBE);
        let k           = self.k.min(MAX_K);
        let n_centroids = self.n_centroids;

        let centroids = unsafe { std::slice::from_raw_parts(self.centroids.as_ptr(), n_centroids) };
        let buckets   = unsafe { std::slice::from_raw_parts(self.buckets.as_ptr(),   n_centroids) };

        // ── Step 1 + 2: find nprobe nearest centroids in one pass ────────
        //
        // Maintain a size-nprobe set as a flat array with explicit tracking
        // of the worst (largest-dist) entry.  Replacement cost = O(nprobe) ≤ 32.

        let mut top_dists = [u32::MAX; MAX_NPROBE];
        let mut top_ids   = [0u32;     MAX_NPROBE];
        let mut worst_top_dist = u32::MAX;
        let mut worst_top_pos  = 0usize;
        let mut filled = 0usize;

        for c in 0..n_centroids {
            let d = sq_dist_u8(&q8, &centroids[c]);

            if filled < nprobe {
                // Fill up the first nprobe slots without pruning.
                top_dists[filled] = d;
                top_ids[filled]   = c as u32;
                filled += 1;

                if filled == nprobe {
                    // Compute initial worst so pruning kicks in from here.
                    worst_top_dist = 0;
                    for i in 0..nprobe {
                        if top_dists[i] > worst_top_dist {
                            worst_top_dist = top_dists[i];
                            worst_top_pos  = i;
                        }
                    }
                }
            } else if d < worst_top_dist {
                top_dists[worst_top_pos] = d;
                top_ids[worst_top_pos]   = c as u32;
                // Recompute worst among the nprobe slots.
                worst_top_dist = 0;
                for i in 0..nprobe {
                    if top_dists[i] > worst_top_dist {
                        worst_top_dist = top_dists[i];
                        worst_top_pos  = i;
                    }
                }
            }
        }

        // ── Step 3: scan probed buckets, maintain size-k result set ──────

        let mut knn_dist  = [u32::MAX; MAX_K];
        let mut knn_label = [0u8;      MAX_K];
        let mut knn_size  = 0usize;
        let mut worst_knn_dist = u32::MAX;
        let mut worst_knn_pos  = 0usize;

        for pi in 0..filled {
            let c    = top_ids[pi] as usize;
            let meta = &buckets[c];
            let start = meta.start as usize;
            let end   = start + meta.len as usize;

            for i in start..end {
                let d   = sq_dist_u8(&q8, unsafe { &*self.vectors.as_ptr().add(i) });
                let lbl = unsafe { *self.labels.as_ptr().add(i) };

                if knn_size < k {
                    knn_dist[knn_size]  = d;
                    knn_label[knn_size] = lbl;
                    knn_size += 1;

                    if knn_size == k {
                        // Compute initial worst; pruning starts now.
                        worst_knn_dist = 0;
                        for j in 0..k {
                            if knn_dist[j] > worst_knn_dist {
                                worst_knn_dist = knn_dist[j];
                                worst_knn_pos  = j;
                            }
                        }
                    }
                } else if d < worst_knn_dist {
                    knn_dist[worst_knn_pos]  = d;
                    knn_label[worst_knn_pos] = lbl;
                    worst_knn_dist = 0;
                    for j in 0..k {
                        if knn_dist[j] > worst_knn_dist {
                            worst_knn_dist = knn_dist[j];
                            worst_knn_pos  = j;
                        }
                    }
                }
            }
        }

        // ── Step 4: count fraud labels in k-NN ───────────────────────────
        knn_label[..knn_size].iter().filter(|&&l| l != 0).count()
    }
}
