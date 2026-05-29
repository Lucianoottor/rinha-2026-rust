use std::ptr::NonNull;

const HEADER:  usize = 32;
const MAGIC:   &[u8] = b"IVF1";
const VERSION: u8    = 1;

const MAX_NPROBE: usize = 32;
const MAX_K:      usize = 32;

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

#[inline(always)]
pub(super) fn quantize(v: &[f32; 16]) -> [u8; 16] {
    let mut q = [0u8; 16];
    for i in 0..16 {
        q[i] = ((v[i] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
    }
    q
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct BucketMeta {
    pub start: u32,
    pub len:   u32,
}

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

#[allow(dead_code)]
pub struct StaticIVF {
    backing:     Backing,
    centroids:   NonNull<[u8; 16]>,
    buckets:     NonNull<BucketMeta>,
    vectors:     NonNull<[u8; 16]>,
    labels:      NonNull<u8>,
    n:           usize,
    n_centroids: usize,
    nprobe:      usize,
    k:           usize,
}

unsafe impl Send for StaticIVF {}
unsafe impl Sync for StaticIVF {}

impl StaticIVF {
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

    pub fn save(&self, path: &str) {
        use std::io::{BufWriter, Write};

        let f = std::fs::File::create(path).expect("create IVF file");
        let mut w = BufWriter::new(f);

        w.write_all(MAGIC).unwrap();
        w.write_all(&[VERSION, 0, 0, 0]).unwrap();
        w.write_all(&(self.n_centroids as u32).to_le_bytes()).unwrap();
        w.write_all(&(self.nprobe      as u32).to_le_bytes()).unwrap();
        w.write_all(&(self.k           as u32).to_le_bytes()).unwrap();
        w.write_all(&[0u8; 4]).unwrap();
        w.write_all(&(self.n           as u64).to_le_bytes()).unwrap();

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

    pub fn prefetch(&self) {
        let vb = unsafe { std::slice::from_raw_parts(
            self.vectors.as_ptr() as *const u8, self.n * 16) };
        let lb = unsafe { std::slice::from_raw_parts(self.labels.as_ptr(), self.n) };
        let mut acc = 0u8;
        for chunk in vb.chunks(4096) { acc = acc.wrapping_add(chunk[0]); }
        for chunk in lb.chunks(4096) { acc = acc.wrapping_add(chunk[0]); }
        std::hint::black_box(acc);
    }

    #[inline]
    pub fn predict(&self, q: [f32; 16]) -> usize {
        if self.n == 0 { return 0; }

        let q8 = quantize(&q);

        let nprobe      = self.nprobe.min(MAX_NPROBE);
        let k           = self.k.min(MAX_K);
        let n_centroids = self.n_centroids;

        let centroids = unsafe { std::slice::from_raw_parts(self.centroids.as_ptr(), n_centroids) };
        let buckets   = unsafe { std::slice::from_raw_parts(self.buckets.as_ptr(),   n_centroids) };

        let mut top_dists = [u32::MAX; MAX_NPROBE];
        let mut top_ids   = [0u32;     MAX_NPROBE];
        let mut worst_top_dist = u32::MAX;
        let mut worst_top_pos  = 0usize;
        let mut filled = 0usize;

        for c in 0..n_centroids {
            let d = sq_dist_u8(&q8, &centroids[c]);

            if filled < nprobe {
                top_dists[filled] = d;
                top_ids[filled]   = c as u32;
                filled += 1;

                if filled == nprobe {
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
                worst_top_dist = 0;
                for i in 0..nprobe {
                    if top_dists[i] > worst_top_dist {
                        worst_top_dist = top_dists[i];
                        worst_top_pos  = i;
                    }
                }
            }
        }

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

        knn_label[..knn_size].iter().filter(|&&l| l != 0).count()
    }
}
