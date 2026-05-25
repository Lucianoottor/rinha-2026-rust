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
pub fn sq_dist_16(a: &[f32; 16], b: &[f32; 16]) -> f32 {
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
pub fn sq_dist_u8(a: &[u8; 16], b: &[u8; 16]) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    return unsafe { sq_dist_u8_avx2(a.as_ptr(), b.as_ptr()) };
    #[allow(unreachable_code)]
    {
        let mut s = 0u32;
        for i in 0..14 { let d = a[i] as i32 - b[i] as i32; s += (d * d) as u32; }
        s
    }
}

#[inline(always)]
pub fn quantize(v: &[f32; 16]) -> [u8; 16] {
    let mut q = [0u8; 16];
    for i in 0..16 { q[i] = ((v[i] + 1.0) * 127.5).clamp(0.0, 255.0) as u8; }
    q
}
