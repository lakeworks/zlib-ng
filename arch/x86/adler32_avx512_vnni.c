/* adler32_avx512_vnni.c -- compute the Adler-32 checksum of a data stream
 * Based on Brian Bockelman's AVX2 version
 * Copyright (C) 1995-2011 Mark Adler
 * Authors:
 *   Adam Stylinski <kungfujesus06@gmail.com>
 *   Brian Bockelman <bockelman@gmail.com>
 * For conditions of distribution and use, see copyright notice in zlib.h
 *
 * Zen 5 optimizations:
 *   - Non-copy path: 512-bit 4-way VPDPBUSD unroll (compute-bound, saturates
 *     2 ports x 2-cycle latency)
 *   - fold_copy path: 512-bit 2-way VPDPBUSD unroll (memory-bound, smaller
 *     icache footprint reduces pressure on deflate's tight inner loops)
 *   - Template merge: single static inline function with const int COPY,
 *     following the pattern from adler32_avx512.c / adler32_avx2.c
 */

#ifdef X86_AVX512VNNI

#include "zbuild.h"
#include "adler32_p.h"
#include "arch_functions.h"
#include <immintrin.h>
#include "x86_intrins.h"
#include "adler32_avx512_p.h"
#include "adler32_avx2_p.h"

static inline uint32_t adler32_fold_copy_vnni_impl(uint32_t adler, uint8_t *dst,
                                                    const uint8_t *src, size_t len,
                                                    const int COPY) {
    if (src == NULL) return 1L;
    if (len == 0) return adler;

    uint32_t adler0, adler1;
    adler1 = (adler >> 16) & 0xffff;
    adler0 = adler & 0xffff;

rem_peel:
    if (len < 32) {
        if (COPY) {
            __mmask32 storemask = (0xFFFFFFFFUL >> (32 - len));
            __m256i copy_vec = _mm256_maskz_loadu_epi8(storemask, src);
            _mm256_mask_storeu_epi8(dst, storemask, copy_vec);
        }
        return adler32_ssse3(adler, src, len);
    }

    if (len < 64) {
        if (COPY) {
            __mmask64 storemask = (0xFFFFFFFFFFFFFFFFULL >> (64 - len));
            __m512i copy_vec = _mm512_maskz_loadu_epi8(storemask, src);
            _mm512_mask_storeu_epi8(dst, storemask, copy_vec);
        }
        return adler32_avx2(adler, src, len);
    }

    const __m512i dot2v = _mm512_set_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                          20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
                                          38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                                          56, 57, 58, 59, 60, 61, 62, 63, 64);

    const __m512i zero = _mm512_setzero_si512();
    __m512i vs1, vs2;

    while (len >= 64) {
        vs1 = _mm512_zextsi128_si512(_mm_cvtsi32_si128(adler0));
        vs2 = _mm512_zextsi128_si512(_mm_cvtsi32_si128(adler1));
        size_t k = MIN(len, NMAX);
        k -= k % 64;
        len -= k;
        __m512i vs1_0 = vs1;
        __m512i vs3 = _mm512_setzero_si512();
        __m512i vs2_1 = _mm512_setzero_si512();

        if (COPY) {
            /* fold_copy path: 2-way unrolled, 128B per iteration.
             * Memory-bound (load+store per chunk), so 2-way is sufficient.
             * Smaller icache footprint than 4-way reduces pressure on
             * deflate_quick's tight inner loop at compression level 1. */

            while (k >= 128) {
                __m512i vbuf0 = _mm512_loadu_si512((__m512i*)src);
                __m512i vbuf1 = _mm512_loadu_si512((__m512i*)(src + 64));
                _mm512_storeu_si512((__m512i*)dst, vbuf0);
                _mm512_storeu_si512((__m512i*)(dst + 64), vbuf1);
                src += 128;
                dst += 128;
                k -= 128;

                /* chunk 0 */
                __m512i vs1_sad = _mm512_sad_epu8(vbuf0, zero);
                vs3 = _mm512_add_epi32(vs3, vs1_0);
                vs1 = _mm512_add_epi32(vs1, vs1_sad);
                vs2 = _mm512_dpbusd_epi32(vs2, vbuf0, dot2v);

                /* chunk 1 */
                vs1_sad = _mm512_sad_epu8(vbuf1, zero);
                vs3 = _mm512_add_epi32(vs3, vs1);
                vs1 = _mm512_add_epi32(vs1, vs1_sad);
                vs2_1 = _mm512_dpbusd_epi32(vs2_1, vbuf1, dot2v);

                vs1_0 = vs1;
            }

            /* Handle remaining 64-byte chunk (0 or 1 iterations) */
            if (k >= 64) {
                __m512i vbuf = _mm512_loadu_si512((__m512i*)src);
                _mm512_storeu_si512((__m512i*)dst, vbuf);
                src += 64;
                dst += 64;
                k -= 64;

                __m512i vs1_sad = _mm512_sad_epu8(vbuf, zero);
                vs3 = _mm512_add_epi32(vs3, vs1_0);
                vs1 = _mm512_add_epi32(vs1, vs1_sad);
                vs2 = _mm512_dpbusd_epi32(vs2, vbuf, dot2v);
                vs1_0 = vs1;
            }
        } else {
            /* Non-copy path: 4-way unrolled, 256B per iteration.
             * Compute-bound by VPDPBUSD chains. 4 independent accumulators
             * saturate Zen 5's 2-cycle latency x 2 execution ports. */
            __m512i vs2_2 = _mm512_setzero_si512();
            __m512i vs2_3 = _mm512_setzero_si512();

            /* Remainder peeling: process up to 3 chunks to align to 256B boundary */
            while (k >= 64 && k % 256) {
                __m512i vbuf = _mm512_loadu_si512((__m512i*)src);
                src += 64;
                k -= 64;

                __m512i vs1_sad = _mm512_sad_epu8(vbuf, zero);
                vs3 = _mm512_add_epi32(vs3, vs1_0);
                vs1 = _mm512_add_epi32(vs1, vs1_sad);
                vs2 = _mm512_dpbusd_epi32(vs2, vbuf, dot2v);
                vs1_0 = vs1;
            }

            while (k >= 256) {
                __m512i vbuf0 = _mm512_loadu_si512((__m512i*)src);
                __m512i vbuf1 = _mm512_loadu_si512((__m512i*)(src + 64));
                __m512i vbuf2 = _mm512_loadu_si512((__m512i*)(src + 128));
                __m512i vbuf3 = _mm512_loadu_si512((__m512i*)(src + 192));
                src += 256;
                k -= 256;

                /* chunk 0 */
                __m512i vs1_sad = _mm512_sad_epu8(vbuf0, zero);
                vs3 = _mm512_add_epi32(vs3, vs1_0);
                vs1 = _mm512_add_epi32(vs1, vs1_sad);
                vs2 = _mm512_dpbusd_epi32(vs2, vbuf0, dot2v);

                /* chunk 1 */
                vs1_sad = _mm512_sad_epu8(vbuf1, zero);
                vs3 = _mm512_add_epi32(vs3, vs1);
                vs1 = _mm512_add_epi32(vs1, vs1_sad);
                vs2_1 = _mm512_dpbusd_epi32(vs2_1, vbuf1, dot2v);

                /* chunk 2 */
                vs1_sad = _mm512_sad_epu8(vbuf2, zero);
                vs3 = _mm512_add_epi32(vs3, vs1);
                vs1 = _mm512_add_epi32(vs1, vs1_sad);
                vs2_2 = _mm512_dpbusd_epi32(vs2_2, vbuf2, dot2v);

                /* chunk 3 */
                vs1_sad = _mm512_sad_epu8(vbuf3, zero);
                vs3 = _mm512_add_epi32(vs3, vs1);
                vs1 = _mm512_add_epi32(vs1, vs1_sad);
                vs2_3 = _mm512_dpbusd_epi32(vs2_3, vbuf3, dot2v);

                vs1_0 = vs1;
            }

            /* Merge extra accumulators into vs2_1 for common reduction */
            vs2_1 = _mm512_add_epi32(vs2_1, vs2_2);
            vs2_1 = _mm512_add_epi32(vs2_1, vs2_3);
        }

        /* Common reduction for both paths */
        vs3 = _mm512_slli_epi32(vs3, 6);
        vs2 = _mm512_add_epi32(vs2, vs3);
        vs2 = _mm512_add_epi32(vs2, vs2_1);

        adler0 = partial_hsum(vs1) % BASE;
        adler1 = _mm512_reduce_add_epu32(vs2) % BASE;
    }

    adler = adler0 | (adler1 << 16);

    /* Process tail (len < 64). */
    if (len) {
        goto rem_peel;
    }

    return adler;
}

Z_INTERNAL uint32_t adler32_avx512_vnni(uint32_t adler, const uint8_t *src, size_t len) {
    return adler32_fold_copy_vnni_impl(adler, NULL, src, len, 0);
}

Z_INTERNAL uint32_t adler32_fold_copy_avx512_vnni(uint32_t adler, uint8_t *dst, const uint8_t *src, size_t len) {
    return adler32_fold_copy_vnni_impl(adler, dst, src, len, 1);
}

#endif
