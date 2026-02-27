/* adler32_avx512_vnni.c -- compute the Adler-32 checksum of a data stream
 * Based on Brian Bockelman's AVX2 version
 * Copyright (C) 1995-2011 Mark Adler
 * Authors:
 *   Adam Stylinski <kungfujesus06@gmail.com>
 *   Brian Bockelman <bockelman@gmail.com>
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifdef X86_AVX512VNNI

#include "zbuild.h"
#include "adler32_p.h"
#include "arch_functions.h"
#include <immintrin.h>
#include "x86_intrins.h"
#include "adler32_avx512_p.h"
#include "adler32_avx2_p.h"

Z_INTERNAL uint32_t adler32_avx512_vnni(uint32_t adler, const uint8_t *src, size_t len) {
    if (src == NULL) return 1L;
    if (len == 0) return adler;

    uint32_t adler0, adler1;
    adler1 = (adler >> 16) & 0xffff;
    adler0 = adler & 0xffff;

rem_peel:
    if (len < 32)
        return adler32_ssse3(adler, src, len);

    if (len < 64)
        return adler32_avx2(adler, src, len);

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
        /* 4 independent dpbusd accumulator chains to saturate Zen 5's 2-cycle
         * VPDPBUSD latency across 2 execution ports (0,1) */
        __m512i vs2_1 = _mm512_setzero_si512();
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

        /* 4-way unrolled loop: 256 bytes per iteration.
         * 4 independent dpbusd chains hide the 2-cycle latency on Zen 5. */
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

        vs3 = _mm512_slli_epi32(vs3, 6);
        vs2 = _mm512_add_epi32(vs2, vs3);
        vs2 = _mm512_add_epi32(vs2, vs2_1);
        vs2 = _mm512_add_epi32(vs2, vs2_2);
        vs2 = _mm512_add_epi32(vs2, vs2_3);

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

Z_INTERNAL uint32_t adler32_fold_copy_avx512_vnni(uint32_t adler, uint8_t *dst, const uint8_t *src, size_t len) {
    if (src == NULL) return 1L;
    if (len == 0) return adler;

    uint32_t adler0, adler1;
    adler1 = (adler >> 16) & 0xffff;
    adler0 = adler & 0xffff;

rem_peel_copy:
    if (len < 64) {
        if (len < 32) {
            /* Masked AVX2 copy + SSSE3 checksum for < 32 bytes */
            __mmask32 storemask = (0xFFFFFFFFUL >> (32 - len));
            __m256i copy_vec = _mm256_maskz_loadu_epi8(storemask, src);
            _mm256_mask_storeu_epi8(dst, storemask, copy_vec);
            return adler32_ssse3(adler, src, len);
        }
        /* Masked 512-bit copy + AVX2 checksum for 32-63 bytes */
        __mmask64 storemask = (0xFFFFFFFFFFFFFFFFULL >> (64 - len));
        __m512i copy_vec = _mm512_maskz_loadu_epi8(storemask, src);
        _mm512_mask_storeu_epi8(dst, storemask, copy_vec);
        return adler32_avx2(adler, src, len);
    }

    /* 512-bit position weights: byte i gets weight (64-i) for VPDPBUSD */
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
        /* 4 independent dpbusd accumulator chains to saturate Zen 5's 2-cycle
         * VPDPBUSD latency across 2 execution ports (0,1) */
        __m512i vs2_1 = _mm512_setzero_si512();
        __m512i vs2_2 = _mm512_setzero_si512();
        __m512i vs2_3 = _mm512_setzero_si512();

        /* Remainder peeling: process up to 3 chunks to align to 256B boundary */
        while (k >= 64 && k % 256) {
            __m512i vbuf = _mm512_loadu_si512((__m512i*)src);
            _mm512_storeu_si512((__m512i*)dst, vbuf);
            dst += 64;
            src += 64;
            k -= 64;

            __m512i vs1_sad = _mm512_sad_epu8(vbuf, zero);
            vs3 = _mm512_add_epi32(vs3, vs1_0);
            vs1 = _mm512_add_epi32(vs1, vs1_sad);
            vs2 = _mm512_dpbusd_epi32(vs2, vbuf, dot2v);
            vs1_0 = vs1;
        }

        /* 4-way unrolled loop: 256 bytes per iteration.
         * 4 independent dpbusd chains hide the 2-cycle latency on Zen 5.
         * 512-bit stores utilize Zen 5's full-width store port (64B/cycle). */
        while (k >= 256) {
            __m512i vbuf0 = _mm512_loadu_si512((__m512i*)src);
            __m512i vbuf1 = _mm512_loadu_si512((__m512i*)(src + 64));
            __m512i vbuf2 = _mm512_loadu_si512((__m512i*)(src + 128));
            __m512i vbuf3 = _mm512_loadu_si512((__m512i*)(src + 192));
            _mm512_storeu_si512((__m512i*)dst, vbuf0);
            _mm512_storeu_si512((__m512i*)(dst + 64), vbuf1);
            _mm512_storeu_si512((__m512i*)(dst + 128), vbuf2);
            _mm512_storeu_si512((__m512i*)(dst + 192), vbuf3);
            dst += 256;
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

        vs3 = _mm512_slli_epi32(vs3, 6);
        vs2 = _mm512_add_epi32(vs2, vs3);
        vs2 = _mm512_add_epi32(vs2, vs2_1);
        vs2 = _mm512_add_epi32(vs2, vs2_2);
        vs2 = _mm512_add_epi32(vs2, vs2_3);

        adler0 = partial_hsum(vs1) % BASE;
        adler1 = _mm512_reduce_add_epu32(vs2) % BASE;
    }

    adler = adler0 | (adler1 << 16);

    /* Process tail (len < 64). */
    if (len) {
        goto rem_peel_copy;
    }

    return adler;
}

#endif
