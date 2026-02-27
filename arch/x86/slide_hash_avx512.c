/*
 * AVX-512 optimized hash slide
 *
 * Copyright (C) 2025 lakeworks
 * Based on slide_hash_avx2.c by Intel Corporation
 *
 * For conditions of distribution and use, see copyright notice in zlib.h
 */
#include "zbuild.h"
#include "deflate.h"

#include <immintrin.h>

static inline void slide_hash_chain(Pos *table, uint32_t entries, const __m512i wsize) {
    table += entries;
    table -= 32;

    do {
        __m512i value, result;

        value = _mm512_load_si512((__m512i *)table);
        result = _mm512_subs_epu16(value, wsize);
        _mm512_store_si512((__m512i *)table, result);

        table -= 32;
        entries -= 32;
    } while (entries > 0);
}

Z_INTERNAL void slide_hash_avx512(deflate_state *s) {
    Assert(s->w_size <= UINT16_MAX, "w_size should fit in uint16_t");
    uint16_t wsize = (uint16_t)s->w_size;
    const __m512i zmm_wsize = _mm512_set1_epi16((short)wsize);

    slide_hash_chain(s->head, HASH_SIZE, zmm_wsize);
    slide_hash_chain(s->prev, wsize, zmm_wsize);
}
