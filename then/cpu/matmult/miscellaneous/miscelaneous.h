//some functions I may want to implement in the future...



//from https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx
//this may come in handy
inline simde__m256d int64_to_double_fast_precise(const simde__m256i v)
/* Optimized full range int64_t to double conversion           */
/* Emulate _mm256_cvtepi64_pd()                                */
{
    simde__m256i magic_i_lo   = simde_mm256_set1_epi64x(0x4330000000000000);                /* 2^52               encoded as floating-point  */
    simde__m256i magic_i_hi32 = simde_mm256_set1_epi64x(0x4530000080000000);                /* 2^84 + 2^63        encoded as floating-point  */
    simde__m256i magic_i_all  = simde_mm256_set1_epi64x(0x4530000080100000);                /* 2^84 + 2^63 + 2^52 encoded as floating-point  */
    simde__m256d magic_d_all  = simde_mm256_castsi256_pd(magic_i_all);

    simde__m256i v_lo         = simde_mm256_blend_epi32(magic_i_lo, v, 0b01010101);         /* Blend the 32 lowest significant bits of v with magic_int_lo                                                   */
    simde__m256i v_hi         = simde_mm256_srli_epi64(v, 32);                              /* Extract the 32 most significant bits of v                                                                     */
                 v_hi         = simde_mm256_xor_si256(v_hi, magic_i_hi32);                  /* Flip the msb of v_hi and blend with 0x45300000                                                                */
    simde__m256d v_hi_dbl     = simde_mm256_sub_pd(simde_mm256_castsi256_pd(v_hi), magic_d_all); /* Compute in double precision:                                                                                  */
    simde__m256d result       = simde_mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo));    /* (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition !!                        */
            return result;                                                        /* With gcc use -O3, then -fno-associative-math is default. Do not use -Ofast, which enables -fassociative-math! */
                                                                                  /* With icc use -fp-model precise                                                                                */
}


inline __m256d uint64_to_double_fast_precise(const __m256i v)                    
/* Optimized full range uint64_t to double conversion          */
/* This code is essentially identical to Mysticial's solution. */
/* Emulate _mm256_cvtepu64_pd()                                */
{
    __m256i magic_i_lo   = _mm256_set1_epi64x(0x4330000000000000);                /* 2^52        encoded as floating-point  */
    __m256i magic_i_hi32 = _mm256_set1_epi64x(0x4530000000000000);                /* 2^84        encoded as floating-point  */
    __m256i magic_i_all  = _mm256_set1_epi64x(0x4530000000100000);                /* 2^84 + 2^52 encoded as floating-point  */
    __m256d magic_d_all  = _mm256_castsi256_pd(magic_i_all);

    __m256i v_lo         = _mm256_blend_epi32(magic_i_lo, v, 0b01010101);         /* Blend the 32 lowest significant bits of v with magic_int_lo                                                   */
    __m256i v_hi         = _mm256_srli_epi64(v, 32);                              /* Extract the 32 most significant bits of v                                                                     */
            v_hi         = _mm256_xor_si256(v_hi, magic_i_hi32);                  /* Blend v_hi with 0x45300000                                                                                    */
    __m256d v_hi_dbl     = _mm256_sub_pd(_mm256_castsi256_pd(v_hi), magic_d_all); /* Compute in double precision:                                                                                  */
    __m256d result       = _mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo));    /* (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition !!                        */
            return result;                                                        /* With gcc use -O3, then -fno-associative-math is default. Do not use -Ofast, which enables -fassociative-math! */
                                                                                  /* With icc use -fp-model precise                                                                                */
}

inline simde__m128d uint64_to_double_full(__m128i x){
    simde__m128i xH = simde_mm_srli_epi64(x, 32);
    xH = simde_mm_or_si128(xH, simde_mm_castpd_si128(simde_mm_set1_pd(19342813113834066795298816.)));          //  2^84
    simde__m128i xL = simde_mm_blend_epi16(x, simde_mm_castpd_si128(simde_mm_set1_pd(0x0010000000000000)), 0xcc);   //  2^52
    simde__m128d f = simde_mm_sub_pd(simde_mm_castsi128_pd(xH), _mm_set1_pd(19342813118337666422669312.));     //  2^84 + 2^52
    return simde_mm_add_pd(f, simde_mm_castsi128_pd(xL));
}

inline simde__m128d int64_to_double_full(__m128i x){
    __m128i xH = _mm_srai_epi32(x, 16);
    xH = _mm_blend_epi16(xH, _mm_setzero_si128(), 0x33);
    xH = _mm_add_epi64(xH, _mm_castpd_si128(_mm_set1_pd(442721857769029238784.)));              //  3*2^67
    __m128i xL = _mm_blend_epi16(x, _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)), 0x88);   //  2^52
    __m128d f = _mm_sub_pd(_mm_castsi128_pd(xH), _mm_set1_pd(442726361368656609280.));          //  3*2^67 + 2^52
    return _mm_add_pd(f, _mm_castsi128_pd(xL));
}


