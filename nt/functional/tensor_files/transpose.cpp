#include "transpose.h"
#include "exceptions.hpp"
#include <algorithm>
#include <simde/x86/avx512.h>
#include <simde/x86/avx2.h>
#include <cstdint>
#include <vector>
#include <array>
#include <stdexcept>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/blocked_range3d.h>
#include <tbb/global_control.h>
#include <type_traits>
#include <unordered_set>

namespace nt{
namespace functional{

#if defined(SIMDE_X86_AVX512F_NATIVE) || defined(__arm64__)
// Function to transpose an 8x8 matrix of int64_t using AVX-512 intrinsics
void transpose8x8_int64_intrinsics(const int64_t* in, int64_t* out, const int64_t& in_stride, const int64_t& out_stride) {
    // Load 8 rows into 8 simde__m512i registers
    simde__m512i row0 = simde_mm512_loadu_si512(reinterpret_cast<const simde__m512i*>(in + 0 * in_stride)); //0, 1, 2, 3, 4, 5, 6, 7,
    simde__m512i row1 = simde_mm512_loadu_si512(reinterpret_cast<const simde__m512i*>(in + 1 * in_stride)); //8, 9 , 10, 11, 12, 13, 14, 15
    simde__m512i row2 = simde_mm512_loadu_si512(reinterpret_cast<const simde__m512i*>(in + 2 * in_stride));
    simde__m512i row3 = simde_mm512_loadu_si512(reinterpret_cast<const simde__m512i*>(in + 3 * in_stride));
    simde__m512i row4 = simde_mm512_loadu_si512(reinterpret_cast<const simde__m512i*>(in + 4 * in_stride));
    simde__m512i row5 = simde_mm512_loadu_si512(reinterpret_cast<const simde__m512i*>(in + 5 * in_stride));
    simde__m512i row6 = simde_mm512_loadu_si512(reinterpret_cast<const simde__m512i*>(in + 6 * in_stride));
    simde__m512i row7 = simde_mm512_loadu_si512(reinterpret_cast<const simde__m512i*>(in + 7 * in_stride));

    // --- Step 1: Transpose 2x2 blocks of int64_t elements ---
    // (treating each simde__m512i as 4 pairs of 2 int64_t elements)
    // We use unpacklo/hi to interleave the elements from adjacent rows.
    simde__m512i t0 = simde_mm512_unpacklo_epi64(row0, row1); // {row0[0], row1[0], row0[1], row1[1], ...}
    simde__m512i t1 = simde_mm512_unpackhi_epi64(row0, row1); // {row0[4], row1[4], row0[5], row1[5], ...}
    simde__m512i t2 = simde_mm512_unpacklo_epi64(row2, row3);
    simde__m512i t3 = simde_mm512_unpackhi_epi64(row2, row3);
    simde__m512i t4 = simde_mm512_unpacklo_epi64(row4, row5);
    simde__m512i t5 = simde_mm512_unpackhi_epi64(row4, row5);
    simde__m512i t6 = simde_mm512_unpacklo_epi64(row6, row7);
    simde__m512i t7 = simde_mm512_unpackhi_epi64(row6, row7);

    static simde__m512i perm_idx = simde_mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);
    static simde__m512i perm_idxB = simde_mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);

    simde__m512i u0 = simde_mm512_permutex2var_epi64(t0, perm_idx , t2);
    simde__m512i u1 = simde_mm512_permutex2var_epi64(t0, perm_idxB, t2); 
    simde__m512i u2 = simde_mm512_permutex2var_epi64(t1, perm_idx , t3);
    simde__m512i u3 = simde_mm512_permutex2var_epi64(t1, perm_idxB, t3);
    simde__m512i u4 = simde_mm512_permutex2var_epi64(t4, perm_idx , t6);
    simde__m512i u5 = simde_mm512_permutex2var_epi64(t4, perm_idxB, t6);
    simde__m512i u6 = simde_mm512_permutex2var_epi64(t5, perm_idx , t7);
    simde__m512i u7 = simde_mm512_permutex2var_epi64(t5, perm_idxB, t7);

    simde__m256i u0_lo = simde_mm512_extracti64x4_epi64(u0, 0); // lower 256-bit of u0
    simde__m256i u0_hi = simde_mm512_extracti64x4_epi64(u0, 1); // upper 256-bit of u0
    simde__m256i u1_lo = simde_mm512_extracti64x4_epi64(u1, 0);
    simde__m256i u1_hi = simde_mm512_extracti64x4_epi64(u1, 1);
    simde__m256i u2_lo = simde_mm512_extracti64x4_epi64(u2, 0);
    simde__m256i u2_hi = simde_mm512_extracti64x4_epi64(u2, 1);
    simde__m256i u3_lo = simde_mm512_extracti64x4_epi64(u3, 0);
    simde__m256i u3_hi = simde_mm512_extracti64x4_epi64(u3, 1);
    simde__m256i u4_lo = simde_mm512_extracti64x4_epi64(u4, 0);
    simde__m256i u4_hi = simde_mm512_extracti64x4_epi64(u4, 1);
    simde__m256i u5_lo = simde_mm512_extracti64x4_epi64(u5, 0);
    simde__m256i u5_hi = simde_mm512_extracti64x4_epi64(u5, 1);
    simde__m256i u6_lo = simde_mm512_extracti64x4_epi64(u6, 0);
    simde__m256i u6_hi = simde_mm512_extracti64x4_epi64(u6, 1);
    simde__m256i u7_lo = simde_mm512_extracti64x4_epi64(u7, 0);
    simde__m256i u7_hi = simde_mm512_extracti64x4_epi64(u7, 1);

    simde__m512i final0 = simde_mm512_inserti64x4(simde_mm512_castsi256_si512(u0_lo), u4_lo, 1);
    simde__m512i final1 = simde_mm512_inserti64x4(simde_mm512_castsi256_si512(u2_lo), u6_lo, 1);
    simde__m512i final2 = simde_mm512_inserti64x4(simde_mm512_castsi256_si512(u1_lo), u5_lo, 1);
    simde__m512i final3 = simde_mm512_inserti64x4(simde_mm512_castsi256_si512(u3_lo), u7_lo, 1);
    simde__m512i final4 = simde_mm512_inserti64x4(simde_mm512_castsi256_si512(u0_hi), u4_hi, 1);
    simde__m512i final5 = simde_mm512_inserti64x4(simde_mm512_castsi256_si512(u2_hi), u6_hi, 1);
    simde__m512i final6 = simde_mm512_inserti64x4(simde_mm512_castsi256_si512(u1_hi), u5_hi, 1);
    simde__m512i final7 = simde_mm512_inserti64x4(simde_mm512_castsi256_si512(u3_hi), u7_hi, 1);

    // Store the transposed rows
    simde_mm512_store_si512(reinterpret_cast<simde__m512i*>(out + 0 * out_stride), final0);
    simde_mm512_store_si512(reinterpret_cast<simde__m512i*>(out + 1 * out_stride), final1);
    simde_mm512_store_si512(reinterpret_cast<simde__m512i*>(out + 2 * out_stride), final2);
    simde_mm512_store_si512(reinterpret_cast<simde__m512i*>(out + 3 * out_stride), final3);
    simde_mm512_store_si512(reinterpret_cast<simde__m512i*>(out + 4 * out_stride), final4);
    simde_mm512_store_si512(reinterpret_cast<simde__m512i*>(out + 5 * out_stride), final5);
    simde_mm512_store_si512(reinterpret_cast<simde__m512i*>(out + 6 * out_stride), final6);
    simde_mm512_store_si512(reinterpret_cast<simde__m512i*>(out + 7 * out_stride), final7);
}

#else
// Function to transpose an 8x8 matrix of int64_t using AVX-512 intrinsics
void transpose8x8_int64_intrinsics(const int64_t* in, int64_t* out, const int64_t& in_stride, const int64_t& out_stride) {
    // Load 8 rows into 15 simde__m256i registers
    simde__m256i row0   = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(in + 0 * in_stride)); //0, 1, 2, 3, 4, 5, 6, 7,
    simde__m256i row0_1 = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(in + 0 * in_stride + 4)); //0, 1, 2, 3, 4, 5, 6, 7,
    simde__m256i row1   = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(in + 1 * in_stride)); //8, 9 , 10, 11, 12, 13, 14, 15
    simde__m256i row1_1 = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(in + 1 * in_stride + 4));
    simde__m256i row2   = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(in + 2 * in_stride));
    simde__m256i row2_1 = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(in + 2 * in_stride + 4));
    simde__m256i row3 = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(in + 3 * in_stride));
    simde__m256i row3_1 = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(in + 3 * in_stride + 4));
    simde__m256i row4 = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(in + 4 * in_stride));
    simde__m256i row4_1 = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(in + 4 * in_stride + 4));
    simde__m256i row5 = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(in + 5 * in_stride));
    simde__m256i row5_1 = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(in + 5 * in_stride + 4));
    simde__m256i row6 = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(in + 6 * in_stride));
    simde__m256i row6_1 = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(in + 6 * in_stride + 4));
    simde__m256i row7 = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(in + 7 * in_stride));
    simde__m256i row7_1 = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(in + 7 * in_stride + 4));

    // --- Step 1: Transpose 2x2 blocks of int64_t elements ---
    // (treating each simde__m512i as 4 pairs of 2 int64_t elements)
    // We use unpacklo/hi to interleave the elements from adjacent rows.
    simde__m256i t0 = simde_mm256_unpacklo_epi64(row0, row1); // {row0[0], row1[0], row0[1], row1[1], ...}
    simde__m256i t0_1 = simde_mm256_unpacklo_epi64(row0_1, row1_1); // {row0[4], row1[4], row0[5], row1[5], ...}
    simde__m256i t1 = simde_mm256_unpackhi_epi64(row0, row1); // {row0[4], row1[4], row0[5], row1[5], ...}
    simde__m256i t1_1 = simde_mm256_unpackhi_epi64(row0_1, row1_1); // {row0[4], row1[4], row0[5], row1[5], ...}
    simde__m256i t2 = simde_mm256_unpacklo_epi64(row2, row3);
    simde__m256i t2_1 = simde_mm256_unpacklo_epi64(row2_1, row3_1);
    simde__m256i t3 = simde_mm256_unpackhi_epi64(row2, row3);
    simde__m256i t3_1 = simde_mm256_unpackhi_epi64(row2_1, row3_1);
    simde__m256i t4 = simde_mm256_unpacklo_epi64(row4, row5);
    simde__m256i t4_1 = simde_mm256_unpacklo_epi64(row4_1, row5_1);
    simde__m256i t5 = simde_mm256_unpackhi_epi64(row4, row5);
    simde__m256i t5_1 = simde_mm256_unpackhi_epi64(row4_1, row5_1);
    simde__m256i t6 = simde_mm256_unpacklo_epi64(row6, row7);
    simde__m256i t6_1 = simde_mm256_unpacklo_epi64(row6_1, row7_1);
    simde__m256i t7 = simde_mm256_unpackhi_epi64(row6, row7);
    simde__m256i t7_1 = simde_mm256_unpackhi_epi64(row6_1, row7_1);


    simde__m128i t0_lo = simde_mm256_extracti128_si256(t0, 0); 
    simde__m128i t0_1_lo = simde_mm256_extracti128_si256(t0_1, 0);
    simde__m128i t0_hi = simde_mm256_extracti128_si256(t0, 1); 
    simde__m128i t0_1_hi = simde_mm256_extracti128_si256(t0_1, 1);
    simde__m128i t1_lo = simde_mm256_extracti128_si256(t1, 0); // {row0[4], row1[4], row0[5], row1[5], ...}
    simde__m128i t1_1_lo = simde_mm256_extracti128_si256(t1_1, 0); // {row0[4], row1[4], row0[5], row1[5], ...}
    simde__m128i t1_hi = simde_mm256_extracti128_si256(t1, 1); // {row0[4], row1[4], row0[5], row1[5], ...}
    simde__m128i t1_1_hi = simde_mm256_extracti128_si256(t1_1, 1); // {row0[4], row1[4], row0[5], row1[5], ...}
    simde__m128i t2_lo = simde_mm256_extracti128_si256(t2, 0);
    simde__m128i t2_1_lo = simde_mm256_extracti128_si256(t2_1, 0);
    simde__m128i t2_hi = simde_mm256_extracti128_si256(t2, 1);
    simde__m128i t2_1_hi = simde_mm256_extracti128_si256(t2_1, 1);
    simde__m128i t3_lo = simde_mm256_extracti128_si256(t3, 0);
    simde__m128i t3_1_lo = simde_mm256_extracti128_si256(t3_1, 0);
    simde__m128i t3_hi = simde_mm256_extracti128_si256(t3, 1);
    simde__m128i t3_1_hi = simde_mm256_extracti128_si256(t3_1, 1);
    simde__m128i t4_lo = simde_mm256_extracti128_si256(t4, 0);
    simde__m128i t4_1_lo = simde_mm256_extracti128_si256(t4_1, 0);
    simde__m128i t4_hi = simde_mm256_extracti128_si256(t4, 1);
    simde__m128i t4_1_hi = simde_mm256_extracti128_si256(t4_1, 1);
    simde__m128i t5_lo = simde_mm256_extracti128_si256(t5, 0);
    simde__m128i t5_1_lo = simde_mm256_extracti128_si256(t5_1, 0);
    simde__m128i t5_hi = simde_mm256_extracti128_si256(t5, 1);
    simde__m128i t5_1_hi = simde_mm256_extracti128_si256(t5_1, 1);
    simde__m128i t6_lo = simde_mm256_extracti128_si256(t6, 0);
    simde__m128i t6_1_lo = simde_mm256_extracti128_si256(t6_1, 0);
    simde__m128i t6_hi = simde_mm256_extracti128_si256(t6, 1);
    simde__m128i t6_1_hi = simde_mm256_extracti128_si256(t6_1, 1);
    simde__m128i t7_lo = simde_mm256_extracti128_si256(t7, 0);
    simde__m128i t7_1_lo = simde_mm256_extracti128_si256(t7_1, 0);
    simde__m128i t7_hi = simde_mm256_extracti128_si256(t7, 1);
    simde__m128i t7_1_hi = simde_mm256_extracti128_si256(t7_1, 1);

    simde__m256i result0 = simde_mm256_castsi128_si256(t0_lo);
    simde__m256i final0 = simde_mm256_inserti128_si256(result0, t2_lo, 1);
    simde__m256i result0_1 = simde_mm256_castsi128_si256(t0_1_lo);
    simde__m256i final0_1 = simde_mm256_inserti128_si256(result0_1, t2_1_lo, 1);
    simde__m256i result1 = simde_mm256_castsi128_si256(t4_lo);
    simde__m256i final1 = simde_mm256_inserti128_si256(result1, t6_lo, 1);
    simde__m256i result1_1 = simde_mm256_castsi128_si256(t4_1_lo);
    simde__m256i final1_1 = simde_mm256_inserti128_si256(result1_1, t6_1_lo, 1);

    simde__m256i result2 = simde_mm256_castsi128_si256(t1_lo);
    simde__m256i final2 = simde_mm256_inserti128_si256(result2, t3_lo, 1);
    simde__m256i result2_1 = simde_mm256_castsi128_si256(t1_1_lo);
    simde__m256i final2_1 = simde_mm256_inserti128_si256(result2_1, t3_1_lo, 1);
    simde__m256i result3 = simde_mm256_castsi128_si256(t5_lo);
    simde__m256i final3 = simde_mm256_inserti128_si256(result3, t7_lo, 1);
    simde__m256i result3_1 = simde_mm256_castsi128_si256(t5_1_lo);
    simde__m256i final3_1 = simde_mm256_inserti128_si256(result3_1, t7_1_lo, 1);

    simde__m256i result4 = simde_mm256_castsi128_si256(t0_hi);
    simde__m256i final4 = simde_mm256_inserti128_si256(result4, t2_hi, 1);
    simde__m256i result4_1 = simde_mm256_castsi128_si256(t0_1_hi);
    simde__m256i final4_1 = simde_mm256_inserti128_si256(result4_1, t2_1_hi, 1);
    simde__m256i result5 = simde_mm256_castsi128_si256(t4_hi);
    simde__m256i final5 = simde_mm256_inserti128_si256(result5, t6_hi, 1);
    simde__m256i result5_1 = simde_mm256_castsi128_si256(t4_1_hi);
    simde__m256i final5_1 = simde_mm256_inserti128_si256(result5_1, t6_1_hi, 1);

    simde__m256i result6 = simde_mm256_castsi128_si256(t1_hi);
    simde__m256i final6 = simde_mm256_inserti128_si256(result6, t3_hi, 1);
    simde__m256i result6_1 = simde_mm256_castsi128_si256(t1_1_hi);
    simde__m256i final6_1 = simde_mm256_inserti128_si256(result6_1, t3_1_hi, 1);
    simde__m256i result7 = simde_mm256_castsi128_si256(t5_hi);
    simde__m256i final7 = simde_mm256_inserti128_si256(result7, t7_hi, 1);
    simde__m256i result7_1 = simde_mm256_castsi128_si256(t5_1_hi);
    simde__m256i final7_1 = simde_mm256_inserti128_si256(result7_1, t7_1_hi, 1);




    // // Store the transposed rows
    simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(out + 0 * out_stride), final0);
    simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(out + 0 * out_stride + 4), final1);
    simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(out + 1 * out_stride), final2);
    simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(out + 1 * out_stride + 4), final3);
    simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(out + 2 * out_stride), final4);
    simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(out + 2 * out_stride + 4), final5);
    simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(out + 3 * out_stride), final6);
    simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(out + 3 * out_stride + 4), final7);

    simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(out + 4 * out_stride), final0_1);
    simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(out + 4 * out_stride + 4), final1_1);
    simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(out + 5 * out_stride), final2_1);
    simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(out + 5 * out_stride + 4), final3_1);
    simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(out + 6 * out_stride), final4_1);
    simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(out + 6 * out_stride + 4), final5_1);
    simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(out + 7 * out_stride), final6_1);
    simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(out + 7 * out_stride + 4), final7_1);

}

#endif

#define _NT_UNROLL_TRANSPOSE_SUB_(out, packed, r_o, packed_cols)\
    out[(c+r_o) * in_rows + r] = packed[(r_o * packed_cols) + 0];\
    out[(c+r_o) * in_rows + (r+1)] = packed[(r_o * packed_cols) + 1];\
    out[(c+r_o)  * in_rows + (r+2)] = packed[(r_o * packed_cols) + 2];\
    out[(c+r_o)  * in_rows + (r+3)] = packed[(r_o * packed_cols) + 3];\
    out[(c+r_o)  * in_rows + (r+4)] = packed[(r_o * packed_cols) + 4];\
    out[(c+r_o)  * in_rows + (r+5)] = packed[(r_o * packed_cols) + 5];\
    out[(c+r_o)  * in_rows + (r+6)] = packed[(r_o * packed_cols) + 6];\
    out[(c+r_o)  * in_rows + (r+7)] = packed[(r_o * packed_cols) + 7];\

#define _NT_UNROLL_TRANSPOSE_(out, packed, packed_cols)\
   _NT_UNROLL_TRANSPOSE_SUB_(out, packed, 0, packed_cols)\
   _NT_UNROLL_TRANSPOSE_SUB_(out, packed, 1, packed_cols)\
   _NT_UNROLL_TRANSPOSE_SUB_(out, packed, 2, packed_cols)\
   _NT_UNROLL_TRANSPOSE_SUB_(out, packed, 3, packed_cols)\
   _NT_UNROLL_TRANSPOSE_SUB_(out, packed, 4, packed_cols)\
   _NT_UNROLL_TRANSPOSE_SUB_(out, packed, 5, packed_cols)\
   _NT_UNROLL_TRANSPOSE_SUB_(out, packed, 6, packed_cols)\
   _NT_UNROLL_TRANSPOSE_SUB_(out, packed, 7, packed_cols)\


#define _NT_TRANSPOSE_NTHREADS_ 21

alignas(64) static int64_t block_packed[64 * _NT_TRANSPOSE_NTHREADS_];

void transpose_last_intrinsics(const int64_t* in, int64_t* out, const int64_t in_rows, const int64_t in_cols, const int64_t batches){
    tbb::global_control control(tbb::global_control::max_allowed_parallelism, _NT_TRANSPOSE_NTHREADS_);
    const int64_t* in_cpy = in;
    int64_t* out_cpy = out;
    constexpr int64_t tile_size = 8;
    const int64_t row_r = in_rows % tile_size;
    const int64_t col_r = in_cols % tile_size;
    const bool extra_cols = (row_r != 0);
    const bool extra_rows = (col_r != 0);
    const int64_t rows = in_rows - row_r;
    const int64_t cols = in_cols - col_r;
    const int64_t mat_size = in_rows * in_cols;
    if(batches < _NT_TRANSPOSE_NTHREADS_){
        tbb::parallel_for(tbb::blocked_range<int64_t>(0, batches),
        [&](const tbb::blocked_range<int64_t>& range){
        const int64_t* cur_in = &in_cpy[range.begin() * mat_size];
        int64_t* cur_out = &out_cpy[range.begin() * mat_size];
        for(int64_t b = range.begin(); b != range.end(); ++b, cur_out += mat_size, cur_in += mat_size){
            int64_t* cur_packed = &block_packed[tile_size * b];
            for(int64_t r = 0; r < rows; r += tile_size){
                for(int64_t c = 0; c < cols; c += tile_size){
                    transpose8x8_int64_intrinsics(&cur_in[r * in_cols + c], cur_packed, in_cols, tile_size * _NT_TRANSPOSE_NTHREADS_);
                    _NT_UNROLL_TRANSPOSE_(cur_out, cur_packed, tile_size * _NT_TRANSPOSE_NTHREADS_)
                }
            }
            for(int64_t r = rows; r < in_rows; ++r){
                for(int64_t c = 0; c < in_cols; ++c){
                    cur_out[c * in_rows + r] = cur_in[r * in_cols + c];
                }
            }
            for(int64_t c = cols; c < in_cols; ++c){
                for(int64_t r = 0; r < in_rows; ++r){
                    cur_out[c * in_rows + r] = cur_in[r * in_cols + c];
                }
            }

        }
        });
    }else{
        //batches >= _NT_TRANSPOSE_NTHREADS_
        const int64_t mult = batches / _NT_TRANSPOSE_NTHREADS_;
        tbb::parallel_for(tbb::blocked_range<int64_t>(0, _NT_TRANSPOSE_NTHREADS_),
        [&](const tbb::blocked_range<int64_t>& range){
            int64_t b_begin = range.begin() * mult;
            int64_t b_end = std::min(range.end() * mult, batches);
            if(range.end() == _NT_TRANSPOSE_NTHREADS_ && b_end < batches) b_end = batches;
            // std::cout << b_begin <<"->"<<b_end<<std::endl;
            int64_t* cur_packed = &block_packed[tile_size * range.begin()];
            const int64_t* cur_in = &in_cpy[b_begin * mat_size];
            int64_t* cur_out = &out_cpy[b_begin * mat_size];
            for(int64_t b = b_begin; b != b_end; ++b, cur_out += mat_size, cur_in += mat_size){
                for(int64_t r = 0; r < rows; r += tile_size){
                    for(int64_t c = 0; c < cols; c += tile_size){
                        transpose8x8_int64_intrinsics(&cur_in[r * in_cols + c], cur_packed, in_cols, tile_size * _NT_TRANSPOSE_NTHREADS_);
                        _NT_UNROLL_TRANSPOSE_(cur_out, cur_packed, tile_size * _NT_TRANSPOSE_NTHREADS_)
                    }
                }
                cur_packed += tile_size;
                for(int64_t r = rows; r < in_rows; ++r){
                    for(int64_t c = 0; c < in_cols; ++c){
                        cur_out[c * in_rows + r] = cur_in[r * in_cols + c];
                    }
                }
                for(int64_t c = cols; c < in_cols; ++c){
                    for(int64_t r = 0; r < in_rows; ++r){
                        cur_out[c * in_rows + r] = cur_in[r * in_cols + c];
                    }
                }
            }
        });
    }
}

#undef _NT_UNROLL_TRANSPOSE_
#undef _NT_UNROLL_TRANSPOSE_SUB_ 

void transpose_last_manual(const std::uintptr_t* _in, std::uintptr_t* _out, const int64_t in_rows, const int64_t in_cols, const int64_t batches){
    const int64_t mat_size = in_rows * in_cols;
    tbb::parallel_for(tbb::blocked_range2d<int64_t>(0, batches, 0, in_rows * in_cols),
    [&](const tbb::blocked_range2d<int64_t>& range){
    const std::uintptr_t* in = _in + (mat_size * range.rows().begin());
    std::uintptr_t* out = _out + (mat_size * range.rows().begin());
    for(int64_t b = range.rows().begin(); b != range.rows().end(); ++b, in += mat_size, out += mat_size){
        for(int64_t n = range.cols().begin(); n != range.cols().end(); n++) {
            int64_t i = n/in_rows;
            int64_t j = n%in_rows;
            out[n] = in[in_cols*j + i];
        }
    }
    });

}


#ifdef _MSC_VER
void transpose_last_manual_MSVC(void** _in, void** _out, const int64_t in_rows, const int64_t in_cols, const int64_t batches){
    const int64_t mat_size = in_rows * in_cols;
    tbb::parallel_for(tbb::blocked_range2d<int64_t>(0, batches, 0, in_rows * in_cols),
    [&](const tbb::blocked_range2d<int64_t>& range){
    void** in = &_in[(mat_size * range.rows().begin())];
    void** out = &_out[(mat_size * range.rows().begin())];
    for(int64_t b = range.rows().begin(); b != range.rows().end(); ++b, in += mat_size, out += mat_size){
        for(int64_t n = range.cols().begin(); n != range.cols().end(); n++) {
            int64_t i = n/in_rows;
            int64_t j = n%in_rows;
            out[n] = in[in_cols*j + i];
        }
    }
    });

}

#endif

#define NT_MAKE_LOAD_STRIDE_IDX_(amt) simde_mm512_set_epi64(amt*7, amt*6, amt*5, amt*4, amt*3, amt*2, amt, 0)


//TODO: The below is a function that can perform transposes across different dimensions
//the only difference is that it uses the gather instead of the load function
//make an AVX2 version of this
//and then there needs to be an intrinsics "any" functon
//  - Important: it will only work when transposing and dimension with rows and columns
//      - so you can load lower dimensions from void** to void*** and reinterpret_cast to int64_t*
//      - then flatten that out to lower dimensions
// Function to transpose an 8x8 matrix of int64_t using AVX-512 intrinsics
// void transpose8x8_int64_strided_intrinsics(const int64_t* in, int64_t* out, const int64_t& in_stride, const int64_t& out_stride, const int64_t& stride) {
//     simde__m512i index = NT_MAKE_LOAD_STRIDE_IDX_(stride); 
//     // Load 8 rows into 8 simde__m512i registers
//     simde__m512i row0 = simde_mm512_i64gather_epi64(index, in + 0 * in_stride, 8); 
//     simde__m512i row1 = simde_mm512_i64gather_epi64(index, in + 1 * in_stride, 8);
//     simde__m512i row2 = simde_mm512_i64gather_epi64(index, in + 2 * in_stride, 8);
//     simde__m512i row3 = simde_mm512_i64gather_epi64(index, in + 3 * in_stride, 8);
//     simde__m512i row4 = simde_mm512_i64gather_epi64(index, in + 4 * in_stride, 8);
//     simde__m512i row5 = simde_mm512_i64gather_epi64(index, in + 5 * in_stride, 8);
//     simde__m512i row6 = simde_mm512_i64gather_epi64(index, in + 6 * in_stride, 8);
//     simde__m512i row7 = simde_mm512_i64gather_epi64(index, in + 7 * in_stride, 8);
    

//     // std::cout << in[0] << std::endl;
//     // std::cout << in[1] << std::endl;
//     // std::cout << in[stride] << std::endl;

//     // --- Step 1: Transpose 2x2 blocks of int64_t elements ---
//     // (treating each simde__m512i as 4 pairs of 2 int64_t elements)
//     // We use unpacklo/hi to interleave the elements from adjacent rows.
//     simde__m512i t0 = simde_mm512_unpacklo_epi64(row0, row1); // {row0[0], row1[0], row0[1], row1[1], ...}
//     simde__m512i t1 = simde_mm512_unpackhi_epi64(row0, row1); // {row0[4], row1[4], row0[5], row1[5], ...}
//     simde__m512i t2 = simde_mm512_unpacklo_epi64(row2, row3);
//     simde__m512i t3 = simde_mm512_unpackhi_epi64(row2, row3);
//     simde__m512i t4 = simde_mm512_unpacklo_epi64(row4, row5);
//     simde__m512i t5 = simde_mm512_unpackhi_epi64(row4, row5);
//     simde__m512i t6 = simde_mm512_unpacklo_epi64(row6, row7);
//     simde__m512i t7 = simde_mm512_unpackhi_epi64(row6, row7);

//     static simde__m512i perm_idx = simde_mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);
//     static simde__m512i perm_idxB = simde_mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);

//     simde__m512i u0 = simde_mm512_permutex2var_epi64(t0, perm_idx , t2);
//     simde__m512i u1 = simde_mm512_permutex2var_epi64(t0, perm_idxB, t2); 
//     simde__m512i u2 = simde_mm512_permutex2var_epi64(t1, perm_idx , t3);
//     simde__m512i u3 = simde_mm512_permutex2var_epi64(t1, perm_idxB, t3);
//     simde__m512i u4 = simde_mm512_permutex2var_epi64(t4, perm_idx , t6);
//     simde__m512i u5 = simde_mm512_permutex2var_epi64(t4, perm_idxB, t6);
//     simde__m512i u6 = simde_mm512_permutex2var_epi64(t5, perm_idx , t7);
//     simde__m512i u7 = simde_mm512_permutex2var_epi64(t5, perm_idxB, t7);

//     simde__m256i u0_lo = simde_mm512_extracti64x4_epi64(u0, 0); // lower 256-bit of u0
//     simde__m256i u0_hi = simde_mm512_extracti64x4_epi64(u0, 1); // upper 256-bit of u0
//     simde__m256i u1_lo = simde_mm512_extracti64x4_epi64(u1, 0);
//     simde__m256i u1_hi = simde_mm512_extracti64x4_epi64(u1, 1);
//     simde__m256i u2_lo = simde_mm512_extracti64x4_epi64(u2, 0);
//     simde__m256i u2_hi = simde_mm512_extracti64x4_epi64(u2, 1);
//     simde__m256i u3_lo = simde_mm512_extracti64x4_epi64(u3, 0);
//     simde__m256i u3_hi = simde_mm512_extracti64x4_epi64(u3, 1);
//     simde__m256i u4_lo = simde_mm512_extracti64x4_epi64(u4, 0);
//     simde__m256i u4_hi = simde_mm512_extracti64x4_epi64(u4, 1);
//     simde__m256i u5_lo = simde_mm512_extracti64x4_epi64(u5, 0);
//     simde__m256i u5_hi = simde_mm512_extracti64x4_epi64(u5, 1);
//     simde__m256i u6_lo = simde_mm512_extracti64x4_epi64(u6, 0);
//     simde__m256i u6_hi = simde_mm512_extracti64x4_epi64(u6, 1);
//     simde__m256i u7_lo = simde_mm512_extracti64x4_epi64(u7, 0);
//     simde__m256i u7_hi = simde_mm512_extracti64x4_epi64(u7, 1);

//     simde__m512i final0 = simde_mm512_inserti64x4(simde_mm512_castsi256_si512(u0_lo), u4_lo, 1);
//     simde__m512i final1 = simde_mm512_inserti64x4(simde_mm512_castsi256_si512(u2_lo), u6_lo, 1);
//     simde__m512i final2 = simde_mm512_inserti64x4(simde_mm512_castsi256_si512(u1_lo), u5_lo, 1);
//     simde__m512i final3 = simde_mm512_inserti64x4(simde_mm512_castsi256_si512(u3_lo), u7_lo, 1);
//     simde__m512i final4 = simde_mm512_inserti64x4(simde_mm512_castsi256_si512(u0_hi), u4_hi, 1);
//     simde__m512i final5 = simde_mm512_inserti64x4(simde_mm512_castsi256_si512(u2_hi), u6_hi, 1);
//     simde__m512i final6 = simde_mm512_inserti64x4(simde_mm512_castsi256_si512(u1_hi), u5_hi, 1);
//     simde__m512i final7 = simde_mm512_inserti64x4(simde_mm512_castsi256_si512(u3_hi), u7_hi, 1);

//     // Store the transposed rows
//     simde_mm512_store_si512(reinterpret_cast<simde__m512i*>(out + 0 * out_stride), final0);
//     simde_mm512_store_si512(reinterpret_cast<simde__m512i*>(out + 1 * out_stride), final1);
//     simde_mm512_store_si512(reinterpret_cast<simde__m512i*>(out + 2 * out_stride), final2);
//     simde_mm512_store_si512(reinterpret_cast<simde__m512i*>(out + 3 * out_stride), final3);
//     simde_mm512_store_si512(reinterpret_cast<simde__m512i*>(out + 4 * out_stride), final4);
//     simde_mm512_store_si512(reinterpret_cast<simde__m512i*>(out + 5 * out_stride), final5);
//     simde_mm512_store_si512(reinterpret_cast<simde__m512i*>(out + 6 * out_stride), final6);
//     simde_mm512_store_si512(reinterpret_cast<simde__m512i*>(out + 7 * out_stride), final7);
// }


#undef NT_MAKE_LOAD_STRIDE_IDX_


inline int64_t unravel_and_compute(int64_t flat_index, const int64_t& total_size, const std::vector<int64_t>& shape, const std::vector<int64_t>& strides){
    int64_t index = 0;
    for(int64_t i = shape.size()-1; i >= 0; --i){
        index += (flat_index % shape[i]) * strides[i];
        flat_index /= shape[i];
    }
    return index;
}

void transpose_any_manual(void** _in, void** _out, 
                            int64_t axis0, int64_t axis1, 
                            std::vector<int64_t> shape){
    int64_t ndim = shape.size();
    axis0 = axis0 < 0 ? axis0 + ndim : axis0;
    axis1 = axis1 < 0 ? axis1 + ndim : axis1;
    if(axis0 > axis1) std::swap(axis0, axis1);
    
    
#ifdef _MSC_VER    
    void** __in = _in;
    void** __out = _out;
#else
    using ptr_type = typename std::conditional_t<sizeof(int64_t) == sizeof(void*),
                                    int64_t, std::uintptr_t>;
    const ptr_type* __in = reinterpret_cast<const ptr_type*>(_in);
    ptr_type* __out = reinterpret_cast<ptr_type*>(_out);
#endif

    //ensure axis0 < axis1
    int64_t total_size = 1;
    for(int64_t i = 0; i < ndim; ++i)
        total_size *= shape[i];

    
    std::vector<int64_t> strides(ndim-axis0);
    for(int64_t i = axis0; i < ndim; ++i){
        strides[i-axis0] = 1;
        for(int64_t j = i+1; j < ndim; ++j)
            strides[i-axis0] *= shape[j];
    }
    int64_t batches = 1;
    for(int64_t i = 0; i < axis0; ++i)
        batches *= shape[i];

    std::swap(strides[0], strides[axis1-axis0]);
    std::vector<int64_t> n_shape(ndim-axis0);
    std::copy(shape.begin()+axis0, shape.end(), n_shape.begin());
    std::swap(n_shape[0], n_shape[axis1-axis0]);
    int64_t total_inner = total_size / batches;
#ifndef _MSC_VER
    tbb::parallel_for(tbb::blocked_range2d<int64_t>(0, batches, 0, total_inner),
    [&](const tbb::blocked_range2d<int64_t>& range){
    const ptr_type* in = &__in[(range.rows().begin() * total_inner)];
    ptr_type* out = &__out[(range.rows().begin() * total_inner)];
    for(int64_t b = range.rows().begin(); b < range.rows().end(); ++b){
        for(int64_t i = range.cols().begin(); i < range.cols().end(); ++i){
            int64_t index = unravel_and_compute(i, total_inner, n_shape, strides);
            out[i] = in[index];
        }
        in += total_inner;
        out += total_inner;
    }});
#else
    tbb::parallel_for(tbb::blocked_range2d<int64_t>(0, batches, 0, total_inner),
    [&](const tbb::blocked_range2d<int64_t>& range){
    void** in = &__in[(range.rows().begin() * total_inner)];
    void** out = &__out[(range.rows().begin() * total_inner)];
    for(int64_t b = range.rows().begin(); b < range.rows().end(); ++b){
        for(int64_t i = range.cols().begin(); i < range.cols().end(); ++i){
            int64_t index = unravel_and_compute(i, total_inner, n_shape, strides);
            out[i] = in[index];
        }
        in += total_inner;
        out += total_inner;
    }});


#endif

}



void transpose_row_col(void** _in, void** _out, std::vector<int64_t> shape){
    const int64_t in_rows = shape[shape.size()-2];
    const int64_t in_cols = shape[shape.size()-1];
    int64_t batches = 1;
    for(int64_t b = 0; b < shape.size()-2; ++b){
        batches *= shape[b];
    }
#ifndef _MSC_VER
    using ptr_type = typename std::conditional_t<sizeof(int64_t) == sizeof(void*),
                                    int64_t, std::uintptr_t>;
    const ptr_type* in = reinterpret_cast<ptr_type*>(_in);
    ptr_type* out = reinterpret_cast<ptr_type*>(_out);
    if constexpr(sizeof(int64_t) == sizeof(void*)){
        transpose_last_intrinsics(in, out, in_rows, in_cols, batches);
    }else{
        transpose_last_manual(reinterpret_cast<const std::uintptr_t*>(in), reinterpret_cast<std::uintptr_t*>(out), in_rows, in_cols, batches); 
    }
#else
    transpose_last_manual_MSVC(_in, _out, in_rows, in_cols, batches);
#endif
}

void transpose(void** _in, void** _out, 
               int64_t axis0, int64_t axis1, 
                std::vector<int64_t> shape){
    int64_t ndim = shape.size();
    axis0 = axis0 < 0 ? axis0 + ndim : axis0;
    axis1 = axis1 < 0 ? axis1 + ndim : axis1;
    if(axis0 > axis1) std::swap(axis0, axis1);
    //makes axis0 < axis1
    if(axis0 == ndim-2 && axis1 == ndim-1){
        transpose_row_col(_in, _out, std::move(shape));
        return;
    }
    transpose_any_manual(_in, _out, axis0, axis1, std::move(shape));
}


void permute(void** _in, void** _out, 
                            std::vector<int64_t> Perm, 
                            std::vector<int64_t> shape){
    using size_value_t = int64_t;
    int64_t ndim = shape.size();

#ifdef _MSC_VER
    void** __in = _in;
    void** __out = _out;
#else
    using ptr_type = typename std::conditional_t<sizeof(int64_t) == sizeof(void*),
                                    int64_t, std::uintptr_t>;
    const ptr_type* __in = reinterpret_cast<const ptr_type*>(_in);
    ptr_type* __out = reinterpret_cast<ptr_type*>(_out);
#endif

    
    if(Perm.size() > shape.size()){
        throw std::invalid_argument("Got invalid size for the number of permutations");
    }


    int64_t total_size = 1;
    for(int64_t i = 0; i < ndim; ++i)
        total_size *= shape[i];

    
    std::vector<int64_t> strides(ndim);
    for(int64_t i = 0; i < ndim; ++i){
        strides[i] = 1;
        for(int64_t j = i+1; j < ndim; ++j)
            strides[i] *= shape[j];
    }
    
    std::vector<int64_t> n_shape(ndim);
    // std::copy(shape.begin(), shape.end(), n_shape.begin());

    std::vector<int64_t> n_stride(ndim);
    for(uint32_t i = 0; i < Perm.size(); ++i){
        n_shape[i] = shape[Perm[i]];
        n_stride[i] = strides[Perm[i]];
    }
    std::copy(n_stride.begin(), n_stride.end(), strides.begin());
    int64_t min_dim = ndim;
    for(uint32_t i = 0; i < Perm.size(); ++i){
        if(Perm[i] == i) continue;
        min_dim = std::min<size_value_t>(min_dim, Perm[i]);
        min_dim = std::min<size_value_t>(min_dim, i);
    }

    int64_t batches = 1;
    for(int64_t i = 0; i < min_dim; ++i)
        batches *= shape[i];

    if(min_dim != 0){
        n_shape.erase(n_shape.begin(), n_shape.begin()+min_dim);
        strides.erase(strides.begin(), strides.begin()+min_dim);
    }

    int64_t total_inner = total_size / batches;
#ifndef _MSC_VER
    tbb::parallel_for(tbb::blocked_range2d<int64_t>(0, batches, 0, total_inner),
    [&](const tbb::blocked_range2d<int64_t>& range){
    const ptr_type* in = &__in[(total_inner * range.rows().begin())];
    ptr_type* out = &__out[(total_inner * range.rows().begin())];
    for(int64_t b = range.rows().begin(); b != range.rows().end(); ++b){
        for(int64_t i = range.cols().begin(); i !=  range.cols().end(); ++i){
            int64_t index = unravel_and_compute(i, total_inner, n_shape, strides);
            out[i] = in[index];
            // std::cout << "out["<<i<<"] = in["<<index<<']'<<std::endl;
        }
        in += total_inner;
        out += total_inner;
    }});
#else
    tbb::parallel_for(tbb::blocked_range2d<int64_t>(0, batches, 0, total_inner),
    [&](const tbb::blocked_range2d<int64_t>& range){
    void** in = &__in[(total_inner * range.rows().begin())];
    void** out = &__out[(total_inner * range.rows().begin())];
    for(int64_t b = range.rows().begin(); b != range.rows().end(); ++b){
        for(int64_t i = range.cols().begin(); i !=  range.cols().end(); ++i){
            int64_t index = unravel_and_compute(i, total_inner, n_shape, strides);
            out[i] = in[index];
            // std::cout << "out["<<i<<"] = in["<<index<<']'<<std::endl;
        }
        in += total_inner;
        out += total_inner;
    }});
#endif
}

SizeRef squeeze_and_adjust_transpose(std::vector<Tensor::size_value_t> size_vec, Tensor::size_value_t& a, Tensor::size_value_t& b){
    //a < b
    for(int i = size_vec.size()-1; i >= 0; --i){
        if(size_vec[i] != 1) continue;
        if(i > b) continue;
        b = (b == 0) ? 0 : b-1;
        if(i > a) continue;
        a = (a == 0) ? 0 : a-1;
    }
    size_vec.erase(std::remove(size_vec.begin(), size_vec.end(), 1), size_vec.end());
    return SizeRef(std::move(size_vec));
}

Tensor transpose(const Tensor& _this, Tensor::size_value_t _a, Tensor::size_value_t _b){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(_this);
    using size_value_t = Tensor::size_value_t;
    _a = _a < 0 ? _this.dims() + _a : _a;
    _b = _b < 0 ? _this.dims() + _b : _b;
    if(std::all_of(_this.shape().begin(), _this.shape().end(), [](const int64_t& i){return i == 1;})){
        return _this;
    }
    if (_a == _b) {
        return _this;
    }

    utils::THROW_EXCEPTION(
    (_a < _this.dims() && _b < _this.dims()) && (_a >= 0 && _b >= 0),
    "a and b ($,$) are out of range for tensor with dimensionality $", _a,
    _b, _this.dims());

    if (_a > _b) {
        std::swap(_a, _b);
    }

    if(std::any_of(_this.shape().begin(), _this.shape().end(), [](const int64_t& i){return i == 1;})){
        //fill in squeeze and adjust
        SizeRef out_shape = _this.shape().transpose(_a, _b);
        SizeRef n_shape = squeeze_and_adjust_transpose(_this.shape().Vec(), _a, _b);
        return transpose(_this.view(n_shape), _a, _b).view(out_shape).set_mutability(_this.is_mutable());
    }
    
    std::vector<size_value_t> cur_strides = _this.getChangedStrides();
    std::swap(cur_strides[_a + 1], cur_strides[_b + 1]);
    ArrayVoid in_vals = _this.arr_void().get_bucket().is_strided()
                                     ? _this.arr_void()
                                     : _this.arr_void().bucket_all_indices();
    void **_in = in_vals.stride_begin();
    ArrayVoid out_vals = _this.arr_void().new_strides(_this.numel());
    void **_out = out_vals.stride_begin();
    transpose(_in, _out, _a, _b, _this.shape().Vec());
    Tensor out(out_vals, _this.shape().transpose(_a, _b));
    out.set_mutability(_this.is_mutable());
    out.set_stored_strides(cur_strides);
    return std::move(out);
}

Tensor& row_col_swap_(Tensor& _this){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(_this);
    check_mutability(_this);
    Tensor transposed = transpose(_this, -1, -2);
    std::swap(_this, transposed);
    return _this;
}

Tensor permute(const Tensor& _this, std::vector<Tensor::size_value_t> Perm){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(_this);
    using size_value_t = Tensor::size_value_t;
    static_assert(std::is_same_v<size_value_t, int64_t>, "INTERNAL LOGIC ERROR: Pemute data type needs to be updated");
    int64_t ndim = _this.dims();
    for(size_t i = 0; i < Perm.size(); ++i)
        Perm[i] = (Perm[i] < 0 ? Perm[i] + ndim : Perm[i]);
    while(Perm.size() < ndim){
        Perm.push_back(Perm.size()-1);
    }
    for(size_t i = 0; i < Perm.size(); ++i){
        utils::throw_exception(Perm[i] >= 0 && Perm[i] < ndim,
                               "Got invalid dimension ($) to permute over at index ($) for tensor of shape ($)",
                               (Perm[i] < 0 ? Perm[i] - ndim : Perm[i]), i, _this.shape());
    }
    if(std::all_of(_this.shape().begin(), _this.shape().end(), [](const int64_t& i){return i == 1;})){
        return _this;
    }
    std::unordered_set<int64_t> set;
    for(size_t i = 0; i < Perm.size(); ++i){
        utils::throw_exception(!set.count(Perm[i]),
                               "Error: Cannot repeat dimensions when permuting");
        set.insert(Perm[i]);
    }
    
    const std::vector<size_value_t> shape = _this.shape().Vec();
    const std::vector<size_value_t> cur_strides = _this.getChangedStrides();
    std::vector<size_value_t> out_shape(shape.size());
    std::vector<size_value_t> out_strides(cur_strides.size());
    for(size_t i = 0; i < Perm.size(); ++i){
        out_shape[i] = shape[Perm[i]];
        out_strides[i] = cur_strides[Perm[i]];
    }

    ArrayVoid in_vals = _this.arr_void().get_bucket().is_strided()
                                     ? _this.arr_void()
                                     : _this.arr_void().bucket_all_indices();
    void **_in = in_vals.stride_begin();
    ArrayVoid out_vals = _this.arr_void().new_strides(_this.numel());
    void **_out = out_vals.stride_begin();
    permute(_in, _out, std::move(Perm), shape);
    Tensor out(out_vals, SizeRef(std::move(out_shape)));
    out.set_mutability(_this.is_mutable());
    out.set_stored_strides(out_strides);
    return std::move(out);
}


}
}

