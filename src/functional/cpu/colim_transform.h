#ifndef __NT_CPU_UNFOLD_ARRAY_VOID_H__
#define __NT_CPU_UNFOLD_ARRAY_VOID_H__
#include "../../dtype/ArrayVoid.h"

namespace nt {
namespace functional {
namespace cpu {

void unfold2d_(ArrayVoid &im, ArrayVoid &col, const int64_t &channels,
               const int64_t &height, const int64_t &width, const int64_t &k_r,
               const int64_t &k_c, const int64_t &s_r, const int64_t &s_c,
               const int64_t &p_r, const int64_t &p_c, const int64_t &d_r,
               const int64_t &d_c, const int64_t &batch_size,
               bool transpose_out);

void unfold2d_backward_(ArrayVoid &output, ArrayVoid &grad,
                        const int64_t &channels, const int64_t &height,
                        const int64_t &width, const int64_t &k_r,
                        const int64_t &k_c, const int64_t &s_r,
                        const int64_t &s_c, const int64_t &p_r,
                        const int64_t &p_c, const int64_t &d_r,
                        const int64_t &d_c, const int64_t &batch_size);

void fold2d_(ArrayVoid &im, ArrayVoid &col, const int64_t &col_add,
             const int64_t &L_r, const int64_t &L_c, const int64_t &o_r,
             const int64_t &o_c, const int64_t &k_r, const int64_t k_c,
             const int64_t &s_r, const int64_t &s_c, const int64_t &d_r,
             const int64_t &d_c, const int64_t &p_r, const int64_t p_c,
             const int64_t &batches, const int64_t &channels);

void fold2d_backward_(ArrayVoid &output, ArrayVoid &grad,
                      const int64_t &col_add, const int64_t &L_r,
                      const int64_t &L_c, const int64_t &o_r,
                      const int64_t &o_c, const int64_t &k_r, const int64_t k_c,
                      const int64_t &s_r, const int64_t &s_c,
                      const int64_t &d_r, const int64_t &d_c,
                      const int64_t &p_r, const int64_t p_c,
                      const int64_t &batches, const int64_t &channels);

void unfold3d_(ArrayVoid &im, ArrayVoid &col, const int64_t &channels,
               const int64_t &depth, const int64_t &height,
               const int64_t &width, const int64_t &k_d, const int64_t &k_r,
               const int64_t &k_c, const int64_t &s_d, const int64_t &s_r,
               const int64_t &s_c, const int64_t &p_d, const int64_t &p_r,
               const int64_t &p_c, const int64_t &d_d, const int64_t &d_r,
               const int64_t &d_c, const int64_t &batch_size);

void unfold3d_backward_(ArrayVoid &output, ArrayVoid &grad,
                        const int64_t &channels, const int64_t &depth,
                        const int64_t &height, const int64_t &width,
                        const int64_t &k_d, const int64_t &k_r,
                        const int64_t &k_c, const int64_t &s_d,
                        const int64_t &s_r, const int64_t &s_c,
                        const int64_t &p_d, const int64_t &p_r,
                        const int64_t &p_c, const int64_t &d_d,
                        const int64_t &d_r, const int64_t &d_c,
                        const int64_t &batch_size);

void unfold1d_(ArrayVoid &im, ArrayVoid &col, const int64_t& channels,
    const int64_t& width,
    const int64_t& k_w,
    const int64_t& s_w,
    const int64_t& p_w,
    const int64_t& d_w,
    const int64_t& batch_size);

void unfold1d_(ArrayVoid &output, ArrayVoid &grad, const int64_t& channels,
    const int64_t& width,
    const int64_t& k_w,
    const int64_t& s_w,
    const int64_t& p_w,
    const int64_t& d_w,
    const int64_t& batch_size);

void unfold1d_backward_(ArrayVoid &output, ArrayVoid &grad, const int64_t& channels,
    const int64_t& width,
    const int64_t& k_w,
    const int64_t& s_w,
    const int64_t& p_w,
    const int64_t& d_w,
    const int64_t& batch_size);

} // namespace cpu
} // namespace functional
} // namespace nt

#endif
