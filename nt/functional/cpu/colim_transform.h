#ifndef NT_CPU_UNFOLD_ARRAY_VOID_H__
#define NT_CPU_UNFOLD_ARRAY_VOID_H__
#include "../../dtype/ArrayVoid.h"

namespace nt {
namespace functional {
namespace cpu {

NEUROTENSOR_API void unfold1d_(ArrayVoid &im, ArrayVoid &col, const int64_t& channels,
    const int64_t& width,
    const int64_t& k_w,
    const int64_t& s_w,
    const int64_t& p_w,
    const int64_t& d_w,
    const int64_t& batch_size);


NEUROTENSOR_API void unfold1d_backward_(ArrayVoid &output, ArrayVoid &grad, const int64_t& channels,
    const int64_t& width,
    const int64_t& k_w,
    const int64_t& s_w,
    const int64_t& p_w,
    const int64_t& d_w,
    const int64_t& batch_size);

NEUROTENSOR_API void unfold2d_(ArrayVoid &im, ArrayVoid &col, const int64_t &channels,
               const int64_t &height, const int64_t &width, const int64_t &k_r,
               const int64_t &k_c, const int64_t &s_r, const int64_t &s_c,
               const int64_t &p_r, const int64_t &p_c, const int64_t &d_r,
               const int64_t &d_c, const int64_t &batch_size,
               bool transpose_out);

NEUROTENSOR_API void unfold2d_backward_(ArrayVoid &output, ArrayVoid &grad,
                       const int64_t &channels, const int64_t &height,
                       const int64_t &width, const int64_t &k_r,
                       const int64_t &k_c, const int64_t &s_r,
                       const int64_t &s_c, const int64_t &p_r,
                       const int64_t &p_c, const int64_t &d_r,
                       const int64_t &d_c, const int64_t &batch_size); 

NEUROTENSOR_API void unfold3d_(ArrayVoid &im, ArrayVoid &col, const int64_t &channels,
               const int64_t &depth, const int64_t &height,
               const int64_t &width, const int64_t &k_d, const int64_t &k_r,
               const int64_t &k_c, const int64_t &s_d, const int64_t &s_r,
               const int64_t &s_c, const int64_t &p_d, const int64_t &p_r,
               const int64_t &p_c, const int64_t &d_d, const int64_t &d_r,
               const int64_t &d_c, const int64_t &batch_size);

NEUROTENSOR_API void unfold3d_backward_(ArrayVoid &output, ArrayVoid &grad,
                        const int64_t &channels, const int64_t &depth,
                        const int64_t &height, const int64_t &width,
                        const int64_t &k_d, const int64_t &k_r,
                        const int64_t &k_c, const int64_t &s_d,
                        const int64_t &s_r, const int64_t &s_c,
                        const int64_t &p_d, const int64_t &p_r,
                        const int64_t &p_c, const int64_t &d_d,
                        const int64_t &d_r, const int64_t &d_c,
                        const int64_t &batch_size);

NEUROTENSOR_API void unfoldnd_(ArrayVoid& im, ArrayVoid& col, const int64_t& channels,
                            const std::vector<int64_t>& size, 
                            const std::vector<int64_t>& kernels,
                            const std::vector<int64_t>& strides, 
                            const std::vector<int64_t>& paddings,
                            const std::vector<int64_t>& dilations, 
                            const int64_t& batch_size);

NEUROTENSOR_API void unfoldnd_backward_(ArrayVoid &output, ArrayVoid &grad, const int64_t& channels,
                                const std::vector<int64_t>& size, const std::vector<int64_t>& kernels,
                                const std::vector<int64_t>& strides, const std::vector<int64_t>& paddings,
                                const std::vector<int64_t>& dilations, const int64_t& batch_size);

NEUROTENSOR_API void fold1d_(ArrayVoid &im, ArrayVoid &col, const int64_t& channels,
                                const int64_t& width,
                                const int64_t& k_w,
                                const int64_t& s_w,
                                const int64_t& p_w,
                                const int64_t& d_w,
                                const int64_t& batch_size);

NEUROTENSOR_API void fold1d_backward_(ArrayVoid &output, ArrayVoid &grad, const int64_t& channels,
                                        const int64_t& width,
                                        const int64_t& k_w,
                                        const int64_t& s_w,
                                        const int64_t& p_w,
                                        const int64_t& d_w,
                                        const int64_t& batch_size);


NEUROTENSOR_API void fold2d_(ArrayVoid &im, ArrayVoid &col, const int64_t &channels,
             const int64_t &height, const int64_t& width,
             const int64_t &k_r, const int64_t &k_c,
             const int64_t &s_r, const int64_t &s_c, 
             const int64_t &p_r, const int64_t &p_c,
             const int64_t &d_r, const int64_t &d_c, 
             const int64_t &batches);

NEUROTENSOR_API void fold2d_backward_(ArrayVoid &output, ArrayVoid &grad, const int64_t& channels,
                      const int64_t& height, const int64_t& width,
                      const int64_t &k_r, const int64_t &k_c,
                      const int64_t &s_r, const int64_t &s_c,
                      const int64_t &p_r, const int64_t &p_c,
                      const int64_t &d_r, const int64_t &d_c,
                      const int64_t &batches);



NEUROTENSOR_API void fold3d_(ArrayVoid &im, ArrayVoid &col,
                        const int64_t &channels, const int64_t &depth,
                        const int64_t &height, const int64_t &width,
                        const int64_t &k_d, const int64_t &k_r,
                        const int64_t &k_c, const int64_t &s_d,
                        const int64_t &s_r, const int64_t &s_c,
                        const int64_t &p_d, const int64_t &p_r,
                        const int64_t &p_c, const int64_t &d_d,
                        const int64_t &d_r, const int64_t &d_c,
                        const int64_t &batch_size);


NEUROTENSOR_API void fold3d_backward_(ArrayVoid &output, ArrayVoid &grad, const int64_t& channels,
                        const int64_t& depth, const int64_t& height, const int64_t& width,
                        const int64_t& k_d, const int64_t& k_r, const int64_t& k_c, 
                        const int64_t& s_d, const int64_t& s_r, const int64_t& s_c, 
                        const int64_t& p_d, const int64_t& p_r, const int64_t& p_c, 
                        const int64_t& d_d, const int64_t& d_r, const int64_t& d_c,
                        const int64_t& batch_size);

NEUROTENSOR_API void foldnd_(ArrayVoid &im, ArrayVoid &col, const int64_t& channels,
                                const std::vector<int64_t>& size, const std::vector<int64_t>& kernels,
                                const std::vector<int64_t>& strides, const std::vector<int64_t>& paddings,
                                const std::vector<int64_t>& dilations, const int64_t& batch_size);

NEUROTENSOR_API void foldnd_backward_(ArrayVoid &output, ArrayVoid &grad, const int64_t& channels,
                                const std::vector<int64_t>& size, const std::vector<int64_t>& kernels,
                                const std::vector<int64_t>& strides, const std::vector<int64_t>& paddings,
                                const std::vector<int64_t>& dilations, const int64_t& batch_size); 

} // namespace cpu
} // namespace functional
} // namespace nt

#endif
