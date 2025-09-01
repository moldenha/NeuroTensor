#include "../../dtype/ArrayVoid_NTensor.hpp"
#include "../../mp/Threading.h"
// #include "colim_transform.h"

// Look back at this, for now going to use Unfold2D backward

namespace nt {
namespace functional {
namespace cpu {
// inline static constexpr auto col2im_cpu =
//     [](auto a_begin, auto a_end, auto o_begin, const int64_t &col_add,
//        const int64_t &L_r, const int64_t &L_c, const int64_t &o_r,
//        const int64_t &o_c, const int64_t &k_r, const int64_t k_c,
//        const int64_t &s_r, const int64_t &s_c, const int64_t &d_r,
//        const int64_t &d_c, const int64_t &p_r, const int64_t p_c,
//        const int64_t &batches, const int64_t &channels) {
//         const int64_t &row_upper_limit = k_c;
//         const int64_t &col_upper_limit = k_r;
//         const int64_t &kernel_row_upper_limit = L_c;
//         const int64_t &kernel_col_upper_limit = L_r;
//         const int64_t o_channel_add = o_r * o_c;
//         const int64_t a_channel_add = k_r * k_c * col_add;
//         const int64_t C =  (channels * col_add) / (k_r * k_c);
//         for (int64_t n = 0; n < batches; ++n) {
//             for (int64_t i = 0; i < OH; ++i) {
//                 for (int64_t j = 0; j < OW; ++j) {
//                     int64_t col_idx = i * OW + j;

//                     for (int64_t kh = 0; kh < KH; ++kh) {
//                         for (int64_t kw = 0; kw < KW; ++kw) {
//                             int64_t out_h = i * SH - PH + kh * DH;
//                             int64_t out_w = j * SW - PW + kw * DW;

//                             if (out_h >= 0 && out_h < H_out && out_w >= 0 && out_w < W_out) {
//                                 for (int64_t c = 0; c < C; ++c) {
//                                     int64_t patch_idx = c * KH * KW + kh * KW + kw;
//                                     output(n, c, out_h, out_w) += input(n, patch_idx, col_idx);
//                                     counter(n, c, out_h, out_w) += 1;
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//         // for (int64_t n = 0; n < batches; ++n) {
//         //     for (int64_t ch = 0; ch < channels; ++ch) {
//         //         for (int64_t r = 0; r < L_r; ++r) {
//         //             for (int64_t c = 0; c < L_c; ++c) {
//         //                 int64_t col_index = r * L_c + c;

//         //                 for (int64_t kr = 0; kr < k_r; ++kr) {
//         //                     for (int64_t kc = 0; kc < k_c; ++kc) {
//         //                         int64_t out_row = r * s_r - p_r + kr * d_r;
//         //                         int64_t out_col = c * s_c - p_c + kc * d_c;

//         //                         if (out_row < 0 || out_row >= o_r || out_col < 0 || out_col >= o_c)
//         //                             continue;

//         //                         // Index in output tensor (o_begin points to NCHW flat array)
//         //                         int64_t out_index = (
//         //                             n * channels * o_r * o_c +
//         //                             ch * o_r * o_c +
//         //                             out_row * o_c +
//         //                             out_col
//         //                         );

//         //                         // Index in input column tensor
//         //                         int64_t col_data_index = (
//         //                             ((ch * k_r + kr) * k_c + kc) * L_r * L_c +
//         //                             col_index
//         //                         );

//         //                         // SAFETY CHECK (optional but good for debugging)
//         //                         if (col_data_index < 0 || out_index < 0) {
//         //                             std::cerr << "Invalid index: col=" << col_data_index << ", out=" << out_index << "\n";
//         //                             continue;
//         //                         }

//         //                         o_begin[out_index] += a_begin[col_data_index];
//         //                     }
//         //                 }
//         //             }
//         //         }
//         //     }
//         // }

//     };


inline static constexpr auto col2im_cpu =
    [](auto data_im_ptr, auto data_im_ptr_end, auto data_col,
       const int64_t &channels, const int64_t &height, const int64_t &width,
       const int64_t &k_r, const int64_t &k_c, const int64_t &s_r,
       const int64_t &s_c, const int64_t &p_r, const int64_t &p_c,
       const int64_t &d_r, const int64_t &d_c, const int64_t &batch_size) {
        const int64_t height_col =
            (height + 2 * p_r - d_r * (k_r - 1) - 1) / s_r + 1;
        const int64_t width_col =
            (width + 2 * p_c - d_c * (k_c - 1) - 1) / s_c + 1;
        const int64_t channels_col = channels * k_r * k_c;
        const int64_t col_matrix_size =
            batch_size * channels_col * height_col * width_col;

        /* auto data_col_ptr = data_col; */
        const int64_t batch_im_add = (channels * height * width);
        const int64_t batch_col_add = (channels_col * height_col * width_col);
        nt::threading::iterator_parallel_for(
            [&](auto range,
                auto data_im) { // works whether parallel or not, but obviously
                                // faster when parallel
                auto data_col_ptr = data_col + (batch_col_add * range.begin[0]);
                for (int64_t batch = range.begin[0]; batch < range.end[0];
                     ++batch, data_im += batch_im_add) {

                    for (int64_t c = 0; c < channels_col; ++c) {
                        int64_t w_offset = c % k_c;
                        int64_t h_offset = (c / k_c) % k_r;
                        int64_t c_im = c / k_r / k_c;
                        for (int64_t h = 0; h < height_col; ++h) {
                            for (int64_t w = 0; w < width_col; ++w) {
                                int64_t h_pad = h * s_r - p_r + h_offset * d_r;
                                int64_t w_pad = w * s_c - p_c + w_offset * d_c;
                                if (h_pad >= 0 && h_pad < height &&
                                    w_pad >= 0 && w_pad < width) {
                                    data_im[c_im * height * width +
                                            h_pad * width + w_pad] +=
                                        *data_col_ptr;
                                }
                                ++data_col_ptr;
                            }
                        }
                    }
                }

            },
            0, batch_size, data_im_ptr, batch_im_add);

    };


void fold2d_(ArrayVoid &im, ArrayVoid &col, const int64_t &channels,
             const int64_t &height, const int64_t& width,
             const int64_t &k_r, const int64_t &k_c,
             const int64_t &s_r, const int64_t &s_c, 
             const int64_t &p_r, const int64_t &p_c,
             const int64_t &d_r, const int64_t &d_c, 
             const int64_t &batches) {
    im.execute_function<WRAP_DTYPES<NumberTypesL>>( col2im_cpu, col,
            channels, height, width, k_r, k_c,
                      s_r, s_c, p_r, p_c, d_r, d_c, batches
    );
}

} // namespace cpu
} // namespace functional
} // namespace nt
