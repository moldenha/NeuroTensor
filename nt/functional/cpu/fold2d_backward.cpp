#include "../../dtype/ArrayVoid_NTensor.hpp"
#include "../../mp/Threading.h"
// #include "colim_transform.h"

namespace nt {
namespace functional {
namespace cpu {

// // basically just one line to switch:
// inline static constexpr auto col2im_cpu_backward =
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
//         for (int64_t n = 0; n < batches; ++n) {
//             for (int64_t ch = 0; ch < channels;
//                  ++ch, o_begin += o_channel_add, a_begin += a_channel_add) {
//                 for (int64_t r = 0; r < row_upper_limit; ++r) {
//                     for (int64_t c = 0; c < col_upper_limit; ++c) {
//                         int64_t o_row = (r * s_r) + (c * d_r) - p_r;
//                         if (o_row >= o_r || o_row < 0) {
//                             continue;
//                         }
//                         int64_t a_col_index = r * L_c;
//                         int64_t a_row_index = c * L_r;
//                         auto o_cpy = o_begin + (o_row * o_c);
//                         auto a_cpy =
//                             a_begin + (a_row_index * col_add + a_col_index);
//                         for (int64_t kr = 0; kr < kernel_row_upper_limit;
//                              ++kr) {
//                             for (int64_t kc = 0; kc < kernel_col_upper_limit;
//                                  ++kc) {
//                                 int64_t o_col = (kc * d_c) + (kr * s_c) - p_c;
//                                 if (o_col < o_c && o_col >= 0) {
//                                     a_cpy[col_add * kc + kr] +=
//                                         o_cpy[o_col]; // switched line
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     };


inline static constexpr auto col2im_cpu_backward =
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
                                    *data_col_ptr +=
                                        data_im[c_im * height * width +
                                            h_pad * width + w_pad];
                                }
                                ++data_col_ptr;
                            }
                        }
                    }
                }

            },
            0, batch_size, data_im_ptr, batch_im_add);
            // auto data_col_ptr = data_col + (batch_col_add * 0);
            // for (int64_t batch = 0; batch < batch_size;
            //      ++batch, data_im_ptr += batch_im_add) {
            //     for (int64_t c = 0; c < channels_col; ++c) {
            //         int64_t w_offset = c % k_c;
            //         int64_t h_offset = (c / k_c) % k_r;
            //         int64_t c_im = c / k_r / k_c;
            //         for (int64_t h = 0; h < height_col; ++h) {
            //             for (int64_t w = 0; w < width_col; ++w) {
            //                 int64_t h_pad = h * s_r - p_r + h_offset * d_r;
            //                 int64_t w_pad = w * s_c - p_c + w_offset * d_c;
            //                 if (h_pad >= 0 && h_pad < height &&
            //                     w_pad >= 0 && w_pad < width) {
            //                     *data_col_ptr +=
            //                         data_im_ptr[c_im * height * width +
            //                             h_pad * width + w_pad];
            //                 }
            //                 ++data_col_ptr;
            //             }
            //         }
            //     }
            // }


    };




void fold2d_backward_(ArrayVoid &output, ArrayVoid &grad, const int64_t& channels,
                      const int64_t& height, const int64_t& width,
                      const int64_t &k_r, const int64_t &k_c,
                      const int64_t &s_r, const int64_t &s_c,
                      const int64_t &p_r, const int64_t &p_c,
                      const int64_t &d_r, const int64_t &d_c,
                      const int64_t &batches) {

    grad.execute_function<WRAP_DTYPES<NumberTypesL>>( col2im_cpu_backward, output,
            channels, height, width, k_r, k_c,
                      s_r, s_c, p_r, p_c, d_r, d_c, batches
    );

    // output.execute_function<WRAP_DTYPES<NumberTypesL>>(
    //     col2im_cpu_backward, grad, col_add, L_r, L_c, o_r, o_c, k_r, k_c, s_r,
    //     s_c, d_r, d_c, p_r, p_c, batches, channels);
}

} // namespace cpu
} // namespace functional
} // namespace nt
