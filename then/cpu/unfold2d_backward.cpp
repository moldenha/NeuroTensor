#include "../../dtype/ArrayVoid_NTensor.hpp"
#include "../../mp/Threading.h"
// #include "colim_transform.h"

namespace nt {
namespace functional {
namespace cpu {

inline static constexpr auto im2col_backward_cpu =
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
        // TODO: once the error at the todo below is fixed, delete everything
        // below this to there
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

        // TODO: why the fuck is this causing an error with calling this
        // function in the first place:
        /* const int64_t hw_col = height_col * width_col; */

        /* /1* auto data_col_ptr = data_col; *1/ */
        /* const int64_t batch_im_add = (channels * height * width); */
        /* const int64_t batch_col_add = (channels_col * height_col *
         * width_col); */
        /* #ifdef USE_PARALLEL */
        /* if constexpr (std::is_pointer_v<decltype(data_im_ptr)>){ */
        /* threading::preferential_parallel_for(threading::block_ranges<2>(0,
         * batch_size, 0, channels_col), */
        /* [&](threading::blocked_range<2> block){ */
        /* for(int64_t batch = block.begin[0]; batch < block.end[0]; ++batch){
         */
        /* auto data_col_ptr = data_col + batch * batch_col_add; */
        /* auto data_im = data_im_ptr + batch * batch_im_add; */
        /* for(int64_t c = block.begin[1]; c < block.end[1]; ++c){ */
        /* int64_t w_offset = c % k_c; */
        /* int64_t h_offset = (c / k_c) % k_r; */
        /* int64_t c_im = c / k_r / k_c; */
        /* // Calculate the valid range for h_pad */
        /* int64_t h_pad_start = std::max(int64_t(0), (p_r - h_offset * d_r + s_r - 1)
         * / s_r); */
        /* int64_t h_pad_end = std::min(height_col, (height + p_r - h_offset *
         * d_r + s_r - 1) / s_r); */

        /* // Calculate the valid range for w_pad */
        /* int64_t w_pad_start = std::max(int64_t(0), (p_c - w_offset * d_c + s_c - 1)
         * / s_c); */
        /* int64_t w_pad_end = std::min(width_col, (width + p_c - w_offset * d_c
         * + s_c - 1) / s_c); */

        /* for (int64_t h = h_pad_start; h < h_pad_end; ++h) { */
        /* int64_t h_pad = h * s_r - p_r + h_offset * d_r; */
        /* for (int64_t w = w_pad_start; w < w_pad_end; ++w) { */
        /* int64_t w_pad = w * s_c - p_c + w_offset * d_c; */
        /* data_im[c_im * height * width + h_pad * width + w_pad] */
        /* += data_col_ptr[c * height_col * width_col + h * width_col + w]; */
        /* /1* data_col_ptr[c * height_col * width_col + h * width_col + w] *1/
         */
        /* } */
        /* } */
        /* } */
        /* } */
        /* }); */
        /*   return; */
        /* } */
        /* #endif */
        /* nt::threading::iterator_parallel_for([&](auto range, auto data_im){
         * //works whether parallel or not, but obviously faster when parallel
         */
        /* auto data_col_ptr = data_col + (batch_col_add * range.begin[0]); */
        /* for (int64_t batch = range.begin[0]; batch < range.end[0]; ++batch,
         * data_im += batch_im_add) { */
        /* for (int64_t c = 0; c < channels_col; ++c) { */
        /* int64_t w_offset = c % k_c; */
        /* int64_t h_offset = (c / k_c) % k_r; */
        /* int64_t c_im = c / k_r / k_c; */
        /* // Calculate the valid range for h_pad */
        /*             int64_t h_pad_start = std::max(int64_t(0), (p_r - h_offset * d_r
         * + s_r - 1) / s_r); */
        /*             int64_t h_pad_end = std::min(height_col, (height + p_r -
         * h_offset * d_r + s_r - 1) / s_r); */

        /*             // Calculate the valid range for w_pad */
        /*             int64_t w_pad_start = std::max(int64_t(0), (p_c - w_offset * d_c
         * + s_c - 1) / s_c); */
        /*             int64_t w_pad_end = std::min(width_col, (width + p_c -
         * w_offset * d_c + s_c - 1) / s_c); */

        /* //this is a really good check that negates a lot of edge case errors,
         * and skips doing a lot of loops */
        /* if(h_pad_start >= height_col || h_pad_end <= h_pad_start */
        /* || w_pad_start >= width_col || w_pad_end <= w_pad_start){ */
        /* data_col_ptr += hw_col; */
        /* continue; */
        /* } */
        /* data_col_ptr += (h_pad_start * width_col); */
        /* for(int64_t h = h_pad_start; h < h_pad_end; ++h){ */
        /* data_col_ptr += w_pad_start; */
        /* int64_t h_pad = h * s_r - p_r + h_offset * d_r; */
        /* for(int64_t w = w_pad_start; w < w_pad_end; ++w){ */
        /* int64_t w_pad = w * s_c - p_c + w_offset * d_c; */
        /* data_im[c_im * height * width + h_pad * width + w_pad] +=
         * *data_col_ptr; */
        /* ++data_col_ptr; */
        /* } */
        /* // Skip to the next valid row in data_col_ptr */
        /* data_col_ptr += width_col - w_pad_end; */
        /* } */
        /* // Adjust data_col_ptr for the skipped rows and columns */
        /*             data_col_ptr += (height_col - h_pad_end) * width_col; */
        /*     } */
        /* } */

        /*  }, 0, batch_size, data_im_ptr, batch_im_add); */
    };

void unfold2d_backward_(ArrayVoid &output, ArrayVoid &grad,
                       const int64_t &channels, const int64_t &height,
                       const int64_t &width, const int64_t &k_r,
                       const int64_t &k_c, const int64_t &s_r,
                       const int64_t &s_c, const int64_t &p_r,
                       const int64_t &p_c, const int64_t &d_r,
                       const int64_t &d_c, const int64_t &batch_size) {
    output.execute_function < WRAP_DTYPES<NumberTypesL> >(
                                  im2col_backward_cpu, grad, channels, height, width, k_r, k_c,
                                  s_r, s_c, p_r, p_c, d_r, d_c, batch_size);
}

} // namespace cpu
} // namespace functional
} // namespace nt
