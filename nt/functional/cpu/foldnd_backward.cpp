#include "../../dtype/ArrayVoid_NTensor.hpp"
#include "../../mp/Threading.h"
#include <vector>
#include <numeric>
#include <algorithm>
// #include "colim_transform.h"

namespace nt {
namespace functional {
namespace cpu{

// NT_ALWAYS_INLINE int64_t multiply(const std::vector<int64_t>& vals, const int64_t& init){
//     return std::accumulate<int64_t>(vals.cbegin(), vals.cend(), init, std::multiplies<int64_t>());
// }


// // offsets.size() == (any of the sizes + 1)
// // look at lines 70, 85, and 108 for why
// template<typename T, typename T2>
// NT_ALWAYS_INLINE void run_inner_loop(size_t i, int64_t col_add, int64_t img_add, std::vector<int64_t> indexes, const std::vector<int64_t>& pad_start, 
//                        const std::vector<int64_t>& pad_end, const std::vector<int64_t>& offsets,
//                        const std::vector<int64_t>& cols, const std::vector<int64_t>& size, const std::vector<int64_t>& kernel, 
//                        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation,
//                         T& im_ptr, T2& col_ptr){
//     if(i == indexes.size()){
//         col_ptr[col_add] = im_ptr[im_add];
//         return;
//     }
//     const int64_t& my_index = indexes[i];
//     if(my_index < pad_end[i]-1){
//         std::vector<int64_t> index_cpy(indexes.size());
//         std::copy(indexes.cbegin(), indexes.cend(), index_cpy.begin());
//         index_cpy[i] = my_index+1;
//         run_inner_loop(i, col_add, img_add, std::move(index_cpy), pad_start, pad_end, offsets, cols, size, kernel, stride, padding, dilation, im_ptr, col_ptr);
//     }
//     const int64_t my_col_add = i == 0 ? my_index : std::accumulate<int64_t>(cols.cbegin(), cols.cbegin() + i, my_index, std::multiplies<int64_t>());
//     col_add += my_col_add;
//     const int64_t cur_pad = (my_index * stride[i] - padding[i] + offset[i] * dilation[i]);
//     img_add += (i == 0) ? cur_pad : std::accumulate<int64_t>(size.cbegin(), size.cbegin()+i, cur_pad, std::multiplies<int64_t>());
//     run_inner_loop(i+1, col_add, img_add, indexes, pad_start,
//                    pad_end, offsets, cols, size, kernel,
//                    stride, padding, dilation, im_ptr, col_ptr);
// }

// std::vector<int64_t> compute_strides(const std::vector<int64_t>& dims) {
//     std::vector<int64_t> strides(dims.size(), 1);
//     for (int i = dims.size() - 2; i >= 0; --i)
//         strides[i] = strides[i+1] * dims[i+1];
//     return strides;
// }

// inline static constexpr auto im2col_cpu_nd = [](auto data_im_ptr, auto data_im_ptr_end, auto data_col,
//     const int64_t& channels,
//     const std::vector<int64_t>& size,     // width, height, depth, ... [or more] stops at channels
//     const std::vector<int64_t>& kernel,   // k_c, k_r, k_d, ...
//     const std::vector<int64_t>& stride,   // s_c, s_r, s_d, ...
//     const std::vector<int64_t>& padding,  // p_c, p_r, p_d, ...
//     const std::vector<int64_t>& dilation, // d_c, d_r, d_d, ...
//     const int64_t& batch_size) {
    
//     std::vector<int64_t> cols(size.size(), 0);
//     for(size_t i = 0; i < size.size(); ++i){
//         cols[i] = (size[i] + 2 * padding[i] - dilation[i] * (kernel[i] - 1) - 1) / stride[i] + 1;
//     }
//     const int64_t channels_col = multiply(kernel, channels);
//     const int64_t col_matrix_size = multiply(cols, channels_col) * batch_size;

//     std::vector<std::vector<int64_t>> offsets(channels_col);
//     std::vector<std::vector<int64_t>> pad_starts(channels_col);
//     std::vector<std::vector<int64_t>> pad_ends(channels_col);
//     pad_starts[0] = std::vector<int64_t>(size.size(), 0);
//     pad_ends[0] = std::vector<int64_t>(size.size(), 0);
//     for(size_t i = 0; i < size.size(); ++i){
//         pad_starts[0][i] = std::max(int64_t(0), (padding[i] - 0 * dilation[i] + stride[i] - 1) / stride[i]);
//         pad_ends[0][i] = std::min(cols[i], (size[i] + padding[i] - 0 * dilation[i] + stride[i] - 1) / stride[i]);
//     }
//     offsets[0] = std::vector<int64_t>(size.size()+1, 0);
//     for(int64_t c = 1; c < channels_col; ++c){
//         std::vector<int64_t> offset(size.size() + 1);
//         offset[0] = c % kernel[0];
//         pad_starts[c] = std::vector<int64_t>(size.size());
//         pad_ends[c] = std::vector<int64_t>(size.size());
//         pad_starts[c][0] = std::max(int64_t(0), (padding[0] - offset[0] * dilation[0] + stride[0] - 1) / stride[0]);
//         pad_ends[c][0] = std::min(cols[0], (size[0] + padding[0] - offset[0] * dilation[0] + stride[0] - 1) / stride[0]);

//         for(size_t i = 1; i < size.size(); ++i){
//             int64_t cur_offset = std::accumulate<int64_t>(size.cbegin(), size.cbegin() + i, c, std::divides<int64_t>()) % size[i];
//             pad_starts[c][i] = std::max(int64_t(0), (padding[i] - cur_offset * dilation[i] + stride[i] - 1) / stride[i]);
//             pad_ends[c][i] = std::min(cols[i], (size[i] + padding[i] - cur_offset * dilation[i] + stride[i] - 1) / stride[i]); 
//             offset[i] = cur_offset;
//         }
//         offset.back() = std::accumulate<int64_t>(size.cbegin(), size.cend(), c, std::divides<int64_t>());
//         offsets[c] = offset;
//     }
    

    


//     const int64_t batch_im_add = multiply(size, channels);
//     const int64_t batch_col_add = multiply(cols, channels_col);
//     // const int64_t dhw_col = depth_col * height_col * width_col;
//     // const int64_t hw_col =  height_col * width_col;

// 	threading::preferential_parallel_for(threading::block_ranges<1>(0, batch_size),
// 		[&](threading::blocked_range<1> block){
// 	for(int64_t batch = block.begin[0]; batch < block.end[0]; ++batch){
// 		auto data_col_ptr = data_col + batch * batch_col_add;
// 		auto data_im = data_im_ptr + batch * batch_im_add;
// 		for(int64_t c = 0; c < channels_col; ++c){
// 			// int64_t w_offset = c % k_c;
// 			// int64_t h_offset = (c / k_c) % k_r;
// 			// int64_t d_offset = (c / k_c / k_r) % k_d;
//             const std::vector<int64_t>& offset = offsets[c]; // size = 4
// 			const int64_t& c_im = offset.back();
// 			// Calculate the valid range for h_pad
// 			// int64_t h_pad_start = std::max(int64_t(0), (p_r - h_offset * d_r + s_r - 1) / s_r);
// 			// int64_t h_pad_end = std::min(height_col, (height + p_r - h_offset * d_r + s_r - 1) / s_r);

// 			// // Calculate the valid range for w_pad
// 			// int64_t w_pad_start = std::max(int64_t(0), (p_c - w_offset * d_c + s_c - 1) / s_c);
// 			// int64_t w_pad_end = std::min(width_col, (width + p_c - w_offset * d_c + s_c - 1) / s_c);

// 			// // Calculate the valid range for d_pad
// 			// int64_t d_pad_start = std::max(int64_t(0), (p_d - d_offset * d_d + s_d - 1) / s_d);
// 			// int64_t d_pad_end = std::min(depth_col, (depth + p_d - d_offset * d_d + s_d - 1) / s_d);
//             const std::vector<int64_t>& pad_start = pad_starts[c];
//             const std::vector<int64_t>& pad_end = pad_ends[c];
//             const int64_t col_add = multiply(cols, c);
//             const int64_t im_add = multiply(size, c_im);
//             std::vector<int64_t> indexes = pad_start;
//             run_inner_loop(0, col_add, im_add, indexes, pad_start, pad_end, offset, cols, size, kernel, stride, padding, dilation, data_im, data_col_ptr);
// 		}
// 	}
// 	});
// };


NT_ALWAYS_INLINE int64_t multiply(const std::vector<int64_t>& vals, int64_t init = 1) {
    return std::accumulate(vals.begin(), vals.end(), init, std::multiplies<int64_t>());
}

NT_ALWAYS_INLINE std::vector<int64_t> compute_strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (int i = dims.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * dims[i + 1];
    return strides;
}

template<typename T, typename T2>
inline void run_inner_foldnd_backward_loop(
    size_t i,
    int64_t col_add,
    int64_t img_add,
    std::vector<int64_t>& indexes,
    const std::vector<int64_t>& pad_start,
    const std::vector<int64_t>& pad_end,
    const std::vector<int64_t>& offset,
    const std::vector<int64_t>& cols,
    const std::vector<int64_t>& img_strides,
    const std::vector<int64_t>& col_strides,
    const std::vector<int64_t>& size,
    const std::vector<int64_t>& kernel,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    T& im_ptr,
    T2& col_ptr
) {
    if (i == indexes.size()) {
        col_ptr[col_add] += im_ptr[img_add];
        return;
    }

    for (int64_t idx = pad_start[i]; idx < pad_end[i]; ++idx) {
        indexes[i] = idx;
        int64_t col_off = idx * col_strides[i];
        int64_t im_off = (idx * stride[i] - padding[i] + offset[i] * dilation[i]) * img_strides[i];
        run_inner_foldnd_backward_loop(i + 1,
                       col_add + col_off,
                       img_add + im_off,
                       indexes,
                       pad_start,
                       pad_end,
                       offset,
                       cols,
                       img_strides,
                       col_strides,
                       size,
                       kernel,
                       stride,
                       padding,
                       dilation,
                       im_ptr,
                       col_ptr);
    }
}

inline static constexpr auto col2im_cpu_nd_backward = [](auto data_im_ptr, auto data_im_ptr_end, auto data_col,
    const int64_t& channels,
    const std::vector<int64_t>& size,      // spatial dims (e.g., D, H, W)
    const std::vector<int64_t>& kernel,    // kernel size
    const std::vector<int64_t>& stride,    // stride
    const std::vector<int64_t>& padding,   // padding
    const std::vector<int64_t>& dilation,  // dilation
    const int64_t& batch_size) {

    const size_t dims = size.size();
    std::vector<int64_t> cols(dims);
    for (size_t i = 0; i < dims; ++i) {
        cols[i] = (size[i] + 2 * padding[i] - dilation[i] * (kernel[i] - 1) - 1) / stride[i] + 1;
    }

    const int64_t channels_col = channels * multiply(kernel);
    const int64_t batch_im_add = channels * multiply(size);
    const int64_t batch_col_add = channels_col * multiply(cols);

    std::vector<int64_t> img_strides = compute_strides(size);
    std::vector<int64_t> col_strides = compute_strides(cols);

    // Precompute offsets, pad ranges
    std::vector<std::vector<int64_t>> offsets(channels_col, std::vector<int64_t>(dims));
    std::vector<int64_t> channel_map(channels_col);
    std::vector<std::vector<int64_t>> pad_starts(channels_col, std::vector<int64_t>(dims));
    std::vector<std::vector<int64_t>> pad_ends(channels_col, std::vector<int64_t>(dims));

    for (int64_t c = 0; c < channels_col; ++c) {
        int64_t rem = c;
        int64_t channel = rem / multiply(kernel);
        channel_map[c] = channel;
        rem = rem % multiply(kernel);

        for (int64_t i = dims - 1; i >= 0; --i) {
            int64_t k = kernel[i];
            offsets[c][i] = rem % k;
            rem /= k;
        }

        for (size_t i = 0; i < dims; ++i) {
            int64_t o = offsets[c][i];
            pad_starts[c][i] = std::max(int64_t(0), (padding[i] - o * dilation[i] + stride[i] - 1) / stride[i]);
            pad_ends[c][i] = std::min(cols[i], (size[i] + padding[i] - o * dilation[i] + stride[i] - 1) / stride[i]);
        }
    }

    threading::preferential_parallel_for(threading::block_ranges<1>(0, batch_size),
        [&](threading::blocked_range<1> block) {
            for (int64_t batch = block.begin[0]; batch < block.end[0]; ++batch) {
                auto data_col_ptr = data_col + batch * batch_col_add;
                auto data_im = data_im_ptr + batch * batch_im_add;

                for (int64_t c = 0; c < channels_col; ++c) {
                    const auto& offset = offsets[c];
                    const auto& pad_start = pad_starts[c];
                    const auto& pad_end = pad_ends[c];
                    int64_t col_add = multiply(cols) * c;
                    int64_t im_add = multiply(size) * channel_map[c];

                    std::vector<int64_t> indexes = pad_start;
                    run_inner_foldnd_backward_loop(0, col_add, im_add, indexes,
                                   pad_start, pad_end, offset,
                                   cols, img_strides, col_strides,
                                   size, kernel, stride, padding, dilation,
                                   data_im, data_col_ptr);
                }
            }
        });
};



void foldnd_backward_(ArrayVoid &output, ArrayVoid &grad, const int64_t& channels,
        const std::vector<int64_t>& size, const std::vector<int64_t>& kernels,
        const std::vector<int64_t>& strides, const std::vector<int64_t>& paddings,
        const std::vector<int64_t>& dilations, const int64_t& batch_size){
    grad.execute_function<WRAP_DTYPES<NumberTypesL> >(
       col2im_cpu_nd_backward, output, channels, size, kernels, strides, paddings, dilations,
                    batch_size);
}


} // namespace cpu
} // namespace functional
} // namespace nt
