#include "../../dtype/ArrayVoid_NTensor.hpp"
#include "../../mp/Threading.h"
// #include "colim_transform.h"

namespace nt {
namespace functional {
namespace cpu{


inline static constexpr auto im2col_cpu_1d = [](auto data_im_ptr, auto data_im_ptr_end, auto data_col,
    const int64_t& channels,
    const int64_t& width,
    const int64_t& k_w,
    const int64_t& s_w,
    const int64_t& p_w,
    const int64_t& d_w,
    const int64_t& batch_size) {
    
    const int64_t width_col = (width + 2 * p_w - d_w * (k_w - 1) - 1) / s_w + 1;
    const int64_t channels_col = channels * k_w;
    const int64_t col_matrix_size = batch_size * channels_col * width_col;

    const int64_t batch_im_add = channels * width;
    const int64_t batch_col_add = channels_col * width_col;


#ifdef USE_PARALLEL
    if constexpr (std::is_pointer_v<decltype(data_im_ptr)>){
	    threading::preferential_parallel_for(threading::block_ranges<2>(0, batch_size, 0, channels_col),
		[&](threading::blocked_range<2> block){
	for(int64_t batch = block.begin[0]; batch < block.end[0]; ++batch){
		auto data_col_ptr = data_col + batch * batch_col_add;
		auto data_im = data_im_ptr + batch * batch_im_add;
		for(int64_t c = block.begin[1]; c < block.end[1]; ++c){
			int64_t w_offset = c % k_w;
			int64_t c_im = c / k_w;
			// Calculate the valid range for w_pad
			int64_t w_pad_start = std::max(int64_t(0), (p_w - w_offset * d_w + s_w - 1) / s_w);
			int64_t w_pad_end = std::min(width_col, (width + p_w - w_offset * d_w + s_w - 1) / s_w);

			for (int64_t w = w_pad_start; w < w_pad_end; ++w) {
				int64_t w_pad = w * s_w - p_w + w_offset * d_w;
				data_col_ptr[c * width_col +  w] =
					data_im[c_im * width + w_pad];
			}
		}
	}
	});
      return;
    }
#endif
    nt::threading::iterator_parallel_for([&](auto range, auto data_im){ //works whether parallel or not, but obviously faster when parallel
	auto data_col_ptr = data_col + (batch_col_add * range.begin[0]);
	for (int64_t batch = range.begin[0]; batch < range.end[0]; ++batch, data_im += batch_im_add) {
		for (int64_t c = 0; c < channels_col; ++c) {
		int64_t w_offset = c % k_w;
		int64_t c_im = c / k_w;

                // Calculate the valid range for w_pad
                int64_t w_pad_start = std::max(int64_t(0), (p_w - w_offset * d_w + s_w - 1) / s_w);
                int64_t w_pad_end = std::min(width_col, (width + p_w - w_offset * d_w + s_w - 1) / s_w);
		
		//this is a really good check that negates a lot of edge case errors, and skips doing a lot of loops
		if(w_pad_start >= width_col || w_pad_end <= w_pad_start){
			data_col_ptr += width_col;
			continue;
		}
		
		data_col_ptr += w_pad_start;
		for(int64_t w = w_pad_start; w < w_pad_end; ++w){
			int64_t w_pad = w * s_w - p_w + w_offset * d_w;
			*data_col_ptr = data_im[c_im * width + w_pad];
			++data_col_ptr;
		}
		// Skip to the next valid row in data_col_ptr
		data_col_ptr += width_col - w_pad_end;
		
        }
    }


     }, 0, batch_size, data_im_ptr, batch_im_add);
};


void unfold1d_(ArrayVoid &im, ArrayVoid &col, const int64_t& channels,
    const int64_t& width,
    const int64_t& k_w,
    const int64_t& s_w,
    const int64_t& p_w,
    const int64_t& d_w,
    const int64_t& batch_size) {
    im.execute_function < WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool>> >(
                              im2col_cpu_1d, col, 
                                channels, width,
                                k_w, 
                                s_w, 
                                p_w, 
                                d_w, batch_size);
}


} // namespace cpu
} // namespace functional
} // namespace nt
