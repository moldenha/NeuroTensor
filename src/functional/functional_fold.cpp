#include "functional_fold.h"
#include "../Tensor.h"
#include "../dtype/ArrayVoid.hpp"
#include "../mp/Threading.h"


namespace nt{
namespace functional{

//apparently the old functional fold was just complete and utter shit
//so this actually not only works, but is also a lot faster
//so thats nice


inline static constexpr auto im2col_cpu = [](auto data_im_ptr, auto data_im_ptr_end, auto data_col,
		const int64_t& channels,
            const int64_t& height, const int64_t& width,
            const int64_t& k_r, const int64_t& k_c, 
            const int64_t& s_r, const int64_t& s_c, 
            const int64_t& p_r, const int64_t& p_c, 
            const int64_t& d_r, const int64_t& d_c,
	    const int64_t& batch_size) {
    const int64_t height_col = (height + 2 * p_r - d_r * (k_r - 1) - 1) / s_r + 1;
    const int64_t width_col = (width + 2 * p_c - d_c * (k_c - 1) - 1) / s_c + 1;
    const int64_t channels_col = channels * k_r * k_c;
    const int64_t col_matrix_size = batch_size * channels_col * height_col * width_col;
    const int64_t hw_col = height_col * width_col;

    /* auto data_col_ptr = data_col; */
    const int64_t batch_im_add = (channels * height * width);
    const int64_t batch_col_add = (channels_col * height_col * width_col);
#ifdef USE_PARALLEL
    if constexpr (std::is_pointer_v<decltype(data_im_ptr)>){
	    threading::preferential_parallel_for(threading::block_ranges<2>(0, batch_size, 0, channels_col),
		[&](threading::blocked_range<2> block){
	for(int64_t batch = block.begin[0]; batch < block.end[0]; ++batch){
		auto data_col_ptr = data_col + batch * batch_col_add;
		auto data_im = data_im_ptr + batch * batch_im_add;
		for(int64_t c = block.begin[1]; c < block.end[1]; ++c){
			int64_t w_offset = c % k_c;
			int64_t h_offset = (c / k_c) % k_r;
			int64_t c_im = c / k_r / k_c;
			// Calculate the valid range for h_pad
			int64_t h_pad_start = std::max(0LL, (p_r - h_offset * d_r + s_r - 1) / s_r);
			int64_t h_pad_end = std::min(height_col, (height + p_r - h_offset * d_r + s_r - 1) / s_r);

			// Calculate the valid range for w_pad
			int64_t w_pad_start = std::max(0LL, (p_c - w_offset * d_c + s_c - 1) / s_c);
			int64_t w_pad_end = std::min(width_col, (width + p_c - w_offset * d_c + s_c - 1) / s_c);

			for (int64_t h = h_pad_start; h < h_pad_end; ++h) {
				int64_t h_pad = h * s_r - p_r + h_offset * d_r;
				for (int64_t w = w_pad_start; w < w_pad_end; ++w) {
					int64_t w_pad = w * s_c - p_c + w_offset * d_c;
					data_col_ptr[c * height_col * width_col + h * width_col + w] =
						data_im[c_im * height * width + h_pad * width + w_pad];
					/* data_col[(batch * channels_col + c) * height_col * width_col */
					/* 		+ h * width_col + w] = */
					/* 	data_im[batch * channels * height * width */ 
					/* 		+ c_im * height * width + h_pad * width + w_pad]; */
				}
			}
		}
	}
	});
	/* tbb::parallel_for(tbb::blocked_range3d<int64_t>(0, batch_size, 0, channels_col, 0, height_col), */
        /* [&](const tbb::blocked_range3d<int64_t>& r) { */
        /* for (int64_t batch = r.pages().begin(); batch < r.pages().end(); ++batch) { */
            /* auto data_col_ptr = data_col + batch * batch_col_add; */
            /* auto data_im = data_im_ptr + batch * batch_im_add; */

            /* for (int64_t c = r.rows().begin(); c < r.rows().end(); ++c) { */
                /* int64_t w_offset = c % k_c; */
                /* int64_t h_offset = (c / k_c) % k_r; */
                /* int64_t c_im = c / k_r / k_c; */

                /* for (int64_t h = r.cols().begin(); h < r.cols().end(); ++h) { */
                    /* for (int64_t w = 0; w < width_col; ++w) { */
                        /* int64_t h_pad = h * s_r - p_r + h_offset * d_r; */
                        /* int64_t w_pad = w * s_c - p_c + w_offset * d_c; */

                        /* if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) { */
                            /* data_col_ptr[c * height_col * width_col + h * width_col + w] = */
                                /* data_im[c_im * height * width + h_pad * width + w_pad]; */
                        /* } */ 
                    /* } */
                /* } */
            /* } */
        /* } */
      /* }); */
      return;
    }
#endif
    nt::threading::iterator_parallel_for([&](auto range, auto data_im){ //works whether parallel or not, but obviously faster when parallel
	auto data_col_ptr = data_col + (batch_col_add * range.begin[0]);
	for (int64_t batch = range.begin[0]; batch < range.end[0]; ++batch, data_im += batch_im_add) {
		for (int64_t c = 0; c < channels_col; ++c) {
		int64_t w_offset = c % k_c;
		int64_t h_offset = (c / k_c) % k_r;
		int64_t c_im = c / k_r / k_c;
		// Calculate the valid range for h_pad
                int64_t h_pad_start = std::max(0LL, (p_r - h_offset * d_r + s_r - 1) / s_r);
                int64_t h_pad_end = std::min(height_col, (height + p_r - h_offset * d_r + s_r - 1) / s_r);

                // Calculate the valid range for w_pad
                int64_t w_pad_start = std::max(0LL, (p_c - w_offset * d_c + s_c - 1) / s_c);
                int64_t w_pad_end = std::min(width_col, (width + p_c - w_offset * d_c + s_c - 1) / s_c);
		
		//this is a really good check that negates a lot of edge case errors, and skips doing a lot of loops
		if(h_pad_start >= height_col || h_pad_end <= h_pad_start 
				|| w_pad_start >= width_col || w_pad_end <= w_pad_start){
			data_col_ptr += hw_col;
			continue;
		}
		data_col_ptr += (h_pad_start * width_col);
		for(int64_t h = h_pad_start; h < h_pad_end; ++h){
			data_col_ptr += w_pad_start;
			int64_t h_pad = h * s_r - p_r + h_offset * d_r;			
			for(int64_t w = w_pad_start; w < w_pad_end; ++w){
				int64_t w_pad = w * s_c - p_c + w_offset * d_c;
				*data_col_ptr = data_im[c_im * height * width + h_pad * width + w_pad];
				++data_col_ptr;
			}
			// Skip to the next valid row in data_col_ptr
			data_col_ptr += width_col - w_pad_end;
		}
		// Adjust data_col_ptr for the skipped rows and columns
                data_col_ptr += (height_col - h_pad_end) * width_col;
        }
    }


     }, 0, batch_size, data_im_ptr, batch_im_add);

};




inline static constexpr auto im2col_cpu_transposed = [](auto data_im_ptr, auto data_im_ptr_end, auto data_col,
        const int64_t& channels,
        const int64_t& height, const int64_t& width,
        const int64_t& k_r, const int64_t& k_c, 
        const int64_t& s_r, const int64_t& s_c, 
        const int64_t& p_r, const int64_t& p_c, 
        const int64_t& d_r, const int64_t& d_c,
        const int64_t& batch_size) {
    const int64_t height_col = (height + 2 * p_r - d_r * (k_r - 1) - 1) / s_r + 1;
    const int64_t width_col = (width + 2 * p_c - d_c * (k_c - 1) - 1) / s_c + 1;
    const int64_t channels_col = channels * k_r * k_c;
    const int64_t col_matrix_size = batch_size * height_col * width_col * channels_col;

    const int64_t batch_im_add = (channels * height * width);
    const int64_t batch_col_add = (height_col * width_col * channels_col);

    tbb::parallel_for(tbb::blocked_range3d<int64_t>(0, batch_size, 0, height_col, 0, width_col),
    [&](const tbb::blocked_range3d<int64_t>& r) {
        for (int64_t batch = r.pages().begin(); batch < r.pages().end(); ++batch) {
            auto data_col_ptr = data_col + batch * batch_col_add;
            auto data_im = data_im_ptr + batch * batch_im_add;

            for (int64_t h = r.rows().begin(); h < r.rows().end(); ++h) {
                for (int64_t w = r.cols().begin(); w < r.cols().end(); ++w) {
                    for (int64_t c = 0; c < channels_col; ++c) {
                        int64_t w_offset = c % k_c;
                        int64_t h_offset = (c / k_c) % k_r;
                        int64_t c_im = c / k_r / k_c;

                        int64_t h_pad = h * s_r - p_r + h_offset * d_r;
                        int64_t w_pad = w * s_c - p_c + w_offset * d_c;

                        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
                            data_col_ptr[c * height_col * width_col + h * width_col + w] =
                                data_im[c_im * height * width + h_pad * width + w_pad];
                        } 
                    }
                }
            }
        }
    });
};


Tensor unfold(const Tensor& x, utils::my_tuple kernel_size, utils::my_tuple dilation, utils::my_tuple padding, utils::my_tuple stride, bool transpose_out){
	utils::throw_exception(x.dims() >= 3, "Expected input tensot to unfold to have dims greater than or equal to 3 but got $D", x.dims());

	const int64_t LKern = kernel_size[0] * kernel_size[1];

	const int64_t L_r = ((x.shape()[-2] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
	const int64_t L_c = ((x.shape()[-1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
	const int64_t L = L_r * L_c;

	utils::throw_exception(L > 0, "Given input with spatial size ($, $), kernel_size=($, $), dilation=($, $), padding=($, $), calculated shape of the array of sliding blocks as ($, $), but its components must be at least one.", x.shape()[-2], x.shape()[-1], kernel_size[0], kernel_size[1], dilation[0], dilation[1], padding[0], padding[1], L_r, L_c);
	Tensor im = (x.dims() == 3) ? x.view(1, x.shape()[0], x.shape()[1], x.shape()[2]) : x.flatten(0, -4);
	const int64_t& channels = im.shape()[1];

	Tensor col = zeros({im.shape()[0], im.shape()[1] * LKern, L});
	if(transpose_out){
		im.arr_void().execute_function(im2col_cpu, col.arr_void(), channels,
				x.shape()[-2], x.shape()[-1],
				kernel_size[0], kernel_size[1],
				stride[0], stride[1],
				padding[0], padding[1],
				dilation[0], dilation[1],
				im.shape()[0]);
		return std::move(col);
	}
	im.arr_void().execute_function(im2col_cpu_transposed, col.arr_void(), channels,
				x.shape()[-2], x.shape()[-1],
				kernel_size[0], kernel_size[1],
				stride[0], stride[1],
				padding[0], padding[1],
				dilation[0], dilation[1],
				im.shape()[0]);

	return col.view(col.shape().transpose(-1,-2));
}

inline static constexpr auto im2col_backward_cpu = [](auto data_im_ptr, auto data_im_ptr_end, auto data_col,
		const int64_t& channels,
            const int64_t& height, const int64_t& width,
            const int64_t& k_r, const int64_t& k_c, 
            const int64_t& s_r, const int64_t& s_c, 
            const int64_t& p_r, const int64_t& p_c, 
            const int64_t& d_r, const int64_t& d_c,
	    const int64_t& batch_size) {
    const int64_t height_col = (height + 2 * p_r - d_r * (k_r - 1) - 1) / s_r + 1;
    const int64_t width_col = (width + 2 * p_c - d_c * (k_c - 1) - 1) / s_c + 1;
    const int64_t channels_col = channels * k_r * k_c;
    const int64_t col_matrix_size = batch_size * channels_col * height_col * width_col;

    /* auto data_col_ptr = data_col; */
    const int64_t batch_im_add = (channels * height * width);
    const int64_t batch_col_add = (channels_col * height_col * width_col);
    //TODO: once the error at the todo below is fixed, delete everything below this to there
    nt::threading::iterator_parallel_for([&](auto range, auto data_im){ //works whether parallel or not, but obviously faster when parallel
     auto data_col_ptr = data_col + (batch_col_add * range.begin[0]);
     for (int64_t batch = range.begin[0]; batch < range.end[0]; ++batch, data_im += batch_im_add) {

        for (int64_t c = 0; c < channels_col; ++c) {
            int64_t w_offset = c % k_c;
            int64_t h_offset = (c / k_c) % k_r;
            int64_t c_im = c / k_r / k_c;
            for (int64_t h = 0; h < height_col; ++h) {
                for (int64_t w = 0; w < width_col; ++w) {
                    int64_t h_pad = h * s_r - p_r + h_offset * d_r;
                    int64_t w_pad = w * s_c - p_c + w_offset * d_c;
                    if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
                        data_im[c_im * height * width + 
                                                h_pad * width + 
                                                w_pad] += *data_col_ptr;
                    } 
		    ++data_col_ptr;
                }
            }
        }
    }


     }, 0, batch_size, data_im_ptr, batch_im_add);

    //TODO: why the fuck is this causing an error with calling this function in the first place:
    /* const int64_t hw_col = height_col * width_col; */


    /* /1* auto data_col_ptr = data_col; *1/ */
    /* const int64_t batch_im_add = (channels * height * width); */
    /* const int64_t batch_col_add = (channels_col * height_col * width_col); */
/* #ifdef USE_PARALLEL */
    /* if constexpr (std::is_pointer_v<decltype(data_im_ptr)>){ */
	    /* threading::preferential_parallel_for(threading::block_ranges<2>(0, batch_size, 0, channels_col), */
		/* [&](threading::blocked_range<2> block){ */
	/* for(int64_t batch = block.begin[0]; batch < block.end[0]; ++batch){ */
		/* auto data_col_ptr = data_col + batch * batch_col_add; */
		/* auto data_im = data_im_ptr + batch * batch_im_add; */
		/* for(int64_t c = block.begin[1]; c < block.end[1]; ++c){ */
			/* int64_t w_offset = c % k_c; */
			/* int64_t h_offset = (c / k_c) % k_r; */
			/* int64_t c_im = c / k_r / k_c; */
			/* // Calculate the valid range for h_pad */
			/* int64_t h_pad_start = std::max(0LL, (p_r - h_offset * d_r + s_r - 1) / s_r); */
			/* int64_t h_pad_end = std::min(height_col, (height + p_r - h_offset * d_r + s_r - 1) / s_r); */

			/* // Calculate the valid range for w_pad */
			/* int64_t w_pad_start = std::max(0LL, (p_c - w_offset * d_c + s_c - 1) / s_c); */
			/* int64_t w_pad_end = std::min(width_col, (width + p_c - w_offset * d_c + s_c - 1) / s_c); */

			/* for (int64_t h = h_pad_start; h < h_pad_end; ++h) { */
				/* int64_t h_pad = h * s_r - p_r + h_offset * d_r; */
				/* for (int64_t w = w_pad_start; w < w_pad_end; ++w) { */
					/* int64_t w_pad = w * s_c - p_c + w_offset * d_c; */
					/* data_im[c_im * height * width + h_pad * width + w_pad] */
						/* += data_col_ptr[c * height_col * width_col + h * width_col + w]; */
					/* /1* data_col_ptr[c * height_col * width_col + h * width_col + w] *1/ */
				/* } */
			/* } */
		/* } */
	/* } */
	/* }); */
    /*   return; */
    /* } */
/* #endif */
    /* nt::threading::iterator_parallel_for([&](auto range, auto data_im){ //works whether parallel or not, but obviously faster when parallel */
	/* auto data_col_ptr = data_col + (batch_col_add * range.begin[0]); */
	/* for (int64_t batch = range.begin[0]; batch < range.end[0]; ++batch, data_im += batch_im_add) { */
		/* for (int64_t c = 0; c < channels_col; ++c) { */
		/* int64_t w_offset = c % k_c; */
		/* int64_t h_offset = (c / k_c) % k_r; */
		/* int64_t c_im = c / k_r / k_c; */
		/* // Calculate the valid range for h_pad */
    /*             int64_t h_pad_start = std::max(0LL, (p_r - h_offset * d_r + s_r - 1) / s_r); */
    /*             int64_t h_pad_end = std::min(height_col, (height + p_r - h_offset * d_r + s_r - 1) / s_r); */

    /*             // Calculate the valid range for w_pad */
    /*             int64_t w_pad_start = std::max(0LL, (p_c - w_offset * d_c + s_c - 1) / s_c); */
    /*             int64_t w_pad_end = std::min(width_col, (width + p_c - w_offset * d_c + s_c - 1) / s_c); */
		
		/* //this is a really good check that negates a lot of edge case errors, and skips doing a lot of loops */
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
				/* data_im[c_im * height * width + h_pad * width + w_pad] += *data_col_ptr; */
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


Tensor unfold_backward(const Tensor& x, const utils::my_tuple& output_size, const utils::my_tuple& kernel_size, const utils::my_tuple& dilation, const utils::my_tuple& padding, const utils::my_tuple& stride, const bool& transpose_out){
	utils::throw_exception(x.dims() == 2 || x.dims() == 3, "Expected to get a shape with a dimensionality of 3D or 2D, but got $ dimensions", x.dims());
	const int64_t& BROWS = output_size[0];
	const int64_t& BCOLS = output_size[1];
	const int64_t L_r = ((output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
	const int64_t L_c = ((output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
	const int64_t L = L_r * L_c;
	const int64_t LKern = kernel_size[0] * kernel_size[1];
	const int64_t batches = x.dims() == 3 ? x.shape()[0] : 1;
	utils::throw_exception((L_r*L_c) > 0, "Given input with spatial size ($, $), kernel_size=($, $), dilation=($, $), padding=($, $), calculated shape of the array of sliding blocks as ($, $), but its components must be at least one.", output_size[0], output_size[1], kernel_size[0], kernel_size[1], dilation[0], dilation[1], padding[0], padding[1], L_r, L_c);
	Tensor Z = (x.dims() == 2) ? x.unsqueeze(0) : x;
	if(!transpose_out)
		Z = Z.contiguous().RowColSwap();
	utils::throw_exception(Z.shape()[1] % LKern == 0, "Given kernel size of $, $ * $ = $ must be able to divide input at dimension $, but got input.shape()[$] = $", kernel_size, kernel_size[0], kernel_size[1], LKern, (x.dims() == 3) ? 1 : 0, (x.dims() == 3) ? 1 : 0, Z.shape()[1]);
	const int64_t channels = Z.shape()[1] / LKern;
	SizeRef output_shape({batches, channels, BROWS, BCOLS});
	utils::throw_exception(Z.shape()[2] == L, "Given output_size=($, $), kernel_size=($, $), dilation=($, $), padding=($, $), stride=($, $), expected size of input's dimension 2 to match the calculated number of sliding blocks $ * $ = $, but got input.size(2)=$.", output_size[0], output_size[1], kernel_size[0], kernel_size[1], dilation[0], dilation[1], padding[0], padding[1], stride[0], stride[1], L_r, L_c, L, Z.shape()[2]);
	Tensor output = zeros(output_shape, x.dtype);
	output.arr_void().execute_function_nbool(im2col_backward_cpu, 
			Z.arr_void(), channels, BROWS, BCOLS,
			kernel_size[0], kernel_size[1],
			stride[0], stride[1],
			padding[0], padding[1],
			dilation[0], dilation[1],
			batches);
	if(!transpose_out && x.is_contiguous())
		Z.RowColSwap();
	return std::move(output);
}

Tensor& unfold_backward(const Tensor& x, Tensor& output, const utils::my_tuple& output_size, const utils::my_tuple& kernel_size, const utils::my_tuple& dilation, const utils::my_tuple& padding, const utils::my_tuple& stride, const bool& transpose_out){
	utils::throw_exception(x.dims() == 2 || x.dims() == 3, "Expected to get a shape with a dimensionality of 3D or 2D, but got $ dimensions", x.dims());
	const int64_t& BROWS = output_size[0];
	const int64_t& BCOLS = output_size[1];
	const int64_t L_r = ((output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
	const int64_t L_c = ((output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
	const int64_t L = L_r * L_c;
	const int64_t LKern = kernel_size[0] * kernel_size[1];
	const int64_t batches = x.dims() == 3 ? x.shape()[0] : 1;
	utils::throw_exception((L_r*L_c) > 0, "Given input with spatial size ($, $), kernel_size=($, $), dilation=($, $), padding=($, $), calculated shape of the array of sliding blocks as ($, $), but its components must be at least one.", output_size[0], output_size[1], kernel_size[0], kernel_size[1], dilation[0], dilation[1], padding[0], padding[1], L_r, L_c);
	Tensor Z = (x.dims() == 2) ? x.unsqueeze(0) : x;
	if(!transpose_out)
		Z = Z.contiguous().RowColSwap();
	utils::throw_exception(Z.shape()[1] % LKern == 0, "Given kernel size of $, $ * $ = $ must be able to divide input at dimension $, but got input.shape()[$] = $", kernel_size, kernel_size[0], kernel_size[1], LKern, (x.dims() == 3) ? 1 : 0, (x.dims() == 3) ? 1 : 0, Z.shape()[1]);
	const int64_t channels = Z.shape()[1] / LKern;
	SizeRef output_shape({batches, channels, BROWS, BCOLS});
	utils::throw_exception(Z.shape()[2] == L, "Given output_size=($, $), kernel_size=($, $), dilation=($, $), padding=($, $), stride=($, $), expected size of input's dimension 2 to match the calculated number of sliding blocks $ * $ = $, but got input.size(2)=$.", output_size[0], output_size[1], kernel_size[0], kernel_size[1], dilation[0], dilation[1], padding[0], padding[1], stride[0], stride[1], L_r, L_c, L, Z.shape()[2]);
	utils::throw_exception(output.dtype == x.dtype, "Expected dtypes of grad and output to match for fold backward but got $ for output and $ for the grad", output.dtype, x.dtype);
	utils::throw_exception(output.shape().multiply() == output_shape.multiply(), "Expected to get same shape for output for unfold backward of $ but got $", output_shape, output.shape());
	output.arr_void().execute_function_nbool(im2col_backward_cpu, 
			Z.arr_void(), channels, BROWS, BCOLS,
			kernel_size[0], kernel_size[1],
			stride[0], stride[1],
			padding[0], padding[1],
			dilation[0], dilation[1],
			batches);
	if(!transpose_out && x.is_contiguous())
		Z.RowColSwap();
	return output;
}


inline static constexpr auto col2im_cpu = [](auto a_begin, auto a_end, auto o_begin,
		const int64_t& col_add, const int64_t& L_r, const int64_t& L_c,
		const int64_t& o_r, const int64_t& o_c,
		const int64_t& k_r, const int64_t k_c,
		const int64_t& s_r, const int64_t& s_c,
		const int64_t& d_r, const int64_t& d_c,
		const int64_t& p_r, const int64_t p_c,
		const int64_t& batches, const int64_t& channels){
	const int64_t& row_upper_limit = k_c;
	const int64_t& col_upper_limit = k_r;
	const int64_t& kernel_row_upper_limit = L_c;
	const int64_t& kernel_col_upper_limit = L_r;
	const int64_t o_channel_add = o_r * o_c;
	const int64_t a_channel_add = k_r * k_c * col_add;
	for(int64_t n = 0; n < batches; ++n){
		for(int64_t ch = 0; ch < channels; ++ch, o_begin += o_channel_add, a_begin += a_channel_add){
			for(int64_t r = 0; r < row_upper_limit; ++r){
				for(int64_t c = 0; c < col_upper_limit; ++c){
					int64_t o_row = (r * s_r) + (c * d_r) - p_r;
					if(o_row >= o_r || o_row < 0){continue;}
					int64_t a_col_index = r * L_c;
					int64_t a_row_index = c * L_r;
					auto o_cpy = o_begin + (o_row * o_c);
					auto a_cpy = a_begin + (a_row_index * col_add + a_col_index);
					for(int64_t kr = 0; kr < kernel_row_upper_limit; ++kr){
						for(int64_t kc = 0; kc < kernel_col_upper_limit; ++kc){
							int64_t o_col = (kc * d_c) + (kr * s_c) - p_c;
							if(o_col < o_c && o_col >= 0){
								o_cpy[o_col] += a_cpy[col_add * kc + kr];
							}
						}
					}
				}
			}
		}
	}

};


//basically just one line to switch:
inline static constexpr auto col2im_cpu_backward = [](auto a_begin, auto a_end, auto o_begin,
		const int64_t& col_add, const int64_t& L_r, const int64_t& L_c,
		const int64_t& o_r, const int64_t& o_c,
		const int64_t& k_r, const int64_t k_c,
		const int64_t& s_r, const int64_t& s_c,
		const int64_t& d_r, const int64_t& d_c,
		const int64_t& p_r, const int64_t p_c,
		const int64_t& batches, const int64_t& channels){
	const int64_t& row_upper_limit = k_c;
	const int64_t& col_upper_limit = k_r;
	const int64_t& kernel_row_upper_limit = L_c;
	const int64_t& kernel_col_upper_limit = L_r;
	const int64_t o_channel_add = o_r * o_c;
	const int64_t a_channel_add = k_r * k_c * col_add;
	for(int64_t n = 0; n < batches; ++n){
		for(int64_t ch = 0; ch < channels; ++ch, o_begin += o_channel_add, a_begin += a_channel_add){
			for(int64_t r = 0; r < row_upper_limit; ++r){
				for(int64_t c = 0; c < col_upper_limit; ++c){
					int64_t o_row = (r * s_r) + (c * d_r) - p_r;
					if(o_row >= o_r || o_row < 0){continue;}
					int64_t a_col_index = r * L_c;
					int64_t a_row_index = c * L_r;
					auto o_cpy = o_begin + (o_row * o_c);
					auto a_cpy = a_begin + (a_row_index * col_add + a_col_index);
					for(int64_t kr = 0; kr < kernel_row_upper_limit; ++kr){
						for(int64_t kc = 0; kc < kernel_col_upper_limit; ++kc){
							int64_t o_col = (kc * d_c) + (kr * s_c) - p_c;
							if(o_col < o_c && o_col >= 0){
								a_cpy[col_add * kc + kr] += o_cpy[o_col]; //switched line
							}
						}
					}
				}
			}
		}
	}

};



Tensor fold(const Tensor& x, utils::my_tuple output_size, utils::my_tuple kernel_size, utils::my_tuple dilation, utils::my_tuple padding, utils::my_tuple stride){
    utils::throw_exception(x.dims() == 2 || x.dims() == 3, "Expected to get a shape with a dimensionality of 3D or 2D, but got $D", x.dims());
    const int64_t& BROWS = output_size[0];
    const int64_t& BCOLS = output_size[1];
    const int64_t L_r = ((BROWS + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
    const int64_t L_c = ((BCOLS + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
    const int64_t L = L_r * L_c;
    const int64_t LKern = kernel_size[0] * kernel_size[1];
    const int64_t batches = (x.dims() == 3) ? x.shape()[0] : 1;
    Tensor Z = (x.dims() == 2) ? x.unsqueeze(0) : x;


    utils::throw_exception(Z.shape()[1] % LKern == 0, "Given kernel size of $, $ * $ = $ must be able to divide input at dimension $, but got input.shape()[$] = $", kernel_size, kernel_size[0], kernel_size[1], LKern, (x.dims() == 3) ? 1 : 0, (x.dims() == 3) ? 1 : 0, Z.shape()[1]);
    utils::throw_exception(Z.shape()[2] == L, "Given output_size = $, kernel_size = $, dilation = $, padding = $, stride = $, expected size of input's dimension $ to match the calculated number of sliding blocks $ * $ = $ but got input.shape()[$] = $",
			output_size, kernel_size, dilation, padding, stride, (x.dims() == 3) ? 2 : 1, L_r, L_c, L, (x.dims() == 3) ? 2 : 1, Z.shape()[2]);
    
    const int64_t channels = Z.shape()[1] / LKern;
    std::vector<int64_t> output_shape = {batches, channels, BROWS, BCOLS};
    Tensor output = zeros(output_shape, x.dtype);


    Z.arr_void().execute_function_nbool(col2im_cpu, output.arr_void(),
		    L, L_r, L_c,
		    BROWS, BCOLS,
		    kernel_size[0], kernel_size[1],
		    stride[0], stride[1],
		    dilation[0], dilation[1],
		    padding[0], padding[1],
		    batches, channels);
    return std::move(output);
}



Tensor fold_backward(const Tensor& grad_output, const utils::my_tuple& output_size, const utils::my_tuple& kernel_size, const utils::my_tuple& dilation, const utils::my_tuple& padding, const utils::my_tuple& stride){
    utils::throw_exception(grad_output.dims() == 4 || grad_output.dims() == 3, "Expected to get a shape with a dimensionality of 4D or 3D, but got $D for fold backward", grad_output.dims());
    const int64_t& BROWS = output_size[0];
    const int64_t& BCOLS = output_size[1];
    const int64_t L_r = ((BROWS + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
    const int64_t L_c = ((BCOLS + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
    const int64_t L = L_r * L_c;
    const int64_t LKern = kernel_size[0] * kernel_size[1];
    const int64_t batches = (grad_output.dims() == 4) ? grad_output.shape()[0] : 1;
    const int64_t channels = grad_output.shape()[-3];
    Tensor Z = (grad_output.dims() == 3) ? grad_output.unsqueeze(0) : grad_output;
    utils::throw_exception(Z.shape()[-1] == BROWS && Z.shape()[-2] == BCOLS, "Expected last dimensions of grad output for fold backward to match output size $ but got ($,$)", output_size, Z.shape()[-2], Z.shape()[-1]);

    /* std::vector<int64_t> output_shape = {batches, channels, BROWS, BCOLS}; */
    SizeRef output_shape({batches, channels * LKern, L});
    Tensor output = zeros(output_shape, grad_output.dtype);


    output.arr_void().execute_function_nbool(col2im_cpu_backward, Z.arr_void(),
		    L, L_r, L_c,
		    BROWS, BCOLS,
		    kernel_size[0], kernel_size[1],
		    stride[0], stride[1],
		    dilation[0], dilation[1],
		    padding[0], padding[1],
		    batches, channels);
    return std::move(output);
}

Tensor& fold_backward(const Tensor& grad_output, Tensor& output, const utils::my_tuple& output_size, const utils::my_tuple& kernel_size, const utils::my_tuple& dilation, const utils::my_tuple& padding, const utils::my_tuple& stride){
    utils::throw_exception(grad_output.dims() == 4 || grad_output.dims() == 3, "Expected to get a shape with a dimensionality of 4D or 3D, but got $D for fold backward", grad_output.dims());
    const int64_t& BROWS = output_size[0];
    const int64_t& BCOLS = output_size[1];
    const int64_t L_r = ((BROWS + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
    const int64_t L_c = ((BCOLS + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
    const int64_t L = L_r * L_c;
    const int64_t LKern = kernel_size[0] * kernel_size[1];
    const int64_t batches = (grad_output.dims() == 4) ? grad_output.shape()[0] : 1;
    const int64_t channels = grad_output.shape()[-3];
    Tensor Z = (grad_output.dims() == 3) ? grad_output.unsqueeze(0) : grad_output;
    utils::throw_exception(Z.shape()[-1] == BROWS && Z.shape()[-2] == BCOLS, "Expected last dimensions of grad output for fold backward to match output size $ but got ($,$)", output_size, Z.shape()[-2], Z.shape()[-1]);

    /* std::vector<int64_t> output_shape = {batches, channels, BROWS, BCOLS}; */
    SizeRef output_shape({batches, channels * LKern, L});
    //make sure the amount of elements are the same
    utils::throw_exception(output_shape.multiply() == output.shape().multiply(), "Expected output shape to match $ but got $ for fold backward", output_shape, output.shape());
    utils::throw_exception(output.dtype == grad_output.dtype, "Expected dtypes of grad and output to match for fold backward but got $ for output and $ for the grad", output.dtype, grad_output.dtype);


    output.arr_void().execute_function_nbool(col2im_cpu_backward, Z.arr_void(),
		    L, L_r, L_c,
		    BROWS, BCOLS,
		    kernel_size[0], kernel_size[1],
		    stride[0], stride[1],
		    dilation[0], dilation[1],
		    padding[0], padding[1],
		    batches, channels);
    return output;
}


inline static constexpr auto im2col_cpu_3d = [](auto data_im_ptr, auto data_im_ptr_end, auto data_col,
    const int64_t& channels,
    const int64_t& depth, const int64_t& height, const int64_t& width,
    const int64_t& k_d, const int64_t& k_r, const int64_t& k_c, 
    const int64_t& s_d, const int64_t& s_r, const int64_t& s_c, 
    const int64_t& p_d, const int64_t& p_r, const int64_t& p_c, 
    const int64_t& d_d, const int64_t& d_r, const int64_t& d_c,
    const int64_t& batch_size) {
    
    const int64_t depth_col = (depth + 2 * p_d - d_d * (k_d - 1) - 1) / s_d + 1;
    const int64_t height_col = (height + 2 * p_r - d_r * (k_r - 1) - 1) / s_r + 1;
    const int64_t width_col = (width + 2 * p_c - d_c * (k_c - 1) - 1) / s_c + 1;
    const int64_t channels_col = channels * k_d * k_r * k_c;
    const int64_t col_matrix_size = batch_size * channels_col * depth_col * height_col * width_col;

    const int64_t batch_im_add = channels * depth * height * width;
    const int64_t batch_col_add = channels_col * depth_col * height_col * width_col;
    const int64_t dhw_col = depth_col * height_col * width_col;
    const int64_t hw_col =  height_col * width_col;

#ifdef USE_PARALLEL
    if constexpr (std::is_pointer_v<decltype(data_im_ptr)>){
	    threading::preferential_parallel_for(threading::block_ranges<2>(0, batch_size, 0, channels_col),
		[&](threading::blocked_range<2> block){
	for(int64_t batch = block.begin[0]; batch < block.end[0]; ++batch){
		auto data_col_ptr = data_col + batch * batch_col_add;
		auto data_im = data_im_ptr + batch * batch_im_add;
		for(int64_t c = block.begin[1]; c < block.end[1]; ++c){
			int64_t w_offset = c % k_c;
			int64_t h_offset = (c / k_c) % k_r;
			int64_t d_offset = (c / k_c / k_r) % k_d;
			int64_t c_im = c / k_d / k_r / k_c;
			// Calculate the valid range for h_pad
			int64_t h_pad_start = std::max(0LL, (p_r - h_offset * d_r + s_r - 1) / s_r);
			int64_t h_pad_end = std::min(height_col, (height + p_r - h_offset * d_r + s_r - 1) / s_r);

			// Calculate the valid range for w_pad
			int64_t w_pad_start = std::max(0LL, (p_c - w_offset * d_c + s_c - 1) / s_c);
			int64_t w_pad_end = std::min(width_col, (width + p_c - w_offset * d_c + s_c - 1) / s_c);

			// Calculate the valid range for d_pad
			int64_t d_pad_start = std::max(0LL, (p_d - d_offset * d_d + s_d - 1) / s_d);
			int64_t d_pad_end = std::min(depth_col, (depth + p_d - d_offset * d_d + s_d - 1) / s_d);
			for(int64_t d = d_pad_start; d < d_pad_end; ++d){
				int64_t d_pad = d * s_d - p_d + d_offset * d_d;
				for (int64_t h = h_pad_start; h < h_pad_end; ++h) {
					int64_t h_pad = h * s_r - p_r + h_offset * d_r;
					for (int64_t w = w_pad_start; w < w_pad_end; ++w) {
						int64_t w_pad = w * s_c - p_c + w_offset * d_c;
						data_col_ptr[c * height_col * width_col * depth_col
							+ d * height_col * width_col +
							h * width_col + w] = 
								data_im[c_im * depth * height * width +
								d_pad * height * width +
								h_pad * width 
								+ w_pad];

					}
				}
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
		int64_t w_offset = c % k_c;
		int64_t h_offset = (c / k_c) % k_r;
		int64_t d_offset = (c / k_c / k_r) % k_d;
		int64_t c_im = c / k_d / k_r / k_c;	
		// Calculate the valid range for h_pad
                int64_t h_pad_start = std::max(0LL, (p_r - h_offset * d_r + s_r - 1) / s_r);
                int64_t h_pad_end = std::min(height_col, (height + p_r - h_offset * d_r + s_r - 1) / s_r);

                // Calculate the valid range for w_pad
                int64_t w_pad_start = std::max(0LL, (p_c - w_offset * d_c + s_c - 1) / s_c);
                int64_t w_pad_end = std::min(width_col, (width + p_c - w_offset * d_c + s_c - 1) / s_c);
		
		// Calculate the valid range for d_pad
		int64_t d_pad_start = std::max(0LL, (p_d - d_offset * d_d + s_d - 1) / s_d);
		int64_t d_pad_end = std::min(depth_col, (depth + p_d - d_offset * d_d + s_d - 1) / s_d);
		//this is a really good check that negates a lot of edge case errors, and skips doing a lot of loops
		if(h_pad_start >= height_col || h_pad_end <= h_pad_start 
				|| w_pad_start >= width_col || w_pad_end <= w_pad_start
				|| d_pad_start >= depth_col || d_pad_end <= d_pad_start){
			data_col_ptr += dhw_col;
			continue;
		}
		//account fo d_pad_start
		data_col_ptr += (d_pad_start * hw_col);
		for(int64_t d = d_pad_start; d < d_pad_end; ++d){
			//account for h_pad_start
			data_col_ptr += h_pad_start * width_col;
			int64_t d_pad = d * s_d - p_d + d_offset * d_d;
			for(int64_t h = h_pad_start; h < h_pad_end; ++h){
				data_col_ptr += w_pad_start;
				int64_t h_pad = h * s_r - p_r + h_offset * d_r;
				for(int64_t w = w_pad_start; w < w_pad_end; ++w){
					int64_t w_pad = w * s_c - p_c + w_offset * d_c;
					*data_col_ptr = data_im[c_im * height * width * depth
							+ d_pad * height * width
							+ h_pad * width 
							+ w_pad];
					++data_col_ptr;
				}
				// Skip to the next valid row in data_col_ptr
				data_col_ptr += width_col - w_pad_end;
			}
			// Adjust data_col_ptr for the skipped rows and columns
			data_col_ptr += (height_col - h_pad_end) * width_col;
		}
		//Adjust data_col_ptr for the skipped rows, depth, and columns
		data_col_ptr += (depth_col - d_pad_end) * hw_col;
        }
    }


     }, 0, batch_size, data_im_ptr, batch_im_add);
};



Tensor unfold3d(const Tensor& x, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> dilation, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> stride, bool transpose_out) {
    utils::throw_exception(x.dims() >= 4, "Expected input tensor to unfold to have dims greater than or equal to 4 but got $D", x.dims());

    const int64_t LKern = kernel_size[0] * kernel_size[1] * kernel_size[2];

    const int64_t L_d = ((x.shape()[-3] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
    const int64_t L_r = ((x.shape()[-2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
    const int64_t L_c = ((x.shape()[-1] + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) / stride[2]) + 1;
    const int64_t L = L_d * L_r * L_c;

    utils::throw_exception(L > 0, "Given input with spatial size ($, $, $), kernel_size=($, $, $), dilation=($, $, $), padding=($, $, $), calculated shape of the array of sliding blocks as ($, $, $), but its components must be at least one.",
                           x.shape()[-3], x.shape()[-2], x.shape()[-1], 
                           kernel_size[0], kernel_size[1], kernel_size[2], 
                           dilation[0], dilation[1], dilation[2], 
                           padding[0], padding[1], padding[2], 
                           L_d, L_r, L_c);

    Tensor im = (x.dims() == 4) ? x.view(1, x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]) : x.flatten(0, -5);

    const int64_t& channels = im.shape()[1];
    Tensor col = zeros({im.shape()[0], im.shape()[1] * LKern, L});
    im.arr_void().execute_function(im2col_cpu_3d, col.arr_void(), channels,
                                   x.shape()[-3], x.shape()[-2], x.shape()[-1],
                                   kernel_size[0], kernel_size[1], kernel_size[2],
                                   stride[0], stride[1], stride[2],
                                   padding[0], padding[1], padding[2],
                                   dilation[0], dilation[1], dilation[2],
                                   im.shape()[0]);

    if (!transpose_out)
        col.RowColSwap();

    return std::move(col);
}


inline static constexpr auto im2col_cpu_3d_backward = [](auto data_im_ptr, auto data_im_ptr_end, auto data_col,
    const int64_t& channels,
    const int64_t& depth, const int64_t& height, const int64_t& width,
    const int64_t& k_d, const int64_t& k_r, const int64_t& k_c, 
    const int64_t& s_d, const int64_t& s_r, const int64_t& s_c, 
    const int64_t& p_d, const int64_t& p_r, const int64_t& p_c, 
    const int64_t& d_d, const int64_t& d_r, const int64_t& d_c,
    const int64_t& batch_size) {
    
    const int64_t depth_col = (depth + 2 * p_d - d_d * (k_d - 1) - 1) / s_d + 1;
    const int64_t height_col = (height + 2 * p_r - d_r * (k_r - 1) - 1) / s_r + 1;
    const int64_t width_col = (width + 2 * p_c - d_c * (k_c - 1) - 1) / s_c + 1;
    const int64_t channels_col = channels * k_d * k_r * k_c;
    const int64_t col_matrix_size = batch_size * channels_col * depth_col * height_col * width_col;

    const int64_t batch_im_add = channels * depth * height * width;
    const int64_t batch_col_add = channels_col * depth_col * height_col * width_col;
    const int64_t dhw_col = depth_col * height_col * width_col;
    const int64_t hw_col = height_col * width_col;

#ifdef USE_PARALLEL
    if constexpr (std::is_pointer_v<decltype(data_im_ptr)>){
	    threading::preferential_parallel_for(threading::block_ranges<2>(0, batch_size, 0, channels_col),
		[&](threading::blocked_range<2> block){
	for(int64_t batch = block.begin[0]; batch < block.end[0]; ++batch){
		auto data_col_ptr = data_col + batch * batch_col_add;
		auto data_im = data_im_ptr + batch * batch_im_add;
		for(int64_t c = block.begin[1]; c < block.end[1]; ++c){
			int64_t w_offset = c % k_c;
			int64_t h_offset = (c / k_c) % k_r;
			int64_t d_offset = (c / k_c / k_r) % k_d;
			int64_t c_im = c / k_d / k_r / k_c;
			// Calculate the valid range for h_pad
			int64_t h_pad_start = std::max(0LL, (p_r - h_offset * d_r + s_r - 1) / s_r);
			int64_t h_pad_end = std::min(height_col, (height + p_r - h_offset * d_r + s_r - 1) / s_r);

			// Calculate the valid range for w_pad
			int64_t w_pad_start = std::max(0LL, (p_c - w_offset * d_c + s_c - 1) / s_c);
			int64_t w_pad_end = std::min(width_col, (width + p_c - w_offset * d_c + s_c - 1) / s_c);

			// Calculate the valid range for d_pad
			int64_t d_pad_start = std::max(0LL, (p_d - d_offset * d_d + s_d - 1) / s_d);
			int64_t d_pad_end = std::min(depth_col, (depth + p_d - d_offset * d_d + s_d - 1) / s_d);
			for(int64_t d = d_pad_start; d < d_pad_end; ++d){
				int64_t d_pad = d * s_d - p_d + d_offset * d_d;
				for (int64_t h = h_pad_start; h < h_pad_end; ++h) {
					int64_t h_pad = h * s_r - p_r + h_offset * d_r;
					for (int64_t w = w_pad_start; w < w_pad_end; ++w) {
						int64_t w_pad = w * s_c - p_c + w_offset * d_c;
						data_im[c_im * depth * height * width +
							d_pad * height * width +
							h_pad * width 
							+ w_pad]
							+= data_col_ptr[c * height_col * width_col * depth_col
							+ d * height_col * width_col +
							h * width_col + w];
					}
				}
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
		int64_t w_offset = c % k_c;
		int64_t h_offset = (c / k_c) % k_r;
		int64_t d_offset = (c / k_c / k_r) % k_d;
		int64_t c_im = c / k_d / k_r / k_c;	
		// Calculate the valid range for h_pad
                int64_t h_pad_start = std::max(0LL, (p_r - h_offset * d_r + s_r - 1) / s_r);
                int64_t h_pad_end = std::min(height_col, (height + p_r - h_offset * d_r + s_r - 1) / s_r);

                // Calculate the valid range for w_pad
                int64_t w_pad_start = std::max(0LL, (p_c - w_offset * d_c + s_c - 1) / s_c);
                int64_t w_pad_end = std::min(width_col, (width + p_c - w_offset * d_c + s_c - 1) / s_c);
		
		// Calculate the valid range for d_pad
		int64_t d_pad_start = std::max(0LL, (p_d - d_offset * d_d + s_d - 1) / s_d);
		int64_t d_pad_end = std::min(depth_col, (depth + p_d - d_offset * d_d + s_d - 1) / s_d);
		//this is a really good check that negates a lot of edge case errors, and skips doing a lot of loops
		if(h_pad_start >= height_col || h_pad_end <= h_pad_start 
				|| w_pad_start >= width_col || w_pad_end <= w_pad_start
				|| d_pad_start >= depth_col || d_pad_end <= d_pad_start){
			data_col_ptr += dhw_col;
			continue;
		}
		//account fo d_pad_start
		data_col_ptr += (d_pad_start * hw_col);
		for(int64_t d = d_pad_start; d < d_pad_end; ++d){
			//account for h_pad_start
			data_col_ptr += h_pad_start * width_col;
			int64_t d_pad = d * s_d - p_d + d_offset * d_d;
			for(int64_t h = h_pad_start; h < h_pad_end; ++h){
				data_col_ptr += w_pad_start;
				int64_t h_pad = h * s_r - p_r + h_offset * d_r;
				for(int64_t w = w_pad_start; w < w_pad_end; ++w){
					int64_t w_pad = w * s_c - p_c + w_offset * d_c;
					data_im[c_im * height * width * depth
						+ d_pad * height * width
						+ h_pad * width 
						+ w_pad] += *data_col_ptr;
					++data_col_ptr;
				}
				// Skip to the next valid row in data_col_ptr
				data_col_ptr += width_col - w_pad_end;
			}
			// Adjust data_col_ptr for the skipped rows and columns
			data_col_ptr += (height_col - h_pad_end) * width_col;
		}
		//Adjust data_col_ptr for the skipped rows, depth, and columns
		data_col_ptr += (depth_col - d_pad_end) * hw_col;
        }
    }


     }, 0, batch_size, data_im_ptr, batch_im_add);
};


Tensor unfold3d_backward(const Tensor& x, const utils::my_n_tuple<3>& output_size, const utils::my_n_tuple<3>& kernel_size, const utils::my_n_tuple<3>& dilation, const utils::my_n_tuple<3>& padding, const utils::my_n_tuple<3>& stride, const bool transpose_out){

	utils::throw_exception(x.dims() == 2 || x.dims() == 3, "Expected to get a shape with a dimensionality of 3D or 2D, but got $ dimensions", x.dims());
	const int64_t& BDEPTH = output_size[0];
	const int64_t& BROWS = output_size[1];
	const int64_t& BCOLS = output_size[2];
	const int64_t L_d = ((output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
	const int64_t L_r = ((output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
	const int64_t L_c = ((output_size[2] + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) / stride[2]) + 1;

	const int64_t L = L_d * L_r * L_c;
	const int64_t LKern = kernel_size[0] * kernel_size[1] * kernel_size[2];
	const int64_t batches = x.dims() == 3 ? x.shape()[0] : 1;
	utils::throw_exception(L > 0, "Given input with spatial size ($, $, $), kernel_size=($, $, $), dilation=($, $, $), padding=($, $, $), calculated shape of the array of sliding blocks as ($, $, $), but its components must be at least one.",
                           x.shape()[-3], x.shape()[-2], x.shape()[-1], 
                           kernel_size[0], kernel_size[1], kernel_size[2], 
                           dilation[0], dilation[1], dilation[2], 
                           padding[0], padding[1], padding[2], 
                           L_d, L_r, L_c);




	Tensor Z = (x.dims() == 2) ? x.unsqueeze(0) : x;
	if(!transpose_out)
		Z = Z.contiguous().RowColSwap();
	utils::throw_exception(Z.shape()[1] % LKern == 0, "Given kernel size of $, $ * $ * $ = $ must be able to divide input at dimension $, but got input.shape()[$] = $", kernel_size, kernel_size[0], kernel_size[1], kernel_size[2], LKern, (x.dims() == 3) ? 1 : 0, (x.dims() == 3) ? 1 : 0, Z.shape()[1]);
	const int64_t channels = Z.shape()[1] / LKern;
	SizeRef output_shape({batches, channels, BDEPTH, BROWS, BCOLS});
	utils::throw_exception(Z.shape()[2] == L, "Given output_size=($, $, $), kernel_size=($, $, $), dilation=($, $, $), padding=($, $, $), stride=($, $, $), and transpose_out = $, expected size of input's dimension 2 to match the calculated number of sliding blocks $ * $ * $ = $, but got input.size(2)=$.", 
			output_size[0], output_size[1], output_size[2], 
			kernel_size[0], kernel_size[1], kernel_size[2],
			dilation[0], dilation[1], dilation[2],
			padding[0], padding[1], padding[2],
			stride[0], stride[1], stride[2],
			(transpose_out) ? "true" : "false",
			L_d, L_r, L_c, L, Z.shape()[1]);
	
	Tensor output = zeros(output_shape, x.dtype);
	output.arr_void().execute_function_nbool(im2col_cpu_3d_backward, 
			Z.arr_void(),  channels,
			BDEPTH, BROWS, BCOLS,
			kernel_size[0], kernel_size[1], kernel_size[2],
			stride[0], stride[1], stride[2],
			padding[0], padding[1], padding[2],
			dilation[0], dilation[1], dilation[2],
			batches);

	
	if(!transpose_out && x.is_contiguous())
		Z.RowColSwap();
	return std::move(output);
}


Tensor& unfold3d_backward(const Tensor& x, Tensor& output, const utils::my_n_tuple<3>& output_size, const utils::my_n_tuple<3>& kernel_size, const utils::my_n_tuple<3>& dilation, const utils::my_n_tuple<3>& padding, const utils::my_n_tuple<3>& stride, const bool transpose_out){

	utils::throw_exception(x.dims() == 2 || x.dims() == 3, "Expected to get a shape with a dimensionality of 3D or 2D, but got $ dimensions", x.dims());
	const int64_t& BDEPTH = output_size[0];
	const int64_t& BROWS = output_size[1];
	const int64_t& BCOLS = output_size[2];
	const int64_t L_d = ((output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
	const int64_t L_r = ((output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
	const int64_t L_c = ((output_size[2] + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) / stride[2]) + 1;

	const int64_t L = L_d * L_r * L_c;
	const int64_t LKern = kernel_size[0] * kernel_size[1] * kernel_size[2];
	const int64_t batches = x.dims() == 3 ? x.shape()[0] : 1;
	utils::throw_exception(L > 0, "Given input with spatial size ($, $, $), kernel_size=($, $, $), dilation=($, $, $), padding=($, $, $), calculated shape of the array of sliding blocks as ($, $, $), but its components must be at least one.",
                           x.shape()[-3], x.shape()[-2], x.shape()[-1], 
                           kernel_size[0], kernel_size[1], kernel_size[2], 
                           dilation[0], dilation[1], dilation[2], 
                           padding[0], padding[1], padding[2], 
                           L_d, L_r, L_c);




	Tensor Z = (x.dims() == 2) ? x.unsqueeze(0) : x;
	if(!transpose_out)
		Z = Z.contiguous().RowColSwap();
	utils::throw_exception(Z.shape()[1] % LKern == 0, "Given kernel size of $, $ * $ * $ = $ must be able to divide input at dimension $, but got input.shape()[$] = $", kernel_size, kernel_size[0], kernel_size[1], kernel_size[2], LKern, (x.dims() == 3) ? 1 : 0, (x.dims() == 3) ? 1 : 0, Z.shape()[1]);
	const int64_t channels = Z.shape()[1] / LKern;
	SizeRef output_shape({batches, channels, BDEPTH, BROWS, BCOLS});
	utils::throw_exception(Z.shape()[2] == L, "Given output_size=($, $, $), kernel_size=($, $, $), dilation=($, $, $), padding=($, $, $), stride=($, $, $), and transpose_out = $, expected size of input's dimension 2 to match the calculated number of sliding blocks $ * $ * $ = $, but got input.size(2)=$.", 
			output_size[0], output_size[1], output_size[2], 
			kernel_size[0], kernel_size[1], kernel_size[2],
			dilation[0], dilation[1], dilation[2],
			padding[0], padding[1], padding[2],
			stride[0], stride[1], stride[2],
			(transpose_out) ? "true" : "false",
			L_d, L_r, L_c, L, Z.shape()[1]);
	
	
	utils::throw_exception(output_shape.multiply() == output.shape().multiply(), "Expected output shape to match $ but got $ for fold backward", output_shape, output.shape());
	utils::throw_exception(output.dtype == x.dtype, "Expected dtypes of grad and output to match for fold backward but got $ for output and $ for the grad", output.dtype, x.dtype);
	output.arr_void().execute_function_nbool(im2col_cpu_3d_backward, 
			Z.arr_void(), channels,
			BDEPTH, BROWS, BCOLS,
			kernel_size[0], kernel_size[1], kernel_size[2],
			stride[0], stride[1], stride[2],
			padding[0], padding[1], padding[2],
			dilation[0], dilation[1], dilation[2],
			batches);

	
	if(!transpose_out && x.is_contiguous())
		Z.RowColSwap();
	return output;
}



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
			int64_t w_pad_start = std::max(0LL, (p_w - w_offset * d_w + s_w - 1) / s_w);
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
                int64_t w_pad_start = std::max(0LL, (p_w - w_offset * d_w + s_w - 1) / s_w);
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


Tensor unfold1d(const Tensor& x, int64_t kernel_size, int64_t dilation, int64_t padding, int64_t stride, bool transpose_out){
	utils::throw_exception(x.dims() >= 2, "Expected input tensor to unfold to have dims greater than or equal to 2 but got $D", x.dims());

	const int64_t LKern = kernel_size;

	const int64_t L = ((x.shape()[-1] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

	utils::throw_exception(L > 0, "Given input with width ($), kernel_size=($), dilation=($), padding=($), calculated shape of the array of sliding blocks as ($), but its components must be at least one.", x.shape()[-1], kernel_size, dilation, padding, L);
	Tensor im = (x.dims() == 2) ? x.view(1, x.shape()[0], x.shape()[1]) : x.flatten(0, -3);

	Tensor col = zeros({im.shape()[0], im.shape()[1] * LKern, L});
	const int64_t& channels = im.shape()[1];
	im.arr_void().execute_function(im2col_cpu_1d, col.arr_void(), channels,
			x.shape()[-1],
			kernel_size,
			stride,
			padding,
			dilation,
			im.shape()[0]);
	if(!transpose_out)
		col.RowColSwap();
	return std::move(col);
}

inline static constexpr auto im2col_cpu_1d_backward = [](auto data_im_ptr, auto data_im_ptr_end, auto data_col,
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
			int64_t w_pad_start = std::max(0LL, (p_w - w_offset * d_w + s_w - 1) / s_w);
			int64_t w_pad_end = std::min(width_col, (width + p_w - w_offset * d_w + s_w - 1) / s_w);

			for (int64_t w = w_pad_start; w < w_pad_end; ++w) {
				int64_t w_pad = w * s_w - p_w + w_offset * d_w;
				data_im[c_im * width + w_pad] += data_col_ptr[c * width_col +  w];
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
                int64_t w_pad_start = std::max(0LL, (p_w - w_offset * d_w + s_w - 1) / s_w);
                int64_t w_pad_end = std::min(width_col, (width + p_w - w_offset * d_w + s_w - 1) / s_w);
		
		//this is a really good check that negates a lot of edge case errors, and skips doing a lot of loops
		if(w_pad_start >= width_col || w_pad_end <= w_pad_start){
			data_col_ptr += width_col;
			continue;
		}
		
		data_col_ptr += w_pad_start;
		for(int64_t w = w_pad_start; w < w_pad_end; ++w){
			int64_t w_pad = w * s_w - p_w + w_offset * d_w;
			data_im[c_im * width + w_pad] += *data_col_ptr;
			++data_col_ptr;
		}
		// Skip to the next valid row in data_col_ptr
		data_col_ptr += width_col - w_pad_end;
		
        }
    }


     }, 0, batch_size, data_im_ptr, batch_im_add);

};


Tensor unfold1d_backward(const Tensor& x, Tensor::size_value_t output_size, Tensor::size_value_t kernel_size, Tensor::size_value_t dilation, Tensor::size_value_t padding, Tensor::size_value_t stride, bool transpose_out){
	utils::throw_exception(x.dims() == 2 || x.dims() == 3, "Expected to get a shape with a dimensionality of 3D or 2D, but got $ dimensions", x.dims());
	const int64_t& BCOLS = output_size;
	const int64_t L_c = ((output_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
	const int64_t& L = L_c;
	const int64_t& LKern = kernel_size;
	const int64_t batches = x.dims() == 3 ? x.shape()[0] : 1;
	utils::throw_exception(L_c > 0, "Given input with spatial size ($), kernel_size=($), dilation=($), padding=($), calculated shape of the array of sliding blocks as ($), but its components must be at least one.", output_size, kernel_size, dilation, padding, L_c);
	Tensor Z = (x.dims() == 2) ? x.unsqueeze(0) : x;
	if(!transpose_out)
		Z = Z.contiguous().RowColSwap();
	utils::throw_exception(Z.shape()[1] % LKern == 0, "Given kernel size of $ must be able to divide input at dimension $, but got input.shape()[$] = $", kernel_size, (x.dims() == 3) ? 1 : 0, (x.dims() == 3) ? 1 : 0, Z.shape()[1]);
	const int64_t channels = Z.shape()[1] / LKern;
	SizeRef output_shape({batches, channels, BCOLS});
	utils::throw_exception(Z.shape()[2] == L, "Given output_size=($), kernel_size=($), dilation=($), padding=($), stride=($), expected size of input's dimension 2 to match the calculated number of sliding blocks $, but got input.size(2)=$.", output_size, kernel_size, dilation, padding, stride, L, Z.shape()[2]);
	Tensor output = zeros(output_shape, x.dtype);
	output.arr_void().execute_function_nbool(im2col_cpu_1d_backward, 
			Z.arr_void(), channels, BCOLS,
			kernel_size,
			stride,
			padding,
			dilation,
			batches);
	if(!transpose_out && x.is_contiguous())
		Z.RowColSwap();
	return std::move(output);
}


Tensor& unfold1d_backward(const Tensor& x, Tensor& output, Tensor::size_value_t output_size, Tensor::size_value_t kernel_size, Tensor::size_value_t dilation, Tensor::size_value_t padding, Tensor::size_value_t stride, bool transpose_out){
	utils::throw_exception(x.dims() == 2 || x.dims() == 3, "Expected to get a shape with a dimensionality of 3D or 2D, but got $ dimensions", x.dims());
	const int64_t& BCOLS = output_size;
	const int64_t L_c = ((output_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
	const int64_t& L = L_c;
	const int64_t& LKern = kernel_size;
	const int64_t batches = x.dims() == 3 ? x.shape()[0] : 1;
	utils::throw_exception(L_c > 0, "Given input with spatial size ($), kernel_size=($), dilation=($), padding=($), calculated shape of the array of sliding blocks as ($), but its components must be at least one.", output_size, kernel_size, dilation, padding, L_c);
	Tensor Z = (x.dims() == 2) ? x.unsqueeze(0) : x;
	if(!transpose_out)
		Z = Z.contiguous().RowColSwap();
	utils::throw_exception(Z.shape()[1] % LKern == 0, "Given kernel size of $ must be able to divide input at dimension $, but got input.shape()[$] = $", kernel_size, (x.dims() == 3) ? 1 : 0, (x.dims() == 3) ? 1 : 0, Z.shape()[1]);
	const int64_t channels = Z.shape()[1] / LKern;
	SizeRef output_shape({batches, channels, BCOLS});
	utils::throw_exception(Z.shape()[2] == L, "Given output_size=($), kernel_size=($), dilation=($), padding=($), stride=($), expected size of input's dimension 2 to match the calculated number of sliding blocks $, but got input.size(2)=$.", output_size, kernel_size, dilation, padding, stride, L, Z.shape()[2]);
	utils::throw_exception(output_shape.multiply() == output.shape().multiply(), "Expected output shape to match $ but got $ for fold backward", output_shape, output.shape());
	utils::throw_exception(output.dtype == x.dtype, "Expected dtypes of grad and output to match for fold backward but got $ for output and $ for the grad", output.dtype, x.dtype);
	output.arr_void().execute_function_nbool(im2col_cpu_1d_backward, 
			Z.arr_void(), channels, BCOLS,
			kernel_size,
			stride,
			padding,
			dilation,
			batches);
	if(!transpose_out && x.is_contiguous())
		Z.RowColSwap();
	return output;
}



}} //nt::functional
