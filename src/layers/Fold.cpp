#include "layers.h"


#include <sys/types.h>
#include <variant>
#include <tuple>
#include "../dtype/ArrayVoid.hpp"

#ifdef USE_PARALLEL
	#include <tbb/parallel_for_each.h>
	#include <tbb/parallel_for.h>
	#include <tbb/blocked_range.h>
	#include <tbb/blocked_range2d.h>
	#include <tbb/blocked_range3d.h>
	/* #include <tbb/blocked_rangeNd.h> */
#endif

namespace nt{
namespace layers{


Unfold::Unfold(utils::my_tuple kernel_size, 
			utils::my_tuple dilation,
			utils::my_tuple padding,
			utils::my_tuple stride,
			bool transpose_out)
	:kernel_size(kernel_size), dilation(dilation), padding(padding), stride(stride), out_transpose(transpose_out)
{
	LKern = kernel_size[0] * kernel_size[1];
}


//while the following works, when it comes to larger tensors it is simply too slow
//going to implement a version that is purely for loops
//should not only be faster, but easier for parallelization
//maybe going to use the pool module that I had created for a previous project, should potentially work better
/* Tensor Unfold::forward(const Tensor& x){ */
/* 	utils::throw_exception(x.dims() == 4, "Expected dimensions of Tensor to unfold to be 4 but got $", x.dims()); */
/* 	uint32_t L = ((x.shape()[-1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1; */ 
/* 	L *= ((x.shape()[-2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1; */
/* 	SizeRef outp_shape_ut({x.shape()[0], L, x.shape()[1] * LKern}); */
/* 	bool dilate = (dilation[0] > 1 || dilation[1] > 1); */
/* 	Tensor X = dilate ? x.dilate_mem_(dilation[0]) : x; */
/* 	if(padding[0] != 0 || padding[1] != 0) */
/* 		X = X.pad({padding[0], padding[1]}); */
	
/* 	Tensor unfolded = X.unfold(2, kernel_size[0], stride[0]).unfold(3, kernel_size[1], stride[1]); */
/* 	uint32_t neg_one = unfolded.numel() / (x.shape()[0] * x.shape()[1] * kernel_size[0] * kernel_size[1]); */
/* 	std::cout << "finished unfold"<<std::endl; */
/* 	unfolded = unfolded.view({x.shape()[0], x.shape()[1], neg_one, kernel_size[0], kernel_size[1]}); */
/* 	std::cout<<"doing permute"<<std::endl; */
/* 	unfolded = unfolded.permute({0, 2, 1, 3, 4}).contiguous().view(outp_shape_ut); */
/* 	std::cout<<"doing row col swap"<<std::endl; */
/* 	/1* unfolded.RowColSwap(); *1/ */
/* 	unfolded = unfolded.transpose(-1,-2); */
/* 	std::cout<<"did row col swap"<<std::endl; */
/* 	/1* unfolded = unfolded.view({unfolded.shape()[0], unfolded.shape()[2], unfolded.shape()[1]}); *1/ */

/* 	return std::move(unfolded); */
/* } */


//%s/const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, const uint32_t& d_r, const uint32_t& d_c/const uint32_t\& k_r, const uint32_t\& k_c, const uint32_t\& s_r, const uint32_t\& s_c, const uint32_t\& d_r, const uint32_t\& d_c/g

inline static constexpr auto im2col_nn_layer_2d = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, const uint32_t& d_r, const uint32_t& d_c, const uint32_t& t_r, const uint32_t& t_c){
	uint32_t a_current_add = ((t_c * d_r) - (k_c*d_c)); 
	for(uint32_t r = 0; r < t_r - ((k_r-1)*d_c); r += s_r){
		for(uint32_t c = 0; c < t_c - ((k_c-1)*d_c); c += s_c){
			auto a_current = a_begin + (r * t_c + c);
			for(uint32_t rr = 0; rr < k_r; rr += d_r, a_current += a_current_add){
				for(uint32_t cc = 0; cc < k_c; cc += d_c, a_current += d_c, ++b_begin){
					*b_begin = *a_current;
				}
			}
		}
	}
};


//%s/const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, int32_t d_r, const uint32_t& d_c/const uint32_t\& k_r, const uint32_t\& k_c, const uint32_t\& s_r, const uint32_t\& s_c, int32_t d_r, const uint32_t\& d_c
inline static constexpr auto im2col_nn_layer_2d_pad = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, int32_t d_r, const uint32_t& d_c, int32_t p_r, int32_t p_c, int32_t t_r, int32_t t_c){
	uint32_t a_current_add = (t_c * d_r); 
	uint32_t upper_row2 = t_r - ((k_r-1)*d_r) + (p_r * 2);
	uint32_t upper_col2 = t_c - ((k_c-1)*d_c) + (p_c * 2);
	uint32_t kd_r = (k_r/d_r);
	uint32_t kd_c = (k_c/d_c);
	for(int32_t r = 0; r < upper_row2; r += s_r){
		for(int32_t c = 0; c < upper_col2; c += s_c){
			uint32_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
			uint32_t c_add = c - p_c <= 0 ? 0 : c - p_c;
			auto a_current = a_begin + r_add + c_add;
			for(int32_t rr = 0; rr < k_r; rr += d_r){
				if((r - p_r + rr) < 0){
					if((r - p_r + rr + d_r) > 0){
						a_current += t_c * (r - p_r + rr + d_r);
					}
					b_begin += kd_c;
					continue;
				}
				if((r - p_r + rr) >= t_r){
					b_begin += kd_c * ((kd_r - (rr+d_r))+1);
					break;
				}
				uint32_t pad_counter = 0;
				for(int32_t cc = 0; cc < k_c; cc += d_c, ++b_begin){
					if((c - p_c + cc) < t_c && (c - p_c + cc) >= 0){
						*b_begin = *a_current;
						a_current += d_c;
						pad_counter += d_c;

					}
				}
				a_current += a_current_add - pad_counter;
			}
		}
	}
};


inline static constexpr auto im2col_nn_layer_2d_backward = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, const uint32_t& d_r, const uint32_t& d_c, const uint32_t& t_r, const uint32_t& t_c){
	uint32_t a_current_add = ((t_c * d_r) - (k_c*d_c)); 
	for(uint32_t r = 0; r < t_r - ((k_r-1)*d_c); r += s_r){
		for(uint32_t c = 0; c < t_c - ((k_c-1)*d_c); c += s_c){
			auto a_current = a_begin + (r * t_c + c);
			for(uint32_t rr = 0; rr < k_r*d_r; rr += d_r, a_current += a_current_add){
				for(uint32_t cc = 0; cc < k_c*d_c; cc += d_c, a_current += d_c, ++b_begin){
					*a_current = *b_begin;
				}
			}
		}
	}
};

inline static constexpr auto im2col_nn_layer_2d_pad_backward = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, int32_t d_r, const uint32_t& d_c, int32_t p_r, int32_t p_c, int32_t t_r, int32_t t_c){
	uint32_t a_current_add = (t_c * d_r); 
	uint32_t upper_row2 = t_r - ((k_r-1)*d_r) + (p_r * 2);
	uint32_t upper_col2 = t_c - ((k_c-1)*d_c) + (p_c * 2);
	uint32_t kd_r = (k_r/d_r);
	uint32_t kd_c = (k_c/d_c);
	for(int32_t r = 0; r < upper_row2; r += s_r){
		for(int32_t c = 0; c < upper_col2; c += s_c){
			uint32_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
			uint32_t c_add = c - p_c <= 0 ? 0 : c - p_c;
			auto a_current = a_begin + r_add + c_add;
			for(int32_t rr = 0; rr < k_r*d_r; rr += d_r){
				if((r - p_r + rr) < 0){
					if((r - p_r + rr + d_r) > 0){
						a_current += t_c * (r - p_r + rr + d_r);
					}
					b_begin += kd_c;
					continue;
				}
				if((r - p_r + rr) >= t_r){
					b_begin += kd_c * ((kd_r - (rr+d_r))+1);
					break;
				}
				uint32_t pad_counter = 0;
				for(int32_t cc = 0; cc < k_c*d_c; cc += d_c, ++b_begin){
					if((c - p_c + cc) < t_c && (c - p_c + cc) >= 0){
						*a_current = *b_begin;
						a_current += d_c;
						pad_counter += d_c;

					}
				}
				a_current += a_current_add - pad_counter;
			}
		}
	}
};


#ifndef USE_PARALLEL
inline static constexpr auto im2col_nn_layer_nd = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, const uint32_t& d_r, const uint32_t& d_c, const uint32_t& t_r, const uint32_t& t_c, const uint32_t& LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	uint32_t a_current_add = ((t_c * d_r) - (k_c*d_c));
	const uint32_t mat_size = t_c * t_r;
	const uint32_t upper_rows = t_r - ((k_r-1) * d_r);
	const uint32_t upper_cols = t_c - ((k_c-1) * d_c);
	const uint32_t upper_rows2 = k_r*d_r;
	const uint32_t upper_cols2 = k_c * d_c;
	for(uint32_t ba = 0; ba < batches; ++ba){
		uint32_t batch_add = (ba * channels * mat_size);
		for(uint32_t r = 0; r < upper_rows; r += s_r){
			for(uint32_t c = 0; c < upper_cols; c += s_c){
				for(uint32_t ch = 0; ch < channels; ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(uint32_t rr = 0; rr < upper_rows2; rr += d_r, a_current += a_current_add){
						for(uint32_t cc = 0; cc < upper_cols2; cc += d_c, a_current += d_c, ++b_begin){
							*b_begin = *a_current;
						}
					}
				}
			}
		}
	}
};

inline static constexpr auto im2col_nn_layer_nd_none = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& t_r, const uint32_t& t_c, const uint32_t& LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	uint32_t a_current_add = (t_c - k_c);
	const uint32_t mat_size = t_c * t_r;
	const uint32_t upper_rows = t_r - (k_r - 1);
	const uint32_t upper_cols = t_c - (k_c - 1);
	for(uint32_t ba = 0; ba < batches; ++ba){
		uint32_t batch_add = (ba * channels * mat_size);
		for(uint32_t r = 0; r < upper_rows; ++r){
			for(uint32_t c = 0; c < upper_cols; ++c){
				for(uint32_t ch = 0; ch < channels; ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(uint32_t rr = 0; rr < k_r; ++rr, a_current += a_current_add){
						for(uint32_t cc = 0; cc < k_c; ++cc, ++a_current, ++b_begin){
							*b_begin = *a_current;
						}
					}
				}
			}
		}
	}
};

inline static constexpr auto im2col_nn_layer_nd_stride = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, const uint32_t& t_r, const uint32_t& t_c, const uint32_t& LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	uint32_t a_current_add = (t_c - k_c);
	const uint32_t mat_size = t_c * t_r;
	const uint32_t upper_rows = t_r - (k_r - 1);
	const uint32_t upper_cols = t_c - (k_c - 1);
	for(uint32_t ba = 0; ba < batches; ++ba){
		uint32_t batch_add = (ba * channels * mat_size);
		for(uint32_t r = 0; r < upper_rows; r += s_r){
			for(uint32_t c = 0; c < upper_cols; c += s_c){
				for(uint32_t ch = 0; ch < channels; ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(uint32_t rr = 0; rr < k_r; ++rr, a_current += a_current_add){
						for(uint32_t cc = 0; cc < k_c; ++cc, ++a_current, ++b_begin){
							*b_begin = *a_current;
						}
					}
				}
			}
		}
	}
};

inline static constexpr auto im2col_nn_layer_nd_dilation = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& d_r, const uint32_t& d_c, const uint32_t& t_r, const uint32_t& t_c, const uint32_t& LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	uint32_t a_current_add = ((t_c * d_r) - (k_c*d_c));
	const uint32_t mat_size = t_c * t_r;
	const uint32_t upper_rows = t_r - ((k_r-1) * d_r);
	const uint32_t upper_cols = t_c - ((k_c-1) * d_c);
	const uint32_t upper_rows2 = k_r*d_r;
	const uint32_t upper_cols2 = k_c * d_c;
	for(uint32_t ba = 0; ba < batches; ++ba){
		uint32_t batch_add = (ba * channels * mat_size);
		for(uint32_t r = 0; r < upper_rows; ++r){
			for(uint32_t c = 0; c < upper_cols; ++c){
				for(uint32_t ch = 0; ch < channels; ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(uint32_t rr = 0; rr < upper_rows2; rr += d_r, a_current += a_current_add){
						for(uint32_t cc = 0; cc < upper_cols2; cc += d_c, a_current += d_c, ++b_begin){
							*b_begin = *a_current;
						}
					}
				}
			}
		}
	}
};
#else
inline static constexpr auto im2col_nn_layer_nd = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, const uint32_t& d_r, const uint32_t& d_c, const uint32_t& t_r, const uint32_t& t_c, const uint32_t& LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){

	uint32_t row_max = (t_r - ((k_r-1) * d_r));
	uint32_t col_max = t_c - ((k_c-1) * d_c);
	const uint32_t upper_row2 = k_r * d_r;
	const uint32_t upper_col2 = k_c * d_c;

	uint32_t a_current_add = ((t_c * d_r) - (k_c * d_c));
	const uint32_t mat_size = t_c * t_r;
	for(uint32_t ba = 0; ba < batches; ++ba){
		uint32_t batch_add = (ba * channels * mat_size);
		//there might be an issue with the stride on this one, need to look into that.
		tbb::parallel_for(tbb::blocked_range3d<uint32_t>(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<uint32_t> range){
		uint32_t row_begin = range.rows().begin() % s_r == 0 ? range.rows().begin() : range.rows().begin() + range.rows().begin() % s_r;
		uint32_t col_begin = range.cols().begin() % s_c == 0 ? range.cols().begin() : range.cols().begin() + range.cols().begin() % s_c;
		uint32_t m_begin_index = ba * L * LKern * channels; //batches 
		m_begin_index += ((row_begin) * col_max * LKern * channels) / (s_r * s_c); //rows;
		m_begin_index += ((col_begin) * LKern * channels) / s_c;//cols
		m_begin_index += range.pages().begin() * LKern;
		auto m_begin = b_begin + m_begin_index;
		for(uint32_t r = row_begin; r < range.rows().end(); r += s_r){
			for(uint32_t c = col_begin; c < range.cols().end(); c += s_c){
				for(uint32_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(uint32_t rr = 0; rr < upper_row2; rr += d_r, a_current += a_current_add){
						for(uint32_t cc = 0; cc < upper_col2; cc += d_c, a_current += d_c, ++m_begin){
							*m_begin = *a_current;
						}
					}
				}
			}
		}
		});
	}
};

inline static constexpr auto im2col_nn_layer_nd_dilation = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& d_r, const uint32_t& d_c, const uint32_t& t_r, const uint32_t& t_c, const uint32_t& LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){

	uint32_t row_max = (t_r - ((k_r-1) * d_r));
	uint32_t col_max = t_c - ((k_c-1) * d_c);
	const uint32_t upper_row2 = k_r * d_r;
	const uint32_t upper_col2 = k_c * d_c;

	uint32_t a_current_add = ((t_c * d_r) - (k_c * d_c));
	const uint32_t mat_size = t_c * t_r;
	for(uint32_t ba = 0; ba < batches; ++ba){
		uint32_t batch_add = (ba * channels * mat_size);
		//there might be an issue with the stride on this one, need to look into that.
		tbb::parallel_for(tbb::blocked_range3d<uint32_t>(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<uint32_t> range){
		uint32_t row_begin = range.rows().begin();
		uint32_t col_begin = range.cols().begin();
		uint32_t m_begin_index = ba * L * LKern * channels; //batches 
		m_begin_index += (row_begin * col_max * LKern * channels); //rows;
		m_begin_index += (col_begin * LKern * channels);//cols
		m_begin_index += range.pages().begin() * LKern;
		auto m_begin = b_begin + m_begin_index;
		for(uint32_t r = row_begin; r < range.rows().end(); ++r){
			for(uint32_t c = col_begin; c < range.cols().end(); ++c){
				for(uint32_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(uint32_t rr = 0; rr < upper_row2; rr += d_r, a_current += a_current_add){
						for(uint32_t cc = 0; cc < upper_col2; cc += d_c, a_current += d_c, ++m_begin){
							*m_begin = *a_current;
						}
					}
				}
			}
		}
		});
	}
};

inline static constexpr auto im2col_nn_layer_nd_stride = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, const uint32_t& t_r, const uint32_t& t_c, const uint32_t& LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){

	uint32_t row_max = (t_r - (k_r-1));
	uint32_t col_max = t_c - (k_c-1);
	const uint32_t upper_row2 = k_r;
	const uint32_t upper_col2 = k_c;

	uint32_t a_current_add = (t_c - k_c);
	const uint32_t mat_size = t_c * t_r;
	for(uint32_t ba = 0; ba < batches; ++ba){
		uint32_t batch_add = (ba * channels * mat_size);
		//there might be an issue with the stride on this one, need to look into that.
		tbb::parallel_for(tbb::blocked_range3d<uint32_t>(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<uint32_t> range){
		uint32_t row_begin = range.rows().begin() % s_r == 0 ? range.rows().begin() : range.rows().begin() + range.rows().begin() % s_r;
		uint32_t col_begin = range.cols().begin() % s_c == 0 ? range.cols().begin() : range.cols().begin() + range.cols().begin() % s_c;
		uint32_t m_begin_index = ba * L * LKern * channels; //batches 
		m_begin_index += ((row_begin) * col_max * LKern * channels) / (s_r * s_c); //rows;
		m_begin_index += ((col_begin) * LKern * channels) / s_c;//cols
		m_begin_index += range.pages().begin() * LKern;
		auto m_begin = b_begin + m_begin_index;
		for(uint32_t r = row_begin; r < range.rows().end(); r += s_r){
			for(uint32_t c = col_begin; c < range.cols().end(); c += s_c){
				for(uint32_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(uint32_t rr = 0; rr < upper_row2; ++rr, a_current += a_current_add){
						for(uint32_t cc = 0; cc < upper_col2; ++cc, ++a_current, ++m_begin){
							*m_begin = *a_current;
						}
					}
				}
			}
		}
		});
	}
};


inline static constexpr auto im2col_nn_layer_nd_none = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& t_r, const uint32_t& t_c, const uint32_t& LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){

	uint32_t row_max = (t_r - (k_r-1));
	uint32_t col_max = t_c - (k_c-1);
	const uint32_t upper_row2 = k_r;
	const uint32_t upper_col2 = k_c;

	uint32_t a_current_add = (t_c - k_c);
	const uint32_t mat_size = t_c * t_r;
	for(uint32_t ba = 0; ba < batches; ++ba){
		uint32_t batch_add = (ba * channels * mat_size);
		//there might be an issue with the stride on this one, need to look into that.
		tbb::parallel_for(tbb::blocked_range3d<uint32_t>(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<uint32_t> range){
		uint32_t m_begin_index = ba * L * LKern * channels; //batches 
		m_begin_index += (range.rows().begin() * col_max * LKern * channels); //rows;
		m_begin_index += (range.cols().begin() * LKern * channels);//cols
		m_begin_index += range.pages().begin() * LKern;
		auto m_begin = b_begin + m_begin_index;
		for(uint32_t r = range.rows().begin(); r < range.rows().end(); ++r){
			for(uint32_t c = range.cols().begin(); c < range.cols().end(); ++c){
				for(uint32_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(uint32_t rr = 0; rr < upper_row2; ++rr, a_current += a_current_add){
						for(uint32_t cc = 0; cc < upper_col2; ++cc, ++a_current, ++m_begin){
							*m_begin = *a_current;
						}
					}
				}
			}
		}
		});
	}
};


//7448432
//1196112

#endif

#ifndef USE_PARALLEL
inline static constexpr auto im2col_nn_layer_nd_backward = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, const uint32_t& d_r, const uint32_t& d_c, const uint32_t& t_r, const uint32_t& t_c, const uint32_t& LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	uint32_t a_current_add = ((t_c * d_r) - (k_c*d_c));
	const uint32_t mat_size = t_c * t_r;
	for(uint32_t ba = 0; ba < batches; ++ba){
		uint32_t batch_add = (ba * channels * mat_size);
		for(uint32_t r = 0; r < t_r - ((k_r-1) * d_r); r += s_r){
			for(uint32_t c = 0; c < t_c - ((k_c-1) * d_c); c += s_c){
				for(uint32_t ch = 0; ch < channels; ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(uint32_t rr = 0; rr < k_r*d_r; rr += d_r, a_current += a_current_add){
						for(uint32_t cc = 0; cc < k_c*d_c; cc += d_c, a_current += d_c, ++b_begin){
							*a_current = *b_begin;
						}
					}
				}
			}
		}
	}
};
#else
inline static constexpr auto im2col_nn_layer_nd_backward = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, const uint32_t& d_r, const uint32_t& d_c, const uint32_t& t_r, const uint32_t& t_c, const uint32_t& LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){

	uint32_t row_max = (t_r - ((k_r-1) * d_r));
	uint32_t col_max = t_c - ((k_c-1) * d_c);

	uint32_t a_current_add = ((t_c * d_r) - (k_c * d_c));
	const uint32_t mat_size = t_c * t_r;
	for(uint32_t ba = 0; ba < batches; ++ba){
		uint32_t batch_add = (ba * channels * mat_size);
		//there might be an issue with the stride on this one, need to look into that.
		tbb::parallel_for(tbb::blocked_range3d<uint32_t>(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<uint32_t> range){
		uint32_t row_begin = range.rows().begin() % s_r == 0 ? range.rows().begin() : range.rows().begin() + range.rows().begin() % s_r;
		uint32_t col_begin = range.cols().begin() % s_c == 0 ? range.cols().begin() : range.cols().begin() + range.cols().begin() % s_c;
		uint32_t m_begin_index = ba * L * LKern * channels; //batches 
		m_begin_index += ((row_begin) * col_max * LKern * channels) / (s_r * s_c); //rows;
		m_begin_index += ((col_begin) * LKern * channels) / s_c;//cols
		m_begin_index += range.pages().begin() * LKern;
		auto m_begin = b_begin + m_begin_index;
		for(uint32_t r = row_begin; r < range.rows().end(); r += s_r){
			for(uint32_t c = col_begin; c < range.cols().end(); c += s_c){
				for(uint32_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(uint32_t rr = 0; rr < k_r*d_r; rr += d_r, a_current += a_current_add){
						for(uint32_t cc = 0; cc < k_c*d_c; cc += d_c, a_current += d_c, ++m_begin){
							*a_current = *m_begin;
						}
					}
				}
			}
		}
		});
	}
};

//7448432
//1196112

#endif

#ifndef USE_PARALLEL

inline static constexpr auto im2col_nn_layer_nd_pad = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, int32_t d_r, const uint32_t& d_c, int32_t p_r, int32_t p_c, int32_t t_r, int32_t t_c, const uint32_t& LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	uint32_t a_current_add = (t_c * d_r); 
	uint32_t upper_row2 = t_r - ((k_r-1) * d_r) + (p_r * 2);
	uint32_t upper_col2 = t_c - ((k_c-1) * d_c) + (p_c * 2);
	uint32_t kd_r = (k_r*d_r);
	uint32_t kd_c = (k_c*d_c);
	const uint32_t mat_size = t_c * t_r;
	for(uint32_t ba = 0; ba < batches; ++ba){
		uint32_t batch_add = (ba * channels * mat_size);
		for(int32_t r = 0; r < upper_row2; r += s_r){
			for(int32_t c = 0; c < upper_col2; c += s_c){
				for(uint32_t ch = 0; ch < channels; ++ch){
					uint32_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					uint32_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int32_t rr = 0; rr < kd_r; rr += d_r){
						if((r - p_r + rr) < 0){
							if((r - p_r + rr + d_r) > 0){
								a_current += t_c * (r - p_r + rr + d_r);
							}
							b_begin += k_c;
							continue;
						}
						if((r - p_r + rr) >= t_r){
							b_begin += k_c * ((k_r - (rr/d_r)));
							break;
						}
						uint32_t pad_counter = 0;
						for(int32_t cc = 0; cc < kd_c; cc += d_c, ++b_begin){
							if((c - p_c + cc) < t_c && (c - p_c + cc) >= 0){
								*b_begin = *a_current;
								a_current += d_c;
								pad_counter += d_c;
							}
							else if((c - p_c + cc + d_c) > 0 && (c - p_c + cc + d_c) < t_c){
								a_current += (c - p_c + cc + d_c);
								pad_counter += (c - p_c + cc + d_c);
							}
						}
						a_current += a_current_add - pad_counter;
					}
				}
			}
		}
	}
};

inline static constexpr auto im2col_nn_layer_nd_pad_dilation = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, int32_t d_r, const uint32_t& d_c, int32_t p_r, int32_t p_c, int32_t t_r, int32_t t_c, const uint32_t& LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	uint32_t a_current_add = (t_c * d_r); 
	uint32_t upper_row2 = t_r - ((k_r-1) * d_r) + (p_r * 2);
	uint32_t upper_col2 = t_c - ((k_c-1) * d_c) + (p_c * 2);
	uint32_t kd_r = (k_r*d_r);
	uint32_t kd_c = (k_c*d_c);
	const uint32_t mat_size = t_c * t_r;
	for(uint32_t ba = 0; ba < batches; ++ba){
		uint32_t batch_add = (ba * channels * mat_size);
		for(int32_t r = 0; r < upper_row2; ++r){
			for(int32_t c = 0; c < upper_col2; ++c){
				for(uint32_t ch = 0; ch < channels; ++ch){
					uint32_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					uint32_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int32_t rr = 0; rr < kd_r; rr += d_r){
						if((r - p_r + rr) < 0){
							if((r - p_r + rr + d_r) > 0){
								a_current += t_c * (r - p_r + rr + d_r);
							}
							b_begin += k_c;
							continue;
						}
						if((r - p_r + rr) >= t_r){
							b_begin += k_c * ((k_r - (rr/d_r)));
							break;
						}
						uint32_t pad_counter = 0;
						for(int32_t cc = 0; cc < kd_c; cc += d_c, ++b_begin){
							if((c - p_c + cc) < t_c && (c - p_c + cc) >= 0){
								*b_begin = *a_current;
								a_current += d_c;
								pad_counter += d_c;
							}
							else if((c - p_c + cc + d_c) > 0 && (c - p_c + cc + d_c) < t_c){
								a_current += (c - p_c + cc + d_c);
								pad_counter += (c - p_c + cc + d_c);
							}
						}
						a_current += a_current_add - pad_counter;
					}
				}
			}
		}
	}
};

inline static constexpr auto im2col_nn_layer_nd_pad_stride = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, int32_t p_r, int32_t p_c, int32_t t_r, int32_t t_c, const uint32_t& LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	uint32_t a_current_add = (t_c * d_r); 
	uint32_t upper_row2 = t_r - ((k_r-1)) + (p_r * 2);
	uint32_t upper_col2 = t_c - ((k_c-1)) + (p_c * 2);
	uint32_t kd_r = (k_r);
	uint32_t kd_c = (k_c);
	const uint32_t mat_size = t_c * t_r;
	for(uint32_t ba = 0; ba < batches; ++ba){
		uint32_t batch_add = (ba * channels * mat_size);
		for(int32_t r = 0; r < upper_row2; r += s_r){
			for(int32_t c = 0; c < upper_col2; c += s_c){
				for(uint32_t ch = 0; ch < channels; ++ch){
					uint32_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					uint32_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int32_t rr = 0; rr < kd_r; ++rr){
						if((r - p_r + rr) < 0){
							b_begin += k_c;
							continue;
						}
						if((r - p_r + rr) >= t_r){
							b_begin += k_c * (k_r - rr);
							break;
						}
						uint32_t pad_counter = 0;
						for(int32_t cc = 0; cc < kd_c; ++cc, ++b_begin){
							if((c - p_c + cc) < t_c && (c - p_c + cc) >= 0){
								*b_begin = *a_current;
								++a_current;
								++pad_counter;
							}
						}
						a_current += a_current_add - pad_counter;
					}
				}
			}
		}
	}
};

inline static constexpr auto im2col_nn_layer_nd_pad_none = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, int32_t p_r, int32_t p_c, int32_t t_r, int32_t t_c, const uint32_t& LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	uint32_t a_current_add = (t_c * d_r); 
	uint32_t upper_row2 = t_r - ((k_r-1)) + (p_r * 2);
	uint32_t upper_col2 = t_c - ((k_c-1)) + (p_c * 2);
	uint32_t kd_r = (k_r);
	uint32_t kd_c = (k_c);
	const uint32_t mat_size = t_c * t_r;
	for(uint32_t ba = 0; ba < batches; ++ba){
		uint32_t batch_add = (ba * channels * mat_size);
		for(int32_t r = 0; r < upper_row2; ++r){
			for(int32_t c = 0; c < upper_col2; ++c){
				for(uint32_t ch = 0; ch < channels; ++ch){
					uint32_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					uint32_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int32_t rr = 0; rr < kd_r; ++rr){
						if((r - p_r + rr) < 0){
							b_begin += k_c;
							continue;
						}
						if((r - p_r + rr) >= t_r){
							b_begin += k_c * (k_r - rr);
							break;
						}
						uint32_t pad_counter = 0;
						for(int32_t cc = 0; cc < kd_c; ++cc, ++b_begin){
							if((c - p_c + cc) < t_c && (c - p_c + cc) >= 0){
								*b_begin = *a_current;
								++a_current;
								++pad_counter;
							}
						}
						a_current += a_current_add - pad_counter;
					}
				}
			}
		}
	}
};



#else

inline static constexpr auto im2col_nn_layer_nd_pad = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, int32_t d_r, const uint32_t& d_c, int32_t p_r, int32_t p_c, int32_t t_r, int32_t t_c, const uint32_t& LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	uint32_t a_current_add = (t_c * d_r); 
	uint32_t row_max = t_r - ((k_r-1) * d_r) + (p_r * 2);
	uint32_t col_max = t_c - ((k_c-1) * d_c) + (p_c * 2);
	uint32_t kd_r = (k_r*d_r);
	uint32_t kd_c = (k_c*d_c);
	const uint32_t mat_size = t_c * t_r;
	for(uint32_t ba = 0; ba < batches; ++ba){
		uint32_t batch_add = (ba * channels * mat_size);
		tbb::parallel_for(tbb::blocked_range3d<uint32_t>(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<uint32_t> range){
		uint32_t row_begin = range.rows().begin() % s_r == 0 ? range.rows().begin() : range.rows().begin() + range.rows().begin() % s_r;
		uint32_t col_begin = range.cols().begin() % s_c == 0 ? range.cols().begin() : range.cols().begin() + range.cols().begin() % s_c;
		uint32_t m_begin_index = ba * L * LKern * channels; //batches 
		m_begin_index += ((row_begin) * col_max * LKern * channels) / (s_r * s_c); //rows;
		m_begin_index += ((col_begin) * LKern * channels) / s_c;//cols
		m_begin_index += range.pages().begin() * LKern;
		auto m_begin = b_begin + m_begin_index;
		for(int32_t r = row_begin; r < range.rows().end(); r += s_r){
			for(int32_t c = col_begin; c < range.cols().end(); c += s_c){
				for(uint32_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					uint32_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					uint32_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int32_t rr = 0; rr < kd_r; rr += d_r){
						if((r - p_r + rr) < 0){
							if((r - p_r + rr + d_r) > 0){
								a_current += t_c * (r - p_r + rr + d_r);
							}
							m_begin += k_c;
							continue;
						}
						if((r - p_r + rr) >= t_r){
							m_begin += k_c * ((k_r - (rr/d_r)));
							break;
						}
						uint32_t pad_counter = 0;
						for(int32_t cc = 0; cc < kd_c; cc += d_c, ++m_begin){
							if((c - p_c + cc) < t_c && (c - p_c + cc) >= 0){
								*m_begin = *a_current;
								a_current += d_c;
								pad_counter += d_c;
							}
							else if((c - p_c + cc + d_c) > 0 && (c - p_c + cc + d_c) < t_c){
								a_current += (c - p_c + cc + d_c);
								pad_counter += (c - p_c + cc + d_c);
							}
						}
						a_current += a_current_add - pad_counter;
					}
				}
			}
		}
		});
	}
};

inline static constexpr auto im2col_nn_layer_nd_pad_stride = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, int32_t p_r, int32_t p_c, int32_t t_r, int32_t t_c, const uint32_t& LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	uint32_t a_current_add = (t_c); 
	uint32_t row_max = t_r - ((k_r-1)) + (p_r * 2);
	uint32_t col_max = t_c - ((k_c-1)) + (p_c * 2);
	uint32_t kd_r = (k_r);
	uint32_t kd_c = (k_c);
	const uint32_t mat_size = t_c * t_r;
	for(uint32_t ba = 0; ba < batches; ++ba){
		uint32_t batch_add = (ba * channels * mat_size);
		tbb::parallel_for(tbb::blocked_range3d<uint32_t>(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<uint32_t> range){
		uint32_t row_begin = range.rows().begin() % s_r == 0 ? range.rows().begin() : range.rows().begin() + range.rows().begin() % s_r;
		uint32_t col_begin = range.cols().begin() % s_c == 0 ? range.cols().begin() : range.cols().begin() + range.cols().begin() % s_c;
		uint32_t m_begin_index = ba * L * LKern * channels; //batches 
		m_begin_index += ((row_begin) * col_max * LKern * channels) / (s_r * s_c); //rows;
		m_begin_index += ((col_begin) * LKern * channels) / s_c;//cols
		m_begin_index += range.pages().begin() * LKern;
		auto m_begin = b_begin + m_begin_index;
		for(int32_t r = row_begin; r < range.rows().end(); r += s_r){
			for(int32_t c = col_begin; c < range.cols().end(); c += s_c){
				for(uint32_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					uint32_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					uint32_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int32_t rr = 0; rr < kd_r; ++rr){
						if((r - p_r + rr) < 0){
							m_begin += k_c;
							continue;
						}
						if((r - p_r + rr) >= t_r){
							m_begin += k_c * (k_r - rr);
							break;
						}
						uint32_t pad_counter = 0;
						for(int32_t cc = 0; cc < kd_c; ++cc, ++m_begin){
							if((c - p_c + cc) < t_c && (c - p_c + cc) >= 0){
								*m_begin = *a_current;
								++a_current;
								++pad_counter;
							}
						}
						a_current += a_current_add - pad_counter;
					}
				}
			}
		}
		});
	}
};

inline static constexpr auto im2col_nn_layer_nd_pad_dilation = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, int32_t d_r, const uint32_t& d_c, int32_t p_r, int32_t p_c, int32_t t_r, int32_t t_c, const uint32_t& LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	uint32_t a_current_add = (t_c * d_r); 
	uint32_t row_max = t_r - ((k_r-1) * d_r) + (p_r * 2);
	uint32_t col_max = t_c - ((k_c-1) * d_c) + (p_c * 2);
	uint32_t kd_r = (k_r*d_r);
	uint32_t kd_c = (k_c*d_c);
	const uint32_t mat_size = t_c * t_r;
	for(uint32_t ba = 0; ba < batches; ++ba){
		uint32_t batch_add = (ba * channels * mat_size);
		tbb::parallel_for(tbb::blocked_range3d<uint32_t>(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<uint32_t> range){
		uint32_t m_begin_index = ba * L * LKern * channels; //batches 
		m_begin_index += (range.rows().begin() * col_max * LKern * channels); //rows;
		m_begin_index += (range.cols().begin() * LKern * channels);//cols
		m_begin_index += range.pages().begin() * LKern;
		auto m_begin = b_begin + m_begin_index;
		for(int32_t r = range.rows().begin(); r < range.rows().end(); ++r){
			for(int32_t c = range.cols().begin(); c < range.cols().end(); ++c){
				for(uint32_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					uint32_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					uint32_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int32_t rr = 0; rr < kd_r; rr += d_r){
						if((r - p_r + rr) < 0){
							if((r - p_r + rr + d_r) > 0){
								a_current += t_c * (r - p_r + rr + d_r);
							}
							m_begin += k_c;
							continue;
						}
						if((r - p_r + rr) >= t_r){
							m_begin += k_c * ((k_r - (rr/d_r)));
							break;
						}
						uint32_t pad_counter = 0;
						for(int32_t cc = 0; cc < kd_c; cc += d_c, ++m_begin){
							if((c - p_c + cc) < t_c && (c - p_c + cc) >= 0){
								*m_begin = *a_current;
								a_current += d_c;
								pad_counter += d_c;
							}
							else if((c - p_c + cc + d_c) > 0 && (c - p_c + cc + d_c) < t_c){
								a_current += (c - p_c + cc + d_c);
								pad_counter += (c - p_c + cc + d_c);
							}
						}
						a_current += a_current_add - pad_counter;
					}
				}
			}
		}
		});
	}
};


inline static constexpr auto im2col_nn_layer_nd_pad_none = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, int32_t p_r, int32_t p_c, int32_t t_r, int32_t t_c, const uint32_t& LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	uint32_t a_current_add = (t_c); 
	uint32_t row_max = t_r - ((k_r-1)) + (p_r * 2);
	uint32_t col_max = t_c - ((k_c-1)) + (p_c * 2);
	uint32_t kd_r = (k_r);
	uint32_t kd_c = (k_c);
	const uint32_t mat_size = t_c * t_r;
	for(uint32_t ba = 0; ba < batches; ++ba){
		uint32_t batch_add = (ba * channels * mat_size);
		tbb::parallel_for(tbb::blocked_range3d<uint32_t>(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<uint32_t> range){
		uint32_t m_begin_index = ba * L * LKern * channels; //batches 
		m_begin_index += (range.rows().begin() * col_max * LKern * channels); //rows;
		m_begin_index += (range.cols().begin() * LKern * channels);//cols
		m_begin_index += range.pages().begin() * LKern;
		auto m_begin = b_begin + m_begin_index;
		for(int32_t r = range.rows().begin(); r < range.rows().end(); ++r){
			for(int32_t c = range.cols().begin(); c < range.cols().end(); ++c){
				for(uint32_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					uint32_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					uint32_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int32_t rr = 0; rr < kd_r; ++rr){
						if((r - p_r + rr) < 0){
							m_begin += k_c;
							continue;
						}
						if((r - p_r + rr) >= t_r){
							m_begin += k_c * (k_r - rr);
							break;
						}
						uint32_t pad_counter = 0;
						for(int32_t cc = 0; cc < kd_c; ++cc, ++m_begin){
							if((c - p_c + cc) < t_c && (c - p_c + cc) >= 0){
								*m_begin = *a_current;
								++a_current;
								++pad_counter;
							}
						}
						a_current += a_current_add - pad_counter;
					}
				}
			}
		}
		});
	}
};


#endif

#ifndef USE_PARALLEL

inline static constexpr auto im2col_nn_layer_nd_pad_backward = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, int32_t d_r, const uint32_t& d_c, int32_t p_r, int32_t p_c, int32_t t_r, int32_t t_c, const uint32_t& LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	uint32_t a_current_add = (t_c * d_r); 
	uint32_t upper_row2 = t_r - ((k_r-1) * d_r) + (p_r * 2);
	uint32_t upper_col2 = t_c - ((k_c-1) * d_c) + (p_c * 2);
	uint32_t kd_r = (k_r*d_r);
	uint32_t kd_c = (k_c*d_c);
	const uint32_t mat_size = t_c * t_r;
	for(uint32_t ba = 0; ba < batches; ++ba){
		uint32_t batch_add = (ba * channels * mat_size);
		for(int32_t r = 0; r < upper_row2; r += s_r){
			for(int32_t c = 0; c < upper_col2; c += s_r){
				for(uint32_t ch = 0; ch < channels; ++ch){
					uint32_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					uint32_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int32_t rr = 0; rr < kd_r; rr += d_r){
						if((r - p_r + rr) < 0){
							if((r - p_r + rr + d_r) > 0){
								a_current += t_c * (r - p_r + rr + d_r);
							}
							b_begin += k_c;
							continue;
						}
						if((r - p_r + rr) >= t_r){
							b_begin += k_c * ((k_r - (rr/d_r)));
							break;
						}
						uint32_t pad_counter = 0;
						for(int32_t cc = 0; cc < kd_c; cc += d_c, ++b_begin){
							if((c - p_c + cc) < t_c && (c - p_c + cc) >= 0){
								*a_current = *b_begin;
								a_current += d_c;
								pad_counter += d_c;
							}
							else if((c - p_c + cc + d_c) > 0 && (c - p_c + cc + d_c) < t_c){
								a_current += (c - p_c + cc + d_c);
								pad_counter += (c - p_c + cc + d_c);
							}
						}
						a_current += a_current_add - pad_counter;
					}
				}
			}
		}
	}
};

#else

inline static constexpr auto im2col_nn_layer_nd_pad_backward = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, int32_t d_r, const uint32_t& d_c, int32_t p_r, int32_t p_c, int32_t t_r, int32_t t_c, const uint32_t& LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	uint32_t a_current_add = (t_c * d_r); 
	uint32_t row_max = t_r - ((k_r-1) * d_r) + (p_r * 2);
	uint32_t col_max = t_c - ((k_c-1) * d_c) + (p_c * 2);
	uint32_t kd_r = (k_r*d_r);
	uint32_t kd_c = (k_c*d_c);
	const uint32_t mat_size = t_c * t_r;
	for(uint32_t ba = 0; ba < batches; ++ba){
		uint32_t batch_add = (ba * channels * mat_size);
		tbb::parallel_for(tbb::blocked_range3d<uint32_t>(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<uint32_t> range){
		uint32_t row_begin = range.rows().begin() % s_r == 0 ? range.rows().begin() : range.rows().begin() + range.rows().begin() % s_r;
		uint32_t col_begin = range.cols().begin() % s_c == 0 ? range.cols().begin() : range.cols().begin() + range.cols().begin() % s_c;
		uint32_t m_begin_index = ba * L * LKern * channels; //batches 
		m_begin_index += ((row_begin) * col_max * LKern * channels) / (s_r * s_c); //rows;
		m_begin_index += ((col_begin) * LKern * channels) / s_c;//cols
		m_begin_index += range.pages().begin() * LKern;
		auto m_begin = b_begin + m_begin_index;
		for(int32_t r = row_begin; r < range.rows().end(); r += s_r){
			for(int32_t c = col_begin; c < range.cols().end(); c += s_r){
				for(uint32_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					uint32_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					uint32_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int32_t rr = 0; rr < kd_r; rr += d_r){
						if((r - p_r + rr) < 0){
							if((r - p_r + rr + d_r) > 0){
								a_current += t_c * (r - p_r + rr + d_r);
							}
							m_begin += k_c;
							continue;
						}
						if((r - p_r + rr) >= t_r){
							m_begin += k_c * ((k_r - (rr/d_r)));
							break;
						}
						uint32_t pad_counter = 0;
						for(int32_t cc = 0; cc < kd_c; cc += d_c, ++m_begin){
							if((c - p_c + cc) < t_c && (c - p_c + cc) >= 0){
								*a_current = *m_begin;
								a_current += d_c;
								pad_counter += d_c;
							}
							else if((c - p_c + cc + d_c) > 0 && (c - p_c + cc + d_c) < t_c){
								a_current += (c - p_c + cc + d_c);
								pad_counter += (c - p_c + cc + d_c);
							}
						}
						a_current += a_current_add - pad_counter;
					}
				}
			}
		}
		});
	}
};

#endif



//normally it would be an LxLKern
//now it goes into is that it would be (row * LKern + col)
//for a transpose would output a matrix of LKernxL
// a normal transpose algorithm looks like the following:
/*
Tensor<T> output(_cols, _rows);
for(uint32_t x = 0; x < _rows; ++x){
	for(uint32_t y = 0; y < _cols; ++y)
		output.at(y,x) = at(x,y);
}
or:
for(uint32_t x = 0; x < _rows; ++x){
	for(uint32_t y = 0; y < _cols; ++y){
		*(o + (y * _rows +x)) = *(t + (x * _cols + y))
	}
}


 */

class im2col_nn_layer_transpose_index{
	uint32_t index, row, col, LKern, L; //originally _cols = LKern, _rows = L
	uint32_t adding;
	public:
		im2col_nn_layer_transpose_index(uint32_t lkern, uint32_t l)
			:index(0), row(0), col(0), LKern(lkern), L(l), adding(0)
		{}
		const uint32_t& operator*(){return index;}
		void operator++(){
			if(col == (LKern-1)){
				col = 0;
				++row;
				index = row;
				return;
			}
			++col;
			index += L;
		}
		void operator+=(const uint32_t& c){
			if((col + c) >= LKern){
				row += (col + c) / LKern;
				col = (col + c) % LKern;
				index = (col * L) + row + adding;
				return;
			}
			col += c;
			index += (L * c);
		}
		im2col_nn_layer_transpose_index operator+(const uint32_t& c) const{
			im2col_nn_layer_transpose_index outp(LKern, L);
			outp.adding = adding;
			if((col + c) >= LKern){
				outp.row = row + (col + c) / LKern;
				outp.col = (col + c) % LKern;
				outp.index = (outp.col * L) + outp.row + adding; 
				return outp;
			}
			outp.row = row;
			outp.col = col + c;
			outp.index = index + (L * c);
			return outp;
		}

		im2col_nn_layer_transpose_index& zero(){
			index = adding;
			row = 0;
			col = 0;
			return *this;
		}
		void set_adding(const uint32_t& c){
			adding = c;
		}

};

inline static constexpr auto im2col_nn_layer_2d_T = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, const uint32_t& d_r, const uint32_t& d_c, const uint32_t& t_r, const uint32_t& t_c, const uint32_t& LKern, const uint32_t& L){
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	uint32_t a_current_add = ((t_c * d_r) - (k_c/d_c)); 
	for(uint32_t r = 0; r < t_r - (k_r-1); r += s_r){
		for(uint32_t c = 0; c < t_c - (k_c-1); c += s_r){
			auto a_current = a_begin + (r * t_c + c);
			for(uint32_t rr = 0; rr < k_r; rr += d_r, a_current += a_current_add){
				for(uint32_t cc = 0; cc < k_c; cc += d_c, a_current += d_c, ++b_indexer){
					*(b_begin + *b_indexer) = *a_current;
				}
			}
		}
	}
};

inline static constexpr auto im2col_nn_layer_2d_T_backward = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, const uint32_t& d_r, const uint32_t& d_c, const uint32_t& t_r, const uint32_t& t_c, const uint32_t& LKern, const uint32_t& L){
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	uint32_t a_current_add = ((t_c * d_r) - (k_c/d_c)); 
	for(uint32_t r = 0; r < t_r - (k_r-1); r += s_r){
		for(uint32_t c = 0; c < t_c - (k_c-1); c += s_r){
			auto a_current = a_begin + (r * t_c + c);
			for(uint32_t rr = 0; rr < k_r; rr += d_r, a_current += a_current_add){
				for(uint32_t cc = 0; cc < k_c; cc += d_c, a_current += d_c, ++b_indexer){
					*a_current = *(b_begin + *b_indexer);
				}
			}
		}
	}
};

#ifndef USE_PARALLEL
inline static constexpr auto im2col_nn_layer_nd_T = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, const uint32_t& d_r, const uint32_t& d_c, const uint32_t& t_r, const uint32_t& t_c, const uint32_t LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	uint32_t a_current_add = ((t_c * d_r) - (k_c*d_c));
	const uint32_t mat_size = t_c * t_r;
	const uint32_t upper_rows = t_r - ((k_r-1) * d_r);
	const uint32_t upper_cols = t_c - ((k_c-1) * d_c);
	const uint32_t upper_rows2 = k_r*d_r;
	const uint32_t upper_cols2 = k_c * d_c;
	uint32_t adding = 0;
	for(uint32_t ba = 0; ba < batches; ++ba, b_indexer.zero()){
		adding = (ba * L * LKern);
		uint32_t batch_add = (ba * channels * mat_size);
		for(uint32_t r = 0; r < upper_rows; r += s_r){
			for(uint32_t c = 0; c <upper_cols; c += s_c){
				for(uint32_t ch = 0; ch < channels; ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(uint32_t rr = 0; rr < upper_rows2; rr += d_r, a_current += a_current_add){
						for(uint32_t cc = 0; cc < upper_cols2; cc += d_c, a_current += d_c, ++b_indexer){
							*(b_begin + *b_indexer + adding) = *a_current;
						}
					}
				}
			}
		}
	}
};

inline static constexpr auto im2col_nn_layer_nd_T_stride = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, const uint32_t& t_r, const uint32_t& t_c, const uint32_t LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	uint32_t a_current_add = (t_c - k_c);
	const uint32_t mat_size = t_c * t_r;
	const uint32_t upper_rows = t_r - (k_r-1);
	const uint32_t upper_cols = t_c - (k_c-1);
	const uint32_t upper_rows2 = k_r;
	const uint32_t upper_cols2 = k_c;
	uint32_t adding = 0;
	for(uint32_t ba = 0; ba < batches; ++ba, b_indexer.zero()){
		adding = (ba * L * LKern);
		uint32_t batch_add = (ba * channels * mat_size);
		for(uint32_t r = 0; r < upper_rows; r += s_r){
			for(uint32_t c = 0; c <upper_cols; c += s_c){
				for(uint32_t ch = 0; ch < channels; ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(uint32_t rr = 0; rr < upper_rows2; ++rr, a_current += a_current_add){
						for(uint32_t cc = 0; cc < upper_cols2; ++cc, ++a_current, ++b_indexer){
							*(b_begin + *b_indexer + adding) = *a_current;
						}
					}
				}
			}
		}
	}
};

inline static constexpr auto im2col_nn_layer_nd_T_dilation = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& d_r, const uint32_t& d_c, const uint32_t& t_r, const uint32_t& t_c, const uint32_t LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	uint32_t a_current_add = ((t_c * d_r) - (k_c*d_c));
	const uint32_t mat_size = t_c * t_r;
	const uint32_t upper_rows = t_r - ((k_r-1) * d_r);
	const uint32_t upper_cols = t_c - ((k_c-1) * d_c);
	const uint32_t upper_rows2 = k_r*d_r;
	const uint32_t upper_cols2 = k_c * d_c;
	uint32_t adding = 0;
	for(uint32_t ba = 0; ba < batches; ++ba, b_indexer.zero()){
		adding = (ba * L * LKern);
		uint32_t batch_add = (ba * channels * mat_size);
		for(uint32_t r = 0; r < upper_rows; ++r){
			for(uint32_t c = 0; c <upper_cols; ++c){
				for(uint32_t ch = 0; ch < channels; ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(uint32_t rr = 0; rr < upper_rows2; rr += d_r, a_current += a_current_add){
						for(uint32_t cc = 0; cc < upper_cols2; cc += d_c, a_current += d_c, ++b_indexer){
							*(b_begin + *b_indexer + adding) = *a_current;
						}
					}
				}
			}
		}
	}
};


inline static constexpr auto im2col_nn_layer_nd_T_none = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& t_r, const uint32_t& t_c, const uint32_t LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	uint32_t a_current_add = (t_c - k_c);
	const uint32_t mat_size = t_c * t_r;
	const uint32_t upper_rows = t_r - (k_r-1);
	const uint32_t upper_cols = t_c - (k_c-1);
	const uint32_t upper_rows2 = k_r;
	const uint32_t upper_cols2 = k_c;
	uint32_t adding = 0;
	for(uint32_t ba = 0; ba < batches; ++ba, b_indexer.zero()){
		adding = (ba * L * LKern);
		uint32_t batch_add = (ba * channels * mat_size);
		for(uint32_t r = 0; r < upper_rows; ++r){
			for(uint32_t c = 0; c <upper_cols; ++c){
				for(uint32_t ch = 0; ch < channels; ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(uint32_t rr = 0; rr < upper_rows2; ++rr, a_current += a_current_add){
						for(uint32_t cc = 0; cc < upper_cols2; ++cc, ++a_current, ++b_indexer){
							*(b_begin + *b_indexer + adding) = *a_current;
						}
					}
				}
			}
		}
	}
};

#else

inline static constexpr auto im2col_nn_layer_nd_T = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, const uint32_t& d_r, const uint32_t& d_c, const uint32_t& t_r, const uint32_t& t_c, const uint32_t LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	uint32_t row_max = (t_r - ((k_r-1) * d_r));
	uint32_t col_max = t_c - ((k_c-1) * d_c);
	uint32_t a_current_add = ((t_c * d_r) - (k_c*d_c));
	const uint32_t mat_size = t_c * t_r;
	uint32_t adding = 0;
	for(uint32_t ba = 0; ba < batches; ++ba, b_indexer.zero()){
		adding = (ba * L * LKern);
		uint32_t batch_add = (ba * channels * mat_size);
		tbb::parallel_for(tbb::blocked_range3d<uint32_t>(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<uint32_t> range){
		uint32_t row_begin = range.rows().begin() % s_r == 0 ? range.rows().begin() : range.rows().begin() + range.rows().begin() % s_r;
		uint32_t col_begin = range.cols().begin() % s_c == 0 ? range.cols().begin() : range.cols().begin() + range.cols().begin() % s_c;
		uint32_t m_index = ((row_begin) * col_max * LKern) / (s_r * s_c); // rows
		m_index += ((col_begin) * LKern) / s_c;//cols
		m_index += range.pages().begin() * (LKern/channels);
		im2col_nn_layer_transpose_index m_begin_index = b_indexer + m_index;  
		for(uint32_t r = row_begin; r < range.rows().end(); r += s_r){
			for(uint32_t c = col_begin; c < range.cols().end(); c += s_c){
				for(uint32_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(uint32_t rr = 0; rr < k_r*d_r; rr += d_r, a_current += a_current_add){
						for(uint32_t cc = 0; cc < k_c*d_c; cc += d_c, a_current += d_c, ++m_begin_index){
							*(b_begin + adding + *m_begin_index) = *a_current;
						}
					}
				}
			}
		}
		});
	}
};


#endif


#ifndef USE_PARALLEL
inline static constexpr auto im2col_nn_layer_nd_T_backward = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, const uint32_t& d_r, const uint32_t& d_c, const uint32_t& t_r, const uint32_t& t_c, const uint32_t LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	uint32_t a_current_add = ((t_c * d_r) - (k_c*d_c));
	const uint32_t mat_size = t_c * t_r;
	uint32_t adding = 0;
	for(uint32_t ba = 0; ba < batches; ++ba, b_indexer.zero()){
		adding = (ba * L * LKern);
		uint32_t batch_add = (ba * channels * mat_size);
		for(uint32_t r = 0; r < t_r - ((k_r-1) * d_r); r += s_r){
			for(uint32_t c = 0; c < t_c - ((k_c-1) * d_c); c += s_c){
				for(uint32_t ch = 0; ch < channels; ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(uint32_t rr = 0; rr < k_r*d_r; rr += d_r, a_current += a_current_add){
						for(uint32_t cc = 0; cc < k_c*d_c; cc += d_c, a_current += d_c, ++b_indexer){
							*a_current = *(b_begin + adding + *m_begin_index) ;
						}
					}
				}
			}
		}
	}
};

#else

inline static constexpr auto im2col_nn_layer_nd_T_backward = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, const uint32_t& d_r, const uint32_t& d_c, const uint32_t& t_r, const uint32_t& t_c, const uint32_t LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	uint32_t row_max = (t_r - ((k_r-1) * d_r));
	uint32_t col_max = t_c - ((k_c-1) * d_c);
	uint32_t a_current_add = ((t_c * d_r) - (k_c*d_c));
	const uint32_t mat_size = t_c * t_r;
	uint32_t adding = 0;
	for(uint32_t ba = 0; ba < batches; ++ba, b_indexer.zero()){
		adding = (ba * L * LKern);
		uint32_t batch_add = (ba * channels * mat_size);
		tbb::parallel_for(tbb::blocked_range3d<uint32_t>(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<uint32_t> range){
		uint32_t row_begin = range.rows().begin() % s_r == 0 ? range.rows().begin() : range.rows().begin() + range.rows().begin() % s_r;
		uint32_t col_begin = range.cols().begin() % s_c == 0 ? range.cols().begin() : range.cols().begin() + range.cols().begin() % s_c;
		uint32_t m_index = ((row_begin) * col_max * LKern) / (s_r * s_c); // rows
		m_index += ((col_begin) * LKern) / s_c;//cols
		m_index += range.pages().begin() * (LKern/channels);
		im2col_nn_layer_transpose_index m_begin_index = b_indexer + m_index;  
		for(uint32_t r = row_begin; r < range.rows().end(); r += s_r){
			for(uint32_t c = col_begin; c < range.cols().end(); c += s_c){
				for(uint32_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(uint32_t rr = 0; rr < k_r*d_r; rr += d_r, a_current += a_current_add){
						for(uint32_t cc = 0; cc < k_c*d_c; cc += d_c, a_current += d_c, ++m_begin_index){
							*a_current = *(b_begin + adding + *m_begin_index) ;
						}
					}
				}
			}
		}
		});
	}
};


#endif


//%s/const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, int32_t d_r, const uint32_t& d_c/const uint32_t\& k_r, const uint32_t\& k_c, const uint32_t\& s_r, const uint32_t\& s_c, int32_t d_r, const uint32_t\& d_c
inline static constexpr auto im2col_nn_layer_2d_T_pad = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, int32_t d_r, const uint32_t& d_c, int32_t p_r, int32_t p_c, int32_t t_r, int32_t t_c, const uint32_t& LKern, const uint32_t& L){
	uint32_t a_current_add = (t_c * d_r); 
	uint32_t upper_row2 = t_r - (k_r-1) + (p_r * 2);
	uint32_t upper_col2 = t_c - (k_c-1) + (p_c * 2);
	uint32_t kd_r = (k_r/d_r);
	uint32_t kd_c = (k_c/d_c);
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	for(int32_t r = 0; r < upper_row2; r += s_r){
		for(int32_t c = 0; c < upper_col2; c += s_r){
			uint32_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
			uint32_t c_add = c - p_c <= 0 ? 0 : c - p_c;
			auto a_current = a_begin + r_add + c_add;
			for(int32_t rr = 0; rr < k_r; rr += d_r){
				if((r - p_r + rr) < 0){
					if((r - p_r + rr + d_r) > 0){
						a_current += t_c * (r - p_r + rr + d_r);
					}
					b_indexer += kd_c;
					continue;
				}
				if((r - p_r + rr) >= t_r){
					b_indexer += kd_c * ((kd_r - (rr+d_r))+1);
					break;
				}
				uint32_t pad_counter = 0;
				for(int32_t cc = 0; cc < k_c; cc += d_c, ++b_indexer){
					if((c - p_c + cc) < t_c && (c - p_c + cc) >= 0){
						*(b_begin + *b_indexer) = *a_current;
						a_current += d_c;
						pad_counter += d_c;
					}
				}
				a_current += a_current_add - pad_counter;
			}
		}
	}
};


inline static constexpr auto im2col_nn_layer_2d_T_pad_backward = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, int32_t d_r, const uint32_t& d_c, int32_t p_r, int32_t p_c, int32_t t_r, int32_t t_c, const uint32_t& LKern, const uint32_t& L){
	uint32_t a_current_add = (t_c * d_r); 
	uint32_t upper_row2 = t_r - (k_r-1) + (p_r * 2);
	uint32_t upper_col2 = t_c - (k_c-1) + (p_c * 2);
	uint32_t kd_r = (k_r/d_r);
	uint32_t kd_c = (k_c/d_c);
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	for(int32_t r = 0; r < upper_row2; r += s_r){
		for(int32_t c = 0; c < upper_col2; c += s_r){
			uint32_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
			uint32_t c_add = c - p_c <= 0 ? 0 : c - p_c;
			auto a_current = a_begin + r_add + c_add;
			for(int32_t rr = 0; rr < k_r; rr += d_r){
				if((r - p_r + rr) < 0){
					if((r - p_r + rr + d_r) > 0){
						a_current += t_c * (r - p_r + rr + d_r);
					}
					b_indexer += kd_c;
					continue;
				}
				if((r - p_r + rr) >= t_r){
					b_indexer += kd_c * ((kd_r - (rr+d_r))+1);
					break;
				}
				uint32_t pad_counter = 0;
				for(int32_t cc = 0; cc < k_c; cc += d_c, ++b_indexer){
					if((c - p_c + cc) < t_c && (c - p_c + cc) >= 0){
						*a_current = *(b_begin + *b_indexer);
						a_current += d_c;
						pad_counter += d_c;
					}
				}
				a_current += a_current_add - pad_counter;
			}
		}
	}
};


#ifndef USE_PARALLEL
inline static constexpr auto im2col_nn_layer_nd_T_pad = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, int32_t d_r, const uint32_t& d_c, int32_t p_r, int32_t p_c, int32_t t_r, int32_t t_c, uint32_t LKern, uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	uint32_t a_current_add = (t_c * d_r); 
	uint32_t upper_row2 = t_r - ((k_r-1) * d_r) + (p_r * 2);
	uint32_t upper_col2 = t_c - ((k_c-1) * d_c) + (p_c * 2);
	uint32_t kd_r = (k_r*d_r);
	uint32_t kd_c = (k_c*d_c);
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	const uint32_t mat_size = t_c * t_r;
	uint32_t adding = 0;
	for(uint32_t ba = 0; ba < batches; ++ba, b_indexer.zero()){
		adding = (ba * L * LKern);
		uint32_t batch_add = ba * channels * mat_size;
		for(int32_t r = 0; r < upper_row2; r += s_r){
			for(int32_t c = 0; c < upper_col2; c += s_r){
				for(uint32_t ch = 0; ch < channels; ++ch){
					uint32_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					uint32_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int32_t rr = 0; rr < kd_r; rr += d_r){
						if((r - p_r + rr) < 0){
							if((r - p_r + rr + d_r) > 0){
								a_current += t_c * (r - p_r + rr + d_r);
							}
							b_indexer += k_c;
							continue;
						}
						if((r - p_r + rr) >= t_r){
							b_indexer += k_c * ((k_r - (rr/d_r)));
							break;
						}
						uint32_t pad_counter = 0;
						for(int32_t cc = 0; cc < kd_c; cc += d_c, ++b_indexer){
							if((c - p_c + cc) < t_c && (c - p_c + cc) >= 0){
								*(b_begin + *b_indexer + adding) = *a_current;
								a_current += d_c;
								pad_counter += d_c;
							}
							else if((c - p_c + cc + d_c) > 0 && (c - p_c + cc + d_c) < t_c){
								a_current += (c - p_c + cc + d_c);
								pad_counter += (c - p_c + cc + d_c);
							}
						}
						a_current += a_current_add - pad_counter;
					}
				}
			}
		}	
	}
};

#else

inline static constexpr auto im2col_nn_layer_nd_T_pad = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, int32_t d_r, const uint32_t& d_c, int32_t p_r, int32_t p_c, int32_t t_r, int32_t t_c, uint32_t LKern, uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	uint32_t a_current_add = (t_c * d_r); 
	uint32_t row_max = t_r - ((k_r-1) * d_r) + (p_r * 2);
	uint32_t col_max = t_c - ((k_c-1) * d_c) + (p_c * 2);
	uint32_t kd_r = (k_r*d_r);
	uint32_t kd_c = (k_c*d_c);
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	const uint32_t mat_size = t_c * t_r;
	uint32_t adding = 0;
	for(uint32_t ba = 0; ba < batches; ++ba, b_indexer.zero()){
		adding = (ba * L * LKern);
		uint32_t batch_add = ba * channels * mat_size;
		tbb::parallel_for(tbb::blocked_range3d<uint32_t>(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<uint32_t> range){
		uint32_t row_begin = range.rows().begin() % s_r == 0 ? range.rows().begin() : range.rows().begin() + range.rows().begin() % s_r;
		uint32_t col_begin = range.cols().begin() % s_c == 0 ? range.cols().begin() : range.cols().begin() + range.cols().begin() % s_c;
		uint32_t m_index = ((row_begin) * col_max * LKern) / (s_r * s_c); // rows
		m_index += ((col_begin) * LKern) / s_c;//cols
		m_index += range.pages().begin() * (LKern/channels);
		im2col_nn_layer_transpose_index m_begin_index = b_indexer + m_index;  
		for(int32_t r = row_begin; r < range.rows().end(); r += s_r){
			for(int32_t c = col_begin; c < range.cols().end(); c += s_r){
				for(uint32_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					uint32_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					uint32_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int32_t rr = 0; rr < kd_r; rr += d_r){
						if((r - p_r + rr) < 0){
							if((r - p_r + rr + d_r) > 0){
								a_current += t_c * (r - p_r + rr + d_r);
							}
							m_begin_index += k_c;
							continue;
						}
						if((r - p_r + rr) >= t_r){
							m_begin_index += k_c * ((k_r - (rr/d_r)));
							break;
						}
						uint32_t pad_counter = 0;
						for(int32_t cc = 0; cc < kd_c; cc += d_c, ++m_begin_index){
							if((c - p_c + cc) < t_c && (c - p_c + cc) >= 0){
								*(b_begin + *m_begin_index + adding) = *a_current;
								a_current += d_c;
								pad_counter += d_c;
							}
							else if((c - p_c + cc + d_c) > 0 && (c - p_c + cc + d_c) < t_c){
								a_current += (c - p_c + cc + d_c);
								pad_counter += (c - p_c + cc + d_c);
							}
						}
						a_current += a_current_add - pad_counter;
					}
				}
			}
		}});	
	}
};


#endif


#ifndef USE_PARALLEL
inline static constexpr auto im2col_nn_layer_nd_T_pad_backward = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, int32_t d_r, const uint32_t& d_c, int32_t p_r, int32_t p_c, int32_t t_r, int32_t t_c, const uint32_t LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	uint32_t a_current_add = (t_c * d_r); 
	uint32_t upper_row2 = t_r - ((k_r-1) * d_r) + (p_r * 2);
	uint32_t upper_col2 = t_c - ((k_c-1) * d_c) + (p_c * 2);
	uint32_t kd_r = (k_r*d_r);
	uint32_t kd_c = (k_c*d_c);
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	const uint32_t mat_size = t_c * t_r;
	uint32_t adding = 0;
	for(uint32_t ba = 0; ba < batches; ++ba, b_indexer.zero()){
		adding = (ba * L * LKern);
		uint32_t batch_add = ba * channels * mat_size;
		for(int32_t r = 0; r < upper_row2; r += s_r){
			for(int32_t c = 0; c < upper_col2; c += s_r){
				for(uint32_t ch = 0; ch < channels; ++ch){
					uint32_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					uint32_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int32_t rr = 0; rr < kd_r; rr += d_r){
						if((r - p_r + rr) < 0){
							if((r - p_r + rr + d_r) > 0){
								a_current += t_c * (r - p_r + rr + d_r);
							}
							b_indexer += k_c;
							continue;
						}
						if((r - p_r + rr) >= t_r){
							b_indexer += k_c * ((k_r - (rr/d_r)));
							break;
						}
						uint32_t pad_counter = 0;
						for(int32_t cc = 0; cc < kd_c; cc += d_c, ++b_indexer){
							if((c - p_c + cc) < t_c && (c - p_c + cc) >= 0){
								*a_current = *(b_begin + *b_indexer + adding);
								a_current += d_c;
								pad_counter += d_c;
							}
							else if((c - p_c + cc + d_c) > 0 && (c - p_c + cc + d_c) < t_c){
								a_current += (c - p_c + cc + d_c);
								pad_counter += (c - p_c + cc + d_c);
							}
						}
						a_current += a_current_add - pad_counter;
					}
				}
			}
		}	
	}
};

#else

inline static constexpr auto im2col_nn_layer_nd_T_pad_backward = [](auto a_begin, auto a_end, auto b_begin, const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, int32_t d_r, const uint32_t& d_c, int32_t p_r, int32_t p_c, int32_t t_r, int32_t t_c, const uint32_t LKern, const uint32_t& L, const uint32_t& channels, const uint32_t& batches){
	uint32_t a_current_add = (t_c * d_r); 
	uint32_t row_max = t_r - ((k_r-1) * d_r) + (p_r * 2);
	uint32_t col_max = t_c - ((k_c-1) * d_c) + (p_c * 2);
	uint32_t kd_r = (k_r*d_r);
	uint32_t kd_c = (k_c*d_c);
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	const uint32_t mat_size = t_c * t_r;
	uint32_t adding = 0;
	for(uint32_t ba = 0; ba < batches; ++ba, b_indexer.zero()){
		adding = (ba * L * LKern);
		uint32_t batch_add = ba * channels * mat_size;
		tbb::parallel_for(tbb::blocked_range3d<uint32_t>(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<uint32_t> range){
		uint32_t row_begin = range.rows().begin() % s_r == 0 ? range.rows().begin() : range.rows().begin() + range.rows().begin() % s_r;
		uint32_t col_begin = range.cols().begin() % s_c == 0 ? range.cols().begin() : range.cols().begin() + range.cols().begin() % s_c;
		uint32_t m_index = ((row_begin) * col_max * LKern) / (s_r * s_c); // rows
		m_index += ((col_begin) * LKern) / s_c;//cols
		m_index += range.pages().begin() * (LKern/channels);
		im2col_nn_layer_transpose_index m_begin_index = b_indexer + m_index;  
		for(int32_t r = row_begin; r < range.rows().end(); r += s_r){
			for(int32_t c = col_begin; c < range.cols().end(); c += s_r){
				for(uint32_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					uint32_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					uint32_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int32_t rr = 0; rr < kd_r; rr += d_r){
						if((r - p_r + rr) < 0){
							if((r - p_r + rr + d_r) > 0){
								a_current += t_c * (r - p_r + rr + d_r);
							}
							m_begin_index += k_c;
							continue;
						}
						if((r - p_r + rr) >= t_r){
							m_begin_index += k_c * ((k_r - (rr/d_r)));
							break;
						}
						uint32_t pad_counter = 0;
						for(int32_t cc = 0; cc < kd_c; cc += d_c, ++m_begin_index){
							if((c - p_c + cc) < t_c && (c - p_c + cc) >= 0){
								*a_current = *(b_begin + *b_indexer + adding);
								a_current += d_c;
								pad_counter += d_c;
							}
							else if((c - p_c + cc + d_c) > 0 && (c - p_c + cc + d_c) < t_c){
								a_current += (c - p_c + cc + d_c);
								pad_counter += (c - p_c + cc + d_c);
							}
						}
						a_current += a_current_add - pad_counter;
					}
				}
			}
		}});	
	}
};


#endif


/* Tensor im2col_2_dim */

//im2col_nn_layer_2d_T function: 142
//im2col_nn_layer_2d + RowColSwap_contiguous: 221
//im2col_nn_layer_2d + transpose(-1,-2): 180
//im2col_nn_layer_2d + RowColSwap: 168
//clearly the im2col_nn_layer_2d_T function is most efficient 

Tensor Unfold::forward(const Tensor& x){
	utils::throw_exception(x.dims() >= 2, "Expected dimensions of Tensor to unfold to be greater than 1 but got $", x.dims());
	int32_t L_r = ((x.shape()[-1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
	int32_t L_c = ((x.shape()[-2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1; 
	uint32_t L = L_r * L_c;
	utils::throw_exception((L_r*L_c) > 0, "Given input with spatial size ($, $), kernel_size=($, $), dilation=($, $), padding=($, $), calculated shape of the array of sliding blocks as ($, $), but its components must be at least one.", x.shape()[-2], x.shape()[-1], kernel_size[0], kernel_size[1], dilation[0], dilation[1], padding[0], padding[1], L_r, L_c); 
	if(x.dims() > 2){
		Tensor X = (x.dims() == 3) ? x.view({1, x.shape()[0], x.shape()[1], x.shape()[2]}) : x.flatten(0, -4);
		if(!(padding == 0)){
			if(out_transpose){
				SizeRef outp_shape_ut({x.shape()[0], X.shape()[1]*LKern, L});
				Tensor output(outp_shape_ut, x.dtype);
				X.arr_void().execute_function(im2col_nn_layer_nd_T_pad, output.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0], dilation[1], padding[0], padding[1], x.shape()[-2], x.shape()[-1], X.shape()[1]*LKern, L, X.shape()[1], X.shape()[0]);
				return std::move(output);
			}

			SizeRef outp_shape_ut({X.shape()[0], L, X.shape()[1]*LKern});
			Tensor output(outp_shape_ut, x.dtype);
			if(stride == 1 && dilation == 1){
				X.arr_void().execute_function(im2col_nn_layer_nd_pad_none, output.arr_void(), kernel_size[0], kernel_size[1], padding[0], padding[1], x.shape()[-2], x.shape()[-1], LKern, L, X.shape()[1], X.shape()[0]);
			}
			else if(stride == 1 && !(dilation == 1)){
				X.arr_void().execute_function(im2col_nn_layer_nd_pad_dilation, output.arr_void(), kernel_size[0], kernel_size[1], dilation[0], dilation[1], padding[0], padding[1], x.shape()[-2], x.shape()[-1], LKern, L, X.shape()[1], X.shape()[0]);
			}
			else if(!(stride == 1) && dilation == 1){
				X.arr_void().execute_function(im2col_nn_layer_nd_pad_stride, output.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], padding[0], padding[1], x.shape()[-2], x.shape()[-1], LKern, L, X.shape()[1], X.shape()[0]);
			}
			else{
				X.arr_void().execute_function(im2col_nn_layer_nd_pad, output.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0], dilation[1], padding[0], padding[1], x.shape()[-2], x.shape()[-1], LKern, L, X.shape()[1], X.shape()[0]);
			}
			return std::move(output);	
		}
		if(out_transpose){
			SizeRef outp_shape_ut({X.shape()[0], X.shape()[1]*LKern, L});
			Tensor output(outp_shape_ut, x.dtype);
			X.arr_void().execute_function(im2col_nn_layer_nd_T, output.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0], dilation[1], x.shape()[-2], x.shape()[-1], X.shape()[1]*LKern, L, X.shape()[1], X.shape()[0]);
			return std::move(output);
		}

		SizeRef outp_shape_ut({X.shape()[0], L, X.shape()[1]*LKern});
		Tensor output(outp_shape_ut, x.dtype);
		if(stride == 1 && dilation == 1){
			X.arr_void().execute_function(im2col_nn_layer_nd_none, output.arr_void(), kernel_size[0], kernel_size[1], x.shape()[-2], x.shape()[-1], LKern, L, X.shape()[1], X.shape()[0]);
		}
		else if(stride == 1 && !(dilation == 1)){
			X.arr_void().execute_function(im2col_nn_layer_nd_dilation, output.arr_void(), kernel_size[0], kernel_size[1], dilation[0], dilation[1], x.shape()[-2], x.shape()[-1], LKern, L, X.shape()[1], X.shape()[0]);
		}
		else if(!(stride == 1) && dilation == 1){
			X.arr_void().execute_function(im2col_nn_layer_nd_stride, output.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], x.shape()[-2], x.shape()[-1], LKern, L, X.shape()[1], X.shape()[0]);
		}
		else{
			X.arr_void().execute_function(im2col_nn_layer_nd, output.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0], dilation[1], x.shape()[-2], x.shape()[-1], LKern, L, X.shape()[1], X.shape()[0]);
		}
		return std::move(output);
	
	}
	Tensor X = x;
	if(padding[0] != 0 || padding[1] != 0){
		if(out_transpose){
			SizeRef outp_shape_ut({LKern, L});
			Tensor output(outp_shape_ut, x.dtype);
			X.arr_void().execute_function(im2col_nn_layer_2d_T_pad, output.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0], dilation[1], padding[0], padding[1], x.shape()[-2], x.shape()[-1], LKern, L);
			return std::move(output);
		}

		SizeRef outp_shape_ut({L, LKern});
		Tensor output(outp_shape_ut, x.dtype);
		X.arr_void().execute_function(im2col_nn_layer_2d_pad, output.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0], dilation[1], padding[0], padding[1], x.shape()[-2], x.shape()[-1]);
		return std::move(output);	
	}
		
	if(out_transpose){
		SizeRef outp_shape_ut({LKern, L});
		Tensor output(outp_shape_ut, x.dtype);
		X.arr_void().execute_function(im2col_nn_layer_2d_T, output.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0], dilation[1], x.shape()[-2], x.shape()[-1], LKern, L);
		return std::move(output);
	}

	SizeRef outp_shape_ut({L, LKern});
	Tensor output(outp_shape_ut, x.dtype);
	X.arr_void().execute_function(im2col_nn_layer_2d, output.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0], dilation[1], x.shape()[-2], x.shape()[-1]);
	return std::move(output);
}

Tensor Unfold::backward(const Tensor& dz){
	return dz.contiguous();
}

Tensor Unfold::eval(const Tensor& x) const {
	return x.contiguous();
}


}
}
