#include "functional_fold.h"
#include "../Tensor.h"
#include "../layers/TensorGrad.h"
#include "../dtype/ArrayVoid.hpp"
#include "../mp/Threading.h"


namespace nt{
namespace functional{



inline static constexpr auto im2col_nn_layer_2d = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, const int64_t& d_r, const int64_t& d_c, const int64_t& t_r, const int64_t& t_c){
	int64_t a_current_add = ((t_c * d_r) - (k_c*d_c)); 
	for(int64_t r = 0; r < t_r - ((k_r-1)*d_c); r += s_r){
		for(int64_t c = 0; c < t_c - ((k_c-1)*d_c); c += s_c){
			auto a_current = a_begin + (r * t_c + c);
			for(int64_t rr = 0; rr < k_r; rr += d_r, a_current += a_current_add){
				for(int64_t cc = 0; cc < k_c; cc += d_c, a_current += d_c, ++b_begin){
					*b_begin = *a_current;
				}
			}
		}
	}
};




inline static constexpr auto im2col_nn_layer_2d_pad = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, int64_t d_r, const int64_t& d_c, int64_t p_r, int64_t p_c, int64_t t_r, int64_t t_c){
	int64_t a_current_add = (t_c * d_r); 
	int64_t upper_row2 = t_r - ((k_r-1)*d_r) + (p_r * 2);
	int64_t upper_col2 = t_c - ((k_c-1)*d_c) + (p_c * 2);
	int64_t kd_r = (k_r/d_r);
	int64_t kd_c = (k_c/d_c);
	for(int64_t r = 0; r < upper_row2; r += s_r){
		for(int64_t c = 0; c < upper_col2; c += s_c){
			int64_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
			int64_t c_add = c - p_c <= 0 ? 0 : c - p_c;
			auto a_current = a_begin + r_add + c_add;
			for(int64_t rr = 0; rr < k_r; rr += d_r){
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
				int64_t pad_counter = 0;
				for(int64_t cc = 0; cc < k_c; cc += d_c, ++b_begin){
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



inline static constexpr auto im2col_nn_layer_2d_backward = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, const int64_t& d_r, const int64_t& d_c, const int64_t& t_r, const int64_t& t_c){
	int64_t a_current_add = ((t_c * d_r) - (k_c*d_c)); 
	for(int64_t r = 0; r < t_r - ((k_r-1)*d_c); r += s_r){
		for(int64_t c = 0; c < t_c - ((k_c-1)*d_c); c += s_c){
			auto a_current = a_begin + (r * t_c + c);
			for(int64_t rr = 0; rr < k_r*d_r; rr += d_r, a_current += a_current_add){
				for(int64_t cc = 0; cc < k_c*d_c; cc += d_c, a_current += d_c, ++b_begin){
					*a_current = *b_begin;
				}
			}
		}
	}
};


inline static constexpr auto im2col_nn_layer_2d_pad_backward = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, int64_t d_r, const int64_t& d_c, int64_t p_r, int64_t p_c, int64_t t_r, int64_t t_c){
	int64_t a_current_add = (t_c * d_r); 
	int64_t upper_row2 = t_r - ((k_r-1)*d_r) + (p_r * 2);
	int64_t upper_col2 = t_c - ((k_c-1)*d_c) + (p_c * 2);
	int64_t kd_r = (k_r/d_r);
	int64_t kd_c = (k_c/d_c);
	for(int64_t r = 0; r < upper_row2; r += s_r){
		for(int64_t c = 0; c < upper_col2; c += s_c){
			int64_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
			int64_t c_add = c - p_c <= 0 ? 0 : c - p_c;
			auto a_current = a_begin + r_add + c_add;
			for(int64_t rr = 0; rr < k_r*d_r; rr += d_r){
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
				int64_t pad_counter = 0;
				for(int64_t cc = 0; cc < k_c*d_c; cc += d_c, ++b_begin){
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
inline static constexpr auto im2col_nn_layer_nd = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, const int64_t& d_r, const int64_t& d_c, const int64_t& t_r, const int64_t& t_c, const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t a_current_add = ((t_c * d_r) - (k_c*d_c));
	const int64_t mat_size = t_c * t_r;
	const int64_t upper_rows = t_r - ((k_r-1) * d_r);
	const int64_t upper_cols = t_c - ((k_c-1) * d_c);
	const int64_t upper_rows2 = k_r*d_r;
	const int64_t upper_cols2 = k_c * d_c;
	int64_t batch_add = 0;
	for(int64_t ba = 0; ba < batches; ++ba, batch_add += (channels * mat_size)){
		for(int64_t r = 0; r < upper_rows; r += s_r){
			for(int64_t c = 0; c < upper_cols; c += s_c){
				for(int64_t ch = 0; ch < channels; ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < upper_rows2; rr += d_r, a_current += a_current_add){
						for(int64_t cc = 0; cc < upper_cols2; cc += d_c, a_current += d_c, ++b_begin){
							*b_begin = *a_current;
						}
					}
				}
			}
		}
	}
};



inline static constexpr auto im2col_nn_layer_nd_none = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& t_r, const int64_t& t_c, const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t a_current_add = (t_c - k_c);
	const int64_t mat_size = t_c * t_r;
	const int64_t upper_rows = t_r - (k_r - 1);
	const int64_t upper_cols = t_c - (k_c - 1);
	const int64_t kc_add = k_c-1; 
	const int64_t batch_add = ((channels - 1) * mat_size) + ((k_r - 1) * t_c);
	const int64_t a_currentCA = mat_size - ((k_r*k_c) + (a_current_add * k_r));
	for(int64_t ba = 0; ba < batches; ++ba, a_begin += batch_add){
		for(int64_t r = 0; r < upper_rows; ++r, a_begin += kc_add){
			for(int64_t c = 0; c < upper_cols; ++c, ++a_begin){
				auto a_current = a_begin;
				for(int64_t ch = 0; ch < channels; ++ch, a_current += a_currentCA){
					for(int64_t rr = 0; rr < k_r; ++rr, a_current += a_current_add){
						for(int64_t cc = 0; cc < k_c; ++cc, ++a_current, ++b_begin){
							*b_begin = *a_current;
						}
					}
				}
			}
		}
	}
};


inline static constexpr auto im2col_nn_layer_nd_dilation = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& d_r, const int64_t& d_c, const int64_t& t_r, const int64_t& t_c, const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t a_current_add = ((t_c * d_r) - (k_c*d_c));
	const int64_t mat_size = t_c * t_r;
	const int64_t upper_rows = t_r - ((k_r-1) * d_r);
	const int64_t upper_cols = t_c - ((k_c-1) * d_c);
	const int64_t upper_rows2 = k_r*d_r;
	const int64_t upper_cols2 = k_c * d_c;
	for(int64_t ba = 0; ba < batches; ++ba){
		int64_t batch_add = (ba * channels * mat_size);
		for(int64_t r = 0; r < upper_rows; ++r){
			for(int64_t c = 0; c < upper_cols; ++c){
				for(int64_t ch = 0; ch < channels; ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < upper_rows2; rr += d_r, a_current += a_current_add){
						for(int64_t cc = 0; cc < upper_cols2; cc += d_c, a_current += d_c, ++b_begin){
							*b_begin = *a_current;
						}
					}
				}
			}
		}
	}
};

inline static constexpr auto im2col_nn_layer_nd_stride = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, const int64_t& t_r, const int64_t& t_c, const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t a_current_add = (t_c - k_c);
	const int64_t mat_size = t_c * t_r;
	const int64_t upper_rows = t_r - (k_r - 1);
	const int64_t upper_cols = t_c - (k_c - 1);
	const int64_t kc_add = k_c-1 + (t_c * (s_r-1));
	const int64_t a_currentCA = mat_size - ((k_r*k_c) + (a_current_add * k_r));
	/* const int64_t batch_add = ((channels-1)*mat_size) + ((k_r - 1) * t_c); */
	const int64_t batch_add = (channels * mat_size);
	int64_t counter = 0;
	for(int64_t ba = 0; ba < batches; ++ba, a_begin += batch_add){
		for(int64_t r = 0; r < upper_rows; r += s_r){
			for(int64_t c = 0; c < upper_cols; c += s_c){
				for(int64_t ch = 0; ch < channels; ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch);
					for(int64_t rr = 0; rr < k_r; ++rr, a_current += a_current_add){
						for(int64_t cc = 0; cc < k_c; ++cc, ++a_current, ++b_begin){
							*b_begin = *a_current;
						}
					}
				}
			}
		}
	}
};

#else


inline static constexpr auto im2col_nn_layer_nd = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, const int64_t& d_r, const int64_t& d_c, const int64_t& t_r, const int64_t& t_c, const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t a_current_add = ((t_c * d_r) - (k_c*d_c));
	const int64_t mat_size = t_c * t_r;
	const int64_t upper_rows = t_r - ((k_r-1) * d_r);
	const int64_t upper_cols = t_c - ((k_c-1) * d_c);
	const int64_t upper_rows2 = k_r*d_r;
	const int64_t upper_cols2 = k_c * d_c;
	int64_t batch_add = 0;
	for(int64_t ba = 0; ba < batches; ++ba, batch_add += (channels * mat_size)){
		for(int64_t r = 0; r < upper_rows; r += s_r){
			for(int64_t c = 0; c < upper_cols; c += s_c){
				for(int64_t ch = 0; ch < channels; ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < upper_rows2; rr += d_r, a_current += a_current_add){
						for(int64_t cc = 0; cc < upper_cols2; cc += d_c, a_current += d_c, ++b_begin){
							*b_begin = *a_current;
						}
					}
				}
			}
		}
	}
};


//needs to be corrected
/* inline static constexpr auto im2col_nn_layer_nd = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, const int64_t& d_r, const int64_t& d_c, const int64_t& t_r, const int64_t& t_c, const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){ */

/* 	int64_t row_max = (t_r - ((k_r-1) * d_r)); */
/* 	int64_t col_max = t_c - ((k_c-1) * d_c); */
/* 	const int64_t upper_row2 = k_r * d_r; */
/* 	const int64_t upper_col2 = k_c * d_c; */

/* 	int64_t a_current_add = ((t_c * d_r) - (k_c * d_c)); */
/* 	const int64_t mat_size = t_c * t_r; */
/* 	for(int64_t ba = 0; ba < batches; ++ba){ */
/* 		int64_t batch_add = (ba * channels * mat_size); */
/* 		//there might be an issue with the stride on this one, need to look into that. */
/* 		tbb::parallel_for(utils::calculateGrainSize3D(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<int64_t> range){ */
/* 		int64_t row_begin = range.rows().begin() % s_r == 0 ? range.rows().begin() : range.rows().begin() + range.rows().begin() % s_r; */
/* 		int64_t col_begin = range.cols().begin() % s_c == 0 ? range.cols().begin() : range.cols().begin() + range.cols().begin() % s_c; */
/* 		int64_t m_begin_index = ba * L * LKern * channels; //batches */ 
/* 		m_begin_index += ((row_begin) * col_max * LKern * channels) / (s_r * s_c); //rows; */
/* 		m_begin_index += ((col_begin) * LKern * channels) / s_c;//cols */
/* 		m_begin_index += range.pages().begin() * LKern; */
/* 		auto m_begin = b_begin + m_begin_index; */
/* 		for(int64_t r = row_begin; r < range.rows().end(); r += s_r){ */
/* 			for(int64_t c = col_begin; c < range.cols().end(); c += s_c){ */
/* 				for(int64_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){ */
/* 					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add; */
/* 					for(int64_t rr = 0; rr < upper_row2; rr += d_r, a_current += a_current_add){ */
/* 						for(int64_t cc = 0; cc < upper_col2; cc += d_c, a_current += d_c, ++m_begin){ */
/* 							*m_begin = *a_current; */
/* 						} */
/* 					} */
/* 				} */
/* 			} */
/* 		} */
/* 		}); */
/* 	} */
/* }; */



inline static constexpr auto im2col_nn_layer_nd_dilation = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& d_r, const int64_t& d_c, const int64_t& t_r, const int64_t& t_c, const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){

	int64_t row_max = (t_r - ((k_r-1) * d_r));
	int64_t col_max = t_c - ((k_c-1) * d_c);
	const int64_t upper_row2 = k_r * d_r;
	const int64_t upper_col2 = k_c * d_c;

	int64_t a_current_add = ((t_c * d_r) - (k_c * d_c));
	const int64_t mat_size = t_c * t_r;
	for(int64_t ba = 0; ba < batches; ++ba){
		int64_t batch_add = (ba * channels * mat_size);
		//there might be an issue with the stride on this one, need to look into that.
		tbb::parallel_for(utils::calculateGrainSize3D(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<int64_t> range){
		int64_t row_begin = range.rows().begin();
		int64_t col_begin = range.cols().begin();
		int64_t m_begin_index = ba * L * LKern * channels; //batches 
		m_begin_index += (row_begin * col_max * LKern * channels); //rows;
		m_begin_index += (col_begin * LKern * channels);//cols
		m_begin_index += range.pages().begin() * LKern;
		auto m_begin = b_begin + m_begin_index;
		for(int64_t r = row_begin; r < range.rows().end(); ++r){
			for(int64_t c = col_begin; c < range.cols().end(); ++c){
				for(int64_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < upper_row2; rr += d_r, a_current += a_current_add){
						for(int64_t cc = 0; cc < upper_col2; cc += d_c, a_current += d_c, ++m_begin){
							*m_begin = *a_current;
						}
					}
				}
			}
		}
		});
	}
};


inline static constexpr auto im2col_nn_layer_nd_stride = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, const int64_t& t_r, const int64_t& t_c, const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t a_current_add = (t_c - k_c);
	const int64_t mat_size = t_c * t_r;
	const int64_t upper_rows = t_r - (k_r - 1);
	const int64_t upper_cols = t_c - (k_c - 1);
	const int64_t kc_add = k_c-1 + (t_c * (s_r-1));
	const int64_t a_currentCA = mat_size - ((k_r*k_c) + (a_current_add * k_r));
	/* const int64_t batch_add = ((channels-1)*mat_size) + ((k_r - 1) * t_c); */
	const int64_t batch_add = (channels * mat_size);
	int64_t counter = 0;
	for(int64_t ba = 0; ba < batches; ++ba, a_begin += batch_add){
		for(int64_t r = 0; r < upper_rows; r += s_r){
			for(int64_t c = 0; c < upper_cols; c += s_c){
				for(int64_t ch = 0; ch < channels; ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch);
					for(int64_t rr = 0; rr < k_r; ++rr, a_current += a_current_add){
						for(int64_t cc = 0; cc < k_c; ++cc, ++a_current, ++b_begin){
							*b_begin = *a_current;
						}
					}
				}
			}
		}
	}
};

/* inline static constexpr auto im2col_nn_layer_nd_stride = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, const int64_t& t_r, const int64_t& t_c, const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){ */

/* 	int64_t row_max = (t_r - (k_r-1)); */
/* 	int64_t col_max = t_c - (k_c-1); */
/* 	const int64_t upper_row2 = k_r; */
/* 	const int64_t upper_col2 = k_c; */

/* 	int64_t a_current_add = (t_c - k_c); */
/* 	const int64_t mat_size = t_c * t_r; */
/* 	for(int64_t ba = 0; ba < batches; ++ba){ */
/* 		int64_t batch_add = (ba * channels * mat_size); */
/* 		//there might be an issue with the stride on this one, need to look into that. */
/* 		tbb::parallel_for(utils::calculateGrainSize3D(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<int64_t> range){ */
/* 		int64_t row_begin = range.rows().begin() % s_r == 0 ? range.rows().begin() : range.rows().begin() + range.rows().begin() % s_r; */
/* 		int64_t col_begin = range.cols().begin() % s_c == 0 ? range.cols().begin() : range.cols().begin() + range.cols().begin() % s_c; */
/* 		int64_t m_begin_index = ba * L * LKern * channels; //batches */ 
/* 		m_begin_index += ((row_begin) * col_max * LKern * channels) / (s_r * s_c); //rows; */
/* 		m_begin_index += ((col_begin) * LKern * channels) / s_c;//cols */
/* 		m_begin_index += range.pages().begin() * LKern; */
/* 		auto m_begin = b_begin + m_begin_index; */
/* 		for(int64_t r = row_begin; r < range.rows().end(); r += s_r){ */
/* 			for(int64_t c = col_begin; c < range.cols().end(); c += s_c){ */
/* 				for(int64_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){ */
/* 					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add; */
/* 					for(int64_t rr = 0; rr < upper_row2; ++rr, a_current += a_current_add){ */
/* 						for(int64_t cc = 0; cc < upper_col2; ++cc, ++a_current, ++m_begin){ */
/* 							*m_begin = *a_current; */
/* 						} */
/* 					} */
/* 				} */
/* 			} */
/* 		} */
/* 		}); */
/* 	} */
/* }; */


inline static constexpr auto im2col_nn_layer_nd_none = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& t_r, const int64_t& t_c, const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){

	int64_t row_max = (t_r - (k_r-1));
	int64_t col_max = t_c - (k_c-1);
	const int64_t upper_row2 = k_r;
	const int64_t upper_col2 = k_c;

	int64_t a_current_add = (t_c - k_c);
	const int64_t mat_size = t_c * t_r;
	for(int64_t ba = 0; ba < batches; ++ba){
		int64_t batch_add = (ba * channels * mat_size);
		//there might be an issue with the stride on this one, need to look into that.
		tbb::parallel_for(utils::calculateGrainSize3D(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<int64_t> range){
		int64_t m_begin_index = ba * L * LKern * channels; //batches 
		m_begin_index += (range.rows().begin() * col_max * LKern * channels); //rows;
		m_begin_index += (range.cols().begin() * LKern * channels);//cols
		m_begin_index += range.pages().begin() * LKern;
		auto m_begin = b_begin + m_begin_index;
		for(int64_t r = range.rows().begin(); r < range.rows().end(); ++r){
			for(int64_t c = range.cols().begin(); c < range.cols().end(); ++c){
				for(int64_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < upper_row2; ++rr, a_current += a_current_add){
						for(int64_t cc = 0; cc < upper_col2; ++cc, ++a_current, ++m_begin){
							*m_begin = *a_current;
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
inline static constexpr auto im2col_nn_layer_nd_backward = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, const int64_t& d_r, const int64_t& d_c, const int64_t& t_r, const int64_t& t_c, const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t a_current_add = ((t_c * d_r) - (k_c*d_c));
	const int64_t mat_size = t_c * t_r;
	for(int64_t ba = 0; ba < batches; ++ba){
		int64_t batch_add = (ba * channels * mat_size);
		for(int64_t r = 0; r < t_r - ((k_r-1) * d_r); r += s_r){
			for(int64_t c = 0; c < t_c - ((k_c-1) * d_c); c += s_c){
				for(int64_t ch = 0; ch < channels; ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < k_r*d_r; rr += d_r, a_current += a_current_add){
						for(int64_t cc = 0; cc < k_c*d_c; cc += d_c, a_current += d_c, ++b_begin){
							*a_current = *b_begin;
						}
					}
				}
			}
		}
	}
};
#else
inline static constexpr auto im2col_nn_layer_nd_backward = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, const int64_t& d_r, const int64_t& d_c, const int64_t& t_r, const int64_t& t_c, const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){

	int64_t row_max = (t_r - ((k_r-1) * d_r));
	int64_t col_max = t_c - ((k_c-1) * d_c);

	int64_t a_current_add = ((t_c * d_r) - (k_c * d_c));
	const int64_t mat_size = t_c * t_r;
	for(int64_t ba = 0; ba < batches; ++ba){
		int64_t batch_add = (ba * channels * mat_size);
		//there might be an issue with the stride on this one, need to look into that.
		tbb::parallel_for(utils::calculateGrainSize3D(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<int64_t> range){
		int64_t row_begin = range.rows().begin() % s_r == 0 ? range.rows().begin() : range.rows().begin() + range.rows().begin() % s_r;
		int64_t col_begin = range.cols().begin() % s_c == 0 ? range.cols().begin() : range.cols().begin() + range.cols().begin() % s_c;
		int64_t m_begin_index = ba * L * LKern * channels; //batches 
		m_begin_index += ((row_begin) * col_max * LKern * channels) / (s_r * s_c); //rows;
		m_begin_index += ((col_begin) * LKern * channels) / s_c;//cols
		m_begin_index += range.pages().begin() * LKern;
		auto m_begin = b_begin + m_begin_index;
		for(int64_t r = row_begin; r < range.rows().end(); r += s_r){
			for(int64_t c = col_begin; c < range.cols().end(); c += s_c){
				for(int64_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < k_r*d_r; rr += d_r, a_current += a_current_add){
						for(int64_t cc = 0; cc < k_c*d_c; cc += d_c, a_current += d_c, ++m_begin){
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

inline static constexpr auto im2col_nn_layer_nd_pad = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, int64_t d_r, const int64_t& d_c, int64_t p_r, int64_t p_c, int64_t t_r, int64_t t_c, const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t a_current_add = (t_c * d_r); 
	int64_t upper_row2 = t_r - ((k_r-1) * d_r) + (p_r * 2);
	int64_t upper_col2 = t_c - ((k_c-1) * d_c) + (p_c * 2);
	int64_t kd_r = (k_r*d_r);
	int64_t kd_c = (k_c*d_c);
	const int64_t mat_size = t_c * t_r;
	for(int64_t ba = 0; ba < batches; ++ba){
		int64_t batch_add = (ba * channels * mat_size);
		for(int64_t r = 0; r < upper_row2; r += s_r){
			for(int64_t c = 0; c < upper_col2; c += s_c){
				for(int64_t ch = 0; ch < channels; ++ch){
					int64_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					int64_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < kd_r; rr += d_r){
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
						int64_t pad_counter = 0;
						for(int64_t cc = 0; cc < kd_c; cc += d_c, ++b_begin){
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


inline static constexpr auto im2col_nn_layer_nd_pad_dilation = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, int64_t d_r, const int64_t& d_c, int64_t p_r, int64_t p_c, int64_t t_r, int64_t t_c, const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t a_current_add = (t_c * d_r); 
	int64_t upper_row2 = t_r - ((k_r-1) * d_r) + (p_r * 2);
	int64_t upper_col2 = t_c - ((k_c-1) * d_c) + (p_c * 2);
	int64_t kd_r = (k_r*d_r);
	int64_t kd_c = (k_c*d_c);
	const int64_t mat_size = t_c * t_r;
	for(int64_t ba = 0; ba < batches; ++ba){
		int64_t batch_add = (ba * channels * mat_size);
		for(int64_t r = 0; r < upper_row2; ++r){
			for(int64_t c = 0; c < upper_col2; ++c){
				for(int64_t ch = 0; ch < channels; ++ch){
					int64_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					int64_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < kd_r; rr += d_r){
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
						int64_t pad_counter = 0;
						for(int64_t cc = 0; cc < kd_c; cc += d_c, ++b_begin){
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



inline static constexpr auto im2col_nn_layer_nd_pad_stride = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, int64_t p_r, int64_t p_c, int64_t t_r, int64_t t_c, const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t a_current_add = t_c; 
	int64_t upper_row2 = t_r - ((k_r-1)) + (p_r * 2);
	int64_t upper_col2 = t_c - ((k_c-1)) + (p_c * 2);
	int64_t kd_r = (k_r);
	int64_t kd_c = (k_c);
	const int64_t mat_size = t_c * t_r;
	for(int64_t ba = 0; ba < batches; ++ba){
		int64_t batch_add = (ba * channels * mat_size);
		for(int64_t r = 0; r < upper_row2; r += s_r){
			for(int64_t c = 0; c < upper_col2; c += s_c){
				for(int64_t ch = 0; ch < channels; ++ch){
					int64_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					int64_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < kd_r; ++rr){
						if((r - p_r + rr) < 0){
							b_begin += k_c;
							continue;
						}
						if((r - p_r + rr) >= t_r){
							b_begin += k_c * (k_r - rr);
							break;
						}
						int64_t pad_counter = 0;
						for(int64_t cc = 0; cc < kd_c; ++cc, ++b_begin){
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


inline static constexpr auto im2col_nn_layer_nd_pad_none = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, int64_t p_r, int64_t p_c, int64_t t_r, int64_t t_c, const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t a_current_add = t_c; 
	int64_t upper_row2 = t_r - ((k_r-1)) + (p_r * 2);
	int64_t upper_col2 = t_c - ((k_c-1)) + (p_c * 2);
	int64_t kd_r = (k_r);
	int64_t kd_c = (k_c);
	const int64_t mat_size = t_c * t_r;
	for(int64_t ba = 0; ba < batches; ++ba){
		int64_t batch_add = (ba * channels * mat_size);
		for(int64_t r = 0; r < upper_row2; ++r){
			for(int64_t c = 0; c < upper_col2; ++c){
				for(int64_t ch = 0; ch < channels; ++ch){
					int64_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					int64_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < kd_r; ++rr){
						if((r - p_r + rr) < 0){
							b_begin += k_c;
							continue;
						}
						if((r - p_r + rr) >= t_r){
							b_begin += k_c * (k_r - rr);
							break;
						}
						int64_t pad_counter = 0;
						for(int64_t cc = 0; cc < kd_c; ++cc, ++b_begin){
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
inline static constexpr auto im2col_nn_layer_nd_pad = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, int64_t d_r, const int64_t& d_c, int64_t p_r, int64_t p_c, int64_t t_r, int64_t t_c, const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t a_current_add = (t_c * d_r); 
	int64_t row_max = t_r - ((k_r-1) * d_r) + (p_r * 2);
	int64_t col_max = t_c - ((k_c-1) * d_c) + (p_c * 2);
	int64_t kd_r = (k_r*d_r);
	int64_t kd_c = (k_c*d_c);
	const int64_t mat_size = t_c * t_r;
	for(int64_t ba = 0; ba < batches; ++ba){
		int64_t batch_add = (ba * channels * mat_size);
		tbb::parallel_for(utils::calculateGrainSize3D(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<int64_t> range){
		int64_t row_begin = range.rows().begin() % s_r == 0 ? range.rows().begin() : range.rows().begin() + range.rows().begin() % s_r;
		int64_t col_begin = range.cols().begin() % s_c == 0 ? range.cols().begin() : range.cols().begin() + range.cols().begin() % s_c;
		int64_t m_begin_index = ba * L * LKern * channels; //batches 
		m_begin_index += ((row_begin) * col_max * LKern * channels) / (s_r * s_c); //rows;
		m_begin_index += ((col_begin) * LKern * channels) / s_c;//cols
		m_begin_index += range.pages().begin() * LKern;
		auto m_begin = b_begin + m_begin_index;
		for(int64_t r = row_begin; r < range.rows().end(); r += s_r){
			for(int64_t c = col_begin; c < range.cols().end(); c += s_c){
				for(int64_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					int64_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					int64_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < kd_r; rr += d_r){
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
						int64_t pad_counter = 0;
						for(int64_t cc = 0; cc < kd_c; cc += d_c, ++m_begin){
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


inline static constexpr auto im2col_nn_layer_nd_pad_stride = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, int64_t p_r, int64_t p_c, int64_t t_r, int64_t t_c, const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t a_current_add = (t_c); 
	int64_t row_max = t_r - ((k_r-1)) + (p_r * 2);
	int64_t col_max = t_c - ((k_c-1)) + (p_c * 2);
	int64_t kd_r = (k_r);
	int64_t kd_c = (k_c);
	const int64_t mat_size = t_c * t_r;
	for(int64_t ba = 0; ba < batches; ++ba){
		int64_t batch_add = (ba * channels * mat_size);
		tbb::parallel_for(utils::calculateGrainSize3D(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<int64_t> range){
		int64_t row_begin = range.rows().begin() % s_r == 0 ? range.rows().begin() : range.rows().begin() + range.rows().begin() % s_r;
		int64_t col_begin = range.cols().begin() % s_c == 0 ? range.cols().begin() : range.cols().begin() + range.cols().begin() % s_c;
		int64_t m_begin_index = ba * L * LKern * channels; //batches 
		m_begin_index += ((row_begin) * col_max * LKern * channels) / (s_r * s_c); //rows;
		m_begin_index += ((col_begin) * LKern * channels) / s_c;//cols
		m_begin_index += range.pages().begin() * LKern;
		auto m_begin = b_begin + m_begin_index;
		for(int64_t r = row_begin; r < range.rows().end(); r += s_r){
			for(int64_t c = col_begin; c < range.cols().end(); c += s_c){
				for(int64_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					int64_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					int64_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < kd_r; ++rr){
						if((r - p_r + rr) < 0){
							m_begin += k_c;
							continue;
						}
						if((r - p_r + rr) >= t_r){
							m_begin += k_c * (k_r - rr);
							break;
						}
						int64_t pad_counter = 0;
						for(int64_t cc = 0; cc < kd_c; ++cc, ++m_begin){
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


inline static constexpr auto im2col_nn_layer_nd_pad_dilation = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, int64_t d_r, const int64_t& d_c, int64_t p_r, int64_t p_c, int64_t t_r, int64_t t_c, const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t a_current_add = (t_c * d_r); 
	int64_t row_max = t_r - ((k_r-1) * d_r) + (p_r * 2);
	int64_t col_max = t_c - ((k_c-1) * d_c) + (p_c * 2);
	int64_t kd_r = (k_r*d_r);
	int64_t kd_c = (k_c*d_c);
	const int64_t mat_size = t_c * t_r;
	for(int64_t ba = 0; ba < batches; ++ba){
		int64_t batch_add = (ba * channels * mat_size);
		tbb::parallel_for(utils::calculateGrainSize3D(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<int64_t> range){
		int64_t m_begin_index = ba * L * LKern * channels; //batches 
		m_begin_index += (range.rows().begin() * col_max * LKern * channels); //rows;
		m_begin_index += (range.cols().begin() * LKern * channels);//cols
		m_begin_index += range.pages().begin() * LKern;
		auto m_begin = b_begin + m_begin_index;
		for(int64_t r = range.rows().begin(); r < range.rows().end(); ++r){
			for(int64_t c = range.cols().begin(); c < range.cols().end(); ++c){
				for(int64_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					int64_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					int64_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < kd_r; rr += d_r){
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
						int64_t pad_counter = 0;
						for(int64_t cc = 0; cc < kd_c; cc += d_c, ++m_begin){
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


inline static constexpr auto im2col_nn_layer_nd_pad_none = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, int64_t p_r, int64_t p_c, int64_t t_r, int64_t t_c, const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t a_current_add = (t_c); 
	int64_t row_max = t_r - ((k_r-1)) + (p_r * 2);
	int64_t col_max = t_c - ((k_c-1)) + (p_c * 2);
	int64_t kd_r = (k_r);
	int64_t kd_c = (k_c);
	const int64_t mat_size = t_c * t_r;
	for(int64_t ba = 0; ba < batches; ++ba){
		int64_t batch_add = (ba * channels * mat_size);
		tbb::parallel_for(utils::calculateGrainSize3D(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<int64_t> range){
		int64_t m_begin_index = ba * L * LKern * channels; //batches 
		m_begin_index += (range.rows().begin() * col_max * LKern * channels); //rows;
		m_begin_index += (range.cols().begin() * LKern * channels);//cols
		m_begin_index += range.pages().begin() * LKern;
		auto m_begin = b_begin + m_begin_index;
		for(int64_t r = range.rows().begin(); r < range.rows().end(); ++r){
			for(int64_t c = range.cols().begin(); c < range.cols().end(); ++c){
				for(int64_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					int64_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					int64_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < kd_r; ++rr){
						if((r - p_r + rr) < 0){
							m_begin += k_c;
							continue;
						}
						if((r - p_r + rr) >= t_r){
							m_begin += k_c * (k_r - rr);
							break;
						}
						int64_t pad_counter = 0;
						for(int64_t cc = 0; cc < kd_c; ++cc, ++m_begin){
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

inline static constexpr auto im2col_nn_layer_nd_pad_backward = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, int64_t d_r, const int64_t& d_c, int64_t p_r, int64_t p_c, int64_t t_r, int64_t t_c, const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t a_current_add = (t_c * d_r); 
	int64_t upper_row2 = t_r - ((k_r-1) * d_r) + (p_r * 2);
	int64_t upper_col2 = t_c - ((k_c-1) * d_c) + (p_c * 2);
	int64_t kd_r = (k_r*d_r);
	int64_t kd_c = (k_c*d_c);
	const int64_t mat_size = t_c * t_r;
	for(int64_t ba = 0; ba < batches; ++ba){
		int64_t batch_add = (ba * channels * mat_size);
		for(int64_t r = 0; r < upper_row2; r += s_r){
			for(int64_t c = 0; c < upper_col2; c += s_r){
				for(int64_t ch = 0; ch < channels; ++ch){
					int64_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					int64_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < kd_r; rr += d_r){
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
						int64_t pad_counter = 0;
						for(int64_t cc = 0; cc < kd_c; cc += d_c, ++b_begin){
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
inline static constexpr auto im2col_nn_layer_nd_pad_backward = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, int64_t d_r, const int64_t& d_c, int64_t p_r, int64_t p_c, int64_t t_r, int64_t t_c, const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t a_current_add = (t_c * d_r); 
	int64_t row_max = t_r - ((k_r-1) * d_r) + (p_r * 2);
	int64_t col_max = t_c - ((k_c-1) * d_c) + (p_c * 2);
	int64_t kd_r = (k_r*d_r);
	int64_t kd_c = (k_c*d_c);
	const int64_t mat_size = t_c * t_r;
	for(int64_t ba = 0; ba < batches; ++ba){
		int64_t batch_add = (ba * channels * mat_size);
		tbb::parallel_for(utils::calculateGrainSize3D(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<int64_t> range){
		int64_t row_begin = range.rows().begin() % s_r == 0 ? range.rows().begin() : range.rows().begin() + range.rows().begin() % s_r;
		int64_t col_begin = range.cols().begin() % s_c == 0 ? range.cols().begin() : range.cols().begin() + range.cols().begin() % s_c;
		int64_t m_begin_index = ba * L * LKern * channels; //batches 
		m_begin_index += ((row_begin) * col_max * LKern * channels) / (s_r * s_c); //rows;
		m_begin_index += ((col_begin) * LKern * channels) / s_c;//cols
		m_begin_index += range.pages().begin() * LKern;
		auto m_begin = b_begin + m_begin_index;
		for(int64_t r = row_begin; r < range.rows().end(); r += s_r){
			for(int64_t c = col_begin; c < range.cols().end(); c += s_r){
				for(int64_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					int64_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					int64_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < kd_r; rr += d_r){
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
						int64_t pad_counter = 0;
						for(int64_t cc = 0; cc < kd_c; cc += d_c, ++m_begin){
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

class im2col_nn_layer_transpose_index{
	int64_t index, row, col, LKern, L; //originally _cols = LKern, _rows = L
	int64_t adding;
	public:
		im2col_nn_layer_transpose_index(int64_t lkern, int64_t l)
			:index(0), row(0), col(0), LKern(lkern), L(l), adding(0)
		{}
		const int64_t& operator*(){return index;}
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
		void operator+=(const int64_t& c){
			if((col + c) >= LKern){
				row += (col + c) / LKern;
				col = (col + c) % LKern;
				index = (col * L) + row + adding;
				return;
			}
			col += c;
			index += (L * c);
		}
		im2col_nn_layer_transpose_index operator+(const int64_t& c) const{
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
		void set_adding(const int64_t& c){
			adding = c;
		}

};

inline static constexpr auto im2col_nn_layer_2d_T = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, const int64_t& d_r, const int64_t& d_c, const int64_t& t_r, const int64_t& t_c, const int64_t& LKern, const int64_t& L){
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	int64_t a_current_add = ((t_c * d_r) - (k_c/d_c)); 
	for(int64_t r = 0; r < t_r - (k_r-1); r += s_r){
		for(int64_t c = 0; c < t_c - (k_c-1); c += s_r){
			auto a_current = a_begin + (r * t_c + c);
			for(int64_t rr = 0; rr < k_r; rr += d_r, a_current += a_current_add){
				for(int64_t cc = 0; cc < k_c; cc += d_c, a_current += d_c, ++b_indexer){
					*(b_begin + *b_indexer) = *a_current;
				}
			}
		}
	}
};

inline static constexpr auto im2col_nn_layer_2d_T_backward = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, const int64_t& d_r, const int64_t& d_c, const int64_t& t_r, const int64_t& t_c, const int64_t& LKern, const int64_t& L){
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	int64_t a_current_add = ((t_c * d_r) - (k_c/d_c)); 
	for(int64_t r = 0; r < t_r - (k_r-1); r += s_r){
		for(int64_t c = 0; c < t_c - (k_c-1); c += s_r){
			auto a_current = a_begin + (r * t_c + c);
			for(int64_t rr = 0; rr < k_r; rr += d_r, a_current += a_current_add){
				for(int64_t cc = 0; cc < k_c; cc += d_c, a_current += d_c, ++b_indexer){
					*a_current = *(b_begin + *b_indexer);
				}
			}
		}
	}
};

#ifndef USE_PARALLEL
inline static constexpr auto im2col_nn_layer_nd_T = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, const int64_t& d_r, const int64_t& d_c, const int64_t& t_r, const int64_t& t_c, const int64_t LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	int64_t a_current_add = ((t_c * d_r) - (k_c*d_c));
	const int64_t mat_size = t_c * t_r;
	const int64_t upper_rows = t_r - ((k_r-1) * d_r);
	const int64_t upper_cols = t_c - ((k_c-1) * d_c);
	const int64_t upper_rows2 = k_r*d_r;
	const int64_t upper_cols2 = k_c * d_c;
	int64_t adding = 0;
	for(int64_t ba = 0; ba < batches; ++ba, b_indexer.zero()){
		adding = (ba * L * LKern);
		int64_t batch_add = (ba * channels * mat_size);
		for(int64_t r = 0; r < upper_rows; r += s_r){
			for(int64_t c = 0; c <upper_cols; c += s_c){
				for(int64_t ch = 0; ch < channels; ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < upper_rows2; rr += d_r, a_current += a_current_add){
						for(int64_t cc = 0; cc < upper_cols2; cc += d_c, a_current += d_c, ++b_indexer){
							*(b_begin + *b_indexer + adding) = *a_current;
						}
					}
				}
			}
		}
	}
};

inline static constexpr auto im2col_nn_layer_nd_T_stride = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, const int64_t& t_r, const int64_t& t_c, const int64_t LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	int64_t a_current_add = (t_c - k_c);
	const int64_t mat_size = t_c * t_r;
	const int64_t upper_rows = t_r - (k_r-1);
	const int64_t upper_cols = t_c - (k_c-1);
	const int64_t upper_rows2 = k_r;
	const int64_t upper_cols2 = k_c;
	int64_t adding = 0;
	for(int64_t ba = 0; ba < batches; ++ba, b_indexer.zero()){
		adding = (ba * L * LKern);
		int64_t batch_add = (ba * channels * mat_size);
		for(int64_t r = 0; r < upper_rows; r += s_r){
			for(int64_t c = 0; c <upper_cols; c += s_c){
				for(int64_t ch = 0; ch < channels; ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < upper_rows2; ++rr, a_current += a_current_add){
						for(int64_t cc = 0; cc < upper_cols2; ++cc, ++a_current, ++b_indexer){
							*(b_begin + *b_indexer + adding) = *a_current;
						}
					}
				}
			}
		}
	}
};

inline static constexpr auto im2col_nn_layer_nd_T_dilation = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& d_r, const int64_t& d_c, const int64_t& t_r, const int64_t& t_c, const int64_t LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	int64_t a_current_add = ((t_c * d_r) - (k_c*d_c));
	const int64_t mat_size = t_c * t_r;
	const int64_t upper_rows = t_r - ((k_r-1) * d_r);
	const int64_t upper_cols = t_c - ((k_c-1) * d_c);
	const int64_t upper_rows2 = k_r*d_r;
	const int64_t upper_cols2 = k_c * d_c;
	int64_t adding = 0;
	for(int64_t ba = 0; ba < batches; ++ba, b_indexer.zero()){
		adding = (ba * L * LKern);
		int64_t batch_add = (ba * channels * mat_size);
		for(int64_t r = 0; r < upper_rows; ++r){
			for(int64_t c = 0; c <upper_cols; ++c){
				for(int64_t ch = 0; ch < channels; ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < upper_rows2; rr += d_r, a_current += a_current_add){
						for(int64_t cc = 0; cc < upper_cols2; cc += d_c, a_current += d_c, ++b_indexer){
							*(b_begin + *b_indexer + adding) = *a_current;
						}
					}
				}
			}
		}
	}
};


inline static constexpr auto im2col_nn_layer_nd_T_none = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& t_r, const int64_t& t_c, const int64_t LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	int64_t a_current_add = (t_c - k_c);
	const int64_t mat_size = t_c * t_r;
	const int64_t upper_rows = t_r - (k_r-1);
	const int64_t upper_cols = t_c - (k_c-1);
	const int64_t upper_rows2 = k_r;
	const int64_t upper_cols2 = k_c;
	int64_t adding = 0;
	for(int64_t ba = 0; ba < batches; ++ba, b_indexer.zero()){
		adding = (ba * L * LKern);
		int64_t batch_add = (ba * channels * mat_size);
		for(int64_t r = 0; r < upper_rows; ++r){
			for(int64_t c = 0; c <upper_cols; ++c){
				for(int64_t ch = 0; ch < channels; ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < upper_rows2; ++rr, a_current += a_current_add){
						for(int64_t cc = 0; cc < upper_cols2; ++cc, ++a_current, ++b_indexer){
							*(b_begin + *b_indexer + adding) = *a_current;
						}
					}
				}
			}
		}
	}
};

#else

inline static constexpr auto im2col_nn_layer_nd_T = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, const int64_t& d_r, const int64_t& d_c, const int64_t& t_r, const int64_t& t_c, const int64_t LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	int64_t a_current_add = ((t_c * d_r) - (k_c*d_c));
	const int64_t mat_size = t_c * t_r;
	const int64_t upper_rows = t_r - ((k_r-1) * d_r);
	const int64_t upper_cols = t_c - ((k_c-1) * d_c);
	const int64_t upper_rows2 = k_r*d_r;
	const int64_t upper_cols2 = k_c * d_c;
	int64_t adding = 0;
	for(int64_t ba = 0; ba < batches; ++ba, b_indexer.zero()){
		adding = (ba * L * LKern);
		int64_t batch_add = (ba * channels * mat_size);
		for(int64_t r = 0; r < upper_rows; r += s_r){
			for(int64_t c = 0; c <upper_cols; c += s_c){
				for(int64_t ch = 0; ch < channels; ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < upper_rows2; rr += d_r, a_current += a_current_add){
						for(int64_t cc = 0; cc < upper_cols2; cc += d_c, a_current += d_c, ++b_indexer){
							*(b_begin + *b_indexer + adding) = *a_current;
						}
					}
				}
			}
		}
	}
};


//need to fix this and model it after the one above it
/* inline static constexpr auto im2col_nn_layer_nd_T = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, const int64_t& d_r, const int64_t& d_c, const int64_t& t_r, const int64_t& t_c, const int64_t LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){ */
/* 	im2col_nn_layer_transpose_index b_indexer(LKern, L); */
/* 	int64_t row_max = (t_r - ((k_r-1) * d_r)); */
/* 	int64_t col_max = t_c - ((k_c-1) * d_c); */
/* 	int64_t a_current_add = ((t_c * d_r) - (k_c*d_c)); */
/* 	const int64_t mat_size = t_c * t_r; */
/* 	int64_t adding = 0; */
/* 	for(int64_t ba = 0; ba < batches; ++ba, b_indexer.zero()){ */
/* 		adding = (ba * L * LKern); */
/* 		int64_t batch_add = (ba * channels * mat_size); */
/* 		tbb::parallel_for(utils::calculateGrainSize3D(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<int64_t> range){ */
/* 		int64_t row_begin = range.rows().begin() % s_r == 0 ? range.rows().begin() : range.rows().begin() + range.rows().begin() % s_r; */
/* 		int64_t col_begin = range.cols().begin() % s_c == 0 ? range.cols().begin() : range.cols().begin() + range.cols().begin() % s_c; */
/* 		int64_t m_index = ((row_begin) * col_max * LKern) / (s_r * s_c); // rows */
/* 		m_index += ((col_begin) * LKern) / s_c;//cols */
/* 		m_index += range.pages().begin() * (LKern/channels); */
/* 		im2col_nn_layer_transpose_index m_begin_index = b_indexer + m_index; */  
/* 		for(int64_t r = row_begin; r < range.rows().end(); r += s_r){ */
/* 			for(int64_t c = col_begin; c < range.cols().end(); c += s_c){ */
/* 				for(int64_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){ */
/* 					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add; */
/* 					for(int64_t rr = 0; rr < k_r*d_r; rr += d_r, a_current += a_current_add){ */
/* 						for(int64_t cc = 0; cc < k_c*d_c; cc += d_c, a_current += d_c, ++m_begin_index){ */
/* 							*(b_begin + adding + *m_begin_index) = *a_current; */
/* 						} */
/* 					} */
/* 				} */
/* 			} */
/* 		} */
/* 		}); */
/* 	} */
/* }; */

#endif


#ifndef USE_PARALLEL
inline static constexpr auto im2col_nn_layer_nd_T_backward = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, const int64_t& d_r, const int64_t& d_c, const int64_t& t_r, const int64_t& t_c, const int64_t LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	int64_t a_current_add = ((t_c * d_r) - (k_c*d_c));
	const int64_t mat_size = t_c * t_r;
	int64_t adding = 0;
	for(int64_t ba = 0; ba < batches; ++ba, b_indexer.zero()){
		adding = (ba * L * LKern);
		int64_t batch_add = (ba * channels * mat_size);
		for(int64_t r = 0; r < t_r - ((k_r-1) * d_r); r += s_r){
			for(int64_t c = 0; c < t_c - ((k_c-1) * d_c); c += s_c){
				for(int64_t ch = 0; ch < channels; ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < k_r*d_r; rr += d_r, a_current += a_current_add){
						for(int64_t cc = 0; cc < k_c*d_c; cc += d_c, a_current += d_c, ++b_indexer){
							*a_current = *(b_begin + adding + *b_indexer) ;
						}
					}
				}
			}
		}
	}
};

#else

inline static constexpr auto im2col_nn_layer_nd_T_backward = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, const int64_t& d_r, const int64_t& d_c, const int64_t& t_r, const int64_t& t_c, const int64_t LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	int64_t row_max = (t_r - ((k_r-1) * d_r));
	int64_t col_max = t_c - ((k_c-1) * d_c);
	int64_t a_current_add = ((t_c * d_r) - (k_c*d_c));
	const int64_t mat_size = t_c * t_r;
	int64_t adding = 0;
	for(int64_t ba = 0; ba < batches; ++ba, b_indexer.zero()){
		adding = (ba * L * LKern);
		int64_t batch_add = (ba * channels * mat_size);
		tbb::parallel_for(utils::calculateGrainSize3D(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<int64_t> range){
		int64_t row_begin = range.rows().begin() % s_r == 0 ? range.rows().begin() : range.rows().begin() + range.rows().begin() % s_r;
		int64_t col_begin = range.cols().begin() % s_c == 0 ? range.cols().begin() : range.cols().begin() + range.cols().begin() % s_c;
		int64_t m_index = ((row_begin) * col_max * LKern) / (s_r * s_c); // rows
		m_index += ((col_begin) * LKern) / s_c;//cols
		m_index += range.pages().begin() * (LKern/channels);
		im2col_nn_layer_transpose_index m_begin_index = b_indexer + m_index;  
		for(int64_t r = row_begin; r < range.rows().end(); r += s_r){
			for(int64_t c = col_begin; c < range.cols().end(); c += s_c){
				for(int64_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					auto a_current = a_begin + (r * t_c + c) + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < k_r*d_r; rr += d_r, a_current += a_current_add){
						for(int64_t cc = 0; cc < k_c*d_c; cc += d_c, a_current += d_c, ++m_begin_index){
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

inline static constexpr auto im2col_nn_layer_2d_T_pad = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, int64_t d_r, const int64_t& d_c, int64_t p_r, int64_t p_c, int64_t t_r, int64_t t_c, const int64_t& LKern, const int64_t& L){
	int64_t a_current_add = (t_c * d_r); 
	int64_t upper_row2 = t_r - (k_r-1) + (p_r * 2);
	int64_t upper_col2 = t_c - (k_c-1) + (p_c * 2);
	int64_t kd_r = (k_r/d_r);
	int64_t kd_c = (k_c/d_c);
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	for(int64_t r = 0; r < upper_row2; r += s_r){
		for(int64_t c = 0; c < upper_col2; c += s_r){
			int64_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
			int64_t c_add = c - p_c <= 0 ? 0 : c - p_c;
			auto a_current = a_begin + r_add + c_add;
			for(int64_t rr = 0; rr < k_r; rr += d_r){
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
				int64_t pad_counter = 0;
				for(int64_t cc = 0; cc < k_c; cc += d_c, ++b_indexer){
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


inline static constexpr auto im2col_nn_layer_2d_T_pad_backward = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, int64_t d_r, const int64_t& d_c, int64_t p_r, int64_t p_c, int64_t t_r, int64_t t_c, const int64_t& LKern, const int64_t& L){
	int64_t a_current_add = (t_c * d_r); 
	int64_t upper_row2 = t_r - (k_r-1) + (p_r * 2);
	int64_t upper_col2 = t_c - (k_c-1) + (p_c * 2);
	int64_t kd_r = (k_r/d_r);
	int64_t kd_c = (k_c/d_c);
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	for(int64_t r = 0; r < upper_row2; r += s_r){
		for(int64_t c = 0; c < upper_col2; c += s_r){
			int64_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
			int64_t c_add = c - p_c <= 0 ? 0 : c - p_c;
			auto a_current = a_begin + r_add + c_add;
			for(int64_t rr = 0; rr < k_r; rr += d_r){
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
				int64_t pad_counter = 0;
				for(int64_t cc = 0; cc < k_c; cc += d_c, ++b_indexer){
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
inline static constexpr auto im2col_nn_layer_nd_T_pad = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, int64_t d_r, const int64_t& d_c, int64_t p_r, int64_t p_c, int64_t t_r, int64_t t_c, int64_t LKern, int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t a_current_add = (t_c * d_r); 
	int64_t upper_row2 = t_r - ((k_r-1) * d_r) + (p_r * 2);
	int64_t upper_col2 = t_c - ((k_c-1) * d_c) + (p_c * 2);
	int64_t kd_r = (k_r*d_r);
	int64_t kd_c = (k_c*d_c);
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	const int64_t mat_size = t_c * t_r;
	int64_t adding = 0;
	for(int64_t ba = 0; ba < batches; ++ba, b_indexer.zero()){
		adding = (ba * L * LKern);
		int64_t batch_add = ba * channels * mat_size;
		for(int64_t r = 0; r < upper_row2; r += s_r){
			for(int64_t c = 0; c < upper_col2; c += s_r){
				for(int64_t ch = 0; ch < channels; ++ch){
					int64_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					int64_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < kd_r; rr += d_r){
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
						int64_t pad_counter = 0;
						for(int64_t cc = 0; cc < kd_c; cc += d_c, ++b_indexer){
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

inline static constexpr auto im2col_nn_layer_nd_T_pad = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, int64_t d_r, const int64_t& d_c, int64_t p_r, int64_t p_c, int64_t t_r, int64_t t_c, int64_t LKern, int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t a_current_add = (t_c * d_r); 
	int64_t row_max = t_r - ((k_r-1) * d_r) + (p_r * 2);
	int64_t col_max = t_c - ((k_c-1) * d_c) + (p_c * 2);
	int64_t kd_r = (k_r*d_r);
	int64_t kd_c = (k_c*d_c);
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	const int64_t mat_size = t_c * t_r;
	int64_t adding = 0;
	for(int64_t ba = 0; ba < batches; ++ba, b_indexer.zero()){
		adding = (ba * L * LKern);
		int64_t batch_add = ba * channels * mat_size;
		tbb::parallel_for(utils::calculateGrainSize3D(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<int64_t> range){
		int64_t row_begin = range.rows().begin() % s_r == 0 ? range.rows().begin() : range.rows().begin() + range.rows().begin() % s_r;
		int64_t col_begin = range.cols().begin() % s_c == 0 ? range.cols().begin() : range.cols().begin() + range.cols().begin() % s_c;
		int64_t m_index = ((row_begin) * col_max * LKern) / (s_r * s_c); // rows
		m_index += ((col_begin) * LKern) / s_c;//cols
		m_index += range.pages().begin() * (LKern/channels);
		im2col_nn_layer_transpose_index m_begin_index = b_indexer + m_index;  
		for(int64_t r = row_begin; r < range.rows().end(); r += s_r){
			for(int64_t c = col_begin; c < range.cols().end(); c += s_r){
				for(int64_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					int64_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					int64_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < kd_r; rr += d_r){
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
						int64_t pad_counter = 0;
						for(int64_t cc = 0; cc < kd_c; cc += d_c, ++m_begin_index){
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
inline static constexpr auto im2col_nn_layer_nd_T_pad_backward = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, int64_t d_r, const int64_t& d_c, int64_t p_r, int64_t p_c, int64_t t_r, int64_t t_c, const int64_t LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t a_current_add = (t_c * d_r); 
	int64_t upper_row2 = t_r - ((k_r-1) * d_r) + (p_r * 2);
	int64_t upper_col2 = t_c - ((k_c-1) * d_c) + (p_c * 2);
	int64_t kd_r = (k_r*d_r);
	int64_t kd_c = (k_c*d_c);
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	const int64_t mat_size = t_c * t_r;
	int64_t adding = 0;
	for(int64_t ba = 0; ba < batches; ++ba, b_indexer.zero()){
		adding = (ba * L * LKern);
		int64_t batch_add = ba * channels * mat_size;
		for(int64_t r = 0; r < upper_row2; r += s_r){
			for(int64_t c = 0; c < upper_col2; c += s_r){
				for(int64_t ch = 0; ch < channels; ++ch){
					int64_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					int64_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < kd_r; rr += d_r){
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
						int64_t pad_counter = 0;
						for(int64_t cc = 0; cc < kd_c; cc += d_c, ++b_indexer){
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

inline static constexpr auto im2col_nn_layer_nd_T_pad_backward = [](auto a_begin, auto a_end, auto b_begin, const int64_t& k_r, const int64_t& k_c, const int64_t& s_r, const int64_t& s_c, int64_t d_r, const int64_t& d_c, int64_t p_r, int64_t p_c, int64_t t_r, int64_t t_c, const int64_t LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t a_current_add = (t_c * d_r); 
	int64_t row_max = t_r - ((k_r-1) * d_r) + (p_r * 2);
	int64_t col_max = t_c - ((k_c-1) * d_c) + (p_c * 2);
	int64_t kd_r = (k_r*d_r);
	int64_t kd_c = (k_c*d_c);
	im2col_nn_layer_transpose_index b_indexer(LKern, L);
	const int64_t mat_size = t_c * t_r;
	int64_t adding = 0;
	for(int64_t ba = 0; ba < batches; ++ba, b_indexer.zero()){
		adding = (ba * L * LKern);
		int64_t batch_add = ba * channels * mat_size;
		tbb::parallel_for(utils::calculateGrainSize3D(0, channels, 0, row_max, 0, col_max), [&](tbb::blocked_range3d<int64_t> range){
		int64_t row_begin = range.rows().begin() % s_r == 0 ? range.rows().begin() : range.rows().begin() + range.rows().begin() % s_r;
		int64_t col_begin = range.cols().begin() % s_c == 0 ? range.cols().begin() : range.cols().begin() + range.cols().begin() % s_c;
		int64_t m_index = ((row_begin) * col_max * LKern) / (s_r * s_c); // rows
		m_index += ((col_begin) * LKern) / s_c;//cols
		m_index += range.pages().begin() * (LKern/channels);
		im2col_nn_layer_transpose_index m_begin_index = b_indexer + m_index;  
		for(int64_t r = row_begin; r < range.rows().end(); r += s_r){
			for(int64_t c = col_begin; c < range.cols().end(); c += s_r){
				for(int64_t ch = range.pages().begin(); ch < range.pages().end(); ++ch){
					int64_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
					int64_t c_add = c - p_c <= 0 ? 0 : c - p_c;
					auto a_current = a_begin + r_add + c_add + (mat_size * ch) + batch_add;
					for(int64_t rr = 0; rr < kd_r; rr += d_r){
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
						int64_t pad_counter = 0;
						for(int64_t cc = 0; cc < kd_c; cc += d_c, ++m_begin_index){
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


//some benchmarks:
/* Tensor im2col_2_dim */

//im2col_nn_layer_2d_T function: 142
//im2col_nn_layer_2d + RowColSwap_contiguous: 221
//im2col_nn_layer_2d + transpose(-1,-2): 180
//im2col_nn_layer_2d + RowColSwap: 168
//clearly the im2col_nn_layer_2d_T function is most efficient

Tensor unfold(const Tensor& x, utils::my_tuple kernel_size, utils::my_tuple dilation, utils::my_tuple padding, utils::my_tuple stride, bool transpose_out){

	int64_t LKern = kernel_size[0] * kernel_size[1];

	int64_t L_r = ((x.shape()[-2] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
	int64_t L_c = ((x.shape()[-1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
	int64_t L = L_r * L_c;

	utils::throw_exception((L_r*L_c) > 0, "Given input with spatial size ($, $), kernel_size=($, $), dilation=($, $), padding=($, $), calculated shape of the array of sliding blocks as ($, $), but its components must be at least one.", x.shape()[-2], x.shape()[-1], kernel_size[0], kernel_size[1], dilation[0], dilation[1], padding[0], padding[1], L_r, L_c);
	if(x.dims() > 2){
		Tensor X = (x.dims() == 3) ? x.view({1, x.shape()[0], x.shape()[1], x.shape()[2]}) : x.flatten(0, -4);
		if(!(padding == 0)){
			if(transpose_out){
				SizeRef outp_shape_ut({x.shape()[0], X.shape()[1]*LKern, L});
				Tensor output = zeros(outp_shape_ut, x.dtype);
				X.arr_void().execute_function(im2col_nn_layer_nd_T_pad, output.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0], dilation[1], padding[0], padding[1], x.shape()[-2], x.shape()[-1], X.shape()[1]*LKern, L, X.shape()[1], X.shape()[0]);
				return std::move(output);
			}

			SizeRef outp_shape_ut({X.shape()[0], L, X.shape()[1]*LKern});
			Tensor output = zeros(outp_shape_ut, x.dtype);
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
		if(transpose_out){
			SizeRef outp_shape_ut({X.shape()[0], X.shape()[1]*LKern, L});
			Tensor output = zeros(outp_shape_ut, x.dtype);
			X.arr_void().execute_function(im2col_nn_layer_nd_T, output.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0], dilation[1], x.shape()[-2], x.shape()[-1], X.shape()[1]*LKern, L, X.shape()[1], X.shape()[0]);
			return std::move(output);
		}

		SizeRef outp_shape_ut({X.shape()[0], L, X.shape()[1]*LKern});
		Tensor output = zeros(outp_shape_ut, x.dtype);
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
		if(transpose_out){
			SizeRef outp_shape_ut({LKern, L});
			Tensor output = zeros(outp_shape_ut, x.dtype);
			X.arr_void().execute_function(im2col_nn_layer_2d_T_pad, output.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0], dilation[1], padding[0], padding[1], x.shape()[-2], x.shape()[-1], LKern, L);
			return std::move(output);
		}

		SizeRef outp_shape_ut({L, LKern});
		Tensor output = zeros(outp_shape_ut, x.dtype);
		X.arr_void().execute_function(im2col_nn_layer_2d_pad, output.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0], dilation[1], padding[0], padding[1], x.shape()[-2], x.shape()[-1]);
		return std::move(output);	
	}
		
	if(transpose_out){
		SizeRef outp_shape_ut({LKern, L});
		Tensor output = zeros(outp_shape_ut, x.dtype);
		X.arr_void().execute_function(im2col_nn_layer_2d_T, output.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0], dilation[1], x.shape()[-2], x.shape()[-1], LKern, L);
		return std::move(output);
	}

	SizeRef outp_shape_ut({L, LKern});
	Tensor output = zeros(outp_shape_ut, x.dtype);
	X.arr_void().execute_function(im2col_nn_layer_2d, output.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0], dilation[1], x.shape()[-2], x.shape()[-1]);
	return std::move(output);
}



//this is the backward pass of unfold
Tensor unfold_backward(const Tensor& x, const utils::my_tuple& output_size, const utils::my_tuple& kernel_size, const utils::my_tuple& dilation, const utils::my_tuple& padding, const utils::my_tuple& stride, const bool& transpose_out){
	utils::throw_exception(x.dims() == 2 || x.dims() == 3, "Expected to get a shape with a dimensionality of 3D or 2D, but got $ dimensions", x.dims());
	const int64_t& BROWS = output_size[0];
	const int64_t& BCOLS = output_size[1];
	const int64_t L_r = ((output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
	const int64_t L_c = ((output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
	const int64_t L = L_r * L_c;
	const int64_t LKern = kernel_size[0] * kernel_size[1];
	const int64_t& batches = x.shape()[0];
	utils::throw_exception((L_r*L_c) > 0, "Given input with spatial size ($, $), kernel_size=($, $), dilation=($, $), padding=($, $), calculated shape of the array of sliding blocks as ($, $), but its components must be at least one.", output_size[0], output_size[1], kernel_size[0], kernel_size[1], dilation[0], dilation[1], padding[0], padding[1], L_r, L_c);
	if(transpose_out){
		Tensor Z = (x.dims() == 2) ? x.unsqueeze(0) : x;
		const int64_t channels = Z.shape()[1] / LKern;
		SizeRef output_shape({batches, channels, BROWS, BCOLS});
		utils::throw_exception(Z.shape()[2] == L, "Given output_size=($, $), kernel_size=($, $), dilation=($, $), padding=($, $), stride=($, $), and transpose_out = true, expected size of input's dimension 2 to match the calculated number of sliding blocks $ * $ = $, but got input.size(2)=$.", output_size[0], output_size[1], kernel_size[0], kernel_size[1], dilation[0], dilation[1], padding[0], padding[1], stride[0], stride[1], L_r, L_c, L, Z.shape()[2]);
		Tensor output = zeros(output_shape, x.dtype);
		if(!(padding == 0)){
			output.arr_void().execute_function_nbool(
				 im2col_nn_layer_nd_T_backward,  Z.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0], dilation[1], BROWS, BCOLS, Z.shape()[1], L, channels, batches);
		}
		else{
			output.arr_void().execute_function_nbool(im2col_nn_layer_nd_T_backward,  Z.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0], dilation[1], BROWS, BCOLS, Z.shape()[1], L, channels, batches);
		}
		return std::move(output);
	}
	Tensor Z = (x.dims() == 2) ? x.unsqueeze(0) : x;
	const int64_t channels = Z.shape()[2] / LKern;
	SizeRef output_shape({batches, channels, BROWS, BCOLS});
	utils::throw_exception(Z.shape()[1] == L, "Given output_size=($, $), kernel_size=($, $), dilation=($, $), padding=($, $), stride=($, $), and transpose_out = false, expected size of input's dimension 2 to match the calculated number of sliding blocks $ * $ = $, but got input.size(1)=$.", output_size[0], output_size[1], kernel_size[0], kernel_size[1], dilation[0], dilation[1], padding[0], padding[1], stride[0], stride[1], L_r, L_c, L, Z.shape()[1]);
	Tensor output = zeros(output_shape, x.dtype);
	if(!(padding == 0)){
		output.arr_void().execute_function_nbool(
				im2col_nn_layer_nd_pad_backward, Z.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0], dilation[1], padding[0], padding[1], BROWS, BCOLS, LKern, L, channels, batches);
	}
	else{
		output.arr_void().execute_function_nbool(im2col_nn_layer_nd_backward,  Z.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0], dilation[1], BROWS, BCOLS, LKern, L, channels, batches);
	}
	return std::move(output);
}


void unfold_backward_mem(const Tensor& x, Tensor& output, const utils::my_tuple& output_size, const utils::my_tuple& kernel_size, const utils::my_tuple& dilation, const utils::my_tuple& padding, const utils::my_tuple& stride, const bool& transpose_out){
	utils::throw_exception(x.dims() == 2 || x.dims() == 3, "Expected to get a shape with a dimensionality of 3D or 2D, but got $ dimensions", x.dims());
	const int64_t& BROWS = output_size[0];
	const int64_t& BCOLS = output_size[1];
	const int64_t L_r = ((output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
	const int64_t L_c = ((output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
	const int64_t L = L_r * L_c;
	const int64_t LKern = kernel_size[0] * kernel_size[1];
	const int64_t& batches = x.shape()[0];
	utils::throw_exception((L_r*L_c) > 0, "Given input with spatial size ($, $), kernel_size=($, $), dilation=($, $), padding=($, $), calculated shape of the array of sliding blocks as ($, $), but its components must be at least one.", output_size[0], output_size[1], kernel_size[0], kernel_size[1], dilation[0], dilation[1], padding[0], padding[1], L_r, L_c);
	if(transpose_out){
		Tensor Z = (x.dims() == 2) ? x.unsqueeze(0) : x;
		const int64_t channels = Z.shape()[1] / LKern;
		SizeRef output_shape({batches, channels, BROWS, BCOLS});
		utils::throw_exception(Z.shape()[2] == L, "Given output_size=($, $), kernel_size=($, $), dilation=($, $), padding=($, $), stride=($, $), and transpose_out = true, expected size of input's dimension 2 to match the calculated number of sliding blocks $ * $ = $, but got input.size(2)=$.", output_size[0], output_size[1], kernel_size[0], kernel_size[1], dilation[0], dilation[1], padding[0], padding[1], stride[0], stride[1], L_r, L_c, L, Z.shape()[2]);
		utils::throw_exception(output.shape().multiply() == output_shape.multiply(), "Expected to get same shape for output for unfold backward of $ but got $", output_shape, output.shape());
		utils::throw_exception(output.dtype == x.dtype, "Expected to get same dtype for output as the input of $ but got $", x.dtype, output.dtype);
		if(!(padding == 0)){
			output.arr_void().execute_function_nbool(
				 im2col_nn_layer_nd_T_backward,  Z.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0], dilation[1], BROWS, BCOLS, Z.shape()[1], L, channels, batches);
		}
		else{
			output.arr_void().execute_function_nbool(im2col_nn_layer_nd_T_backward,  Z.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0], dilation[1], BROWS, BCOLS, Z.shape()[1], L, channels, batches);
		}
		return;
	}
	Tensor Z = (x.dims() == 2) ? x.unsqueeze(0) : x;
	const int64_t channels = Z.shape()[2] / LKern;
	SizeRef output_shape({batches, channels, BROWS, BCOLS});
	utils::throw_exception(Z.shape()[1] == L, "Given output_size=($, $), kernel_size=($, $), dilation=($, $), padding=($, $), stride=($, $), and transpose_out = false, expected size of input's dimension 2 to match the calculated number of sliding blocks $ * $ = $, but got input.size(1)=$.", output_size[0], output_size[1], kernel_size[0], kernel_size[1], dilation[0], dilation[1], padding[0], padding[1], stride[0], stride[1], L_r, L_c, L, Z.shape()[1]);
	utils::throw_exception(output.shape().multiply() == output_shape.multiply(), "Expected to get same shape for output for unfold backward of $ but got $", output_shape, output.shape());
	utils::throw_exception(output.dtype == x.dtype, "Expected to get same dtype for output as the input of $ but got $", x.dtype, output.dtype);
	if(!(padding == 0)){
		output.arr_void().execute_function_nbool(
				im2col_nn_layer_nd_pad_backward, Z.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0], dilation[1], padding[0], padding[1], BROWS, BCOLS, LKern, L, channels, batches);
	}
	else{
		output.arr_void().execute_function_nbool(im2col_nn_layer_nd_backward,  Z.arr_void(), kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0], dilation[1], BROWS, BCOLS, LKern, L, channels, batches);
	}
}




TensorGrad unfold(const TensorGrad& x, utils::my_tuple kernel_size, utils::my_tuple dilation, utils::my_tuple padding, utils::my_tuple stride, bool transpose_out){
	TensorGrad result(unfold(x.tensor, kernel_size, dilation, padding, stride, transpose_out));
	result.track_tensors(x);
	result.create_backward_function([kernel_size, dilation, padding, stride, transpose_out]
			(const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
		utils::my_tuple output_size(parents[0]->grad->tensor.shape()[-2], parents[0]->grad->tensor.shape()[-1]);
		unfold_backward_mem(grad, parents[0]->grad->tensor, output_size, kernel_size, dilation, padding, stride, transpose_out);
	});
	return std::move(result);
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

void fold_backward_mem(const Tensor& grad_output, Tensor& output, const utils::my_tuple& output_size, const utils::my_tuple& kernel_size, const utils::my_tuple& dilation, const utils::my_tuple& padding, const utils::my_tuple& stride){
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
}



TensorGrad fold(const TensorGrad& x, utils::my_tuple output_size, utils::my_tuple kernel_size, utils::my_tuple dilation, utils::my_tuple padding, utils::my_tuple stride){
	TensorGrad result(fold(x.tensor, output_size, kernel_size, dilation, padding, stride));
	result.track_tensors(x);
	//it is coppied because the backward pass will go out of scope of this function
	//and so I dont want that memory to try to be referenced
	result.create_backward_function([output_size, kernel_size, dilation, padding, stride]
			(const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
			
		fold_backward_mem(grad, parents[0]->grad->tensor, output_size, kernel_size, dilation, padding, stride);

	});
	return std::move(result);



}


inline static constexpr auto im3col_nn_layer_nd_pad = [](auto a_begin, auto a_end, auto b_begin, 
		const int64_t& k_d, const int64_t& k_r, const int64_t& k_c, 
		const int64_t& s_d, const int64_t& s_r, const int64_t& s_c, 
		const int64_t& d_d, const int64_t& d_r, const int64_t& d_c, 
		const int64_t& p_d, const int64_t& p_r, const int64_t& p_c, 
		const int64_t& t_d, const int64_t& t_r, const int64_t& t_c, 
		const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t a_current_add = (t_c * d_r); 
	int64_t upper_depth2 = t_d - ((k_d - 1) * d_d) + (p_d * 2);
	int64_t upper_row2 = t_r - ((k_r-1) * d_r) + (p_r * 2);
	int64_t upper_col2 = t_c - ((k_c-1) * d_c) + (p_c * 2);
	int64_t kd_d = (k_d * d_d);
	int64_t kd_r = (k_r*d_r);
	int64_t kd_c = (k_c*d_c);
	const int64_t mat_size = t_d * t_c * t_r;
	for(int64_t ba = 0; ba < batches; ++ba){
		int64_t batch_add = (ba * channels * mat_size);
		for(int64_t d = 0; d < upper_depth2; d += s_d){
			for(int64_t r = 0; r < upper_row2; r += s_r){
				for(int64_t c = 0; c < upper_col2; c += s_c){
					for(int64_t ch = 0; ch < channels; ++ch){
						int64_t d_add = d - p_d <= 0 ? 0 : (d - p_d) * t_r * t_c;
						int64_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
						int64_t c_add = c - p_c <= 0 ? 0 : c - p_c;
						auto a_current = a_begin + d_add + r_add + c_add + (mat_size * ch) + batch_add;
						for(int64_t dd = 0; dd < kd_d; dd += d_d){
							if((d - p_d + dd) < 0){
								if((d - p_d + dd + d_d) > 0){
								    a_current += t_r * t_c * (d - p_d + dd + d_d);
								}
								b_begin += k_r * k_c;
								continue;
							    }
							    if((d - p_d + dd) >= t_d){
								b_begin += k_r * k_c * ((k_d - (dd / d_d)));
								break;
							}
							for(int64_t rr = 0; rr < kd_r; rr += d_r){
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
								int64_t pad_counter = 0;
								for(int64_t cc = 0; cc < kd_c; cc += d_c, ++b_begin){
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
		}
	}
};


Tensor unfold3d(const Tensor& x, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> dilation, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> stride, bool transpose_out){
	int64_t LKern = kernel_size[0] * kernel_size[1] * kernel_size[2];

	int64_t L_d = ((x.shape()[-3] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
	int64_t L_r = ((x.shape()[-2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
	int64_t L_c = ((x.shape()[-1] + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) / stride[2]) + 1;
	int64_t L = L_d * L_r * L_c;

	utils::throw_exception(x.dims() >= 4, "Expected to get an input tensor to unfold 3d of at least 4D but got $D", x.dims());
	Tensor X = (x.dims() == 4) ? x.unsqueeze(0) : x.flatten(0, -5);


	utils::throw_exception(L > 0, "Given input with spatial size ($, $, $), kernel_size=($, $, $), dilation=($, $, $), padding=($, $, $), calculated shape of the array of sliding blocks as ($, $, $), but its components must be at least one.", x.shape()[-3], x.shape()[-2], x.shape()[-1], 
			kernel_size[0], kernel_size[1], kernel_size[2],
			dilation[0], dilation[1], dilation[2],
			padding[0], padding[1], padding[2],
			L_d, L_r, L_c);
	
	SizeRef outp_shape_ut({X.shape()[0], L, X.shape()[1]*LKern});
	Tensor output = zeros(outp_shape_ut, x.dtype);
	
	X.arr_void().execute_function(im3col_nn_layer_nd_pad, output.arr_void(), 
				kernel_size[0], kernel_size[1], kernel_size[2],
				stride[0], stride[1], stride[2],
				dilation[0], dilation[1], dilation[2],
				padding[0], padding[1], padding[2],
				X.shape()[-3], X.shape()[-2], X.shape()[-1],
				LKern, L, X.shape()[1], X.shape()[0]);
	
	if(transpose_out)
		output.RowColSwap();
	return std::move(output);
}




inline static constexpr auto im3col_nn_layer_nd_pad_backward = [](auto a_begin, auto a_end, auto b_begin, 
		const int64_t& k_d, const int64_t& k_r, const int64_t& k_c, 
		const int64_t& s_d, const int64_t& s_r, const int64_t& s_c, 
		const int64_t& d_d, const int64_t& d_r, const int64_t& d_c, 
		const int64_t& p_d, const int64_t& p_r, const int64_t& p_c, 
		const int64_t& t_d, const int64_t& t_r, const int64_t& t_c, 
		const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t a_current_add = (t_c * d_r); 
	int64_t upper_depth2 = t_d - ((k_d - 1) * d_d) + (p_d * 2);
	int64_t upper_row2 = t_r - ((k_r-1) * d_r) + (p_r * 2);
	int64_t upper_col2 = t_c - ((k_c-1) * d_c) + (p_c * 2);
	int64_t kd_d = (k_d * d_d);
	int64_t kd_r = (k_r*d_r);
	int64_t kd_c = (k_c*d_c);
	const int64_t mat_size = t_d * t_c * t_r;
	for(int64_t ba = 0; ba < batches; ++ba){
		int64_t batch_add = (ba * channels * mat_size);
		for(int64_t d = 0; d < upper_depth2; d += s_d){
			for(int64_t r = 0; r < upper_row2; r += s_r){
				for(int64_t c = 0; c < upper_col2; c += s_c){
					for(int64_t ch = 0; ch < channels; ++ch){
						int64_t d_add = d - p_d <= 0 ? 0 : (d - p_d) * t_r * t_c;
						int64_t r_add = r - p_r <= 0 ? 0 : (r - p_r) * t_c;
						int64_t c_add = c - p_c <= 0 ? 0 : c - p_c;
						auto a_current = a_begin + d_add + r_add + c_add + (mat_size * ch) + batch_add;
						for(int64_t dd = 0; dd < kd_d; dd += d_d){
							if((d - p_d + dd) < 0){
								if((d - p_d + dd + d_d) > 0){
								    a_current += t_r * t_c * (d - p_d + dd + d_d);
								}
								b_begin += k_r * k_c;
								continue;
							    }
							    if((d - p_d + dd) >= t_d){
								b_begin += k_r * k_c * ((k_d - (dd / d_d)));
								break;
							}
							for(int64_t rr = 0; rr < kd_r; rr += d_r){
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
								int64_t pad_counter = 0;
								for(int64_t cc = 0; cc < kd_c; cc += d_c, ++b_begin){
									if((c - p_c + cc) < t_c && (c - p_c + cc) >= 0){
										/* *b_begin = *a_current; */
										*a_current += *b_begin;
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
		}
	}
};

Tensor unfold3d_backward(const Tensor& x, const utils::my_n_tuple<3>& output_size, const utils::my_n_tuple<3>& kernel_size, const utils::my_n_tuple<3>& dilation, const utils::my_n_tuple<3>& padding, const utils::my_n_tuple<3>& stride, bool transpose_out){

	utils::throw_exception(x.dims() == 2 || x.dims() == 3, "Expected to get a shape with a dimensionality of 3D or 2D, but got $ dimensions", x.dims());
	const int64_t& BDEPTH = output_size[0];
	const int64_t& BROWS = output_size[1];
	const int64_t& BCOLS = output_size[2];
	const int64_t& batches = x.shape()[0];
	int64_t LKern = kernel_size[0] * kernel_size[1] * kernel_size[2];

	int64_t L_d = ((output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
	int64_t L_r = ((output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
	int64_t L_c = ((output_size[2] + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) / stride[2]) + 1;
	int64_t L = L_d * L_r * L_c;
	

	Tensor Z = (x.dims() == 2) ? x.unsqueeze(0) : x;
	if(transpose_out)
		Z = Z.transpose(-1,-2);

	utils::throw_exception(L > 0, "Given input with spatial size ($, $, $), kernel_size=($, $, $), dilation=($, $, $), padding=($, $, $), calculated shape of the array of sliding blocks as ($, $, $), but its components must be at least one.", x.shape()[-3], x.shape()[-2], x.shape()[-1], 
			kernel_size[0], kernel_size[1], kernel_size[2],
			dilation[0], dilation[1], dilation[2],
			padding[0], padding[1], padding[2],
			L_d, L_r, L_c);

	const int64_t channels = Z.shape()[2] / LKern;
	SizeRef output_shape({batches, channels, BDEPTH, BROWS, BCOLS});
	utils::throw_exception(Z.shape()[1] == L, "Given output_size=($, $, $), kernel_size=($, $, $), dilation=($, $, $), padding=($, $, $), stride=($, $, $), and transpose_out = $, expected size of input's dimension 2 to match the calculated number of sliding blocks $ * $ * $ = $, but got input.size(1)=$.", 
			output_size[0], output_size[1], output_size[2], 
			kernel_size[0], kernel_size[1], kernel_size[2],
			dilation[0], dilation[1], dilation[2],
			padding[0], padding[1], padding[2],
			stride[0], stride[1], stride[2],
			(transpose_out) ? "true" : "false",
			L_d, L_r, L_c, L, Z.shape()[1]);
	Tensor output = zeros(output_shape, x.dtype);
	
	Z.arr_void().execute_function_nbool(im3col_nn_layer_nd_pad_backward, output.arr_void(), 
				kernel_size[0], kernel_size[1], kernel_size[2],
				stride[0], stride[1], stride[2],
				dilation[0], dilation[1], dilation[2],
				padding[0], padding[1], padding[2],
				output_size[0], output_size[1], output_size[2],
				LKern, L, channels, batches);
	
	return std::move(output);
}



void unfold3d_backward_mem(const Tensor& x, Tensor& output, const utils::my_n_tuple<3>& output_size, const utils::my_n_tuple<3>& kernel_size, const utils::my_n_tuple<3>& dilation, const utils::my_n_tuple<3>& padding, const utils::my_n_tuple<3>& stride, bool transpose_out){

	utils::throw_exception(x.dims() == 2 || x.dims() == 3, "Expected to get a shape with a dimensionality of 3D or 2D, but got $ dimensions", x.dims());
	const int64_t& BDEPTH = output_size[0];
	const int64_t& BROWS = output_size[1];
	const int64_t& BCOLS = output_size[2];
	const int64_t& batches = x.shape()[0];
	int64_t LKern = kernel_size[0] * kernel_size[1] * kernel_size[2];

	int64_t L_d = ((output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
	int64_t L_r = ((output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
	int64_t L_c = ((output_size[2] + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) / stride[2]) + 1;
	int64_t L = L_d * L_r * L_c;
	

	Tensor Z = (x.dims() == 2) ? x.unsqueeze(0) : x;
	if(transpose_out)
		Z = Z.transpose(-1,-2);

	utils::throw_exception(L > 0, "Given input with spatial size ($, $, $), kernel_size=($, $, $), dilation=($, $, $), padding=($, $, $), calculated shape of the array of sliding blocks as ($, $, $), but its components must be at least one.", x.shape()[-3], x.shape()[-2], x.shape()[-1], 
			kernel_size[0], kernel_size[1], kernel_size[2],
			dilation[0], dilation[1], dilation[2],
			padding[0], padding[1], padding[2],
			L_d, L_r, L_c);

	const int64_t channels = Z.shape()[2] / LKern;
	SizeRef output_shape({batches, channels, BDEPTH, BROWS, BCOLS});
	utils::throw_exception(Z.shape()[1] == L, "Given output_size=($, $, $), kernel_size=($, $, $), dilation=($, $, $), padding=($, $, $), stride=($, $, $), and transpose_out = $, expected size of input's dimension 2 to match the calculated number of sliding blocks $ * $ * $ = $, but got input.size(1)=$.", 
			output_size[0], output_size[1], output_size[2], 
			kernel_size[0], kernel_size[1], kernel_size[2],
			dilation[0], dilation[1], dilation[2],
			padding[0], padding[1], padding[2],
			stride[0], stride[1], stride[2],
			(transpose_out) ? "true" : "false",
			L_d, L_r, L_c, L, Z.shape()[1]);
	
	utils::throw_exception(output_shape.multiply() == output.shape().multiply(), "Expected output shape to match $ but got $ for unfold3d backward", output_shape, output.shape());
	utils::throw_exception(output.dtype == x.dtype, "Expected to get a dtype matching the grad input for unfold3d backward of $ but got $", x.dtype, output.dtype);
	
	Z.arr_void().execute_function_nbool(im3col_nn_layer_nd_pad_backward, output.arr_void(), 
				kernel_size[0], kernel_size[1], kernel_size[2],
				stride[0], stride[1], stride[2],
				dilation[0], dilation[1], dilation[2],
				padding[0], padding[1], padding[2],
				output_size[0], output_size[1], output_size[2],
				LKern, L, channels, batches);
	
}


TensorGrad unfold3d(const TensorGrad& x, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> dilation, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> stride, bool transpose_out){
	TensorGrad result(unfold3d(x.tensor, kernel_size, dilation, padding, stride, transpose_out));
	result.track_tensors(x);
	result.create_backward_function([kernel_size, dilation, padding, stride, transpose_out]
			(const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
		utils::my_n_tuple<3> output_size(parents[0]->grad->tensor.shape()[-3], parents[0]->grad->tensor.shape()[-2], parents[0]->grad->tensor.shape()[-1]);
		unfold3d_backward_mem(grad, parents[0]->grad->tensor, output_size, kernel_size, dilation, padding, stride, transpose_out);
	});
	return std::move(result);

}




inline static constexpr auto im1col_nn_layer_nd_pad = [](auto a_begin, auto a_end, auto b_begin, 
		const int64_t& k_c, 
		const int64_t& s_c, 
		const int64_t& d_c, 
		const int64_t& p_c, 
		const int64_t& t_c, 
		const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t upper_col2 = t_c - ((k_c-1) * d_c) + (p_c * 2);
	int64_t kd_c = (k_c*d_c);
	const int64_t& mat_size = t_c;
	for(int64_t ba = 0; ba < batches; ++ba){
		int64_t batch_add = (ba * channels * mat_size);
		for(int64_t c = 0; c < upper_col2; c += s_c){
			for(int64_t ch = 0; ch < channels; ++ch){
				int64_t c_add = c - p_c <= 0 ? 0 : c - p_c;
				auto a_current = a_begin + c_add + (mat_size * ch) + batch_add;
				for(int64_t cc = 0; cc < kd_c; cc += d_c, ++b_begin){
					if((c - p_c + cc) < t_c && (c - p_c + cc) >= 0){
						*b_begin = *a_current;
						a_current += d_c;
					}
					else if((c - p_c + cc + d_c) > 0 && (c - p_c + cc + d_c) < t_c){
						a_current += (c - p_c + cc + d_c);
					}
				}
			}
		}
	}
};


Tensor unfold1d(const Tensor& x, Tensor::size_value_t kernel_size, Tensor::size_value_t dilation, Tensor::size_value_t padding, Tensor::size_value_t stride, bool transpose_out){
	const int64_t& LKern = kernel_size;

	int64_t L_c = ((x.shape()[-1] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
	const int64_t& L = L_c;

	utils::throw_exception(x.dims() >= 2, "Expected to get an input tensor to unfold 1d of at least 2D but got $D", x.dims());
	Tensor X = (x.dims() == 2) ? x.unsqueeze(0) : x.flatten(0, -3);


	utils::throw_exception(L > 0, "Given input with spatial size ($), kernel_size=($), dilation=($), padding=($), calculated shape of the array of sliding blocks as ($), but its components must be at least one.", x.shape()[-1], 
			kernel_size,
			dilation,
			padding,
			L_c);
	
	SizeRef outp_shape_ut({X.shape()[0], L, X.shape()[1]*LKern});
	Tensor output = zeros(outp_shape_ut, x.dtype);
	
	X.arr_void().execute_function(im1col_nn_layer_nd_pad, output.arr_void(), 
				kernel_size,
				stride,
				dilation,
				padding,
				X.shape()[-1],
				LKern, L, X.shape()[1], X.shape()[0]);
	
	if(transpose_out)
		output.RowColSwap();
	return std::move(output);
}


inline static constexpr auto im1col_nn_layer_nd_pad_backward = [](auto a_begin, auto a_end, auto b_begin, 
		const int64_t& k_c, 
		const int64_t& s_c, 
		const int64_t& d_c, 
		const int64_t& p_c, 
		const int64_t& t_c, 
		const int64_t& LKern, const int64_t& L, const int64_t& channels, const int64_t& batches){
	int64_t upper_col2 = t_c - ((k_c-1) * d_c) + (p_c * 2);
	int64_t kd_c = (k_c*d_c);
	const int64_t& mat_size = t_c;
	for(int64_t ba = 0; ba < batches; ++ba){
		int64_t batch_add = (ba * channels * mat_size);
		for(int64_t c = 0; c < upper_col2; c += s_c){
			for(int64_t ch = 0; ch < channels; ++ch){
				int64_t c_add = c - p_c <= 0 ? 0 : c - p_c;
				auto a_current = a_begin + c_add + (mat_size * ch) + batch_add;
				for(int64_t cc = 0; cc < kd_c; cc += d_c, ++b_begin){
					if((c - p_c + cc) < t_c && (c - p_c + cc) >= 0){
						/* *b_begin = *a_current; */
						*a_current += *b_begin;
						a_current += d_c;
					}
					else if((c - p_c + cc + d_c) > 0 && (c - p_c + cc + d_c) < t_c){
						a_current += (c - p_c + cc + d_c);
					}
				}
			}
		}
	}
};


Tensor unfold1d_backward(const Tensor& x, Tensor::size_value_t output_size, Tensor::size_value_t kernel_size, Tensor::size_value_t dilation, Tensor::size_value_t padding, Tensor::size_value_t stride, bool transpose_out){

	utils::throw_exception(x.dims() == 2 || x.dims() == 3, "Expected to get a shape with a dimensionality of 3D or 2D, but got $ dimensions", x.dims());
	const int64_t& BCOLS = output_size;
	const int64_t& batches = x.shape()[0];
	int64_t LKern = kernel_size;

	int64_t L_c = ((output_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
	int64_t L = L_c;
	

	Tensor Z = (x.dims() == 2) ? x.unsqueeze(0) : x;
	if(transpose_out)
		Z = Z.transpose(-1,-2);

	utils::throw_exception(L > 0, "Given input with spatial size ($), kernel_size=($), dilation=($), padding=($), calculated shape of the array of sliding blocks as ($), but its components must be at least one.", x.shape()[-1], 
			kernel_size,
			dilation,
			padding,
			L_c);

	const int64_t channels = Z.shape()[2] / LKern;
	SizeRef output_shape({batches, channels, BCOLS});
	utils::throw_exception(Z.shape()[1] == L, "Given output_size=($), kernel_size=($), dilation=($), padding=($), stride=($), and transpose_out = $, expected size of input's dimension 2 to match the calculated number of sliding blocks $, but got input.size(1)=$.", 
			output_size, 
			kernel_size,
			dilation,
			padding,
			stride,
			(transpose_out) ? "true" : "false",
			L, Z.shape()[1]);
	Tensor output = zeros(output_shape, x.dtype);
	
	Z.arr_void().execute_function_nbool(im1col_nn_layer_nd_pad_backward, output.arr_void(), 
				kernel_size,
				stride,
				dilation,
				padding,
				output_size,
				LKern, L, channels, batches);
	
	return std::move(output);
}


void unfold1d_backward_mem(const Tensor& x, Tensor& output, Tensor::size_value_t output_size, Tensor::size_value_t kernel_size, Tensor::size_value_t dilation, Tensor::size_value_t padding, Tensor::size_value_t stride, bool transpose_out){

	utils::throw_exception(x.dims() == 2 || x.dims() == 3, "Expected to get a shape with a dimensionality of 3D or 2D, but got $ dimensions", x.dims());
	const int64_t& BCOLS = output_size;
	const int64_t& batches = x.shape()[0];
	int64_t LKern = kernel_size;

	int64_t L_c = ((output_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
	int64_t L = L_c;
	

	Tensor Z = (x.dims() == 2) ? x.unsqueeze(0) : x;
	if(transpose_out)
		Z = Z.transpose(-1,-2);

	utils::throw_exception(L > 0, "Given input with spatial size ($), kernel_size=($), dilation=($), padding=($), calculated shape of the array of sliding blocks as ($), but its components must be at least one.", x.shape()[-1], 
			kernel_size,
			dilation,
			padding,
			L_c);

	const int64_t channels = Z.shape()[2] / LKern;
	SizeRef output_shape({batches, channels, BCOLS});
	utils::throw_exception(Z.shape()[1] == L, "Given output_size=($), kernel_size=($), dilation=($), padding=($), stride=($), and transpose_out = $, expected size of input's dimension 2 to match the calculated number of sliding blocks $, but got input.size(1)=$.", 
			output_size, 
			kernel_size,
			dilation,
			padding,
			stride,
			(transpose_out) ? "true" : "false",
			L, Z.shape()[1]);
	utils::throw_exception(output_shape.multiply() == output.shape().multiply(), "Expected output shape to match $ but got $ for unfold1d backward", output_shape, output.shape());
	utils::throw_exception(output.dtype == x.dtype, "Expected to get a dtype matching the grad input for unfold1d backward of $ but got $", x.dtype, output.dtype);
	
	Z.arr_void().execute_function_nbool(im1col_nn_layer_nd_pad_backward, output.arr_void(), 
				kernel_size,
				stride,
				dilation,
				padding,
				output_size,
				LKern, L, channels, batches);
	
}

TensorGrad unfold1d(const TensorGrad& x, Tensor::size_value_t kernel_size, Tensor::size_value_t dilation, Tensor::size_value_t padding, Tensor::size_value_t stride, bool transpose_out){
	TensorGrad result(unfold1d(x.tensor, kernel_size, dilation, padding, stride, transpose_out));
	result.track_tensors(x);
	result.create_backward_function([kernel_size, dilation, padding, stride, transpose_out]
			(const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
		Tensor::size_value_t output_size = parents[0]->grad->tensor.shape()[-1];
		unfold1d_backward_mem(grad, parents[0]->grad->tensor, output_size, kernel_size, dilation, padding, stride, transpose_out);
	});
	return std::move(result);

}





inline static constexpr auto col2im3d_cpu = [](auto a_begin, auto a_end, auto o_begin,
		const int64_t& col_add, const int64_t& L_d, const int64_t& L_r, const int64_t& L_c,
		const int64_t& o_d, const int64_t& o_r, const int64_t& o_c,
		const int64_t& k_d, const int64_t k_r, const int64_t k_c,
		const int64_t& s_d, const int64_t& s_r, const int64_t& s_c,
		const int64_t& d_d, const int64_t& d_r, const int64_t& d_c,
		const int64_t& p_d, const int64_t p_r, const int64_t p_c,
		const int64_t& batches, const int64_t& channels){
	const int64_t& depth_upper_limit = k_d;
	const int64_t& row_upper_limit = k_r;
	const int64_t& col_upper_limit = k_c;
	const int64_t& kernel_depth_upper_limit = L_d;
	const int64_t& kernel_row_upper_limit = L_r;
	const int64_t& kernel_col_upper_limit = L_c;
	const int64_t o_channel_add = o_d * o_r * o_c;
	const int64_t a_channel_add = k_d * k_r * k_c * col_add;
	for(int64_t n = 0; n < batches; ++n){
		for(int64_t ch = 0; ch < channels; ++ch, o_begin += o_channel_add, a_begin += a_channel_add){
			for(int64_t d = 0; d < depth_upper_limit; ++d){
				for(int64_t r = 0; r < row_upper_limit; ++r){
					for(int64_t c = 0; c < col_upper_limit; ++c){
						int64_t o_depth = (d * s_d) + (c * d_d) - p_d;
						if(o_depth >= o_d || o_depth < 0){ continue; }
						int64_t o_row = (r * s_r) + (c * d_r) - p_r;
						if(o_row >= o_r || o_row < 0){ continue; }
						int64_t a_col_index = d * L_r * L_c + r * L_c;
						int64_t a_row_index = c * L_d * L_r;
						auto o_cpy = o_begin + (o_depth * o_r * o_c) + (o_row * o_c);
						auto a_cpy = a_begin + (a_row_index * col_add + a_col_index);
						for(int64_t kd = 0; kd < kernel_depth_upper_limit; ++kd){
							for(int64_t kr = 0; kr < kernel_row_upper_limit; ++kr){
								for(int64_t kc = 0; kc < kernel_col_upper_limit; ++kc){
									int64_t o_col = (kc * d_c) + (kr * s_c) - p_c;
									if(o_col < o_c && o_col >= 0){
										o_cpy[o_col] += a_cpy[col_add * (kd * L_r * L_c + kr * L_c + kc)];
									}
								}
							}
						}
					}
				}
			}
		}
	}
};

inline static constexpr auto col3im_cpu = [](auto a_begin, auto a_end, auto o_begin,
		const int64_t& col_add, const int64_t& L_d, const int64_t& L_r, const int64_t& L_c,
		const int64_t& o_d, const int64_t& o_r, const int64_t& o_c,
		const int64_t& k_d, const int64_t& k_r, const int64_t k_c,
		const int64_t& s_d, const int64_t& s_r, const int64_t& s_c,
		const int64_t& d_d, const int64_t& d_r, const int64_t& d_c,
		const int64_t& p_d, const int64_t& p_r, const int64_t p_c,
		const int64_t& batches, const int64_t& channels){
	const int64_t& depth_upper_limit = k_d;
	const int64_t& row_upper_limit = k_c;
	const int64_t& col_upper_limit = k_r;
	const int64_t& kernel_depth_upper_limit = L_d;
	const int64_t& kernel_row_upper_limit = L_c;
	const int64_t& kernel_col_upper_limit = L_r;
	const int64_t o_channel_add = o_d * o_r * o_c;
	const int64_t a_channel_add = k_d * k_r * k_c * col_add;
	for(int64_t n = 0; n < batches; ++n){
		for(int64_t ch = 0; ch < channels; ++ch, o_begin += o_channel_add, a_begin += a_channel_add){
			for(int64_t d = 0; d < depth_upper_limit; ++d){
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
	}

};


//basically just one line to switch:
inline static constexpr auto col3im_cpu_backward = [](auto a_begin, auto a_end, auto o_begin,
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


}
}
