#ifndef _NT_MATMULT_HPP_
#define _NT_MATMULT_HPP_
#include <iostream>
#include <string.h>
#include <random>

#include <chrono>
#include <cstdlib> // For std::aligned_alloc
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/blocked_range3d.h>
#include <tbb/global_control.h>
#include <tbb/task_group.h>
#include <functional>
#include <unistd.h>
#include "nt_matmult_macros.h"
//#include "matmult_simde_fma_avx.hpp" <- old
#include "nt_matmult_blocks.h"
#include "nt_kmatmult_simde.hpp"
#include "nt_matmult_simde.hpp"


namespace nt{
namespace functional{
namespace std_functional{




// Zero out an array using SIMD instructions
template<typename T>
void zero_memory(T* array, size_t num_elements) {
    constexpr size_t simd_size = mp::pack_size_v<T>; // simde__m256 processes 8 floats at a time
    size_t i = 0;

    //create a zero vector
    mp::simde_type<T> zero = mp::SimdTraits<T>::zero();
    for(;i < num_elements; i += simd_size){
	    if constexpr (std::is_integral<T>::value){
		    mp::SimdTraits<T>::store(reinterpret_cast<mp::simde_type<T>*>(array+i), zero);
	    }else{
		    mp::SimdTraits<T>::store(array+i, zero);
	    }
    }
}

template<typename T, size_t num_elements>
void kzero_memory(T* array) {
    constexpr size_t simd_size = mp::pack_size_v<T>; // simde__m256 processes 8 floats at a time
    size_t i = 0;

    // Create a zero vector
    mp::simde_type<T> zero = mp::SimdTraits<T>::zero();
    for(;i < num_elements; i += simd_size){
	    mp::SimdTraits<T>::store(array+i, zero);
    }
}



/* template <typename T> */
/* static T* get_blockA_packed() { */
/*     static_assert(simde_supported_v<T>, "Expected supported type"); */
/*     if constexpr (std::is_same_v<T, float>) { */
/*         return blockA_packed_float; */
/*     } else if constexpr (std::is_same_v<T, double>) { */
/*         return blockA_packed_double; */
/*     } else if constexpr (std::is_same_v<T, int8_t>) { */
/*         return blockA_packed_int8_t; */
/*     } else if constexpr (std::is_same_v<T, uint8_t>) { */
/*         return blockA_packed_uint8_t; */
/*     } else if constexpr (std::is_same_v<T, int16_t>) { */
/*         return blockA_packed_int16_t; */
/*     } else if constexpr (std::is_same_v<T, uint16_t>) { */
/*         return blockA_packed_uint16_t; */
/*     } else if constexpr (std::is_same_v<T, int32_t>) { */
/*         return blockA_packed_int32_t; */
/*     } else if constexpr (std::is_same_v<T, uint32_t>) { */
/*         return blockA_packed_uint32_t; */
/*     } else if constexpr (std::is_same_v<T, int64_t>) { */
/*         return blockA_packed_int64_t; */
/*     } else if constexpr (std::is_same_v<T, uint64_t>) { */
/*         return blockA_packed_uint64_t; */
/*     } else { */
/*         static_assert(simde_supported_v<T>, "Unsupported type"); */
/*     } */
/* } */


/* template <typename T> */
/* static T* get_blockB_packed() { */
/*     static_assert(simde_supported_v<T>, "Expected supported type"); */
/*     if constexpr (std::is_same_v<T, float>) { */
/*         return blockB_packed_float; */
/*     } else if constexpr (std::is_same_v<T, double>) { */
/*         return blockB_packed_double; */
/*     } else if constexpr (std::is_same_v<T, int8_t>) { */
/*         return blockB_packed_int8_t; */
/*     } else if constexpr (std::is_same_v<T, uint8_t>) { */
/*         return blockB_packed_uint8_t; */
/*     } else if constexpr (std::is_same_v<T, int16_t>) { */
/*         return blockB_packed_int16_t; */
/*     } else if constexpr (std::is_same_v<T, uint16_t>) { */
/*         return blockB_packed_uint16_t; */
/*     } else if constexpr (std::is_same_v<T, int32_t>) { */
/*         return blockB_packed_int32_t; */
/*     } else if constexpr (std::is_same_v<T, uint32_t>) { */
/*         return blockB_packed_uint32_t; */
/*     } else if constexpr (std::is_same_v<T, int64_t>) { */
/*         return blockB_packed_int64_t; */
/*     } else if constexpr (std::is_same_v<T, uint64_t>) { */
/*         return blockB_packed_uint64_t; */
/*     } else { */
/*         static_assert(simde_supported_v<T>, "Unsupported type"); */
/*     } */
/* } */



/* template<typename T> */
/* void print_matrix(const T* M, int64_t rows, int64_t cols) { */
/*     for (int i = 0; i < rows; ++i) { */
/*         for (int j = 0; j < cols; ++j) { */
/*             std::cout << M[i * cols + j] << " "; */
/*         } */
/*         std::cout << std::endl; */
/*     } */
/* } */

/* bool has_avx_support() { */
/* #if defined(__GNUC__) || defined(__clang__) */
/*     return __builtin_cpu_supports("avx"); */
/* #else */
/*     return false; // or use an alternative approach on non-GNU/Clang compilers */
/* #endif */
/* } */

/* bool has_avx2_support() { */
/* #if defined(__GNUC__) || defined(__clang__) */
/*     return __builtin_cpu_supports("avx2"); */
/* #else */
/*     return false; // or use an alternative approach on non-GNU/Clang compilers */
/* #endif */
/* } */

/* bool has_avx512_support() { */
/* #if defined(__GNUC__) || defined(__clang__) */
/*     return __builtin_cpu_supports("avx512f"); */
/* #else */
/*     return false; // or use an alternative approach on non-GNU/Clang compilers */
/* #endif */
/* } */


template<typename T, size_t COLS>
inline void kpack_rowA_threaded(const T* src_matrix, T* block_matrix, int64_t src_cols) noexcept {
	int64_t max = _NT_MATMULT_MIN_(COLS, src_cols);
	for(int i = 0; i < max; ++i)
		*block_matrix++ = src_matrix[i];
	for(int i = max; i < COLS; ++i)
		*block_matrix++ = 0;
}


//example use:
//kpack_blockA_threaded<T, TILE_SIZE * _NT_MATMULT_NTHREADS_, TILE_SIZE * _NT_MATMULT_NTHREADS_>(src + (start_row * src_cols), blockA_packed, start_col, src_rows - start_row, src_cols)
template<typename T, size_t ROWS, size_t COLS>
inline void kpack_blockA_threaded(const T* src_matrix, T* block_matrix, int64_t start_col, int64_t src_rows, int64_t src_cols) noexcept{
	int64_t max = _NT_MATMULT_MIN_(ROWS, src_rows);
	int64_t col_amt = src_cols - start_col;
	tbb::parallel_for(tbb::blocked_range<int64_t>(0, max),
			[&](const tbb::blocked_range<int64_t>& r){
	for(int i = r.begin(); i < r.end(); ++i){
		kpack_rowA_threaded<T, COLS>(src_matrix + (i * src_cols) + start_col, block_matrix + (COLS * i), col_amt);
	}
	});
	zero_memory(block_matrix + (COLS * max), (ROWS - max) * COLS);
}


template<typename T, size_t COLS>
inline void kpack_rowB_threaded(const T* src_matrix, T* block_matrix, const int64_t& row_amt, const int64_t& src_cols, const int64_t& cur_row) noexcept {
	int64_t max = _NT_MATMULT_MIN_(COLS, src_cols);
	for(int i = 0; i < max; ++i)
		block_matrix[i * COLS + cur_row]  = *src_matrix++;
}


//example use:
//kpack_blockB_threaded<TILE_SIZE * _NT_MATMULT_NTHREADS_, TILE_SIZE * _NT_MATMULT_NTHREADS_>(src + (start_row * src_cols), blockA_packed, start_col, src_rows - start_row, src_cols)
template<typename T, size_t ROWS, size_t COLS>
inline void kpack_blockB_threaded(const T* src_matrix, T* block_matrix, int64_t start_col, int64_t src_rows, int64_t src_cols) noexcept {
	int64_t max = _NT_MATMULT_MIN_(ROWS, src_rows); //row_amt
	int64_t col_amt = src_cols - start_col;
	tbb::parallel_for(tbb::blocked_range<int64_t>(0, max),
			[&](const tbb::blocked_range<int64_t>& r){
	for(int i = r.begin(); i < r.end(); ++i){
		kpack_rowB_threaded<T, COLS>(src_matrix + (i * src_cols) + start_col, block_matrix, max, col_amt, i);
	}
	});
	zero_memory(block_matrix + (COLS * max), (ROWS - max) * COLS);

}



template<typename T, size_t TILE_SIZE, size_t left_bc, size_t left_ar, size_t addition, size_t a_pack_cols, bool parallel, std::enable_if_t<left_bc == 0 && left_ar == 0, bool> = true>
inline void perform_dots(const T* A, const T* B, T* C, const int64_t& src_c_cols, const int64_t& j, const int64_t& i, const int64_t& irMax, const int64_t& irnmax, const int64_t& jrMax, const int64_t& jrnmax) noexcept{
	if constexpr(parallel){
	tbb::parallel_for(tbb::blocked_range2d<int64_t>(0, irMax/TILE_SIZE, 0, jrMax/TILE_SIZE),
		[&](const tbb::blocked_range2d<int64_t>& range){
	for(int64_t ir = range.rows().begin(); ir < range.rows().end(); ++ir){
		for(int64_t jr = range.cols().begin(); jr < range.cols().end(); ++jr){
			kmatmult_simdeT_threaded_fma<T, TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE, addition>(
				&A[(ir* TILE_SIZE) * a_pack_cols], &B[jr * TILE_SIZE], C + ((i+(ir*TILE_SIZE)) * src_c_cols) + (j+(jr*TILE_SIZE)), src_c_cols);
		}
	}
	});
	}else{
		for(int64_t ir = 0; ir < irMax; ir += TILE_SIZE){
			for(int64_t jr = 0; jr < jrMax; jr += TILE_SIZE){
				kmatmult_simdeT_threaded_fma<T, TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE, addition>(
					&A[ir * a_pack_cols], &B[jr], C + ((i+ir) * src_c_cols) + (j+jr), src_c_cols);
				
			}
		}
	}
}

template<typename T, size_t TILE_SIZE, size_t left_bc, size_t left_ar, size_t addition, size_t a_pack_cols, bool parallel, std::enable_if_t<(left_bc == 0 && left_ar > 0), bool> = true>
inline void perform_dots(const T* A, const T* B, T* C, const int64_t& src_c_cols, const int64_t& j, const int64_t& i, const int64_t& irMax, const int64_t& irnmax, const int64_t& jrMax, const int64_t& jrnmax) noexcept{
	
	if constexpr(parallel){
	const int64_t irmax_t = irMax / TILE_SIZE;
	tbb::parallel_for(tbb::blocked_range2d<int64_t>(0, (irMax == irnmax) ? irmax_t : irmax_t + 1, 0, jrMax/TILE_SIZE),
		[&](const tbb::blocked_range2d<int64_t>& range){
	int64_t ir;
	for(int64_t ir = range.rows().begin(); ir < range.rows().end(); ++ir){
		if(ir == irmax_t){
			for(int64_t jr = range.cols().begin(); jr < range.cols().end(); ++jr){
				kmatmult_simdeT_threaded_fma<T, left_ar, TILE_SIZE, TILE_SIZE, TILE_SIZE, addition>(
					&A[(ir* TILE_SIZE) * a_pack_cols], &B[jr * TILE_SIZE], C + ((i+(ir*TILE_SIZE)) * src_c_cols) + (j+(jr*TILE_SIZE)), src_c_cols);
			}
		}else{
			for(int64_t jr = range.cols().begin(); jr < range.cols().end(); ++jr){
				kmatmult_simdeT_threaded_fma<T, TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE, addition>(
					&A[(ir* TILE_SIZE) * a_pack_cols], &B[jr * TILE_SIZE], C + ((i+(ir*TILE_SIZE)) * src_c_cols) + (j+(jr*TILE_SIZE)), src_c_cols);
			}
		
		}
	}
	});
	}else{
		for(int64_t ir = 0; ir < irMax; ir += TILE_SIZE){
			for(int64_t jr = 0; jr < jrMax; jr += TILE_SIZE){
				kmatmult_simdeT_threaded_fma<T, TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE, addition>(
					&A[ir * a_pack_cols], &B[jr], C + ((i+ir) * src_c_cols) + (j+jr), src_c_cols);
				
			}
		}
		for(int64_t ir = irMax; ir < irnmax; ir += TILE_SIZE){
			for(int64_t jr = 0; jr < jrMax; jr += TILE_SIZE){
				kmatmult_simdeT_threaded_fma<T, left_ar, TILE_SIZE, TILE_SIZE, TILE_SIZE, addition>(
					&A[ir * a_pack_cols], &B[jr], C + ((i+ir) * src_c_cols) + (j+jr), src_c_cols);
				
			}
	
		}
	}
}

template<typename T, size_t TILE_SIZE, size_t left_bc, size_t left_ar, size_t addition, size_t a_pack_cols, bool parallel, std::enable_if_t<(left_bc > 0 && left_ar == 0), bool> = true>
inline void perform_dots(const T* A, const T* B, T* C, const int64_t& src_c_cols, const int64_t& j, const int64_t& i, const int64_t& irMax, const int64_t& irnmax, const int64_t& jrMax, const int64_t& jrnmax) noexcept{
	
	if constexpr(parallel){
	const int64_t jrmax_t = jrMax / TILE_SIZE;
	tbb::parallel_for(tbb::blocked_range2d<int64_t>(0, (jrMax == jrnmax) ? jrmax_t : jrmax_t + 1, 0, irMax/TILE_SIZE),
		[&](const tbb::blocked_range2d<int64_t>& range){
	for(int64_t jr = range.rows().begin(); jr < range.rows().end(); ++jr){
		if(jr == jrmax_t){
			for(int64_t ir = range.cols().begin(); ir < range.cols().end(); ++ir){
				kmatmult_simdeT_threaded_fma<T, TILE_SIZE, TILE_SIZE, TILE_SIZE, left_bc, addition>(
					&A[(ir* TILE_SIZE) * a_pack_cols], &B[jr * TILE_SIZE], C + ((i+(ir*TILE_SIZE)) * src_c_cols) + (j+(jr*TILE_SIZE)), src_c_cols);
			}
	
		}else{
			for(int64_t ir = range.cols().begin(); ir < range.cols().end(); ++ir){
				kmatmult_simdeT_threaded_fma<T, TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE, addition>(
					&A[(ir* TILE_SIZE) * a_pack_cols], &B[jr * TILE_SIZE], C + ((i+(ir*TILE_SIZE)) * src_c_cols) + (j+(jr*TILE_SIZE)), src_c_cols);
			}

		}
	}
	});
	}else{
		for(int64_t ir = 0; ir < irMax; ir += TILE_SIZE){
			for(int64_t jr = 0; jr < jrMax; jr += TILE_SIZE){
				kmatmult_simdeT_threaded_fma<T, TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE, addition>(
					&A[ir * a_pack_cols], &B[jr], C + ((i+ir) * src_c_cols) + (j+jr), src_c_cols);
				
			}
			for(int64_t jr = jrMax; jr < jrnmax; jr += TILE_SIZE){
				kmatmult_simdeT_threaded_fma<T, TILE_SIZE, TILE_SIZE, TILE_SIZE, left_bc, addition>(
					&A[ir * a_pack_cols], &B[jr], C + ((i+ir) * src_c_cols) + (j+jr), src_c_cols);
				
			}
		}
	}
}


template<typename T, size_t TILE_SIZE, size_t left_bc, size_t addition>
inline void initiate_matmult_ntg(const T* A, const T* B, T* C, const int64_t& src_c_cols, const int64_t& j, const int64_t& jrMax, const int64_t& jrnmax) noexcept {
	if constexpr (left_bc > 0){
		int64_t jr;
		for(jr = 0; jr < jrMax; jr += TILE_SIZE){
			kmatmult_simdeT_threaded_fma<T, TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE, addition>(
				A, &B[jr], C + (j+jr), src_c_cols);
		}
		for(jr = jrMax; jr < jrnmax; jr +=TILE_SIZE){
			kmatmult_simdeT_threaded_fma<T, TILE_SIZE, TILE_SIZE, TILE_SIZE, left_bc, addition>(
				A, &B[jr], C + (j+jr), src_c_cols);
			
		}
	}else{
		int64_t jr;
		for(jr = 0; jr < jrMax; jr += TILE_SIZE){
			kmatmult_simdeT_threaded_fma<T, TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE, addition>(
				A, &B[jr], C + (j+jr), src_c_cols);
		}
		
	}

}

template<typename T, size_t TILE_SIZE, size_t left_bc, size_t left_ar, size_t addition>
inline void initiate_matmult_ntg_lar(const T* A, const T* B, T* C, const int64_t& src_c_cols, const int64_t& j, const int64_t& jrMax, const int64_t& jrnmax) noexcept {
	if constexpr (left_bc > 0){
		int64_t jr;
		for(jr = 0; jr < jrMax; jr += TILE_SIZE){
			kmatmult_simdeT_threaded_fma<T, left_ar, TILE_SIZE, TILE_SIZE, TILE_SIZE, addition>(
				A, &B[jr], C + (j+jr), src_c_cols);
		}
		for(jr = jrMax; jr < jrnmax; jr +=TILE_SIZE){
			kmatmult_simdeT_threaded_fma<T, left_ar, TILE_SIZE, TILE_SIZE, left_bc, addition>(
				A, &B[jr], C + (j+jr), src_c_cols);
			
		}
	}else{
		int64_t jr;
		for(jr = 0; jr < jrMax; jr += TILE_SIZE){
			kmatmult_simdeT_threaded_fma<T, left_ar, TILE_SIZE, TILE_SIZE, TILE_SIZE, addition>(
				A, &B[jr], C + (j+jr), src_c_cols);
		}
	}


}

template<typename T, size_t TILE_SIZE, size_t left_bc, size_t left_ar, size_t addition, size_t a_pack_cols, bool parallel, std::enable_if_t<(left_bc > 0 && left_ar > 0), bool> = true>
inline void perform_dots(const T* A, const T* B, T* C, const int64_t& src_c_cols, const int64_t& j, const int64_t& i, const int64_t& irMax, const int64_t& irnmax, const int64_t& jrMax, const int64_t& jrnmax) noexcept{
	
	if constexpr(parallel){
	tbb::parallel_for(tbb::blocked_range2d<int64_t>(0, jrMax / TILE_SIZE, 0, irMax/TILE_SIZE),
		[&](const tbb::blocked_range2d<int64_t>& range){
	for(int64_t jr = range.rows().begin(); jr < range.rows().end(); ++jr){
		for(int64_t ir = range.cols().begin(); ir < range.cols().end(); ++ir){
			kmatmult_simdeT_threaded_fma<T, TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE, addition>(
				&A[(ir* TILE_SIZE) * a_pack_cols], &B[jr * TILE_SIZE], C + ((i+(ir*TILE_SIZE)) * src_c_cols) + (j+(jr*TILE_SIZE)), src_c_cols);
		}

	}
	});

	for(int64_t jr = jrMax; jr < jrnmax; jr += TILE_SIZE){
		tbb::parallel_for(tbb::blocked_range<int64_t>(0, irMax/TILE_SIZE),
		[&](const tbb::blocked_range<int64_t>& range){
		for(int64_t ir = range.begin(); ir < range.end(); ++ir){
			kmatmult_simdeT_threaded_fma<T, TILE_SIZE, TILE_SIZE, TILE_SIZE, left_bc, addition>(
				&A[(ir* TILE_SIZE) * a_pack_cols], &B[jr], C + ((i+(ir*TILE_SIZE)) * src_c_cols) + (j+jr), src_c_cols);
			
		}
		});
	}
	for(int64_t ir = irMax; ir < irnmax; ir += TILE_SIZE){
		tbb::parallel_for(tbb::blocked_range<int64_t>(0, jrMax/TILE_SIZE),
		[&](const tbb::blocked_range<int64_t>& range){
		for(int64_t jr = range.begin(); jr < range.end(); ++jr){
			kmatmult_simdeT_threaded_fma<T, left_ar, TILE_SIZE, TILE_SIZE, TILE_SIZE, addition>(
				&A[(ir) * a_pack_cols], &B[jr * TILE_SIZE], C + ((i+(ir)) * src_c_cols) + (j+(jr*TILE_SIZE)), src_c_cols);
			
		}
		});
	}
	for(int64_t jr = jrMax; jr < jrnmax; jr += TILE_SIZE){
		for(int64_t ir = irMax; ir < irnmax; ir += TILE_SIZE){
			kmatmult_simdeT_threaded_fma<T, left_ar, TILE_SIZE, TILE_SIZE, left_bc, addition>(
				&A[(ir) * a_pack_cols], &B[jr], C + ((i+(ir)) * src_c_cols) + (j+jr), src_c_cols);
			
		}
	}
	}else{
		for(int64_t ir = 0; ir < irMax; ir += TILE_SIZE){
			initiate_matmult_ntg<T, TILE_SIZE, left_bc, addition>(&A[ir * a_pack_cols], B, C + ((i+ir) * src_c_cols), src_c_cols, j, jrMax, jrnmax);
		}
		for(int64_t ir = irMax; ir < irnmax; ir += TILE_SIZE){
			initiate_matmult_ntg_lar<T, TILE_SIZE, left_bc, left_ar, addition>(&A[ir * a_pack_cols], B, C + ((i+ir) * src_c_cols), src_c_cols, j, jrMax, jrnmax);
		}
	}
}


template<typename T>
void transpose_RowColSwap(T* first, T* last, const int64_t& n, const int64_t& mn1, const int64_t& total)
{
    std::vector<bool> visited(total);
    T* cycle = first;
    while (++cycle != last) {
        if (visited[cycle - first])
            continue;
        int a = cycle - first;
        do  {
            a = a == mn1 ? mn1 : (n * a) % mn1;
            std::swap(*(first + a), *cycle);
            visited[a] = true;
        } while ((first + a) != cycle);
    }
}

//if either matrix needs to be transposed, send in the transposed matrix size
template<typename T, size_t left_ar, size_t left_bc, bool parallel = true>
void kpack_multiply_directly_var_threaded(const T* A, const T* B, T* C, const int64_t a_rows, const int64_t a_cols, const int64_t b_rows, const int64_t b_cols, bool transpose_a = false, bool transpose_b = false){
	constexpr size_t TILE_SIZE = tile_size_v<T>;
	tbb::global_control control(tbb::global_control::max_allowed_parallelism, _NT_MATMULT_NTHREADS_);
	/* tbb::task_group pool; */
	/* ThreadPool pool(_NT_MATMULT_NTHREADS_); */

	//going to handle this differently than before, where pretty much everything now just gets packed
	//also, B no longer gets transposed
	T* blockA_packed = get_blockA_packed<T>();
	T* blockB_packed = get_blockB_packed<T>();
	//TODO: this is very inefficient, and there is a much better way to handle this
	const int64_t a_total = a_rows * a_cols;
	const int64_t t_a_rows = a_cols;
	const int64_t a_mn1 = a_total-1;
	const int64_t b_total = b_rows * b_cols;
	const int64_t t_b_rows = b_cols;
	const int64_t b_mn1 = b_total-1;

	if(transpose_a){
		transpose_RowColSwap(const_cast<T*>(A), const_cast<T*>(A) + a_total, a_mn1, t_a_rows, a_total);
	}
	if(transpose_b){
		transpose_RowColSwap(const_cast<T*>(B), const_cast<T*>(B) + b_total, b_mn1, t_b_rows, a_total);
	}
	//TODO: eliminate the need for the above
	constexpr size_t a_pack_rows = (TILE_SIZE * _NT_MATMULT_NTHREADS_ * _NT_MATMULT_NTHREADS_);
	constexpr size_t a_pack_cols = TILE_SIZE;
	
	constexpr size_t b_pack_rows = (TILE_SIZE);
	constexpr size_t b_pack_cols = (TILE_SIZE * _NT_MATMULT_NTHREADS_ * _NT_MATMULT_NTHREADS_);

	constexpr size_t ALIGNED_SIZE = _NT_MATMULT_ENSURE_ALIGNMENT_(T, 64, a_pack_rows * a_pack_cols);

	const int64_t i_max = a_rows - left_ar;
	const int64_t j_max = b_cols - left_bc;

	const int64_t jr_size = _NT_MATMULT_DETERMINE_SIZE_(b_cols, b_pack_cols);
	const int64_t ir_size = _NT_MATMULT_DETERMINE_SIZE_(a_rows, a_pack_rows);

	
	int64_t jrMaxs[jr_size];
	int64_t jrLefts[jr_size];
	for(int i = 0; i < jr_size; ++i){
		jrMaxs[i] = b_pack_cols;
		jrLefts[i] = 0;
	}
	
	int64_t irMaxs[ir_size];
	int64_t irLefts[ir_size];
	for(int i = 0; i < ir_size; ++i){
		irMaxs[i] = a_pack_rows;
		irLefts[i] = 0;
	}

	if(b_cols % b_pack_cols != 0){
		jrMaxs[jr_size-1] = (b_cols % b_pack_cols) - left_bc;
		jrLefts[jr_size-1] = left_bc;
	}
	if(a_rows % a_pack_rows != 0){
		irMaxs[ir_size-1] = (a_rows % a_pack_rows) - left_ar;
		irLefts[ir_size-1] = left_ar;
	}

	int64_t k;
	int64_t j;
	int64_t i;
	int64_t ir, kr;
	int64_t j_counter = 0;
	int64_t i_counter = 0;
	for( k = 0; k < a_cols; k += TILE_SIZE){
		const int64_t krMax = _NT_MATMULT_MIN_(TILE_SIZE, a_cols - k);
		i_counter = 0;
		for(i = 0; i < a_rows; i += a_pack_rows, ++i_counter){
			const int64_t& irMax = irMaxs[i_counter];
			const int64_t& irLeft = irLefts[i_counter];
			const int64_t ir_nmax = irMax + irLeft;
			kpack_blockA_threaded<T, a_pack_rows, a_pack_cols>(A + (i * a_cols), blockA_packed, k, a_rows - i, a_cols);
			j_counter = 0;
			for(j = 0; j < b_cols; j += b_pack_cols, ++j_counter){
				const int64_t& jrMax = jrMaxs[j_counter];
				const int64_t& jrLeft = jrLefts[j_counter];
				kpack_blockA_threaded<T, b_pack_rows, b_pack_cols>(B + (k * b_cols), blockB_packed, j, b_rows - k, b_cols); //no longer transposed
				const int64_t jr_nmax = jrMax + jrLeft;
				perform_dots<T, TILE_SIZE, left_bc, left_ar, b_pack_cols, a_pack_cols, parallel>(blockA_packed, blockB_packed, C, b_cols, j, i, irMax, ir_nmax, jrMax, jr_nmax);	
			}
		}
	}
	//TODO: this is very inefficient, and there is a much better way to handle this
	if(transpose_a){
		transpose_RowColSwap(const_cast<T*>(A), const_cast<T*>(A) + a_total, a_mn1, a_rows, a_total);
	}
	if(transpose_b){
		transpose_RowColSwap(const_cast<T*>(B), const_cast<T*>(B) + b_total, b_mn1, b_rows, a_total);
	}
	//TODO: eliminate the need for the above


}


template<typename T, size_t left_ar, size_t left_bc, bool parallel = true>
void kpack_multiply_directly_var_threaded_batch(const T** A_batch, const T** B_batch, T** C_batch, const int64_t batches, const int64_t a_rows, const int64_t a_cols, const int64_t b_rows, const int64_t b_cols, bool transpose_a = false, bool transpose_b = false){
	constexpr size_t TILE_SIZE = tile_size_v<T>;
	/* tbb::global_control control(tbb::global_control::max_allowed_parallelism, _NT_MATMULT_NTHREADS_); */
	/* tbb::task_group pool; */
	/* ThreadPool pool(_NT_MATMULT_NTHREADS_); */

	//going to handle this differently than before, where pretty much everything now just gets packed
	//also, B no longer gets transposed
	/* T* blockA_packed = get_blockA_packed<T>(); */
	/* T* blockB_packed = get_blockB_packed<T>(); */
	//TODO: this is very inefficient, and there is a much better way to handle this


	if(transpose_a){
		const int64_t a_total = a_rows * a_cols;
		const int64_t t_a_rows = a_cols;
		const int64_t a_mn1 = a_total-1;
		const int64_t b_total = b_rows * b_cols;
		const int64_t t_b_rows = b_cols;
		const int64_t b_mn1 = b_total-1;
		tbb::parallel_for(tbb::blocked_range<int64_t>(0, batches),
			[&](const tbb::blocked_range<int64_t>& range){
				for(int64_t i = range.begin(); i < range.end(); ++i){
					transpose_RowColSwap(const_cast<T*>(A_batch[i]), const_cast<T*>(A_batch[i]) + (a_total), a_mn1, t_a_rows, a_total);
				}
		});
	}
	if(transpose_b){
		const int64_t a_total = a_rows * a_cols;
		const int64_t t_a_rows = a_cols;
		const int64_t a_mn1 = a_total-1;
		const int64_t b_total = b_rows * b_cols;
		const int64_t t_b_rows = b_cols;
		const int64_t b_mn1 = b_total-1;

		tbb::parallel_for(tbb::blocked_range<int64_t>(0, batches),
			[&](const tbb::blocked_range<int64_t>& range){
				for(int64_t i = range.begin(); i < range.end(); ++i){
					transpose_RowColSwap(const_cast<T*>(B_batch[i]), const_cast<T*>(B_batch[i]) + (b_total), b_mn1, t_b_rows, b_total);
				}
		});

	}
	//TODO: eliminate the need for the above
	/* constexpr size_t a_pack_rows = (TILE_SIZE * _NT_MATMULT_NTHREADS_ * _NT_MATMULT_NTHREADS_); */
	/* constexpr size_t a_pack_cols = TILE_SIZE; */
	
	/* constexpr size_t b_pack_rows = (TILE_SIZE); */
	/* constexpr size_t b_pack_cols = (TILE_SIZE * _NT_MATMULT_NTHREADS_ * _NT_MATMULT_NTHREADS_); */

	/* constexpr size_t ALIGNED_SIZE = _NT_MATMULT_ENSURE_ALIGNMENT_(T, 64, a_pack_rows * a_pack_cols); */

	/* const int64_t i_max = a_rows - left_ar; */
	/* const int64_t j_max = b_cols - left_bc; */

	/* const int64_t jr_size = _NT_MATMULT_DETERMINE_SIZE_(b_cols, b_pack_cols); */
	/* const int64_t ir_size = _NT_MATMULT_DETERMINE_SIZE_(a_rows, a_pack_rows); */

	
	/* int64_t jrMaxs[jr_size]; */
	/* int64_t jrLefts[jr_size]; */
	/* for(int i = 0; i < jr_size; ++i){ */
	/* 	jrMaxs[i] = b_pack_cols; */
	/* 	jrLefts[i] = 0; */
	/* } */
	
	/* int64_t irMaxs[ir_size]; */
	/* int64_t irLefts[ir_size]; */
	/* for(int i = 0; i < ir_size; ++i){ */
	/* 	irMaxs[i] = a_pack_rows; */
	/* 	irLefts[i] = 0; */
	/* } */

	/* if(b_cols % b_pack_cols != 0){ */
	/* 	jrMaxs[jr_size-1] = (b_cols % b_pack_cols) - left_bc; */
	/* 	jrLefts[jr_size-1] = left_bc; */
	/* } */
	/* if(a_rows % a_pack_rows != 0){ */
	/* 	irMaxs[ir_size-1] = (a_rows % a_pack_rows) - left_ar; */
	/* 	irLefts[ir_size-1] = left_ar; */
	/* } */

	/* int64_t k; */
	/* int64_t j; */
	/* int64_t i; */
	/* int64_t ir, kr; */
	/* int64_t j_counter = 0; */
	/* int64_t i_counter = 0; */
	/* for(int64_t b = 0; b < batches; ++b){ */
	/* 	const T* A = A_batch[b]; */
	/* 	const T* B = B_batch[b]; */
	/* 	T* C = C_batch[b]; */
	/* 	for( k = 0; k < a_cols; k += TILE_SIZE){ */
	/* 		const int64_t krMax = _NT_MATMULT_MIN_(TILE_SIZE, a_cols - k); */
	/* 		i_counter = 0; */
	/* 		for(i = 0; i < a_rows; i += a_pack_rows, ++i_counter){ */
	/* 			const int64_t& irMax = irMaxs[i_counter]; */
	/* 			const int64_t& irLeft = irLefts[i_counter]; */
	/* 			const int64_t ir_nmax = irMax + irLeft; */
	/* 			kpack_blockA_threaded<T, a_pack_rows, a_pack_cols>(A + (i * a_cols), blockA_packed, k, a_rows - i, a_cols); */
	/* 			j_counter = 0; */
	/* 			for(j = 0; j < b_cols; j += b_pack_cols, ++j_counter){ */
	/* 				const int64_t& jrMax = jrMaxs[j_counter]; */
	/* 				const int64_t& jrLeft = jrLefts[j_counter]; */
	/* 				kpack_blockA_threaded<T, b_pack_rows, b_pack_cols>(B + (k * b_cols), blockB_packed, j, b_rows - k, b_cols); //no longer transposed */
	/* 				const int64_t jr_nmax = jrMax + jrLeft; */
	/* 				perform_dots<T, TILE_SIZE, left_bc, left_ar, b_pack_cols, a_pack_cols, parallel>(blockA_packed, blockB_packed, C, b_cols, j, i, irMax, ir_nmax, jrMax, jr_nmax); */	
	/* 			} */
	/* 		} */
	/* 	} */
	/* } */
	for(int64_t b = 0; b < batches; ++b){
		kpack_multiply_directly_var_threaded<T, left_ar, left_bc, parallel>(A_batch[b], B_batch[b], C_batch[b], a_rows, a_cols, b_rows, b_cols);
	}
	//TODO: this is very inefficient, and there is a much better way to handle this
	if(transpose_a){
		const int64_t a_total = a_rows * a_cols;
		const int64_t t_a_rows = a_cols;
		const int64_t a_mn1 = a_total-1;
		const int64_t b_total = b_rows * b_cols;
		const int64_t t_b_rows = b_cols;
		tbb::parallel_for(tbb::blocked_range<int64_t>(0, batches),
			[&](const tbb::blocked_range<int64_t>& range){
				for(int64_t i = range.begin(); i < range.end(); ++i){
					transpose_RowColSwap(const_cast<T*>(A_batch[i]), const_cast<T*>(A_batch[i]) + (a_total), a_mn1, a_rows, a_total);
				}
		});
	}
	if(transpose_b){
		const int64_t a_total = a_rows * a_cols;
		const int64_t t_a_rows = a_cols;
		const int64_t a_mn1 = a_total-1;
		const int64_t b_total = b_rows * b_cols;
		const int64_t t_b_rows = b_cols;
		const int64_t b_mn1 = b_total-1;
		tbb::parallel_for(tbb::blocked_range<int64_t>(0, batches),
			[&](const tbb::blocked_range<int64_t>& range){
				for(int64_t i = range.begin(); i < range.end(); ++i){
					transpose_RowColSwap(const_cast<T*>(B_batch[i]), const_cast<T*>(B_batch[i]) + (b_total), b_mn1, b_rows, b_total);
				}
		});

	}
	//TODO: eliminate the need for the above


}


template<typename T, size_t left_ar, size_t left_bc>
inline void kmatmult_simde_step(const T* A, const T* B, T* C, const int64_t a_rows, const int64_t a_cols, const int64_t b_rows, const int64_t b_cols, bool transpose_a, bool transpose_b, const size_t left_ar_max, const size_t left_bc_max) noexcept {
	constexpr size_t TILE_SIZE = tile_size_v<T>;
	if constexpr (left_ar < (TILE_SIZE-1)){
		if(left_ar < left_ar_max){
			return kmatmult_simde_step<T, left_ar+1, left_bc>(A, B, C, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b, left_ar_max, left_bc_max);
		}
	}
	if constexpr (left_bc < (TILE_SIZE-1)){
		if(left_bc < left_bc_max){
			return kmatmult_simde_step<T, left_ar, left_bc+1>(A, B, C, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b, left_ar_max, left_bc_max);
		}
	}
	kpack_multiply_directly_var_threaded<T, left_ar, left_bc>(A, B, C, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b);
}




template<typename T, size_t left_ar, size_t left_bc>
inline void kmatmult_simde_step_batch(const T** A, const T** B, T** C, const int64_t batches, const int64_t a_rows, const int64_t a_cols, const int64_t b_rows, const int64_t b_cols, bool transpose_a, bool transpose_b, const size_t left_ar_max, const size_t left_bc_max) noexcept {
	constexpr size_t TILE_SIZE = tile_size_v<T>;
	if constexpr (left_ar < (TILE_SIZE-1)){
		if(left_ar < left_ar_max){
			return kmatmult_simde_step_batch<T, left_ar+1, left_bc>(A, B, C, batches, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b, left_ar_max, left_bc_max);
		}
	}
	if constexpr (left_bc < (TILE_SIZE-1)){
		if(left_bc < left_bc_max){
			return kmatmult_simde_step_batch<T, left_ar, left_bc+1>(A, B, C, batches, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b, left_ar_max, left_bc_max);
		}
	}
	kpack_multiply_directly_var_threaded_batch<T, left_ar, left_bc>(A, B, C, batches, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b);
}



template<typename T>
inline void kmatmult_simde(const T* A, const T* B, T* C, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b) noexcept {
	
	size_t left_ar = (a_rows % tile_size_v<T>);
	/* size_t left_ac = (a_cols % tile_size_v<T>); */
	/* size_t left_br = (b_rows % TILE_SIZE); */ //should be same as left_ac
	size_t left_bc = (b_cols % tile_size_v<T>);
	kmatmult_simde_step<T, 0, 0>(A, B, C, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b, left_ar, left_bc);

}

template<typename T>
inline void kmatmult_simde_batch(const T** A, const T** B, T** C, int64_t batches, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b) noexcept {
	
	size_t left_ar = (a_rows % tile_size_v<T>);
	/* size_t left_ac = (a_cols % tile_size_v<T>); */
	/* size_t left_br = (b_rows % TILE_SIZE); */ //should be same as left_ac
	size_t left_bc = (b_cols % tile_size_v<T>);
	kmatmult_simde_step_batch<T, 0, 0>(A, B, C, batches, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b, left_ar, left_bc);

}

template<typename T>
void pack_multiply_directly_var_threaded(const T* A, const T* B, T* C, const int64_t a_rows, const int64_t a_cols, const int64_t b_rows, const int64_t b_cols, bool transpose_a = false, bool transpose_b = false){
	constexpr size_t TILE_SIZE = tile_size_v<T>;
	tbb::global_control control(tbb::global_control::max_allowed_parallelism, _NT_MATMULT_NTHREADS_);
	/* tbb::task_group pool; */
	/* ThreadPool pool(NTHREADS); */

	//going to handle this differently than before, where pretty much everything now just gets packed
	//also, B no longer gets transposed
	T* blockA_packed = get_blockA_packed<T>();
	T* blockB_packed = get_blockB_packed<T>();

	const int64_t a_total = a_rows * a_cols;
	const int64_t t_a_rows = a_cols;
	const int64_t a_mn1 = a_total-1;
	const int64_t b_total = b_rows * b_cols;
	const int64_t t_b_rows = b_cols;
	const int64_t b_mn1 = b_total-1;

	//TODO: this is very inefficient, and there is a much better way to handle this
	if(transpose_a){
		transpose_RowColSwap(const_cast<T*>(A), const_cast<T*>(A) + a_total, a_mn1, t_a_rows, a_total);
	}
	if(transpose_b){
		transpose_RowColSwap(const_cast<T*>(B), const_cast<T*>(B) + b_total, b_mn1, t_b_rows, b_total);
	}
	//TODO: eliminate the need for the above
	constexpr size_t a_pack_rows = (TILE_SIZE * _NT_MATMULT_NTHREADS_ * _NT_MATMULT_NTHREADS_);
	constexpr size_t a_pack_cols = TILE_SIZE;
	
	constexpr size_t b_pack_rows = (TILE_SIZE);
	constexpr size_t b_pack_cols = (TILE_SIZE * _NT_MATMULT_NTHREADS_ * _NT_MATMULT_NTHREADS_);

	constexpr size_t ALIGNED_SIZE = _NT_MATMULT_ENSURE_ALIGNMENT_(T, 64, a_pack_rows * a_pack_cols);

	/* const int64_t i_max = a_rows - left_ar; */
	/* const int64_t j_max = b_cols - left_bc; */

	const int64_t jr_size = _NT_MATMULT_DETERMINE_SIZE_(b_cols, b_pack_cols);
	const int64_t ir_size = _NT_MATMULT_DETERMINE_SIZE_(a_rows, a_pack_rows);

	//so the point of this is to have the max's there so that divides don't have tp happen later
	size_t jrMaxs[jr_size];
	size_t jrTBBs[jr_size];
	for(int i = 0; i < jr_size; ++i){
		jrMaxs[i] = b_pack_cols;
		jrTBBs[i] = b_pack_cols / TILE_SIZE;
	}
	//basically if it is not divisible, and there is anything left over
	if(b_cols % b_pack_cols != 0){
		jrMaxs[jr_size-1] = (b_cols - (b_pack_cols * (jr_size-1)));
		if(b_cols % TILE_SIZE == 0){
			jrTBBs[jr_size-1] = ((b_cols - (b_pack_cols * (jr_size-1))) / TILE_SIZE);
		}else{
			size_t start = (b_cols - (b_pack_cols * (jr_size-1)));
			jrTBBs[jr_size-1] = (start < TILE_SIZE ? 1 : (start + (TILE_SIZE % b_cols)) / TILE_SIZE);
			
		}
	}
	
	size_t irMaxs[ir_size];
	size_t irTBBs[ir_size];
	for(int i = 0; i < ir_size; ++i){
		irMaxs[i] = a_pack_rows;
		irTBBs[i] = a_pack_rows / TILE_SIZE;
	}
	if(a_rows % a_pack_rows != 0){
		irMaxs[ir_size-1] = (a_rows - (a_pack_rows * (ir_size-1)));
		if(a_rows % TILE_SIZE == 0){
			irTBBs[ir_size-1] = (a_rows - (a_pack_rows * (ir_size-1))) / TILE_SIZE;
		}else{
			size_t start = (a_rows - (a_pack_rows * (ir_size-1)));
			irTBBs[ir_size-1] = (start < TILE_SIZE ? 1 : (start + (TILE_SIZE % a_rows)) / TILE_SIZE);
			
		}
	}

	int64_t k;
	int64_t j;
	int64_t i;
	/* int64_t ir, kr; */
	int64_t j_counter = 0;
	int64_t i_counter = 0;
	for( k = 0; k < a_cols; k += TILE_SIZE){
		const int64_t krMax = _NT_MATMULT_MIN_(TILE_SIZE, a_cols - k);
		i_counter = 0;
		for(i = 0; i < a_rows; i += a_pack_rows, ++i_counter){
			const size_t& irMax = irMaxs[i_counter];
			kpack_blockA_threaded<T, a_pack_rows, a_pack_cols>(A + (i * a_cols), blockA_packed, k, a_rows - i, a_cols);
			j_counter = 0;
			for(j = 0; j < b_cols; j += b_pack_cols, ++j_counter){
				const size_t& jrMax = jrMaxs[j_counter];
				kpack_blockA_threaded<T, b_pack_rows, b_pack_cols>(B + (k * b_cols), blockB_packed, j, b_rows - k, b_cols); //no longer transposed

				tbb::parallel_for(tbb::blocked_range2d<size_t>(0, irTBBs[i_counter], 0, jrTBBs[j_counter]),
					[&](const tbb::blocked_range2d<size_t>& range){
				for(size_t ir = range.rows().begin(); ir < range.rows().end(); ++ir){
					const size_t na_rows = _NT_MATMULT_MIN_(TILE_SIZE, irMax - (ir * TILE_SIZE));
					for(size_t jr = range.cols().begin(); jr < range.cols().end(); ++jr){
						const size_t nb_cols = _NT_MATMULT_MIN_(TILE_SIZE, jrMax - (jr * TILE_SIZE));
						matmult_simdeT_directly_threaded<T, b_pack_cols>
							(&blockA_packed[(ir * TILE_SIZE) * a_pack_cols], &blockB_packed[jr * TILE_SIZE], C + ((i + (ir * TILE_SIZE)) * b_cols) + (j+(jr * TILE_SIZE)), b_cols,
							 nb_cols, na_rows);
					}
				}
				});
			}
		}
	}
	//TODO: this is very inefficient, and there is a much better way to handle this
	if(transpose_a){
		transpose_RowColSwap(const_cast<T*>(A), const_cast<T*>(A) + a_total, a_mn1, a_rows, a_total);
	}
	if(transpose_b){
		transpose_RowColSwap(const_cast<T*>(B), const_cast<T*>(B) + b_total, b_mn1, b_rows, b_total);
	}
	//TODO: eliminate the need for the above
}


template<typename T>
void pack_multiply_directly_var_threaded_batch(const T** A_batch, const T** B_batch, T** C_batch, const int64_t batches, const int64_t a_rows, const int64_t a_cols, const int64_t b_rows, const int64_t b_cols, bool transpose_a = false, bool transpose_b = false){
	constexpr size_t TILE_SIZE = tile_size_v<T>;
	/* tbb::global_control control(tbb::global_control::max_allowed_parallelism, _NT_MATMULT_NTHREADS_); */
	/* tbb::task_group pool; */
	/* ThreadPool pool(NTHREADS); */

	//going to handle this differently than before, where pretty much everything now just gets packed
	//also, B no longer gets transposed
	/* T* blockA_packed = get_blockA_packed<T>(); */
	/* T* blockB_packed = get_blockB_packed<T>(); */

	//TODO: this is very inefficient, and there is a much better way to handle this
	

	if(transpose_a){
		const int64_t a_total = a_rows * a_cols;
		const int64_t t_a_rows = a_cols;
		const int64_t a_mn1 = a_total-1;
		const int64_t b_total = b_rows * b_cols;
		const int64_t t_b_rows = b_cols;
		const int64_t b_mn1 = b_total-1;
		tbb::parallel_for(tbb::blocked_range<int64_t>(0, batches),
			[&](const tbb::blocked_range<int64_t>& range){
				for(int64_t i = range.begin(); i < range.end(); ++i){
					transpose_RowColSwap(const_cast<T*>(A_batch[i]), const_cast<T*>(A_batch[i]) + (a_total), a_mn1, t_a_rows, a_total);
				}
		});
	}
	if(transpose_b){
		const int64_t a_total = a_rows * a_cols;
		const int64_t t_a_rows = a_cols;
		const int64_t a_mn1 = a_total-1;
		const int64_t b_total = b_rows * b_cols;
		const int64_t t_b_rows = b_cols;
		const int64_t b_mn1 = b_total-1;
		tbb::parallel_for(tbb::blocked_range<int64_t>(0, batches),
			[&](const tbb::blocked_range<int64_t>& range){
				for(int64_t i = range.begin(); i < range.end(); ++i){
					transpose_RowColSwap(const_cast<T*>(B_batch[i]), const_cast<T*>(B_batch[i]) + (b_total), b_mn1, t_b_rows, b_total);
				}
		});

	}
	//TODO: eliminate the need for the above
	/* constexpr size_t a_pack_rows = (TILE_SIZE * _NT_MATMULT_NTHREADS_ * _NT_MATMULT_NTHREADS_); */
	/* constexpr size_t a_pack_cols = TILE_SIZE; */
	
	/* constexpr size_t b_pack_rows = (TILE_SIZE); */
	/* constexpr size_t b_pack_cols = (TILE_SIZE * _NT_MATMULT_NTHREADS_ * _NT_MATMULT_NTHREADS_); */

	/* constexpr size_t ALIGNED_SIZE = _NT_MATMULT_ENSURE_ALIGNMENT_(T, 64, a_pack_rows * a_pack_cols); */

	/* /1* const int64_t i_max = a_rows - left_ar; *1/ */
	/* /1* const int64_t j_max = b_cols - left_bc; *1/ */

	/* const int64_t jr_size = _NT_MATMULT_DETERMINE_SIZE_(b_cols, b_pack_cols); */
	/* const int64_t ir_size = _NT_MATMULT_DETERMINE_SIZE_(a_rows, a_pack_rows); */

	/* //so the point of this is to have the max's there so that divides don't have tp happen later */
	/* size_t jrMaxs[jr_size]; */
	/* size_t jrTBBs[jr_size]; */
	/* for(int i = 0; i < jr_size; ++i){ */
	/* 	jrMaxs[i] = b_pack_cols; */
	/* 	jrTBBs[i] = b_pack_cols / TILE_SIZE; */
	/* } */
	/* //basically if it is not divisible, and there is anything left over */
	/* if(b_cols % b_pack_cols != 0){ */
	/* 	jrMaxs[jr_size-1] = (b_cols - (b_pack_cols * (jr_size-1))); */
	/* 	if(b_cols % TILE_SIZE == 0){ */
	/* 		jrTBBs[jr_size-1] = ((b_cols - (b_pack_cols * (jr_size-1))) / TILE_SIZE); */
	/* 	}else{ */
	/* 		size_t start = (b_cols - (b_pack_cols * (jr_size-1))); */
	/* 		jrTBBs[jr_size-1] = (start < TILE_SIZE ? 1 : (start + (TILE_SIZE % b_cols)) / TILE_SIZE); */
			
	/* 	} */
	/* } */
	
	/* size_t irMaxs[ir_size]; */
	/* size_t irTBBs[ir_size]; */
	/* for(int i = 0; i < ir_size; ++i){ */
	/* 	irMaxs[i] = a_pack_rows; */
	/* 	irTBBs[i] = a_pack_rows / TILE_SIZE; */
	/* } */
	/* if(a_rows % a_pack_rows != 0){ */
	/* 	irMaxs[ir_size-1] = (a_rows - (a_pack_rows * (ir_size-1))); */
	/* 	if(a_rows % TILE_SIZE == 0){ */
	/* 		irTBBs[ir_size-1] = (a_rows - (a_pack_rows * (ir_size-1))) / TILE_SIZE; */
	/* 	}else{ */
	/* 		size_t start = (a_rows - (a_pack_rows * (ir_size-1))); */
	/* 		irTBBs[ir_size-1] = (start < TILE_SIZE ? 1 : (start + (TILE_SIZE % a_rows)) / TILE_SIZE); */
			
	/* 	} */
	/* } */

	/* int64_t k; */
	/* int64_t j; */
	/* int64_t i; */
	/* /1* int64_t ir, kr; *1/ */
	/* int64_t j_counter = 0; */
	/* int64_t i_counter = 0; */
	/* for(int64_t b = 0; b < batches; ++b){ */
	/* 	const T* A = A_batch[b]; */
	/* 	const T* B = B_batch[b]; */
	/* 	T* C = C_batch[b]; */
	/* 	for( k = 0; k < a_cols; k += TILE_SIZE){ */
	/* 		const int64_t krMax = _NT_MATMULT_MIN_(TILE_SIZE, a_cols - k); */
	/* 		i_counter = 0; */
	/* 		for(i = 0; i < a_rows; i += a_pack_rows, ++i_counter){ */
	/* 			const size_t& irMax = irMaxs[i_counter]; */
	/* 			kpack_blockA_threaded<T, a_pack_rows, a_pack_cols>(A + (i * a_cols), blockA_packed, k, a_rows - i, a_cols); */
	/* 			j_counter = 0; */
	/* 			for(j = 0; j < b_cols; j += b_pack_cols, ++j_counter){ */
	/* 				const size_t& jrMax = jrMaxs[j_counter]; */
	/* 				kpack_blockA_threaded<T, b_pack_rows, b_pack_cols>(B + (k * b_cols), blockB_packed, j, b_rows - k, b_cols); //no longer transposed */

	/* 				tbb::parallel_for(tbb::blocked_range2d<size_t>(0, irTBBs[i_counter], 0, jrTBBs[j_counter]), */
	/* 					[&](const tbb::blocked_range2d<size_t>& range){ */
	/* 				for(size_t ir = range.rows().begin(); ir < range.rows().end(); ++ir){ */
	/* 					const size_t na_rows = _NT_MATMULT_MIN_(TILE_SIZE, irMax - (ir * TILE_SIZE)); */
	/* 					for(size_t jr = range.cols().begin(); jr < range.cols().end(); ++jr){ */
	/* 						const size_t nb_cols = _NT_MATMULT_MIN_(TILE_SIZE, jrMax - (jr * TILE_SIZE)); */
	/* 						matmult_simdeT_directly_threaded<T, b_pack_cols> */
	/* 							(&blockA_packed[(ir * TILE_SIZE) * a_pack_cols], &blockB_packed[jr * TILE_SIZE], C + ((i + (ir * TILE_SIZE)) * b_cols) + (j+(jr * TILE_SIZE)), b_cols, */
	/* 							 nb_cols, na_rows); */
	/* 					} */
	/* 				} */
	/* 				}); */
	/* 			} */
	/* 		} */
	/* 	} */
	/* } */
	for(int64_t b = 0; b < batches; ++b){
		pack_multiply_directly_var_threaded(A_batch[b], B_batch[b], C_batch[b], a_rows, a_cols, b_rows, b_cols, false, false);
	}
	//TODO: this is very inefficient, and there is a much better way to handle this
	if(transpose_a){
		const int64_t a_total = a_rows * a_cols;
		const int64_t t_a_rows = a_cols;
		const int64_t a_mn1 = a_total-1;
		const int64_t b_total = b_rows * b_cols;
		const int64_t t_b_rows = b_cols;
		const int64_t b_mn1 = b_total-1;

		tbb::parallel_for(tbb::blocked_range<int64_t>(0, batches),
			[&](const tbb::blocked_range<int64_t>& range){
				for(int64_t i = range.begin(); i < range.end(); ++i){
					transpose_RowColSwap(const_cast<T*>(A_batch[i]), const_cast<T*>(A_batch[i]) + (a_total), a_mn1, a_rows, a_total);
				}
		});
	}
	if(transpose_b){
		const int64_t a_total = a_rows * a_cols;
		const int64_t t_a_rows = a_cols;
		const int64_t a_mn1 = a_total-1;
		const int64_t b_total = b_rows * b_cols;
		const int64_t t_b_rows = b_cols;
		const int64_t b_mn1 = b_total-1;

		tbb::parallel_for(tbb::blocked_range<int64_t>(0, batches),
			[&](const tbb::blocked_range<int64_t>& range){
				for(int64_t i = range.begin(); i < range.end(); ++i){
					transpose_RowColSwap(const_cast<T*>(B_batch[i]), const_cast<T*>(B_batch[i]) + (b_total), b_mn1, b_rows, b_total);
				}
		});

	}
	//TODO: eliminate the need for the above


}


template<typename T>
void matmult_naive(const T* A, const T* B, T* C, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b) {
  	const int64_t a_total = a_rows * a_cols;
	const int64_t t_a_rows = a_cols;
	const int64_t a_mn1 = a_total-1;
	const int64_t b_total = b_rows * b_cols;
	const int64_t t_b_rows = b_cols;
	const int64_t b_mn1 = b_total-1;

	if(transpose_a){
		transpose_RowColSwap(const_cast<T*>(A), const_cast<T*>(A) + a_total, a_mn1, t_a_rows, a_total);
	}
	if(transpose_b){
		transpose_RowColSwap(const_cast<T*>(B), const_cast<T*>(B) + b_total, b_mn1, t_b_rows, b_total);
	}
	tbb::parallel_for(tbb::blocked_range2d<int64_t>(0, a_rows, 0, b_cols),
		[&](const tbb::blocked_range2d<int64_t>& range){
	for (int64_t i = range.rows().begin(); i < range.rows().end(); ++i) {
		for (int64_t j = range.cols().begin(); j < range.cols().end(); ++j) {
			for (int64_t k = 0; k < a_cols; ++k) {
				C[i * b_cols + j] += A[i * a_cols + k] * B[k * b_cols + j];
			}
		}
	}});	
	if(transpose_a){
		transpose_RowColSwap(const_cast<T*>(A), const_cast<T*>(A) + a_total, a_mn1, a_rows, a_total);
	}
	if(transpose_b){
		transpose_RowColSwap(const_cast<T*>(B), const_cast<T*>(B) + b_total, b_mn1, b_rows, b_total);
	}
}

template<typename T>
void matmult_naive_batch(const T** A_batch, const T** B_batch, T** C_batch, int64_t batches, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b) {
  	const int64_t a_total = a_rows * a_cols;
	const int64_t t_a_rows = a_cols;
	const int64_t a_mn1 = a_total-1;
	const int64_t b_total = b_rows * b_cols;
	const int64_t t_b_rows = b_cols;
	const int64_t b_mn1 = b_total-1;

	if(transpose_a){
		tbb::parallel_for(tbb::blocked_range<int64_t>(0, batches),
			[&](const tbb::blocked_range<int64_t>& range){
				for(int64_t i = range.begin(); i < range.end(); ++i){
					transpose_RowColSwap(const_cast<T*>(A_batch[i]), const_cast<T*>(A_batch[i]) + (a_total), a_mn1, t_a_rows, a_total);
				}
		});
	}
	if(transpose_b){
		tbb::parallel_for(tbb::blocked_range<int64_t>(0, batches),
			[&](const tbb::blocked_range<int64_t>& range){
				for(int64_t i = range.begin(); i < range.end(); ++i){
					transpose_RowColSwap(const_cast<T*>(B_batch[i]), const_cast<T*>(B_batch[i]) + (b_total), b_mn1, t_b_rows, b_total);
				}
		});

	}
	tbb::parallel_for(tbb::blocked_range3d<int64_t>(0, batches, 0, a_rows, 0, b_cols),
		[&](const tbb::blocked_range3d<int64_t>& range){
	for(int64_t b = range.pages().begin(); b < range.pages().end(); ++b){
		const T* A = A_batch[b];
		const T* B = B_batch[b];
		T* C = C_batch[b];
		for (int64_t i = range.rows().begin(); i < range.rows().end(); ++i) {
			for (int64_t j = range.cols().begin(); j < range.cols().end(); ++j) {
				for (int64_t k = 0; k < a_cols; ++k) {
					C[i * b_cols + j] += A[i * a_cols + k] * B[k * b_cols + j];
				}
			}
		}
	}});
	if(transpose_a){
		tbb::parallel_for(tbb::blocked_range<int64_t>(0, batches),
			[&](const tbb::blocked_range<int64_t>& range){
				for(int64_t i = range.begin(); i < range.end(); ++i){
					transpose_RowColSwap(const_cast<T*>(A_batch[i]), const_cast<T*>(A_batch[i]) + (a_total), a_mn1, a_rows, a_total);
				}
		});
	}
	if(transpose_b){
		tbb::parallel_for(tbb::blocked_range<int64_t>(0, batches),
			[&](const tbb::blocked_range<int64_t>& range){
				for(int64_t i = range.begin(); i < range.end(); ++i){
					transpose_RowColSwap(const_cast<T*>(B_batch[i]), const_cast<T*>(B_batch[i]) + (b_total), b_mn1, b_rows, b_total);
				}
		});

	}
}

template<typename T>
void nt_matmult(const T* A, const T* B, T* C, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b){
	/* if constexpr (mp::simde_supported_v<T>){ */
	/* 	//differentiating based on tile size is a good way to reduce compile time */
	/* 	//this is why this was done in the first place */
	/* 	if constexpr (tile_size_v<T> <= 32){ */
	/* 		kmatmult_simde<T>(A, B, C, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b); */
	/* 	}else{ */
	/* 		pack_multiply_directly_var_threaded<T>(A, B, C, a_rows, a_cols, b_rows, b_cols,transpose_a, transpose_b); */
	/* 	} */
	/* } */
	if constexpr (mp::simde_supported_v<T>){
		pack_multiply_directly_var_threaded<T>(A, B, C, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b);
	}
	else{
		matmult_naive<T>(A, B, C, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b);
	}
}






//batch functions
template<typename T>
void nt_matmult_batch(const T** A, const T** B, T** C, int64_t batches, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b){
	/* if constexpr (mp::simde_supported_v<T>){ */
	/* 	if constexpr (tile_size_v<T> <= 32){ */
	/* 		kmatmult_simde_batch<T>(A, B, C, batches, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b); */
	/* 	}else{ */
	/* 		pack_multiply_directly_var_threaded_batch<T>(A, B, C, batches, a_rows, a_cols, b_rows, b_cols,transpose_a, transpose_b); */
	/* 	} */
	/* } */
	if constexpr (mp::simde_supported_v<T>){
		pack_multiply_directly_var_threaded_batch<T>(A, B, C, batches, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b);
	}
	else{
		matmult_naive_batch<T>(A, B, C, batches, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b);
	}
}




}}} //nt::functional::std_functional::

#endif // _NT_MATMULT_HPP_
