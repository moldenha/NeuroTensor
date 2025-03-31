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
#include "nt_matmult_simde.hpp"


namespace nt{
namespace functional{
namespace cpu{




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
    constexpr size_t simd_size = nt::mp::pack_size_v<T>; // simde__m256 processes 8 floats at a time
    size_t i = 0;

    // Create a zero vector
    nt::mp::simde_type<T> zero = nt::mp::SimdTraits<T>::zero();
    if constexpr (std::is_integral<T>::value){
	for(;i < num_elements; i += simd_size){
	    nt::mp::SimdTraits<T>::store(reinterpret_cast<nt::mp::simde_type<T>*>(array+i), zero);
	}
	
    }else{
	    for(;i < num_elements; i += simd_size){
		    nt::mp::SimdTraits<T>::store(array+i, zero);
	    }
    }
}






template<typename T, size_t COLS>
inline void kpack_rowA_threaded(const T* src_matrix, T* block_matrix, int64_t src_cols) noexcept {
	int64_t max = _NT_MATMULT_MIN_(COLS, src_cols);
	for(int i = 0; i < max; ++i)
		*block_matrix++ = src_matrix[i];
	for(int i = max; i < COLS; ++i)
		*block_matrix++ = 0;
}

//pack_threaded<T, a_pack_rows, a_pack_cols>(A, blockA_packed, i, k, a_rows, a_cols);
template<typename T, size_t ROWS, size_t COLS>
inline void pack_threaded(const T* src_matrix_pre, T* block_matrix, const int64_t& start_row, const int64_t& start_col, const int64_t& src_rows, const int64_t& src_cols) noexcept{
	/* kpack_blockA_threaded<T, COLS, ROWS>(src_matrix_pre + (start_row * src_cols), block_matrix, start_col, src_rows-start_row, src_cols); */
	const T* src_matrix = src_matrix_pre + (start_row * src_cols);
	int64_t max =  _NT_MATMULT_MIN_(ROWS, (src_rows-start_row));
	int64_t col_amt = src_cols - start_col;
	tbb::parallel_for(tbb::blocked_range<int64_t>(0, max),
			[&](const tbb::blocked_range<int64_t>& r){
	for(int64_t i = r.begin(); i < r.end(); ++i){
		kpack_rowA_threaded<T, COLS>(src_matrix + (i * src_cols) + start_col, block_matrix + (COLS * i), col_amt);
	}});
	zero_memory(block_matrix + (COLS * max), (ROWS - max) * COLS);
}



template<typename T, size_t ROWS, size_t COLS>
inline void kpack_rowB_threaded(const T* src_matrix, T* block_matrix, const int64_t& row_amt, const int64_t& src_cols, const int64_t& cur_row) noexcept {
	int64_t max = _NT_MATMULT_MIN_(ROWS, src_cols);
	for(int i = 0; i < max; ++i)
		block_matrix[i * COLS + cur_row]  = *src_matrix++;
}


//kpack_blockB_threaded<T, b_pack_rows, b_pack_cols>(B + (j * b_rows), blockB_packed, k, j, b_cols, b_rows);
template<typename T, size_t ROWS, size_t COLS>
inline void pack_transpose_threaded(const T* src_matrix_pre, T* block_matrix, const int64_t& start_col, const int64_t& start_row, const int64_t& src_cols, const int64_t& src_rows) noexcept{
	const T* src_matrix = src_matrix_pre + (start_row * src_cols);
	int64_t max =  _NT_MATMULT_MIN_(COLS, (src_rows-start_row)); //row_amt
	int64_t col_amt = src_cols - start_col;
	/* std::cout << "first max for row amount: "<<max<<std::endl; */
	kzero_memory<T, ROWS * COLS>(block_matrix);
	tbb::parallel_for(tbb::blocked_range<int64_t>(0, max),
			[&](const tbb::blocked_range<int64_t>& r){
	for(int i = r.begin(); i < r.end(); ++i){
		kpack_rowB_threaded<T, ROWS, COLS>(src_matrix + (i * src_cols) + start_col, block_matrix, max, col_amt, i);
	}
	});
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

	constexpr size_t a_pack_rows = (TILE_SIZE * _NT_MATMULT_NTHREADS_ * _NT_MATMULT_NTHREADS_);
	constexpr size_t a_pack_cols = TILE_SIZE;
	
	constexpr size_t b_pack_rows = (TILE_SIZE);
	constexpr size_t b_pack_cols = (TILE_SIZE * _NT_MATMULT_NTHREADS_ * _NT_MATMULT_NTHREADS_);
	auto block_a_func = (transpose_a ? &pack_transpose_threaded<T, a_pack_rows, a_pack_cols> : &pack_threaded<T, a_pack_rows, a_pack_cols>);
	auto block_b_func = (transpose_b ? &pack_transpose_threaded<T, b_pack_rows, b_pack_cols> : &pack_threaded<T, b_pack_rows, b_pack_cols>);

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
			block_a_func(A, blockA_packed, i, k, a_rows, a_cols);
			j_counter = 0;
			for(j = 0; j < b_cols; j += b_pack_cols, ++j_counter){
				const size_t& jrMax = jrMaxs[j_counter];
				block_b_func(B , blockB_packed, k, j, b_rows, b_cols);
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

}


template<typename T>
void pack_multiply_directly_var_threaded_batch(const T** A_batch, const T** B_batch, T** C_batch, const int64_t batches, const int64_t a_rows, const int64_t a_cols, const int64_t b_rows, const int64_t b_cols, bool transpose_a = false, bool transpose_b = false){
	constexpr size_t TILE_SIZE = tile_size_v<T>;
	for(int64_t b = 0; b < batches; ++b){
		pack_multiply_directly_var_threaded(A_batch[b], B_batch[b], C_batch[b], a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b);
	}

}


template<typename T>
void matmult_naive(const T* A, const T* B, T* C, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b) {


	if(transpose_a){
	if(transpose_b){
	tbb::parallel_for(tbb::blocked_range2d<int64_t>(0, a_rows, 0, b_cols),
		[&](const tbb::blocked_range2d<int64_t>& range){
	for (int64_t i = range.rows().begin(); i < range.rows().end(); ++i) {
		for (int64_t j = range.cols().begin(); j < range.cols().end(); ++j) {
			for (int64_t k = 0; k < a_cols; ++k) {
				C[i * b_cols + j] += A[k * a_rows + i] * B[j * b_rows + k];
			}
		}
	}});
	
	}else{
	//in this case, a_rows corresponds to the collumns of inputted a, it would be after the transpose
	tbb::parallel_for(tbb::blocked_range2d<int64_t>(0, a_rows, 0, b_cols),
		[&](const tbb::blocked_range2d<int64_t>& range){
	for (int64_t i = range.rows().begin(); i < range.rows().end(); ++i) {
		for (int64_t j = range.cols().begin(); j < range.cols().end(); ++j) {
			for (int64_t k = 0; k < a_cols; ++k) {
				C[i * b_cols + j] += A[k * a_rows + i] * B[k * b_cols + j];
			}
		}
	}});
	}

	}
	else if(transpose_b){
	tbb::parallel_for(tbb::blocked_range2d<int64_t>(0, a_rows, 0, b_cols),
		[&](const tbb::blocked_range2d<int64_t>& range){
	for (int64_t i = range.rows().begin(); i < range.rows().end(); ++i) {
		for (int64_t j = range.cols().begin(); j < range.cols().end(); ++j) {
			for (int64_t k = 0; k < a_cols; ++k) {
				C[i * b_cols + j] += A[i * a_cols + k] * B[j * b_rows + k];
			}
		}
	}});
	
	}
	else{
		tbb::parallel_for(tbb::blocked_range2d<int64_t>(0, a_rows, 0, b_cols),
			[&](const tbb::blocked_range2d<int64_t>& range){
		for (int64_t i = range.rows().begin(); i < range.rows().end(); ++i) {
			for (int64_t j = range.cols().begin(); j < range.cols().end(); ++j) {
				for (int64_t k = 0; k < a_cols; ++k) {
					C[i * b_cols + j] += A[i * a_cols + k] * B[k * b_cols + j];
				}
			}
		}});
	}

}


template<typename T>
void matmult_naive_batch(const T** A_batch, const T** B_batch, T** C_batch, int64_t batches, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b) {

		if(transpose_a){
	if(transpose_b){
	tbb::parallel_for(tbb::blocked_range3d<int64_t>(0, batches, 0, a_rows, 0, b_cols),
		[&](const tbb::blocked_range3d<int64_t>& range){
	for(int64_t b = range.pages().begin(); b < range.pages().end(); ++b){
		const T* A = A_batch[b];
		const T* B = B_batch[b];
		T* C = C_batch[b];
		for (int64_t i = range.rows().begin(); i < range.rows().end(); ++i) {
			for (int64_t j = range.cols().begin(); j < range.cols().end(); ++j) {
				for (int64_t k = 0; k < a_cols; ++k) {
					C[i * b_cols + j] += A[k * a_rows + i] * B[j * b_rows + k];
				}
			}
		}
	}});
	
	}else{
	//in this case, a_rows corresponds to the collumns of inputted a, it would be after the transpose
	tbb::parallel_for(tbb::blocked_range3d<int64_t>(0, batches, 0, a_rows, 0, b_cols),
		[&](const tbb::blocked_range3d<int64_t>& range){
	for(int64_t b = range.pages().begin(); b < range.pages().end(); ++b){
		const T* A = A_batch[b];
		const T* B = B_batch[b];
		T* C = C_batch[b];
		for (int64_t i = range.rows().begin(); i < range.rows().end(); ++i) {
			for (int64_t j = range.cols().begin(); j < range.cols().end(); ++j) {
				for (int64_t k = 0; k < a_cols; ++k) {
					C[i * b_cols + j] += A[k * a_rows + i] * B[k * b_cols + j];
				}
			}
		}
	}});
	}

	}
	else if(transpose_b){
	tbb::parallel_for(tbb::blocked_range3d<int64_t>(0, batches, 0, a_rows, 0, b_cols),
		[&](const tbb::blocked_range3d<int64_t>& range){
	for(int64_t b = range.pages().begin(); b < range.pages().end(); ++b){
		const T* A = A_batch[b];
		const T* B = B_batch[b];
		T* C = C_batch[b];
		for (int64_t i = range.rows().begin(); i < range.rows().end(); ++i) {
			for (int64_t j = range.cols().begin(); j < range.cols().end(); ++j) {
				for (int64_t k = 0; k < a_cols; ++k) {
					C[i * b_cols + j] += A[i * a_cols + k] * B[j * b_rows + k];
				}
			}
		}
	}});
	
	}
	else{
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
	}

}

template<typename T>
void nt_matmult(const T* A, const T* B, T* C, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b){
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
	if constexpr (mp::simde_supported_v<T>){
		pack_multiply_directly_var_threaded_batch<T>(A, B, C, batches, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b);
	}
	else{
		matmult_naive_batch<T>(A, B, C, batches, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b);
	}
}




}}} //nt::functional::cpu::

#endif // _NT_MATMULT_HPP_
