#ifndef _MATMULT_STD_CPP_
#define _MATMULT_STD_CPP_

#include <cstdint>

#include "../Tensor.h"
#include "../memory/iterator.h"
#include "../refs/SizeRef.h"
#include "../dtype/ArrayVoid.h"
#include "../dtype/DType.h"
#include "../dtype/DType_enum.h"




#include <atomic>
#include <functional>
//#include <i386/types.h>
#include <memory.h>
#include <memory>
#include <algorithm>
#include <numeric>
#include <random>
#include <ratio>
#include <iterator>

#include <cassert>
//#include <format>
#include <sys/_types/_int32_t.h>
#include <sys/_types/_int64_t.h>
#include <sys/types.h>
#include <type_traits>
#include <vector>
#include "../utils/utils.h"
#include <chrono>
#include "functional.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <random>
#include <cmath>
#include "../dtype/ArrayVoid.hpp"
#include "functional_operator.h"
#include "../mp/Threading.h"


namespace nt{
namespace functional{
namespace std_functional{

void determine_transposes(const Tensor& a, const Tensor& b, bool& a_trans, bool& b_trans, const bool& trans_a, const bool& trans_b){
	const auto& a_rows = a.shape()[-2];
	const auto& a_cols = a.shape()[-1];
	const auto& b_rows = b.shape()[-2];
	const auto& b_cols = b.shape()[-1];
	a_trans = trans_a;
	b_trans = trans_b;
	if(!a_trans && !b_trans){
		utils::THROW_EXCEPTION(b_rows == a_cols, "Expected dimensions -2 and -1 of shapes ($,$) to be equal for matrix multiplication", a.shape(), b.shape());
	}
	else if(a_trans && !b_trans){
		utils::THROW_EXCEPTION(b_rows == a_rows, "Expected dimensions -2 and -1 of shapes ($,$) to be equal for matrix multiplication", a.shape().transpose(-1,-2), b.shape());
	}
	else if(a_trans && b_trans){
		utils::THROW_EXCEPTION(b_cols == a_rows, "Expected dimensions -2 and -1 of shapes ($,$) to be equal for matrix multiplication", a.shape().transpose(-1,-2), b.shape().transpose(-1,-2));
	}
	else if(!a_trans && b_trans){
		utils::THROW_EXCEPTION(b_cols == a_cols, "Expected dimensions -2 and -1 of shapes ($,$) to be equal for matrix multiplication", a.shape(), b.shape().transpose(-1,-2));
	}
	return;
	
	

/* 	if(trans_a != -1 && trans_b != -1){ */
/* 		a_trans = (trans_a >= 1); */
/* 		b_trans = (trans_b >= 1); */
/* 		if(!a_trans && !b_trans){ */
/* 			utils::THROW_EXCEPTION(b_rows == a_cols, "Expected dimensions -2 and -1 of shapes ($,$) to be equal for matrix multiplication", a.shape(), b.shape()); */
/* 		} */
/* 		else if(a_trans && !b_trans){ */
/* 			utils::THROW_EXCEPTION(b_rows == a_rows, "Expected dimensions -2 and -1 of shapes ($,$) to be equal for matrix multiplication", a.shape().transpose(-1,-2), b.shape()); */
/* 		} */
/* 		else if(a_trans && b_trans){ */
/* 			utils::THROW_EXCEPTION(b_cols == a_rows, "Expected dimensions -2 and -1 of shapes ($,$) to be equal for matrix multiplication", a.shape().transpose(-1,-2), b.shape().transpose(-1,-2)); */
/* 		} */
/* 		else if(!a_trans && b_trans){ */
/* 			utils::THROW_EXCEPTION(b_cols == a_cols, "Expected dimensions -2 and -1 of shapes ($,$) to be equal for matrix multiplication", a.shape(), b.shape().transpose(-1,-2)); */
/* 		} */
/* 		return; */
/* 	} */
/* 	if(trans_a != -1){ */
/* 		a_trans = (trans_a >= 1); */
/* 	} */
/* 	if(trans_b != -1){ */
/* 		b_trans = (trans_b >= 1); */
/* 	} */

/* 	if(trans_a != -1){ */
/* 		if(a_trans == true){ */
/* 			if(b_cols == a_cols){ */
/* 				b_trans = false; */
/* 				return; */
/* 			} */
/* 			utils::THROW_EXCEPTION(b_rows == a_cols, "Expected dimensions -2 and -1 of shapes ($,$) to be equal for matrix multiplication", a.shape().transpose(-1,-2), b.shape().transpose(-1,-2)); */
/* 			b_trans = true; */
/* 			return; */
/* 		} */
/* 		else if(a_trans == false){ */
/* 			if(b_cols == a_rows){ */
/* 				b_trans = false; */
/* 				return; */
/* 			} */
/* 			utils::THROW_EXCEPTION(b_rows == a_rows, "Expected dimensions -2 and -1 of shapes ($,$) to be equal for matrix multiplication", a.shape(), b.shape().transpose(-1,-2)); */
/* 			b_trans = true; */
/* 			return; */
/* 		} */
/* 	} */
/* 	if(trans_b != -1){ */
/* 		if(b_trans == true){ */
/* 			if(b_rows == a_cols){ */
/* 				a_trans = false; */
/* 				return; */
/* 			} */
/* 			utils::THROW_EXCEPTION(b_rows == a_cols, "Expected dimensions -2 and -1 of shapes ($,$) to be equal for matrix multiplication", a.shape().transpose(-1,-2), b.shape().transpose(-1,-2)); */
/* 			a_trans = true; */
/* 			return; */
/* 		} */
/* 		else if(b_trans == false){ */
/* 			if(b_cols == a_rows){ */
/* 				a_trans = false; */
/* 				return; */
/* 			} */
/* 			utils::THROW_EXCEPTION(b_cols == a_cols, "Expected dimensions -2 and -1 of shapes ($,$) to be equal for matrix multiplication", a.shape().transpose(-1,-2), b.shape()); */
/* 			b_trans = true; */
/* 			return; */
/* 		} */
/* 	} */

/* 	if(b_cols == a_rows){ */

/* 		a_trans = false; */
/* 		b_trans = false; */
/* 		return; */
/* 	} */
/* 	if(b_rows == a_rows){ */

/* 		a_trans = false; */
/* 		b_trans = true; */
/* 		return; */
/* 	} */
/* 	a_trans = true; */
/* 	b_trans = true; */
/* 	utils::THROW_EXCEPTION(b_rows == a_cols, "Expected dimensions -2 and -1 of shapes ($,$) to be equal for matrix multiplication", a.shape().transpose(-1,-2), b.shape().transpose(-1,-2)); */
/* 	return; */
}










/* template<typename T> */
/* void _matrix_multiplication_typed_(const T* A, const T* B, T* C, int64_t A_rows, int64_t B_cols, int64_t A_cols) noexcept { */
/*     for (int64_t i = 0; i < A_rows; ++i) { */
/*         for (int64_t j = 0; j < B_cols; ++j) { */
/*             T sum = 0; */
/*             for (int64_t k = 0; k < A_cols; ++k) { */
/*                 sum += A[i * A_cols + k] * B[k * B_cols + j]; */
/*             } */
/*             C[i * B_cols + j] = sum; */
/*         } */
/*     } */
/* } */



template<typename T>
void _matrix_multiplication_typed_(const T* A, const T* B, T* C, int64_t A_rows, int64_t B_cols, int64_t A_cols) noexcept {
    threading::preferential_parallel_for(threading::block_ranges<2>(0, A_rows, 0, B_cols),
        [&](const auto& range) {
            for (int64_t i = range.begin[0]; i < range.end[0]; ++i) {
                for (int64_t j = range.begin[1]; j < range.end[1]; ++j) {
                    C[i * B_cols + j] = threading::strided_dot_product(
                        A + i * A_cols,    // Beginning of A
                        B + j,             // Beginning of B
                        A + (i + 1) * A_cols, // End of A (one beyond the last element)
                        T{0},             // Initial value for the final product
                        1,                // Stride for A (it's a row-wise traversal)
                        B_cols            // Stride for B (it's a column-wise traversal)
                    );
                }
            }
        });
}

/* template<typename T> */
/* void _matrix_multiplication_typed_transposed_B(const T* A, const T* B, T* C, int64_t A_rows, int64_t B_cols, int64_t A_cols) noexcept { */
/*     for (int64_t i = 0; i < A_rows; ++i) { */
/*         for (int64_t j = 0; j < B_cols; ++j) { */
/*             T sum = 0; */
/*             for (int64_t k = 0; k < A_cols; ++k) { */
/*                 sum += A[i * A_cols + k] * B[j * A_cols + k]; */
/*             } */
/*             C[i * B_cols + j] = sum; */
/*         } */
/*     } */
/* } */


template<typename T>
void _matrix_multiplication_typed_transposed_B(const T* A, const T* B, T* C, int64_t A_rows, int64_t B_cols, int64_t A_cols) noexcept {
    
	threading::preferential_parallel_for(threading::block_ranges<2>(0, A_rows, 0, B_cols),
        [&](const auto& range) {
            for (int64_t i = range.begin[0]; i < range.end[0]; ++i) {
                for (int64_t j = range.begin[1]; j < range.end[1]; ++j) {
			C[i * B_cols + j] = threading::dot_product(A + i * A_cols, B + j * A_cols, A + (i+1) * A_cols, T{0}); //not needed to be strided which is nice
		}
	}
	});
}




//this is basically the transposed_matmult
/* template<typename T> */
/* void _matrix_multiplication_typed_transposed_B(const T* A, const T* B, T* C, int64_t A_rows, int64_t B_cols, int64_t A_cols) noexcept { */
/*     threading::preferential_parallel_for(threading::block_ranges<2>(0, A_rows, 0, B_cols), */
/*         [&](const auto& range) { */
/*             for (int64_t i = range.begin[0]; i < range.end[0]; ++i) { */
/*                 for (int64_t j = range.begin[1]; j < range.end[1]; ++j) { */
/*                     C[i * B_cols + j] = threading::dot_product(A + i * A_cols, A + (i + 1) * A_cols, B + j * A_cols, T{0}); */
/*                 } */
/*             } */
/*         }); */
/* } */

template<typename T>
void _matrix_multiplication_typed_transposed_A(const T* A, const T* B, T* C, int64_t A_rows, int64_t B_cols, int64_t A_cols) noexcept {
    threading::preferential_parallel_for(threading::block_ranges<2>(0, A_cols, 0, B_cols),
        [&](const auto& range) {
            for (int64_t i = range.begin[0]; i < range.end[0]; ++i) {
                for (int64_t j = range.begin[1]; j < range.end[1]; ++j) {
                    C[i * B_cols + j] = threading::strided_dot_product(
                        A + i,                // Beginning of A (transposed)
                        B + j,                // Beginning of B
                        A + A_rows * A_cols,  // End of A (one beyond the last element, considering transposition)
                        T{0},                // Initial value for the final product
                        A_cols,              // Stride for A (it's a column-wise traversal)
                        B_cols               // Stride for B (it's a column-wise traversal)
                    );
                }
            }
        });
}

/* template<typename T> */
/* void _matrix_multiplication_typed_transposed_A(const T* A, const T* B, T* C, int64_t A_rows, int64_t B_cols, int64_t A_cols) noexcept { */
/*     for (int64_t i = 0; i < A_rows; ++i) { */
/*         for (int64_t j = 0; j < B_cols; ++j) { */
/*             T sum = 0; */
/*             for (int64_t k = 0; k < A_cols; ++k) { */
/*                 sum += A[i * A_cols + k] * B[j * A_cols + k]; */
/*             } */
/*             C[i * B_cols + j] = sum; */
/*         } */
/*     } */
/* } */




template<typename T>
void _matrix_multiplication_typed_both_transposed(const T* A, const T* B, T* C, int64_t A_rows, int64_t B_cols, int64_t A_cols) noexcept {
    threading::preferential_parallel_for(threading::block_ranges<2>(0, A_cols, 0, B_cols),
        [&](const auto& range) {
            for (int64_t i = range.begin[0]; i < range.end[0]; ++i) {
                for (int64_t j = range.begin[1]; j < range.end[1]; ++j) {
                    C[i * B_cols + j] = threading::strided_dot_product(
                        A + i,                // Beginning of A (transposed)
                        B + j * A_cols,       // Beginning of B (transposed)
                        A + A_rows * A_cols,  // End of A (one beyond the last element, considering transposition)
                        T{0},                // Initial value for the final product
                        A_cols,              // Stride for A (it's a column-wise traversal)
                        1                    // Stride for B (it's a column-wise traversal)
                    );
                }
            }
        });

}

/* template<typename T> */
/* void _matrix_multiplication_typed_both_transposed(const T* A, const T* B, T* C, int64_t A_rows, int64_t B_cols, int64_t A_cols) noexcept { */
/*     for (int64_t i = 0; i < A_rows; ++i) { */
/*         for (int64_t j = 0; j < B_cols; ++j) { */
/*             T sum = 0; */
/*             for (int64_t k = 0; k < A_cols; ++k) { */
/*                 sum += A[i * A_cols + k] * B[j * A_cols + k]; */
/*             } */
/*             C[i * B_cols + j] = sum; */
/*         } */
/*     } */
/* } */

template<typename T>
void matrix_multiplication_typed(const bool& transpose_a, const bool& transpose_b, const T* A, const T* B, T* C, int64_t A_rows, int64_t B_cols, int64_t A_cols) noexcept {
	if(transpose_a && transpose_b){
		_matrix_multiplication_typed_both_transposed<T>(A, B, C, A_rows, B_cols, A_cols);
	}
	else if(transpose_a){
		_matrix_multiplication_typed_transposed_A<T>(A, B, C, A_rows, B_cols, A_cols);
	}
	else if(transpose_b){
		_matrix_multiplication_typed_transposed_B<T>(A, B, C, A_rows, B_cols, A_cols);
	}
	else{
		_matrix_multiplication_typed_<T>(A, B, C, A_rows, B_cols, A_cols);
	}
}


template<typename T>
void _batch_matrix_multiplication_typed_(const T** A, const T** B, T** C, const int64_t& A_rows, const int64_t& B_cols, const int64_t& A_cols, const int64_t& group_count, const int64_t* group_sizes){
	threading::preferential_parallel_for(threading::block_ranges<3>(0, group_count, 0, A_rows, 0, B_cols),
		[&](const auto& range){
	int64_t batch_begin = 0;
	for (int64_t i = 0; i < range.begin[0]; ++i){batch_begin += group_sizes[i];}
	for (int64_t g = range.begin[0]; g != range.end[0]; ++g) {
            const int64_t& batch_end = group_sizes[g] + batch_begin;
            for (int64_t i = range.begin[1]; i != range.end[1]; ++i) {
                for (int64_t j = range.begin[2]; j != range.end[2]; ++j) {
		    const int64_t C_index = i * B_cols + j;
                    for (int64_t b = batch_begin; b < batch_end; ++b) {
			C[b][i * B_cols + j] = threading::strided_dot_product(
				A[b] + i * A_cols,    // Beginning of A
				B[b] + j,             // Beginning of B
				A[b] + (i + 1) * A_cols, // End of A (one beyond the last element)
				T{0},             // Initial value for the final product
				1,                // Stride for A (it's a row-wise traversal)
				B_cols            // Stride for B (it's a column-wise traversal)
			    );
                    }
                }
            }
	    batch_begin = batch_end;
        }
	});
}




//this is with no threading no matter what:
//switching A_cols and B_cols makes it work and just easier in this case that rewriting all the logic
template<typename T>
void _batch_matrix_multiplication_typed_transposed_B(const T** A, const T** B, T** C, const int64_t& A_rows, const int64_t& B_cols, const int64_t& A_cols, const int64_t& group_count, const int64_t* group_sizes){
	/* std::cout << "A_rows: "<<A_rows << "B_cols: "<<B_cols << "A_cols: "<<A_cols<<std::endl; */
	//in this case C_rows = A_rows and C_cols = A_cols
	threading::preferential_parallel_for(threading::block_ranges<3>(0, group_count, 0, A_rows, 0, A_cols),
		[&](const auto& range){
	int64_t batch_begin = 0;
	for(int64_t i = 0; i < range.begin[0]; ++i){batch_begin += group_sizes[i];}
	for(int64_t g = range.begin[0]; g < range.end[0]; ++g){
		const int64_t& batch_end = group_sizes[g] + batch_begin;
		for(int64_t i = range.begin[1]; i < range.end[1]; ++i){
			for(int64_t j = range.begin[2]; j < range.end[2]; ++j){
				const int64_t C_index = i * A_cols + j;
				const int64_t A_begin = i * B_cols;
				const int64_t B_index = j * B_cols;
				const int64_t A_end = A_begin + B_cols;
				/* std::cout << "C_index: "<<C_index << ',' << "A_begin: "<<A_begin << ", B_index: "<<B_index << "A_end: "<<A_end << " i: "<<i<<" j: "<<j<<std::endl; */
				for(int64_t b = batch_begin; b < batch_end; ++b){
					/* std::cout << "writing batch "<<b<<std::endl; */
					C[b][C_index] = threading::dot_product(
						A[b] + A_begin, //the beggining of A
						B[b] + B_index, //the beggining of B
						A[b] + A_end, //the end of A
						T{0}); //the initial value
	
				}
			}
		}
		batch_begin = batch_end;
	}});
}


//this works, but doesn't have the threading:
/* template<typename T> */
/* void _batch_matrix_multiplication_typed_transposed_B(const T** A, const T** B, T** C, const int64_t& A_rows, const int64_t& B_cols, const int64_t& A_cols, const int64_t& group_count, const int64_t* group_sizes){ */
/* 	/1* std::cout << "A_rows: "<<A_rows << "B_cols: "<<B_cols << "A_cols: "<<A_cols<<std::endl; *1/ */
/* 	//in this case C_rows = A_rows and C_cols = A_cols */
/* /1* 	threading::preferential_parallel_for(threading::block_ranges<3>(0, group_count, 0, A_rows, 0, B_cols), *1/ */
/* /1* 		[&](const auto& range){ *1/ */
/* 	int64_t batch_begin = 0; */
/* 	for(int64_t g = 0; g < group_count; ++g){ */
/* 		const int64_t& batch_end = group_sizes[g] + batch_begin; */
/* 		for(int64_t i = 0; i < A_rows; ++i){ */
/* 			for(int64_t j = 0; j < A_cols; ++j){ */
/* 				const int64_t C_index = i * A_cols + j; */
/* 				const int64_t A_begin = i * B_cols; */
/* 				const int64_t B_index = j * B_cols; */
/* 				const int64_t A_end = A_begin + B_cols; */
/* 				/1* std::cout << "C_index: "<<C_index << ',' << "A_begin: "<<A_begin << ", B_index: "<<B_index << "A_end: "<<A_end << " i: "<<i<<" j: "<<j<<std::endl; *1/ */
/* 				for(int64_t b = batch_begin; b < batch_end; ++b){ */
/* 					/1* std::cout << "writing batch "<<b<<std::endl; *1/ */
/* 					C[b][C_index] = threading::dot_product( */
/* 						A[b] + A_begin, //the beggining of A */
/* 						B[b] + B_index, //the beggining of B */
/* 						A[b] + A_end, //the end of A */
/* 						T{0}); //the initial value */
	
/* 				} */
/* 			} */
/* 		} */
/* 		batch_begin = batch_end; */
/* 	} */
/* } */





template<typename T>
void _batch_matrix_multiplication_typed_transposed_A(const T** A, const T** B, T** C, const int64_t& A_rows, const int64_t& B_cols, const int64_t& A_cols, const int64_t& group_count, const int64_t* group_sizes){
	threading::preferential_parallel_for(threading::block_ranges<3>(0, group_count, 0, A_cols, 0, B_cols),
		[&](const auto& range){
	int64_t batch_begin = 0;
	for (int64_t i = 0; i < range.begin[0]; ++i){batch_begin += group_sizes[i];}
	for(int64_t g = range.begin[0]; g != range.end[0]; ++g){
		const int64_t& batch_end = group_sizes[g] + batch_begin;
		for(int64_t i = range.begin[1]; i < range.end[1]; ++i){
			for(int64_t j = range.begin[2]; j < range.end[2]; ++j){
				const int64_t C_index = i * B_cols + j;
				const int64_t A_begin = i;
				const int64_t B_index = j;
				const int64_t A_end = A_rows * A_cols;
				/* std::cout << "C_index: "<<C_index << ',' << "A_begin: "<<A_begin << ", B_index: "<<B_index << "A_end: "<<A_end << " i: "<<i<<" j: "<<j<<std::endl; */
				for(int64_t b = batch_begin; b < batch_end; ++b){
					/* std::cout << "writing batch "<<b<<" of "<<batch_end<<std::endl; */
					C[b][C_index] = threading::strided_dot_product(
						A[b] + A_begin, //the beggining of A
						B[b] + B_index, //the beggining of B
						A[b] + A_end, //the end of A
						T{0},
						A_cols,
						B_cols); //the initial value
	
				}
	
			}
		}
		batch_begin = batch_end;
	}});
}

/* template<typename T> */
/* void _batch_matrix_multiplication_typed_transposed_A(const T** A, const T** B, T** C, const int64_t& A_rows, const int64_t& B_cols, const int64_t& A_cols, const int64_t& group_count, const int64_t* group_sizes){ */
/* 	threading::preferential_parallel_for(threading::block_ranges<3>(0, group_count, 0, A_cols, 0, B_cols), */
/* 		[&](const auto& range){ */
/* 	int64_t batch_begin = 0; */
/* 	for (int64_t i = 0; i < range.begin[0]; ++i){batch_begin += group_sizes[i];} */
/* 	for (int64_t g = range.begin[0]; g != range.end[0]; ++g) { */
/*             const int64_t& batch_end = group_sizes[g] + batch_begin; */
/*             for (int64_t i = range.begin[1]; i != range.end[1]; ++i) { */
/*                 for (int64_t j = range.begin[2]; j != range.end[2]; ++j) { */
/* 			for (int64_t b = batch_begin; b < batch_end; ++b) { */
/* 			C[b][i * B_cols + j] = threading::strided_dot_product( */
/* 				A[b] + i,                // Beginning of A (transposed) */
/* 				B[b] + j,                // Beginning of B */
/* 				A[b] + A_rows * A_cols,  // End of A (one beyond the last element, considering transposition) */
/* 				T{0},                // Initial value for the final product */
/* 				A_cols,              // Stride for A (it's a column-wise traversal) */
/* 				B_cols               // Stride for B (it's a column-wise traversal) */
/* 			    ); */
/*                     } */
/*                 } */
/*             } */
/* 	    batch_begin = batch_end; */
/*         } */
/* 	}); */
/* } */




template<typename T>
void _batch_matrix_multiplication_typed_both_transposed(const T** A, const T** B, T** C, const int64_t& A_rows, const int64_t& B_cols, const int64_t& A_cols, const int64_t& group_count, const int64_t* group_sizes){
    int64_t batch_begin = 0;
    for(int64_t g = 0; g < group_count; ++g){
	const int64_t& batch_end = group_sizes[g] + batch_begin;
	for(int64_t i = 0; i < A_cols; ++i){
	    for(int64_t j = 0; j < B_cols; ++j){
		const int64_t C_index = i * B_cols + j;
		const int64_t A_begin = i;
		const int64_t B_index = j * A_rows;
		const int64_t A_end = A_rows * A_cols;
		for(int64_t b = batch_begin; b < batch_end; ++b){
		    C[b][C_index] = threading::strided_dot_product(
			A[b] + A_begin, // The beginning of A
			B[b] + B_index, // The beginning of B
			A[b] + A_end, // The end of A
			T{0}, // Initial value
			A_cols, // Stride for A
			1); // Stride for B
		}
	    }
	}
	batch_begin = batch_end;
    }
}


/* template<typename T> */
/* void _batch_matrix_multiplication_typed_both_transposed(const T** A, const T** B, T** C, const int64_t& A_rows, const int64_t& B_cols, const int64_t& A_cols, const int64_t& group_count, const int64_t* group_sizes){ */
/*     threading::preferential_parallel_for(threading::block_ranges<3>(0, group_count, 0, A_cols, 0, B_cols), */
/*         [&](const auto& range){ */
/*             int64_t batch_begin = 0; */
/*             for (int64_t i = 0; i < range.begin[0]; ++i){batch_begin += group_sizes[i];} */
/*             for(int64_t g = range.begin[0]; g != range.end[0]; ++g){ */
/*                 const int64_t& batch_end = group_sizes[g] + batch_begin; */
/*                 for(int64_t i = range.begin[1]; i < range.end[1]; ++i){ */
/*                     for(int64_t j = range.begin[2]; j < range.end[2]; ++j){ */
/*                         const int64_t C_index = i * B_cols + j; */
/*                         const int64_t A_begin = i; */
/*                         const int64_t B_index = j * A_cols; */
/*                         const int64_t A_end = A_rows * A_cols; */
/*                         for(int64_t b = batch_begin; b < batch_end; ++b){ */
/*                             C[b][C_index] = threading::strided_dot_product( */
/*                                 A[b] + A_begin, // The beginning of A */
/*                                 B[b] + B_index, // The beginning of B */
/*                                 A[b] + A_end, // The end of A */
/*                                 T{0}, // Initial value */
/*                                 A_cols, // Stride for A */
/*                                 1); // Stride for B */
/*                         } */
/*                     } */
/*                 } */
/*                 batch_begin = batch_end; */
/*             } */
/*     }); */
/* } */





template<typename T>
void batched_matrix_multiplication_typed(const bool& transpose_a, const bool& transpose_b, const T** A, const T** B, T** C, int64_t A_rows, int64_t B_cols, int64_t A_cols, const int64_t& group_count, const int64_t* group_size) noexcept {
	if(transpose_a && transpose_b){
		_batch_matrix_multiplication_typed_both_transposed<T>(A, B, C, A_rows, B_cols, A_cols, group_count, group_size);
	}
	else if(transpose_a){
		_batch_matrix_multiplication_typed_transposed_A<T>(A, B, C, A_rows, B_cols, A_cols, group_count, group_size);
	}
	else if(transpose_b){
		_batch_matrix_multiplication_typed_transposed_B<T>(A, B, C, A_rows, B_cols, A_cols, group_count, group_size);
	}
	else{
		_batch_matrix_multiplication_typed_<T>(A, B, C, A_rows, B_cols, A_cols, group_count, group_size);
	}
}

/* #ifndef USE_PARALLEL*/
/* //if the tensor dim is 2, then theres no need for the mn1, i, or b_end variables, which could have its efficiency increased through parallelization*/
/* inline static constexpr auto transposed_matmult = [](auto a_begin, auto a_end, auto b_begin, void* o_ptr, const typename SizeRef::value_type& rows, const typename SizeRef::value_type& cols, const typename SizeRef::value_type& inter, const typename SizeRef::value_type& m_st){*/
/* 	typename SizeRef::value_type a_dist = std::distance(a_begin, a_end);*/
/* 	typename SizeRef::value_type a_mat_size = rows * inter;*/
/* 	typename SizeRef::value_type b_mat_size = cols * inter;*/
/* 	using value_t = utils::IteratorBaseType_t<decltype(a_begin)>;*/
/* 	value_t initial(0);*/
/* 	value_t* o_begin = reinterpret_cast<value_t*>(o_ptr) + m_st;*/
/* 	auto b_copy = b_begin;*/
/* 	auto b_end = b_begin + b_mat_size;*/
/* 	typename SizeRef::value_type i;*/
/* 	auto a_copy = a_begin + a_mat_size;*/
/* 	for(i = 0; i < a_dist; i += a_mat_size){*/
/* 		for(;a_begin != a_copy; a_begin += inter){*/
/* 			b_begin = b_copy;*/
/* 			for(;b_begin != b_end; b_begin += inter, ++o_begin){*/
/* 				*o_begin = std::inner_product(a_begin, a_begin + inter, b_begin, initial);*/
/* 			}*/
/* 		}*/
/* 		b_copy += b_mat_size;*/
/* 		b_end += b_mat_size;*/
/* 		a_copy += a_mat_size;*/
/* 	}*/
/* };*/

/* #else*/
/* going to make this parallelized*/
/*  */




/* //this version is capable of using with shared memory get, it is meant to be used over multiple CPU's with threading*/
/* inline static constexpr auto transposed_matmult = [](auto a_begin, auto a_end, auto b_begin, void* o_ptr, const typename SizeRef::value_type& rows, const typename SizeRef::value_type& cols, const typename SizeRef::value_type& inter, const typename SizeRef::value_type& m_st, const typename SizeRef::value_type& a_dist, const typename SizeRef::value_type& inter_begin, const typename SizeRef::value_type& inter_end){*/
/* 	std::cout << "transposed matmult parallel"<<std::endl; */
/* 	typename SizeRef::value_type a_mat_size = rows * inter;*/
/* 	typename SizeRef::value_type b_mat_size = cols * inter;*/
/* 	using value_t = utils::IteratorBaseType_t<decltype(a_begin)>;*/
	

/* 	if(a_dist == a_mat_size){*/
/* 		tbb::parallel_for(utils::calculateGrainSize3D(0, rows, 0, cols, inter_begin, inter_end), [&](const tbb::blocked_range3d<typename SizeRef::value_type>& range){*/
/* 		//pages() are now "rows" and rows() are now "cols" and cols() are now "inter"*/
/* 		auto ma_begin = a_begin + (inter * range.pages().begin());*/
/* 		auto a_copy = ma_begin + (inter * (range.pages().end() - range.pages().begin()));*/
/* 		typename SizeRef::value_type o_begin_index = (range.pages().begin() * cols) + range.rows().begin();*/
/* 		value_t* o_begin = reinterpret_cast<value_t*>(o_ptr) + m_st + o_begin_index;*/
/* 		value_t initial(0);*/
/* 		auto mb_begin = b_begin + (inter * range.rows().begin());*/
/* 		auto b_end = mb_begin + (inter * (range.rows().end() - range.rows().begin()));*/
/* 		auto b_copy = mb_begin;*/
/* 		typename SizeRef::value_type cols_o_add = cols - (range.rows().end() - range.rows().begin());*/

/* 		for(;ma_begin != a_copy; ma_begin += inter, o_begin += cols_o_add){*/
/* 			mb_begin = b_copy;*/
/* 			for(;mb_begin != b_end; mb_begin += inter, ++o_begin){*/
/* 				*o_begin += std::inner_product(ma_begin + range.cols().begin(), ma_begin + range.cols().end(), mb_begin + range.cols().begin(), initial);*/
/* 			}*/
/* 		}*/
/* 		});*/
/* 		return;*/
		
/* 	}*/
	/* std::vector<threading::block_ranges<3>> my_blocks; */
	/* my_blocks.reserve(numSegments); */
	/* for(typename SizeRef::value_type i = 0; i < numSegments; ++i){ */
	/* 	const typename SizeRef::value_type& row_begin = row_begins[i]; */
	/* 	const typename SizeRef::value_type& row_end = row_begins[i+1]; */
	/* 	my_blocks.push_back(threading::block_ranges<3>(0, first_arg, row_begin, row_end, 0, outpSize.back())); */
	/* } */
/* 	auto block = threading::block_ranges<3>(0, a_dist / a_mat_size, 0, rows, 0, cols);*/
	/* std::cout << "going to do parallel for"<<std::endl; */
/* 	threading::parallel_for(block, [&](const threading::blocked_range<3> &range){*/
	/* threading::parallel_for(utils::calculateGrainSize3D(a_dist / a_mat_size, rows, cols), [&](const tbb::blocked_range3d<typename SizeRef::value_type>& range){ */
	/* std::cout << "inside parallel for"<<std::endl; */
	/* tbb::parallel_for(tbb::blocked_range3d<typename SizeRef::value_type>(0, a_dist / a_mat_size, 0, rows, 0, cols), [&](const tbb::blocked_range3d<typename SizeRef::value_type>& range){ */
	
/* 	auto ma_begin = a_begin + (a_mat_size * range.begin[0]) + (inter * range.begin[1]);*/
	/* auto ma_begin = a_begin + (a_mat_size * range.pages().begin()) + (inter * range.begin[1]); */
/* 	auto a_copy = ma_begin + (inter * (range.end[1] - range.begin[1]));*/
/* 	typename SizeRef::value_type o_begin_index = (range.begin[0] * rows * cols) + (range.begin[1] * cols) + range.begin[2];*/
/* 	value_t* o_begin = reinterpret_cast<value_t*>(o_ptr) + m_st + o_begin_index;*/
/* 	value_t initial(0);*/
/* 	auto mb_begin = b_begin + (b_mat_size * range.begin[0]) + (inter * range.begin[2]);*/
/* 	auto b_end = mb_begin + (inter * (range.end[2] - range.begin[2]));*/
/* 	auto b_copy = mb_begin;*/
	

/* 	typename SizeRef::value_type cols_o_add = cols - (range.end[2] - range.begin[2]);*/
/* 	typename SizeRef::value_type rows_o_add = (range.end[1] - range.begin[1]) * cols_o_add; // this might be wrong, it might be more like:*/
/* 	//typename SizeRef::value_type rows_o_add = (rows * cols) - ((range.end[1] - range.begin[1]) * cols);*/
	

/* 	for(typename SizeRef::value_type i = range.begin[0]; i != range.end[0]; ++i, o_begin += rows_o_add){*/
/* 		for(;ma_begin != a_copy; ma_begin += inter, o_begin += cols_o_add){*/
/* 			mb_begin = b_copy;*/
/* 			for(;mb_begin != b_end; mb_begin += inter, ++o_begin){*/
				/* value_t initial(0); */
				/* ParallelInnerProduct innerProduct(ma_begin, mb_begin, *o_begin); */
				/* tbb::parallel_reduce(tbb::blocked_range<size_t>(0, inter), innerProduct); */
				/* *o_begin = innerProduct.result; */
/* 				*o_begin += std::inner_product(ma_begin + inter_begin, ma_begin + inter_end, mb_begin, value_t(0));*/
/* 			}*/
/* 		}*/
/* 		b_copy += b_mat_size;*/
/* 		b_end += b_mat_size;*/
/* 		a_copy += a_mat_size;*/
/* 	}*/
/* 	});*/
/* };*/


/* //this version is meant to be used with pipe, when the previous version uses too much of the shared_memory get*/
/* //there will be a write and read*/ 
/* inline static constexpr auto transposed_matmult_piped = [](auto a_begin, auto a_end, auto b_begin, void* o_ptr, const typename SizeRef::value_type& rows, const typename SizeRef::value_type& cols, const typename SizeRef::value_type& inter, const typename SizeRef::value_type& m_st, const typename SizeRef::value_type& a_dist, const typename SizeRef::value_type& row_begin, const typename SizeRef::value_type& row_end, int (*pipes)[2], const threading::block_ranges<3>& block){*/
/* 	std::cout << "transposed matmult piped"<<std::endl;*/
/* 	typename SizeRef::value_type a_mat_size = rows * inter;*/
/* 	typename SizeRef::value_type b_mat_size = cols * inter;*/
/* 	//this version is going to split by the rows based on the number of processes and the current process to split the multiplication by process*/
	
/* 	const typename SizeRef::value_type my_row_increment = row_end - row_begin;*/
/* 	using value_t = utils::IteratorBaseType_t<decltype(a_begin)>;*/
	/* std::cout << "my_block: "<<block<<std::endl; */
/* 	std::cout << "going to do parallel_for"<<std::endl;*/
/* 	threading::parallel_for(block, [&](const threading::blocked_range<3> &range){*/
/* 		std::cout << "inside parallel_for"<<std::endl;*/
/* 		int fd = pipes[range.index][1];*/
/* 		auto ma_begin = a_begin + (a_mat_size * range.begin[0]) + (inter * range.begin[1]);*/
/* 		auto a_copy = ma_begin + (inter * (range.end[1] - range.begin[1]));*/
/* 		value_t initial(0);*/
/* 		//for tbb:parallel_for conversion:*/
/* 		//range.pages().begin() == range.begin[0];*/
/* 		//range.rows().begin() == range.begin[1];*/
/* 		//range.cols().begin() == range.begin[2];*/
		/* typename SizeRef::value_type o_begin_index = (range.pages().begin() * my_row_increment * cols) + (range.rows().begin() * cols) + range.cols().begin(); */
/* 		typename SizeRef::value_type o_begin_index = (range.begin[0] * my_row_increment * cols) + (range.begin[1] * cols) + range.begin[2];*/
/* 		value_t* o_begin = reinterpret_cast<value_t*>(o_ptr) + m_st + o_begin_index;*/
/* 		value_t* o_begin_copy = o_begin;*/
/* 		auto mb_begin = b_begin + (b_mat_size * range.begin[0]) + (inter * range.begin[2]);*/
/* 		auto b_end = mb_begin + (inter * (range.end[2] - range.begin[2]));*/
/* 		auto b_copy = mb_begin;*/
/* 		const typename SizeRef::value_type bytes_to_write = (range.end[2] - range.begin[2]) * sizeof(value_t);*/
/* 		for(typename SizeRef::value_type i = range.begin[0]; i < range.end[0]; ++i){*/
/* 			for(;ma_begin != a_copy; ma_begin += inter){*/
/* 				mb_begin = b_copy;*/
/* 				for(;mb_begin != b_end; mb_begin += inter, ++o_begin){*/
/* 					*o_begin = std::inner_product(ma_begin, ma_begin + inter, mb_begin, initial);*/
/* 				}*/
/* 				write(fd, o_begin_copy, bytes_to_write);*/
/* 			}*/
/* 			b_copy += b_mat_size;*/
/* 			b_end += b_mat_size;*/
/* 			a_copy += a_mat_size;*/
/* 			o_begin = o_begin_copy;*/
/* 		}*/

/* 	});*/

/* };*/	
	

/* #endif*/



/* Tensor matmult_cT(const Tensor& a, const Tensor& b){*/
	/* std::cout << "matmult ct called"<<std::endl; */
/* 	utils::THROW_EXCEPTION(a.dtype != DType::Bool, "RuntimeError: Tensor DType was Bool which is unallowed for matmult function");*/
/* 	utils::THROW_EXCEPTION(a.dtype == b.dtype, "\nRuntimeError: Expected second tensor to have dtype of $, instead had dtype of $", a.dtype, b.dtype);*/
/* 	utils::THROW_EXCEPTION(a.dims() > 1 && b.dims() > 1, "\nRuntimeError: Expected tensors to have dims greater than 1, but instead had dims of $ and $", a.dims(), b.dims());*/
/* 	utils::THROW_EXCEPTION(a.shape()[-1] == b.shape()[-1], "\nRuntimeError: Expected second tensor rows to be $ when it was $",a.shape()[-1],b.shape()[-1]);*/
/* 	if(a.dims() != b.dims()){*/
/* 		if(a.dims() > b.dims()){*/
			/* std::cout << "if statement"<<std::endl; */
/* 			std::vector<typename SizeRef::value_type> size_outp = a.shape().Vec();*/
/* 			size_outp.back() = b.shape()[-2];*/
/* 			typename SizeRef::value_type start = a.dims() - b.dims();*/
/* 			for(typename SizeRef::value_type i = start; i < size_outp.size()-2; ++i){*/
/* 				utils::THROW_EXCEPTION(size_outp[i] == b.shape()[i - start], "RuntimeError: Expected sizes at dimension $ to be $ but got $", i-start, size_outp[i], b.shape()[i-start]);*/ 
/* 			}*/

/* #ifndef USE_PARALLEL*/
/* 			Tensor output = zeros(SizeRef(std::move(size_outp)), a.dtype);*/
/* 			const Tensor a_1 = a.split_axis(start-1);*/
/* 			const Tensor* a1_begin = reinterpret_cast<const Tensor*>(a_1.data_ptr());*/
/* 			const Tensor* a1_end = a1_begin + a_1.numel();*/
/* 			typename SizeRef::value_type multiply = output.shape().multiply(start);*/
/* 			typename SizeRef::value_type i = 0;*/
/* 			for(;a1_begin != a1_end; ++a1_begin, i += multiply){*/
/* 				a1_begin->arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(transposed_matmult, b.arr_void(), output.data_ptr(), output.shape()[-2], output.shape().back(), a.shape().back(), i);*/
/* 			}*/
/* #else*/
/* 			SizeRef outpSize(std::move(size_outp));*/
/* 			const typename SizeRef::value_type numSegments = utils::getNumCores();*/
/* 			const typename SizeRef::value_type numThreads = utils::getThreadsPerCore();*/
/* 			const typename SizeRef::value_type numParallel = numSegments * numThreads;*/
/* 			const typename SizeRef::value_type& inter = a.shape().back();*/
/* 			if(outpSize.multiply() * DTypeFuncs::size_of_dtype(a.dtype) < utils::get_shared_memory_max()){*/
/* 				const typename SizeRef::value_type multiply = outpSize.multiply(start);*/
/* 				const typename SizeRef::value_type additions = (inter % numSegments == 0 ? inter / numSegments : (inter / numSegments) + 1);*/
/* 				ArrayVoid outpAV = mp::shm_parallel_memset([&b, &multiply, &start, &outpSize, &additions, &inter](const Tensor& a, void* shared_mem, const typename SizeRef::value_type& process_num){*/
/* 						const Tensor a_1 = a.split_axis(start-1);*/
/* 						const Tensor* a1_begin = reinterpret_cast<const Tensor*>(a_1.data_ptr());*/
/* 						const Tensor* a1_end = a1_begin + a_1.numel();*/
/* 						typename SizeRef::value_type i = 0;*/
/* 						const typename SizeRef::value_type inter_begin = additions * process_num;*/
/* 						const typename SizeRef::value_type inter_end = (inter_begin + additions) > inter ? inter : inter_begin + additions;*/
/* 						for(;a1_begin != a1_end; ++a1_begin, i += multiply){*/
/* 							a1_begin->arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(transposed_matmult, b.arr_void(), shared_mem, outpSize[-2], outpSize.back(), inter, i, a1_begin->numel(), inter_begin, inter_end);*/
/* 						}*/
/* 						}, a, outpSize.multiply(), 0);*/
/* 				return Tensor(std::move(outpAV), std::move(outpSize));*/

/* 			}*/
/* 			std::vector<typename SizeRef::value_type> row_begins(numSegments + 1);*/
/* 			row_begins[0] = 0;*/
/* 			const typename SizeRef::value_type& rows = a.shape()[-2];*/
/* 			const typename SizeRef::value_type divs = rows / numSegments;*/
/* 			for(typename SizeRef::value_type i = 0; i < numSegments; ++i){*/
/* 				row_begins[i+1] = row_begins[i] + divs;*/
/* 			}*/
/* 			row_begins.back() = rows;*/

/* 			std::vector<typename SizeRef::value_type> multiplies(numSegments);*/
/* 			const typename SizeRef::value_type multiply_mrow = outpSize.multiply(start) / outpSize[-2];*/
/* 			for(typename SizeRef::value_type i = 0; i < multiplies.size(); ++i){*/
/* 				typename SizeRef::value_type outRowSize = row_begins[i+1] - row_begins[i];*/
/* 				multiplies[i] = multiply_mrow * outRowSize;*/
/* 			}*/
/* 			const typename SizeRef::value_type a_dist = a.shape().multiply(start-1);*/
/* 			const typename SizeRef::value_type a_mat_size = rows * inter;*/
/* 			const typename SizeRef::value_type first_arg = a_dist / a_mat_size;*/
/* 			std::vector<threading::block_ranges<3>> my_blocks;*/

/* 			for(typename SizeRef::value_type i = 0; i < numSegments; ++i){*/
/* 				const typename SizeRef::value_type& row_begin = row_begins[i];*/
/* 				const typename SizeRef::value_type& row_end = row_begins[i+1];*/
/* 				my_blocks.push_back(threading::block_ranges<3>(0, first_arg, row_begin, row_end, 0, outpSize.back()));*/
/* 			}*/


/* 			std::vector<std::tuple<typename SizeRef::value_type, typename SizeRef::value_type, typename SizeRef::value_type> > increments(numParallel);*/
/* 			const typename SizeRef::value_type increment_num = outpSize.multiply(start-1) * DTypeFuncs::size_of_dtype(a.dtype);*/
/* 			auto begin = increments.begin();*/
/* 			typename SizeRef::value_type index = 0;*/
/* 			for(typename SizeRef::value_type i = 0; i < numSegments; ++i){*/
/* 				const std::vector<threading::blocked_range<3> >& blocks = my_blocks[i].getBlocks();*/
/* 				for(typename SizeRef::value_type j = 0; j < blocks.size(); ++j, ++begin){*/
/* 					const typename SizeRef::value_type max_bytes = blocks[j].blockSize * DTypeFuncs::size_of_dtype(a.dtype);*/
/* 					*begin = std::tuple<typename SizeRef::value_type, typename SizeRef::value_type, typename SizeRef::value_type>(index, max_bytes, increment_num);*/
/* 					index += max_bytes;*/
/* 				}*/
/* 			}*/
			

/* 			ArrayVoid outpAV = pool::apply_unary_function(*/
/* 					[&a, &b, &row_begins, &start, &multiplies, &outpSize, &inter, &my_blocks](void* memory, const typename SizeRef::value_type& process_num, int (*pipes)[2]){*/
/* 						const Tensor a_1 = a.split_axis(start-1);*/
/* 						const Tensor* a1_begin = reinterpret_cast<const Tensor*>(a_1.data_ptr());*/
/* 						const Tensor* a1_end = a1_begin + a_1.numel();*/
/* 						const typename SizeRef::value_type& row_begin = row_begins[process_num];*/
/* 						const typename SizeRef::value_type& row_end = row_begins[process_num+1];*/
/* 						typename SizeRef::value_type i = 0;*/
/* 						for(;a1_begin != a1_end; ++a1_begin, i += multiplies[process_num]){*/
/* 							a1_begin->arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(transposed_matmult_piped,*/
/* 									b.arr_void(),*/
/* 									memory,*/
/* 									outpSize[-2],*/
/* 									outpSize.back(),*/
/* 									inter,*/
/* 									i,*/
/* 									a1_begin->numel(),*/
/* 									row_begin,*/
/* 									row_end,*/
/* 									pipes,*/
/* 									my_blocks[process_num]);*/

/* 						}*/

/* 					},*/
/* 					a.dtype,*/
/* 					outpSize.multiply(),*/
/* 					increments);*/
/* 			Tensor output(std::move(outpAV), std::move(outpSize));*/

/* #endif*/

/* 			return std::move(output);*/
/* 		}*/
/* 		//b.dims() > a.dims()*/
/* 		else{*/
/* 			std::vector<typename SizeRef::value_type> size_outp = b.shape().transpose(-1,-2).Vec();*/
/* 			size_outp[size_outp.size()-2] = a.shape()[-2];*/
/* 			typename SizeRef::value_type start = b.dims() - a.dims();*/
/* 			for(typename SizeRef::value_type i = start; i < size_outp.size() - 2; ++i)*/
/* 				utils::THROW_EXCEPTION(size_outp[i] == a.shape()[i - start], "RuntimeError: Expected sizes at dimension $ to be $ but got $", i-start, size_outp[i], a.shape()[i-start]);*/
/* #ifndef USE_PARALLEL*/
/* 			Tensor output = zeros(SizeRef(std::move(size_outp)), a.dtype);*/
/* 			const Tensor b_1 = b.split_axis(start-1);*/
/* 			const Tensor* b1_begin = reinterpret_cast<const Tensor*>(b_1.data_ptr());*/
/* 			const Tensor* b1_end = b1_begin + b_1.numel();*/
/* 			typename SizeRef::value_type multiply = output.shape().multiply(start);*/
/* 			typename SizeRef::value_type i = 0;*/
/* 			for(;b1_begin != b1_end; ++b1_begin, i += multiply){*/
/* 				a.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(transposed_matmult, b1_begin->arr_void(), output.data_ptr(), output.shape()[-2], output.shape().back(), a.shape().back(), i);*/
/* 			}*/
/* #else*/
/* 			SizeRef outpSize(std::move(size_outp));*/
/* 			SizeRef outpSize2 = outpSize.flatten(0, start);*/
/* 			const typename SizeRef::value_type numSegments = utils::getNumCores();*/
/* 			const typename SizeRef::value_type numThreads = utils::getThreadsPerCore();*/
/* 			const typename SizeRef::value_type numParallel = numSegments * numThreads;*/
			/* if(numParallel > outpSize.multiply(1)){ */
			/* 	Tensor output = zeros(SizeRef(std::move(size_outp)), a.dtype); */
			/* 	const Tensor b_1 = b.split_axis(start-1); */
			/* 	const Tensor* b1_begin = reinterpret_cast<const Tensor*>(b_1.data_ptr()); */
			/* 	const Tensor* b1_end = b1_begin + b_1.numel(); */
			/* 	typename SizeRef::value_type multiply = output.shape().multiply(start); */
			/* 	typename SizeRef::value_type i = 0; */
			/* 	for(;b1_begin != b1_end; ++b1_begin, i += multiply){ */
			/* 		a.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(transposed_matmult, b1_begin->arr_void(), output.data_ptr(), output.shape()[-2], output.shape().back(), a.shape().back(), i); */
			/* 	} */
			/* 	return std::move(output); */
	
			/* } */

/* 			const typename SizeRef::value_type& inter = a.shape().back();*/
/* 			if(outpSize.multiply() * DTypeFuncs::size_of_dtype(a.dtype) < utils::get_shared_memory_max()){*/
/* 				const typename SizeRef::value_type multiply = outpSize.multiply(start);*/
/* 				const typename SizeRef::value_type additions = (inter % numSegments == 0 ? inter / numSegments : (inter / numSegments) + 1);*/
/* 				const typename SizeRef::value_type& a_dist = a.numel();*/
/* 				ArrayVoid outpAV = mp::shm_parallel_memset([&b, &multiply, &start, &outpSize, &additions, &a_dist, &inter](const Tensor& a, void* shared_mem, const typename SizeRef::value_type& process_num){*/
/* 						const Tensor b_1 = b.split_axis(start-1);*/
/* 						const Tensor* b1_begin = reinterpret_cast<const Tensor*>(b_1.data_ptr());*/
/* 						const Tensor* b1_end = b1_begin + b_1.numel();*/
/* 						typename SizeRef::value_type i = 0;*/
/* 						const typename SizeRef::value_type inter_begin = additions * process_num;*/
/* 						const typename SizeRef::value_type inter_end = (inter_begin + additions) > inter ? inter : inter_begin + additions;*/
/* 						for(;b1_begin != b1_end; ++b1_begin, i += multiply){*/
/* 							a.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(transposed_matmult, b1_begin->arr_void(), shared_mem, outpSize[-2], outpSize.back(), inter, i, a_dist, inter_begin, inter_end);*/
/* 						}*/
/* 						}, a, outpSize.multiply(), 0);*/
/* 				return Tensor(std::move(outpAV), std::move(outpSize));*/
/* 			}*/
			
/* 			utils::THROW_EXCEPTION(outpSize2.size() == 3, "outpSize2 $ was calculated incorrectly, with having $ arguments instead of 3", outpSize, outpSize.size());*/
/* 			std::vector<typename SizeRef::value_type> row_begins(numSegments + 1);*/
/* 			row_begins[0] = 0;*/
/* 			const typename SizeRef::value_type& rows = a.shape()[-2];*/
/* 			const typename SizeRef::value_type divs = rows / numSegments;*/
/* 			for(typename SizeRef::value_type i = 0; i < numSegments; ++i){*/
/* 				row_begins[i+1] = row_begins[i] + divs;*/
/* 			}*/
/* 			row_begins.back() = rows;*/
			/* std::vector<typename SizeRef::value_type> byteSizes(max_cpu_processes), multiplies(max_cpu_processes); */
/* 			std::vector<typename SizeRef::value_type>  multiplies(numSegments);*/
			/* const typename SizeRef::value_type bytesize_mrow = outpSize.multiply() / outpSize[-2]; */
/* 			const typename SizeRef::value_type multiply_mrow = outpSize.multiply(start) / outpSize[-2];*/
/* 			for(typename SizeRef::value_type i = 0; i < multiplies.size(); ++i){*/
/* 				typename SizeRef::value_type outRowSize = row_begins[i+1] - row_begins[i];*/
/* 				//typename SizeRef::value_type outMatSize = outRowSize * outpSize.back();*/
				/* byteSizes[i] = bytesize_mrow * outRowSize; */
/* 				multiplies[i] = multiply_mrow * outRowSize;*/
/* 			}*/
			
/* 			const typename SizeRef::value_type a_dist = a.numel();*/
/* 			const typename SizeRef::value_type a_mat_size = rows * inter;*/
/* 			const typename SizeRef::value_type first_arg = a_dist / a_mat_size;*/
/* 			std::vector<threading::block_ranges<3>> my_blocks;*/
/* 			for(typename SizeRef::value_type i = 0; i < numSegments; ++i){*/
/* 				const typename SizeRef::value_type& row_begin = row_begins[i];*/
/* 				const typename SizeRef::value_type& row_end = row_begins[i+1];*/
/* 				my_blocks.push_back(threading::block_ranges<3>(0, first_arg, row_begin, row_end, 0, outpSize.back()));*/
/* 			}*/


/* 			const typename SizeRef::value_type increment_num = outpSize2.multiply(1) * DTypeFuncs::size_of_dtype(a.dtype); //the first one is the amount that it is over by, so that is going to be itterated over by a for loop, this it std::get<2>(tupple);*/
/* 			const typename SizeRef::value_type remainder = (increment_num % numParallel);*/
/* 			const typename SizeRef::value_type div = (increment_num / numParallel);*/
/* 			std::vector<std::tuple<typename SizeRef::value_type, typename SizeRef::value_type, typename SizeRef::value_type>> increments(numParallel);*/
/* 			auto begin = increments.begin();*/
/* 			typename SizeRef::value_type index = 0;*/
/* 			for(typename SizeRef::value_type i = 0; i < numSegments; ++i){*/
/* 				const std::vector<threading::blocked_range<3> >& blocks = my_blocks[i].getBlocks();*/
/* 				for(typename SizeRef::value_type j = 0; j < blocks.size(); ++j, ++begin){*/
/* 					const typename SizeRef::value_type max_bytes = blocks[j].blockSize * DTypeFuncs::size_of_dtype(a.dtype);*/
/* 					*begin = std::tuple<typename SizeRef::value_type, typename SizeRef::value_type, typename SizeRef::value_type>(index, max_bytes, increment_num);*/
/* 					index += max_bytes;*/
/* 				}*/
/* 			}*/
			
			/* std::vector<std::tuple<typename SizeRef::value_type, typename SizeRef::value_type, typename SizeRef::value_type>> increments(numParallel); */
			/* for(typename SizeRef::value_type i = 0; i < NumParallel; */ 


/* 			ArrayVoid outpAV = pool::apply_unary_function(*/
/* 					[&a, &b, &row_begins, &start, &multiplies, &outpSize, &inter, &my_blocks](void* memory, const typename SizeRef::value_type& process_num, int (*pipes)[2]){*/
/* 						const Tensor b_1 = b.split_axis(start-1);*/
/* 						const Tensor* b1_begin = reinterpret_cast<const Tensor*>(b_1.data_ptr());*/
/* 						const Tensor* b1_end = b1_begin + b_1.numel();*/
/* 						const typename SizeRef::value_type& row_begin = row_begins[process_num];*/
/* 						const typename SizeRef::value_type& row_end = row_begins[process_num+1];*/
/* 						typename SizeRef::value_type i = 0;*/
/* 						for(;b1_begin != b1_end; ++b1_begin, i += multiplies[process_num]){*/
/* 							a.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(transposed_matmult_piped,*/
/* 									b1_begin->arr_void(),*/
/* 									memory,*/
/* 									outpSize[-2],*/
/* 									outpSize.back(),*/
/* 									inter,*/
/* 									i,*/
/* 									a.numel(),*/
/* 									row_begin,*/
/* 									row_end,*/
/* 									pipes,*/
/* 									my_blocks[process_num]);*/

/* 						}*/

/* 					},*/
/* 					a.dtype,*/
/* 					outpSize.multiply(),*/
/* 					increments);*/

/* 			Tensor output(std::move(outpAV), std::move(outpSize));*/

/* #endif*/
/* 			return std::move(output);*/
/* 		}*/
/* 	}*/
/* 	std::vector<typename SizeRef::value_type> size_outp = a.shape().Vec();*/
/* 	size_outp.back() = b.shape()[-2];*/
/* 	for(typename SizeRef::value_type i = 0; i < size_outp.size()-2; ++i){*/
/* 		utils::THROW_EXCEPTION(size_outp[i] == b.shape()[i], "RuntimeError: Expected sizes at dimension $ to be $ but got $", i, size_outp[i], b.shape()[i]);*/ 
/* 	}*/
/* #ifndef USE_PARALLEL*/
/* 	Tensor output = zeros(SizeRef(std::move(size_outp)), a.dtype);*/
/* 	a.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(transposed_matmult, b.arr_void(), output.data_ptr(), output.shape()[-2], output.shape().back(), a.shape().back(), 0);*/
/* #else*/
/* 	SizeRef outpSize(std::move(size_outp));*/
/* 	const typename SizeRef::value_type numSegments = utils::getNumCores();*/
/* 	const typename SizeRef::value_type numThreads = utils::getThreadsPerCore();*/
/* 	const typename SizeRef::value_type numParallel = numSegments * numThreads;*/
/* 	const typename SizeRef::value_type& inter = a.shape().back();*/
/* 	if(outpSize.multiply() * DTypeFuncs::size_of_dtype(a.dtype) < utils::get_shared_memory_max()){*/
/* 		const typename SizeRef::value_type& a_dist = a.numel();*/
/* 		const typename SizeRef::value_type additions = (inter % numSegments == 0 ? inter / numSegments : (inter / numSegments) + 1);*/
/* 		ArrayVoid outpAV = mp::shm_parallel_memset([&b, &outpSize, &additions, &a_dist, &inter](const Tensor& a, void* shared_mem, const typename SizeRef::value_type& process_num){*/
/* 				const typename SizeRef::value_type inter_begin = additions * process_num;*/
/* 				const typename SizeRef::value_type inter_end = (inter_begin + additions) > inter ? inter : inter_begin + additions;*/
/* 				a.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(transposed_matmult, b.arr_void(), shared_mem, outpSize[-2], outpSize.back(), inter, 0, a_dist, inter_begin, inter_end);*/
/* 				}, a, outpSize.multiply(), 0);*/
/* 		return Tensor(std::move(outpAV), std::move(outpSize));*/
/* 	}*/
/* 	std::vector<typename SizeRef::value_type> row_begins(numSegments + 1);*/
/* 	row_begins[0] = 0;*/
/* 	const typename SizeRef::value_type& rows = a.shape()[-2];*/
/* 	const typename SizeRef::value_type divs = rows / numSegments;*/
/* 	for(typename SizeRef::value_type i = 0; i < numSegments; ++i){*/
/* 		row_begins[i+1] = row_begins[i] + divs;*/
/* 	}*/
/* 	row_begins.back() = rows;*/

/* 	const typename SizeRef::value_type a_dist = a.numel();*/
/* 	const typename SizeRef::value_type a_mat_size = rows * inter;*/
/* 	const typename SizeRef::value_type first_arg = a_dist / a_mat_size;*/
/* 	std::vector<threading::block_ranges<3>> my_blocks;*/
/* 	my_blocks.reserve(numSegments);*/
/* 	for(typename SizeRef::value_type i = 0; i < numSegments; ++i){*/
/* 		const typename SizeRef::value_type& row_begin = row_begins[i];*/
/* 		const typename SizeRef::value_type& row_end = row_begins[i+1];*/
/* 		my_blocks.push_back(threading::block_ranges<3>(0, first_arg, row_begin, row_end, 0, outpSize.back()));*/
/* 	}*/

/* 	std::vector<std::tuple<typename SizeRef::value_type, typename SizeRef::value_type, typename SizeRef::value_type> > increments(numParallel);*/
/* 	const typename SizeRef::value_type increment_num = outpSize.multiply() * DTypeFuncs::size_of_dtype(a.dtype);*/
/* 	auto begin = increments.begin();*/
/* 	typename SizeRef::value_type index = 0;*/
/* 	for(typename SizeRef::value_type i = 0; i < numSegments; ++i){*/
/* 		const std::vector<threading::blocked_range<3> >& blocks = my_blocks[i].getBlocks();*/
/* 		for(typename SizeRef::value_type j = 0; j < blocks.size(); ++j, ++begin){*/
/* 			const typename SizeRef::value_type max_bytes = blocks[j].blockSize * DTypeFuncs::size_of_dtype(a.dtype);*/
/* 			*begin = std::tuple<typename SizeRef::value_type, typename SizeRef::value_type, typename SizeRef::value_type>(index, max_bytes, increment_num);*/
/* 			index += max_bytes;*/
/* 		}*/
/* 	}*/

	
/* 	ArrayVoid outpAV = pool::apply_unary_function(*/
/* 		[&a, &b, &row_begins, &outpSize, &inter, &my_blocks](void* memory, const typename SizeRef::value_type& process_num, int (*pipes)[2]){*/

/* 			const typename SizeRef::value_type& row_begin = row_begins[process_num];*/
/* 			const typename SizeRef::value_type& row_end = row_begins[process_num+1];*/
/* 			typename SizeRef::value_type i = 0;*/
/* 			a.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(transposed_matmult_piped,*/
/* 					b.arr_void(),*/
/* 					memory,*/
/* 					outpSize[-2],*/
/* 					outpSize.back(),*/
/* 					inter,*/
/* 					0,*/
/* 					a.numel(),*/
/* 					row_begin,*/
/* 					row_end,*/
/* 					pipes,*/
/* 					my_blocks[process_num]);*/

/* 		},*/
/* 		a.dtype,*/
/* 		outpSize.multiply(),*/
/* 		increments);*/
/* 	Tensor output(std::move(outpAV), std::move(outpSize));*/


/* #endif*/
/* 	return std::move(output);*/
/* }*/

//void matrix_multiplication_typed(const bool& transpose_a, const bool& transpose_b, const T* A, const T* B, T* C, int64_t A_rows, int64_t B_cols, int64_t A_cols) noexcept {


template<typename T>
Tensor matmult_std_single(const T* A, const T* B, const SizeRef& o_a_shape, const SizeRef& o_b_shape, const bool& transpose_a, const bool& transpose_b){
	SizeRef a_shape = (transpose_a) ? o_a_shape.transpose(-1,-2) : o_a_shape.clone();

	SizeRef b_shape = (transpose_b) ? o_b_shape.transpose(-1,-2) : o_b_shape.clone();
	/* SizeRef a_shape = o_a_shape; */
	/* SizeRef b_shape = o_b_shape; */
	utils::THROW_EXCEPTION(a_shape[-1] == b_shape[-2], "Expected at positions (-1,-2) for shapes $,$ -> ($:$) to be equal", a_shape, b_shape, a_shape[-1], b_shape[-2]);
	std::vector<typename Tensor::size_value_t> vec = a_shape.Vec();
	vec.back() = b_shape.back();
	SizeRef c_shape(std::move(vec));

	Tensor tensor_C(std::move(c_shape), DTypeFuncs::type_to_dtype<T>);
	
	T* C = reinterpret_cast<T*>(tensor_C.data_ptr());

	const int64_t& m = o_a_shape[-2]; // Number of rows in A
	const int64_t& n = o_b_shape[-1]; // Number of columns in B
	const int64_t& k = o_a_shape[-1]; // Number of columns in A and rows in B
	// Perform matrix multiplication: C = A * B
	matrix_multiplication_typed<T>(transpose_a, transpose_b, A, B, C, m, n, k);
	return std::move(tensor_C);

}


template<typename T>
intrusive_ptr<T[]> convert_to_intrusive_bucketed(const Tensor& t){
	constexpr DType dt = DTypeFuncs::type_to_dtype<T>;
	utils::THROW_EXCEPTION(t.dtype == dt, "Expected to convert tensor of dtype $ but got $ to contiguous intrusive_ptr", dt, t.dtype);
	void** strides = t.arr_void().stride_begin();
	const int64_t& stride_size = t.arr_void().get_bucket().stride_amt();
	intrusive_ptr<T[]> output = intrusive_ptr<T[]>::make_aligned(t.numel(), 64);
	T* begin = output.get();
	for(uint64_t i = 0; i < stride_size; ++i){
		const T* sBegin = reinterpret_cast<const T*>(strides[i]);
		++i;
		const T* sEnd = reinterpret_cast<const T*>(strides[i]);
		std::ptrdiff_t distance = (sEnd - sBegin);
		std::copy(sBegin, sEnd, begin);
		begin += distance;
	}

	return std::move(output);
}




template<typename T>
constexpr DType get_convert_to_intrusive_strided_dtype(){
	if constexpr (std::is_same_v<T, std::complex<float>>){
		return DType::Complex64;
	}
	else if constexpr (std::is_same_v<T, std::complex<double>>){
		return DType::Complex128;
	}
	return DTypeFuncs::type_to_dtype<T>;
}

template<typename T>
intrusive_ptr<T[]> convert_to_intrusive_strided(const Tensor& t){
	constexpr DType dt = get_convert_to_intrusive_strided_dtype<T>();

	utils::THROW_EXCEPTION(t.dtype == dt, "Expected to convert tensor of dtype $ but got $ to contiguous intrusive_ptr", dt, t.dtype);
	return t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<dt> > >(
			[&](auto begin, auto end){
				intrusive_ptr<T[]> intrusive_t = intrusive_ptr<T[]>::make_aligned(t.numel(), 64);
				std::copy(begin, end, intrusive_t.get());
				return std::move(intrusive_t);
			});
}

//this is to determine if t was strided, and if so was it because a transpose(-1,-2) happened, then I can avoid copying any memory and just look at the memory that is already there contiguously
bool was_transposed_back(const Tensor& t){
	std::vector<Tensor::size_value_t> current_strides = t.getChangedStrides();
	std::vector<Tensor::size_value_t> supposed_strides(current_strides.cbegin(), current_strides.cend());
	std::sort(supposed_strides.begin(), supposed_strides.end(), [](auto a, auto b){return a > b;});
	if(supposed_strides.back() == current_strides[current_strides.size()-2] && current_strides.back() == supposed_strides[supposed_strides.size()-2]){
		return true;
	}
	return false;
}


//this assumes there is already a transpose(-1,-2) that happened
//it basically sees how far back it can stay contiguous before needing to be bucketed again
int64_t handle_looking_contiguous(const Tensor& t){ 
	std::vector<Tensor::size_value_t> current_strides = t.getChangedStrides();
	std::vector<Tensor::size_value_t> supposed_strides(current_strides.cbegin(), current_strides.cend());
	std::sort(supposed_strides.begin(), supposed_strides.end(), [](auto a, auto b){return a > b;});
	
	int64_t last_change = -1;
	for(uint64_t i = 1; i < current_strides.size()-2; ++i){
		if(supposed_strides[i] != current_strides[i]){last_change = (i-1);}
	}
	return last_change;
}

Tensor make_contiguous_look(const Tensor& t, const int64_t&& last_change){
	if(last_change == -1){
		if(t.arr_void().get_bucket().force_contig_size() != t.numel()
				|| !t.arr_void().get_bucket().can_force_contiguity())
			return t.transpose(-1,-2).contiguous();
		return t.force_contiguity().view(t.shape().transpose(-1,-2));
	}
	const Tensor split = t.split_axis(last_change);
	Tensor output = Tensor::makeNullTensorArray(split.numel());
	const Tensor* begin = reinterpret_cast<const Tensor*>(split.data_ptr());
	const Tensor* end = begin + split.numel();
	Tensor* o_begin = reinterpret_cast<Tensor*>(output.data_ptr());
	const int64_t check_bytes = o_begin->numel() * DTypeFuncs::size_of_dtype(o_begin->dtype);
	for(;begin != end; ++begin, ++o_begin){
		if(begin->arr_void().get_bucket().force_contig_size() != t.numel()
				|| !begin->arr_void().get_bucket().can_force_contiguity_bytes(check_bytes)){
			*o_begin = begin->transpose(-1,-2).contiguous();
		}
		*o_begin = begin->force_contiguity().view(begin->shape().transpose(-1,-2));
	}
	return std::move(output);
}


template<typename T>
Tensor handle_dim2_matmult_std(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){
	uint32_t iterator_type_a = a.arr_void().get_bucket().iterator_type(); // 3 = strided_view, 2 = bucketed, 1 = is_contiguous
	uint32_t iterator_type_b = b.arr_void().get_bucket().iterator_type(); // 
	
	bool transpose_a, transpose_b;
	determine_transposes(a, b, transpose_a, transpose_b, trans_a, trans_b);

	if(iterator_type_a == 1){ // contiguous
		const T* A = a.arr_void().get_bucket().cbegin<1, T>();
		if(iterator_type_b == 1){
			const T* B = b.arr_void().get_bucket().cbegin<1, T>();
			return matmult_std_single<T>(A, B, a.shape(), b.shape(), transpose_a, transpose_b);
		}

		if(iterator_type_b == 3 && was_transposed_back(b)){
			Tensor contiguous_b = make_contiguous_look(b, handle_looking_contiguous(b)); //there is no possibility for it to be an array of tensors since it is only dim() == 2
			transpose_b = !transpose_b;
			const T* B = contiguous_b.arr_void().get_bucket().cbegin<1, T>();
			return matmult_std_single<T>(A, B, a.shape(), contiguous_b.shape(), transpose_a, transpose_b);

		}
		//this is if it is bucketed or strided
		//normally if it were bucketed, it would see if there is any possible way to seperate it into matricies, otherwise just this:
		intrusive_ptr<T[]> intrusive_B = (iterator_type_b == 3) ? convert_to_intrusive_strided<T>(b) : convert_to_intrusive_bucketed<T>(b);
		const T* B = intrusive_B.get();
		return matmult_std_single<T>(A, B, a.shape(), b.shape(), transpose_a, transpose_b);
	}
	if(iterator_type_a == 2){ //bucketed
		intrusive_ptr<T[]> intrusive_A = convert_to_intrusive_bucketed<T>(a);
		const T* A = intrusive_A.get();
		if(iterator_type_b == 1){
			const T* B = b.arr_void().get_bucket().cbegin<1, T>();
			return matmult_std_single<T>(A, B, a.shape(), b.shape(), transpose_a, transpose_b);
		}

		if(iterator_type_b == 3 && was_transposed_back(b)){
			Tensor contiguous_b = make_contiguous_look(b, handle_looking_contiguous(b)); //there is no possibility for it to be an array of tensors since it is only dim() == 2
			transpose_b = !transpose_b;
			const T* B = contiguous_b.arr_void().get_bucket().cbegin<1, T>();
			return matmult_std_single<T>(A, B, a.shape(), contiguous_b.shape(), transpose_a, transpose_b);

		}
		//this is if it is bucketed or strided
		//normally if it were bucketed, it would see if there is any possible way to seperate it into matricies, otherwise just this:
		intrusive_ptr<T[]> intrusive_B = (iterator_type_b == 3) ? convert_to_intrusive_strided<T>(b) : convert_to_intrusive_bucketed<T>(b);
		const T* B = intrusive_B.get();
		return matmult_std_single<T>(A, B, a.shape(), b.shape(), transpose_a, transpose_b);
	}
	if(was_transposed_back(a)){
		Tensor contiguous_a = make_contiguous_look(a, handle_looking_contiguous(a)); //there is no possibility for it to be an array of tensors since it is only dim() == 2
		transpose_a = !transpose_a;
		const T* A = contiguous_a.arr_void().get_bucket().cbegin<1, T>();
		if(iterator_type_b == 1){
			const T* B = b.arr_void().get_bucket().cbegin<1, T>();
			return matmult_std_single<T>(A, B, a.shape(), b.shape(), transpose_a, transpose_b);
		}

		if(iterator_type_b == 3 && was_transposed_back(b)){
			Tensor contiguous_b = make_contiguous_look(b, handle_looking_contiguous(b)); //there is no possibility for it to be an array of tensors since it is only dim() == 2
			transpose_b = !transpose_b;
			const T* B = contiguous_b.arr_void().get_bucket().cbegin<1, T>();
			return matmult_std_single<T>(A, B, a.shape(), contiguous_b.shape(), transpose_a, transpose_b);

		}
		//this is if it is bucketed or strided
		//normally if it were bucketed, it would see if there is any possible way to seperate it into matricies, otherwise just this:
		intrusive_ptr<T[]> intrusive_B = (iterator_type_b == 3) ? convert_to_intrusive_strided<T>(b) : convert_to_intrusive_bucketed<T>(b);
		const T* B = intrusive_B.get();
		return matmult_std_single<T>(A, B, contiguous_a.shape(), b.shape(), transpose_a, transpose_b);
	}
	intrusive_ptr<T[]> intrusive_A = convert_to_intrusive_strided<T>(a);
	const T* A = intrusive_A.get();
	if(iterator_type_b == 1){
		const T* B = b.arr_void().get_bucket().cbegin<1, T>();
		return matmult_std_single<T>(A, B, a.shape(), b.shape(), transpose_a, transpose_b);
	}

	if(iterator_type_b == 3 && was_transposed_back(b)){
		Tensor contiguous_b = make_contiguous_look(b, handle_looking_contiguous(b)); //there is no possibility for it to be an array of tensors since it is only dim() == 2
		transpose_b = !transpose_b;
		const T* B = contiguous_b.arr_void().get_bucket().cbegin<1, T>();
		return matmult_std_single<T>(A, B, a.shape(), contiguous_b.shape(), transpose_a, transpose_b);

	}
	//this is if it is bucketed or strided
	//normally if it were bucketed, it would see if there is any possible way to seperate it into matricies, otherwise just this:
	intrusive_ptr<T[]> intrusive_B = (iterator_type_b == 3) ? convert_to_intrusive_strided<T>(b) : convert_to_intrusive_bucketed<T>(b);
	const T* B = intrusive_B.get();
	return matmult_std_single<T>(A, B, a.shape(), b.shape(), transpose_a, transpose_b);
}

int64_t handle_looking_buckets(const Tensor& t){
	std::size_t dtype_size = sizeof(float);
	void** strides = t.arr_void().stride_begin();
	const int64_t& stride_size = t.arr_void().get_bucket().stride_amt();
	const int64_t& bucket_size = t.arr_void().get_bucket().buckets_amt();
	const int64_t mat_size = t.shape().multiply(-2);
	/* std::cout << "mat size: "<<mat_size << std::endl; */
	
	std::vector<std::ptrdiff_t> sizes(bucket_size);
	int64_t counter = 0;
	for(uint64_t i = 0; i < stride_size; ++i, ++counter){
		const uint8_t* sBegin = reinterpret_cast<const uint8_t*>(strides[i]);
		++i;
		const uint8_t* sEnd = reinterpret_cast<const uint8_t*>(strides[i]);
		std::ptrdiff_t distance = (sEnd - sBegin) / dtype_size;
		if(distance < mat_size || distance % mat_size != 0){return -1;}
		sizes[counter] = distance;
	}
	int64_t axis = -3;
	counter = 3;
	int64_t cur_size = t.shape().multiply((-1)*counter);
	while(true){
		for(const auto& size : sizes){
			/* std::cout << "size: "<<size<<" cur size: "<<cur_size << std::endl; */
			if(size < cur_size || size % cur_size != 0){return axis;}
		}
		++counter;
		--axis;
		if(counter == t.dims()){return axis;}
		cur_size = t.shape().multiply((-1) * counter);
	}
}


Tensor split_and_contigitize(const Tensor& t){
	Tensor splitting = t.split_axis(-3); //first it splits along all matriciess
	Tensor* begin = reinterpret_cast<Tensor*>(splitting.data_ptr());
	Tensor* end = begin + splitting.numel();
	for(;begin != end; ++begin){
		if(!begin->is_contiguous()){
			Tensor contiged = begin->contiguous();
			std::swap(*begin, contiged);
		}
	}
	return std::move(splitting);
}


Tensor return_split_tensor(const Tensor& t, bool& transpose){
	/* uint32_t iterator_type_a = a.arr_void().get_bucket().iterator_type(); // 3 = strided_view, 2 = bucketed, 1 = is_contiguous */
	uint32_t iterator_type = t.arr_void().get_bucket().iterator_type();
	if(iterator_type == 3){ //strided
		if(was_transposed_back(t)){
			transpose = !transpose;
			return make_contiguous_look(t, -3);
		}
		return split_and_contigitize(t);
	}
	if(iterator_type == 2){ //bucketed
		int64_t split_by = handle_looking_buckets(t);
		if(split_by == -1){return split_and_contigitize(t);}
	}
	//already contiguous
	return t.split_axis(-3);
}

void return_split_tensor(const Tensor& t, std::__bit_reference<std::vector<bool>> transpose, Tensor& out_t){
	/* uint32_t iterator_type_a = a.arr_void().get_bucket().iterator_type(); // 3 = strided_view, 2 = bucketed, 1 = is_contiguous */
	uint32_t iterator_type = t.arr_void().get_bucket().iterator_type();
	if(iterator_type == 3){ //strided
		if(was_transposed_back(t)){
			transpose = !transpose;
			out_t = make_contiguous_look(t, -3);
			return;
		}
		out_t = split_and_contigitize(t);
		return;
	}
	if(iterator_type == 2){ //bucketed
		int64_t split_by = handle_looking_buckets(t);
		if(split_by == -1){out_t = split_and_contigitize(t); return;}
	}
	//already contiguous
	out_t = t.split_axis(-3);
}

Tensor return_split_tensor_of_tensors(const Tensor& t, std::vector<bool>& transpose){
	Tensor out = Tensor::makeNullTensorArray(t.numel());
	const Tensor* begin = reinterpret_cast<const Tensor*>(t.data_ptr());
	const Tensor* end = begin + t.numel();
	Tensor* begin_o = reinterpret_cast<Tensor*>(out.data_ptr());
	auto t_begin = transpose.begin();
	for(;begin != end; ++begin, ++begin_o, ++t_begin){
		return_split_tensor(*begin, *t_begin, *begin_o);
	}
	return std::move(out);
}


void calculate_optimal_group_std(int64_t M, int64_t K, int64_t N, int64_t batch_size, int64_t& group_count, int64_t*& group_sizes) {
    // Get cache sizes (assuming L1, L2, and L3)
    /* uint64_t L1_cache_size = 32 * 1024;  // L1 cache size in bytes (32 KB) */
    /* uint64_t L2_cache_size = 256 * 1024; // L2 cache size in bytes (256 KB) */
    uint64_t L3_cache_size = 8 * 1024 * 1024; // L3 cache size in bytes (8 MB)

    // Calculate matrix sizes in bytes
    uint64_t size_A = M * K * sizeof(float);
    uint64_t size_B = K * N * sizeof(float);
    uint64_t size_C = M * N * sizeof(float);

    // Calculate the number of elements that can fit in each cache level
    /* uint64_t elements_L1 = L1_cache_size / sizeof(float); */
    /* uint64_t elements_L2 = L2_cache_size / sizeof(float); */
    uint64_t elements_L3 = L3_cache_size / sizeof(float);

    // Calculate the number of elements for each group
    uint64_t total_size = (size_A + size_B + size_C);
    uint64_t group_elements;
    if(total_size > elements_L3){
	group_elements = elements_L3;
    }
    else{
	group_elements = elements_L3 / total_size;
    }


    // Calculate the number of groups needed
    group_count = ceil((float)(batch_size * M * N) / group_elements);

    // Allocate memory for group sizes array
    group_sizes = new MKL_INT64[group_count];

    // Distribute batch size evenly among groups
    MKL_INT64 batch_per_group = ceil((float)batch_size / group_count);

    // Assign group sizes
    for (MKL_INT64 i = 0; i < group_count; ++i) {
        group_sizes[i] = std::min(batch_per_group, (MKL_INT64)batch_size);
        batch_size -= group_sizes[i];
    }
}


//expects all tensors to only have dim of 2, and all to have the same shape
template<typename T,
	typename outT = T,
	DType DT = DTypeFuncs::type_to_dtype<T>,
	DType outDT = DTypeFuncs::type_to_dtype<outT>>
Tensor handle_tensors_of_tensors_std(Tensor& a, Tensor& b, const bool transpose_a, const bool transpose_b){
	utils::THROW_EXCEPTION(a.dtype == b.dtype && a.dtype == DType::TensorObj, "Expected dtypes to be TensorObj but got $ and $", a.dtype, b.dtype);
	utils::THROW_EXCEPTION(a.numel() == b.numel(), "Expected tensor a and tensor b to have the same numel but got $ and $", a.numel(), b.numel());
	

	const T* A_array[a.numel()];
	const T* B_array[a.numel()];
	outT* C_array[a.numel()];


	Tensor* begin = reinterpret_cast<Tensor*>(a.data_ptr());
	Tensor* end = begin + a.numel();
	Tensor* begin_b = reinterpret_cast<Tensor*>(b.data_ptr());
	
	SizeRef a_shape = (transpose_a) ? begin->shape().transpose(-1,-2) : begin->shape().clone();
	SizeRef b_shape = (transpose_b) ? begin_b->shape().transpose(-1,-2) : begin_b->shape().clone();
	/* SizeRef a_shape = begin->shape(); */
	/* SizeRef b_shape = begin_b->shape(); */

	SizeRef begin_shape = begin->shape();
	SizeRef begin_b_shape = begin_b->shape();
	const int64_t& batch_size = a.numel();
	const int64_t& M = begin_shape[-2]; // Number of rows in A
	const int64_t& N = (transpose_a && transpose_b) ? b_shape[-1] : begin_b_shape[-1]; // Number of columns in B
	const int64_t& K = (transpose_a) ? a_shape[-2] : (transpose_b) ? b_shape[-1] : begin_shape[-1]; // Number of columns in A and rows in B
	
	//on transposed_b, what it generally should be for a (6,4,5) * (6,2,5) (where (6,2,5)->(6,5,2) for a transpose)
	//int m = 4, n = 5, k = 2;
	//m = begin_shape[-2]
	//n = begin_b_shape[-1]
	//k = b_shape[-1] or begin_b_shape[-2]
	//
	//on transposed_a what it should generally be for a (6,4,2) * (6,4,5) after transpose: ((6,2,4) * (6,4,5))
	//int m = 4, n = 5, k = 2;
	//m = begin_shape[-2]
	//n = begin_b_shape[-1]
	//k = a_shape[-2] or begin_shape[-1]
	//
	//k = (transpose_a) ? a_shape[-2] : (transpose_b) ? b_shape[-1] : begin_shape[-1]

	std::vector<typename Tensor::size_value_t> vec = {batch_size, a_shape[-2], b_shape[-1]};
	SizeRef c_shape(std::move(vec));
	Tensor tensor_C = zeros(std::move(c_shape), outDT); 
	outT* C = reinterpret_cast<outT*>(tensor_C.data_ptr());
	const int64_t c_matrix_size = tensor_C.shape()[-1] * tensor_C.shape()[-2];
	
	int64_t counter = 0;
	for(;begin != end; ++begin, ++begin_b, ++counter){
		//just going to make them contiguous, too much work to try and split them all up
		//at this point they should all be contiguous anyways
		//so it's just more of a final check to make sure
		if(!begin->is_contiguous()){
			*begin = begin->contiguous();
		}
		if(!begin_b->is_contiguous()){
			*begin_b = begin_b->contiguous();
		}
		utils::THROW_EXCEPTION(begin->dims() == begin_b->dims() && begin->dims() == 2, "Expected tensors to only contain matricies but got dims $ and $", begin->dims(), begin_b->dims());
		utils::THROW_EXCEPTION(begin->dtype == begin_b->dtype && begin->dtype == DT, "Expected both dtypes to be $ but got $ and $ for tensors of tensors", DT, begin->dtype, begin_b->dtype);
		utils::THROW_EXCEPTION(begin->shape() == begin_shape, "Expected all shapes inside of tensors to match but got $ and $", begin->shape(), begin_shape);
		utils::THROW_EXCEPTION(begin_b->shape() == begin_b_shape, "Expected all shapes inside of tensors to match but got $ and $", begin_b->shape(), begin_b_shape);
		A_array[counter] = begin->arr_void().get_bucket().begin<1, T>();
		B_array[counter] = begin_b->arr_void().get_bucket().begin<1, T>();
		C_array[counter] = C + (counter * c_matrix_size);
	}
	
	int64_t group_count;
	int64_t* group_size;

	/* std::cout << "going to calculate optimal group size..."<<std::endl; */
	calculate_optimal_group_std(M, K, N, batch_size, group_count, group_size);
	/* std::cout << "calculated"<<std::endl; */
	
/* void batched_matrix_multiplication_typed(const bool& transpose_a, const bool& transpose_b, const T* A, const T* B, T* C, int64_t A_rows, int64_t B_cols, int64_t A_cols, const int64_t& group_count, const int64_t* group_sizes) noexcept { */
	
	batched_matrix_multiplication_typed<T>(transpose_a, transpose_b, A_array, B_array, C_array, M, N, K, group_count, group_size);

	delete[] group_size;
	return std::move(tensor_C);

	
}

template<typename T>
Tensor handle_dim_n_k_matmult_std(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){
	
	constexpr DType in_dtype = DTypeFuncs::type_to_dtype<T>;
	utils::THROW_EXCEPTION(a.dims() > b.dims(), "Expected to have larger tensor in terms of dims as a but got $ and $", a.dims(), b.dims());
	utils::THROW_EXCEPTION(a.dtype == in_dtype && b.dtype == in_dtype, "Expected both tensors to have the dtype $ but got $ and $", in_dtype, a.dtype, b.dtype);
	int64_t start = a.dims()-b.dims();
	for(int64_t i = start; i < a.dims() - 2; ++i){
		utils::THROW_EXCEPTION(a.shape()[i] == b.shape()[i-start], "Cannot multiply tensors of shapes $ and $, outter dimensions do not match", a.shape(), b.shape());
	}
	
	bool transpose_a, transpose_b;
	determine_transposes(a, b, transpose_a, transpose_b, trans_a, trans_b);

	auto bigger_shape_vec = a.shape().arr();

	Tensor bigger = return_split_tensor(a, transpose_a);
	Tensor smaller = return_split_tensor(b, transpose_b);
	utils::THROW_EXCEPTION(bigger.numel() % smaller.numel() == 0, "Expected there to be a fraction of arrays in the tensors but got $ and  $ as the amount of matricies", bigger.numel(), smaller.numel());
	smaller = smaller.repeat_(bigger.numel() / smaller.numel());
	Tensor output = handle_tensors_of_tensors_std<T>(bigger, smaller, transpose_a, transpose_b);
	bigger_shape_vec[-1] = output.shape()[-1];
	bigger_shape_vec[-2] = output.shape()[-2];
	return output.view(SizeRef(std::move(bigger_shape_vec)));
}

template<typename T>
Tensor handle_dim_k_n_matmult_std(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){
	constexpr DType in_dtype = DTypeFuncs::type_to_dtype<T>;
	utils::THROW_EXCEPTION(b.dims() > a.dims(), "Expected to have larger tensor in terms of dims as a but got $ > $", b.dims(), a.dims());
	utils::THROW_EXCEPTION(a.dtype == in_dtype && b.dtype == in_dtype, "Expected both tensors to have the dtype $ but got $ and $", in_dtype, a.dtype, b.dtype);
	int64_t start = b.dims()-a.dims();
	for(int64_t i = start; i < b.dims() - 2; ++i){
		utils::THROW_EXCEPTION(a.shape()[i-start] == b.shape()[i], "Cannot multiply tensors of shapes $ and $, outter dimensions do not match", a.shape(), b.shape());
	}
	
	bool transpose_a, transpose_b;
	determine_transposes(a, b, transpose_a, transpose_b, trans_a, trans_b);
	
	auto bigger_shape_vec = b.shape().arr();

	Tensor bigger = return_split_tensor(b, transpose_b);
	Tensor smaller = return_split_tensor(a, transpose_a);
	utils::THROW_EXCEPTION(bigger.numel() % smaller.numel() == 0, "Expected there to be a fraction of arrays in the tensors but got $ and  $ as the amount of matricies", bigger.numel(), smaller.numel());
	smaller = smaller.repeat_(bigger.numel() / smaller.numel());
	Tensor output = handle_tensors_of_tensors_std<T>(smaller, bigger, transpose_a, transpose_b);
	bigger_shape_vec[-1] = output.shape()[-1];
	bigger_shape_vec[-2] = output.shape()[-2];
	return output.view(SizeRef(std::move(bigger_shape_vec)));
}

template<typename T>
Tensor handle_dim_n_n_matmult_std(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){
	constexpr DType in_dtype = DTypeFuncs::type_to_dtype<T>;
	utils::THROW_EXCEPTION(b.dims() == a.dims(), "Expected to have same tensor dims as a but got $ != $", b.dims(), a.dims());
	utils::THROW_EXCEPTION(a.dtype == in_dtype && b.dtype == in_dtype, "Expected both tensors to have the dtype $ but got $ and $", in_dtype, a.dtype, b.dtype);
	int64_t start = b.dims()-a.dims();
	for(int64_t i = start; i < b.dims() - 2; ++i){
		utils::THROW_EXCEPTION(a.shape()[i-start] == b.shape()[i], "Cannot multiply tensors of shapes $ and $, outter dimensions do not match", a.shape(), b.shape());
	}
	
	auto bigger_shape_vec = a.shape().arr();
	bool transpose_a, transpose_b;
	determine_transposes(a, b, transpose_a, transpose_b, trans_a, trans_b);
	Tensor A = return_split_tensor(a, transpose_a);
	Tensor B = return_split_tensor(b, transpose_b);
	utils::THROW_EXCEPTION(A.numel() == B.numel(), "Expected there to be the same number of matricies but got $ and  $ as the amount of matricies", A.numel(), B.numel());
	Tensor output = handle_tensors_of_tensors_std<T>(A, B, transpose_a, transpose_b);
	bigger_shape_vec[-1] = output.shape()[-1];
	bigger_shape_vec[-2] = output.shape()[-2];
	return output.view(SizeRef(std::move(bigger_shape_vec)));

}


//a and be are tensors of tensors of tensors of type T
template<typename T,
	typename outT = T,
	DType DT = DTypeFuncs::type_to_dtype<T>,
	DType outDT = DTypeFuncs::type_to_dtype<outT>>
Tensor handle_tensors_of_tensors_std_subT(Tensor& a, Tensor& b, const bool transpose_a, const bool transpose_b){
	utils::THROW_EXCEPTION(a.dtype == b.dtype && a.dtype == DType::TensorObj, "Expected dtypes to be TensorObj but got $ and $", a.dtype, b.dtype);
	utils::THROW_EXCEPTION(a.numel() == b.numel(), "Expected tensor a and tensor b to have the same numel but got $ and $", a.numel(), b.numel());
	

	const T* A_array[a.numel() * a[0].item<Tensor>().numel()];
	const T* B_array[a.numel() * a[0].item<Tensor>().numel()];
	outT* C_array[a.numel() * a[0].item<Tensor>().numel()];


	Tensor* begin = reinterpret_cast<Tensor*>(a.data_ptr());
	Tensor* end = begin + a.numel();
	Tensor* begin_b = reinterpret_cast<Tensor*>(b.data_ptr());
	
	SizeRef a_shape = (transpose_a) ? (*begin)[0].item<Tensor>().shape().transpose(-1,-2) :(*begin)[0].item<Tensor>().shape().clone();
	SizeRef b_shape = (transpose_b) ? (*begin_b)[0].item<Tensor>().shape().transpose(-1,-2) : (*begin_b)[0].item<Tensor>().shape().clone();
	/* SizeRef a_shape = begin->shape(); */
	/* SizeRef b_shape = begin_b->shape(); */

	SizeRef begin_shape = (*begin)[0].item<Tensor>().shape();
	SizeRef begin_b_shape = (*begin_b)[0].item<Tensor>().shape();
	const int64_t batch_size = a.numel() * begin->numel();
	const int64_t& M = begin_shape[-2]; // Number of rows in A
	const int64_t& N = (transpose_a && transpose_b) ? b_shape[-1] : begin_b_shape[-1]; // Number of columns in B
	const int64_t& K = (transpose_a) ? a_shape[-2] : (transpose_b) ? b_shape[-1] : begin_shape[-1]; // Number of columns in A and rows in B
	
	//on transposed_b, what it generally should be for a (6,4,5) * (6,2,5) (where (6,2,5)->(6,5,2) for a transpose)
	//int m = 4, n = 5, k = 2;
	//m = begin_shape[-2]
	//n = begin_b_shape[-1]
	//k = b_shape[-1] or begin_b_shape[-2]
	//
	//on transposed_a what it should generally be for a (6,4,2) * (6,4,5) after transpose: ((6,2,4) * (6,4,5))
	//int m = 4, n = 5, k = 2;
	//m = begin_shape[-2]
	//n = begin_b_shape[-1]
	//k = a_shape[-2] or begin_shape[-1]
	//
	//k = (transpose_a) ? a_shape[-2] : (transpose_b) ? b_shape[-1] : begin_shape[-1]

	std::vector<typename Tensor::size_value_t> vec = {batch_size, a_shape[-2], b_shape[-1]};
	SizeRef c_shape(std::move(vec));
	Tensor tensor_C = zeros(std::move(c_shape), outDT); 
	outT* C = reinterpret_cast<outT*>(tensor_C.data_ptr());
	const int64_t c_matrix_size = tensor_C.shape()[-1] * tensor_C.shape()[-2];
	
	int64_t counter = 0;
	for(;begin != end; ++begin, ++begin_b){
		//now the second nested for loop for the inner tensors
		Tensor* sub_begin = reinterpret_cast<Tensor*>(begin->data_ptr());
		Tensor* sub_end = sub_begin + begin->numel();
		Tensor* sub_begin_b = reinterpret_cast<Tensor*>(begin_b->data_ptr());
		for(;sub_begin != sub_end; ++sub_begin, ++sub_begin_b, ++counter){
			//just going to make them contiguous, too much work to try and split them all up
			//at this point they should all be contiguous anyways
			//so it's just more of a final check to make sure
			if(!sub_begin->is_contiguous()){
				*sub_begin = sub_begin->contiguous();
			}
			if(!sub_begin_b->is_contiguous()){
				*sub_begin_b = sub_begin_b->contiguous();
			}
			utils::THROW_EXCEPTION(sub_begin->dims() == sub_begin_b->dims() && sub_begin->dims() == 2, "Expected tensors to only contain matricies but got dims $ and $", sub_begin->dims(), sub_begin_b->dims());
			utils::THROW_EXCEPTION(sub_begin->dtype == sub_begin_b->dtype && sub_begin->dtype == DT, "Expected both dtypes to be $ but got $ and $ for tensors of tensors", DT, sub_begin->dtype, sub_begin_b->dtype);
			utils::THROW_EXCEPTION(sub_begin->shape() == begin_shape, "Expected all shapes inside of tensors to match but got $ and $", sub_begin->shape(), begin_shape);
			utils::THROW_EXCEPTION(sub_begin_b->shape() == begin_b_shape, "Expected all shapes inside of tensors to match but got $ and $", sub_begin_b->shape(), begin_b_shape);

			A_array[counter] = sub_begin->arr_void().get_bucket().begin<1, T>();
			B_array[counter] = sub_begin_b->arr_void().get_bucket().begin<1, T>();
			C_array[counter] = C + (counter * c_matrix_size);	
		}

	}
	
	int64_t group_count;
	int64_t* group_size;

	/* std::cout << "going to calculate optimal group size..."<<std::endl; */
	calculate_optimal_group_std(M, K, N, batch_size, group_count, group_size);
	/* std::cout << "calculated"<<std::endl; */
	
/* void batched_matrix_multiplication_typed(const bool& transpose_a, const bool& transpose_b, const T* A, const T* B, T* C, int64_t A_rows, int64_t B_cols, int64_t A_cols, const int64_t& group_count, const int64_t* group_sizes) noexcept { */
	
	batched_matrix_multiplication_typed<T>(transpose_a, transpose_b, A_array, B_array, C_array, M, N, K, group_count, group_size);

	delete[] group_size;
	return std::move(tensor_C);

	
}

template<typename T>
Tensor handle_dim_n_k_subT_matmult_std(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){
	constexpr DType in_dtype = DTypeFuncs::type_to_dtype<T>;
	const Tensor& temp_a = a[0].item<Tensor>();
	const Tensor& temp_b = b[0].item<Tensor>();
	utils::THROW_EXCEPTION(temp_b.dims() < temp_a.dims(), "Expected to have larger tensor in terms of dims as a but got $ < $", b.dims(), a.dims());
	utils::THROW_EXCEPTION(temp_a.dtype == in_dtype && temp_b.dtype == in_dtype, "Expected both tensors to have the dtype $ but got $ and $", in_dtype, temp_a.dtype, temp_b.dtype);
	int64_t start = temp_a.dims()-temp_b.dims();
	for(int64_t i = start; i < temp_a.dims() - 2; ++i){
		utils::THROW_EXCEPTION(temp_b.shape()[i-start] == temp_a.shape()[i], "Cannot multiply tensors of shapes $ and $, outter dimensions do not match", temp_a.shape(), temp_b.shape());
	}
	
	bool transpose_a, transpose_b;
	determine_transposes(temp_a, temp_b, transpose_a, transpose_b, trans_a, trans_b);
	std::vector<bool> n_transpose_a(a.numel(), transpose_a);
	std::vector<bool> n_transpose_b(b.numel(), transpose_b);
	Tensor bigger = return_split_tensor_of_tensors(a, n_transpose_a);
	Tensor smaller = return_split_tensor_of_tensors(b, n_transpose_b);
	const bool transpose_a_get = n_transpose_a[0];
	const bool transpose_b_get = n_transpose_b[0];
	std::for_each(n_transpose_a.cbegin(), n_transpose_a.cend(), [&transpose_a_get](const bool& t){utils::throw_exception(transpose_a_get == t, "Memory error with transposes a");});
	std::for_each(n_transpose_b.cbegin(), n_transpose_b.cend(), [&transpose_b_get](const bool& t){utils::throw_exception(transpose_b_get == t, "Memory error with transposes b");});
	
	auto bigger_shape_vec = temp_a.shape().arr();
	bigger_shape_vec[0] *= a.numel();

	Tensor* bigger_begin = reinterpret_cast<Tensor*>(bigger.data_ptr());
	Tensor* bigger_end = bigger_begin + bigger.numel();
	Tensor* smaller_begin = reinterpret_cast<Tensor*>(smaller.data_ptr());
	for(;bigger_begin != bigger_end; ++bigger_begin, ++smaller_begin){
		utils::THROW_EXCEPTION(bigger_begin->numel() % smaller_begin->numel() == 0, "Expected there to be a fraction of arrays in the tensors but got $ and  $ as the amount of matricies", bigger_begin->numel(), smaller_begin->numel());
		*smaller_begin = smaller_begin->repeat_(bigger_begin->numel() / smaller_begin->numel());
	
	}
	Tensor output = handle_tensors_of_tensors_std_subT<T>(bigger, smaller, transpose_a, transpose_b);
	bigger_shape_vec[-1] = output.shape()[-1];
	bigger_shape_vec[-2] = output.shape()[-2];
	return output.view(SizeRef(std::move(bigger_shape_vec)));
}

template<typename T>
Tensor handle_dim_k_n_subT_matmult_std(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){
	constexpr DType in_dtype = DTypeFuncs::type_to_dtype<T>;
	const Tensor& temp_a = a[0].item<Tensor>();
	const Tensor& temp_b = b[0].item<Tensor>();
	utils::THROW_EXCEPTION(temp_b.dims() > temp_a.dims(), "Expected to have larger tensor in terms of dims as a but got $ > $", b.dims(), a.dims());
	utils::THROW_EXCEPTION(temp_a.dtype == in_dtype && temp_b.dtype == in_dtype, "Expected both tensors to have the dtype $ but got $ and $", in_dtype, temp_a.dtype, temp_b.dtype);
	int64_t start = temp_b.dims()-temp_a.dims();
	for(int64_t i = start; i < temp_b.dims() - 2; ++i){
		utils::THROW_EXCEPTION(temp_a.shape()[i-start] == temp_b.shape()[i], "Cannot multiply tensors of shapes $ and $, outter dimensions do not match", temp_a.shape(), temp_b.shape());
	}
	
	bool transpose_a, transpose_b;
	determine_transposes(temp_a, temp_b, transpose_a, transpose_b, trans_a, trans_b);
	std::vector<bool> n_transpose_a(a.numel(), transpose_a);
	std::vector<bool> n_transpose_b(b.numel(), transpose_b);
	Tensor smaller = return_split_tensor_of_tensors(a, n_transpose_a);
	Tensor bigger = return_split_tensor_of_tensors(b, n_transpose_b);
	const bool transpose_a_get = n_transpose_a[0];
	const bool transpose_b_get = n_transpose_b[0];
	std::for_each(n_transpose_a.cbegin(), n_transpose_a.cend(), [&transpose_a_get](const bool& t){utils::throw_exception(transpose_a_get == t, "Memory error with transposes a");});
	std::for_each(n_transpose_b.cbegin(), n_transpose_b.cend(), [&transpose_b_get](const bool& t){utils::throw_exception(transpose_b_get == t, "Memory error with transposes b");});
	
	auto bigger_shape_vec = temp_b.shape().arr();
	bigger_shape_vec[0] *= b.numel();

	Tensor* bigger_begin = reinterpret_cast<Tensor*>(bigger.data_ptr());
	Tensor* bigger_end = bigger_begin + bigger.numel();
	Tensor* smaller_begin = reinterpret_cast<Tensor*>(smaller.data_ptr());
	for(;bigger_begin != bigger_end; ++bigger_begin, ++smaller_begin){
		utils::THROW_EXCEPTION(bigger_begin->numel() % smaller_begin->numel() == 0, "Expected there to be a fraction of arrays in the tensors but got $ and  $ as the amount of matricies", bigger_begin->numel(), smaller_begin->numel());
		*smaller_begin = smaller_begin->repeat_(bigger_begin->numel() / smaller_begin->numel());
	
	}
	Tensor output = handle_tensors_of_tensors_std_subT<T>(smaller, bigger, transpose_a, transpose_b);
	bigger_shape_vec[-1] = output.shape()[-1];
	bigger_shape_vec[-2] = output.shape()[-2];
	return output.view(SizeRef(std::move(bigger_shape_vec)));
}

//this function is called post tensors_check
template<typename T>
Tensor handle_dim_n_n_subT_matmult_std(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){
	constexpr DType in_dtype = DTypeFuncs::type_to_dtype<T>;
	const Tensor& temp_a = a[0].item<Tensor>();
	const Tensor& temp_b = b[0].item<Tensor>();
	utils::THROW_EXCEPTION(temp_b.dims() == temp_a.dims(), "Expected to have same tensor dims as a but got $ != $", temp_b.dims(), temp_a.dims());
	utils::THROW_EXCEPTION(temp_a.dtype == in_dtype && temp_b.dtype == in_dtype, "Expected both tensors to have the dtype $ but got $ and $", in_dtype, temp_a.dtype, temp_b.dtype);
	int64_t start = temp_b.dims()-temp_a.dims();
	for(int64_t i = start; i < b.dims() - 2; ++i){
		utils::THROW_EXCEPTION(temp_a.shape()[i-start] == temp_b.shape()[i], "Cannot multiply tensors of shapes $ and $, outter dimensions do not match", a.shape(), b.shape());
	}
	
	auto bigger_shape_vec = temp_a.shape().arr();
	bigger_shape_vec[0] *= a.numel();
	bool transpose_a, transpose_b;
	determine_transposes(temp_a, temp_b, transpose_a, transpose_b, trans_a, trans_b);
	std::vector<bool> n_transpose_a(a.numel(), transpose_a);
	std::vector<bool> n_transpose_b(b.numel(), transpose_b);
	Tensor A = return_split_tensor_of_tensors(a, n_transpose_a);
	Tensor B = return_split_tensor_of_tensors(b, n_transpose_b);
	const bool transpose_a_get = n_transpose_a[0];
	const bool transpose_b_get = n_transpose_b[0];
	std::for_each(n_transpose_a.cbegin(), n_transpose_a.cend(), [&transpose_a_get](const bool& t){utils::throw_exception(transpose_a_get == t, "Memory error with transposes a");});
	std::for_each(n_transpose_b.cbegin(), n_transpose_b.cend(), [&transpose_b_get](const bool& t){utils::throw_exception(transpose_b_get == t, "Memory error with transposes b");});

	utils::THROW_EXCEPTION(A.numel() == B.numel(), "Expected there to be the same number of matricies but got $ and  $ as the amount of matricies", A.numel(), B.numel());
	Tensor output = handle_tensors_of_tensors_std_subT<T>(A, B, transpose_a_get, transpose_b_get);
	bigger_shape_vec[-1] = output.shape()[-1];
	bigger_shape_vec[-2] = output.shape()[-2];
	return output.view(SizeRef(std::move(bigger_shape_vec)));

}


DType subT_check(const Tensor& a, const Tensor& b){
	utils::throw_exception(a.dtype == b.dtype, "Expected both tensors to be dtype Tensor but got $ and $", a.dtype, b.dtype);
	utils::throw_exception(a.is_contiguous() && b.is_contiguous(), "Expected both tensor of tensors to be contiguous");
	utils::throw_exception(a.numel() == b.numel(), "Expected both tensors to have the same number of sub-tensors, going to have these multiplied in a list-like fashion, but got $ and $ tensors", a.numel(), b.numel());
	utils::throw_exception(a.dims() == 1 && b.dims() == a.dims(), "Expected both tensor of tensors to have dimensions of 1, but got $ and $ tensors", a.dims(), b.dims());

	const Tensor* begin = reinterpret_cast<const Tensor*>(a.data_ptr());
	const Tensor* end = begin + a.numel();
	const SizeRef& s = begin->shape();
	const DType dt = begin->dtype;
	++begin;
	for(;begin != end; ++begin){
		utils::throw_exception(s == begin->shape(), "Expected all tensors in list of tensors to have the same shape, but a has a tensor of shape $ and $", s, begin->shape());
		utils::throw_exception(dt == begin->dtype, "Expected all tensors in list of tensors to have the same dtype, but a has a tensor of dtype $ and $", dt, begin->dtype);
	}
	begin = reinterpret_cast<const Tensor*>(b.data_ptr());
	end = begin + a.numel();
	const SizeRef &n_s = begin->shape();
	utils::throw_exception(dt == begin->dtype, "Expected all tensors in list of tensors to have the same dtype ($), but b has a tensor of dtype $", dt, begin->dtype);
	++begin;
	for(;begin != end; ++begin){
		utils::throw_exception(n_s == begin->shape(), "Expected all tensors in list of tensors to have the same shape, but a has b tensor of shape $ and $", n_s, begin->shape());
		utils::throw_exception(dt == begin->dtype, "Expected all tensors in list of tensors to have the same dtype ($), but a has b tensor of dtype $", dt, begin->dtype);
	}

	return dt;

}




//this is basically a way to automatically handle matrix multiplication using recusion instead of a bunch of if statements
//mainly used for easier readability
template<DType dt = DType::Integer>
Tensor handle_tensor_tensor_matrix_multiplication_std(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b, const DType& sub_dt){
	if(dt != sub_dt){return handle_tensor_tensor_matrix_multiplication_std<DTypeFuncs::next_dtype_it<dt> >(a, b, trans_a, trans_b, sub_dt);}
	if constexpr (!(dt == DType::TensorObj || dt == DType::Bool)){
		using type_t = DTypeFuncs::dtype_to_type_t<dt>;
		if(a.dims() > b.dims())
			return handle_dim_n_k_subT_matmult_std<type_t>(a,b,trans_a,trans_b);
		if(b.dims() > a.dims())
			return handle_dim_k_n_subT_matmult_std<type_t>(a,b,trans_a,trans_b);
		return handle_dim_n_n_subT_matmult_std<type_t>(a,b,trans_a,trans_b);
	}
	else{
		utils::THROW_EXCEPTION(a.dtype == DType::Bool, "Expected dtype to be numerical, but got dtype $", a.dtype);
		return Tensor();
	}	
}

template<DType dt = DType::Integer>
Tensor handle_matrix_multiplication_std(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){
	if(dt != a.dtype){return handle_matrix_multiplication_std<DTypeFuncs::next_dtype_it<dt> >(a, b, trans_a, trans_b);}
	if constexpr (!(dt == DType::TensorObj || dt == DType::Bool)){
		using type_t = DTypeFuncs::dtype_to_type_t<dt>;
		if(a.dims() == 2 && a.dims() == b.dims())
			return handle_dim2_matmult_std<type_t>(a, b, trans_a, trans_b);
		if(a.dims() > b.dims())
			return handle_dim_n_k_matmult_std<type_t>(a,b,trans_a,trans_b);
		if(b.dims() > a.dims())
			return handle_dim_k_n_matmult_std<type_t>(a,b,trans_a,trans_b);
		return handle_dim_n_n_matmult_std<type_t>(a,b,trans_a,trans_b);
	}
	else if (dt == DType::TensorObj){
		DType sub_dt = subT_check(a, b); //just a list of things to check if the errors
		return handle_tensor_tensor_matrix_multiplication_std(a, b, trans_a, trans_b, sub_dt);
		
	}
	else{
		utils::THROW_EXCEPTION(a.dtype == DType::Bool, "Expected dtype to be numerical, or tensors, but got dtype $", a.dtype);
		return Tensor();
	}


}




Tensor std_matmult(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){
	return handle_matrix_multiplication_std<DType::Integer>(a, b, trans_a, trans_b);
}


/* Tensor std_matmult(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){ */
/* 	utils::THROW_EXCEPTION(a.dims() > 1 && b.dims() > 1, "\nRuntimeError: Expected tensors to have dims greater than 1, but instead had dims of $ and $", a.dims(), b.dims()); */
/* 	/1* utils::THROW_EXCEPTION(a.shape()[-1] == b.shape()[-2] || a.shape()[-1] == b.shape()[-1], "\nRuntimeError: Expected second tensor rows or collumns to be $ when it was $ and $",a.shape()[-1],b.shape()[-2], b.shape()[-1]); *1/ */
/* 	bool transpose_a, transpose_b; */
/* 	determine_transposes(a, b, transpose_a, transpose_b, trans_a, trans_b); */
/* 	bool swapped_a = false; */
/* 	bool swapped_b = false; */
/* 	if(transpose_a){a.RowColSwap(); swapped_a = true;} */
/* 	if(!transpose_b && a.shape()[-1] != b.shape()[-1]){b.RowColSwap(); swapped_b = true;} */
/* 	Tensor output = matmult_cT(a, b); */
/* 	if(swapped_a){a.RowColSwap();} */
/* 	if(swapped_b){b.RowColSwap();} */
/* 	return std::move(output); */
/* } */

}}}
#endif
