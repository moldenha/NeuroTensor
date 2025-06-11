#ifndef _MATMULT_MKL_CPP_
#define _MATMULT_MKL_CPP_

//this is assuming an intel processor, and that mkl is the framework used and available for all functions

#include <mkl.h>
#include "old_matmult_std.cpp"


//Link against Intel MKL library
#pragma comment(lib, "mkl_intel_lp64.lib")
#pragma comment(lib, "mkl_sequential.lib")
#pragma comment(lib, "mkl_core.lib")

//these are the matrix multiplication functions that can be made and supported:


#include <iostream>
#include <complex>
#include <cstdint> // For integer types

namespace nt{
namespace functional{
namespace mkl_functions{


inline void cblas_gemm_s16s16s32_batch_64(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE *TransA, const CBLAS_TRANSPOSE *TransB,
                                    const MKL_INT64 *M, const MKL_INT64 *N, const MKL_INT64 *K,
                                    const int16_t *alpha, const int16_t **A, const MKL_INT64 *lda,
                                    const int16_t **B, const MKL_INT64 *ldb, const int32_t *beta,
                                    int32_t **C, const MKL_INT64 *ldc, const MKL_INT64 group_count, const MKL_INT64 *group_size) {
	
	for(MKL_INT64 i = 0; i < group_count; ++i){
		for(MKL_INT64 j = 0; j < group_size[i]; ++j){
			cblas_gemm_s16s16s32_64(Order, TransA[i], TransB[i], CblasFixOffset, M[i], N[i], K[i],
					(float)alpha[i], A[i] + j * M[i] * K[i], lda[i],
					0, B[i] + j * K[i] * N[i], ldb[i], 0, 
					(float)beta[i], C[i] + j * M[i] * N[i], ldc[i], nullptr);
		}
	}
}


inline void mm_s16s16s32_64(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                                    const MKL_INT64 M, const MKL_INT64 N, const MKL_INT64 K,
                                    const int16_t alpha, const int16_t *A, const MKL_INT64 lda,
                                    const int16_t *B, const MKL_INT64 ldb, const int32_t beta,
                                    int32_t *C, const MKL_INT64 ldc) {

	cblas_gemm_s16s16s32_64(Order, TransA, TransB, CblasFixOffset, M, N, K, float(alpha), A, lda, 0, B, ldb, 0, (float)beta, C, ldc, nullptr);
	
}

//this is just to convert alpha and beta into const void*
inline void mm_complex_float_64(CBLAS_LAYOUT&& layout,
    CBLAS_TRANSPOSE&& transa,
    CBLAS_TRANSPOSE&& transb,
    const MKL_INT64& m,
    const MKL_INT64& n,
    const MKL_INT64& k,
    const std::complex<float>& alpha,
    const void *a,
    const MKL_INT64& lda,
    const void* b,
    const MKL_INT64& ldb,
    const std::complex<float>& beta,
    void *c,
    const MKL_INT64& ldc){
	cblas_cgemm_64(layout, transa, transb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
	
}

//this is just to convert alpha and beta into const void*
inline void mm_complex_double_64(CBLAS_LAYOUT&& layout,
    CBLAS_TRANSPOSE&& transa,
    CBLAS_TRANSPOSE&& transb,
    const MKL_INT64& m,
    const MKL_INT64& n,
    const MKL_INT64& k,
    const std::complex<double>& alpha,
    const void *a,
    const MKL_INT64& lda,
    const void* b,
    const MKL_INT64& ldb,
    const std::complex<double>& beta,
    void *c,
    const MKL_INT64& ldc){

	cblas_zgemm_64(layout, transa, transb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
	
}


//this is just to convert a, b, and c into const void**, const void**, and void**
inline void mm_complex_float_batch_64(CBLAS_LAYOUT&& layout,
    CBLAS_TRANSPOSE* transa,
    CBLAS_TRANSPOSE* transb,
    const MKL_INT64* m,
    const MKL_INT64* n,
    const MKL_INT64* k,
    const std::complex<float>* alpha,
    const std::complex<float>** a,
    const MKL_INT64* lda,
    const std::complex<float>** b,
    const MKL_INT64* ldb,
    const std::complex<float>* beta,
    std::complex<float> **c,
    const MKL_INT64* ldc,
    const MKL_INT64& group_count,
    const MKL_INT64* group_size){
	cblas_cgemm_batch_64(layout, transa, transb, 
			m, n, k, 
			reinterpret_cast<const void*>(alpha), reinterpret_cast<const void**>(a), 
			lda, reinterpret_cast<const void**>(b), ldb, reinterpret_cast<const void*>(beta), reinterpret_cast<void**>(c), ldc,
			group_count, group_size);
	
}

//this is just to convert a, b, and c into const void**, const void**, and void**
inline void mm_complex_double_batch_64(CBLAS_LAYOUT&& layout,
    CBLAS_TRANSPOSE* transa,
    CBLAS_TRANSPOSE* transb,
    const MKL_INT64* m,
    const MKL_INT64* n,
    const MKL_INT64* k,
    const std::complex<double>* alpha,
    const std::complex<double>** a,
    const MKL_INT64* lda,
    const std::complex<double>** b,
    const MKL_INT64* ldb,
    const std::complex<double>* beta,
    std::complex<double> **c,
    const MKL_INT64* ldc,
    const MKL_INT64& group_count,
    const MKL_INT64* group_size){
	cblas_zgemm_batch_64(layout, transa, transb, 
			m, n, k, 
			reinterpret_cast<const void*>(alpha), reinterpret_cast<const void**>(a), 
			lda, reinterpret_cast<const void**>(b), ldb, reinterpret_cast<const void*>(beta), reinterpret_cast<void**>(c), ldc,
			group_count, group_size);
	
}


// Primary template for MKL_MATMUL (will not be used directly)
template<typename T>
struct MKL_MATMUL;

// Template specialization for float (single precision)
template<>
struct MKL_MATMUL<float> {
    static constexpr auto function = cblas_sgemm_64;
};

// Template specialization for double (double precision)
template<>
struct MKL_MATMUL<double> {
    static constexpr auto function = cblas_dgemm_64;
};

// Template specialization for std::complex<float>
template<>
struct MKL_MATMUL<std::complex<float>> {
    static constexpr auto function = mm_complex_float_64;
};

// Template specialization for std::complex<double>
template<>
struct MKL_MATMUL<std::complex<double>> {
    static constexpr auto function = mm_complex_double_64;
};


// Template specialization for int16_t
template<>
struct MKL_MATMUL<int16_t> {
    static constexpr auto function = mm_s16s16s32_64;
};




template<typename T>
struct MKL_MATMUL_BATCH;

template<>
struct MKL_MATMUL_BATCH<float>{
	static constexpr auto function = cblas_sgemm_batch_64;
};


template<>
struct MKL_MATMUL_BATCH<double>{
	static constexpr auto function = cblas_dgemm_batch_64;
};


template<>
struct MKL_MATMUL_BATCH<std::complex<float>>{
	static constexpr auto function = mm_complex_float_batch_64; 
};

template<>
struct MKL_MATMUL_BATCH<std::complex<double>>{
	static constexpr auto function = mm_complex_double_batch_64;
};


template<>
struct MKL_MATMUL_BATCH<int16_t>{
	static constexpr auto function = cblas_gemm_s16s16s32_batch_64;
};


template<typename T>
struct IN_DTYPE;

template<>
struct IN_DTYPE<float>{
	static constexpr DType dtype = DType::Float32;
};

template<>
struct IN_DTYPE<double>{
	static constexpr DType dtype = DType::Float64;
};

template<>
struct IN_DTYPE<std::complex<float>>{
	static constexpr DType dtype = DType::Complex64;
};

template<>
struct IN_DTYPE<std::complex<double>>{
	static constexpr DType dtype = DType::Complex128;
};

template<>
struct IN_DTYPE<int16_t>{
	static constexpr DType dtype = DType::int16;
};

template<typename T>
struct OUT_DTYPE;

template<>
struct OUT_DTYPE<float>{
	static constexpr DType dtype = DType::Float32;
};

template<>
struct OUT_DTYPE<double>{
	static constexpr DType dtype = DType::Float64;
};

template<>
struct OUT_DTYPE<std::complex<float>>{
	static constexpr DType dtype = DType::Complex64;
};

template<>
struct OUT_DTYPE<std::complex<double>>{
	static constexpr DType dtype = DType::Complex128;
};

template<>
struct OUT_DTYPE<int16_t>{
	static constexpr DType dtype = DType::int32;
};



template<typename T>
struct OUT_TYPE;

template<>
struct OUT_TYPE<float>{
	using value_type = float;
};

template<>
struct OUT_TYPE<double>{
	using value_type = double;
};

template<>
struct OUT_TYPE<std::complex<float>>{
	using value_type = std::complex<float>;
};

template<>
struct OUT_TYPE<std::complex<double>>{
	using value_type = std::complex<double>;
};

template<>
struct OUT_TYPE<int16_t>{
	using value_type = int32_t;
};

//converts a pointer to a tensor
	
//%s/\tSizeRef a_shape = (transpose_a) ? o_a_shape.transpose(-1,-2) : o_a_shape;\n\tSizeRef b_shape = (transpose_b) ? o_b_shape.transpose(-1,-2) : o_b_shape;/\/\/\tSizeRef a_shape = (transpose_a) ? o_a_shape.transpose(-1,-2) : o_a_shape;\n\/\/\tSizeRef b_shape = (transpose_b) ? o_b_shape.transpose(-1,-2) : o_b_shape\n\tSizeRef a_shape = o_a_shape;\n\tSizeRef b_shape = o_b_shape;

template<typename T,
	typename outT = typename OUT_TYPE<T>::value_type,
	DType DT = IN_DTYPE<T>::dtype,
	DType outDT = OUT_DTYPE<T>::dtype>
Tensor matmult_mkl_2(const T* A, const T* B, const SizeRef& o_a_shape, const SizeRef& o_b_shape, const bool& transpose_a, const bool& transpose_b){
	SizeRef a_shape = (transpose_a) ? o_a_shape.transpose(-1,-2) : o_a_shape;
	SizeRef b_shape = (transpose_b) ? o_b_shape.transpose(-1,-2) : o_b_shape;
	/* SizeRef a_shape = o_a_shape; */
	/* SizeRef b_shape = o_b_shape; */
	utils::THROW_EXCEPTION(a_shape[-1] == b_shape[-2], "Expected at positions (-1,-2) for shapes $,$ -> ($:$) to be equal", a_shape, b_shape, a_shape[-1], b_shape[-2]);
	std::vector<typename Tensor::size_value_t> vec = a_shape.Vec();
	vec.back() = b_shape.back();
	SizeRef c_shape(std::move(vec));

	outT start_num_a;
	T start_num_b;
	if constexpr (std::is_same_v<outT, std::complex<float>>){
		start_num_a = std::complex<float>(1.0f, 1.0f);
		start_num_b = std::complex<float>(0.0f, 0.0f);
	}
	else if constexpr (std::is_same_v<outT, std::complex<double>>){
		start_num_a = std::complex<double>(1.0, 1.0);
		start_num_b = std::complex<double>(0.0, 0.0);
	}
	else{
		start_num_a = outT(1);
		start_num_b = outT(0);
	}

	Tensor tensor_C = zeros(std::move(c_shape), outDT);

	outT* C = reinterpret_cast<outT*>(tensor_C.data_ptr());
	const MKL_INT64 m = o_a_shape[-2]; // Number of rows in A
	const MKL_INT64 n = o_b_shape[-1]; // Number of columns in B
	const MKL_INT64 k = (transpose_b) ? b_shape[-1] : o_a_shape[-1]; // Number of columns in A and rows in B
	// Perform matrix multiplication: C = A * B
	if(transpose_b && !transpose_a){
		MKL_MATMUL<T>::function(CblasRowMajor, CblasNoTrans, CblasTrans, m, k, n, start_num_a, 
			      A, n, 
			      B, n, 
			      start_num_b, C, k);
	}
	else if(transpose_a && !transpose_b){
		MKL_MATMUL<T>::function(CblasRowMajor, CblasTrans, CblasNoTrans, k, n, m, start_num_a, 
			      A, k, 
			      B, n, 
			      start_num_b, C, n);
		
	}
	else if(transpose_a && transpose_b){
		MKL_MATMUL<T>::function(CblasRowMajor, CblasTrans, CblasTrans, a_shape[-2], b_shape[-1], a_shape[-1], start_num_a, 
			      A, a_shape[-2], 
			      B, b_shape[-2], 
			      start_num_b, C, b_shape[-1]);
	
	}
	else{
		MKL_MATMUL<T>::function(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, start_num_a, 
			      A, k, 
			      B, n, 
			      start_num_b, C, n);

	
	}	

	/* MKL_MATMUL<T>::function(CblasRowMajor, transpose_a ? CblasTrans : CblasNoTrans, transpose_b ? CblasTrans : CblasNoTrans, m, n, k, 1.0f, A, k, B, n, 0.0f, C, n); */

	return std::move(tensor_C);

}





template<typename T,
	typename outT = typename OUT_TYPE<T>::value_type,
	DType DT = IN_DTYPE<T>::dtype,
	DType outDT = OUT_DTYPE<T>::dtype>
Tensor handle_dim2_matmult_mkl(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){
	uint32_t iterator_type_a = a.arr_void().get_bucket().iterator_type(); // 3 = strided_view, 2 = bucketed, 1 = is_contiguous
	uint32_t iterator_type_b = b.arr_void().get_bucket().iterator_type(); // 
	
	bool transpose_a, transpose_b;
	std_functional::determine_transposes(a, b, transpose_a, transpose_b, trans_a, trans_b);

	if(iterator_type_a == 1){
		const T* A = a.arr_void().get_bucket().cbegin<1, T>();
		if(iterator_type_b == 1){
			const T* B = b.arr_void().get_bucket().cbegin<1, T>();
			return matmult_mkl_2<T, outT, DT, outDT>(A, B, a.shape(), b.shape(), transpose_a, transpose_b);
		}

		if(iterator_type_b == 3 && std_functional::was_transposed_back(b)){
			Tensor contiguous_b = std_functional::make_contiguous_look(b, std_functional::handle_looking_contiguous(b)); //there is no possibility for it to be an array of tensors since it is only dim() == 2
			transpose_b = !transpose_b;
			const T* B = contiguous_b.arr_void().get_bucket().cbegin<1, T>();
			return matmult_mkl_2<T, outT, DT, outDT>(A, B, a.shape(), contiguous_b.shape(), transpose_a, transpose_b);

		}
		//this is if it is bucketed or strided
		//normally if it were bucketed, it would see if there is any possible way to seperate it into matricies, otherwise just this:
		intrusive_ptr<T[]> intrusive_B = (iterator_type_b == 3) ? std_functional::convert_to_intrusive_strided<T>(b) : std_functional::convert_to_intrusive_bucketed<T>(b);
		const T* B = intrusive_B.get();
		return matmult_mkl_2<T, outT, DT, outDT>(A, B, a.shape(), b.shape(), transpose_a, transpose_b);

	}
	if(iterator_type_a == 2){ //bucketed
		intrusive_ptr<T[]> intrusive_A = std_functional::convert_to_intrusive_bucketed<T>(a);
		const T* A = intrusive_A.get();
		if(iterator_type_b == 1){
			const T* B = b.arr_void().get_bucket().cbegin<1, T>();
			return matmult_mkl_2<T, outT, DT, outDT>(A, B, a.shape(), b.shape(), transpose_a, transpose_b);
		}

		if(iterator_type_b == 3 && std_functional::was_transposed_back(b)){
			Tensor contiguous_b = std_functional::make_contiguous_look(b, std_functional::handle_looking_contiguous(b)); //there is no possibility for it to be an array of tensors since it is only dim() == 2
			transpose_b = !transpose_b;
			const T* B = contiguous_b.arr_void().get_bucket().cbegin<1, T>();
			return matmult_mkl_2<T, outT, DT, outDT>(A, B, a.shape(), contiguous_b.shape(), transpose_a, transpose_b);

		}
		//this is if it is bucketed or strided
		//normally if it were bucketed, it would see if there is any possible way to seperate it into matricies, otherwise just this:
		intrusive_ptr<T[]> intrusive_B = (iterator_type_b == 3) ? std_functional::convert_to_intrusive_strided<T>(b) : std_functional::convert_to_intrusive_bucketed<T>(b);
		const T* B = intrusive_B.get();
		return matmult_mkl_2<T, outT, DT, outDT>(A, B, a.shape(), b.shape(), transpose_a, transpose_b);
		
	}
	//iterator_type_a == 3
	if(std_functional::was_transposed_back(a)){
		Tensor contiguous_a = std_functional::make_contiguous_look(a, std_functional::handle_looking_contiguous(a)); //there is no possibility for it to be an array of tensors since it is only dim() == 2
		transpose_a = !transpose_a;
		const T* A = contiguous_a.arr_void().get_bucket().cbegin<1, T>();
		if(iterator_type_b == 1){
			const T* B = b.arr_void().get_bucket().cbegin<1, T>();
			return matmult_mkl_2<T, outT, DT, outDT>(A, B, a.shape(), b.shape(), transpose_a, transpose_b);
		}

		if(iterator_type_b == 3 && std_functional::was_transposed_back(b)){
			Tensor contiguous_b = std_functional::make_contiguous_look(b, std_functional::handle_looking_contiguous(b)); //there is no possibility for it to be an array of tensors since it is only dim() == 2
			transpose_b = !transpose_b;
			const T* B = contiguous_b.arr_void().get_bucket().cbegin<1, T>();
			return matmult_mkl_2<T, outT, DT, outDT>(A, B, a.shape(), contiguous_b.shape(), transpose_a, transpose_b);

		}
		//this is if it is bucketed or strided
		//normally if it were bucketed, it would see if there is any possible way to seperate it into matricies, otherwise just this:
		intrusive_ptr<T[]> intrusive_B = (iterator_type_b == 3) ? std_functional::convert_to_intrusive_strided<T>(b) : std_functional::convert_to_intrusive_bucketed<T>(b);
		const T* B = intrusive_B.get();
		return matmult_mkl_2<T, outT, DT, outDT>(A, B, contiguous_a.shape(), b.shape(), transpose_a, transpose_b);
	}
	intrusive_ptr<T[]> intrusive_A = std_functional::convert_to_intrusive_strided<T>(a);
	const T* A = intrusive_A.get();
	if(iterator_type_b == 1){
		const T* B = b.arr_void().get_bucket().cbegin<1, T>();
		return matmult_mkl_2<T, outT, DT, outDT>(A, B, a.shape(), b.shape(), transpose_a, transpose_b);
	}

	if(iterator_type_b == 3 && std_functional::was_transposed_back(b)){
		Tensor contiguous_b = std_functional::make_contiguous_look(b, std_functional::handle_looking_contiguous(b)); //there is no possibility for it to be an array of tensors since it is only dim() == 2
		transpose_b = !transpose_b;
		const T* B = contiguous_b.arr_void().get_bucket().cbegin<1, T>();
		return matmult_mkl_2<T, outT, DT, outDT>(A, B, a.shape(), contiguous_b.shape(), transpose_a, transpose_b);

	}
	//this is if it is bucketed or strided
	//normally if it were bucketed, it would see if there is any possible way to seperate it into matricies, otherwise just this:
	intrusive_ptr<T[]> intrusive_B = (iterator_type_b == 3) ? std_functional::convert_to_intrusive_strided<T>(b) : std_functional::convert_to_intrusive_bucketed<T>(b);
	const T* B = intrusive_B.get();
	return matmult_mkl_2<T, outT, DT, outDT>(A, B, a.shape(), b.shape(), transpose_a, transpose_b);
}


void calculate_optimal_group_mkl(MKL_INT64 M, MKL_INT64 K, MKL_INT64 N, int64_t batch_size, MKL_INT64& group_count, MKL_INT64*& group_sizes) {
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
	typename outT = typename OUT_TYPE<T>::value_type,
	DType DT = IN_DTYPE<T>::dtype,
	DType outDT = OUT_DTYPE<T>::dtype>
Tensor handle_tensors_of_tensors(Tensor& a, Tensor& b, const bool transpose_a, const bool transpose_b){
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
	//these would be what it would be under normal conditions
	
	//for when b is transposed (only), it would be M=M, N=K, and K=N
	//so, M = a_shape[-2], N = b_shape[-1], K = b_shape[-2]
	//
	//for when a is transposed (only), it would be M=K, N=N, K=M
	//so, M = a_shape[-2], N = b_shape[-1], K = a_shape[-1] == b_shape[-2]
	//
	//when both are transposed it would be that M=K, N=M, K=N
	//so, M = a_shape[-2], N = a_shape[-1], K = b_shape[-2]
	//
	
	const int64_t& M = begin_shape[-2];
	const int64_t& N = begin_b_shape[-1];
	const int64_t& K = (transpose_b) ? b_shape[-1] : begin_shape[-1];
	
	/* const int64_t& M = a_shape[-2]; // Number of rows in A */
	/* const int64_t& N = (transpose_a && transpose_b) ? a_shape[-1] : b_shape[-1]; // Number of columns in B */
	/* const int64_t& K = b_shape[-2]; // Number of columns in A and rows in B */
	/* const int64_t& K = (transpose_a) ? a_shape[-2] : (transpose_b) ? b_shape[-1] : begin_shape[-1]; // Number of columns in A and rows in B */
	/* const int64_t& K = begin_shape[-1]; // Number of columns in A and rows in B */
	

	//normally LDA = K, LDB = N, LDC = N
	//but with transposed b (only)
	//it comes out to be LDA = K(N), LDB = N(K), LDC = N(K)
	//
	//with transposed a (only)
	//LDA = N(N), LDB = N(N), LDB = K(M)
	//
	//with both transposed
	//k, m, m
	//which comes out to
	//
	
	


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
	
	MKL_INT64 group_count;
	MKL_INT64* group_size;

	calculate_optimal_group_mkl(M, K, N, batch_size, group_count, group_size);
	
	CBLAS_TRANSPOSE transA[group_count];
	CBLAS_TRANSPOSE transB[group_count];
	MKL_INT64 m[group_count], n[group_count], k[group_count];
	MKL_INT64 lda[group_count], ldb[group_count], ldc[group_count];
	T alpha[group_count]; 
	outT beta[group_count];

	std::fill(transA, transA + group_count, transpose_a ? CblasTrans : CblasNoTrans);
	std::fill(transB, transB + group_count, transpose_b ? CblasTrans : CblasNoTrans);

	if (transpose_a && transpose_b) {
		std::fill(lda, lda + group_count, a_shape[-2]);
		std::fill(ldb, ldb + group_count, b_shape[-2]);
		std::fill(ldc, ldc + group_count, b_shape[-1]);

		
		std::fill(m, m + group_count, a_shape[-2]);
		std::fill(n, n + group_count, b_shape[-1]);
		std::fill(k, k + group_count, a_shape[-1]);

	} else if (transpose_a) {
		std::fill(lda, lda + group_count, K);
		std::fill(ldb, ldb + group_count, N);
		std::fill(ldc, ldc + group_count, N);
		
		std::fill(m, m + group_count, M);
		std::fill(n, n + group_count, N);
		std::fill(k, k + group_count, K);

	} else if (transpose_b) {
		std::fill(lda, lda + group_count, K);
		std::fill(ldb, ldb + group_count, K);
		std::fill(ldc, ldc + group_count, M);

		std::fill(m, m + group_count, M);
		std::fill(n, n + group_count, N);
		std::fill(k, k + group_count, K);

	} else {
		std::fill(lda, lda + group_count, K);
		std::fill(ldb, ldb + group_count, N);
		std::fill(ldc, ldc + group_count, N);

		std::fill(m, m + group_count, M);
		std::fill(n, n + group_count, N);
		std::fill(k, k + group_count, K);
	}


	if constexpr (std::is_same_v<outT, std::complex<float>>){
		std::fill(beta, beta + group_count, std::complex<float>(0.0, 0.0));
	}
	else if constexpr (std::is_same_v<outT, std::complex<double>>){
		std::fill(beta, beta + group_count, std::complex<double>(0.0, 0.0));
	}
	else{
		std::fill(beta, beta + group_count, static_cast<outT>(0.0));
	}

	if constexpr(std::is_same_v<T, std::complex<float>>){
		std::fill(alpha, alpha + group_count, std::complex<float>(1.0, 1.0));
	}
	else if constexpr (std::is_same_v<T, std::complex<double>>){
		std::fill(alpha, alpha + group_count, std::complex<double>(1.0, 1.0));
	}else{
		std::fill(alpha, alpha + group_count, static_cast<T>(1.0));
	}
	
	/* std::cout << "running functional..."<<std::endl; */
	if(transpose_b && !transpose_a){
		MKL_MATMUL_BATCH<T>::function(CblasRowMajor, transA, transB, m, k, n, alpha, 
			      A_array, n, 
			      B_array, n, 
			      beta, C_array, k, group_count, group_size);
	}
	else if(transpose_a && !transpose_b){
		MKL_MATMUL_BATCH<T>::function(CblasRowMajor, transA, transB, k, n, m, alpha, 
			      A_array, k, 
			      B_array, n, 
			      beta, C_array, n, group_count, group_size);
		
	}
	else if(transpose_a && transpose_b){
		MKL_MATMUL_BATCH<T>::function(CblasRowMajor, transA, transB, m, n, k, alpha, 
			      A_array, lda, 
			      B_array, ldb, 
			      beta, C_array, ldc, group_count, group_size);
	
	}
	else{
		MKL_MATMUL_BATCH<T>::function(CblasRowMajor, transA, transB, m, n, k, alpha, 
			      A_array, lda, 
			      B_array, ldb, 
			      beta, C_array, ldc, group_count, group_size);

	
	}
	/* std::cout << "returned functional"<<std::endl; */
	
	delete[] group_size;
	/* std::cout << "deleted group_size"<<std::endl; */
	return std::move(tensor_C);

	
}



template<typename T,
	typename outT = typename OUT_TYPE<T>::value_type,
	DType DT = IN_DTYPE<T>::dtype,
	DType outDT = OUT_DTYPE<T>::dtype>
Tensor handle_tensors_of_tensors_subT(Tensor& a, Tensor& b, const bool transpose_a, const bool transpose_b){
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
	//these would be what it would be under normal conditions
	
	//for when b is transposed (only), it would be M=M, N=K, and K=N
	//so, M = a_shape[-2], N = b_shape[-1], K = b_shape[-2]
	//
	//for when a is transposed (only), it would be M=K, N=N, K=M
	//so, M = a_shape[-2], N = b_shape[-1], K = a_shape[-1] == b_shape[-2]
	//
	//when both are transposed it would be that M=K, N=M, K=N
	//so, M = a_shape[-2], N = a_shape[-1], K = b_shape[-2]
	//
	
	const int64_t& M = begin_shape[-2];
	const int64_t& N = begin_b_shape[-1];
	const int64_t& K = (transpose_b) ? b_shape[-1] : begin_shape[-1];
	
	/* const int64_t& M = a_shape[-2]; // Number of rows in A */
	/* const int64_t& N = (transpose_a && transpose_b) ? a_shape[-1] : b_shape[-1]; // Number of columns in B */
	/* const int64_t& K = b_shape[-2]; // Number of columns in A and rows in B */
	/* const int64_t& K = (transpose_a) ? a_shape[-2] : (transpose_b) ? b_shape[-1] : begin_shape[-1]; // Number of columns in A and rows in B */
	/* const int64_t& K = begin_shape[-1]; // Number of columns in A and rows in B */
	

	//normally LDA = K, LDB = N, LDC = N
	//but with transposed b (only)
	//it comes out to be LDA = K(N), LDB = N(K), LDC = N(K)
	//
	//with transposed a (only)
	//LDA = N(N), LDB = N(N), LDB = K(M)
	//
	//with both transposed
	//k, m, m
	//which comes out to
	//
	
	


	std::vector<typename Tensor::size_value_t> vec = {batch_size, a_shape[-2], b_shape[-1]};
	SizeRef c_shape(std::move(vec));
	Tensor tensor_C = zeros(std::move(c_shape), outDT); 
	outT* C = reinterpret_cast<outT*>(tensor_C.data_ptr());
	const int64_t c_matrix_size = tensor_C.shape()[-1] * tensor_C.shape()[-2];
	
	int64_t counter = 0;
	for(;begin != end; ++begin, ++begin_b){
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
			
			/* std::cout << *sub_begin << std::endl; */
			const T* a_ptr = sub_begin->arr_void().get_bucket().begin<1, T>();
			A_array[counter] = a_ptr;
			B_array[counter] = sub_begin_b->arr_void().get_bucket().begin<1, T>();
			C_array[counter] = C + (counter * c_matrix_size);	
		}
	}
	
	//the rest is handled the same as the original non-subt one
	MKL_INT64 group_count;
	MKL_INT64* group_size;

	calculate_optimal_group_mkl(M, K, N, batch_size, group_count, group_size);
	
	CBLAS_TRANSPOSE transA[group_count];
	CBLAS_TRANSPOSE transB[group_count];
	MKL_INT64 m[group_count], n[group_count], k[group_count];
	MKL_INT64 lda[group_count], ldb[group_count], ldc[group_count];
	T alpha[group_count]; 
	outT beta[group_count];

	std::fill(transA, transA + group_count, transpose_a ? CblasTrans : CblasNoTrans);
	std::fill(transB, transB + group_count, transpose_b ? CblasTrans : CblasNoTrans);

	if (transpose_a && transpose_b) {
		std::fill(lda, lda + group_count, a_shape[-2]);
		std::fill(ldb, ldb + group_count, b_shape[-2]);
		std::fill(ldc, ldc + group_count, b_shape[-1]);

		
		std::fill(m, m + group_count, a_shape[-2]);
		std::fill(n, n + group_count, b_shape[-1]);
		std::fill(k, k + group_count, a_shape[-1]);

	} else if (transpose_a) {
		std::fill(lda, lda + group_count, K);
		std::fill(ldb, ldb + group_count, N);
		std::fill(ldc, ldc + group_count, N);
		
		std::fill(m, m + group_count, M);
		std::fill(n, n + group_count, N);
		std::fill(k, k + group_count, K);

	} else if (transpose_b) {
		std::fill(lda, lda + group_count, K);
		std::fill(ldb, ldb + group_count, K);
		std::fill(ldc, ldc + group_count, M);

		std::fill(m, m + group_count, M);
		std::fill(n, n + group_count, N);
		std::fill(k, k + group_count, K);

	} else {
		std::fill(lda, lda + group_count, K);
		std::fill(ldb, ldb + group_count, N);
		std::fill(ldc, ldc + group_count, N);

		std::fill(m, m + group_count, M);
		std::fill(n, n + group_count, N);
		std::fill(k, k + group_count, K);
	}


	if constexpr (std::is_same_v<outT, std::complex<float>>){
		std::fill(beta, beta + group_count, std::complex<float>(0.0, 0.0));
	}
	else if constexpr (std::is_same_v<outT, std::complex<double>>){
		std::fill(beta, beta + group_count, std::complex<double>(0.0, 0.0));
	}
	else{
		std::fill(beta, beta + group_count, static_cast<outT>(0.0));
	}

	if constexpr(std::is_same_v<T, std::complex<float>>){
		std::fill(alpha, alpha + group_count, std::complex<float>(1.0, 1.0));
	}
	else if constexpr (std::is_same_v<T, std::complex<double>>){
		std::fill(alpha, alpha + group_count, std::complex<double>(1.0, 1.0));
	}else{
		std::fill(alpha, alpha + group_count, static_cast<T>(1.0));
	}
	
	/* std::cout << "running functional..."<<std::endl; */
	if(transpose_b && !transpose_a){
		MKL_MATMUL_BATCH<T>::function(CblasRowMajor, transA, transB, m, k, n, alpha, 
			      A_array, n, 
			      B_array, n, 
			      beta, C_array, k, group_count, group_size);
	}
	else if(transpose_a && !transpose_b){
		MKL_MATMUL_BATCH<T>::function(CblasRowMajor, transA, transB, k, n, m, alpha, 
			      A_array, k, 
			      B_array, n, 
			      beta, C_array, n, group_count, group_size);
		
	}
	else if(transpose_a && transpose_b){
		MKL_MATMUL_BATCH<T>::function(CblasRowMajor, transA, transB, m, n, k, alpha, 
			      A_array, lda, 
			      B_array, ldb, 
			      beta, C_array, ldc, group_count, group_size);
	
	}
	else{
		MKL_MATMUL_BATCH<T>::function(CblasRowMajor, transA, transB, m, n, k, alpha, 
			      A_array, lda, 
			      B_array, ldb, 
			      beta, C_array, ldc, group_count, group_size);

	
	}
	/* std::cout << "returned functional"<<std::endl; */
	
	delete[] group_size;
	/* std::cout << "deleted group_size"<<std::endl; */
	return std::move(tensor_C);

	
}


//this is going to use multi-processing to perform matrix multiplication across multiple batches with sub-batches
//for example (3,4,5,5) x (4,5,5)
//a in this case has 4 dims and b has 3




template<typename T>
Tensor handle_dim_n_k_matmul_mkl(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){
	constexpr DType in_dtype = DTypeFuncs::type_to_dtype<T>;
	utils::THROW_EXCEPTION(a.dims() > b.dims(), "Expected to have larger tensor in terms of dims as a but got $ and $", a.dims(), b.dims());
	utils::THROW_EXCEPTION(a.dtype == in_dtype && b.dtype == in_dtype, "Expected both tensors to have the dtype $ but got $ and $", in_dtype, a.dtype, b.dtype);

	int64_t start = a.dims()-b.dims();
	for(int64_t i = start; i < a.dims() - 2; ++i){
		utils::THROW_EXCEPTION(a.shape()[i] == b.shape()[i-start], "Cannot multiply tensors of shapes $ and $, outter dimensions do not match", a.shape(), b.shape());
	}
	
	bool transpose_a, transpose_b;
	std_functional::determine_transposes(a, b, transpose_a, transpose_b, trans_a, trans_b);
	
	auto bigger_shape_vec = a.shape().arr();
	

	Tensor bigger = std_functional::return_split_tensor(a, transpose_a);
	/* std::cout << "got bigger"<<std::endl; */
	Tensor smaller = std_functional::return_split_tensor(b, transpose_b);
	/* std::cout << "got smaller"<<std::endl; */
	utils::THROW_EXCEPTION(bigger.numel() % smaller.numel() == 0, "Expected there to be a fraction of arrays in the tensors but got $ and  $ as the amount of matricies", bigger.numel(), smaller.numel());
	smaller = smaller.repeat_(bigger.numel() / smaller.numel());
	Tensor output = handle_tensors_of_tensors<T>(bigger, smaller, transpose_a, transpose_b);
	bigger_shape_vec[-1] = output.shape()[-1];
	bigger_shape_vec[-2] = output.shape()[-2];
	return output.view(SizeRef(std::move(bigger_shape_vec)));
}


template<typename T>
Tensor handle_dim_n_k_subT_matmult_mkl(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){
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
	std_functional::determine_transposes(temp_a, temp_b, transpose_a, transpose_b, trans_a, trans_b);
	std::vector<bool> n_transpose_a(a.numel(), transpose_a);
	std::vector<bool> n_transpose_b(b.numel(), transpose_b);
	Tensor bigger = std_functional::return_split_tensor_of_tensors(a, n_transpose_a);
	Tensor smaller = std_functional::return_split_tensor_of_tensors(b, n_transpose_b);
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
	Tensor output = handle_tensors_of_tensors_subT<T>(bigger, smaller, transpose_a, transpose_b);
	bigger_shape_vec[-1] = output.shape()[-1];
	bigger_shape_vec[-2] = output.shape()[-2];
	return output.view(SizeRef(std::move(bigger_shape_vec)));
	
}


template<typename T>
Tensor handle_dim_k_n_matmul_mkl(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){
	constexpr DType in_dtype = DTypeFuncs::type_to_dtype<T>;
	utils::THROW_EXCEPTION(b.dims() > a.dims(), "Expected to have larger tensor in terms of dims as a but got $ > $", b.dims(), a.dims());
	utils::THROW_EXCEPTION(a.dtype == in_dtype && b.dtype == in_dtype, "Expected both tensors to have the dtype $ but got $ and $", in_dtype, a.dtype, b.dtype);
	int64_t start = b.dims()-a.dims();
	for(int64_t i = start; i < b.dims() - 2; ++i){
		utils::THROW_EXCEPTION(a.shape()[i-start] == b.shape()[i], "Cannot multiply tensors of shapes $ and $, outter dimensions do not match", a.shape(), b.shape());
	}
	
	bool transpose_a, transpose_b;
	std_functional::determine_transposes(a, b, transpose_a, transpose_b, trans_a, trans_b);
	
	auto bigger_shape_vec = b.shape().arr();

	Tensor bigger = std_functional::return_split_tensor(b, transpose_b);
	/* std::cout << "got bigger"<<std::endl; */
	Tensor smaller = std_functional::return_split_tensor(a, transpose_a);
	/* std::cout << "got smaller"<<std::endl; */
	utils::THROW_EXCEPTION(bigger.numel() % smaller.numel() == 0, "Expected there to be a fraction of arrays in the tensors but got $ and  $ as the amount of matricies", bigger.numel(), smaller.numel());
	smaller = smaller.repeat_(bigger.numel() / smaller.numel());
	Tensor output = handle_tensors_of_tensors<T>(smaller, bigger, transpose_a, transpose_b);
	bigger_shape_vec[-1] = output.shape()[-1];
	bigger_shape_vec[-2] = output.shape()[-2];
	return output.view(SizeRef(std::move(bigger_shape_vec)));
}


template<typename T>
Tensor handle_dim_k_n_subT_matmult_mkl(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){
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
	std_functional::determine_transposes(temp_a, temp_b, transpose_a, transpose_b, trans_a, trans_b);
	std::vector<bool> n_transpose_a(a.numel(), transpose_a);
	std::vector<bool> n_transpose_b(b.numel(), transpose_b);
	Tensor smaller = std_functional::return_split_tensor_of_tensors(a, n_transpose_a);
	Tensor bigger = std_functional::return_split_tensor_of_tensors(b, n_transpose_b);
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
	Tensor output = handle_tensors_of_tensors_subT<T>(smaller, bigger, transpose_a, transpose_b);
	bigger_shape_vec[-1] = output.shape()[-1];
	bigger_shape_vec[-2] = output.shape()[-2];
	return output.view(SizeRef(std::move(bigger_shape_vec)));
}

template<typename T>
Tensor handle_dim_n_n_matmul_mkl(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){
	constexpr DType in_dtype = DTypeFuncs::type_to_dtype<T>;
	utils::THROW_EXCEPTION(b.dims() == a.dims(), "Expected to have same tensor dims as a but got $ != $", b.dims(), a.dims());
	utils::THROW_EXCEPTION(a.dtype == in_dtype && b.dtype == in_dtype, "Expected both tensors to have the dtype $ but got $ and $", in_dtype, a.dtype, b.dtype);

	int64_t start = b.dims()-a.dims();
	for(int64_t i = start; i < b.dims() - 2; ++i){
		utils::THROW_EXCEPTION(a.shape()[i-start] == b.shape()[i], "Cannot multiply tensors of shapes $ and $, outter dimensions do not match", a.shape(), b.shape());
	}
	bool transpose_a, transpose_b;
	auto bigger_shape_vec = b.shape().arr();
	std_functional::determine_transposes(a, b, transpose_a, transpose_b, trans_a, trans_b);
	Tensor A = std_functional::return_split_tensor(a, transpose_a);
	Tensor B = std_functional::return_split_tensor(b, transpose_b);
	utils::THROW_EXCEPTION(A.numel() == B.numel(), "Expected there to be the same number of matricies but got $ and  $ as the amount of matricies", A.numel(), B.numel());
	Tensor output = handle_tensors_of_tensors<T>(A, B, transpose_a, transpose_b);
	bigger_shape_vec[-1] = output.shape()[-1];
	bigger_shape_vec[-2] = output.shape()[-2];
	return output.view(SizeRef(std::move(bigger_shape_vec)));
	
}


template<typename T>
Tensor handle_dim_n_n_subT_matmult_mkl(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){
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
	bigger_shape_vec[0] *= b.numel();
	bool transpose_a, transpose_b;
	std_functional::determine_transposes(temp_a, temp_b, transpose_a, transpose_b, trans_a, trans_b);
	std::vector<bool> n_transpose_a(a.numel(), transpose_a);
	std::vector<bool> n_transpose_b(b.numel(), transpose_b);
	Tensor A = std_functional::return_split_tensor_of_tensors(a, n_transpose_a);
	Tensor B = std_functional::return_split_tensor_of_tensors(b, n_transpose_b);
	const bool transpose_a_get = n_transpose_a[0];
	const bool transpose_b_get = n_transpose_b[0];
	std::for_each(n_transpose_a.cbegin(), n_transpose_a.cend(), [&transpose_a_get](const bool& t){utils::throw_exception(transpose_a_get == t, "Memory error with transposes a");});
	std::for_each(n_transpose_b.cbegin(), n_transpose_b.cend(), [&transpose_b_get](const bool& t){utils::throw_exception(transpose_b_get == t, "Memory error with transposes b");});

	utils::THROW_EXCEPTION(A.numel() == B.numel(), "Expected there to be the same number of matricies but got $ and  $ as the amount of matricies", A.numel(), B.numel());
	Tensor output = handle_tensors_of_tensors_subT<T>(A, B, transpose_a, transpose_b);
	bigger_shape_vec[-1] = output.shape()[-1];
	bigger_shape_vec[-2] = output.shape()[-2];
	return output.view(SizeRef(std::move(bigger_shape_vec)));

}



#define HANDLE_DIMS_TO_MULT_MKL(type) if(a.dims() == 2 && a.dims() == b.dims()){return handle_dim2_matmult_mkl<type>(a, b, trans_a, trans_b);}\
				      else if(a.dims() > b.dims()){return handle_dim_n_k_matmul_mkl<type>(a, b, trans_a, trans_b);}\
				      else if(b.dims() > a.dims()){return handle_dim_k_n_matmul_mkl<type>(a, b, trans_a, trans_b);}\
				      return handle_dim_n_n_matmul_mkl<type>(a, b, trans_a, trans_b);


#define HANDLE_DIMS_TO_MULT_MKL_SUBT(type) std_functional::subT_check(a, b); \
				      if(tmp_a.dims() > tmp_b.dims()){return handle_dim_n_k_subT_matmult_mkl<type>(a, b, trans_a, trans_b);}\
				      else if(tmp_b.dims() > tmp_a.dims()){return handle_dim_k_n_subT_matmult_mkl<type>(a, b, trans_a, trans_b);}\
				      return handle_dim_n_n_subT_matmult_mkl<type>(a, b, trans_a, trans_b);


Tensor mkl_matmult(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){
#ifdef _HALF_FLOAT_SUPPORT_
	if(a.dtype == DType::Float16){
		utils::throw_exception(b.dtype == DType::Float32 || b.dtype == DType::Float16 || b.dtype == DType::Float64,
				"Expected to get a float16 tensor and a float16, float32, or float64, but got $", b.dtype);
		if(b.dtype == DType::Float16)
			return mkl_matmult(a.to_dtype(DType::Float32), b.to_dtype(DType::Float32), trans_a, trans_b);
		return mkl_matmult(a.to_dtype(b.dtype), b, trans_a, trans_b);	
	}
	if(b.dtype == DType::Float16){
		utils::throw_exception(a.dtype == DType::Float32 || a.dtype == DType::Float64,
				"Expected to get a float16 tensor and a float16, float32, or float64, but got $", b.dtype);
		return mkl_matmult(a, b.to_dtype(a.dtype), trans_a, trans_b);	
	}
#endif
	utils::throw_exception(a.dtype == b.dtype, "Expected to multiply tensors of the same dtype but got $ and $", a.dtype, b.dtype);
	if(a.dtype != DType::TensorObj)
		utils::THROW_EXCEPTION(a.dims() > 1 && b.dims() > 1, "\nRuntimeError: Expected tensors to have dims greater than 1, but instead had dims of $ and $", a.dims(), b.dims());
	/* utils::THROW_EXCEPTION(a.shape()[-1] == b.shape()[-2] || a.shape()[-1] == b.shape()[-1], "\nRuntimeError: Expected second tensor rows or collumns to be $ when it was $ and $",a.shape()[-1],b.shape()[-2], b.shape()[-1]); */
	
	switch(a.dtype){
		case DType::Float32:{
			HANDLE_DIMS_TO_MULT_MKL(float)
		}
		case DType::Float64:{
			HANDLE_DIMS_TO_MULT_MKL(double)
		}
		case DType::Complex64:{
			HANDLE_DIMS_TO_MULT_MKL(std::complex<float>)
		}
		case DType::Complex128:{
			HANDLE_DIMS_TO_MULT_MKL(std::complex<double>)
		}
		case DType::int16:{
			HANDLE_DIMS_TO_MULT_MKL(int16_t)
		}
		case DType::int8:
			return mkl_matmult(a.to_dtype(DType::int16), b.to_dtype(DType::int16), trans_a, trans_b);
		case DType::uint8:
			return mkl_matmult(a.to_dtype(DType::int16), b.to_dtype(DType::int16), trans_a, trans_b);
		case DType::TensorObj:{
			const Tensor& tmp_a = a[0].item<Tensor>();
			const Tensor& tmp_b = b[0].item<Tensor>();
			switch(tmp_a.dtype){
				case DType::Float32:{
					HANDLE_DIMS_TO_MULT_MKL_SUBT(float)
				}
				case DType::Float64:{
					HANDLE_DIMS_TO_MULT_MKL_SUBT(double)
				}
				case DType::Complex64:{
					HANDLE_DIMS_TO_MULT_MKL_SUBT(std::complex<float>)
				}
				case DType::Complex128:{
					HANDLE_DIMS_TO_MULT_MKL_SUBT(std::complex<double>)
				}
				case DType::int16:{
					HANDLE_DIMS_TO_MULT_MKL_SUBT(int16_t)
				}
				default:
					return std_functional::std_matmult(a, b, trans_a, trans_b);
			}
			
		}
		default:
			return std_functional::std_matmult(a, b, trans_a, trans_b);
	}

}


}
}}
#endif
