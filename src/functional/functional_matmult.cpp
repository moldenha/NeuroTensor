#include "functional_matmult.h"
/* #include <immintrin.h> */

#include <iostream>
#include <sys/wait.h>
#include "../convert/Convert.h"
#include "../dtype/Scalar.h"
#include "functional.h"
#include "../dtype/ArrayVoid.hpp"

namespace nt{
namespace functional{

/* float dot_product_8(const float* a, const float* b, size_t size) { */
/*     int remainder = size % 8; */
/*     __m256 sum = _mm256_setzero_ps(); */

/*     for (size_t i = 0; i < size; i += 8) { */
/*         __m256 a_data = _mm256_loadu_ps(&a[i]); */
/*         __m256 b_data = _mm256_loadu_ps(&b[i]); */
/*         sum = _mm256_add_ps(sum, _mm256_mul_ps(a_data, b_data)); */
/*     } */

/*     // Horizontal sum across the 8 float values */
/*     __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1)); */
/*     sum128 = _mm_hadd_ps(sum128, sum128); */
/*     sum128 = _mm_hadd_ps(sum128, sum128); */

/*     // Extract the final result */
/*     std::cout<<"extracting final result"<<std::endl; */
/*     float result = _mm_cvtss_f32(sum128); */
/*     std::cout<<"extracted final result"<<std::endl; */

/*     // Process the remainder */
/*     for (int i = size - remainder; i < size; ++i) { */
/*         result += a[i] * b[i]; */
/*     } */

/*     return result; */
/* } */

/* double dot_product_4(const double* a, const double* b, size_t size) { */
/*     int remainder = size % 4; */
/*     __m256d sum = _mm256_setzero_pd(); */

/*     for (int i = 0; i < size - remainder; i += 4) { */
/*         __m256d a_vec = _mm256_loadu_pd(&a[i]); */
/*         __m256d b_vec = _mm256_loadu_pd(&b[i]); */
/*         sum = _mm256_add_pd(sum, _mm256_mul_pd(a_vec, b_vec)); */
/*     } */

/*     // Sum the remaining elements */
/*     double result = 0.0; */
/*     double* sum_arr = reinterpret_cast<double*>(&sum); */
/*     for (int i = 0; i < 4; ++i) { */
/*         result += sum_arr[i]; */
/*     } */

/*     // Process the remainder */
/*     for (int i = size - remainder; i < size; ++i) { */
/*         result += a[i] * b[i]; */
/*     } */

/*     return result; */
/* } */

inline static constexpr auto dot_product = [](auto begin, auto end, auto begin_2) -> Scalar{
	typedef typename std::remove_const<typename decltype(begin)::value_type>::type value_t;
	return std::inner_product(begin, end, begin_2, value_t(0));
};


Tensor matmult(const Tensor& a, const Tensor& b, bool un_transpose){
	utils::throw_exception(a.dtype != DType::Bool, "RuntimeError: Tensor DType was Bool which is unallowed for matmult function");
	utils::throw_exception(a.dtype == b.dtype, "\nRuntimeError: Expected second tensor to have dtype of $, instead had dtype of $", a.dtype, b.dtype);
	utils::throw_exception(a.dims() > 1 && b.dims() > 1, "\nRuntimeError: Expected tensors to have dims greater than 1, but instead had dims of $ and $", a.dims(), b.dims());
	utils::throw_exception(a.shape()[-1] == b.shape()[-2], "\nRuntimeError: Expected second tensor rows to be $ when it was $",a.shape()[-1],b.shape()[-2]);
	if(a.dims() != b.dims()){
		if(a.dims() > b.dims()){
			std::vector<uint32_t> size_outp = a.shape().Vec();
			size_outp.back() = b.shape().back();
			uint32_t start = a.dims() - b.dims() - 1;
			for(uint32_t i = start; i < size_outp.size()-2; ++i){
				utils::throw_exception(size_outp[i] == b.shape()[i - start], "RuntimeError: Expected sizes at dimension $ to be $ but got $", i-start, size_outp[i], b.shape()[i-start]); 
			}
			Tensor output = zeros(SizeRef(std::move(size_outp)), a.dtype);
			dtype_list o_begin = output.val_begin();
			const Tensor a_1 = a.split_axis(start);
			const Tensor* a1_begin = reinterpret_cast<const Tensor*>(a_1.data_ptr());
			const Tensor* a1_end = a1_begin + a_1.numel();
			b.RowColSwap();
			const Tensor b1 = b.split_axis(-2);
			for(;a1_begin != a1_end; ++a1_begin){
				const Tensor* b1_begin = reinterpret_cast<const Tensor*>(b1.data_ptr());
				const Tensor* b1_end = b1_begin + b1.numel();
				const Tensor a2 = a1_begin->split_axis(-2);
				const Tensor* a2_begin = reinterpret_cast<const Tensor*>(a2.data_ptr());
				/* const Tensor* a2_end = a2_begin + a2.numel(); */
				for(;b1_begin != b1_end; ++b1_begin, ++a2_begin, ++o_begin){ // also itterate the output_iterator
					o_begin.set(b1_begin->arr_void().cexecute_function_nbool(dot_product, a2_begin->arr_void()));
				}

			}
			if(un_transpose){b.RowColSwap();}
			return std::move(output);
		}
		//b.dims() > a.dims()
		else{
			std::vector<uint32_t> size_outp = b.shape().Vec();
			size_outp[size_outp.size()-2] = a.shape()[-2];
			uint32_t start = b.dims() - a.dims() - 1;
			for(uint32_t i = start; i < size_outp.size() - 2; ++i)
				utils::throw_exception(size_outp[i] == a.shape()[i - start], "RuntimeError: Expected sizes at dimension $ to be $ but got $", i-start, size_outp[i], a.shape()[i-start]);
			Tensor output = zeros(SizeRef(std::move(size_outp)), a.dtype);
			dtype_list o_begin = output.val_begin();
			b.RowColSwap();
			const Tensor b_1 = b.split_axis(start);
			const Tensor* b1_begin = reinterpret_cast<const Tensor*>(b_1.data_ptr());
			const Tensor* b1_end = b1_begin + b_1.numel();
			const Tensor a1 = a.split_axis(-2);
			for(;b1_begin != b1_end; ++b1_begin){
				const Tensor b2 = b1_begin->split_axis(-2);
				const Tensor* b2_begin = reinterpret_cast<const Tensor*>(b2.data_ptr());
				const Tensor* a1_begin = reinterpret_cast<const Tensor*>(a1.data_ptr());
				const Tensor* a1_end = a1_begin + a1.numel();
				for(;a1_begin != a1_end; ++a1_begin, ++b2_begin, ++o_begin)
					o_begin.set(a1_begin->arr_void().cexecute_function_nbool(dot_product, b2_begin->arr_void()));

			}
			if(un_transpose){b.RowColSwap();}
			return std::move(output);
		}
	}

	std::vector<uint32_t> size_outp = a.shape().Vec();
	size_outp.back() = b.shape().back();
	for(uint32_t i = 0; i < size_outp.size()-2; ++i){
		utils::throw_exception(size_outp[i] == b.shape()[i], "RuntimeError: Expected sizes at dimension $ to be $ but got $", i, size_outp[i], b.shape()[i]); 
	}
	b.RowColSwap();
	Tensor output = zeros(SizeRef(std::move(size_outp)), a.dtype);
	dtype_list o_begin = output.val_begin();
	const Tensor b1 = b.split_axis(-2);
	const Tensor* b_begin = reinterpret_cast<const Tensor*>(b1.data_ptr());
	const Tensor* b_end = b_begin + b1.numel();
	const Tensor a1 = a.split_axis(-2);
	const Tensor* a_begin = reinterpret_cast<const Tensor*>(a1.data_ptr());
	const Tensor* a_end = a_begin + a1.numel();
	for(;b_begin != b_end; ++b_begin, ++a_begin, ++o_begin)
		o_begin.set(b_begin->arr_void().cexecute_function_nbool(dot_product, a_begin->arr_void()));
	if(un_transpose){b.RowColSwap();}
	return std::move(output);

}

}
}
