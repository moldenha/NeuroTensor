#ifndef _NT_MATMULT_STD_CPP_
#define _NT_MATMULT_STD_CPP_

#include <cstdint>

#include "../../Tensor.h"
#include "../../memory/iterator.h"
#include "../../refs/SizeRef.h"
#include "../../dtype/ArrayVoid.h"
#include "../../dtype/DType.h"
#include "../../dtype/DType_enum.h"




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
/* #include <sys/_types/_int32_t.h> */
/* #include <sys/_types/_int64_t.h> */
#include <sys/types.h>
#include <type_traits>
#include <vector>
#include "../../utils/utils.h"
#include <chrono>
#include "../functional.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <random>
#include <cmath>
#include "../../dtype/ArrayVoid.hpp"
#include "matmult/nt_matmult.h"

namespace nt{
namespace functional{
namespace cpu{

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
	
}



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

	Tensor tensor_C = zeros(std::move(c_shape), DTypeFuncs::type_to_dtype<T>);
	
	T* C = reinterpret_cast<T*>(tensor_C.data_ptr());

	const int64_t& m = a_shape[-2]; // Number of rows in A
	const int64_t& k = a_shape[-1]; // Number of columns in A and rows in B
	const int64_t& n = b_shape[-1]; // Number of columns in B
	// Perform matrix multiplication: C = A * B
	nt_matmult<T>(A, B, C, m, k, k, n, transpose_a, transpose_b);
	return std::move(tensor_C);

}

template<typename T>
Tensor& matmult_std_single(const T* A, const T* B, Tensor& c, const SizeRef& o_a_shape, const SizeRef& o_b_shape, const bool& transpose_a, const bool& transpose_b){

	SizeRef a_shape = (transpose_a) ? o_a_shape.transpose(-1,-2) : o_a_shape.clone();

	SizeRef b_shape = (transpose_b) ? o_b_shape.transpose(-1,-2) : o_b_shape.clone();
	/* SizeRef a_shape = o_a_shape; */
	/* SizeRef b_shape = o_b_shape; */
	utils::THROW_EXCEPTION(a_shape[-1] == b_shape[-2], "Expected at positions (-1,-2) for shapes $,$ -> ($:$) to be equal", a_shape, b_shape, a_shape[-1], b_shape[-2]);
	std::vector<typename Tensor::size_value_t> vec = a_shape.Vec();
	vec.back() = b_shape.back();
	SizeRef c_shape(std::move(vec));

	//Tensor tensor_C = zeros(std::move(c_shape), DTypeFuncs::type_to_dtype<T>);
    utils::THROW_EXCEPTION(c.shape() == c_shape,
                           "Got invalid shape for output tensor $ expected $", c.shape(), c_shape);
    utils::THROW_EXCEPTION(c.is_contiguous(),
                           "Expected output tensor for matrix multiplication to be contiguous");
    utils::THROW_EXCEPTION(c.dtype == DTypeFuncs::type_to_dtype<T>,
                           "Expected dtype $ for output matrix multiplication, C dtype inputted was $",
                            DTypeFuncs::type_to_dtype<T>, c.dtype);
	T* C = reinterpret_cast<T*>(c.data_ptr());

	const int64_t& m = a_shape[-2]; // Number of rows in A
	const int64_t& k = a_shape[-1]; // Number of columns in A and rows in B
	const int64_t& n = b_shape[-1]; // Number of columns in B
	// Perform matrix multiplication: C = A * B
	nt_matmult<T>(A, B, C, m, k, k, n, transpose_a, transpose_b);
	return c;

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

template<typename T>
Tensor& handle_dim2_matmult_std(const Tensor& a, const Tensor& b, Tensor& c, bool trans_a, bool trans_b){
	uint32_t iterator_type_a = a.arr_void().get_bucket().iterator_type(); // 3 = strided_view, 2 = bucketed, 1 = is_contiguous
	uint32_t iterator_type_b = b.arr_void().get_bucket().iterator_type(); // 
	
	bool transpose_a, transpose_b;
	determine_transposes(a, b, transpose_a, transpose_b, trans_a, trans_b);

	if(iterator_type_a == 1){ // contiguous
		const T* A = a.arr_void().get_bucket().cbegin<1, T>();
		if(iterator_type_b == 1){
			const T* B = b.arr_void().get_bucket().cbegin<1, T>();
			return matmult_std_single<T>(A, B, c, a.shape(), b.shape(), transpose_a, transpose_b);
		}

		if(iterator_type_b == 3 && was_transposed_back(b)){
			Tensor contiguous_b = make_contiguous_look(b, handle_looking_contiguous(b)); //there is no possibility for it to be an array of tensors since it is only dim() == 2
			transpose_b = !transpose_b;
			const T* B = contiguous_b.arr_void().get_bucket().cbegin<1, T>();
			return matmult_std_single<T>(A, B, c, a.shape(), contiguous_b.shape(), transpose_a, transpose_b);

		}
		//this is if it is bucketed or strided
		//normally if it were bucketed, it would see if there is any possible way to seperate it into matricies, otherwise just this:
		intrusive_ptr<T[]> intrusive_B = (iterator_type_b == 3) ? convert_to_intrusive_strided<T>(b) : convert_to_intrusive_bucketed<T>(b);
		const T* B = intrusive_B.get();
		return matmult_std_single<T>(A, B, c, a.shape(), b.shape(), transpose_a, transpose_b);
	}
	if(iterator_type_a == 2){ //bucketed
		intrusive_ptr<T[]> intrusive_A = convert_to_intrusive_bucketed<T>(a);
		const T* A = intrusive_A.get();
		if(iterator_type_b == 1){
			const T* B = b.arr_void().get_bucket().cbegin<1, T>();
			return matmult_std_single<T>(A, B, c, a.shape(), b.shape(), transpose_a, transpose_b);
		}

		if(iterator_type_b == 3 && was_transposed_back(b)){
			Tensor contiguous_b = make_contiguous_look(b, handle_looking_contiguous(b)); //there is no possibility for it to be an array of tensors since it is only dim() == 2
			transpose_b = !transpose_b;
			const T* B = contiguous_b.arr_void().get_bucket().cbegin<1, T>();
			return matmult_std_single<T>(A, B, c, a.shape(), contiguous_b.shape(), transpose_a, transpose_b);

		}
		//this is if it is bucketed or strided
		//normally if it were bucketed, it would see if there is any possible way to seperate it into matricies, otherwise just this:
		intrusive_ptr<T[]> intrusive_B = (iterator_type_b == 3) ? convert_to_intrusive_strided<T>(b) : convert_to_intrusive_bucketed<T>(b);
		const T* B = intrusive_B.get();
		return matmult_std_single<T>(A, B, c, a.shape(), b.shape(), transpose_a, transpose_b);
	}
	if(was_transposed_back(a)){
		Tensor contiguous_a = make_contiguous_look(a, handle_looking_contiguous(a)); //there is no possibility for it to be an array of tensors since it is only dim() == 2
		transpose_a = !transpose_a;
		const T* A = contiguous_a.arr_void().get_bucket().cbegin<1, T>();
		if(iterator_type_b == 1){
			const T* B = b.arr_void().get_bucket().cbegin<1, T>();
			return matmult_std_single<T>(A, B, c, a.shape(), b.shape(), transpose_a, transpose_b);
		}

		if(iterator_type_b == 3 && was_transposed_back(b)){
			Tensor contiguous_b = make_contiguous_look(b, handle_looking_contiguous(b)); //there is no possibility for it to be an array of tensors since it is only dim() == 2
			transpose_b = !transpose_b;
			const T* B = contiguous_b.arr_void().get_bucket().cbegin<1, T>();
			return matmult_std_single<T>(A, B, c, a.shape(), contiguous_b.shape(), transpose_a, transpose_b);

		}
		//this is if it is bucketed or strided
		//normally if it were bucketed, it would see if there is any possible way to seperate it into matricies, otherwise just this:
		intrusive_ptr<T[]> intrusive_B = (iterator_type_b == 3) ? convert_to_intrusive_strided<T>(b) : convert_to_intrusive_bucketed<T>(b);
		const T* B = intrusive_B.get();
		return matmult_std_single<T>(A, B, c, contiguous_a.shape(), b.shape(), transpose_a, transpose_b);
	}
	intrusive_ptr<T[]> intrusive_A = convert_to_intrusive_strided<T>(a);
	const T* A = intrusive_A.get();
	if(iterator_type_b == 1){
		const T* B = b.arr_void().get_bucket().cbegin<1, T>();
		return matmult_std_single<T>(A, B, c, a.shape(), b.shape(), transpose_a, transpose_b);
	}

	if(iterator_type_b == 3 && was_transposed_back(b)){
		Tensor contiguous_b = make_contiguous_look(b, handle_looking_contiguous(b)); //there is no possibility for it to be an array of tensors since it is only dim() == 2
		transpose_b = !transpose_b;
		const T* B = contiguous_b.arr_void().get_bucket().cbegin<1, T>();
		return matmult_std_single<T>(A, B, c, a.shape(), contiguous_b.shape(), transpose_a, transpose_b);

	}
	//this is if it is bucketed or strided
	//normally if it were bucketed, it would see if there is any possible way to seperate it into matricies, otherwise just this:
	intrusive_ptr<T[]> intrusive_B = (iterator_type_b == 3) ? convert_to_intrusive_strided<T>(b) : convert_to_intrusive_bucketed<T>(b);
	const T* B = intrusive_B.get();
	return matmult_std_single<T>(A, B, c, a.shape(), b.shape(), transpose_a, transpose_b);
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
    return t.clone().split_axis(-3);
    //this function has an issue with the following:
	// Tensor splitting = t.split_axis(-3); //first it splits along all matriciess
	// Tensor* begin = reinterpret_cast<Tensor*>(splitting.data_ptr());
	// Tensor* end = begin + splitting.numel();
	// for(;begin != end; ++begin){
	// 	if(!begin->is_contiguous()){
	// 		Tensor contiged = begin->contiguous();
	// 		std::swap(*begin, contiged);
	// 	}
	// }
	// return std::move(splitting);
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
        //the following isn't working properly for some reason
        //will have to look into it
        // if(was_transposed_back(t)){
        //     std::cout << "was transposed"<<std::endl;
        //     std::cout << std::boolalpha << transpose << "->";
			// transpose = !transpose;
        //     std::cout << transpose << std::noboolalpha << std::endl;
			// out_t = make_contiguous_look(t, -3);
			// return;
		// }
        // std::cout << "going to split and contigitize"<<std::endl;
		out_t = split_and_contigitize(t);
		return;
	}
	if(iterator_type == 2){ //bucketed
        // std::cout << "going to split and contigitize (2) "<<std::endl;
        int64_t split_by = handle_looking_buckets(t);
		// if(split_by == -1){std::cout << "going to split and contigitize (2->do) "<<std::endl; out_t = split_and_contigitize(t); return;}
        // std::cout << "else"<<std::endl;
	}
	//already contiguous
    // std::cout << "already contiguous"<<std::endl;
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


/* void calculate_optimal_group_std(int64_t M, int64_t K, int64_t N, int64_t batch_size, int64_t& group_count, int64_t*& group_sizes) { */
/*     // Get cache sizes (assuming L1, L2, and L3) */
/*     /1* uint64_t L1_cache_size = 32 * 1024;  // L1 cache size in bytes (32 KB) *1/ */
/*     /1* uint64_t L2_cache_size = 256 * 1024; // L2 cache size in bytes (256 KB) *1/ */
/*     uint64_t L3_cache_size = 8 * 1024 * 1024; // L3 cache size in bytes (8 MB) */

/*     // Calculate matrix sizes in bytes */
/*     uint64_t size_A = M * K * sizeof(float); */
/*     uint64_t size_B = K * N * sizeof(float); */
/*     uint64_t size_C = M * N * sizeof(float); */

/*     // Calculate the number of elements that can fit in each cache level */
/*     /1* uint64_t elements_L1 = L1_cache_size / sizeof(float); *1/ */
/*     /1* uint64_t elements_L2 = L2_cache_size / sizeof(float); *1/ */
/*     uint64_t elements_L3 = L3_cache_size / sizeof(float); */

/*     // Calculate the number of elements for each group */
/*     uint64_t total_size = (size_A + size_B + size_C); */
/*     uint64_t group_elements; */
/*     if(total_size > elements_L3){ */
/* 	group_elements = elements_L3; */
/*     } */
/*     else{ */
/* 	group_elements = elements_L3 / total_size; */
/*     } */


/*     // Calculate the number of groups needed */
/*     group_count = ceil((float)(batch_size * M * N) / group_elements); */

/*     // Allocate memory for group sizes array */
/*     group_sizes = new MKL_INT64[group_count]; */

/*     // Distribute batch size evenly among groups */
/*     MKL_INT64 batch_per_group = ceil((float)batch_size / group_count); */

/*     // Assign group sizes */
/*     for (MKL_INT64 i = 0; i < group_count; ++i) { */
/*         group_sizes[i] = std::min(batch_per_group, (MKL_INT64)batch_size); */
/*         batch_size -= group_sizes[i]; */
/*     } */
/* } */


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
	const int64_t& K = (transpose_a) ? a_shape[-2] : (transpose_b) ? b_shape[-1] : begin_shape[-1]; // Number of columns in A and rows in B
	const int64_t& N = (transpose_a && transpose_b) ? b_shape[-1] : begin_b_shape[-1]; // Number of columns in B
	
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
	
	/* int64_t group_count; */
	/* int64_t* group_size; */

	/* std::cout << "going to calculate optimal group size..."<<std::endl; */
	/* calculate_optimal_group_std(M, K, N, batch_size, group_count, group_size); */
	/* std::cout << "calculated"<<std::endl; */
	
/* void batched_matrix_multiplication_typed(const bool& transpose_a, const bool& transpose_b, const T* A, const T* B, T* C, int64_t A_rows, int64_t B_cols, int64_t A_cols, const int64_t& group_count, const int64_t* group_sizes) noexcept { */
	
	nt_matmult_batch<T>(A_array, B_array, C_array, batch_size, a_shape[-2], a_shape[-1], b_shape[-2], b_shape[-1], transpose_a, transpose_b);
	/* batched_matrix_multiplication_typed<T>(transpose_a, transpose_b, A_array, B_array, C_array, M, N, K, group_count, group_size); */

	/* delete[] group_size; */
	return std::move(tensor_C);

	
}



//expects all tensors to only have dim of 2, and all to have the same shape
template<typename T,
	typename outT = T,
	DType DT = DTypeFuncs::type_to_dtype<T>,
	DType outDT = DTypeFuncs::type_to_dtype<outT>>
Tensor& handle_tensors_of_tensors_std(Tensor& a, Tensor& b, Tensor& c, const bool transpose_a, const bool transpose_b){
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
	const int64_t& K = (transpose_a) ? a_shape[-2] : (transpose_b) ? b_shape[-1] : begin_shape[-1]; // Number of columns in A and rows in B
	const int64_t& N = (transpose_a && transpose_b) ? b_shape[-1] : begin_b_shape[-1]; // Number of columns in B
	
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
    utils::THROW_EXCEPTION(c.dims() >= 3,
                           "Expected the dimension of c to be greater than 2 as the dimensionality of a or b is as well for matrix multiplication");
    utils::THROW_EXCEPTION(c.shape() == c_shape || (c.numel() == (batch_size * a_shape[-2] * b_shape[-1]) &&
                                                    c.shape()[-1] == b_shape[-1] && c.shape()[-2] == a_shape[-2]),
                           "Got invalid shape for output tensor $ expected $", c.shape(), c_shape);
    utils::THROW_EXCEPTION(c.is_contiguous(),
                           "Expected output tensor for matrix multiplication to be contiguous");
    utils::THROW_EXCEPTION(c.dtype == outDT,
                           "Expected dtype $ for output matrix multiplication, C dtype inputted was $",
                            DTypeFuncs::type_to_dtype<T>, c.dtype);

	//Tensor tensor_C = zeros(std::move(c_shape), outDT); 
	outT* C = reinterpret_cast<outT*>(c.data_ptr());
	const int64_t c_matrix_size = c.shape()[-1] * c.shape()[-2];
	
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
	
	/* int64_t group_count; */
	/* int64_t* group_size; */

	/* std::cout << "going to calculate optimal group size..."<<std::endl; */
	/* calculate_optimal_group_std(M, K, N, batch_size, group_count, group_size); */
	/* std::cout << "calculated"<<std::endl; */
	
/* void batched_matrix_multiplication_typed(const bool& transpose_a, const bool& transpose_b, const T* A, const T* B, T* C, int64_t A_rows, int64_t B_cols, int64_t A_cols, const int64_t& group_count, const int64_t* group_sizes) noexcept { */
	
	nt_matmult_batch<T>(A_array, B_array, C_array, batch_size, a_shape[-2], a_shape[-1], b_shape[-2], b_shape[-1], transpose_a, transpose_b);
	/* batched_matrix_multiplication_typed<T>(transpose_a, transpose_b, A_array, B_array, C_array, M, N, K, group_count, group_size); */

	/* delete[] group_size; */
	return c;

	
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
Tensor& handle_dim_n_k_matmult_std(const Tensor& a, const Tensor& b, Tensor& c, bool trans_a, bool trans_b){
	
	constexpr DType in_dtype = DTypeFuncs::type_to_dtype<T>;
	utils::THROW_EXCEPTION(a.dims() > b.dims(), "Expected to have larger tensor in terms of dims as a but got $ and $", a.dims(), b.dims());
	utils::THROW_EXCEPTION(a.dtype == in_dtype && b.dtype == in_dtype, "Expected both tensors to have the dtype $ but got $ and $", in_dtype, a.dtype, b.dtype);
	int64_t start = a.dims()-b.dims();
	for(int64_t i = start; i < a.dims() - 2; ++i){
		utils::THROW_EXCEPTION(a.shape()[i] == b.shape()[i-start], "Cannot multiply tensors of shapes $ and $, outter dimensions do not match", a.shape(), b.shape());
	}
	
	bool transpose_a, transpose_b;
	determine_transposes(a, b, transpose_a, transpose_b, trans_a, trans_b);

	// auto bigger_shape_vec = a.shape().arr();

	Tensor bigger = return_split_tensor(a, transpose_a);
	Tensor smaller = return_split_tensor(b, transpose_b);
	utils::THROW_EXCEPTION(bigger.numel() % smaller.numel() == 0, "Expected there to be a fraction of arrays in the tensors but got $ and  $ as the amount of matricies", bigger.numel(), smaller.numel());
	smaller = smaller.repeat_(bigger.numel() / smaller.numel());
	return handle_tensors_of_tensors_std<T>(bigger, smaller, c, transpose_a, transpose_b);
}

template<typename T>
Tensor handle_dim_k_n_matmult_std(Tensor a, Tensor b, bool trans_a, bool trans_b){
	constexpr DType in_dtype = DTypeFuncs::type_to_dtype<T>;
	utils::THROW_EXCEPTION(b.dims() > a.dims(), "Expected to have larger tensor in terms of dims as a but got $ > $", b.dims(), a.dims());
	utils::THROW_EXCEPTION(a.dtype == in_dtype && b.dtype == in_dtype, "Expected both tensors to have the dtype $ but got $ and $", in_dtype, a.dtype, b.dtype);
	int64_t start = b.dims()-a.dims();
	for(int64_t i = start; i < b.dims() - 2; ++i){
		utils::THROW_EXCEPTION(a.shape()[i-start] == b.shape()[i], "Cannot multiply tensors of shapes $ and $, outter dimensions do not match", a.shape(), b.shape());
	}
	
	bool transpose_a, transpose_b;
	determine_transposes(a, b, transpose_a, transpose_b, trans_a, trans_b);
    utils::THROW_EXCEPTION(!transpose_a && !transpose_b, "Transposing a and b are not currently supported for tensors of tensors, run transpose_Tensors(-1, -2).clone(), and then run matrix multiplication over the tensors");
    //otherwise there is a seg fault
    if(transpose_a){a = a.transpose_Tensors(-1,-2).clone();}
    if(transpose_b){b = b.transpose_Tensors(-1,-2).clone();}
    
	
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
Tensor& handle_dim_k_n_matmult_std(Tensor a, Tensor b, Tensor& c, bool trans_a, bool trans_b){
	constexpr DType in_dtype = DTypeFuncs::type_to_dtype<T>;
	utils::THROW_EXCEPTION(b.dims() > a.dims(), "Expected to have larger tensor in terms of dims as a but got $ > $", b.dims(), a.dims());
	utils::THROW_EXCEPTION(a.dtype == in_dtype && b.dtype == in_dtype, "Expected both tensors to have the dtype $ but got $ and $", in_dtype, a.dtype, b.dtype);
	int64_t start = b.dims()-a.dims();
	for(int64_t i = start; i < b.dims() - 2; ++i){
		utils::THROW_EXCEPTION(a.shape()[i-start] == b.shape()[i], "Cannot multiply tensors of shapes $ and $, outter dimensions do not match", a.shape(), b.shape());
	}
	
	bool transpose_a, transpose_b;
	determine_transposes(a, b, transpose_a, transpose_b, trans_a, trans_b);
    utils::THROW_EXCEPTION(!transpose_a && !transpose_b, "Transposing a and b are not currently supported for tensors of tensors, run transpose_Tensors(-1, -2).clone(), and then run matrix multiplication over the tensors");
    //otherwise there is a seg fault
    if(transpose_a){a = a.transpose_Tensors(-1,-2).clone();}
    if(transpose_b){b = b.transpose_Tensors(-1,-2).clone();}
    
	
	// auto bigger_shape_vec = b.shape().arr();

	Tensor bigger = return_split_tensor(b, transpose_b);
	Tensor smaller = return_split_tensor(a, transpose_a);
	utils::THROW_EXCEPTION(bigger.numel() % smaller.numel() == 0, "Expected there to be a fraction of arrays in the tensors but got $ and  $ as the amount of matricies", bigger.numel(), smaller.numel());
	smaller = smaller.repeat_(bigger.numel() / smaller.numel());
	return handle_tensors_of_tensors_std<T>(smaller, bigger, c, transpose_a, transpose_b);
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

template<typename T>
Tensor& handle_dim_n_n_matmult_std(const Tensor& a, const Tensor& b, Tensor& c, bool trans_a, bool trans_b){
	constexpr DType in_dtype = DTypeFuncs::type_to_dtype<T>;
	utils::THROW_EXCEPTION(b.dims() == a.dims(), "Expected to have same tensor dims as a but got $ != $", b.dims(), a.dims());
	utils::THROW_EXCEPTION(a.dtype == in_dtype && b.dtype == in_dtype, "Expected both tensors to have the dtype $ but got $ and $", in_dtype, a.dtype, b.dtype);
	int64_t start = b.dims()-a.dims();
	for(int64_t i = start; i < b.dims() - 2; ++i){
		utils::THROW_EXCEPTION(a.shape()[i-start] == b.shape()[i], "Cannot multiply tensors of shapes $ and $, outter dimensions do not match", a.shape(), b.shape());
	}
	
	//auto bigger_shape_vec = a.shape().arr();
	bool transpose_a, transpose_b;
	determine_transposes(a, b, transpose_a, transpose_b, trans_a, trans_b);
	Tensor A = return_split_tensor(a, transpose_a);
	Tensor B = return_split_tensor(b, transpose_b);
	utils::THROW_EXCEPTION(A.numel() == B.numel(), "Expected there to be the same number of matricies but got $ and  $ as the amount of matricies", A.numel(), B.numel());
	return handle_tensors_of_tensors_std<T>(A, B, c, transpose_a, transpose_b);
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
	nt_matmult_batch<T>(A_array, B_array, C_array, batch_size, M, K, K, N, transpose_a, transpose_b);
	/* int64_t group_count; */
	/* int64_t* group_size; */

	/* std::cout << "going to calculate optimal group size..."<<std::endl; */
	/* calculate_optimal_group_std(M, K, N, batch_size, group_count, group_size); */
	/* std::cout << "calculated"<<std::endl; */
	
/* void batched_matrix_multiplication_typed(const bool& transpose_a, const bool& transpose_b, const T* A, const T* B, T* C, int64_t A_rows, int64_t B_cols, int64_t A_cols, const int64_t& group_count, const int64_t* group_sizes) noexcept { */


	/* batched_matrix_multiplication_typed<T>(transpose_a, transpose_b, A_array, B_array, C_array, M, N, K, group_count, group_size); */

	/* delete[] group_size; */
	return std::move(tensor_C);

	
}


//a and be are tensors of tensors of tensors of type T
template<typename T,
	typename outT = T,
	DType DT = DTypeFuncs::type_to_dtype<T>,
	DType outDT = DTypeFuncs::type_to_dtype<outT>>
Tensor& handle_tensors_of_tensors_std_subT(Tensor& a, Tensor& b, Tensor& c, const bool transpose_a, const bool transpose_b){
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
    utils::THROW_EXCEPTION(c.dims() == 3,
                           "When multiplying tensors of tensors, the output is expected to be batched matrices, but got"
                           "a shape of $", c.shape());
    utils::THROW_EXCEPTION(c_shape.multiply() == c.numel(),
                           "Expected the number of elements in c to be $ but got $ for shape $",
                           c_shape.multiply(), c.numel(), c.shape());
    utils::THROW_EXCEPTION(c_shape[-1] == c.shape()[-1] && c_shape[-2] == c.shape()[-2],
                           "Expected the matrix shape of C to be {$, $}, but got {$, $}", 
                           c_shape[-2], c_shape[-1], c.shape()[-2], c.shape()[-1]);
    utils::THROW_EXCEPTION(c.dtype == outDT,
                           "Expected dtype of out tensor from matmult to be $ but got $", outDT, c.dtype);
	//Tensor tensor_C = zeros(std::move(c_shape), outDT); 
	outT* C = reinterpret_cast<outT*>(c.data_ptr());
	const int64_t c_matrix_size = c.shape()[-1] * c.shape()[-2];
	
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
	nt_matmult_batch<T>(A_array, B_array, C_array, batch_size, M, K, K, N, transpose_a, transpose_b);
	/* int64_t group_count; */
	/* int64_t* group_size; */

	/* std::cout << "going to calculate optimal group size..."<<std::endl; */
	/* calculate_optimal_group_std(M, K, N, batch_size, group_count, group_size); */
	/* std::cout << "calculated"<<std::endl; */
	
/* void batched_matrix_multiplication_typed(const bool& transpose_a, const bool& transpose_b, const T* A, const T* B, T* C, int64_t A_rows, int64_t B_cols, int64_t A_cols, const int64_t& group_count, const int64_t* group_sizes) noexcept { */


	/* batched_matrix_multiplication_typed<T>(transpose_a, transpose_b, A_array, B_array, C_array, M, N, K, group_count, group_size); */

	/* delete[] group_size; */
	return c;
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
	std::for_each(n_transpose_a.cbegin(), n_transpose_a.cend(), [&transpose_a_get](const bool& t){utils::THROW_EXCEPTION(transpose_a_get == t, "Memory error with transposes a");});
	std::for_each(n_transpose_b.cbegin(), n_transpose_b.cend(), [&transpose_b_get](const bool& t){utils::THROW_EXCEPTION(transpose_b_get == t, "Memory error with transposes b");});
	
	auto bigger_shape_vec = temp_a.shape().arr();
	bigger_shape_vec[0] *= a.numel();

	Tensor* bigger_begin = reinterpret_cast<Tensor*>(bigger.data_ptr());
	Tensor* bigger_end = bigger_begin + bigger.numel();
	Tensor* smaller_begin = reinterpret_cast<Tensor*>(smaller.data_ptr());
	for(;bigger_begin != bigger_end; ++bigger_begin, ++smaller_begin){
		utils::THROW_EXCEPTION(bigger_begin->numel() % smaller_begin->numel() == 0, "Expected there to be a fraction of arrays in the tensors but got $ and  $ as the amount of matricies", bigger_begin->numel(), smaller_begin->numel());
		*smaller_begin = smaller_begin->repeat_(bigger_begin->numel() / smaller_begin->numel());
	
	}
	Tensor output = handle_tensors_of_tensors_std_subT<T>(bigger, smaller, transpose_a_get, transpose_b_get);
	bigger_shape_vec[-1] = output.shape()[-1];
	bigger_shape_vec[-2] = output.shape()[-2];
	return output.view(SizeRef(std::move(bigger_shape_vec)));
}


template<typename T>
Tensor& handle_dim_n_k_subT_matmult_std(const Tensor& a, const Tensor& b, Tensor& c, bool trans_a, bool trans_b){
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
	std::for_each(n_transpose_a.cbegin(), n_transpose_a.cend(), [&transpose_a_get](const bool& t){utils::THROW_EXCEPTION(transpose_a_get == t, "Memory error with transposes a");});
	std::for_each(n_transpose_b.cbegin(), n_transpose_b.cend(), [&transpose_b_get](const bool& t){utils::THROW_EXCEPTION(transpose_b_get == t, "Memory error with transposes b");});
	
	auto bigger_shape_vec = temp_a.shape().arr();
	bigger_shape_vec[0] *= a.numel();

	Tensor* bigger_begin = reinterpret_cast<Tensor*>(bigger.data_ptr());
	Tensor* bigger_end = bigger_begin + bigger.numel();
	Tensor* smaller_begin = reinterpret_cast<Tensor*>(smaller.data_ptr());
	for(;bigger_begin != bigger_end; ++bigger_begin, ++smaller_begin){
		utils::THROW_EXCEPTION(bigger_begin->numel() % smaller_begin->numel() == 0, "Expected there to be a fraction of arrays in the tensors but got $ and  $ as the amount of matricies", bigger_begin->numel(), smaller_begin->numel());
		*smaller_begin = smaller_begin->repeat_(bigger_begin->numel() / smaller_begin->numel());
	
	}
	Tensor& output = handle_tensors_of_tensors_std_subT<T>(bigger, smaller, c, transpose_a_get, transpose_b_get);
    return output;
	// bigger_shape_vec[-1] = output.shape()[-1];
	// bigger_shape_vec[-2] = output.shape()[-2];
	// return output.view(SizeRef(std::move(bigger_shape_vec)));
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



template<typename T>
Tensor& handle_dim_k_n_subT_matmult_std(const Tensor& a, const Tensor& b, Tensor& c, bool trans_a, bool trans_b){
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
	Tensor& output = handle_tensors_of_tensors_std_subT<T>(smaller, bigger,c, transpose_a, transpose_b);
    return output;
	// bigger_shape_vec[-1] = output.shape()[-1];
	// bigger_shape_vec[-2] = output.shape()[-2];
	// return output.view(SizeRef(std::move(bigger_shape_vec)));
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
	for(int64_t i = start; i < temp_b.dims() - 2; ++i){
		utils::THROW_EXCEPTION(temp_a.shape()[i-start] == temp_b.shape()[i], "Cannot multiply tensors of shapes $ and $, outter dimensions do not match at ($) and ($)", temp_a.shape(), temp_b.shape(), temp_a.shape()[i-start], temp_b.shape()[i]);
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


//this function is called post tensors_check
template<typename T>
Tensor& handle_dim_n_n_subT_matmult_std(const Tensor& a, const Tensor& b, Tensor& c, bool trans_a, bool trans_b){
	constexpr DType in_dtype = DTypeFuncs::type_to_dtype<T>;
	const Tensor& temp_a = a[0].item<Tensor>();
	const Tensor& temp_b = b[0].item<Tensor>();
	utils::THROW_EXCEPTION(temp_b.dims() == temp_a.dims(), "Expected to have same tensor dims as a but got $ != $", temp_b.dims(), temp_a.dims());
	utils::THROW_EXCEPTION(temp_a.dtype == in_dtype && temp_b.dtype == in_dtype, "Expected both tensors to have the dtype $ but got $ and $", in_dtype, temp_a.dtype, temp_b.dtype);
	int64_t start = temp_b.dims()-temp_a.dims();
	for(int64_t i = start; i < temp_b.dims() - 2; ++i){
		utils::THROW_EXCEPTION(temp_a.shape()[i-start] == temp_b.shape()[i], "Cannot multiply tensors of shapes $ and $, outter dimensions do not match at ($) and ($)", temp_a.shape(), temp_b.shape(), temp_a.shape()[i-start], temp_b.shape()[i]);
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
	Tensor& output = handle_tensors_of_tensors_std_subT<T>(A, B,c, transpose_a_get, transpose_b_get);
	return output;
    // bigger_shape_vec[-1] = output.shape()[-1];
	// bigger_shape_vec[-2] = output.shape()[-2];
	// return output.view(SizeRef(std::move(bigger_shape_vec)));

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
        const Tensor& temp_a = a.item<Tensor>();
        const Tensor& temp_b = b.item<Tensor>();
		if(temp_a.dims() > temp_b.dims())
			return handle_dim_n_k_subT_matmult_std<type_t>(a,b,trans_a,trans_b);
		if(temp_b.dims() > temp_a.dims())
			return handle_dim_k_n_subT_matmult_std<type_t>(a,b,trans_a,trans_b);
		return handle_dim_n_n_subT_matmult_std<type_t>(a,b,trans_a,trans_b);
	}
	else{
		utils::THROW_EXCEPTION(a.dtype != DType::Bool, "Expected dtype to be numerical, but got dtype $", a.dtype);
		return Tensor();
	}	
}



template<DType dt = DType::Integer>
Tensor& handle_tensor_tensor_matrix_multiplication_std(const Tensor& a, const Tensor& b, Tensor& c, bool trans_a, bool trans_b, const DType& sub_dt){
	if(dt != sub_dt){return handle_tensor_tensor_matrix_multiplication_std<DTypeFuncs::next_dtype_it<dt> >(a, b,c, trans_a, trans_b, sub_dt);}
	if constexpr (!(dt == DType::TensorObj || dt == DType::Bool)){
		using type_t = DTypeFuncs::dtype_to_type_t<dt>;
        const Tensor& temp_a = a.item<Tensor>();
        const Tensor& temp_b = b.item<Tensor>();
		if(temp_a.dims() > temp_b.dims())
			return handle_dim_n_k_subT_matmult_std<type_t>(a,b,c,trans_a,trans_b);
		if(temp_b.dims() > temp_a.dims())
			return handle_dim_k_n_subT_matmult_std<type_t>(a,b,c,trans_a,trans_b);
		return handle_dim_n_n_subT_matmult_std<type_t>(a,b,c,trans_a,trans_b);
	}
	else{
		utils::THROW_EXCEPTION(a.dtype != DType::Bool, "Expected dtype to be numerical, but got dtype $", a.dtype);
		return c;
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



template<DType dt = DType::Integer>
Tensor& handle_matrix_multiplication_std(const Tensor& a, const Tensor& b, Tensor& c, bool trans_a, bool trans_b){
	if(dt != a.dtype){return handle_matrix_multiplication_std<DTypeFuncs::next_dtype_it<dt> >(a, b, c, trans_a, trans_b);}
	if constexpr (!(dt == DType::TensorObj || dt == DType::Bool)){
		using type_t = DTypeFuncs::dtype_to_type_t<dt>;
		if(a.dims() == 2 && a.dims() == b.dims())
			return handle_dim2_matmult_std<type_t>(a, b,c, trans_a, trans_b);
		if(a.dims() > b.dims())
			return handle_dim_n_k_matmult_std<type_t>(a,b,c,trans_a,trans_b);
		if(b.dims() > a.dims())
			return handle_dim_k_n_matmult_std<type_t>(a,b,c,trans_a,trans_b);
		return handle_dim_n_n_matmult_std<type_t>(a,b,c,trans_a,trans_b);
	}
	else if (dt == DType::TensorObj){
		DType sub_dt = subT_check(a, b); //just a list of things to check if the errors
		return handle_tensor_tensor_matrix_multiplication_std(a, b, c, trans_a, trans_b, sub_dt);
		
	}
	else{
		utils::THROW_EXCEPTION(a.dtype != DType::Bool, "Expected dtype to be numerical, or tensors, but got dtype $", a.dtype);
		return c;
	}


}



Tensor matmult(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){
	return handle_matrix_multiplication_std<DType::Integer>(a, b, trans_a, trans_b);
}

Tensor& matmult(const Tensor& a, const Tensor& b, Tensor& c, bool trans_a, bool trans_b){
    return handle_matrix_multiplication_std<DType::Integer>(a, b, c, trans_a, trans_b);
}

Tensor linear(const Tensor& x, const Tensor& w, const Tensor& b, bool trans_a, bool trans_b){
    if(x.dims() == w.dims() && x.dims() == b.dims()){
        utils::throw_exception(x.dims() >= 2, "Cannot perform matrix multiplication with dimensions less than 2 got $ for tensor dimensions", x.dims());
        const int64_t& out_rows = trans_a ? x.shape()[-1] : x.shape()[-2];
        const int64_t& out_cols = trans_b ? w.shape()[-2] : w.shape()[-1];
        utils::THROW_EXCEPTION(b.shape()[-2] == out_rows || b.shape()[-2] == 1,
                               "Got invalid shape for bias ($) to linear function expected rows to be 1 or $", 
                               b.shape(), out_rows);
        utils::THROW_EXCEPTION(b.shape()[-1] == out_cols || b.shape()[-1] == 1,
                               "Got invalid shape for bias ($) to linear function expected cols to be 1 or $",
                               b.shape(), out_cols);
        bool b_matmult_shape = (trans_a ? b.shape()[-2] == x.shape()[-1] : b.shape()[-2] == x.shape()[-2])
                                && (trans_b ? b.shape()[-1] == w.shape()[-2] : b.shape()[-1] == w.shape()[-1]); 
        if(x.dims() == 2){
            if(b_matmult_shape){
                Tensor c = b.clone();
                matmult(x, w, c, trans_a, trans_b);
                return std::move(c);
            }
            else{
                Tensor c = b.expand({out_rows, out_cols}).clone();
                matmult(x, w, c, trans_a, trans_b);
                return std::move(c);
            }
        }
        const int64_t x_mat_size = x.shape()[-1] * x.shape()[-2];
        const int64_t w_mat_size = w.shape()[-1] * w.shape()[-2];
        const int64_t b_mat_size = b.shape()[-1] * b.shape()[-2];

        const int64_t& x_numel = x.numel();
        const int64_t& w_numel = w.numel();
        const int64_t& b_numel = b.numel();
        SizeRef x_batch_shape = x.shape().range(0, -2);
        SizeRef w_batch_shape = w.shape().range(0, -2);
        SizeRef b_batch_shape = b.shape().range(0, -2);
        
        
        
        if(x_batch_shape == w_batch_shape && x_batch_shape == b_batch_shape){
            if(b_matmult_shape){
                Tensor c = b.clone();
                matmult(x, w, c, trans_a, trans_b);
                return std::move(c);
            }
        }
        if(x_batch_shape == w_batch_shape){
            auto vec = x_batch_shape.Vec();
            vec.push_back(out_rows);
            vec.push_back(out_cols);
            // std::cout << "[1] Expanding b as " << SizeRef(vec) << std::endl;
            Tensor c = b.expand_as(SizeRef(vec));
            c = (c.is_contiguous()) ? c.clone() : c.contiguous();
            matmult(x, w, c, trans_a, trans_b);
            return std::move(c);
        }
        if(x_batch_shape == b_batch_shape
            && b_matmult_shape){
            SizeRef WShape = w.shape().clone();
            utils::throw_exception(WShape[0] == 1, "Error, invalid batch shapes for linear: ($, $, $)", x.shape(), w.shape(), b.shape());
            while(WShape[0] == 1 && WShape.size() > 2){WShape = WShape.pop_front();}
            Tensor c = b.clone();
            Tensor nw = w.view(WShape);
            matmult(x, nw, c, trans_a, trans_b);
            return std::move(c);
        }
        if(w_batch_shape == b_batch_shape
            && b_matmult_shape){
            SizeRef XShape = x.shape().clone();
            utils::throw_exception(XShape[0] == 1, "Error, invalid batch shapes for linear: ($, $, $)", x.shape(), w.shape(), b.shape());
            while(XShape[0] == 1 && XShape.size() > 2){XShape = XShape.pop_front();}
            Tensor c = b.clone();
            Tensor nx = x.view(XShape);
            matmult(nx, w, c, trans_a, trans_b);
            return std::move(c);
        }
        SizeRef XShape = x.shape().clone();
        SizeRef WShape = w.shape().clone();
        while(XShape[0] == 1 && XShape.size() > 2){XShape = XShape.pop_front();}
        while(WShape[0] == 1 && WShape.size() > 2){WShape = WShape.pop_front();}
        utils::throw_exception(XShape.size() != WShape.size(), "Error, invalid batch shapes for x and w for linear: ($, $, $)", x.shape(), w.shape(), b.shape());
        auto b_vec = XShape.size() > WShape.size() ? XShape.Vec() : WShape.Vec();
        b_vec.back() = out_cols;
        b_vec[b_vec.size()-2] = out_rows;
        Tensor c = b.expand_as(SizeRef(b_vec));
        c = (c.is_contiguous()) ? c.clone() : c.contiguous();
        matmult(x.view(XShape), w.view(WShape), c, trans_a, trans_b);
        return std::move(c);
    }
    const SizeRef& max_dim = (x.dims() > w.dims() ? (x.dims() > b.dims() ? x.shape() : b.shape()) : (w.dims() > b.dims() ? w.shape() : b.shape()));
    return linear(x.unsqueeze_as(max_dim), w.unsqueeze_as(max_dim), b.unsqueeze_as(max_dim), trans_a, trans_b);
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
