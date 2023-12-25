#include <cstdint>
#include <sys/_types/_size_t.h>
#include "../Tensor.h"
#include "../dtype/DType_list.h"
#include "../refs/SizeRef.h"
#include "../dtype/ArrayVoid.h"
#include "../dtype/DType.h"
#include "../dtype/DType_enum.h"
#include <_types/_uint32_t.h>
#include <_types/_uint16_t.h>
#include <_types/_uint32_t.h>
#include <_types/_uint8_t.h>
#include <atomic>
#include <functional>
#include <i386/types.h>
#include <memory.h>
#include <algorithm>
#include <numeric>
#include <ratio>
#include <sys/_types/_int8_t.h>
#include <cassert>
#include <format>
#include <sys/types.h>
#include <type_traits>
#include <vector>
#include "../utils/utils.h"
#include <chrono>
#include "../permute/permute_old.h"
#include "functional.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include "../dtype/ArrayVoid.hpp"
#include "functional_operator.h"

#ifdef USE_PARALLEL
	#include <tbb/parallel_for_each.h>
	#include <tbb/parallel_for.h>
#endif
#define assertm(exp, msg) assert(((void)msg, exp))


namespace nt{
namespace functional{

void exception_dtypes(const DType& a, const DType& b){
	utils::throw_exception(a == b, "\nRuntimeError: Expected dtype of second tensor to be $ but got $", a, b);
}

void exception_shapes(const SizeRef& a, const SizeRef& b, bool singletons=false){
	if(!singletons && a != b){
		utils::throw_exception(a == b, "\nRuntimeError: Expected shape of second tensor to be $ but got $", a, b);
	}
	if(a != b){
		if(a.size() > b.size()){
			uint32_t start = a.size() - b.size();
			for(uint32_t i = a.size() - b.size(); i < a.size(); ++i){
				if(a[i] != b[i - start] && (b[i - start] != 1 || a[i] != 1)){
					utils::throw_exception(b[i - start] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i], b[i - start], i);
					utils::throw_exception(a[i] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i], b[i - start], i);
				}
			}
		}
		else if(b.size() > a.size()){
			uint32_t start = b.size() - a.size();
			for(uint32_t i = b.size() - a.size(); i < b.size(); ++i){
				if(a[i - start] != b[i] && (b[i] != 1 || a[i - start] != 1)){
					utils::throw_exception(b[i] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i - start], b[i], i);
					utils::throw_exception(a[i - start] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i - start], b[i], i);

				}
			}
		}
		else{
			for(uint32_t i = 0; i < b.size(); ++i){
				if(a[i] != b[i] && (b[i] != 1 || a[i] != 1)){
					utils::throw_exception(b[i] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i], b[i], i);
					utils::throw_exception(a[i] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i], b[i], i);

				}
			}
		}

	}
}



Tensor zeros(SizeRef inp, DType dt){
	uint32_t total_size = inp.multiply();
	Tensor output(std::move(inp), dt);
	output._fill(0);    
	return std::move(output);
}

Tensor nums(SizeRef inp, const float& k, DType dt){
	uint32_t total_size = inp.multiply();
	Tensor output(std::move(inp), dt);
	output._fill(k);
	return std::move(output);
}

Tensor arange(uint32_t total_size, DType dt){
	Tensor output(total_size, dt);
	output.arr_void().iota(0);
	return std::move(output);
}

Tensor arange(SizeRef total_size, DType dt){
	Tensor output(std::move(total_size), dt);
	output.arr_void().iota(0);
	return std::move(output);
}

Tensor randint(int32_t lower, int32_t upper, SizeRef s, DType dt){
	Tensor output(std::move(s), dt);
	srand(time(NULL));
	output.arr_void().for_each_nbool([&lower, &upper](auto& val){val = (rand() % (upper - lower + 1)) + lower;});
	return std::move(output);

}

Tensor randn(SizeRef inp, DType dt){
	Tensor output = randint(0, 20, std::move(inp), dt);
	softmax_(output);
	return std::move(output);
}



inline static constexpr auto transposed_matmult = [](auto a_begin, auto a_end, auto b_begin, void* o_ptr, const uint32_t& rows, const uint32_t& cols, const uint32_t& inter, const uint32_t& m_st){
	typedef typename std::remove_const<typename decltype(a_begin)::value_type>::type value_t;
	value_t initial(0);
	value_t* o_begin = reinterpret_cast<value_t*>(o_ptr) + m_st;
	auto b_copy = b_begin;
	auto b_end = b_begin + (cols * inter);
	uint32_t mn1 = rows-1;
	uint32_t i = 0;
	for(;a_begin != a_end; a_begin += inter){
		for(;b_begin != b_end; b_begin += inter, ++o_begin){
			*o_begin = std::inner_product(a_begin, a_begin + inter, b_begin, initial);
		}
		if(i == mn1){
			i = 0;
			b_copy = b_begin;
			b_end += (cols * inter);
		}
		else{
			++i;
			b_begin = b_copy;
		}
	}
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
			uint32_t start = a.dims() - b.dims();
			for(uint32_t i = start; i < size_outp.size()-2; ++i){
				utils::throw_exception(size_outp[i] == b.shape()[i - start], "RuntimeError: Expected sizes at dimension $ to be $ but got $", i-start, size_outp[i], b.shape()[i-start]); 
			}
			Tensor output = zeros(SizeRef(std::move(size_outp)), a.dtype);
			const Tensor a_1 = a.split_axis(start-1);
			const Tensor* a1_begin = reinterpret_cast<const Tensor*>(a_1.data_ptr());
			const Tensor* a1_end = a1_begin + a_1.numel();
			b.RowColSwap();
			uint32_t multiply = output.shape().multiply(start);
			uint32_t i = 0;
			for(;a1_begin != a1_end; ++a1_begin, i += multiply){
				a1_begin->arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(transposed_matmult, b.arr_void(), output.data_ptr(), output.shape()[-2], output.shape().back(), a.shape().back(), i);
			}
			if(un_transpose){b.RowColSwap();}
			return std::move(output);
		}
		//b.dims() > a.dims()
		else{
			std::vector<uint32_t> size_outp = b.shape().Vec();
			size_outp[size_outp.size()-2] = a.shape()[-2];
			uint32_t start = b.dims() - a.dims();
			for(uint32_t i = start; i < size_outp.size() - 2; ++i)
				utils::throw_exception(size_outp[i] == a.shape()[i - start], "RuntimeError: Expected sizes at dimension $ to be $ but got $", i-start, size_outp[i], a.shape()[i-start]);
			Tensor output = zeros(SizeRef(std::move(size_outp)), a.dtype);
			b.RowColSwap();
			const Tensor b_1 = b.split_axis(start-1);
			const Tensor* b1_begin = reinterpret_cast<const Tensor*>(b_1.data_ptr());
			const Tensor* b1_end = b1_begin + b_1.numel();
			uint32_t multiply = output.shape().multiply(start);
			uint32_t i = 0;
			for(;b1_begin != b1_end; ++b1_begin, i += multiply){
				a.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(transposed_matmult, b1_begin->arr_void(), output.data_ptr(), output.shape()[-2], output.shape().back(), a.shape().back(), i);
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
	a.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(transposed_matmult, b.arr_void(), output.data_ptr(), output.shape()[-2], output.shape().back(), a.shape().back(), 0);
	if(un_transpose){b.RowColSwap();}
	return std::move(output);

}


Tensor cat_0_dim(const Tensor& _a, const Tensor& _b){
	::nt::utils::throw_exception(_a.dims() == _b.dims(), "\nDims must be equal $ != $", _a.dims(), _b.dims());
	::nt::utils::throw_exception(_a.dtype == _b.dtype, "\nDTypes must be equal $ != $", _a.dtype, _b.dtype);
	for(uint32_t i = 1; i < _a.dims(); ++i)
		utils::throw_exception(_a.shape()[i] == _b.shape()[i],
				"\nRuntimeError: Sizes of tensors must match except in dimension 0. Expected size $ but got size $ for second tensor.", _a.shape()[i], _b.shape()[i]);
	std::vector<typename SizeRef::ArrayRefInt::value_type> n_vals = _a.shape().Vec();
	n_vals[0] += _b.shape()[0];
	Tensor output(SizeRef(std::move(n_vals)), _a.dtype);
	_a.arr_void().copy(output.arr_void());
	_b.arr_void().copy(output.arr_void(), _a.numel());
	return std::move(output);
}


inline static constexpr auto cat_arr_void_perms = [](auto begin_a, auto end_a, ArrayVoid& output, permute::PermIndexItND& a_begin, permute::PermIndexItND& a_end, permute::PermIndexItND& o_begin){
	typedef typename std::remove_const<typename decltype(begin_a)::value_type>::type value_t;
	tdtype_list<value_t> begin_o = output.tbegin<value_t>();
	for(;a_begin != a_end; ++a_begin, ++o_begin)
		begin_o[*o_begin] = begin_a[*a_begin];
};

Tensor cat_permute_dim(const Tensor& _a, const Tensor& _b, int8_t dim){
	::nt::utils::throw_exception(_a.dims() == _b.dims(), "\nDims must be equal $ != $", _a.dims(), _b.dims());
	::nt::utils::throw_exception(_a.dtype == _b.dtype, "\nDTypes must be equal $ != $", _a.dtype, _b.dtype);
	for(uint32_t i = 0; i < _a.dims(); ++i){
		if(i == dim) continue;
		utils::throw_exception(_a.shape()[i] == _b.shape()[i],
				"\nRuntimeError: Sizes of tensors must match except in dimension $. Expected size $ but got size $ for second tensor.", i, _a.shape()[i], _b.shape()[i]);
	}
	
	std::vector<uint32_t> _a_strides(_a.strides().cbegin() + 1, _a.strides().cend());
	std::vector<uint32_t> _b_strides(_b.strides().cbegin() + 1, _b.strides().cend());
	std::vector<uint32_t> _a_shape = _a.shape().Vec();
	std::vector<uint32_t> _b_shape = _b.shape().Vec();
	std::vector<uint32_t> _o_shape = _a_shape;
	_o_shape[dim] += _b_shape[dim];
	Tensor outp(SizeRef(_o_shape), _a.dtype);
	std::vector<uint32_t> _o_strides(outp.strides().cbegin() + 1, outp.strides().cend());
	std::swap(_a_shape[0], _a_shape[dim]);	
	std::swap(_a_strides[0], _a_strides[dim]);	
	std::swap(_b_shape[0], _b_shape[dim]);	
	std::swap(_b_strides[0], _b_strides[dim]);	
	std::swap(_o_shape[0], _o_shape[dim]);	
	std::swap(_o_strides[0], _o_strides[dim]);
	std::unique_ptr<permute::PermND> o_perm = permute::create_perm(_o_strides, _o_shape);
	std::unique_ptr<permute::PermND> a_perm = permute::create_perm(_a_strides, _a_shape);
	std::unique_ptr<permute::PermND> b_perm = permute::create_perm(_b_strides, _b_shape);
	std::shared_ptr<permute::PermIndexItND> o_begin = o_perm->begin();
	std::shared_ptr<permute::PermIndexItND> a_begin = a_perm->begin();
	std::shared_ptr<permute::PermIndexItND> b_begin = b_perm->begin();
	std::shared_ptr<permute::PermIndexItND> a_end = a_perm->end();
	std::shared_ptr<permute::PermIndexItND> b_end = b_perm->end();
	_a.arr_void().cexecute_function(cat_arr_void_perms, outp.arr_void(), *a_begin, *a_end, *o_begin);
	_b.arr_void().cexecute_function(cat_arr_void_perms, outp.arr_void(), *b_begin, *b_end, *o_begin);
	return outp;
}

Tensor cat(const Tensor& _a, const Tensor& _b, int8_t dim){
	return (dim == 0) ? cat_0_dim(_a, _b) : cat_permute_dim(_a, _b, dim);
}

Tensor hadamard_multiply(const Tensor& a, const Tensor& b){return functional_operator_out(a, b, functional_operator_num::Multiply);}
Tensor& hadamard_multiply_this(Tensor &a, const Tensor &b){functional_operator_this(a, b, functional_operator_num::Multiply); return a;}

Tensor add(const Tensor& a, const Tensor& b){return functional_operator_out(a, b, functional_operator_num::Add);}
Tensor& add_(Tensor &a, const Tensor &b){functional_operator_this(a, b, functional_operator_num::Add); return a;}

Tensor subtract(const Tensor& a, const Tensor& b){return functional_operator_out(a, b, functional_operator_num::Subtract);}
Tensor& subtract_(Tensor &a, const Tensor &b){functional_operator_this(a, b, functional_operator_num::Subtract); return a;}

Tensor divide(const Tensor& a, const Tensor& b){return functional_operator_out(a, b, functional_operator_num::Divide);}
Tensor& divide_(Tensor &a, const Tensor &b){functional_operator_this(a, b, functional_operator_num::Divide); return a;}


bool all(const Tensor &t){
	exception_dtypes(t.dtype, DType::Bool);
	auto begin = t.arr_void().tcbegin<uint_bool_t>();
	auto end = t.arr_void().tcend<uint_bool_t>();
	return std::all_of(begin, end, [](const uint_bool_t &v){return v.value == 1;});
}

#ifdef USE_PARALLEL

inline static constexpr auto do_conv2d = [](auto image_begin, auto image_end, auto kernel_begin, ArrayVoid& outp, const uint32_t* starts, const uint32_t* ends, const uint32_t* k_starts, const uint32_t& outside_itter, const SizeRef& i_size, const SizeRef& k_size, const SizeRef& o_size){
	typedef typename std::remove_const<typename decltype(image_begin)::value_type>::type value_t;
	const uint32_t& image_itterate_4d = i_size.multiply(1);
	const uint32_t& kernel_itterate_4d = k_size.multiply(1);
	const uint32_t& outp_itterate_4d = o_size.multiply(1);
	const uint32_t& kernel_itterate_3d = k_size.multiply(2);
	const uint32_t& outp_itterate_3d = o_size.multiply(2);
	const uint32_t& image_itterate_3d = i_size.multiply(2);
	tbb::parallel_for(tbb::blocked_range<uint32_t>(0, i_size[0], 10),
			[&](tbb::blocked_range<uint32_t> a){
	auto o_begin = outp.tbegin<value_t>() + (a.begin() * outp_itterate_4d);
	auto cur_img_begin = image_begin + (a.begin() * image_itterate_4d);
	for(uint32_t i = a.begin(); i < a.end(); ++i, o_begin += outp_itterate_4d, cur_img_begin += image_itterate_4d){
		tbb::parallel_for(tbb::blocked_range<uint32_t>(0, k_size[0], 10),
				[&](tbb::blocked_range<uint32_t> b){
		
		auto kernel_copy = kernel_begin + (b.begin() * kernel_itterate_4d);
		auto o_copy = o_begin + (b.begin() * outp_itterate_3d);
		for(uint32_t j = b.begin(); j < b.end(); ++j, o_copy += outp_itterate_3d, kernel_copy += kernel_itterate_4d){
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, image_itterate_4d, 10),
					[&](tbb::blocked_range<uint32_t> c){
					
			auto img_copy = cur_img_begin + (c.begin() * image_itterate_3d);
			auto kern_copy = kernel_copy + (c.begin() * kernel_itterate_3d);
			uint32_t counter2 = 0;
			for(uint32_t k = c.begin(); k < c.end(); ++k, img_copy += image_itterate_3d, kern_copy += kernel_itterate_3d){
				for(uint32_t counter = 0; counter < outside_itter; ++counter){
					for(uint32_t l = 0; l < k_size[-1]; ++l, ++counter2){
						*(o_copy + l) = std::inner_product(img_copy + starts[counter2], img_copy + ends[counter2], kern_copy + k_starts[l], *(o_copy + l));
					}
				}
			}
			});
		}
		
		});

	}

	});
};





#else

inline static constexpr auto do_conv2d = [](auto image_begin, auto image_end, auto kernel_begin, ArrayVoid& outp, const uint32_t* starts, const uint32_t* ends, const uint32_t* k_starts, const uint32_t& outside_itter, const SizeRef& i_size, const SizeRef& k_size, const SizeRef& o_size){
	typedef typename std::remove_const<typename decltype(image_begin)::value_type>::type value_t;
	const uint32_t& image_itterate_4d = i_size.multiply(1);
	const uint32_t& kernel_itterate_4d = k_size.multiply(1);
	const uint32_t& outp_itterate_4d = o_size.multiply(1);
	const uint32_t& kernel_itterate_3d = k_size.multiply(2);
	const uint32_t& outp_itterate_3d = o_size.multiply(2);
	const uint32_t& image_itterate_3d = i_size.multiply(2);
	std::cout << "finished multiplication" << std::endl;
	auto kernel_end = kernel_begin + k_size.multiply();
	auto o_begin = outp.tbegin<value_t>();
	for(;image_begin != image_end; image_begin += image_itterate_4d, o_begin += outp_itterate_4d){
		auto kernel_copy = kernel_begin;
		auto o_copy = o_begin;
		for(;kernel_copy != kernel_end; kernel_copy += kernel_itterate_4d, o_copy += outp_itterate_3d){
			auto img_copy = image_begin;
			auto kern_copy = kernel_copy;
			auto img_copy_end = img_copy + image_itterate_4d;
			for(;img_copy != img_copy_end; img_copy += image_itterate_3d, kern_copy += kernel_itterate_3d){
				//cplus = _vals.get() + i
				uint32_t counter2 = 0;
				for(uint32_t counter = 0; counter < outside_itter; ++counter){
					for(uint32_t i = 0; i < k_size[-2]; ++i, ++counter2){
						*(o_copy + i) = std::inner_product(img_copy + starts[counter2], (img_copy + ends[counter2]), (kern_copy + k_starts[i]), *(o_copy + i));
					}
				}
			}
		}
	}

};

#endif


Tensor conv2d(const Tensor &image, const Tensor &kernel, const uint32_t s_h, const uint32_t s_w){
	utils::throw_exception(image.dims() == 4,"Expected Image dims to be $ but got $", 4, image.dims());
	utils::throw_exception(kernel.dims() == 4,"Expected Kernel dims to be $ but got $", 4, kernel.dims());
	utils::throw_exception(kernel.shape()[-3] == image.shape()[-3],"Image and kernel at dim 3 must be equal, expected kernel to have $ but had $", image.shape()[-3], kernel.shape()[-3]);
	exception_dtypes(image.dtype, kernel.dtype);
	uint32_t row_itters = (image.shape()[-2] - kernel.shape()[-2])/(float)s_h + 1;
	uint32_t col_itters = (image.shape()[-1] - kernel.shape()[-1])/(float)s_w +1;
	uint32_t row_barrier = (image.shape()[-2]-kernel.shape()[-2]) + 1;
	uint32_t col_barrier = (image.shape()[-1]-kernel.shape()[-1]) + 1;
	uint32_t outside_itter = row_itters * col_itters;
	uint32_t* starts = new uint32_t[kernel.shape()[-2] * outside_itter];
	uint32_t* ends = new uint32_t[kernel.shape()[-2] * outside_itter];
	uint32_t* k_starts = new uint32_t[kernel.shape()[-2]];	
	uint32_t current = 0; // current index
	uint32_t total_count = 0;
	uint32_t last_c = 0;
	uint32_t last_r = 0;
	uint32_t row = 0;
	uint32_t col = 0;
	uint32_t counter = 0;
	for(uint32_t i = 0; i < kernel.shape()[-2]; ++i){
		k_starts[i] = kernel.shape()[-1] * i;
	}
	for(counter = 0; row < row_barrier; ++counter){
		current = last_r + col;
		for(uint32_t i = 0; i < kernel.shape()[-2]; ++i){
			starts[total_count] = last_c + current;
			ends[total_count] = starts[total_count] + kernel.shape()[-1];
			last_c += image.shape()[-1];
			++total_count;
		}
		last_c = 0;
		if((col += s_w) >= col_barrier){
			col = 0;
			row += s_h;
			last_r = (row*image.shape()[-1]);
		}
	}
	Tensor outp(SizeRef({image.shape()[-4], kernel.shape()[-4], row_itters, col_itters}), image.dtype);
	//do conv loop
	std::cout << "executing" << std::endl;
	image.arr_void().cexecute_function_nbool(do_conv2d, kernel.arr_void(), outp.arr_void(), starts, ends, k_starts, outside_itter, image.shape(), kernel.shape(), outp.shape());
	std::cout << "finished executing" << std::endl;
	delete[] starts;
	delete[] ends;
	delete[] k_starts;
	return std::move(outp);
}


/* Tensor conv2dT(const Tensor &image, const Tensor &kernel, const uint32_t s_h, const uint32_t s_w){ */
/* 	utils::throw_exception(image.dims() == 4,"Expected Image dims to be $ but got $", 4, image.dims()); */
/* 	utils::throw_exception(kernel.dims() == 4,"Expected Kernel dims to be $ but got $", 4, kernel.dims()); */
/* 	utils::throw_exception(kernel.shape()[-3] == image.shape()[-3],"Image and kernel at dim 3 must be equal, expected kernel to have $ but had $", image.shape()[-3], kernel.shape()[-3]); */
/* 	exception_dtypes(image.dtype, kernel.dtype); */
	
/* } */

//this applies a softmax function over the entire inputted tensor
void softmax_(Tensor& inp){
	inp.exp_();
	Scalar element = inp.sum().toScalar();
	inp._divide(element);
}

void softmax_(Tensor& inp, uint32_t dim){
	inp.exp_();
	Tensor tensors = inp.split_axis(dim);
	tensors.arr_void().for_each<DType::TensorObj>([](auto& val){
				Scalar element = val.sum().toScalar();
				val._divide(element);
			});
}

Tensor softmax(const Tensor& inp){
	Tensor outp = inp.exp();
	Scalar element = outp.sum().toScalar();
	outp._divide(element);
	return std::move(outp);
}

Tensor softmax(const Tensor& inp, uint32_t dim){
	Tensor outp = inp.exp();
	Tensor tensors = outp.split_axis(dim);
	tensors.arr_void().for_each<DType::TensorObj>([](auto& val){
				Scalar element = val.sum().toScalar();
				val._divide(element);
			});
	return std::move(outp);
}

void softmax_stable_(Tensor& inp){
	Scalar max = inp.max().toScalar();
	inp.exp_();
	inp -= max;
	Scalar element = inp.sum().toScalar();
	inp._divide(element);
}

void softmax_stable_(Tensor& inp, uint32_t dim){
	Tensor tensors = inp.split_axis(dim);
	tensors.arr_void().for_each<DType::TensorObj>([](auto& val){
				Scalar max = val.max().toScalar();
				val.exp_();
				val -= max;
				Scalar element = val.sum().toScalar();
				val._divide(element);
			});
}

Tensor softmax_stable(const Tensor& inp){
	Scalar max = inp.max().toScalar();
	Tensor outp = inp.exp() - max;
	Scalar element = outp.sum().toScalar();
	outp._divide(element);
	return std::move(outp);
}


Tensor softmax_stable(const Tensor& inp, uint32_t dim){
	Tensor outp(inp);
	Tensor tensors = outp.split_axis(dim);
	tensors.arr_void().for_each<DType::TensorObj>([](auto& val){
				Scalar max = val.max().toScalar();
				val.exp_();
				val -= max;
				Scalar element = val.sum().toScalar();
				val._divide(element);
			});
	return std::move(outp);

}

Tensor cat(std::vector<Tensor> t){
	uint32_t last_dim = 0;
	for(uint32_t i = 1; i < t.size(); ++i){
		exception_dtypes(t[i-1].dtype, t[i].dtype);
		utils::throw_exception(t[i-1].dims() == t[i].dims(), "Runtime Error: Expected all tensors to have $ dims but got $ instead", t[i-1].dims(), t[i].dims());
		for(uint32_t j = 1; j < t[i].dims(); ++j){
			utils::throw_exception(t[i].shape()[j] == t[i-1].shape()[j],
				"Runtime Error: Expected tensors to have same shape ($) at dim $ but got ($) instead",
				t[i-1].shape()[j], j, t[i].shape()[j]);
		}
		last_dim += t[i-1].shape()[0];
	}
	last_dim += t.back().shape()[0];
	std::vector<uint32_t> vec = t[0].shape().Vec();
	vec[0] = last_dim;
	SizeRef a(std::move(vec));
	Tensor outp(a, t[0].dtype);
	std::vector<nt::my_range> ranges(1);
	uint32_t last = 0;
	for(uint32_t i = 0; i < t.size(); ++i){
		ranges.back() = nt::my_range(last, t[i].shape()[0]);
		last += t[i].shape()[0];
		outp[ranges] = t[i];
	}
	return outp;
}

Tensor cat(std::vector<Tensor> t, int32_t dim){
	dim = dim < 0 ? t[0].dims() + dim : dim;
	uint32_t last_dim = 0;
	for(uint32_t i = 1; i < t.size(); ++i){
		exception_dtypes(t[i-1].dtype, t[i].dtype);
		utils::throw_exception(t[i-1].dims() == t[i].dims(), "Runtime Error: Expected all tensors to have $ dims but got $ instead", t[i-1].dims(), t[i].dims());
		for(uint32_t j = 0; j < t[i].dims(); ++j){
			if(j == dim){
				last_dim += t[i-1].shape()[j];
				continue;
			}
			utils::throw_exception(t[i].shape()[j] == t[i-1].shape()[j],
				"Runtime Error: Expected tensors to have same shape ($) at dim $ but got ($) instead",
				t[i-1].shape()[j], j, t[i].shape()[j]);
		}
	}
	last_dim += t.back().shape()[dim];
	std::vector<uint32_t> vec = t[0].shape().Vec();
	vec[dim] = last_dim;
	SizeRef a(std::move(vec));
	Tensor outp(a, t[0].dtype);
	std::vector<nt::my_range> ranges(dim+1);
	for(uint32_t i = 0; i < ranges.size()-1; ++i){
		ranges[i] = nt::my_range(0, t[0].shape()[i]);
	}
	uint32_t last = 0;
	for(uint32_t i = 0; i < t.size(); ++i){
		ranges.back() = nt::my_range(last, t[i].shape()[dim]);
		last += t[i].shape()[dim];
		outp[ranges] = t[i];
	}
	return outp;
}


Tensor stack(std::vector<Tensor> t){
	for(uint32_t i = 1; i < t.size(); ++i){
		utils::throw_exception(t[i-1].shape() == t[i].shape(),
				"Runtime Error: Expected all Tensors to have same shape of $ but instead got", t[i-1].shape(), t[i].shape());
		exception_dtypes(t[i-1].dtype, t[i].dtype);
	}
	std::vector<uint32_t> vec = t[0].shape().Vec();
	vec.insert(vec.begin(), t.size());
	SizeRef a(std::move(vec));
	Tensor output(std::move(a), t[0].dtype);
	for(uint32_t i = 0; i < t.size(); ++i)
		output[i] = t[i];
	return output;
}

Tensor stack(std::vector<Tensor> t, int8_t dim){
	for(uint32_t i = 1; i < t.size(); ++i){
		utils::throw_exception(t[i-1].shape() == t[i].shape(),
				"Runtime Error: Expected all Tensors to have same shape of $ but instead got", t[i-1].shape(), t[i].shape());
		exception_dtypes(t[i-1].dtype, t[i].dtype);
	}
	std::vector<uint32_t> vec = t[0].shape().Vec();
	dim = dim < 0 ? dim + t[0].dims() : dim;
	vec.insert(vec.begin() + dim, t.size());
	SizeRef a(std::move(vec));
	Tensor output(std::move(a), t[0].dtype);
	Tensor trans = output.transpose(0, dim);
	for(uint32_t i = 0; i < t.size(); ++i)
		trans[i] = t[i];
	return output;

}

}
}
