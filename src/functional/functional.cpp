#include <cstdint>

#include "../Tensor.h"
#include "../memory/iterator.h"
#include "../refs/SizeRef.h"
#include "../dtype/ArrayVoid.h"
#include "../dtype/DType.h"
#include "../dtype/DType_enum.h"
#include "../mp/simde_ops.h"




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
#include "../permute/permute_old.h"
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

#ifdef USE_PARALLEL
	#include <tbb/parallel_for_each.h>
	#include <tbb/parallel_for.h>
	#include <tbb/blocked_range.h>
	#include <tbb/blocked_range2d.h>
	#include <tbb/blocked_range3d.h>
	#include <tbb/parallel_reduce.h>
	#include <thread>
	/* #include "../mp/MP.hpp" */
	/* #include "../mp/Pool.hpp" */
#endif
#define assertm(exp, msg) assert(((void)msg, exp))


namespace nt{
namespace functional{

void exception_dtypes(const DType& a, const DType& b){
	utils::THROW_EXCEPTION(a == b, "\nRuntimeError: Expected dtype of second tensor to be $ but got $", a, b);
}

void exception_shapes(const SizeRef& a, const SizeRef& b, bool singletons=false){
	if(!singletons && a != b){
		utils::THROW_EXCEPTION(a == b, "\nRuntimeError: Expected shape of second tensor to be $ but got $", a, b);
	}
	if(a != b){
		if(a.size() > b.size()){
			typename SizeRef::value_type start = a.size() - b.size();
			for(typename SizeRef::value_type i = a.size() - b.size(); i < a.size(); ++i){
				if(a[i] != b[i - start] && (b[i - start] != 1 || a[i] != 1)){
					utils::THROW_EXCEPTION(b[i - start] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i], b[i - start], i);
					utils::THROW_EXCEPTION(a[i] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i], b[i - start], i);
				}
			}
		}
		else if(b.size() > a.size()){
			typename SizeRef::value_type start = b.size() - a.size();
			for(typename SizeRef::value_type i = b.size() - a.size(); i < b.size(); ++i){
				if(a[i - start] != b[i] && (b[i] != 1 || a[i - start] != 1)){
					utils::THROW_EXCEPTION(b[i] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i - start], b[i], i);
					utils::THROW_EXCEPTION(a[i - start] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i - start], b[i], i);

				}
			}
		}
		else{
			for(typename SizeRef::value_type i = 0; i < b.size(); ++i){
				if(a[i] != b[i] && (b[i] != 1 || a[i] != 1)){
					utils::THROW_EXCEPTION(b[i] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i], b[i], i);
					utils::THROW_EXCEPTION(a[i] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i], b[i], i);

				}
			}
		}

	}
}



Tensor zeros(SizeRef inp, DType dt){
	typename SizeRef::value_type total_size = inp.multiply();
	Tensor output(std::move(inp), dt);
	/* std::memset(output.data_ptr(), 0, total_size * DTypeFuncs::size_of_dtype(dt)); */
	output.arr_void().fill_(0);    
	return std::move(output);
}

Tensor zeros_like(const Tensor& t){
	if(t.dtype != DType::TensorObj){
		return zeros(t.shape(), t.dtype);
	}
	Tensor out = Tensor::makeNullTensorArray(t.numel());
	out = out.view(t.shape());
	Tensor* begin_o = reinterpret_cast<Tensor*>(out.data_ptr());
	t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&begin_o](auto begin, auto end){
		for(;begin != end; ++begin, ++begin_o){*begin_o = zeros_like(*begin);}
	});
	return std::move(out);
}

Tensor ones(SizeRef inp, DType dt){
	typename SizeRef::value_type total_size = inp.multiply();
	Tensor output(std::move(inp), dt);
	/* std::memset(output.data_ptr(), 0, total_size * DTypeFuncs::size_of_dtype(dt)); */
	output.arr_void().fill_(1);    
	return std::move(output);
}

Tensor ones_like(const Tensor& t){
	if(t.dtype != DType::TensorObj){
		return ones(t.shape(), t.dtype);
	}
	Tensor out = Tensor::makeNullTensorArray(t.numel());
	out = out.view(t.shape());
	Tensor* begin_o = reinterpret_cast<Tensor*>(out.data_ptr());
	t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&begin_o](auto begin, auto end){
		for(;begin != end; ++begin, ++begin_o){*begin_o = ones_like(*begin);}
	});
	return std::move(out);
}


Tensor nums(SizeRef inp, const Scalar k, DType dt){
	Tensor output(std::move(inp), dt);
	output.arr_void().fill_(k);
	return std::move(output);
}

Tensor arange(typename SizeRef::value_type total_size, DType dt, Scalar start){
	Tensor output(total_size, dt);
	output.arr_void().iota(start);
	return std::move(output);
}

Tensor arange(SizeRef total_size, DType dt, Scalar start){
	Tensor output(std::move(total_size), dt);
	output.arr_void().iota(start);
	return std::move(output);
}


Tensor randint(Scalar lower, Scalar upper, SizeRef s, DType dt){
	Tensor output(std::move(s), dt);
	std::random_device rd;
	std::minstd_rand gen(rd()); //minimal version
	if(DTypeFuncs::is_unsigned(dt) || DTypeFuncs::is_integer(dt)){
		output.arr_void().execute_function<WRAP_DTYPES<IntegerTypesL>>(
			[&upper, &lower, &gen](auto begin, auto end){
				using value_t = utils::IteratorBaseType_t<decltype(begin)>;
#ifdef __SIZEOF_INT128__
				if constexpr(std::is_same_v<value_t, uint128_t>){
					uint64_t low = lower.to<int64_t>();
					uint64_t up = upper.to<int64_t>();
					std::uniform_int_distribution<uint64_t> dis(low, up);
					std::generate(begin, end, [&]() { return static_cast<value_t>(dis(gen)); });
				}
				else if(std::is_same_v<value_t, int128_t>){
					int64_t low = lower.to<int64_t>();
					int64_t up = upper.to<int64_t>();
					std::uniform_int_distribution<int64_t> dis(low, up);
					std::generate(begin, end, [&]() { return static_cast<value_t>(dis(gen)); });
				}
				else{
					value_t low = lower.to<value_t>();
					value_t up = upper.to<value_t>();
					std::uniform_int_distribution<value_t> dis(low, up);
					std::generate(begin, end, [&]() { return dis(gen); });
				}
#else
				value_t low = lower.to<value_t>();
				value_t up = upper.to<value_t>();
				std::uniform_int_distribution<value_t> dis(low, up);
				std::generate(begin, end, [&]() { return dis(gen); });
#endif
			});
		return std::move(output);
	}
	else if(DTypeFuncs::is_complex(dt)){
		output.arr_void().execute_function<WRAP_DTYPES<ComplexTypesL> >(
		[&upper, &lower, &gen](auto begin, auto end){
			using complex_t = utils::IteratorBaseType_t<decltype(begin)>;
			using value_t = typename complex_t::value_type;
#ifdef _HALF_FLOAT_SUPPORT_
			if constexpr (std::is_same_v<value_t, float16_t>){
				float low = lower.to<float>();
				float up = upper.to<float>();
				std::uniform_real_distribution<float> dis(low, up);
				std::generate(begin, end, [&]() {return complex_t(static_cast<value_t>(std::round(dis(gen))), static_cast<value_t>(std::round(dis(gen))));});
			}
			else{
				value_t low = lower.to<value_t>();
				value_t up = upper.to<value_t>();
				std::uniform_real_distribution<value_t> dis(low, up);
				std::generate(begin, end, [&]() {return complex_t(std::round(dis(gen)), std::round(dis(gen)));}); 
			}
#else
			value_t low = lower.to<value_t>();
			value_t up = upper.to<value_t>();
			std::uniform_real_distribution<value_t> dis(low, up);
			std::generate(begin, end, [&]() {return complex_t(std::round(dis(gen)), std::round(dis(gen)));});
#endif

		});
		return std::move(output);
	}
	else if(DTypeFuncs::is_floating(dt)){
		output.arr_void().execute_function<WRAP_DTYPES<FloatingTypesL> >(
		[&upper, &lower, &gen](auto begin, auto end){
			using value_t = utils::IteratorBaseType_t<decltype(begin)>;
#ifdef _HALF_FLOAT_SUPPORT_
			if constexpr (std::is_same_v<value_t, float16_t>){
				float low = lower.to<float>();
				float up = upper.to<float>();
				std::uniform_real_distribution<float> dis(low, up);
				std::generate(begin, end, [&]() {return static_cast<value_t>(std::round(dis(gen)));});
			}
			else{
				value_t low = lower.to<value_t>();
				value_t up = upper.to<value_t>();
				std::uniform_real_distribution<value_t> dis(low, up);
				std::generate(begin, end, [&]() {return std::round(dis(gen));}); 
			}
#else
			value_t low = lower.to<value_t>();
			value_t up = upper.to<value_t>();
			std::uniform_real_distribution<value_t> dis(low, up);
			std::generate(begin, end, [&]() {return std::round(dis(gen));});
#endif
		});
		return std::move(output);
	}
	return std::move(output);
}

Tensor rand(Scalar lower, Scalar upper, SizeRef s, DType dt){
	Tensor output(std::move(s), dt);
	std::random_device rd;
	std::minstd_rand gen(rd());
	if(DTypeFuncs::is_unsigned(dt) || DTypeFuncs::is_integer(dt)){
		output.arr_void().execute_function<WRAP_DTYPES<IntegerTypesL>>(
			[&upper, &lower, &gen](auto begin, auto end){
				using value_t = utils::IteratorBaseType_t<decltype(begin)>;
#ifdef __SIZEOF_INT128__
				if constexpr(std::is_same_v<value_t, uint128_t>){
					uint64_t low = lower.to<int64_t>();
					uint64_t up = upper.to<int64_t>();
					std::uniform_int_distribution<uint64_t> dis(low, up);
					std::generate(begin, end, [&]() { return static_cast<value_t>(dis(gen)); });
				}
				else if(std::is_same_v<value_t, int128_t>){
					int64_t low = lower.to<int64_t>();
					int64_t up = upper.to<int64_t>();
					std::uniform_int_distribution<int64_t> dis(low, up);
					std::generate(begin, end, [&]() { return static_cast<value_t>(dis(gen)); });
				}
				else{
					value_t low = lower.to<value_t>();
					value_t up = upper.to<value_t>();
					std::uniform_int_distribution<value_t> dis(low, up);
					std::generate(begin, end, [&]() { return dis(gen); });
				}
#else
				value_t low = lower.to<value_t>();
				value_t up = upper.to<value_t>();
				std::uniform_int_distribution<value_t> dis(low, up);
				std::generate(begin, end, [&]() { return dis(gen); });
#endif
			});
		return std::move(output);
	}
	else if(DTypeFuncs::is_complex(dt)){
		output.arr_void().execute_function<WRAP_DTYPES<ComplexTypesL> >(
		[&upper, &lower, &gen](auto begin, auto end){
			using complex_t = utils::IteratorBaseType_t<decltype(begin)>;
			using value_t = typename complex_t::value_type;
#ifdef _HALF_FLOAT_SUPPORT_
			if constexpr (std::is_same_v<value_t, float16_t>){
				float low = lower.to<float>();
				float up = upper.to<float>();
				std::uniform_real_distribution<float> dis(low, up);
				std::generate(begin, end, [&]() {return complex_t(static_cast<value_t>(dis(gen)), static_cast<value_t>(dis(gen)));});
			}
			else{
				value_t low = lower.to<value_t>();
				value_t up = upper.to<value_t>();
				std::uniform_real_distribution<value_t> dis(low, up);
				std::generate(begin, end, [&]() {return complex_t(dis(gen), dis(gen));}); 
			}
#else
			value_t low = lower.to<value_t>();
			value_t up = upper.to<value_t>();
			std::uniform_real_distribution<value_t> dis(low, up);
			std::generate(begin, end, [&]() {return complex_t(dis(gen), dis(gen));});
#endif

		});
		return std::move(output);
	}
	else if(DTypeFuncs::is_floating(dt)){
		output.arr_void().execute_function<WRAP_DTYPES<FloatingTypesL> >(
		[&upper, &lower, &gen](auto begin, auto end){
			using value_t = utils::IteratorBaseType_t<decltype(begin)>;
#ifdef _HALF_FLOAT_SUPPORT_
			if constexpr (std::is_same_v<value_t, float16_t>){
				float low = lower.to<float>();
				float up = upper.to<float>();
				std::uniform_real_distribution<float> dis(low, up);
				std::generate(begin, end, [&]() {return static_cast<value_t>(dis(gen));});
			}
			else{
				value_t low = lower.to<value_t>();
				value_t up = upper.to<value_t>();
				std::uniform_real_distribution<value_t> dis(low, up);
				std::generate(begin, end, [&]() {return dis(gen);}); 
			}
#else
			value_t low = lower.to<value_t>();
			value_t up = upper.to<value_t>();
			std::uniform_real_distribution<value_t> dis(low, up);
			std::generate(begin, end, [&]() {return dis(gen);});
#endif
		});
		return std::move(output);
	}
	return std::move(output);
}

Tensor randn(SizeRef inp, DType dt){
	Tensor output = randint(0, 20, std::move(inp), dt);
	softmax_(output);
	return std::move(output);
}



Tensor cat_0_dim(const Tensor& _a, const Tensor& _b){
	::nt::utils::THROW_EXCEPTION(_a.dims() == _b.dims(), "\nDims must be equal $ != $", _a.dims(), _b.dims());
	::nt::utils::THROW_EXCEPTION(_a.dtype == _b.dtype, "\nDTypes must be equal $ != $", _a.dtype, _b.dtype);
	for(typename SizeRef::value_type i = 1; i < _a.dims(); ++i)
		utils::THROW_EXCEPTION(_a.shape()[i] == _b.shape()[i],
				"\nRuntimeError: Sizes of tensors must match except in dimension 0. Expected size $ but got size $ for second tensor.", _a.shape()[i], _b.shape()[i]);
	std::vector<typename SizeRef::ArrayRefInt::value_type> n_vals = _a.shape().Vec();
	n_vals[0] += _b.shape()[0];
	return Tensor(ArrayVoid::cat(_a.arr_void(), _b.arr_void()), SizeRef(std::move(n_vals)));
}


inline static constexpr auto cat_arr_void_perms = [](auto begin_a, auto end_a, ArrayVoid& output, permute::PermIndexItND& a_begin, permute::PermIndexItND& a_end, permute::PermIndexItND& o_begin){
	using value_t = utils::IteratorBaseType_t<decltype(begin_a)>;
	typename SizeRef::value_type type = output.get_bucket().iterator_type();
	if(type == 1){
		auto begin_o = output.get_bucket().begin_contiguous<value_t>();
		for(;a_begin != a_end; ++a_begin, ++o_begin)
			begin_o[*o_begin] = begin_a[*a_begin];
	}
	else if(type == 2){
		auto begin_o = output.get_bucket().begin_blocked<value_t>();
		for(;a_begin != a_end; ++a_begin, ++o_begin)
			begin_o[*o_begin] = begin_a[*a_begin];
	}
	else if(type == 3){
		auto begin_o = output.get_bucket().begin_blocked<value_t>();
		for(;a_begin != a_end; ++a_begin, ++o_begin)
			begin_o[*o_begin] = begin_a[*a_begin];
	}
	
};

Tensor cat_permute_dim(const Tensor& _a, const Tensor& _b, int8_t dim){
	::nt::utils::THROW_EXCEPTION(_a.dims() == _b.dims(), "\nDims must be equal $ != $", _a.dims(), _b.dims());
	::nt::utils::THROW_EXCEPTION(_a.dtype == _b.dtype, "\nDTypes must be equal $ != $", _a.dtype, _b.dtype);
	for(typename SizeRef::value_type i = 0; i < _a.dims(); ++i){
		if(i == dim) continue;
		utils::THROW_EXCEPTION(_a.shape()[i] == _b.shape()[i],
				"\nRuntimeError: Sizes of tensors must match except in dimension $. Expected size $ but got size $ for second tensor.", i, _a.shape()[i], _b.shape()[i]);
	}
	
	std::vector<typename SizeRef::value_type> _a_strides(_a.strides().cbegin() + 1, _a.strides().cend());
	std::vector<typename SizeRef::value_type> _b_strides(_b.strides().cbegin() + 1, _b.strides().cend());
	std::vector<typename SizeRef::value_type> _a_shape = _a.shape().Vec();
	std::vector<typename SizeRef::value_type> _b_shape = _b.shape().Vec();
	std::vector<typename SizeRef::value_type> _o_shape = _a_shape;
	_o_shape[dim] += _b_shape[dim];
	Tensor outp(SizeRef(_o_shape), _a.dtype);
	std::vector<typename SizeRef::value_type> _o_strides(outp.strides().cbegin() + 1, outp.strides().cend());
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
Tensor& add_(Tensor &a, const Tensor &b){/*std::cout << "adding "<<a.shape() << " and "<<b.shape()<<std::endl;*/functional_operator_this(a, b, functional_operator_num::Add); return a;}

Tensor subtract(const Tensor& a, const Tensor& b){return functional_operator_out(a, b, functional_operator_num::Subtract);}
Tensor& subtract_(Tensor &a, const Tensor &b){functional_operator_this(a, b, functional_operator_num::Subtract); return a;}

Tensor divide(const Tensor& a, const Tensor& b){return functional_operator_out(a, b, functional_operator_num::Divide);}
Tensor& divide_(Tensor &a, const Tensor &b){functional_operator_this(a, b, functional_operator_num::Divide); return a;}


bool all(const Tensor &t){
	if(t.dtype == DType::TensorObj){
		return t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
			return std::all_of(begin, end, [](const Tensor& v){return all(v);});
		});
	}
	exception_dtypes(t.dtype, DType::Bool);
	return t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Bool> > >([](auto begin, auto end){
			return std::all_of(begin, end, [](const uint_bool_t& v){return v.value == 1;});});
}


bool any(const Tensor &t){
	if(t.dtype == DType::TensorObj){
		return t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
			return std::any_of(begin, end, [](const Tensor& v){return any(v);});
		});
	}
	exception_dtypes(t.dtype, DType::Bool);
	return t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Bool> > >([](auto begin, auto end){
			return std::any_of(begin, end, [](const uint_bool_t& v){return v.value == 1;});});

}






/* Tensor conv2dT(const Tensor &image, const Tensor &kernel, const typename SizeRef::value_type s_h, const typename SizeRef::value_type s_w){ */
/* 	utils::THROW_EXCEPTION(image.dims() == 4,"Expected Image dims to be $ but got $", 4, image.dims()); */
/* 	utils::THROW_EXCEPTION(kernel.dims() == 4,"Expected Kernel dims to be $ but got $", 4, kernel.dims()); */
/* 	utils::THROW_EXCEPTION(kernel.shape()[-3] == image.shape()[-3],"Image and kernel at dim 3 must be equal, expected kernel to have $ but had $", image.shape()[-3], kernel.shape()[-3]); */
/* 	exception_dtypes(image.dtype, kernel.dtype); */
	
/* } */

//this applies a softmax function over the entire inputted tensor
void softmax_(Tensor& inp){
	inp.arr_void().execute_function<WRAP_DTYPES<NumberTypesL> >([](auto begin, auto end){
		mp::softmax(begin, end, begin);
	});
	if(inp.dtype == DType::TensorObj){
		inp.exp_();
		Scalar element = inp.sum().toScalar();
		inp.divide_(element);
	}
}

void softmax_(Tensor& inp, typename SizeRef::value_type dim){
	inp.exp_();
	Tensor tensors = inp.split_axis(dim);
	tensors.arr_void().for_each<DType::TensorObj>([](auto& val){
				Scalar element = val.sum().toScalar();
				val.divide_(element);
			});
}

Tensor softmax(const Tensor& inp){
	Tensor outp = inp.clone();
	softmax_(outp);
	return std::move(outp);
}

Tensor softmax(const Tensor& inp, typename SizeRef::value_type dim){
	Tensor outp = inp.exp();
	Tensor tensors = outp.split_axis(dim);
	tensors.arr_void().for_each<DType::TensorObj>([](auto& val){
				Scalar element = val.sum().toScalar();
				val.divide_(element);
			});
	return std::move(outp);
}

void softmax_stable_(Tensor& inp){
	Scalar max = inp.max().values.toScalar();
	inp.exp_();
	inp -= max;
	Scalar element = inp.sum().toScalar();
	inp.divide_(element);
}

void softmax_stable_(Tensor& inp, typename SizeRef::value_type dim){
	Tensor tensors = inp.split_axis(dim);
	tensors.arr_void().for_each<DType::TensorObj>([](auto& val){
				Scalar max = val.max().values.toScalar();
				val.exp_();
				val -= max;
				Scalar element = val.sum().toScalar();
				val.divide_(element);
			});
}

Tensor softmax_stable(const Tensor& inp){
	Scalar max = inp.max().values.toScalar();
	Tensor outp = inp.exp() - max;
	Scalar element = outp.sum().toScalar();
	outp.divide_(element);
	return std::move(outp);
}


Tensor softmax_stable(const Tensor& inp, typename SizeRef::value_type dim){
	Tensor outp(inp);
	Tensor tensors = outp.split_axis(dim);
	tensors.arr_void().for_each<DType::TensorObj>([](auto& val){
				Scalar max = val.max().values.toScalar();
				val.exp_();
				val -= max;
				Scalar element = val.sum().toScalar();
				val.divide_(element);
			});
	return std::move(outp);

}

Tensor cat(std::vector<Tensor> t){
	typename SizeRef::value_type last_dim = 0;
	for(typename SizeRef::value_type i = 1; i < t.size(); ++i){
		exception_dtypes(t[i-1].dtype, t[i].dtype);
		utils::THROW_EXCEPTION(t[i-1].dims() == t[i].dims(), "Runtime Error: Expected all tensors to have $ dims but got $ instead", t[i-1].dims(), t[i].dims());
		for(typename SizeRef::value_type j = 1; j < t[i].dims(); ++j){
			utils::THROW_EXCEPTION(t[i].shape()[j] == t[i-1].shape()[j],
				"Runtime Error: Expected tensors to have same shape ($) at dim $ but got ($) instead",
				t[i-1].shape()[j], j, t[i].shape()[j]);
		}
		last_dim += t[i-1].shape()[0];
	}
	last_dim += t.back().shape()[0];
	std::vector<typename SizeRef::value_type> vec = t[0].shape().Vec();
	vec[0] = last_dim;
	SizeRef a(std::move(vec));
	std::vector<std::reference_wrapper<const ArrayVoid> > arrVds;
	arrVds.reserve(t.size());
	for(Tensor& x : t)
		arrVds.push_back(std::cref(x.arr_void()));
	return Tensor(ArrayVoid::cat(arrVds), a);

}

Tensor cat(const Tensor& t){
	utils::THROW_EXCEPTION(t.dtype == DType::TensorObj,
			"In order to concatenate a tensor, it must hold multiple tensors, but got type $", t.dtype);
	const typename SizeRef::value_type& num = t.numel();
	return t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&num](auto begin, auto end) -> Tensor{
		auto begin_cpy = begin;
		const SizeRef sh = begin->shape();
		++begin;
		for(;begin != end; ++begin){
			utils::THROW_EXCEPTION(begin->shape() == sh, "Expected all shapes in concatenate to be the same, but got $ and $", begin->shape(), sh);
		}
		std::vector<typename SizeRef::value_type> vec = sh.Vec();
		vec[0] *= num;
		std::vector<std::reference_wrapper<const ArrayVoid> > arrVds;
		arrVds.reserve(num); //okay because it is allocating a reference wrapper, putting a number there would cause an allocation error
		begin = begin_cpy;
		typename SizeRef::value_type i = 0;
		for(typename SizeRef::value_type i = 0; begin != end; ++begin, ++i){
			arrVds.push_back(std::cref(begin->arr_void()));
		}
		return Tensor(ArrayVoid::cat(arrVds), SizeRef(std::move(vec)));
		
	});
}

Tensor cat_unordered(const Tensor& t){
	utils::THROW_EXCEPTION(t.dtype == DType::TensorObj,
			"In order to concatenate a single tensor, it must hold multiple tensors, but got type $", t.dtype);
	std::vector<ArrayVoid> arrVds;
	arrVds.reserve(t.numel());
	for(const auto& tensor : t){
		arrVds.push_back(tensor.arr_void());
	}

	ArrayVoid outp = ArrayVoid::cat(arrVds);
	const auto n_numel = outp.Size();
	return Tensor(std::move(outp), {static_cast<typename SizeRef::value_type>(n_numel)});
}

Tensor cat_unordered(const std::vector<Tensor>& t){
	std::vector<std::reference_wrapper<const ArrayVoid> > arrVds;
	arrVds.reserve(t.size());
	for(const auto& tensor : t){
		arrVds.push_back(std::cref(tensor.arr_void()));
	}
	ArrayVoid outp = ArrayVoid::cat(arrVds);
	const auto n_numel = outp.Size();
	return Tensor(std::move(outp), {static_cast<typename SizeRef::value_type>(n_numel)});
}

Tensor cat(std::vector<Tensor> t, int32_t dim){
	dim = dim < 0 ? t[0].dims() + dim : dim;
	typename SizeRef::value_type last_dim = 0;
	for(typename SizeRef::value_type i = 1; i < t.size(); ++i){
		exception_dtypes(t[i-1].dtype, t[i].dtype);
		utils::THROW_EXCEPTION(t[i-1].dims() == t[i].dims(), "Runtime Error: Expected all tensors to have $ dims but got $ instead", t[i-1].dims(), t[i].dims());
		for(typename SizeRef::value_type j = 0; j < t[i].dims(); ++j){
			if(j == dim){
				last_dim += t[i-1].shape()[j];
				continue;
			}
			utils::THROW_EXCEPTION(t[i].shape()[j] == t[i-1].shape()[j],
				"Runtime Error: Expected tensors to have same shape ($) at dim $ but got ($) instead",
				t[i-1].shape()[j], j, t[i].shape()[j]);
		}
	}
	last_dim += t.back().shape()[dim];
	std::vector<typename SizeRef::value_type> vec = t[0].shape().Vec();
	vec[dim] = last_dim;
	SizeRef a(std::move(vec));
	Tensor outp(a, t[0].dtype);
	std::vector<my_range> ranges(dim+1);
	for(typename SizeRef::value_type i = 0; i < ranges.size()-1; ++i){
		ranges[i] = my_range(0, t[0].shape()[i]);
	}
	typename SizeRef::value_type last = 0;
	for(typename SizeRef::value_type i = 0; i < t.size(); ++i){
		ranges.back() = my_range(last, t[i].shape()[dim]);
		last += t[i].shape()[dim];
		outp[ranges] = t[i];
	}
	return outp;
}

Tensor cat(const Tensor& t, int64_t dim){
	utils::THROW_EXCEPTION(t.dtype == DType::TensorObj,
			"In order to concatenate a tensor, it must hold multiple tensors, but got type $", t.dtype);
	const typename SizeRef::value_type& num = t.numel();
	return t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&num, &dim](auto begin, auto end) -> Tensor{
		auto begin_cpy = begin;
		const SizeRef sh = begin->shape().transpose(0, dim);
		int64_t n_dim_size = sh[0];
		const SizeRef sh_smaller = sh.pop_front();
		++begin;
		for(;begin != end; ++begin){
			n_dim_size += begin->shape()[dim];
			utils::THROW_EXCEPTION(begin->shape().transpose(0, dim).pop_front() == sh_smaller, "Expected all shapes in concatenate to be the same, but got $ and $", begin->shape(), sh);
		}
		std::vector<typename SizeRef::value_type> vec = sh.Vec();
		vec[0] = n_dim_size;
		std::vector<std::reference_wrapper<const ArrayVoid> > arrVds;
		arrVds.reserve(num); //okay because it is allocating a reference wrapper, putting a number there would cause an allocation error
		begin = begin_cpy;
		typename SizeRef::value_type i = 0;
		for(typename SizeRef::value_type i = 0; begin != end; ++begin, ++i){
			arrVds.push_back(std::cref(begin->transpose(0, dim).arr_void()));
		}
		return Tensor(ArrayVoid::cat(arrVds), SizeRef(std::move(vec))).transpose(0, dim);
		
	});
	
}


std::vector<Tensor> get_all(Tensor& t){
	std::vector<Tensor> output(t.shape()[0]);
	for(typename SizeRef::value_type i = 0; i < t.shape()[0]; ++i)
		output[i] = t[i];
	return std::move(output);
}

std::vector<Tensor> get_all(std::vector<Tensor>& ts){
	std::vector<Tensor> output(ts[0].shape()[0]*ts.size());
	typename SizeRef::value_type a_counter = 0;
	typename SizeRef::value_type b = ts[0].shape()[0];
	typename SizeRef::value_type a = 0;
	typename SizeRef::value_type ts_counter = 0;
	for(typename SizeRef::value_type i = 0; i < output.size(); ++i){
		output[i] = ts[ts_counter][a_counter];
		if(++a_counter == b){
			++ts_counter;
			a_counter = a;
		}
	}
	return std::move(output);
}

std::vector<Tensor> get_indices(std::vector<Tensor>& ts, int64_t* begin, int64_t* end){
	std::ptrdiff_t diff = std::distance(begin, end);
	std::vector<Tensor> output(diff*ts.size());
	int64_t* begin_cpy = begin;
	uint64_t index = 0;
	for(uint64_t i = 0; i < output.size(); ++i, ++index){
		for(;begin != end; ++begin, ++i)
			output[i] = ts[index][*begin];
		begin = begin_cpy;
	}
	return std::move(output);

}

Tensor index_select(Tensor input, int8_t dim, Tensor index){
	dim = (dim < 0) ? dim + input.dims() : dim;
	utils::THROW_EXCEPTION(dim < input.dims(), "Expected (dim = $) to be less than dims of input which is $", dim, input.dims());
	utils::THROW_EXCEPTION(dim >= 0, "Expected (dim = $) to be greater than or equal to zero", dim);
	utils::THROW_EXCEPTION(index.dims() == 1, "Expected indexing tensor to have a dimensional size of 1 but got $", index.dims());
	utils::THROW_EXCEPTION(index.dtype == DType::int64, "Expected indexing tensor to be dtype int64 but got $", index.dtype);
	if(dim == 0){
		std::vector<Tensor> output(index.numel());
		int64_t* begin = reinterpret_cast<int64_t*>(index.data_ptr());
		int64_t* end = reinterpret_cast<int64_t*>(index.data_ptr_end());
		auto setting = output.begin();
		for(;begin != end; ++begin, ++setting)
			*setting = input[*begin];
		return cat(output);
	}
	auto n_shape = input.shape().Vec();
	n_shape[dim] = index.numel();
	std::vector<Tensor> output = get_all(input);
	--dim;
	while(dim > 0){
		output = get_all(output);
		--dim;
	}

	return cat_unordered(get_indices(output, reinterpret_cast<int64_t*>(index.data_ptr()), reinterpret_cast<int64_t*>(index.data_ptr_end()))).view(SizeRef(std::move(n_shape)));
}

Tensor select(Tensor input, int8_t dim, int64_t index){
	dim = (dim < 0) ? dim + input.dims() : dim;
	if(dim == 0)
		return input[index];
	std::vector<my_range> ranges(dim+1, my_range());
	ranges.back() = my_range(index);
	return input[std::move(ranges)];
}

Tensor split(Tensor input, typename SizeRef::value_type split_size, int8_t dim){
	dim = (dim < 0) ? dim + input.dims() : dim;
	utils::THROW_EXCEPTION(dim < input.dims(), "Expected (dim = $) to be less than dims of input which is $", dim, input.dims());
	utils::THROW_EXCEPTION(dim >= 0, "Expected (dim = $) to be greater than or equal to zero", dim);
	typename SizeRef::value_type total_tensors = input.shape()[dim] / split_size;
	bool remainder = false;
	if(input.shape()[dim] % split_size != 0){++total_tensors; remainder = true;}
	Tensor output({total_tensors}, DType::TensorObj);
	if(dim == 0){
		typename SizeRef::value_type begin = 0;
		typename SizeRef::value_type end = split_size;
		if(!remainder){
			for(typename SizeRef::value_type i = 0; i < total_tensors; ++i){
				output[i] = input[my_range(begin, end)];
				begin += split_size;
				end += split_size;
			}
			return std::move(output);
		}
		for(typename SizeRef::value_type i = 0; i < total_tensors-1; ++i){
			output[i] = input[my_range(begin, end)];
			begin += split_size;
			end += split_size;
		}
		output[total_tensors-1] = input[my_range(begin, -1)];
		return std::move(output);
	}
	std::vector<Tensor> vec = get_all(input);
	int8_t dim_cpy = dim;
	--dim;
	while(dim > 0){
		vec = get_all(vec);
		--dim;
	}
	typename SizeRef::value_type begin = 0;
	typename SizeRef::value_type end = split_size;
	auto n_shape = input.shape().Vec();
	n_shape[dim_cpy] = split_size;
	SizeRef curr_shape(n_shape);
	if(!remainder){
		for(typename SizeRef::value_type i = 0; i < total_tensors; ++i){
			std::vector<Tensor> vec_cpy(vec.size());
			for(typename SizeRef::value_type j = 0; j < vec.size(); ++j){
				vec_cpy[j] = vec[i][my_range(begin, end)];
			}
			output[i] = cat_unordered(vec_cpy).view(curr_shape);
			begin += split_size;
			end += split_size;
		}
		return std::move(output);
	}
	for(typename SizeRef::value_type i = 0; i < total_tensors-1; ++i){
		std::vector<Tensor> vec_cpy(vec.size());
		for(typename SizeRef::value_type j = 0; j < vec.size(); ++j){
			vec_cpy[j] = vec[i][my_range(begin, end)];
		}
		output[i] = cat_unordered(vec_cpy).view(curr_shape);
		begin += split_size;
		end += split_size;
	}
	std::vector<Tensor> vec_cpy(vec.size());
	for(typename SizeRef::value_type j = 0; j < vec.size(); ++j){
		vec_cpy[j] = vec[j][my_range(begin, -1)];
	}
	n_shape[dim_cpy] = input.shape()[dim_cpy] % split_size;
	output[total_tensors-1] = cat_unordered(vec_cpy).view(SizeRef(std::move(n_shape)));
	return std::move(output);
}

Tensor split(Tensor input, std::vector<typename SizeRef::value_type> split_sections, int8_t dim){
	dim = (dim < 0) ? dim + input.dims() : dim;
	typename SizeRef::value_type sum = std::accumulate(split_sections.cbegin(), split_sections.cend(), 0);
	utils::THROW_EXCEPTION(dim < input.dims(), "Expected (dim = $) to be less than dims of input which is $", dim, input.dims());
	utils::THROW_EXCEPTION(dim >= 0, "Expected (dim = $) to be greater than or equal to zero", dim);
	utils::THROW_EXCEPTION(sum == input.shape()[dim], "Expected the sum of split_sections to be equal to the shape along dim $ which is $, instead got $", (int)dim, input.shape()[dim], sum);

	Tensor output({static_cast<typename SizeRef::value_type>(split_sections.size())}, DType::TensorObj);
	if(dim == 0){
		typename SizeRef::value_type begin = 0;
		for(typename SizeRef::value_type i = 0; i < split_sections.size(); ++i){
			output[i] = input[my_range(begin, split_sections[i])];
			begin += split_sections[i];
		}
		return std::move(output);
	}
	std::vector<Tensor> vec = get_all(input);
	int8_t dim_cpy = dim;
	--dim;
	while(dim > 0){
		vec = get_all(vec);
		--dim;
	}
	typename SizeRef::value_type begin = 0;
	auto n_shape = input.shape().Vec();
	for(typename SizeRef::value_type i = 0; i < split_sections.size(); ++i){
		std::vector<Tensor> vec_cpy(vec.size());
		for(typename SizeRef::value_type j = 0; j < vec.size(); ++j){
			vec_cpy[j] = vec[i][my_range(begin, split_sections[i])];
		}
		n_shape[dim_cpy] = split_sections[i];
		output[i] = cat_unordered(vec_cpy).view(SizeRef(n_shape));
		begin += split_sections[i];
	}
	return std::move(output);
}


Tensor chunk(Tensor input, typename SizeRef::value_type chunks, int8_t dim){
	dim = (dim < 0) ? dim + input.dims() : dim;
	utils::THROW_EXCEPTION(dim < input.dims(), "Expected (dim = $) to be less than dims of input which is $", dim, input.dims());
	utils::THROW_EXCEPTION(dim >= 0, "Expected (dim = $) to be greater than or equal to zero", dim);
	Tensor output({chunks}, DType::TensorObj);
	typename SizeRef::value_type adding = input.shape()[dim] / chunks;
	if(dim == 0){
		typename SizeRef::value_type begin = 0;
		typename SizeRef::value_type end = adding;
		for(typename SizeRef::value_type i = 0; i < chunks-1; ++i){
			output[i] = input[my_range(begin, end)];
			begin += adding;
			end += adding;
		}
		output[chunks-1] = input[my_range(begin, -1)];
		return std::move(output);
	}
	std::vector<Tensor> vec = get_all(input);
	int8_t dim_cpy = dim;
	--dim;
	while(dim > 0){
		vec = get_all(vec);
		--dim;
	}
	typename SizeRef::value_type begin = 0;
	typename SizeRef::value_type end = adding;
	auto n_shape = input.shape().Vec();
	n_shape[dim_cpy] = adding;
	SizeRef curr_shape(n_shape);
	for(typename SizeRef::value_type i = 0; i < chunks-1; ++i){
		std::vector<Tensor> vec_cpy(vec.size());
		for(typename SizeRef::value_type j = 0; j < vec.size(); ++j){
			vec_cpy[j] = vec[i][my_range(begin, end)];
		}
		output[i] = cat_unordered(vec_cpy).view(curr_shape);
		begin += adding;
		end += adding;
	}
	n_shape[dim_cpy] = input.shape()[dim_cpy] - (adding * (chunks-1));


	std::vector<Tensor> vec_cpy(vec.size());
	for(typename SizeRef::value_type j = 0; j < vec.size(); ++j){
		vec_cpy[j] = vec[j][my_range(begin, -1)];
	}
	output[chunks-1] = cat_unordered(vec_cpy).view(SizeRef(std::move(n_shape)));
	return std::move(output);
}

Tensor stack(std::vector<std::reference_wrapper<Tensor> > t){
	bool are_same = true;
	for(typename SizeRef::value_type i = 1; i < t.size(); ++i){
		if(t[i-1].get().shape() != t[i].get().shape() || t[i-1].get().dtype != t[i].get().dtype){
			are_same = false;
			break;
		}
	}
	if(are_same){
		std::vector<typename SizeRef::value_type> vec = t[0].get().shape().Vec();
		vec.insert(vec.begin(), t.size());
		SizeRef a(std::move(vec));
		Tensor output(std::move(a), t[0].get().dtype);
		for(typename SizeRef::value_type i = 0; i < t.size(); ++i)
			output[i] = t[i].get();
		return std::move(output);
	}

	Tensor output({static_cast<typename SizeRef::value_type>(t.size())}, DType::TensorObj);
	for(typename SizeRef::value_type i = 0; i < t.size(); ++i){
		output[i] = t[i].get();
	}
	return std::move(output);
}

Tensor stack(std::vector<Tensor> t){
	bool are_same = true;
	for(typename SizeRef::value_type i = 1; i < t.size(); ++i){
		if(t[i-1].shape() != t[i].shape() || t[i-1].dtype != t[i].dtype){
			are_same = false;
			break;
		}
	}
	if(are_same){
		std::vector<typename SizeRef::value_type> vec = t[0].shape().Vec();
		vec.insert(vec.begin(), t.size());
		SizeRef a(std::move(vec));
		Tensor output(std::move(a), t[0].dtype);
		for(typename SizeRef::value_type i = 0; i < t.size(); ++i)
			output[i] = t[i];
		return std::move(output);
	}
	Tensor output({static_cast<typename SizeRef::value_type>(t.size())}, DType::TensorObj);
	for(typename SizeRef::value_type i = 0; i < t.size(); ++i)
		output[i] = t[i];
	return std::move(output);
}

Tensor stack(std::vector<Tensor> t, int8_t dim){
	for(typename SizeRef::value_type i = 1; i < t.size(); ++i){
		utils::THROW_EXCEPTION(t[i-1].shape() == t[i].shape(),
				"Runtime Error: Expected all Tensors to have same shape of $ but instead got", t[i-1].shape(), t[i].shape());
		exception_dtypes(t[i-1].dtype, t[i].dtype);
	}
	std::vector<typename SizeRef::value_type> vec = t[0].shape().Vec();
	dim = dim < 0 ? dim + t[0].dims() : dim;
	vec.insert(vec.begin() + dim, t.size());
	SizeRef a(std::move(vec));
	std::vector<ArrayVoid> arrVds;
	arrVds.reserve(t.size());
	for(Tensor& x : t)
		arrVds.push_back(x.arr_void());

	return Tensor(ArrayVoid::cat(arrVds), a.transpose(0, dim)).transpose(0, dim);
}


Tensor vectorize(std::vector<Tensor> t){
	Tensor output = Tensor::makeNullTensorArray(t.size());
	Tensor* begin = reinterpret_cast<Tensor*>(output.data_ptr());
	Tensor* end = begin + output.numel();
	for(typename SizeRef::value_type i = 0; i < t.size(); ++i, ++begin)
		*begin = t[i];
	return std::move(output);
}

Tensor sigmoid(const Tensor& x){
	if(x.dtype == DType::TensorObj){
		Tensor a = (-1) * x;
		a.exp_();
		a += 1;
		a.inverse_();
		return std::move(a);
	}
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL> >([](auto begin, auto end){
		mp::sigmoid(begin, end, begin);
	});
	return std::move(a);
}

Tensor dsigmoid(const Tensor & x, bool apply_sigmoid){
	if(x.dtype == DType::TensorObj){
		if(!apply_sigmoid)
			return x * (1-x);
		Tensor sigmoid_x = sigmoid(x);
		return sigmoid_x * (1 - sigmoid_x);	
	}
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL> >([&apply_sigmoid](auto begin, auto end){
		mp::dsigmoid(begin, end, begin, apply_sigmoid);
	});
	return std::move(a);
	
}


Tensor tanh(const Tensor& x){
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
		mp::tanh(begin, end, begin);
	});
	return std::move(a);
}

Tensor tan(const Tensor& x){
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
		mp::tan(begin, end, begin);
	});
	return std::move(a);
}

Tensor dtanh(const Tensor& x){
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
		mp::dtanh(begin, end, begin);
	});
	return std::move(a);
}

Tensor dtan(const Tensor& x){
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
		mp::dtan(begin, end, begin);
	});
	return std::move(a);
}

Tensor sinh(const Tensor& x){
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
		mp::sinh(begin, end, begin);
	});
	return std::move(a);
}

Tensor sin(const Tensor& x){
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
		mp::sin(begin, end, begin);
	});
	return std::move(a);
}

Tensor cosh(const Tensor& x){
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
		mp::cosh(begin, end, begin);
	});
	return std::move(a);
}

Tensor cos(const Tensor& x){
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
		mp::cos(begin, end, begin);
	});
	return std::move(a);
}

Tensor sqrt(const Tensor& x){
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
		mp::sqrt(begin, end, begin);
	});
	return std::move(a);
	
}

Tensor invsqrt(const Tensor& x){
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
		mp::invsqrt(begin, end, begin);
	});
	return std::move(a);
}

Tensor dinvsqrt(const Tensor& x){
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
		mp::dinvsqrt(begin, end, begin);
	});
	return std::move(a);
}


Tensor var(const Tensor& x, utils::optional_list dim, int64_t correction, bool keepdim){
	Tensor mean = x.mean(dim, true);
	Tensor squared_diff = std::pow((x - mean), 2);
	int64_t N = 0;
	if(!dim){
		N = x.numel();
	}else{
		N = 1;
		for(const auto& ele : dim){
			N *= x.shape()[ele];
		}
	}
	Tensor variance = squared_diff.sum(dim, keepdim) / (N - correction);
	return std::move(variance);
}
Tensor dvar(const Tensor& dx, const Tensor& x, utils::optional_list dim, int64_t correction){
	//takes both the gradient, and the input given to the variance function
	Tensor mean = x.mean(dim, true);
	int64_t N = 0;
	if(!dim){
		N = x.numel();
	}else{
		N = 1;
		for(const auto& ele : dim){
			N *= x.shape()[ele];
		}
	}
	return (2 / (N - correction)) * (x - mean);
}


size_t amount_of(const Tensor& t, Scalar s){
	if(t.dtype == DType::Bool){
		uint_bool_t b = s.to<uint_bool_t>();
		return t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Bool>>>(
				[&b](auto begin, auto end) -> size_t{
					size_t amt = 0;
					for(;begin != end; ++begin)
						if(*begin == b){++amt;}
					return amt;
				});
	}
	return t.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(
			[&s](auto a_begin, auto a_end) -> size_t{
				using value_t = utils::IteratorBaseType_t<decltype(a_begin)>;
				value_t ns = s.to<value_t>();
				size_t amt = 0;
				for(;a_begin != a_end; ++a_begin)
					if(*a_begin == ns){++amt;}
				return amt;
			});
}

size_t count(const Tensor& t){
	utils::THROW_EXCEPTION(t.dtype == DType::Bool, "Expected to get bool tensor for count function but got $", t.dtype);
	uint_bool_t b = uint_bool_t(true);
	return t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Bool>>>(
			[&b](auto begin, auto end) -> size_t{
				size_t amt = 0;
				for(;begin != end; ++begin)
					if(*begin == b){++amt;}
				return amt;
			});
}


void next_index(const SizeRef& s, std::vector<int64_t>& v, typename SizeRef::value_type index){
	if(v[index] == s[index] - 1){
		v[index] = 0;
		if(index == 0){
			std::fill(v.begin(), v.end(), 0);
			next_index(s, v, v.size()-1);
			return;
		}
		--index;
		next_index(s, v, index);
	}
	else{
		++v[index];
	}
}


//inefficient, but for some reason seems to be the only way it works?
//maybe look back into this function in the future
Tensor where(Tensor t){
	utils::THROW_EXCEPTION(t.dtype == DType::Bool, "Expected dtype to be DType::Bool but got $", t.dtype);
	utils::THROW_EXCEPTION(t.is_contiguous(), "Expected contiguous tensor for where");
	uint_bool_t looking(true);
	size_t amt = amount_of(t, looking);
	Tensor outp({static_cast<typename SizeRef::value_type>(t.dims()), static_cast<typename SizeRef::value_type>(amt)}, DType::int64);
	
	Tensor ts = outp.split_axis_1();
	std::vector<int64_t> indexes(t.dims(), 0);
	uint_bool_t* begin = reinterpret_cast<uint_bool_t*>(t.data_ptr());
	uint_bool_t* end = begin + t.numel();
	const typename SizeRef::value_type index = indexes.size() - 1;
	int64_t keeping = 0;
	Tensor* ts_begin = reinterpret_cast<Tensor*>(ts.data_ptr());
	Tensor* ts_end = ts_begin + ts.numel();
	Tensor* ts_cpy = ts_begin;
	for(;begin != end; ++begin, next_index(t.shape(), indexes, index)){
		if(*begin == looking){
			auto cbegin = indexes.cbegin();
			ts_begin->arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::int64>>>([&cbegin](auto a_begin, auto a_end){
				for(;a_begin != a_end; ++a_begin, ++cbegin){
					*a_begin = *cbegin;
				}
			});
			++ts_begin;
		}
	}
	return outp.split_axis(0);
}

Tensor meshgrid(Tensor&& x, Tensor&& y){
	utils::THROW_EXCEPTION(x.dtype == y.dtype, "Runtime Error: Expected tensors to have same dtype but got $ and $", x.dtype, y.dtype);
	/* utils::THROW_EXCEPTION(a.numel() == b.numel(), "RuntimeError: Expected tensors to have same number of elements but got $ and $", a.numel(), b.numel()) */
	Tensor xy({2}, DType::TensorObj);
	Tensor* xy_p = reinterpret_cast<Tensor*>(xy.data_ptr());
	*xy_p = Tensor({static_cast<typename SizeRef::value_type>(x.numel()), static_cast<typename SizeRef::value_type>(y.numel())}, x.dtype);
	*(xy_p + 1) = Tensor({static_cast<typename SizeRef::value_type>(x.numel()), static_cast<typename SizeRef::value_type>(y.numel())}, x.dtype);
	
	const typename SizeRef::value_type x_n = x.numel();
	const typename SizeRef::value_type y_n = y.numel();
	x.arr_void().execute_function([xy_p, &x_n, &y_n](auto a_begin, auto a_end, auto b_begin){
				Tensor& X = *xy_p;
				Tensor& Y = *(xy_p+1);
				const typename SizeRef::value_type total_size = x_n * y_n;
				using value_t = utils::IteratorBaseType_t<decltype(a_begin)>;
				value_t* x_begin = reinterpret_cast<value_t*>(X.data_ptr());
				value_t* y_begin = reinterpret_cast<value_t*>(Y.data_ptr());
				auto b_end = b_begin + y_n;
				auto b_cpy = b_begin;
				
				for(;a_begin != a_end; ++a_begin){
					for(;b_begin != b_end; ++b_begin, ++x_begin, ++y_begin){
						*x_begin = *a_begin;
						*y_begin = *b_begin;
					}
					b_begin = b_cpy;
				}

			}, y.arr_void());
	return std::move(xy);

}





//this is for when the stride is already changed, makes certain functions easier
//this is more akin to pytorch's version
//however, purely because of the way that the cat function works, this could be dangerous
//for example if I did cat(A, B[2], C)
//it would look at all of B, not just B[2]
Tensor as_strided_force_contiguity(const Tensor& input, const SizeRef& n_size, const SizeRef& n_stride, const int64_t& storage_offset){
	utils::THROW_EXCEPTION(n_size.size() == n_stride.size(), "Expected to have same amount of strides as dimensions for as strided");
	//bound force contiguity bucket basically takes the tensor's memory
	//and looks at each bucket of memory (for example if there was a concatenation that happened)
	//and it looks at all instances of memory
	ArrayVoid strided_vals = input.strides() != input.getChangedStrides() ? input.arr_void().bound_force_contiguity_bucket() : (input.arr_void().get_bucket().is_strided()) ? input.arr_void() : input.arr_void().bucket_all_indices(); //make input strided so that memory can be accessed one at a time
	ArrayVoid output = strided_vals.new_strides(n_size.multiply()); //make the output strided but with the new size
	void** in_begin = strided_vals.stride_begin(); //start at the correct indice to begin
	void** out_begin = output.stride_begin();//the starting point of the new tensor
	// Calculate the total number of elements in the new tensor
	Tensor::size_value_t total_elements = n_size.multiply();
	//get contiguous strides of the new size
	std::vector<Tensor::size_value_t> multiplicities = n_size.strides();
	//make a reference to numel (reduce overhead)
	const uint64_t& num = strided_vals.Size();
	// Fill the output tensor with strided values from the input tensor
	for (Tensor::size_value_t i = 0; i < total_elements; ++i) {
		Tensor::size_value_t offset = storage_offset;
		Tensor::size_value_t index = i;
			for (Tensor::size_value_t j = 0; j < n_size.size()-1; ++j) {
				Tensor::size_value_t mult = (index / multiplicities[j+1]);
				offset += mult * n_stride[j];
				index -= mult * multiplicities[j+1];
			}
		offset += index * n_stride.back();
		//make sure offset isn't out of range, if so, subtract by input.numel() until it is back in range
		offset = (offset < num) ? offset : offset % num;
		out_begin[i] = in_begin[offset];
	}
	
	if(n_stride.size() == n_size.size()){
		std::vector<Tensor::size_value_t> out_strides = {n_size.multiply()};
		out_strides.insert(out_strides.end(), n_stride.begin(), n_stride.end());
		return Tensor(output, std::move(n_size), out_strides);
	}
	return Tensor(output, std::move(n_size), n_stride.Vec());
}


//this goes based off a comparison of the original strides inputted
//so based off of input.strides()
//which is basically contiguous viewing
//but it goes based off of the way the strides are already implanted in memory
Tensor as_strided(const Tensor& input, const SizeRef n_size, SizeRef n_stride, const int64_t storage_offset, bool whole_tensor){
	if(n_stride.size() == n_size.size()+1){n_stride = n_stride.pop_front();}
	if(whole_tensor){return as_strided_force_contiguity(input, n_size, n_stride, storage_offset);}
	utils::THROW_EXCEPTION(n_size.size() == n_stride.size(), "Expected to have same amount of strides as dimensions for as size or one more, where the last dimension represents n_size.multiply()");
	//bound force contiguity bucket basically takes the tensor's memory
	//and looks at each bucket of memory (for example if there was a concatenation that happened)
	//and it looks at all instances of memory
	ArrayVoid strided_vals = (input.arr_void().get_bucket().is_strided()) ? input.arr_void() : input.arr_void().bucket_all_indices(); //make input strided so that memory can be accessed one at a time
	ArrayVoid output = strided_vals.new_strides(n_size.multiply()); //make the output strided but with the new size
	void** in_begin = strided_vals.stride_begin(); //start at the correct indice to begin
	void** out_begin = output.stride_begin();//the starting point of the new tensor
	// Calculate the total number of elements in the new tensor
	Tensor::size_value_t total_elements = n_size.multiply();
	//get contiguous strides of the new size
	std::vector<Tensor::size_value_t> multiplicities = n_size.strides();
	//make a reference to numel (reduce overhead)
	const uint64_t& num = strided_vals.Size();
	// Fill the output tensor with strided values from the input tensor
	for (Tensor::size_value_t i = 0; i < total_elements; ++i) {
		Tensor::size_value_t offset = storage_offset;
		Tensor::size_value_t index = i;
			for (Tensor::size_value_t j = 0; j < n_size.size()-1; ++j) {
				Tensor::size_value_t mult = (index / multiplicities[j+1]);
				offset += mult * n_stride[j];
				index -= mult * multiplicities[j+1];
			}
		offset += index * n_stride.back();
		//make sure offset isn't out of range, if so, subtract by input.numel() until it is back in range
		offset = (offset < num) ? offset : offset % num;
		out_begin[i] = in_begin[offset];
	}
	
	if(n_stride.size() == n_size.size()){
		std::vector<Tensor::size_value_t> out_strides = {n_size.multiply()};
		out_strides.insert(out_strides.end(), n_stride.begin(), n_stride.end());
		return Tensor(output, std::move(n_size), out_strides);
	}
	return Tensor(output, std::move(n_size), n_stride.Vec());
}

}
}
