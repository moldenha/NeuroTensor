#include "fill.h"
#include "../../dtype/ArrayVoid.h"
#include "../../dtype/ArrayVoid.hpp"

namespace nt{
namespace functional{

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

Tensor nums_like(const Tensor& t, const Scalar k){
	if(t.dtype != DType::TensorObj){
		return nums(t.shape(), k, t.dtype);
	}
	Tensor out = Tensor::makeNullTensorArray(t.numel());
	out = out.view(t.shape());
	Tensor* begin_o = reinterpret_cast<Tensor*>(out.data_ptr());
	t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&begin_o, &k](auto begin, auto end){
		for(;begin != end; ++begin, ++begin_o){*begin_o = nums_like(*begin, k);}
	});
	return std::move(out);
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

}
}
