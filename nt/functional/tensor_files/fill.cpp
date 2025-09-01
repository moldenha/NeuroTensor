#include "fill.h"
#include "../../dtype/ArrayVoid.h"
#include "../../dtype/ArrayVoid.hpp"
#include "exceptions.hpp"
#include "../cpu/fill.h"

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
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
	if(t.dtype() != DType::TensorObj){
		return zeros(t.shape(), t.dtype());
	}
    for(const auto& _t : t)
        _NT_FUNCTIONAL_ALWAYS_CHECK_(_t);
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
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
	if(t.dtype() != DType::TensorObj){
		return ones(t.shape(), t.dtype());
	}
    for(const auto& _t : t)
        _NT_FUNCTIONAL_ALWAYS_CHECK_(_t);
	Tensor out = Tensor::makeNullTensorArray(t.numel());
	out = out.view(t.shape());
	Tensor* begin_o = reinterpret_cast<Tensor*>(out.data_ptr());
	t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&begin_o](auto begin, auto end){
		for(;begin != end; ++begin, ++begin_o){*begin_o = ones_like(*begin);}
	});
	return std::move(out);
}

Tensor nums(SizeRef inp, Scalar k, DType dt){
	Tensor output(std::move(inp), dt);
	output.arr_void().fill_(k);
	return std::move(output);
}

Tensor nums_like(const Tensor& t, Scalar k){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
	if(t.dtype() != DType::TensorObj){
		return nums(t.shape(), k, t.dtype());
	}
    for(const auto& _t : t)
        _NT_FUNCTIONAL_ALWAYS_CHECK_(_t);
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

Tensor& fill_diagonal_(Tensor& t, Scalar c){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    check_mutability(t);
    if(t.dims() < 2 && t.dtype() == DType::TensorObj){
        t.arr_void().for_each<DType::TensorObj>([&c](auto& inp){fill_diagonal_(inp, c);});
        return t;
    }
    utils::throw_exception(t.dims() >= 2, "Cannot fill diagonal of tensor shape $ that has less than 2 dimensions", t.shape());
    const int64_t& rows = t.shape()[-2];
    const int64_t& cols = t.shape()[-1];
    const int64_t batches = t.numel() / (rows * cols);
    if(t.dtype() == DType::TensorObj){
        t.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >(
        [&c, &rows, &cols, &batches](auto begin, auto end){
            int64_t min_rows = std::min(rows, cols);
            for(int64_t b = 0; b < batches; ++b){
                auto next_begin = begin + (rows * cols);
                for(int64_t r = 0; r < min_rows; ++r){
                    if(end < begin || begin == end) break;
                    *begin = c;
                    begin += cols + 1;
                }
                begin = next_begin;
            }
        });
        return t;
    }
    cpu::_fill_diagonal_(t.arr_void(), c, batches, rows, cols);
    return t;
}


Tensor& fill_(Tensor& t, Scalar s){
    // _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    // No reason to not allow null tensors to be set by a scalar
    // operations like t[t < 0] = 0 happen in ReLU
    // and if there are no values less than 0, it should not throw an exception
    if(t.is_null())
        return t;
    check_mutability(t);
    if(t.dtype() == DType::TensorObj){
        t.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >(
        [&s](auto begin, auto end){
            for(;begin != end; ++begin)
                fill_(*begin, s);
        });
        return t;
    }
    cpu::_fill_scalar_(t.arr_void(), s);
    return t;

}

Tensor& set_(Tensor& t, const Tensor& s){
    if(t.is_null()){
        _NT_FUNCTIONAL_ALWAYS_CHECK_(s);
        t = s;
        return t;
    }
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t, s);
    check_mutability(t);
    utils::THROW_EXCEPTION(t.dtype() == s.dtype(), "Expected dtype of $ but got $ for set_ function",
                       t.dtype(), s.dtype());
    utils::THROW_EXCEPTION(t.shape() == s.shape(),
                           "Expected shape to be $ but got shape $ for set_ function", s.shape(),
                           t.shape());
    if(t.dtype() == DType::TensorObj){
        t.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >(
        [&s](auto begin, auto end, auto begin2){
            for(;begin != end; ++begin, ++begin2){
                if(begin->dtype() == begin2->dtype() && begin->shape() == begin2->shape()){
                    set_(*begin, *begin2);
                }else{
                    *begin = *begin2;
                }
            }
        }, const_cast<ArrayVoid&>(s.arr_void()));
        return t;
    }
    cpu::_set_(t.arr_void(), s.arr_void());
    return t; 
}

}
}
