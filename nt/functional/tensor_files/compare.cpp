#include "../../Tensor.h"
#include "../../dtype/ArrayVoid.hpp"
#include "../cpu/compare.h"
#include <algorithm>

namespace nt{
namespace functional{


inline void compare_equal(const Tensor& a, const Tensor& b){
    utils::THROW_EXCEPTION(a.shape() == b.shape(),
                           "\nRuntimeError: Expected shape a ($) to be equal to shape b ($) ",
                           a.shape(), b.shape());
    utils::THROW_EXCEPTION(
        a.dtype == b.dtype,
        "\nRuntimeError: Expected dtype a ($) to be equal to dtype b ($)",
        a.dtype, b.dtype);
}


using DualFunc = Tensor (*)(const Tensor&, const Tensor&);
inline Tensor tensor_of_tensors(const Tensor& a, const Tensor& b, DualFunc func){
    Tensor out = Tensor::makeNullTensorArray(a.numel());
    Tensor* o_begin = reinterpret_cast<Tensor*>(out.data_ptr());
    a.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >(
        [&func, &o_begin](auto begin, auto end, auto begin2){
            for(;begin != end; ++begin, ++begin2, ++o_begin){
                *o_begin = func(*begin, *begin2);
            }
    }, b.arr_void());
    return out.view(a.shape());
}

Tensor equal(const Tensor& a, const Tensor& b){
    compare_equal(a, b);
    if(a.dtype == DType::TensorObj){return tensor_of_tensors(a, b, &equal);}
    Tensor out(a.shape(), DType::Bool);
    cpu::_equal(out.arr_void(), a.arr_void(), b.arr_void());
    return out;
}

Tensor not_equal(const Tensor& a, const Tensor& b){
    compare_equal(a, b);
    if(a.dtype == DType::TensorObj){return tensor_of_tensors(a, b, &not_equal);}
    Tensor out(a.shape(), DType::Bool);
    cpu::_not_equal(out.arr_void(), a.arr_void(), b.arr_void());
    return std::move(out);
}
Tensor less_than(const Tensor& a, const Tensor& b){
    compare_equal(a, b);
    if(a.dtype == DType::TensorObj){return tensor_of_tensors(a, b, &less_than);}
    Tensor out(a.shape(), DType::Bool);
    cpu::_less_than(out.arr_void(), a.arr_void(), b.arr_void());
    return std::move(out);
}
Tensor greater_than(const Tensor& a, const Tensor& b){
    compare_equal(a, b);
    if(a.dtype == DType::TensorObj){return tensor_of_tensors(a, b, &greater_than);}
    Tensor out(a.shape(), DType::Bool);
    cpu::_greater_than(out.arr_void(), a.arr_void(), b.arr_void());
    return std::move(out);
}
Tensor less_than_equal(const Tensor& a, const Tensor& b){
    compare_equal(a, b);
    if(a.dtype == DType::TensorObj){return tensor_of_tensors(a, b, &less_than_equal);}
    Tensor out(a.shape(), DType::Bool);
    cpu::_less_than_equal(out.arr_void(), a.arr_void(), b.arr_void());
    return std::move(out);
}
Tensor greater_than_equal(const Tensor& a, const Tensor& b){
    compare_equal(a, b);
    if(a.dtype == DType::TensorObj){return tensor_of_tensors(a, b, &greater_than_equal);}
    Tensor out(a.shape(), DType::Bool);
    cpu::_greater_than_equal(out.arr_void(), a.arr_void(), b.arr_void());
    return std::move(out);
}
Tensor and_op(const Tensor& a, const Tensor& b){
    compare_equal(a, b);
    if(a.dtype == DType::TensorObj){return tensor_of_tensors(a, b, &and_op);}
    utils::throw_exception(a.dtype == DType::Bool,
                           "and operator only works on bool dtypes got $", a.dtype);
    Tensor out(a.shape(), DType::Bool);
    cpu::_and_op(out.arr_void(), a.arr_void(), b.arr_void());
    return std::move(out);
}
Tensor or_op(const Tensor& a, const Tensor& b){
    compare_equal(a, b);
    if(a.dtype == DType::TensorObj){return tensor_of_tensors(a, b, &or_op);}
    utils::throw_exception(a.dtype == DType::Bool,
                           "or operator only works on bool dtypes got $", a.dtype);
    Tensor out(a.shape(), DType::Bool);
    cpu::_or_op(out.arr_void(), a.arr_void(), b.arr_void());
    return std::move(out);
}


using SingleFunc = Tensor (*)(const Tensor&, Scalar);
inline Tensor tensor_of_scalars(const Tensor& a, Scalar b, SingleFunc func){
    Tensor out = Tensor::makeNullTensorArray(a.numel());
    Tensor* o_begin = reinterpret_cast<Tensor*>(out.data_ptr());
    a.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >(
        [&func, &o_begin, &b](auto begin, auto end){
            for(;begin != end; ++begin, ++o_begin){
                *o_begin = func(*begin, b);
            }
    });
    return out.view(a.shape());
}


Tensor equal(const Tensor& a, Scalar b){
    Tensor out(a.shape(), DType::Bool);
    cpu::_equal(out.arr_void(), a.arr_void(), b);
    return std::move(out);
}
Tensor not_equal(const Tensor& a, Scalar b){
    Tensor out(a.shape(), DType::Bool);
    cpu::_not_equal(out.arr_void(), a.arr_void(), b);
    return std::move(out);
}
Tensor less_than(const Tensor& a, Scalar b){
    Tensor out(a.shape(), DType::Bool);
    cpu::_less_than(out.arr_void(), a.arr_void(), b);
    return std::move(out);
}
Tensor greater_than(const Tensor& a, Scalar b){
    Tensor out(a.shape(), DType::Bool);
    cpu::_greater_than(out.arr_void(), a.arr_void(), b);
    return std::move(out);
}
Tensor less_than_equal(const Tensor& a, Scalar b){
    Tensor out(a.shape(), DType::Bool);
    cpu::_less_than_equal(out.arr_void(), a.arr_void(), b);
    return std::move(out);
}
Tensor greater_than_equal(const Tensor& a, Scalar b){
    Tensor out(a.shape(), DType::Bool);
    cpu::_greater_than_equal(out.arr_void(), a.arr_void(), b);
    return std::move(out);
}
bool all(const Tensor & t){
    if(t.dtype == DType::TensorObj){
		return t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
			return std::all_of(begin, end, [](const Tensor& v){return all(v);});
		});
	}
    utils::throw_exception(t.dtype == DType::Bool,
                           "Expected dtype for all to be bool got $", t.dtype);
    return cpu::_all(t.arr_void());
}
bool any(const Tensor & t){
    if(t.dtype == DType::TensorObj){
		return t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
			return std::any_of(begin, end, [](const Tensor& v){return any(v);});
		});
	}
    utils::throw_exception(t.dtype == DType::Bool,
                           "Expected dtype for all to be bool got $", t.dtype);
    return cpu::_any(t.arr_void());
}
bool none(const Tensor & t){
    if(t.dtype == DType::TensorObj){
		return t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
			return std::all_of(begin, end, [](const Tensor& v){return none(v);});
		});
	}
    utils::throw_exception(t.dtype == DType::Bool,
                           "Expected dtype for all to be bool got $", t.dtype);
    return cpu::_none(t.arr_void());
}

}
}
