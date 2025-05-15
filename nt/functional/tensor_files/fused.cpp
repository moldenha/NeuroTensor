#include "fused.h"
#include "../cpu/fused.h"
#include "exceptions.hpp"

namespace nt{
namespace functional{

//returns c + (a * b);
Tensor fused_multiply_add(const Tensor& c, const Tensor& a, const Tensor& b){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(a, b, c);
    utils::throw_exception(c.numel() == a.numel() && c.numel() == b.numel(),
                           "For optimized fused multiply add, expected all tensors to be the same shape a=($), b=($), c=($)",
                           a.shape(), b.shape(), c.shape());
    utils::throw_exception(c.dtype == a.dtype && c.dtype == b.dtype, "Expected dtypes of tensors to match but got $, $, and $", c.dtype, a.dtype, b.dtype);
    utils::throw_exception(c.dtype != DType::TensorObj && c.dtype != DType::Bool, "Optimized fused multiply and op does not support $", c.dtype);
    Tensor out = c.clone();
	ArrayVoid& B_arrv = const_cast<Tensor&>(b).arr_void();
    cpu::_fused_multiply_add(const_cast<ArrayVoid&>(a.arr_void()), B_arrv, out.arr_void());
    return std::move(out);
}

//returns c += (a * b);
Tensor& fused_multiply_add_(Tensor& c, const Tensor& a, const Tensor& b){
    utils::THROW_EXCEPTION(c.is_mutable(), "Output from fused operation must be mutable");
    _NT_FUNCTIONAL_ALWAYS_CHECK_(a, b, c);
    utils::throw_exception(c.numel() == a.numel() && c.numel() == b.numel(),
                           "For optimized fused multiply add, expected all tensors to be the same shape a=($), b=($), c=($)",
                           a.shape(), b.shape(), c.shape());
    utils::throw_exception(c.dtype == a.dtype && c.dtype == b.dtype, "Expected dtypes of tensors to match but got $, $, and $", c.dtype, a.dtype, b.dtype);
    utils::throw_exception(c.dtype != DType::TensorObj && c.dtype != DType::Bool, "Optimized fused multiply and op does not support $", c.dtype);
    cpu::_fused_multiply_add_(c.arr_void(), const_cast<ArrayVoid&>(a.arr_void()), const_cast<ArrayVoid&>(b.arr_void()));
    ArrayVoid& B_arrv = const_cast<Tensor&>(b).arr_void();
    return c;
}


//returns c + (a * b);
Tensor fused_multiply_add(const Tensor& c, const Tensor& a, Scalar b){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(a, c);
    utils::throw_exception(c.numel() == a.numel(),
                           "For optimized fused multiply add, expected all tensors to be the same shape a=($), c=($)",
                           a.shape(), c.shape());
    utils::throw_exception(c.dtype == a.dtype, "Expected dtypes of tensors to match but got $ and $", c.dtype, a.dtype);
    utils::throw_exception(c.dtype != DType::TensorObj && c.dtype != DType::Bool, "Optimized fused multiply and op does not support $", c.dtype);
    Tensor out = c.clone();
    cpu::_fused_multiply_add(const_cast<Tensor&>(a).arr_void(), b, out.arr_void());
    return std::move(out);
}



Tensor& fused_multiply_add_(Tensor& c, const Tensor& a, Scalar b){
    utils::THROW_EXCEPTION(c.is_mutable(), "Output from fused operation must be mutable");
    _NT_FUNCTIONAL_ALWAYS_CHECK_(a, c);
    utils::throw_exception(c.numel() == a.numel(),
                           "For optimized fused multiply add, expected all tensors to be the same shape a=($), c=($)",
                           a.shape(), c.shape());
    utils::throw_exception(c.dtype == a.dtype, "Expected dtypes of tensors to match but got $ and $", c.dtype, a.dtype);
    utils::throw_exception(c.dtype != DType::TensorObj && c.dtype != DType::Bool, "Optimized fused multiply and op does not support $", c.dtype);
    cpu::_fused_multiply_add_(c.arr_void(), const_cast<Tensor&>(a).arr_void(), b);
    return c;
}

//returns c - (a * b);
Tensor fused_multiply_subtract(const Tensor& c, const Tensor& a, const Tensor& b){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(a, c, b);
    utils::throw_exception(c.numel() == a.numel() && c.numel() == b.numel(),
                           "For optimized fused multiply subtract, expected all tensors to be the same shape a=($), b=($), c=($)",
                           a.shape(), b.shape(), c.shape());
    utils::throw_exception(c.dtype == a.dtype && c.dtype == b.dtype, "Expected dtypes of tensors to match but got $, $, and $", c.dtype, a.dtype, b.dtype);
    utils::throw_exception(c.dtype != DType::TensorObj && c.dtype != DType::Bool, "Optimized fused multiply and op does not support $", c.dtype);
    Tensor out(c.shape(), c.dtype);
    cpu::_fused_multiply_subtract(const_cast<Tensor&>(c).arr_void(), const_cast<Tensor&>(a).arr_void(), const_cast<Tensor&>(b).arr_void(), out.arr_void());
    return std::move(out);
}

Tensor fused_multiply_subtract(const Tensor& c, const Tensor& a, Scalar b){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(a, c);
    utils::throw_exception(c.numel() == a.numel(),
                           "For optimized fused multiply subtract, expected all tensors to be the same shape a=($), c=($)",
                           a.shape(), c.shape());
    utils::throw_exception(c.dtype == a.dtype, "Expected dtypes of tensors to match but got $ and $", c.dtype, a.dtype);
    utils::throw_exception(c.dtype != DType::TensorObj && c.dtype != DType::Bool, "Optimized fused multiply and op does not support $", c.dtype);
    Tensor out(c.shape(), c.dtype);
    cpu::_fused_multiply_subtract(const_cast<Tensor&>(c).arr_void(),
                                  const_cast<Tensor&>(a).arr_void(),
                                    b, out.arr_void());
    return std::move(out);
}

//returns c -= (a * b);
Tensor& fused_multiply_subtract_(Tensor& c, const Tensor& a, const Tensor& b){
    utils::THROW_EXCEPTION(c.is_mutable(), "Output from fused operation must be mutable");
    _NT_FUNCTIONAL_ALWAYS_CHECK_(a, c, b);
    utils::throw_exception(c.numel() == a.numel() && c.numel() == b.numel(),
                           "For optimized fused multiply subtract, expected all tensors to be the same shape a=($), b=($), c=($)",
                           a.shape(), b.shape(), c.shape());
    utils::throw_exception(c.dtype == a.dtype && c.dtype == b.dtype, "Expected dtypes of tensors to match but got $, $, and $", c.dtype, a.dtype, b.dtype);
    utils::throw_exception(c.dtype != DType::TensorObj && c.dtype != DType::Bool, "Optimized fused multiply and op does not support $", c.dtype);
    cpu::_fused_multiply_subtract_(c.arr_void(), const_cast<Tensor&>(a).arr_void(), const_cast<Tensor&>(b).arr_void());
    return c;
}


Tensor& fused_multiply_subtract_(Tensor& c, const Tensor& a, Scalar b){
    utils::THROW_EXCEPTION(c.is_mutable(), "Output from fused operation must be mutable");
    _NT_FUNCTIONAL_ALWAYS_CHECK_(a, c);
    utils::throw_exception(c.numel() == a.numel(),
                           "For optimized fused multiply subtract, expected all tensors to be the same shape a=($), c=($)",
                           a.shape(), c.shape());
    utils::throw_exception(c.dtype == a.dtype, "Expected dtypes of tensors to match but got $ and $", c.dtype, a.dtype);
    utils::throw_exception(c.dtype != DType::TensorObj && c.dtype != DType::Bool, "Optimized fused multiply and op does not support $", c.dtype);
    cpu::_fused_multiply_subtract_(const_cast<Tensor&>(c).arr_void(),
                           const_cast<Tensor&>(a).arr_void(), b);
    return c;
}


}
}
