#include "../cpu/round.h"
#include "exceptions.hpp"
#include "round.h"

namespace nt{
namespace functional{


Tensor round(const Tensor& input, int64_t decimals){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    utils::throw_exception(DTypeFuncs::is_floating(input.dtype()) || DTypeFuncs::is_complex(input.dtype()), "Can only call round on a non-floating type");
    Tensor out(input.shape(), input.dtype());
    if(decimals == 0)
        cpu::_round(input.arr_void(), out.arr_void());
    else
        cpu::_round_decimal(input.arr_void(), out.arr_void(), decimals);
    return std::move(out);
}


Tensor floor(const Tensor& input){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    utils::throw_exception(DTypeFuncs::is_floating(input.dtype()) || DTypeFuncs::is_complex(input.dtype()), "Can only call floor on a non-floating type");
    Tensor out(input.shape(), input.dtype());
    cpu::_floor(input.arr_void(), out.arr_void());
    return std::move(out);
} 

Tensor ceil(const Tensor& input){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    utils::throw_exception(DTypeFuncs::is_floating(input.dtype()) || DTypeFuncs::is_complex(input.dtype()), "Can only call ceil on a non-floating type");
    Tensor out(input.shape(), input.dtype());
    cpu::_ceil(input.arr_void(), out.arr_void());
    return std::move(out);
    
}
Tensor trunc(const Tensor& input){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    utils::throw_exception(DTypeFuncs::is_floating(input.dtype()) || DTypeFuncs::is_complex(input.dtype()), "Can only call trunc on a non-floating type");
    Tensor out(input.shape(), input.dtype());
    cpu::_trunc(input.arr_void(), out.arr_void());
    return std::move(out);

}

}
}
