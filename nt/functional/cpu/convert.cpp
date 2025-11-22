#include "convert.h"
#include <stdexcept>
#include "../../dtype/ArrayVoid.hpp"
#include "../../convert/Convert.h"

namespace nt{
namespace functional{
namespace cpu{




void _convert(const ArrayVoid& in, ArrayVoid& out){
    in.cexecute_function<WRAP_DTYPES<AllTypesL> >(
    [&out](auto in_begin, auto in_end){
        out.execute_function<WRAP_DTYPES<AllTypesL> >(
        [&](auto o_begin, auto o_end){
            using to_t = utils::IteratorBaseType_t<decltype(o_begin)>;
            for(;o_begin != o_end; ++o_begin, ++in_begin){
                *o_begin = ::nt::convert::convert<to_t>(*in_begin);
            }
        });
        
    });
}

template<typename T>
void _internal_convert_(ArrayVoid& in){
    in.execute_function<WRAP_DTYPES<AllTypesL>>(
    [](auto begin, auto end){
        using from_t = utils::IteratorBaseType_t<decltype(begin)>;
        if constexpr (sizeof(from_t) == sizeof(T)){
            for(;begin != end; ++begin){
                *((T*)&(begin[0])) = ::nt::convert::convert<T>(*begin);
            }
        }
    });
}


void _internal_convert_(ArrayVoid& in, DType dt){
    ArrayVoid arr(1, dt);
    arr.execute_function<WRAP_DTYPES<AllTypesL>>(
    [&in](auto begin, auto end){
        using to_t = utils::IteratorBaseType_t<decltype(begin)>;
        _internal_convert_<to_t>(in);
    });
}

void _floating_(ArrayVoid& in){
    DType floating_to = DTypeFuncs::floating_size(DTypeFuncs::size_of_dtype(in.dtype()));
    if(!DTypeFuncs::is_floating(floating_to)){
        throw std::invalid_argument("Cannot convert current dtype to float internally");
    }
    _internal_convert_(in, floating_to);
}

void _complex_(ArrayVoid& in){
    DType complex_to = DTypeFuncs::complex_size(DTypeFuncs::size_of_dtype(in.dtype()));
    if(!DTypeFuncs::is_complex(complex_to)){
        throw std::invalid_argument("Cannot convert current dtype to complex internally");
    }
    _internal_convert_(in, complex_to);
}

void _integer_(ArrayVoid& in){
    DType integer_to = DTypeFuncs::integer_size(DTypeFuncs::size_of_dtype(in.dtype()));
    if(!DTypeFuncs::is_integer(integer_to)){
        throw std::invalid_argument("Cannot convert current dtype to integer internally");
    }
    _internal_convert_(in, integer_to);
}

void _unsigned_(ArrayVoid& in){
    DType unsigned_to = DTypeFuncs::unsigned_size(DTypeFuncs::size_of_dtype(in.dtype()));
    if(!DTypeFuncs::is_unsigned(unsigned_to)){
        throw std::invalid_argument("Cannot convert current dtype to unsigned integer internally");
    }
    _internal_convert_(in, unsigned_to);
}


}
}
}
