#include "CommaOperator.h"

namespace nt{

#define _NT_DEFINE_COMMA_SET_FUNCTION_(dtype, type) \
std::size_t DTypeFuncs::detail_comma_operator::set##dtype (void* _ptr, Scalar _s){\
    type s = _s.to<type>();\
    type* ptr = reinterpret_cast<type*>(_ptr);\
    *ptr = s;\
    return sizeof(type);\
}

_NT_DEFINE_COMMA_SET_FUNCTION_(Float, float)
_NT_DEFINE_COMMA_SET_FUNCTION_(Double, double)
_NT_DEFINE_COMMA_SET_FUNCTION_(Complex64, complex_64)
_NT_DEFINE_COMMA_SET_FUNCTION_(Complex128, complex_128)
_NT_DEFINE_COMMA_SET_FUNCTION_(int8, int8_t)
_NT_DEFINE_COMMA_SET_FUNCTION_(uint8, uint8_t)
_NT_DEFINE_COMMA_SET_FUNCTION_(uint16, uint16_t)
_NT_DEFINE_COMMA_SET_FUNCTION_(int16, int16_t)
_NT_DEFINE_COMMA_SET_FUNCTION_(int32, int32_t)
_NT_DEFINE_COMMA_SET_FUNCTION_(uint32, uint32_t)
_NT_DEFINE_COMMA_SET_FUNCTION_(int64, int64_t)
_NT_DEFINE_COMMA_SET_FUNCTION_(Bool, uint_bool_t)
_NT_DEFINE_COMMA_SET_FUNCTION_(Float16, float16_t)
_NT_DEFINE_COMMA_SET_FUNCTION_(Complex32, complex_32)
#ifdef _128_FLOAT_SUPPORT_
_NT_DEFINE_COMMA_SET_FUNCTION_(Float128, float128_t)
#endif
#ifdef __SIZEOF_INT128__
_NT_DEFINE_COMMA_SET_FUNCTION_(int128, int128_t)
_NT_DEFINE_COMMA_SET_FUNCTION_(uint128, uint128_t)
#endif

#undef _NT_DEFINE_COMMA_SET_FUNCTION_

CommaOperator::CommaOperator(DTypeFuncs::detail_comma_operator::SetFunc _s, void* _b, void* _e, DType _d)
:set_func(_s), begin(_b), end(_e), dt(_d)
{}

CommaOperator::CommaOperator(void* b, void* e, DType _dt)
:set_func(DTypeFuncs::detail_comma_operator::getSetFunc(_dt)), begin(b), end(e), dt(_dt)
{}


CommaOperator CommaOperator::operator,(Scalar s){
    utils::throw_exception(begin < end, "Already entered the maximum elements using the comma operator!");
    std::size_t increment = this->set_func(begin, s);
    return CommaOperator(set_func, reinterpret_cast<char*>(begin) + increment, end, dt);
} 
CommaOperator CommaOperator::operator,(Tensor){
    utils::throw_exception(false, "Comma operator for tensors not implemented yet, coming in future versions");
    return CommaOperator(set_func, begin, end, dt);
} 

}
