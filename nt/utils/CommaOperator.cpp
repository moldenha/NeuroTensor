#include "CommaOperator.h"

namespace nt{

#define NT_DEFINE_COMMA_SET_FUNCTION_(dtype, type) \
std::size_t DTypeFuncs::detail_comma_operator::set##dtype (void* _ptr, Scalar _s){\
    type s = _s.to<type>();\
    type* ptr = reinterpret_cast<type*>(_ptr);\
    *ptr = s;\
    return sizeof(type);\
}

#define X(type, dtype_enum_a, dtype_enum_b)\
NT_DEFINE_COMMA_SET_FUNCTION_(dtype_enum_a, type)\

NT_GET_X_FLOATING_DTYPES_ 
NT_GET_X_COMPLEX_DTYPES_
NT_GET_X_SIGNED_INTEGER_DTYPES_
NT_GET_X_UNSIGNED_INTEGER_DTYPES_

NT_DEFINE_COMMA_SET_FUNCTION_(Bool, uint_bool_t)

#undef X
#undef NT_DEFINE_COMMA_SET_FUNCTION_

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
