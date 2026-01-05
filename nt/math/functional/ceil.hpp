#ifndef NT_MATH_FUNCTIONAL_CEIL_HPP__
#define NT_MATH_FUNCTIONAL_CEIL_HPP__

#include "utils.h"
#include "general_include.h"
#include "trunc.hpp"

#include NT_MAKE_FLOAT128_MATH_FUNCTION_ACCESSIBLE__(ceil) // <nt/types/float128/math/ceil.hpp>

namespace nt::math{

NT_ALWAYS_INLINE float16_t ceil(const float16_t& val) noexcept {
    float16_t int_part = trunc(val);
    return val > int_part ? int_part + 1 : int_part;
}

NT_ALWAYS_INLINE float ceil(const float& val) noexcept {
    float int_part = trunc(val);
    return val > int_part ? int_part + 1 : int_part;
}

NT_ALWAYS_INLINE double ceil(const double& val) noexcept {
    double int_part = trunc(val);
    return val > int_part ? int_part + 1 : int_part;
}


#define NT_CEIL_INTEGER_(type, name_a, name_b)\
NT_ALWAYS_INLINE constexpr type ceil(const type& val) noexcept { return val; }\


#define NT_CEIL_COMPLEX_(type, name_a, name_b)\
NT_ALWAYS_INLINE constexpr type ceil(const type& val) noexcept {\
    return type(ceil(val.real()), ceil(val.imag())); \
}\

NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_(NT_CEIL_INTEGER_) 
NT_GET_DEFINE_UNSIGNED_INTEGER_DTYPES_(NT_CEIL_INTEGER_) 
NT_GET_DEFINE_COMPLEX_DTYPES_(NT_CEIL_COMPLEX_) 

#undef NT_CEIL_INTEGER_ 
#undef NT_CEIL_COMPLEX_ 

}

#endif
