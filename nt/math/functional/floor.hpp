#ifndef NT_MATH_FUNCTIONAL_FLOOR_HPP__
#define NT_MATH_FUNCTIONAL_FLOOR_HPP__

#include "utils.h"
#include "general_include.h"
#include "trunc.hpp"

#include NT_MAKE_FLOAT128_MATH_FUNCTION_ACCESSIBLE__(floor) // <nt/types/float128/math/floor.hpp>

namespace nt::math{


NT_ALWAYS_INLINE float16_t floor(const float16_t& val){
    float16_t int_part = trunc(val);
    return val < int_part ? int_part - 1 : int_part;
}

NT_ALWAYS_INLINE float floor(const float& val){
    float int_part = trunc(val);
    return val < int_part ? int_part - 1 : int_part;
}

NT_ALWAYS_INLINE double floor(const double& val){
    double int_part = trunc(val);
    return val < int_part ? int_part - 1 : int_part;
}


#define NT_FLOOR_INTEGER_(type, name_a, name_b)\
NT_ALWAYS_INLINE type floor(const type& val){ return val; }\


#define NT_FLOOR_COMPLEX_(type, name_a, name_b)\
NT_ALWAYS_INLINE type floor(const type& val){\
    return type(floor(val.real()), floor(val.imag())); \
}\

NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_(NT_FLOOR_INTEGER_) 
NT_GET_DEFINE_UNSIGNED_INTEGER_DTYPES_(NT_FLOOR_INTEGER_) 
NT_GET_DEFINE_COMPLEX_DTYPES_(NT_FLOOR_COMPLEX_) 

#undef NT_FLOOR_INTEGER_ 
#undef NT_FLOOR_COMPLEX_ 

}

#endif
