#ifndef NT_MATH_FUNCTIONAL_FMOD_HPP__
#define NT_MATH_FUNCTIONAL_FMOD_HPP__

#include "utils.h"
#include "general_include.h"
#include <cmath>
#include NT_MAKE_FLOAT128_MATH_FUNCTION_ACCESSIBLE__(fmod)


namespace nt::math{ 

NT_ALWAYS_INLINE float fmod(float x, float y) {
    return std::fmod(x, y);
}

NT_ALWAYS_INLINE double fmod(double x, double y) {
    return std::fmod(x, y);
}

NT_ALWAYS_INLINE float16_t fmod(float16_t x, float16_t y) {
    return _NT_FLOAT32_TO_FLOAT16_(std::fmod(_NT_FLOAT16_TO_FLOAT32_(x), _NT_FLOAT16_TO_FLOAT32_(y)));
}

#define NT_COMPLEX_FMOD_(type, name_a, name_b)\
NT_ALWAYS_INLINE type fmod(type x, type y){\
    return type(fmod(x.real(), y.real()), fmod(x.imag(), y.imag()));\
}

NT_GET_DEFINE_COMPLEX_DTYPES_(NT_COMPLEX_FMOD_)

#undef NT_COMPLEX_FMOD_

} // nt::math:: 

#endif
