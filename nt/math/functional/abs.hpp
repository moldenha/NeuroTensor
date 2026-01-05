#ifndef NT_MATH_FUNCTIONAL_ABS_HPP__
#define NT_MATH_FUNCTIONAL_ABS_HPP__

#include "utils.h"
#include "general_include.h"
#include "../../bit/bit_cast.h"

#include NT_MAKE_FLOAT128_MATH_FUNCTION_ACCESSIBLE__(abs) // <nt/types/float128/math/abs.hpp>

namespace nt::math{


// ABSOLUTE VALUE function
// assumes twos-complement [ pretty much always true ]
// fast
static_assert((-1 & 1) == 1,
              "This library requires two's-complement representation for absolute value, please change nt::math::abs func");

// U mask = uv >> (sizeof(T)*8 - 1); -> 0 if v >= 0, all-ones if v < 0
#define NT_ABS_SIGNED_INTEGER_(type, name_a, name_b)\
NT_ALWAYS_INLINE type abs(const type& a) noexcept {\
    using U = ::nt::type_traits::make_unsigned_t<type>;\
    U uv = static_cast<U>(a);\
    U mask = uv >> (sizeof(type)*CHAR_BIT - 1);\
    return static_cast<type>((uv ^ mask) - mask);\
}

#define NT_ABS_UNSIGNED_INTEGER_(type, name_a, name_b)\
NT_ALWAYS_INLINE type abs(const type& a) noexcept { return a; }

#define NT_ABS_COMPLEX_TYPE_(type, name_a, name_b)\
NT_ALWAYS_INLINE type abs(const type& a) { return type(abs(a.real()), abs(a.imag())); }


NT_ALWAYS_INLINE float abs(const float& a){ return std::abs(a); }
NT_ALWAYS_INLINE double abs(const double& a){ return std::abs(a); }
inline float16_t abs(const float16_t& a) noexcept {
    using U = uint16_t;
    U bits = ::nt::bit_cast<U>(a);
    bits &= 0x7FFFu;
    return ::nt::bit_cast<float16_t>(bits);
}

NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_(NT_ABS_SIGNED_INTEGER_) 
NT_GET_DEFINE_UNSIGNED_INTEGER_DTYPES_(NT_ABS_UNSIGNED_INTEGER_) 
NT_GET_DEFINE_COMPLEX_DTYPES_(NT_ABS_COMPLEX_TYPE_) 

#undef NT_ABS_SIGNED_INTEGER_ 
#undef NT_ABS_UNSIGNED_INTEGER_ 
#undef NT_ABS_COMPLEX_TYPE_ 


}

#endif
