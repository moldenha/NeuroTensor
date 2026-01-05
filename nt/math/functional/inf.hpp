#ifndef NT_MATH_FUNCTIONAL_INF_HPP__
#define NT_MATH_FUNCTIONAL_INF_HPP__

// -inf 
// inf
// isinf

#include "../kmath/inf.hpp"
#include "../../bit/bitset.h"
#include "../../bit/float_bits.h"
#include "../../bit/kbit_cast.h"
#include "../../bit/bit_cast.h"
#include "../../utils/always_inline_macro.h"
#include "../../utils/type_traits.h"
#include "../../dtype/compatible/DTypeDeclareMacros.h"
#include "inf_decl.h"
#include "utils.h"
#include <cstdint>

namespace nt::math{

template<>
inline constexpr float16_t inf() noexcept {
    return kmath::generate_inf<float16_t>();
}

template<>
inline constexpr float inf() noexcept {
    return kmath::generate_inf<float>();
}

template<>
inline constexpr double inf() noexcept {
    return kmath::generate_inf<double>();
}

template<>
inline constexpr float16_t neg_inf() noexcept {
    return kmath::generate_neg_inf<float16_t>();
}

template<>
inline constexpr float neg_inf() noexcept {
    return kmath::generate_neg_inf<float>();
}

template<>
inline constexpr double neg_inf() noexcept {
    return kmath::generate_neg_inf<double>();
}



inline constexpr bool isinf(double val) noexcept {
    return (val == inf<double>() || val == neg_inf<double>());
}

inline constexpr bool isinf(float val) noexcept {
    return (val == inf<float>() || val == neg_inf<float>());
}

inline constexpr bool isinf(::nt::float16_t val) noexcept {
    return (val == inf<::nt::float16_t>() || val == neg_inf<::nt::float16_t>());
}





#define NT_INF_SIGNED_INTEGER_(type, name_a, name_b)\
template<>\
inline constexpr type inf() noexcept{\
    nt::bitset<sizeof(type) * CHAR_BIT, type> b{};\
    for(int i = 1; i < sizeof(type) * CHAR_BIT; ++i){\
        b.set(i, true);\
    }\
    return b.lo_type();\
}\
template<>\
inline constexpr type neg_inf() noexcept{\
    nt::bitset<sizeof(type) * CHAR_BIT, type> b{};\
    for(int i = 0; i < sizeof(type) * CHAR_BIT; ++i){\
        b.set(i, true);\
    }\
    return b.lo_type();\
}\
inline constexpr bool isinf(type val) noexcept {return false;}

#define NT_INF_UNSIGNED_INTEGER_(type, name_a, name_b)\
template<>\
inline constexpr type inf() noexcept{\
    nt::bitset<sizeof(type) * CHAR_BIT, type> b{};\
    for(int i = 0; i < sizeof(type) * CHAR_BIT; ++i){\
        b.set(i, true);\
    }\
    return b.lo_type();\
}\
template<>\
inline constexpr type neg_inf() noexcept { return 0; }\
inline constexpr bool isinf(type val) noexcept {return false;}


#define NT_INF_COMPLEX_TYPE_(type, name_a, name_b)\
template<>\
inline constexpr type inf() noexcept {\
    constexpr type::value_type val = inf<type::value_type>();\
    constexpr type ct = type(val, val);\
    return ct;\
}\
template<>\
inline constexpr type neg_inf() noexcept {\
    constexpr type::value_type val = neg_inf<type::value_type>();\
    constexpr type ct = type(val, val);\
    return ct;\
}\
inline constexpr bool isinf(type val) noexcept {return isinf(val.real()) && isinf(val.imag());}


NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_(NT_INF_SIGNED_INTEGER_);
NT_GET_DEFINE_UNSIGNED_INTEGER_DTYPES_(NT_INF_UNSIGNED_INTEGER_); 
NT_GET_DEFINE_COMPLEX_DTYPES_(NT_INF_COMPLEX_TYPE_);

#undef NT_INF_SIGNED_INTEGER_ 
#undef NT_INF_UNSIGNED_INTEGER_ 
#undef NT_INF_COMPLEX_TYPE_ 


}


#include NT_MAKE_FLOAT128_MATH_FUNCTION_ACCESSIBLE__(inf) // <nt/types/float128/math/inf.hpp>

#endif
