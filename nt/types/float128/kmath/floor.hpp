#ifndef NT_MATH_KMATH_FLOAT128_FLOOR_HPP__
#define NT_MATH_KMATH_FLOAT128_FLOOR_HPP__

#include "../float128_impl.h"
#include "trunc.hpp"
#include "../../../math/kmath/floor_decl.h"
#include "../../../utils/type_traits.h"
#include "../../../utils/always_inline_macro.h"

// constexpr math functions
namespace nt::math::kmath{

template<>
NT_ALWAYS_INLINE constexpr float128_t floor(float128_t arg) noexcept {
    // if true -> negative
    float128_t t = trunc<float128_t>(arg);
    return arg.get_bits().sign() ? 
        ((arg == t) ? t : t-1) : t;
}



}

#endif
