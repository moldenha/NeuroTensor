#ifndef NT_MATH_KMATH_FLOAT128_ROUND_HPP__
#define NT_MATH_KMATH_FLOAT128_ROUND_HPP__

#include "../float128_impl.h"
#include "trunc.hpp"
#include "abs.hpp"
#include "floor.hpp"
#include "ceil.hpp"
#include "copysign.hpp"
#include "../../../utils/type_traits.h"
#include "../../../utils/always_inline_macro.h"

// constexpr math functions
namespace nt::math::kmath{


NT_ALWAYS_INLINE constexpr float128_t round(float128_t arg) noexcept
{
    constexpr float128_t half = 0.5_f128;

    float128_t t = ::nt::math::kmath::trunc(arg);
    float128_t frac = ::nt::math::kmath::abs(arg - t);

    if (frac > half)
        return t + (arg.get_bits().sign() ? float128_t(-1) : float128_t(1));

    if (frac < half)
        return t;

    // frac == 0.5 → round away from zero
    return t + (arg.get_bits().sign() ? float128_t(-1) : float128_t(1));
}


}

#endif
