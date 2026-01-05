#ifndef NT_MATH_KMATH_TRUNC_HPP__
#define NT_MATH_KMATH_TRUNC_HPP__

#include "../../utils/type_traits.h"
#include "../../utils/always_inline_macro.h"
#include "floor.hpp"
#include "ceil.hpp"
#include "config.hpp"
#include "trunc_decl.h"
#ifndef NT_KMATH_NO_INCLUDE_F128_
#include NT_KMATH_INCLUDE_F128_(trunc)
#endif
namespace nt::math::kmath{


template<class Real>
NT_ALWAYS_INLINE constexpr Real trunc(Real arg) noexcept {
    NT_KMATH_FUNCTION_CHECK(trunc);
    return (arg > 0) ? ::nt::math::kmath::floor(arg) : ::nt::math::kmath::ceil(arg);
}

}

#endif
