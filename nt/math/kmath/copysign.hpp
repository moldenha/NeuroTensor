#ifndef NT_MATH_KMATH_COPYSIGN_HPP__
#define NT_MATH_KMATH_COPYSIGN_HPP__

#include "../../utils/type_traits.h"
#include "../../utils/always_inline_macro.h"
#include "abs.hpp"
#include "signbit.hpp"
#include "config.hpp"
#include "copysign_decl.h"
#ifndef NT_KMATH_NO_INCLUDE_F128_
#include NT_KMATH_INCLUDE_F128_(copysign)
#endif

namespace nt::math::kmath{


template <typename Real>
NT_ALWAYS_INLINE constexpr Real copysign(const Real mag, const Real sgn) noexcept
{
    NT_KMATH_FUNCTION_CHECK(copysign);
    if(::nt::math::kmath::signbit(sgn)){
        return -::nt::math::kmath::abs(mag); 
    }else{
        return ::nt::math::kmath::abs(mag);
    }
}

}

#endif
