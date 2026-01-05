#ifndef NT_MATH_KMATH_ABS_HPP__
#define NT_MATH_KMATH_ABS_HPP__

#include "abs_decl.h"
#include "../../utils/type_traits.h"
#include "../../utils/always_inline_macro.h"
#include "config.hpp"

#ifndef NT_KMATH_NO_INCLUDE_F128_
#include NT_KMATH_INCLUDE_F128_(abs)
#endif

namespace nt::math::kmath{


template<class Real>
NT_ALWAYS_INLINE constexpr Real abs(Real arg) noexcept {
    NT_KMATH_FUNCTION_CHECK(abs);
    if(arg == Real(-0)){return Real(0);}
    return (arg > 0) ? arg : -arg;
}

}

#endif
