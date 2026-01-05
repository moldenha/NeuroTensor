#ifndef NT_MATH_KMATH_CEIL_HPP__
#define NT_MATH_KMATH_CEIL_HPP__

#include "../../utils/type_traits.h"
#include "../../utils/always_inline_macro.h"
#include "floor.hpp"
#include "config.hpp"
#include "ceil_decl.h"
#ifndef NT_KMATH_NO_INCLUDE_F128_
#include NT_KMATH_INCLUDE_F128_(ceil)
#endif


namespace nt::math::kmath{


template<class Real>
NT_ALWAYS_INLINE constexpr Real ceil(Real arg) noexcept {
    NT_KMATH_FUNCTION_CHECK(ceil)
    Real result = ::nt::math::kmath::floor(arg);
    if(result == arg) return result;
    else return result + 1;
}

}

#endif
