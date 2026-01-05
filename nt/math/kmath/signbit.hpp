#ifndef NT_MATH_KMATH_SIGNBIT_HPP__
#define NT_MATH_KMATH_SIGNBIT_HPP__

#include "../../utils/type_traits.h"
#include "../../utils/always_inline_macro.h"
#include "abs.hpp"
#include "config.hpp"
#include "signbit_decl.h"
#ifndef NT_KMATH_NO_INCLUDE_F128_
#include NT_KMATH_INCLUDE_F128_(signbit)
#endif

namespace nt::math::kmath{

template<typename Real>
NT_ALWAYS_INLINE constexpr bool signbit(Real val) noexcept {
    NT_KMATH_FUNCTION_CHECK(signbit);
    if(val == Real(0.0)){
        return false;
    }else if(val == Real(-0.0)){
        return true;
    }else{
        return val < Real(0.0);
    }
}



}

#endif
