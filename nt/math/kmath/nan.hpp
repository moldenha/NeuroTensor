#ifndef NT_MATH_KMATH_NAN_HPP__
#define NT_MATH_KMATH_NAN_HPP__

#include "../../utils/type_traits.h"
#include "../../utils/always_inline_macro.h"
#include "../../bit/float_bits.h"
#include "nan_decl.h"

#ifndef NT_KMATH_NO_INCLUDE_F128_
#include NT_KMATH_INCLUDE_F128_(nan)
#endif

namespace nt::math::kmath{

// by default does quiet
template<typename FP>
inline constexpr FP generate_qNaN() noexcept {
    static_assert(type_traits::numeric_limits<FP>::has_quiet_NaN || type_traits::numeric_limits<FP>::has_signaling_NaN);
    if constexpr (type_traits::numeric_limits<FP>::has_quiet_NaN){
        return type_traits::numeric_limits<FP>::quiet_NaN();
    }else{
        return type_traits::numeric_limits<FP>::signaling_NaN();
    }
}


template<class Real>
inline constexpr bool isnan(Real arg) noexcept {
    return arg != arg;
}

}

#endif
