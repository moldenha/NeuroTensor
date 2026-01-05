#ifndef NT_MATH_KMATH_INF_HPP__
#define NT_MATH_KMATH_INF_HPP__

#include "../../utils/type_traits.h"
#include "../../utils/always_inline_macro.h"
#include "../../bit/float_bits.h"
#include "inf_decl.h"
#ifndef NT_KMATH_NO_INCLUDE_F128_
#include NT_KMATH_INCLUDE_F128_(inf)
#endif
namespace nt::math::kmath{

template<typename FP>
inline constexpr FP generate_inf() noexcept {
    return type_traits::numeric_limits<FP>::infinity();
}

template<typename FP>
inline constexpr FP generate_neg_inf() noexcept {
    return -type_traits::numeric_limits<FP>::infinity();
}

template<class Real>
inline constexpr bool isinf(Real arg) noexcept {
    constexpr Real inf = generate_inf<type_traits::decay_t<Real>>();
    constexpr Real neg_inf = generate_neg_inf<type_traits::decay_t<Real>>();
    return arg == inf || arg == neg_inf;
}

}

#endif
