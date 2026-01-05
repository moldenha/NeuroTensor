#ifndef NT_MATH_KMATH_FLOAT128_NAN_HPP__
#define NT_MATH_KMATH_FLOAT128_NAN_HPP__

#include "../float128_impl.h"
#include "../../../math/kmath/nan_decl.h"
#include "../../../utils/type_traits.h"
#include "../../../utils/always_inline_macro.h"

// constexpr math functions
namespace nt::math::kmath{

template<>
inline constexpr float128_t generate_qNaN() noexcept {
    return float128_t::make_nan();
}


template<>
inline constexpr bool isnan(float128_t arg) noexcept {
    return arg.get_bits().is_nan();
}



}

#endif
