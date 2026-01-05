#ifndef NT_MATH_KMATH_FLOAT128_COPYSIGN_HPP__
#define NT_MATH_KMATH_FLOAT128_COPYSIGN_HPP__

#include "../float128_impl.h"
#include "../../../math/kmath/copysign_decl.h"
#include "../../../utils/type_traits.h"
#include "../../../utils/always_inline_macro.h"

// constexpr math functions
namespace nt::math::kmath{

template<>
NT_ALWAYS_INLINE constexpr float128_t copysign(const float128_t mag, const float128_t sgn) noexcept {
    return sgn.get_bits().sign() ? float128_t(mag.get_bits().negative_sign_bit()) : float128_t(mag.get_bits().positive_sign_bit());
}



}

#endif
