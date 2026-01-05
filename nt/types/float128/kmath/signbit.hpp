#ifndef NT_MATH_KMATH_FLOAT128_SIGNBIT_HPP__
#define NT_MATH_KMATH_FLOAT128_SIGNBIT_HPP__

#include "../float128_impl.h"
#include "../../../math/kmath/signbit_decl.h"
#include "../../../utils/type_traits.h"
#include "../../../utils/always_inline_macro.h"

// constexpr math functions
namespace nt::math::kmath{

template<>
NT_ALWAYS_INLINE constexpr bool signbit(float128_t arg) noexcept {
    return arg.get_bits().sign();
}



}

#endif
