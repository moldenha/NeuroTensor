#ifndef NT_MATH_KMATH_FLOAT128_CEIL_HPP__
#define NT_MATH_KMATH_FLOAT128_CEIL_HPP__

#include "../float128_impl.h"
#include "floor.hpp"
#include "../../../math/kmath/ceil_decl.h"
#include "../../../utils/type_traits.h"
#include "../../../utils/always_inline_macro.h"

// constexpr math functions
namespace nt::math::kmath{

template<>
NT_ALWAYS_INLINE constexpr float128_t ceil(float128_t arg) noexcept {
    float128_t res = floor(arg);
    return (res == arg) ? res : res + 1;
}



}

#endif
