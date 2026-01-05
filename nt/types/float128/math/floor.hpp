#ifndef NT_MATH_FLOAT128_FLOOR_HPP__
#define NT_MATH_FLOAT128_FLOOR_HPP__

#include "../float128_impl.h"
#include "../../../utils/always_inline_macro.h"
#include "../kmath/floor.hpp"

// constexpr math functions
namespace nt::math{


NT_ALWAYS_INLINE float128_t floor(const float128_t& x) noexcept {
    return kmath::floor(x);
}


}

#endif
