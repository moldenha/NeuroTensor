#ifndef NT_MATH_FLOAT128_ROUND_HPP__
#define NT_MATH_FLOAT128_ROUND_HPP__

#include "../float128_impl.h"
#include "../../../utils/always_inline_macro.h"
#include "../kmath/round.hpp"

// constexpr math functions
namespace nt::math{


NT_ALWAYS_INLINE float128_t round(const float128_t& x) noexcept {
    return kmath::round(x);
}


}

#endif
