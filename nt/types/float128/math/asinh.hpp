#ifndef NT_MATH_FLOAT128_ASINH_HPP__
#define NT_MATH_FLOAT128_ASINH_HPP__

#include "../float128_impl.h"
#include "../b128.h"
#include "../../../utils/always_inline_macro.h"
#include "constants.h"
#include "sqrt.hpp"
#include "log.hpp"
#include "../kmath/abs.hpp"

// constexpr math functions
namespace nt::math{

inline float128_t asinh(const float128_t& x) noexcept {
    float128_bits bits = x.get_bits();
    if(bits.is_nan()) return x;
    
    float128_t ax = kmath::abs(x);
    // small x -> good numerically
    if(ax < f128_constants::ASINH_BOUND_SMALL)
        return x;
    
    // large x -> log(2x)
    if(ax > f128_constants::ASINH_BOUND_LARGE){
        float128_t r = ::nt::math::log(ax) + f128_constants::LN2;
        return bits.sign() ? -r : r;
    }

    float128_t r = ::nt::math::log(ax + ::nt::math::sqrt(ax * ax + 1));
    return bits.sign() ? -r : r;

}

}

#endif
