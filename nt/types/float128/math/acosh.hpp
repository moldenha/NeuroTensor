#ifndef NT_MATH_FLOAT128_ACOSH_HPP__
#define NT_MATH_FLOAT128_ACOSH_HPP__

#include "../float128_impl.h"
#include "../b128.h"
#include "../../../utils/always_inline_macro.h"
#include "constants.h"
#include "sqrt.hpp"
#include "log.hpp"

// constexpr math functions
namespace nt::math{

inline float128_t acosh(const float128_t& x) noexcept {
    float128_bits bits = x.get_bits();
    if(bits.is_nan()) return x;
    
    // domain violation
    if(x < 1)
        return float128_t::make_nan();

    if(x == float128_t(1))
        return float128_t::make_zero();
    
    // large x -> log(2x)
    if(x > f128_constants::ASINH_BOUND_LARGE){
        return ::nt::math::log(x) + f128_constants::LN2;
    }

    return::nt::math::log(x + ::nt::math::sqrt(x * x - 1));
}

}

#endif
