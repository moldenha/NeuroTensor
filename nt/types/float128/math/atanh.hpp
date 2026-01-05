#ifndef NT_MATH_FLOAT128_ATANH_HPP__
#define NT_MATH_FLOAT128_ATANH_HPP__

#include "../float128_impl.h"
#include "../b128.h"
#include "../../../utils/always_inline_macro.h"
#include "sqrt.hpp"
#include "log.hpp"
#include "../kmath/abs.hpp"

// constexpr math functions
namespace nt::math{

inline float128_t atanh(const float128_t& x) noexcept {
    float128_bits bits = x.get_bits();
    if(bits.is_nan()) return x;
    
    float128_t ax = kmath::abs(x);
    if(ax >= 1)
        return float128_t::make_nan();

    if(bits.is_zero())
        return x;

    float128_t r = float128_t(0.5) * ::nt::math::log((float128_t(1) + ax) / (float128_t(1) - ax));
    return bits.sign() ? -r : r;

}

}

#endif
