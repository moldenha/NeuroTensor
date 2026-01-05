#ifndef NT_MATH_FLOAT128_COS_HPP__
#define NT_MATH_FLOAT128_COS_HPP__

#include "../float128_impl.h"
#include "../b128.h"
#include "../b256.h"
#include "../from_string.hpp"
#include "../../../utils/always_inline_macro.h"
#include "constants.h"
#include "trig.hpp"

// constexpr math functions
namespace nt::math{

inline float128_t cos(const float128_t& x) noexcept {
    float128_bits bits = x.get_bits();
    if(bits.is_nan() || bits.is_inf()) return float128_t::make_nan();
    if(bits.is_zero()) return float128_t(1);

    auto [q, r] = trig::reduce_trig(x);
    switch(q){
        case 0 : return trig::cos_kernel(r);
        case 1 : return -trig::sin_kernel(r);
        case 2 : return -trig::cos_kernel(r);
        default: return trig::sin_kernel(r);
    }
}

}

#endif
