#ifndef NT_MATH_FLOAT128_LOG_HPP__
#define NT_MATH_FLOAT128_LOG_HPP__

#include "../float128_impl.h"
#include "../b256.h"
#include "../from_string.hpp"
#include "../../../utils/type_traits.h"
#include "../../../utils/always_inline_macro.h"
#include "constants.h"
#include "../kmath/round.hpp"
#include "../kmath/ldexp.hpp"

// constexpr math functions
namespace nt::math{


inline float128_t log(const float128_t& x) noexcept {
    float128_bits bits = x.get_bits();
    if(bits.is_nan()) return x;
    if(bits.is_zero()) return float128_t::make_inf(true); // -inf
    if(bits.sign()) return float128_t::make_nan();
    if(bits.is_inf()) return x;
    
    
    int32_t e = int32_t(bits.exponent()) - 16383;
    float128_t m = float128_t( 
        float128_bits::pack(
            false,
            16383,
            bits.mantissa_with_hidden_bit()
        )
    );

    // force m into [sqrt(1/2), sqrt(2)]
    if (m > f128_constants::SQRT2) {
        m *= f128_constants::HALF;
        ++e;
    }

    float128_t f = m - f128_constants::ONE;

    // log(1+f) polynomial
    float128_t term = f;
    float128_t sum = f;

    for (int i = 2; i <= 10; ++i) {
        term *= -f;
        sum += term / float128_t::from_integer(i);
    }

    return sum + float128_t::from_integer(e) * f128_constants::LN2;
}


}

#endif
