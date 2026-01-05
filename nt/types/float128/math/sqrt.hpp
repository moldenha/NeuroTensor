#ifndef NT_MATH_FLOAT128_SQRT_HPP__
#define NT_MATH_FLOAT128_SQRT_HPP__

#include "../float128_impl.h"
#include "../b256.h"
#include "../../../utils/type_traits.h"
#include "../../../utils/always_inline_macro.h"

// constexpr math functions
namespace nt::math{

inline float128_t sqrt(const float128_t& arg) noexcept {
    float128_bits bits = arg.get_bits();
    if(bits.is_nan()) return arg;
    if(bits.is_zero()) return arg;
    if(bits.sign()) return float128_t::make_nan();
    if(bits.is_inf()) return arg;

    double initial_guess_d = std::sqrt(double(arg));
    float128_t y = float128_t(initial_guess_d);
    constexpr float128_t two = nt::float128_t(2);
    // Refine the guess using Newton-Raphson iteration (usually 3-4 iterations are enough for full precision)
    y = (y + arg / y) / two;
    y = (y + arg / y) / two;
    y = (y + arg / y) / two;
    y = (y + arg / y) / two; // More iterations increase precision

    return y;
}


}

#endif
