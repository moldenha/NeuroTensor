#ifndef NT_MATH_FLOAT128_TANH_HPP__
#define NT_MATH_FLOAT128_TANH_HPP__

#include "../float128_impl.h"
#include "../../../utils/always_inline_macro.h"
#include "exp.hpp"

// constexpr math functions
namespace nt::math{

// [(e + (1 / e)) / 2] / [(e - (1 / e)) / 2]
// == [(e + (1 / e)) * 2] / [2 * (e - (1 / e))]
// == (e + (1 / e)) / (e - (1 / e))

NT_ALWAYS_INLINE float128_t tanh(const float128_t& x) noexcept {
    float128_t e = ::nt::math::exp(x);
    float128_t neg_e = float128_t(1) / e;
    float128_t cosh_ = (e + neg_e);
    float128_t sinh_ = (e - neg_e); 
    return sinh_ / cosh_;
    // return (::nt::math::sinh(x) / ::nt::math::cosh(-x));
}

}

#endif
