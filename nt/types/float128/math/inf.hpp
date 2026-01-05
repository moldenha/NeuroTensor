#ifndef NT_MATH_FLOAT128_INF_HPP__
#define NT_MATH_FLOAT128_INF_HPP__

#include "../float128_impl.h"
#include "../../../utils/always_inline_macro.h"
#include "../kmath/inf.hpp"
#include "../../../math/functional/inf_decl.h"

// constexpr math functions
namespace nt::math{

template<>
inline constexpr float128_t inf() noexcept {
    return kmath::generate_inf<float128_t>();
}

template<>
inline constexpr float128_t neg_inf() noexcept {
    return kmath::generate_neg_inf<float128_t>();
}

inline constexpr bool isinf(const float128_t& x) noexcept {
    return kmath::isinf<float128_t>(x);
}

}

#endif
