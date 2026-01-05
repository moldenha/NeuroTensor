#ifndef NT_MATH_FLOAT128_NAN_HPP__
#define NT_MATH_FLOAT128_NAN_HPP__

#include "../float128_impl.h"
#include "../../../utils/always_inline_macro.h"
#include "../kmath/nan.hpp"
#include "../../../math/functional/nan_decl.h"

// constexpr math functions
namespace nt::math{

template<>
inline constexpr float128_t nan() noexcept {
    return kmath::generate_qNaN<float128_t>();
}

inline bool isnan(const float128_t& x) noexcept {
    return kmath::isnan<float128_t>(x);
}

}

#endif
