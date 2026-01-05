#ifndef NT_MATH_KMATH_INF_DECL_H__
#define NT_MATH_KMATH_INF_DECL_H__

namespace nt::math::kmath{

template<typename FP>
inline constexpr FP generate_inf() noexcept;

template<typename FP>
inline constexpr FP generate_neg_inf() noexcept;

template<class Real>
constexpr bool isinf(Real arg) noexcept;

}

#endif


