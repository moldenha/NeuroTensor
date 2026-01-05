#ifndef NT_MATH_KMATH_NAN_DECL_H__
#define NT_MATH_KMATH_NAN_DECL_H__

namespace nt::math::kmath{

// by default does quiet
template<typename FP>
inline constexpr FP generate_qNaN() noexcept;

template<class Real>
inline constexpr bool isnan(Real arg) noexcept;

}

#endif
