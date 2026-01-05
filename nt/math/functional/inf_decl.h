#ifndef NT_MATH_FUNCTIONAL_INF_DECL_H__
#define NT_MATH_FUNCTIONAL_INF_DECL_H__

// -inf 
// inf
// isinf

namespace nt::math{

template<typename T>
inline constexpr T inf() noexcept;

template<typename T>
inline constexpr T neg_inf() noexcept;

}

#endif
