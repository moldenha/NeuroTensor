#ifndef NT_MATH_FUNCTIONAL_LDEXP_DECL_H__
#define NT_MATH_FUNCTIONAL_LDEXP_DECL_H__

#include "../../utils/always_inline_macro.h"

namespace nt::math::kmath{
template<class Real>
inline constexpr Real ldexp(Real arg, int exp) noexcept;
}

#endif
