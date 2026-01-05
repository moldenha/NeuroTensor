#ifndef NT_MATH_FUNCTIONAL_COPYSIGN_DECL_H__
#define NT_MATH_FUNCTIONAL_COPYSIGN_DECL_H__

#include "decl.h"
namespace nt::math::kmath{
template <typename Real>
NT_ALWAYS_INLINE constexpr Real copysign(const Real mag, const Real sgn) noexcept;
}

#endif
