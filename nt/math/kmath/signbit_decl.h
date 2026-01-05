#ifndef NT_MATH_FUNCTIONAL_SIGNBIT_DECL_H__
#define NT_MATH_FUNCTIONAL_SIGNBIT_DECL_H__

#include "decl.h"
namespace nt::math::kmath{

template<typename Real>
NT_ALWAYS_INLINE constexpr bool signbit(Real val) noexcept;

}


#endif
