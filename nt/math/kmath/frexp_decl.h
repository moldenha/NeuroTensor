#ifndef NT_MATH_FUNCTIONAL_FREXP_DECL_H__
#define NT_MATH_FUNCTIONAL_FREXP_DECL_H__

#include "../../utils/always_inline_macro.h"
namespace nt::math::kmath{
template<typename Real>
inline constexpr Real frexp(Real arg, int* exp = nullptr);
}


#endif
