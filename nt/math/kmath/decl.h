#ifndef NT_MATH_KMATH_DECL

#include "../../utils/always_inline_macro.h"

#define NT_MATH_KMATH_DECL(name)\
namespace nt::math::kmath{\
template<class Real>\
NT_ALWAYS_INLINE constexpr Real name(Real arg) noexcept;\
}

#endif
