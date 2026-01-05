#ifndef NT_MATH_KMATH_LDEXP_HPP__
#define NT_MATH_KMATH_LDEXP_HPP__

#include "../../utils/type_traits.h"
#include "config.hpp"
#include "ldexp_decl.h"
#ifndef NT_KMATH_NO_INCLUDE_F128_
#include NT_KMATH_INCLUDE_F128_(ldexp)
#endif
namespace nt::math::kmath{

template<class Real>
inline constexpr Real ldexp(Real arg, int exp) noexcept {
    NT_KMATH_FUNCTION_CHECK(ldexp)
    constexpr Real two = 2;
    while(exp > 0)
    {
        arg *= two;
        --exp;
    }
    while(exp < 0)
    {
        arg /= two;
        ++exp;
    }

    return arg;
}

}

#endif
