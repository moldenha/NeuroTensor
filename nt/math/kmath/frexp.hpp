#ifndef NT_MATH_KMATH_FREXP_HPP__
#define NT_MATH_KMATH_FREXP_HPP__

#include "../../utils/type_traits.h"
#include "config.hpp"
#include "frexp_decl.h"
#ifndef NT_KMATH_NO_INCLUDE_F128_
#include NT_KMATH_INCLUDE_F128_(frexp)
#endif

namespace nt::math::kmath{


template<typename Real>
inline constexpr Real frexp(Real arg, int* exp){
    NT_KMATH_FUNCTION_CHECK(frexp);
    const bool negative_arg = (arg < Real(0));
    
    Real f = negative_arg ? -arg : arg;
    int e2 = 0;
    constexpr Real two_pow_32 = Real(4294967296);

    while (f >= two_pow_32)
    {
        f = f / two_pow_32;
        e2 += 32;
    }

    while(f >= Real(1))
    {
        f = f / Real(2);
        ++e2;
    }
    
    if(exp != nullptr)
    {
        *exp = e2;
    }

    return !negative_arg ? f : -f;

}

}

#endif
