#ifndef NT_MATH_KMATH_FLOOR_HPP__
#define NT_MATH_KMATH_FLOOR_HPP__

#include "../../utils/type_traits.h"
#include "../../utils/always_inline_macro.h"
#include "config.hpp"
#include "floor_decl.h"
#ifndef NT_KMATH_NO_INCLUDE_F128_
#include NT_KMATH_INCLUDE_F128_(floor)
#endif

namespace nt::math::kmath{

namespace details{

template <typename T>
inline constexpr T floor_pos_impl(T arg) noexcept
{
    constexpr T max_comp_val = T(1) / type_traits::numeric_limits<T>::epsilon();

    if (arg >= max_comp_val){
        return arg;
    }

    T result = 1;

    if(result < arg)
    {
        while(result < arg)
        {
            result *= 2;
        }
        while(result > arg)
        {
            --result;
        }

        return result;
    }
    else
    {
        return T(0);
    }
}

template <typename T>
inline constexpr T floor_neg_impl(T arg) noexcept
{
    T result = -1;

    if(result > arg)
    {
        while(result > arg)
        {
            result *= 2;
        }
        while(result < arg)
        {
            ++result;
        }
        if(result != arg)
        {
            --result;
        }
    }

    return result;
}

}

template<class Real>
NT_ALWAYS_INLINE constexpr Real floor(Real arg) noexcept {
    NT_KMATH_FUNCTION_CHECK(floor)
    if(arg == 0){return arg;}
    else if(arg > 0)
    {
        return details::floor_pos_impl(arg);
    }
    else
    {
        return details::floor_neg_impl(arg);
    }
}

}

#endif
