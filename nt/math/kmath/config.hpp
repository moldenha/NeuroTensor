/*
 * This creates a macro to ensure that the way the function is being used is correct
*/


#ifndef NT_KMATH_FUNCTION_CHECK
#define NT_KMATH_FUNCTION_CHECK(name)\
    static_assert(type_traits::is_floating_point_v<Real>, "Error, cannot perform kmath constexpr " #name " function on a type that isn't a real floating point");
#endif


#ifndef NT_KMATH_INCLUDE_F128_

#ifndef _NT_EXPAND_
#define _NT_EXPAND_(x) x
#endif

#ifndef NT_GLUE_EXPAND
#define NT_GLUE_EXPAND(x, y) _NT_EXPAND_(x)y
#endif

#ifndef NT_STRINGIFY
#define NT_STRINGIFY__(x) #x
#define NT_STRINGIFY(x) NT_STRINGIFY__(x)
#endif

// # define P_INCLUDE_FILE P_XSTR(P_CONCAT(Fonts/,P_CONCAT(LED_FONT,.h)))
// # include P_INCLUDE_FILE

#define NT_KMATH_INCLUDE_F128_(name) NT_STRINGIFY(NT_GLUE_EXPAND(../../types/float128/kmath/, NT_GLUE_EXPAND(name, .hpp)))


#endif

