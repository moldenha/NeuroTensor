#ifndef NT_FLOAT128_MATH_CONSTANTS
#define NT_FLOAT128_MATH_CONSTANTS

#include "../float128_impl.h"
#include "../from_string.hpp"

namespace nt::math::f128_constants{

inline constexpr float128_t INV_LN2 = 1.4426950408889634073599246810018921_f128; 
inline constexpr float128_t LN2 = 0.6931471805599453094172321214581766_f128;


// underflow/overflow clamps
inline constexpr float128_t EXP_MAX = 11356.5234062941439488_f128;
inline constexpr float128_t EXP_MIN = -11399.4985314888605581_f128;

inline constexpr float128_t SQRT2 = 1.414213562373095048801688724209698079_f128;

inline constexpr float128_t PI = 3.14159265358979323846264338327950288_f128;
inline constexpr float128_t PI_2    = PI / float128_t(2);
inline constexpr float128_t PI_4    = PI / float128_t(4);
inline constexpr float128_t INV_PI_2    = float128_t(2) / PI;
inline constexpr float128_t TAN_PI_8 = 0.414213562373095048801688724209698_f128;

inline constexpr float128_t ASINH_BOUND_SMALL = float128_t::from_double(1e-10);
inline constexpr float128_t ASINH_BOUND_LARGE = float128_t::from_double(1e+10);

}


#endif
