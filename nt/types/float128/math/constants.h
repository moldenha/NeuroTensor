#ifndef NT_FLOAT128_MATH_CONSTANTS
#define NT_FLOAT128_MATH_CONSTANTS

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicitly-unsigned-literal"
#endif

#include "../float128_impl.h"
#include "../from_string.hpp"


// The reason I am using this macro instead of the _f128 operator:
//  - compile times are faster when using the number just from an integer
//  - While the run times would stay the same, this just really speeds up compile
//      times without having to deal with strings and division and multiplication operators

#define NT_F128_CONST_FROM_INT(hi_bits, lo_bits)\
    ::nt::float128_t(::nt::float128_bits(::nt::b128(uint64_t(hi_bits), uint64_t(lo_bits))))

namespace nt::math::f128_constants{

inline constexpr float128_t INV_LN2 = NT_F128_CONST_FROM_INT(uint64_t(4611529151027001391), uint64_t(16246591688395510317));
// inline constexpr float128_t INV_LN2 = 1.4426950408889634073599246810018921_f128; 
inline constexpr float128_t LN2 = NT_F128_CONST_FROM_INT(uint64_t(4611231800670266270), uint64_t(17534646158829291480));
// inline constexpr float128_t LN2 = 0.6931471805599453094172321214581766_f128;


// underflow/overflow clamps
inline constexpr float128_t EXP_MAX = NT_F128_CONST_FROM_INT(uint64_t(4615172450344215454), uint64_t(17534207595406138384));
// inline constexpr float128_t EXP_MAX = 11356.5234062941439488_f128;
inline constexpr float128_t EXP_MIN = NT_F128_CONST_FROM_INT(uint64_t(13838545963813049285), uint64_t(8453824611189993686));
// inline constexpr float128_t EXP_MIN = -11399.4985314888605581_f128;

inline constexpr float128_t SQRT2 = NT_F128_CONST_FROM_INT(uint64_t(4611521134203499452), uint64_t(13838545963813049285));
// inline constexpr float128_t SQRT2 = 1.414213562373095048801688724209698079_f128;

inline constexpr float128_t PI = NT_F128_CONST_FROM_INT(uint64_t(4611846683310179025), uint64_t(9541308523256152487));
// inline constexpr float128_t PI = 3.14159265358979323846264338327950288_f128;
inline constexpr float128_t PI_2 = NT_F128_CONST_FROM_INT(uint64_t(4611565208333468369), uint64_t(9541308523256152487));
// inline constexpr float128_t PI_2    = PI / float128_t(2);
inline constexpr float128_t PI_4 = NT_F128_CONST_FROM_INT(uint64_t(4611283733356757713), uint64_t(9541308523256152487));
// inline constexpr float128_t PI_4    = PI / float128_t(4);
inline constexpr float128_t INV_PI_2 = NT_F128_CONST_FROM_INT(uint64_t(4611199978568457352), uint64_t(3050054389497850488));
// inline constexpr float128_t INV_PI_2    = float128_t(2) / PI;
inline constexpr float128_t TAN_PI_8 = NT_F128_CONST_FROM_INT(uint64_t(4611026481531834099), uint64_t(2603867750348335686));
// inline constexpr float128_t TAN_PI_8 = 0.414213562373095048801688724209698_f128;

inline constexpr float128_t ASINH_BOUND_SMALL = NT_F128_CONST_FROM_INT(uint64_t(4602036489593650139), uint64_t(12357150490933114128));
// inline constexpr float128_t ASINH_BOUND_SMALL = 0.0000000001_f128;
inline constexpr float128_t ASINH_BOUND_LARGE = NT_F128_CONST_FROM_INT(uint64_t(4620739422705418240), uint64_t(0));
// inline constexpr float128_t ASINH_BOUND_LARGE = float128_t(uint64_t(10000000000));

inline constexpr float128_t HALF = NT_F128_CONST_FROM_INT(uint64_t(4611123068473966592), uint64_t(0));
inline constexpr float128_t ONE = NT_F128_CONST_FROM_INT(uint64_t(4611404543450677248), uint64_t(0));
//inline constexpr float128_t HALF = 0.5_f128;


}

#undef NT_F128_CONST_FROM_INT

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif
