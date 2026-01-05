//This is a header file that ensures that nt::float16_t is included
#ifndef _NT_TYPES_FLOAT16_ENSURE_H_
#define _NT_TYPES_FLOAT16_ENSURE_H_


#include <simde/simde-f16.h> //using this to convert between float32 and float16
#include "../utils/type_traits.h"

#define _HALF_FLOAT_SUPPORT_ //by default is going to have half float support
#ifdef SIMDE_FLOAT16_IS_SCALAR
namespace nt{
    using float16_t = simde_float16;
	#define _NT_FLOAT16_TO_FLOAT32_(f16) simde_float16_to_float32(f16)
	#define _NT_FLOAT32_TO_FLOAT16_(f)    simde_float16_from_float32(f)
	inline std::ostream& operator<<(std::ostream& os, const float16_t& val){
		return os << _NT_FLOAT16_TO_FLOAT32_(val);
	}
namespace type_traits{
constexpr bool simde_float16 = true;
}
}
#else
#include <half/half.hpp>
namespace nt{
	using float16_t = half_float::half;
	#define _NT_FLOAT16_TO_FLOAT32_(f16) float(f16)
	#define _NT_FLOAT32_TO_FLOAT16_(f) half_float::half(f)
	//by default has a way to print out the half_float::half
namespace type_traits{
constexpr bool simde_float16 = true;
}
}

namespace std{
inline ::nt::float16_t pow(const half_float::half& a, const half_float::half& b){
    return half_float::half(half_float::detail::pow(a, b));
}
}

#endif

namespace nt{
inline int16_t _NT_FLOAT16_TO_INT16_(float16_t fp) noexcept {
	int16_t out = *reinterpret_cast<int16_t*>(&fp);
	return out;
}

namespace type_traits{
template<>
struct is_floating_point<float16_t> : true_type {};
template<>
struct is_floating_point<const float16_t> : true_type {};
template<>
struct is_floating_point<const volatile float16_t> : true_type {};
template<>
struct is_floating_point<volatile float16_t> : true_type {};
template<>
struct make_unsigned<float16_t>{
    using type = uint16_t;
};

template<>
struct numeric_num_digits<float16_t>{
    static constexpr std::size_t value = 11;
};


}}



#ifdef SIMDE_FLOAT16_IS_SCALAR
// numeric_limits already defined in std for half float
// Therefore they are going to be defined here for the other version
#include "../bit/float_bits.h"

#if SIMDE_FLOAT16_API == SIMDE_FLOAT16_API_FLOAT16
//typedef _Float16 simde_float16;
#define NT_HALF_MAKE_BINARY_TO_FUNC__(name, binary)\
inline static constexpr ::nt::float16_t name() noexcept{\
    return _Float16(::nt::type_traits::numeric_limits<float>::name());\
}
#elif SIMDE_FLOAT16_API == SIMDE_FLOAT16_API_FP16_NO_ABI
// typedef struct { __fp16 value; } simde_float16;
#define NT_HALF_MAKE_BINARY_TO_FUNC__(name, binary)\
inline static constexpr ::nt::float16_t name() noexcept{\
    return simde_float16(__fp16(::nt::type_traits::numeric_limits<float>::name()));\
}

#elif SIMDE_FLOAT16_API == SIMDE_FLOAT16_API_FP16
//typedef __fp16 simde_float16;

#define NT_HALF_MAKE_BINARY_TO_FUNC__(name, binary)\
inline static constexpr ::nt::float16_t name() noexcept{\
    return __fp16(::nt::type_traits::numeric_limits<float>::name());\
}
#elif SIMDE_FLOAT16_API == SIMDE_FLOAT16_API_PORTABLE
//typedef struct { uint16_t value; } simde_float16;
#define NT_HALF_MAKE_BINARY_TO_FUNC__(name, binary)\
inline static constexpr ::nt::float16_t name() noexcept{\
    return simde_float16{.value = uint16_t(binary)};\
}
#else
#error "Error, simde float16 used when half library should have been used!"
#endif


// #define NT_HALF_MAKE_BINARY_TO_FUNC__(name, binary)\
// inline static constexpr ::nt::float16_t name() noexcept{\
//     return ::nt::float_bits<::nt::float16_t>(nt::bitset<16, uint16_t>(binary)).get();\
// }

// #define NT_HALF_MAKE_FLOAT16_FROM_FLOAT_(name)\
// inline static constexpr ::nt::float16_t name() noexcept {\
//     return _NT_FLOAT32_TO_FLOAT16_(::nt::type_traits::numeric_limits<float>::name());\
// }

namespace nt::type_traits{

template<> class numeric_limits<::nt::float16_t> : public numeric_limits<float> {
public:
    /// Supports signed values.
    static constexpr bool is_signed = true;

    /// Is not exact.
    static constexpr bool is_exact = false;

    /// Doesn't provide modulo arithmetic.
    static constexpr bool is_modulo = false;

    /// IEEE conformant.
    static constexpr bool is_iec559 = true;

    /// Supports infinity.
    static constexpr bool has_infinity = true;

    /// Supports quiet NaNs.
    static constexpr bool has_quiet_NaN = true;

    /// Supports subnormal values.
    static constexpr std::float_denorm_style has_denorm = std::denorm_present;

    /// Rounding mode.
    /// Due to the mix of internal single-precision computations (using the rounding mode of the underlying 
    /// single-precision implementation) with explicit truncation of the single-to-half conversions, the actual rounding 
    /// mode is indeterminate.
    static constexpr std::float_round_style round_style = std::numeric_limits<float>::round_style;

    /// Significant digits.
    static constexpr int digits = 11;

    /// Significant decimal digits.
    static constexpr int digits10 = 3;

    /// Required decimal digits to represent all possible values.
    static constexpr int max_digits10 = 5;

    /// Number base.
    static constexpr int radix = 2;

    /// One more than smallest exponent.
    static constexpr int min_exponent = -13;

    /// Smallest normalized representable power of 10.
    static constexpr int min_exponent10 = -4;

    /// One more than largest exponent
    static constexpr int max_exponent = 16;

    /// Largest finitely representable power of 10.
    static constexpr int max_exponent10 = 4;

    /// Smallest positive normal value.
    NT_HALF_MAKE_BINARY_TO_FUNC__(min, 0x0400);
    /// Smallest finite value.
    NT_HALF_MAKE_BINARY_TO_FUNC__(lowest, 0xFBFF);
    /// Largest finite value.
    NT_HALF_MAKE_BINARY_TO_FUNC__(max, 0x7BFF);
    /// Difference between one and next representable value.
    NT_HALF_MAKE_BINARY_TO_FUNC__(epsilon, 0x1400);
    /// Maximum rounding error.
    NT_HALF_MAKE_BINARY_TO_FUNC__(round_error, 0x3800);
    /// Positive infinity.
    NT_HALF_MAKE_BINARY_TO_FUNC__(infinity, 0x7C00);
    /// Quiet NaN.
    NT_HALF_MAKE_BINARY_TO_FUNC__(quiet_NaN, 0x7FFF);
    /// Signalling NaN.
    NT_HALF_MAKE_BINARY_TO_FUNC__(signaling_NaN, 0x7DFF)
    /// Smallest positive subnormal value.
    NT_HALF_MAKE_BINARY_TO_FUNC__(denorm_min, 0x0001);

};
}

#undef NT_HALF_MAKE_BINARY_TO_FUNC__ 

#endif



#endif
