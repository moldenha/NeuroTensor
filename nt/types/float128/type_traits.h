#ifndef NT_TYPES_FLOAT128_TYPE_TRAITS_H__
#define NT_TYPES_FLOAT128_TYPE_TRAITS_H__

#include "../../utils/type_traits.h"
#include "float128_impl.h"
#include "b128.h"
#include "kmath/ldexp.hpp"
#include <limits>

namespace std{

template<>
class numeric_limits<nt::float128_t> {
public:
    static constexpr bool is_specialized = true;

    static constexpr nt::float128_t min() noexcept {
        // smallest positive normal
        return nt::float128_t(nt::float128_bits::pack(false, 1, nt::b128(0, 0)));
    }

    static constexpr nt::float128_t max() noexcept {
        // largest finite value
        return nt::float128_t(nt::float128_bits::pack(
            false,
            0x7FFE,                     // max exponent before INF
            (nt::b128(1) << 112) - 1        // all mantissa bits set
        ));
    }

    static constexpr nt::float128_t lowest() noexcept {
        return -max();
    }

    static constexpr int digits       = 113; // includes hidden bit
    static constexpr int digits10     = 33;  // IEEE quad precision
    static constexpr int max_digits10 = 36;

    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;

    static constexpr int radix = 2;

    static constexpr nt::float128_t epsilon() noexcept {
        // difference between 1 and next representable
        return nt::math::kmath::ldexp(nt::float128_t(1), -112);
    }

    static constexpr nt::float128_t round_error() noexcept {
        return 0.5_f128;
    }

    static constexpr int min_exponent   = -16382; // unbiased
    static constexpr int min_exponent10 = -4931;

    static constexpr int max_exponent   = 16384;
    static constexpr int max_exponent10 = 4932;

    static constexpr bool has_infinity      = true;
    static constexpr bool has_quiet_NaN     = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr float_denorm_style has_denorm = denorm_present;

    static constexpr bool has_denorm_loss = true;

    static constexpr nt::float128_t infinity() noexcept {
        return nt::float128_t::make_inf(false);
    }

    static constexpr nt::float128_t quiet_NaN() noexcept {
        return nt::float128_t::make_nan();
    }

    static constexpr nt::float128_t signaling_NaN() noexcept {
        return nt::float128_t::make_snan();
    }

    static constexpr nt::float128_t denorm_min() noexcept {
        // Smallest positive subnormal:
        return nt::float128_t(nt::float128_bits::pack(false, 0, nt::b128(0, 1)));
    }

    static constexpr bool is_iec559  = true;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo  = false;

    static constexpr bool traps = false;  // depends on platform
    static constexpr bool tinyness_before = false;

    static constexpr float_round_style round_style = round_to_nearest;
};

template<>
class numeric_limits<const nt::float128_t>
    : public numeric_limits<nt::float128_t> {};

}

namespace nt::type_traits{

template<>
struct is_floating_point<float128_t> : true_type {};
template<>
struct is_floating_point<const float128_t> : true_type {};
template<>
struct is_floating_point<const volatile float128_t> : true_type {};
template<>
struct is_floating_point<volatile float128_t> : true_type {};

// template<>
// struct make_unsigned<float128_t>{
//     using type = b128;
// };

template<>
struct numeric_num_digits<float128_t>{
    static constexpr std::size_t value = 113;
};


template<>
class numeric_limits<float128_t> : public std::numeric_limits<float128_t> {};
template<>
class numeric_limits<const float128_t> : 
        public std::numeric_limits<const float128_t> {};

}


#endif
