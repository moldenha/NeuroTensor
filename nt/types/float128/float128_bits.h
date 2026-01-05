#ifndef NT_TYPES_128_BIT_TYPES_FLOAT128_BITS_H__
#define NT_TYPES_128_BIT_TYPES_FLOAT128_BITS_H__ 
// #include "../uint128.h"
#include "../../utils/always_inline_macro.h"
// #include "../../utils/type_traits.h"
// #include "../../math/kmath/ldexp.hpp"
// #include "../../math/kmath/abs.hpp"
#include "../../bit/float_bits.h"
#include "../../bit/bit_cast.h"
#include "../../bit/bitset.h"
// #include "../../bit/kbit_cast.h"
#include "../../math/kmath/ldexp.hpp"
#include "../../math/kmath/frexp.hpp"
#include "../../math/kmath/signbit.hpp"
#include <cstddef>
#include <vector>
#include <exception>
#include <string> 
#include <type_traits>
#include <iostream>
#include <cmath>

#include "b128.h"

namespace nt{


struct float128_bits{
    b128 bits;
    static constexpr b128 b128_0 = b128(0);
    static constexpr b128 b128_1 = b128(1);
    static constexpr std::size_t SIGN_BIT = 127;
    static constexpr std::size_t EXP_BITS = 15;
    static constexpr std::size_t FRAC_BITS = 112;
    static constexpr std::size_t EXP_SHIFT = FRAC_BITS;
    static constexpr b128 SIGN_MASK = (b128_1 << SIGN_BIT);
    static constexpr b128 EXP_MASK = (((b128_1 << EXP_BITS) - 1) << EXP_SHIFT);
    static constexpr b128 FRAC_MASK = ((b128_1 << FRAC_BITS) - 1);
    static constexpr b128 HIDDEN_BIT = (b128_1 << FRAC_BITS);

    constexpr float128_bits() : bits(0) {;}
    constexpr explicit float128_bits(b128 b) : bits(b) {;}
    
    NT_ALWAYS_INLINE static constexpr float128_bits from_bits(b128 b) noexcept { return float128_bits(b); }
    NT_ALWAYS_INLINE constexpr b128 raw() const { return bits; }

    NT_ALWAYS_INLINE static constexpr float128_bits make_zero(bool neg=false) noexcept {
        return float128_bits( neg ? (b128_1 << SIGN_BIT) : b128_0 );
    }
    
    NT_ALWAYS_INLINE static constexpr float128_bits make_inf(bool neg=false) noexcept {
        constexpr b128 b = (((b128_1 << EXP_BITS) -1) << EXP_SHIFT); // exp all ones
        if (neg) return float128_bits(b | (b128_1 << SIGN_BIT));
        return float128_bits(b);
    }
    NT_ALWAYS_INLINE static constexpr float128_bits make_nan() noexcept {
        constexpr b128 exp_all_ones = ((b128_1 << EXP_BITS) - 1) << EXP_SHIFT;
        constexpr b128 quiet_bit    = (b128_1 << (EXP_SHIFT - 1)); // bit 111
        return float128_bits(exp_all_ones | quiet_bit);
    }
    NT_ALWAYS_INLINE static constexpr float128_bits make_snan() noexcept {
        constexpr b128 exp_all_ones = ((b128_1 << EXP_BITS) - 1) << EXP_SHIFT;
        constexpr b128 payload_bit = b128_1;  // NOT the MSB
        return float128_bits(exp_all_ones | payload_bit);
    }



    // bitwise operators
    NT_ALWAYS_INLINE constexpr float128_bits operator<<(int s) const noexcept { return float128_bits(bits << s); }
    NT_ALWAYS_INLINE constexpr float128_bits operator>>(int s) const noexcept { return float128_bits(bits >> s); }
    NT_ALWAYS_INLINE constexpr float128_bits& operator<<=(int s) noexcept { bits <<= s; return *this; }
    NT_ALWAYS_INLINE constexpr float128_bits& operator>>=(int s) noexcept { bits >>= s; return *this; }
    NT_ALWAYS_INLINE constexpr float128_bits operator&(const float128_bits& f) const noexcept {return float128_bits(bits & f.bits);}
    NT_ALWAYS_INLINE constexpr float128_bits operator|(const float128_bits& f) const noexcept {return float128_bits(bits | f.bits);}
    NT_ALWAYS_INLINE constexpr float128_bits operator^(const float128_bits& f) const noexcept {return float128_bits(bits ^ f.bits);}
    NT_ALWAYS_INLINE constexpr float128_bits& operator&=(const float128_bits& f) noexcept {bits &= f.bits; return *this;}
    NT_ALWAYS_INLINE constexpr float128_bits& operator|=(const float128_bits& f) noexcept {bits |= f.bits; return *this;}
    NT_ALWAYS_INLINE constexpr float128_bits& operator^=(const float128_bits& f) noexcept {bits ^= f.bits; return *this;}
    NT_ALWAYS_INLINE constexpr float128_bits operator~() const noexcept {return float128_bits(~bits);}
    
    NT_ALWAYS_INLINE constexpr bool sign() const noexcept {
        return ((bits.hi_bits) & (1ULL <<(63)));
    }
    
    NT_ALWAYS_INLINE constexpr int exponent() const noexcept {
        return uint16_t(((bits.hi_bits) & EXP_MASK.hi_bits) >> (64 - 16));
    }

    NT_ALWAYS_INLINE constexpr b128 mantissa() const noexcept {
        return bits & FRAC_MASK;
    }
    NT_ALWAYS_INLINE constexpr b128 mantissa_with_hidden_bit() const noexcept {
        // 1. Get the fractional part
        b128 mantissa = bits & FRAC_MASK; 
        
        // 2. Get the exponent
        b128 exponent = bits & EXP_MASK;

        // 3. If the exponent is non-zero (normalized number)
        // Check for (exponent != 0) AND (exponent != all-ones)
        // We only check for != 0 here, as denormalized numbers (exponent == 0) 
        // DO NOT have the hidden bit set.
        if (!exponent.is_zero()) {
            mantissa |= HIDDEN_BIT; // Set the implicit leading '1'
        }
        
        return mantissa;
    }
    
    NT_ALWAYS_INLINE static constexpr float128_bits pack(bool sign, uint16_t exp_raw, uint64_t frac_hi48, uint64_t frac_lo64) noexcept {
        return float128_bits(b128::pack_ieee128(sign, exp_raw, frac_hi48, frac_lo64));
    }

    NT_ALWAYS_INLINE static constexpr float128_bits pack(bool sign, uint16_t exp_raw, b128 res) noexcept {
        uint64_t hi = 0;
        if (sign) hi |= (1ull << 63);
        hi |= (uint64_t(exp_raw & 0x7FFFu) << 48);
        hi |= (res.hi_bits & 0x0000FFFFFFFFFFFFull);
        return float128_bits(b128(hi, res.lo_bits));
    }

    // sets sign bit to 1
    NT_ALWAYS_INLINE constexpr float128_bits negative_sign_bit() const noexcept {
        return float128_bits(b128(bits.hi_bits | (uint64_t(1) << 63), bits.lo_bits));
    }

    // sets sign bit to 0
    NT_ALWAYS_INLINE constexpr float128_bits positive_sign_bit() const noexcept {
        return float128_bits(b128(bits.hi_bits & ~(uint64_t(1) << 63), bits.lo_bits));
    }

    // toggles sign bit on and off
    NT_ALWAYS_INLINE constexpr float128_bits change_sign_bit() const noexcept {
        return float128_bits(b128(bits.hi_bits ^ (uint64_t(1) << 63), bits.lo_bits));
    }

    NT_ALWAYS_INLINE constexpr bool is_zero() const noexcept {return positive_sign_bit().bits.is_zero(); }



    // changed it to int16_t for [Wconstant-conversion]
    NT_ALWAYS_INLINE constexpr bool is_inf() const noexcept {
        uint16_t exp = exponent();
        uint16_t c = 0x7FFF;
        if(exp != c) return false;
        // if it is zero, then it is infinity
        return (negative_sign_bit().bits & FRAC_MASK).is_zero();
    }

    // changed it to int16_t for [Wconstant-conversion]
    NT_ALWAYS_INLINE constexpr bool is_nan() const noexcept {
        uint16_t exp = exponent();
        uint16_t c = 0x7FFF;
        if(exp != c) return false;
        // if it is not zero then it is nan
        return !(negative_sign_bit().bits & FRAC_MASK).is_zero();
    }
};



}

inline std::ostream& operator<<(std::ostream& os, nt::float128_bits b) {
    nt::bitset<128, uint64_t> set;
    // very specific use case
    const_cast<uint64_t&>(set.raw()[0]) = b.bits.hi_bits;
    const_cast<uint64_t&>(set.raw()[1]) = b.bits.lo_bits;
    return os << set;
}


#endif
