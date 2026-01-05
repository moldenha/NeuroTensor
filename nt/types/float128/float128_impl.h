#ifndef NT_TYPES_128_BIT_TYPES_FLOAT128_IMPL_H__
#define NT_TYPES_128_BIT_TYPES_FLOAT128_IMPL_H__ 
// so that there are no float bits dependency errors
#define NT_KMATH_NO_INCLUDE_F128_

namespace nt{
class float128_t;
}

#include "../../utils/always_inline_macro.h"
#include "../../utils/type_traits.h"
#include "../../bit/bit_cast.h"
#include "../../bit/float_bits.h"
#include <cstddef>
#include <vector>
#include <exception>
#include <string> 
#include <type_traits>
#include <iostream>
#include <cmath>
#include <limits>

#include "b128.h"
#include "float128_bits.h"

namespace nt{


// op and op(single) -> NT_MAKE_THIS_MATH_OPERATOR__(+=, +)
#define NT_MAKE_THIS_MATH_OPERATOR__(op, ops)\
    NT_ALWAYS_INLINE constexpr float128_t& operator op (const float128_t& other){\
        *this = (*this ops other);\
        return *this;\
    }\
    \
    template<typename Float,\
        typename std::enable_if_t<std::is_floating_point_v<Float>, bool> = true>\
    NT_ALWAYS_INLINE constexpr float128_t& operator op (const Float& other){\
        *this = (*this op other);\
        return *this;\
    }\
    \
    template<typename Integer, \
        typename std::enable_if_t<std::is_integral_v<Integer>, bool> = true>\
    NT_ALWAYS_INLINE constexpr float128_t& operator op (const Integer& other){\
        *this = (*this op other);\
        return *this;\
    }\
        
        

class float128_t{
    float128_bits bits;
    static constexpr b128 b128_0 = b128(0);
    static constexpr b128 b128_1 = b128(1);
    static constexpr int MAX_EXP = 16383;
    
    // this is a way to make it non-constexpr (faster than constexpr route during runtime)
    inline static float128_bits nkfrom_double(double d) noexcept {
        if(std::isnan(d)){
            return float128_bits::make_nan();
        }else if (std::isinf(d)){
            return float128_bits::make_inf(d < 0);
        }
        uint64_t bits64 = ::nt::bit_cast<uint64_t>(d); 
        uint16_t sign = (bits64 >> 63) & 1;
        uint16_t exp  = (bits64 >> 52) & 0x7FF;
        uint64_t frac = bits64 & 0xFFFFFFFFFFFFFull;

        uint16_t new_exp = exp ? (exp - 1023 + 16383) : 0; 

        uint64_t frac_hi48 = (frac >> 4) & 0xFFFFFFFFFFFFull;
        uint64_t frac_lo64 = (frac << 60);

        return float128_bits(b128::pack_ieee128(sign, new_exp, frac_hi48, frac_lo64));
 
    }


    // inline constexpr float128_t ldexp(const float128_t& x, int n) noexcept {
    //     const auto bits = x.bits;
    //     if(bits.is_zero() || bits.is_nan() || bits.is_inf()) return x;

    //     uint16_t exp = bits.exponent();
    //     b128 mant = bits.mantissa_with_hidden_bit();
    //     bool sign = bits.sign();

    //     // If subnormal, normalize first
    //     if (exp == 0) {
    //         int lz = mant.clz();   // count leading zeros in 128-bit mantissa
    //         int shift = lz - (128 - 113);
    //         mant <<= shift;
    //         exp = 1;
    //         n -= shift;
    //     }

    //     // Apply exponent shift
    //     int32_t new_exp = int32_t(exp) + n;

    //     // Underflow into subnormal range?
    //     if (new_exp <= 0) {
    //         return float128_t(float128_bits::pack(
    //             sign,
    //             0,
    //             underflow_to_subnormal(mant, new_exp)
    //         ));
    //     }

    //     // Overflow → Inf
    //     if (new_exp >= EXP_INF_NAN) {
    //         return float128_t(float128_bits::pack(
    //             sign,
    //             EXP_INF_NAN,
    //             b128(0,0)
    //         ));
    //     }

    //     // Normal case: pack directly
    //     return float128_t(float128_bits::pack(
    //         sign,
    //         uint16_t(new_exp),
    //         mant
    //     ));
    // }

    // inline static constexpr float128_t internal_ldexp(const float128_t& x, int n) {
    //     if(x.bits.is_nan()) return float128_t{float128_bits::make_nan()};
    //     if(x.bits.is_zero()) return float128_t{float128_bits::make_zero()};
    //     if(x.bits.is_inf()) return x;
    //     float128_bits bits = x.bits;
        
    //     int32_t e = int32_t(b.exponent()) + n;
    //     if (e <= 0) return underflow_to_subnormal(...);
    //     if (e >= MAX_EXP) return float128_t{float128_bits::make_inf(x.bits.sign());}

    //     return float128_t{float128_bits::pack(b.sign(), uint16_t(e), b.mantissa())};
    // }


public:
    constexpr float128_t() : bits(b128_0) {};
    constexpr explicit float128_t(float128_bits b) : bits(b) {};
    float128_t(double val) : bits(float128_t::nkfrom_double(val)) {}
    float128_t(float val) : float128_t(double(val)) {}
    template<typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
    constexpr float128_t(T val) : float128_t(float128_t::from_integer(val)) {}
    
    constexpr float128_t& operator=(const float128_t&) = default;
    template<typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
    inline constexpr float128_t& operator=(const float128_t& other){return *this = float128_t(other);}
    template<typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
    inline float128_t& operator=(const float128_t& other){return *this = float128_t(other);}
    
    inline constexpr float128_bits get_bits() const noexcept { return bits; }
    
    
        
    
    // a constexpr way to convert from a float128_t to a double
    // This is not the default due to (math::kmath::frexp and math::kmath::ldexp being slower)
    NT_ALWAYS_INLINE static constexpr float128_t from_double(double d) noexcept {
        if(d != d){ // constexpr check nan values do not equal themselves
            return float128_t::make_nan();
        }else if (d == std::numeric_limits<double>::infinity()){
            return float128_t::make_inf();
        }else if (d == -std::numeric_limits<double>::infinity()){
            return float128_t::make_inf(true);
        }
        uint64_t bits64 = ::nt::float_bits<double>(d).get_tracker().lo_type(); 
        uint16_t sign = (bits64 >> 63) & 1;
        uint16_t exp  = (bits64 >> 52) & 0x7FF;
        uint64_t frac = bits64 & 0xFFFFFFFFFFFFFull;

        uint16_t new_exp = exp ? (exp - 1023 + 16383) : 0; 

        uint64_t frac_hi48 = (frac >> 4) & 0xFFFFFFFFFFFFull;
        uint64_t frac_lo64 = (frac << 60);

        return float128_t{ float128_bits(b128::pack_ieee128(sign, new_exp, frac_hi48, frac_lo64)) };
    }

    NT_ALWAYS_INLINE static constexpr float128_t from_double(float d) noexcept {
        return from_double(double(d));
    }

    inline static constexpr float128_t from_uint128(const b128 &x) noexcept {
        if(x.is_zero()){
            return float128_t::make_zero();
        }

        // Step 1: find MSB position
        int lz  = x.clz();
        int msb = 127 - lz;  // index of highest set bit, 0..127

        // Step 2: compute unbiased exponent
        int e = msb + 16383;

        // Step 3: convert to fraction
        b128 fraction;
        int shift = msb - 112;

        bool round_up = false;

        if (shift > 0) {
            // Need to right-shift to fit into 112 bits
            b128 shifted = x >> shift;
            b128 leftover = x & ((b128(1) << shift) - 1);

            // Rounding: round to nearest even
            b128 half = b128(1) << (shift - 1);
            if (leftover > half ||
                (leftover == half && (shifted & b128(1)) != b128(0)))
            {
                round_up = true;
            }

            fraction = shifted & ((b128(1) << 112) - 1);
        }
        else {
            // shift <= 0, left-shift into mantissa
            fraction = x << (-shift);
            fraction &= ((b128(1) << 112) - 1);
        }

        if (round_up) {
            fraction += b128(1);
            if (fraction >> 112) {
                // Mantissa overflow: renormalize
                fraction &= ((b128(1) << 112) - 1);
                e += 1;
            }
        }

        return float128_t(float128_bits::pack(false, uint16_t(e), fraction));
    }
    
    NT_ALWAYS_INLINE static constexpr float128_t from_integer(b128 i) noexcept {
        return from_uint128(i);
    }

    NT_ALWAYS_INLINE static constexpr float128_t from_integer(uint64_t i) noexcept {
        return from_uint128(b128(0, i));
    }

    NT_ALWAYS_INLINE static constexpr float128_t from_integer(int64_t i) noexcept {
        return i < 0 ? -from_integer(uint64_t(-i)) : from_integer(uint64_t(i));
    }

    NT_ALWAYS_INLINE static constexpr float128_t from_integer(uint32_t i) noexcept {
        return from_integer(uint64_t(i));
    }

    NT_ALWAYS_INLINE static constexpr float128_t from_integer(int32_t i) noexcept {
        return i < 0 ? -from_integer(uint64_t(-i)) : from_integer(uint64_t(i));
    }

    NT_ALWAYS_INLINE static constexpr float128_t from_integer(uint16_t i) noexcept {
        return from_integer(uint64_t(i));
    }

    NT_ALWAYS_INLINE static constexpr float128_t from_integer(int16_t i) noexcept {
        return i < 0 ? -from_integer(uint64_t(-i)) : from_integer(uint64_t(i));
    }

    NT_ALWAYS_INLINE static constexpr float128_t from_integer(uint8_t i) noexcept {
        return from_integer(uint64_t(i));
    }

    NT_ALWAYS_INLINE static constexpr float128_t from_integer(int8_t i) noexcept {
        return i < 0 ? -from_integer(uint64_t(-i)) : from_integer(uint64_t(i));
    }


    

    NT_ALWAYS_INLINE static constexpr float128_t make_zero(bool negative = false) noexcept {
        return float128_t(float128_bits::make_zero(negative));
    }
    
    NT_ALWAYS_INLINE static constexpr float128_t make_nan() noexcept {
        return float128_t(float128_bits::make_nan());
    }
    
    NT_ALWAYS_INLINE static constexpr float128_t make_snan() noexcept {
        return float128_t(float128_bits::make_snan());
    }

    NT_ALWAYS_INLINE static constexpr float128_t make_inf(bool negative = false) noexcept {
        return float128_t(float128_bits::make_inf(negative));
    }


    inline operator double() const noexcept {
        if (bits.is_zero()) return 0.0;
        if (bits.is_nan())  return std::numeric_limits<double>::quiet_NaN();
        if (bits.is_inf())  return bits.sign() ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();

        // Extract sign
        bool sign = (bits.bits.hi_bits >> 63) & 1;

        // Extract exponent (15-bit)
        int32_t exp128 = (bits.bits.hi_bits >> 48) & 0x7FFF;

        // Extract fraction (112-bit)
        uint64_t frac_hi = bits.bits.hi_bits & 0x0000FFFFFFFFFFFFull; // top 48 bits
        uint64_t frac_lo = bits.bits.lo_bits;                         // bottom 64 bits

        // Reconstruct fraction as double: 1.fraction
        double fraction = 1.0; // implicit leading 1

        // Scale fraction_hi and fraction_lo to [0, 1)
        fraction += double(frac_hi) / double(1ull << 48);
        // fraction += double(frac_lo) / double(1ull << 112); // note: loses precision but acceptable

        // Adjust exponent from binary128 bias (16383) to double bias (1023)
        int32_t exp_double = exp128 - 16383;

        // Compute final value
        double result = std::ldexp(fraction, exp_double);

        return sign ? -result : result;
    }

    inline operator float() const noexcept { return float(double(*this)); }

    inline constexpr operator int64_t() const noexcept {
        auto bits = this->get_bits();
        if(bits.is_nan() || bits.is_inf())
            return 0;

        bool sign = bits.sign();
        uint16_t exp = bits.exponent();

        if(exp == 0)
            return 0;

        int32_t E = int32_t(exp) - 16383;   // unbiased exponent
        if(E < 0)
            return 0;
        
        // overflow
        if(E > 127)
            return 0;

        // Get full significand
        b128 sig = bits.mantissa_with_hidden_bit();

        // Shift to integer
        b128 result = (E >= 112) ? (sig << (E-112)) : (sig >> (112 - E));
        
        return sign ? -int64_t(result.lo_bits) : int64_t(result.lo_bits);

    }
    
    inline constexpr float128_t operator+(const float128_t& b) const noexcept {
        // 1. Unpack
        bool sa = bits.sign();
        bool sb = b.bits.sign();
        uint16_t ea = bits.exponent();
        uint16_t eb = b.bits.exponent();
        b128 ma = bits.mantissa_with_hidden_bit();
        b128 mb = b.bits.mantissa_with_hidden_bit();

        // --- ADDED: Create room for GRS (Guard, Round, Sticky) bits ---
        // We shift both up by 3. Bit 0 becomes Sticky, 1 Round, 2 Guard.
        // The new "hidden bit" position is 112 + 3 = 115.
        ma <<= 3;
        mb <<= 3;

        // 2. Align exponents
        int16_t de = int16_t(ea) - int16_t(eb);
        if (de != 0) {
            if (de > 0) {
                // Check for Sticky bit before shifting
                // If any bits are about to be shifted out, set the Sticky bit (LSB)
                b128 lost_mask = (de >= 128) ? ~b128(0) : ((b128(1) << de) - 1);
                bool sticky = (mb & lost_mask) != 0;
                
                if (de >= 128) mb.zero();
                else mb >>= de;
                
                if (sticky) mb |= 1; // OR the sticky bit back into LSB
            } else {
                int shift = -de;
                b128 lost_mask = (shift >= 128) ? ~b128(0) : ((b128(1) << shift) - 1);
                bool sticky = (ma & lost_mask) != 0;

                if (shift >= 128) ma.zero();
                else ma >>= shift;

                if (sticky) ma |= 1; 
            }
        }

        uint16_t e_res = (ea > eb) ? ea : eb; 

        // 3. Add or subtract mantissas (Now including GRS bits)
        b128 m_res;
        bool sign_res = false;
        
        if (sa == sb) {
            m_res = ma + mb;
            sign_res = sa;
            
            // Check for mantissa overflow (Bit 116 is set)
            // Original hidden bit was 112. Shifted +3 = 115. Carry = 116.
            if (m_res > (b128(1) << 115 | (b128(1) << 115) - 1)) { 
                 // Logic: if m_res >= 2^116. 
                 // Faster check: if (m_res >> 116) != 0
                 
                 bool sticky = (m_res & 1) != 0; // Capture sticky from LSB
                 m_res >>= 1;
                 m_res |= sticky; // Maintain sticky after shift
                 e_res += 1;
            }
        } else {
            if (ma >= mb) {
                m_res = ma - mb;
                sign_res = sa;
            } else {
                m_res = mb - ma;
                sign_res = sb;
            }

            // Normalize subtraction result
            if (m_res.is_zero()) {
                 // Exact zero result
                 return float128_t(float128_bits::pack(false, 0, b128(0)));
            }

            // Find position of first set bit
            // We expect the hidden bit at 115.
            // If clz() counts leading zeros on 128-bit type:
            uint8_t leading = m_res.clz(); // e.g., if bit 115 is set, clz is 12 (128-116)
            int target_lead = 128 - 1 - 115; // 12
            
            int shift_needed = leading - target_lead;
            
            if (shift_needed > 0) {
                m_res <<= shift_needed;
                e_res -= shift_needed;
            } else if (shift_needed < 0) {
                // This handles the rare case where we might need to shift right? 
                // Usually subtraction only shrinks, but being safe:
                // (Only happens if alignment logic was flawed, but standard logic is valid)
            }
        }

        // --- ADDED: Rounding Step (Round to Nearest, Ties to Even) ---
        // Current format: [ Integer Part | . | frac ... | G | R | S ]
        // The G, R, S are the bottom 3 bits (mask 0x7).
        
        uint8_t grs = (uint8_t)(m_res & 7); // Extract last 3 bits
        bool guard = (grs & 4) != 0;
        bool round_bit = (grs & 2) != 0;
        bool sticky = (grs & 1) != 0;
        bool lsb = (m_res & 8) != 0; // The least significant bit of the ACTUAL mantissa

        // Remove GRS bits now, positioning for check
        // We do not shift yet, because we might need to add 1 to the bit at pos 3.
        
        bool round_up = false;
        if (guard) {
            if (round_bit || sticky) round_up = true; // > 0.5
            else if (lsb) round_up = true;            // == 0.5, tie to even (odd -> up)
        }

        if (round_up) {
            m_res += 8; // Add 1 to the position 3 (where the LSB lives)
            
            // Adding 1 might cause an overflow (e.g. 1.111 + 1 = 10.000)
            if (m_res >= (b128(1) << 116)) {
                m_res >>= 1;
                e_res += 1;
            }
        }

        // Final shift to remove GRS bits
        m_res >>= 3; 

        // 4. Pack
        // Mask out the hidden bit (bit 112) for storage
        // FRAC_MASK is (1<<112) - 1
        return float128_t(float128_bits::pack(sign_res, e_res, m_res & float128_bits::FRAC_MASK));
    }
    
    // just changes the sign and adds
    NT_ALWAYS_INLINE constexpr float128_t operator-(const float128_t& b) const noexcept {
        return *this + (float128_t(b.bits.change_sign_bit()));
    }
    
    NT_ALWAYS_INLINE constexpr float128_t operator-() const noexcept {
        return float128_t(bits.change_sign_bit());
    }

    NT_ALWAYS_INLINE constexpr float128_t abs() const noexcept {
        return float128_t(bits.positive_sign_bit());
    }

    inline constexpr float128_t operator*(const float128_t& b) const noexcept {
        if(b.bits.is_nan() || this->bits.is_nan()){
            return float128_t::make_nan();
        }else if(b.bits.is_inf() || this->bits.is_inf()){
            return float128_t::make_inf(bits.sign() ^ b.bits.sign());
        }else if(b.bits.is_zero() || this->bits.is_zero()){
            return float128_t::make_zero();
        }
        // 1. Unpack
        bool sa = bits.sign();
        bool sb = b.bits.sign();
        uint16_t ea = bits.exponent();
        uint16_t eb = b.bits.exponent();
        b128 ma = bits.mantissa_with_hidden_bit(); // 113-bit mantissa
        b128 mb = b.bits.mantissa_with_hidden_bit();


        // 2. Compute sign
        bool sign_res = sa ^ sb;
        
        // 3. add exponents = (add biased exponents, then subtract bias (in this case 16383 for float128)
        int32_t e_res = int32_t(ea) + int32_t(eb) - 16383;
        
        // 4. Multiply the mantissa bits
        //  This is done so that there is no potential overflow
        //  This allows rounding and normalization to be done properly
        std::pair<b128, b128> bmult = b128::multiply_128x128(ma, mb);
        // bool overflow = (bmult.first.hi_bits & (b128(1) << 97)) != 0;
        bool overflow = bmult.first.bit(225 - 128);
        int shift = overflow ? 113 : 112;
        if (overflow){
            e_res += 1;
            // bmult = b128::shift_256_right(bmult.first, bmult.second, 1);
        }
        
        // this is now a 116 bit window containing information for rounding
        b128 mres_work = b128::shift_256_right(bmult.first, bmult.second, shift - 3).second;
        
        // mres_work is the 116-bit window (M...T|G|R|S)

        // GRS bits are the three lowest bits (indices 2, 1, 0)
        constexpr uint8_t GRS_MASK = 0b111; // 7

        // T-bit is the Least Significant Bit of the final 112 fractional bits, 
        // which sits at index 3 in mres_work.
        constexpr b128 T_BIT_MASK = b128(1) << 3; // 8

        uint8_t grs_bits = (uint8_t)((mres_work & GRS_MASK).lo_bits);
        bool G = (grs_bits & 0b100); // Guard bit (index 2)
        bool R = (grs_bits & 0b010); // Round bit (index 1)
        bool S = (grs_bits & 0b001); // Sticky bit (index 0)
        bool T = (mres_work & T_BIT_MASK) != 0; // T-bit (index 3)
        
        bool round_up = false;

        // Case 1: Strictly Greater Than Halfway (G=1 and R or S is 1)
        if (G && (R || S)) {
            round_up = true;
        } 
        // Case 2: Exactly Halfway (G=1, R=0, S=0)
        else if (G && !R && !S) {
            // Check T-bit (LSB of the mantissa) for "Ties to Even"
            if (T) {
                round_up = true; // Round up to make the LSB (T-bit) zero (even)
            }
        }

        if (round_up) {
            // Add 8 (b128(1) << 3) to round up the T-bit
            mres_work += T_BIT_MASK;

            // Check for overflow due to rounding (e.g., 1.11...11 + 1 -> 10.00...00)
            // The hidden bit is at index 115. Overflow means index 116 is now set.
            constexpr b128 CARRY_MASK = b128(1) << 116; 
            
            if (mres_work & CARRY_MASK) {
                // Mantissa overflowed: shift right by 1 and adjust exponent
                mres_work >>= 1;
                e_res += 1;
            }
        }

        // 4. Final Truncation and Packing

        // Remove the GRS bits (and the extra T-bit alignment)
        b128 mres = mres_work >> 3;

        return float128_t(float128_bits::pack(sign_res, uint16_t(e_res), mres));
    }

    inline constexpr float128_t operator/(const float128_t& b) const noexcept {
        if(b.bits.is_nan() || this->bits.is_nan() || b.bits.is_zero()){
            return float128_t::make_nan();
        }else if(this->bits.is_zero()){
            return float128_t::make_zero();
        }else if (b.bits.is_inf()){
            return float128_t::make_nan();
        }else if(this->bits.is_inf()){
            return float128_t::make_inf(b.bits.sign() ^ bits.sign()); 
        }

        // 1. Convert exponents to decimal numbers
        // 2. Prepend implicit 1 to mantissas
        // 3. Determine sign
        // 4. Subtract exponents and add bias
        // 5. Divide mantissas
        // 6. Normalization
        // 7. Rounding
        // 8. Convert exponent to binary number
        // 9. Assemble floating point number
        

        // 1. Convert exponents to decimal numbers
        
        // the exponent function returns the exponent bits
        int32_t ea = int32_t(bits.exponent());
        int32_t eb = int32_t(b.bits.exponent());
        // 2. Prepend implicit 1 to mantissas
        // Get 113-bit mantissas (1.f)
        //  - 112 mantissa bits + the hidden bit
        b128 ma = bits.mantissa_with_hidden_bit();
        b128 mb = b.bits.mantissa_with_hidden_bit();
        

        // 3. Determine sign
        bool sa = bits.sign();
        bool sb = b.bits.sign();
        bool sign_res = sa ^ sb;

        // 4. Subtract exponents and add bias
        int32_t e_res = ea - eb + 16383;

        // 5. Divide mantissas        
        
        //  This is done so that there is no potential overflow
        //  This allows rounding and normalization to be done properly
        // The mantissa bits will never be 0 because of the inf check 
        while (ma < mb) {
            ma <<= 1;
            e_res -= 1;
        }

        // Assume now ma >= mb
        b128 R = ma;
        b128 Q(0);
        // Subtract out the integer part (should be 1 since ma>=mb after normalization)
        R -= mb; 
        Q |= 1;  // hidden 1 at MSB
        // Generate 112+3 additional bits (112 fraction + 3 guard/round/sticky)
        for(int i = 0; i < 112+3; i++) {
            R <<= 1;
            Q <<= 1;
            if (R >= mb) {
                R -= mb;
                Q |= 1;
            }
        }

        if (Q & (b128(1) << 116)) {  
            Q >>= 1; 
            e_res += 1;
        }

        b128 mres_work = Q;
        bool G = (mres_work & b128(0b100)) != 0;
        bool Rb = (mres_work & b128(0b010)) != 0;
        bool S = (mres_work & b128(0b001)) != 0;
        bool T = (mres_work & b128(1<<3)) != 0;
        if ((G && (Rb || S)) || (G && !Rb && !S && T)) {
            mres_work += b128(1<<3);
            // handle carry-out:
            if (mres_work & (b128(1) << 116)) {
                // Mantissa overflowed (1.xxx + 1 -> 10.0xxx)
                mres_work >>= 1;
                e_res += 1;
            }
        }

        b128 fractional_bits = (mres_work >> 3) & ((b128(1)<<112)-1);
        return float128_t(float128_bits::pack(sign_res, uint16_t(e_res), fractional_bits));
    }

    // purposely using std::is_integral_v and std::is_floating_point_v
    //  for types that would be used by something like f + 2.13 for example
    template<typename Float,
        typename std::enable_if_t<std::is_floating_point_v<Float>, bool> = true>
    NT_ALWAYS_INLINE constexpr float128_t operator+(const Float& other) noexcept { return *this + float128_t::from_double(other); }
    template<typename Float,
        typename std::enable_if_t<std::is_floating_point_v<Float>, bool> = true>
    NT_ALWAYS_INLINE constexpr float128_t operator*(const Float& other) noexcept { return *this * float128_t::from_double(other); }
    template<typename Float,
        typename std::enable_if_t<std::is_floating_point_v<Float>, bool> = true>
    NT_ALWAYS_INLINE constexpr float128_t operator-(const Float& other) noexcept { return *this - float128_t::from_double(other); }
    template<typename Float,
        typename std::enable_if_t<std::is_floating_point_v<Float>, bool> = true>
    NT_ALWAYS_INLINE constexpr float128_t operator/(const Float& other) noexcept { return *this / float128_t::from_double(other); }
    
    template <typename Int, 
          typename = std::enable_if_t<std::is_integral_v<Int>>>
    NT_ALWAYS_INLINE constexpr float128_t operator+(Int a) const noexcept {
        return *this + float128_t::from_integer(a);
    }
    
    template <typename Int, 
          typename = std::enable_if_t<std::is_integral_v<Int>>>
    NT_ALWAYS_INLINE constexpr float128_t operator-(Int a) const noexcept {
        return *this - float128_t::from_integer(a);
    }

    template <typename Int, 
          typename = std::enable_if_t<std::is_integral_v<Int>>>
    NT_ALWAYS_INLINE constexpr float128_t operator*(Int a) const noexcept {
        return *this * float128_t::from_integer(a);
    }

    template <typename Int, 
          typename = std::enable_if_t<std::is_integral_v<Int>>>
    NT_ALWAYS_INLINE constexpr float128_t operator/(Int a) const noexcept {
        return *this / float128_t::from_integer(a);
    }

    NT_MAKE_THIS_MATH_OPERATOR__(+=, +);
    NT_MAKE_THIS_MATH_OPERATOR__(*=, *);
    NT_MAKE_THIS_MATH_OPERATOR__(-=, -);
    NT_MAKE_THIS_MATH_OPERATOR__(/=, /);

    // PREFIX ++
    NT_ALWAYS_INLINE constexpr float128_t& operator++() noexcept {
        *this = *this + 1;
        return *this;
    }

    // POSTFIX ++
    NT_ALWAYS_INLINE constexpr float128_t operator++(int) noexcept {
        float128_t tmp = *this;
        ++(*this);
        return tmp;
    }

    // PREFIX --
    NT_ALWAYS_INLINE constexpr float128_t& operator--() noexcept {
        *this = *this - 1;
        return *this;
    }

    // POSTFIX --
    NT_ALWAYS_INLINE constexpr float128_t operator--(int) noexcept {
        float128_t tmp = *this;
        --(*this);
        return tmp;
    }


    // NT_MAKE_INTEGER_FLOATING128_OP_(b128);
    // NT_MAKE_INTEGER_FLOATING128_OP_(uint64_t);
    // NT_MAKE_INTEGER_FLOATING128_OP_(int64_t);
    // NT_MAKE_INTEGER_FLOATING128_OP_(uint32_t);
    // NT_MAKE_INTEGER_FLOATING128_OP_(int32_t);
    // NT_MAKE_INTEGER_FLOATING128_OP_(uint16_t);
    // NT_MAKE_INTEGER_FLOATING128_OP_(int16_t);
    // NT_MAKE_INTEGER_FLOATING128_OP_(uint8_t);
    // NT_MAKE_INTEGER_FLOATING128_OP_(int8_t);
    
    NT_ALWAYS_INLINE constexpr bool operator==(const float128_t& other) const noexcept {
        if(bits.is_nan() || other.bits.is_nan()) return false;
        if(bits.is_zero() && other.bits.is_zero()) return true;
        return bits.bits == other.bits.bits;
    }
    
    NT_ALWAYS_INLINE constexpr bool operator!=(const float128_t& other) const noexcept {
        if(bits.is_nan() || other.bits.is_nan()) return true;
        if(bits.is_zero() && other.bits.is_zero()) return false;
        return bits.bits != other.bits.bits;
    }
    
    inline constexpr bool operator>(const float128_t& b) const noexcept {
        // 0. NaN handling
        if (bits.is_nan() || b.bits.is_nan())
            return false;
        // 1. Zero handling (+0 == -0)
        if (bits.is_zero() && b.bits.is_zero())
            return false;
        bool sa = bits.sign();
        bool sb = b.bits.sign();
        // 3. Different signs
        if (sa != sb)
            return sb; // positive > negative
        // 4. Same sign → compare magnitude
        int32_t ea = int32_t(bits.exponent());
        int32_t eb = int32_t(b.bits.exponent());

        b128 ma = bits.mantissa_with_hidden_bit();
        b128 mb = b.bits.mantissa_with_hidden_bit();

        // 5. Compare exponent first
        if (ea != eb) {
            if (!sa)
                return ea > eb; // both positive
            else
                return ea < eb; // both negative (reversed)
        }

        // 6. Exponents equal → compare mantissa
        if (ma != mb) {
            if (!sa)
                return ma > mb;
            else
                return ma < mb;
        }

        // 7. Exactly equal
        return false;
    }

    inline constexpr bool operator<(const float128_t& b) const noexcept {
        // 0. NaN handling
        if (bits.is_nan() || b.bits.is_nan())
            return false;
        // 1. Zero handling (+0 == -0)
        if (bits.is_zero() && b.bits.is_zero())
            return false;
        bool sa = bits.sign();
        bool sb = b.bits.sign();
        // 3. Different signs
        if (sa != sb)
            return sa; // negative < positive
        // 4. Same sign → compare magnitude
        int32_t ea = int32_t(bits.exponent());
        int32_t eb = int32_t(b.bits.exponent());

        b128 ma = bits.mantissa_with_hidden_bit();
        b128 mb = b.bits.mantissa_with_hidden_bit();

        // 5. Compare exponent first
        if (ea != eb) {
            if (!sa)
                return ea < eb; // both positive
            else
                return ea > eb; // both negative (reversed)
        }

        // 6. Exponents equal → compare mantissa
        if (ma != mb) {
            if (!sa)
                return ma < mb;
            else
                return ma > mb;
        }

        // 7. Exactly equal
        return false;
    }

    inline constexpr bool operator>=(const float128_t& b) const noexcept {
        // 0. NaN handling
        if (bits.is_nan() || b.bits.is_nan())
            return false;
        // 1. Zero handling (+0 == -0)
        if (bits.is_zero() && b.bits.is_zero())
            return false;
        bool sa = bits.sign();
        bool sb = b.bits.sign();
        // 3. Different signs
        if (sa != sb)
            return sb; // positive > negative
        // 4. Same sign → compare magnitude
        int32_t ea = int32_t(bits.exponent());
        int32_t eb = int32_t(b.bits.exponent());

        b128 ma = bits.mantissa_with_hidden_bit();
        b128 mb = b.bits.mantissa_with_hidden_bit();

        // 5. Compare exponent first
        if (ea != eb) {
            if (!sa)
                return ea > eb; // both positive
            else
                return ea < eb; // both negative (reversed)
        }

        // 6. Exponents equal → compare mantissa
        if (ma != mb) {
            if (!sa)
                return ma > mb;
            else
                return ma < mb;
        }

        // 7. Exactly equal
        return true;
    }

    inline constexpr bool operator<=(const float128_t& b) const noexcept {
        // 0. NaN handling
        if (bits.is_nan() || b.bits.is_nan())
            return false;
        // 1. Zero handling (+0 == -0)
        if (bits.is_zero() && b.bits.is_zero())
            return false;
        bool sa = bits.sign();
        bool sb = b.bits.sign();
        // 3. Different signs
        if (sa != sb)
            return sa; // negative < positive
        // 4. Same sign → compare magnitude
        int32_t ea = int32_t(bits.exponent());
        int32_t eb = int32_t(b.bits.exponent());

        b128 ma = bits.mantissa_with_hidden_bit();
        b128 mb = b.bits.mantissa_with_hidden_bit();

        // 5. Compare exponent first
        if (ea != eb) {
            if (!sa)
                return ea < eb; // both positive
            else
                return ea > eb; // both negative (reversed)
        }

        // 6. Exponents equal → compare mantissa
        if (ma != mb) {
            if (!sa)
                return ma < mb;
            else
                return ma > mb;
        }

        // 7. Exactly equal
        return true;
    }

    template<typename Float,
        typename std::enable_if_t<std::is_floating_point_v<Float>, bool> = true>
    NT_ALWAYS_INLINE constexpr bool operator==(const Float& other) noexcept { return *this == float128_t::from_double(other); }
    template<typename Float,
        typename std::enable_if_t<std::is_floating_point_v<Float>, bool> = true>
    NT_ALWAYS_INLINE constexpr bool operator!=(const Float& other) noexcept { return *this != float128_t::from_double(other); }
    template<typename Float,
        typename std::enable_if_t<std::is_floating_point_v<Float>, bool> = true>
    NT_ALWAYS_INLINE constexpr bool operator<(const Float& other) noexcept { return *this < float128_t::from_double(other); }
    template<typename Float,
        typename std::enable_if_t<std::is_floating_point_v<Float>, bool> = true>
    NT_ALWAYS_INLINE constexpr bool operator>(const Float& other) noexcept { return *this > float128_t::from_double(other); }
    template<typename Float,
        typename std::enable_if_t<std::is_floating_point_v<Float>, bool> = true>
    NT_ALWAYS_INLINE constexpr bool operator<=(const Float& other) noexcept { return *this <= float128_t::from_double(other); }
    template<typename Float,
        typename std::enable_if_t<std::is_floating_point_v<Float>, bool> = true>
    NT_ALWAYS_INLINE constexpr bool operator>=(const Float& other) noexcept { return *this >= float128_t::from_double(other); }
    
    template <typename Int, 
          typename = std::enable_if_t<std::is_integral_v<Int>>>
    NT_ALWAYS_INLINE constexpr bool operator==(Int a) const noexcept {
        return *this == float128_t::from_integer(a);
    }
    
    template <typename Int, 
          typename = std::enable_if_t<std::is_integral_v<Int>>>
    NT_ALWAYS_INLINE constexpr bool operator!=(Int a) const noexcept {
        return *this != float128_t::from_integer(a);
    }

    template <typename Int, 
          typename = std::enable_if_t<std::is_integral_v<Int>>>
    NT_ALWAYS_INLINE constexpr bool operator<(Int a) const noexcept {
        return *this < float128_t::from_integer(a);
    }
    
    template <typename Int, 
          typename = std::enable_if_t<std::is_integral_v<Int>>>
    NT_ALWAYS_INLINE constexpr bool operator>(Int a) const noexcept {
        return *this > float128_t::from_integer(a);
    }

    template <typename Int, 
          typename = std::enable_if_t<std::is_integral_v<Int>>>
    NT_ALWAYS_INLINE constexpr bool operator<=(Int a) const noexcept {
        return *this <= float128_t::from_integer(a);
    }

    template <typename Int, 
          typename = std::enable_if_t<std::is_integral_v<Int>>>
    NT_ALWAYS_INLINE constexpr bool operator>=(Int a) const noexcept {
        return *this >= float128_t::from_integer(a);
    }

};


#undef NT_MAKE_THIS_MATH_OPERATOR__ 






#define NT_MAKE_F128_OPERATOR_OTHER_(out_type, op)\
template<typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>\
NT_ALWAYS_INLINE constexpr out_type operator op (const T& a, const ::nt::float128_t& b) noexcept {\
    return ::nt::float128_t(a) op b;\
}\
\
template<typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>\
NT_ALWAYS_INLINE out_type operator op (const T& a, const ::nt::float128_t& b) noexcept {\
    return ::nt::float128_t(a) op b;\
}\

// math operators

NT_MAKE_F128_OPERATOR_OTHER_(::nt::float128_t, /);
NT_MAKE_F128_OPERATOR_OTHER_(::nt::float128_t, *);
NT_MAKE_F128_OPERATOR_OTHER_(::nt::float128_t, -);
NT_MAKE_F128_OPERATOR_OTHER_(::nt::float128_t, +);

// compare operators

NT_MAKE_F128_OPERATOR_OTHER_(bool, ==);
NT_MAKE_F128_OPERATOR_OTHER_(bool, !=);
NT_MAKE_F128_OPERATOR_OTHER_(bool, <);
NT_MAKE_F128_OPERATOR_OTHER_(bool, >);
NT_MAKE_F128_OPERATOR_OTHER_(bool, <=);
NT_MAKE_F128_OPERATOR_OTHER_(bool, >=);

#undef NT_MAKE_F128_OPERATOR_OTHER_


}

// so that kmath can include float128_t
#undef NT_KMATH_NO_INCLUDE_F128_
#endif
