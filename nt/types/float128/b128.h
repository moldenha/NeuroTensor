#ifndef NT_TYPES_128_BIT_TYPES_B128_H__
#define NT_TYPES_128_BIT_TYPES_B128_H__ 
// This is a struct which acts in many ways like a uint128
// Some of the calculations are similar to what is seen in https://github.com/calccrypto/uint128_t
// The main difference is that b128 is guaranteed to be constexpr - usefull for various reasons
// This struct may replace the use of the current uint128 to something native (as in this below)

#include "../../utils/always_inline_macro.h"
#include <cstddef>
#include <vector>
#include <exception>
#include <iostream>
#include "eddian.h"


namespace nt{

#define NT_B128_SHIFT_OPERATORS_(type)\
 NT_ALWAYS_INLINE constexpr b128 operator>>(const type& s) const noexcept {return *this >> (int64_t)s;}\
 NT_ALWAYS_INLINE constexpr b128 operator<<(const type& s) const noexcept {return *this << (int64_t)s;}\
 NT_ALWAYS_INLINE constexpr b128& operator>>=(const type& s) noexcept {return *this >>= (int64_t)s;}\
 NT_ALWAYS_INLINE constexpr b128& operator<<=(const type& s) noexcept {return *this <<= (int64_t)s;}\

#define NT_BIT_OPERATORS_UNSIGNED_(type)\
    NT_ALWAYS_INLINE constexpr b128 operator&(const type& s) const noexcept {return *this & (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128& operator&=(const type& s) noexcept {return *this &= (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128 operator^(const type& s) const noexcept {return *this ^ (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128& operator^=(const type& s) noexcept {return *this ^= (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128 operator|(const type& s) const noexcept {return *this | (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128& operator|=(const type& s) noexcept {return *this |= (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128 operator-(const type& s) const noexcept {return *this - (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128& operator-=(const type& s) noexcept {return *this -= (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128 operator+(const type& s) const noexcept {return *this + (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128& operator+=(const type& s) noexcept {return *this += (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128 operator*(const type& s) const noexcept {return *this * (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128& operator*=(const type& s) noexcept {return *this *= (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128 operator/(const type& s) const noexcept {return *this / (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128& operator/=(const type& s) noexcept {return *this /= (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128 operator%(const type& s) const noexcept {return *this % (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128& operator%=(const type& s) noexcept {return *this %= (uint64_t)(s);}\

#define NT_BIT_OPERATORS_SIGNED_(type)\
    NT_ALWAYS_INLINE constexpr b128 operator&(const type& s) const noexcept {return *this & (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128& operator&=(const type& s) noexcept {return *this &= (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128 operator^(const type& s) const noexcept {return *this ^ (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128& operator^=(const type& s) noexcept {return *this ^= (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128 operator|(const type& s) const noexcept {return *this | (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128& operator|=(const type& s) noexcept {return *this |= (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128 operator-(const type& s) const noexcept {return *this - (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128 operator-=(const type& s) noexcept {return *this -= (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128 operator+(const type& s) const noexcept {return *this + (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128 operator+=(const type& s) noexcept {return *this += (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128 operator*(const type& s) const noexcept {return *this * (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128 operator*=(const type& s) noexcept {return *this *= (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128 operator/(const type& s) const noexcept {return *this / (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128 operator/=(const type& s) noexcept {return *this /= (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128 operator%(const type& s) const noexcept {return *this % (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b128 operator%=(const type& s) noexcept {return *this %= (int64_t)(s);}\


struct b128{
#ifdef NT_128BITS_BIG_ENDIAN__
        uint64_t hi_bits, lo_bits;
#endif
#ifdef NT_128BITS_LITTLE_ENDIAN__
        uint64_t lo_bits, hi_bits;
#endif
    constexpr b128() noexcept : hi_bits(0), lo_bits(0) {}
    constexpr b128(uint64_t h, uint64_t l) noexcept : hi_bits(h), lo_bits(l) {}
    constexpr explicit b128(uint64_t l) noexcept : hi_bits(0), lo_bits(l) {}
    constexpr b128(const b128&) noexcept = default;
    constexpr b128(b128&&) noexcept = default;
    NT_ALWAYS_INLINE constexpr b128& operator=(const b128&) = default;
    NT_ALWAYS_INLINE constexpr b128& operator=(b128&&) = default;


    // == helpers ==
    NT_ALWAYS_INLINE constexpr bool is_zero() const noexcept { return hi_bits == 0 && lo_bits == 0; }
    NT_ALWAYS_INLINE constexpr b128& zero() noexcept { hi_bits = 0; lo_bits = 0; return *this; }

    // compare
    NT_ALWAYS_INLINE constexpr int cmp(const b128& o) const noexcept {
        if (hi_bits < o.hi_bits) return -1;
        if (hi_bits > o.hi_bits) return  1;
        if (lo_bits < o.lo_bits) return -1;
        if (lo_bits > o.lo_bits) return  1;
        return 0;
    }

    NT_ALWAYS_INLINE constexpr bool operator==(const b128& o) const noexcept { return hi_bits == o.hi_bits && lo_bits == o.lo_bits; }
    NT_ALWAYS_INLINE constexpr bool operator!=(const b128& o) const noexcept { return !(*this == o); }
    NT_ALWAYS_INLINE constexpr bool operator<(const b128& o) const noexcept { return cmp(o) < 0; }
    NT_ALWAYS_INLINE constexpr bool operator<=(const b128& o) const noexcept { return cmp(o) <= 0; }
    NT_ALWAYS_INLINE constexpr bool operator>(const b128& o) const noexcept { return cmp(o) > 0; }
    NT_ALWAYS_INLINE constexpr bool operator>=(const b128& o) const noexcept { return cmp(o) >= 0; }

    NT_ALWAYS_INLINE constexpr int bit_at(int i) const noexcept {
        return (i < 64) ? int((lo_bits >> i) & 1ull) : int((hi_bits >> (i - 64)) & 1ull);
    }


    // add (b128 + b128)
    NT_ALWAYS_INLINE constexpr b128 operator+(const b128& o) const noexcept {
        uint64_t new_lo = lo_bits + o.lo_bits;
        uint64_t carry = (new_lo < lo_bits) ? 1ull : 0ull; // overflow detection
        uint64_t new_hi = hi_bits + o.hi_bits + carry;
        return b128(new_hi, new_lo);
    }

    NT_ALWAYS_INLINE constexpr b128& operator+=(const b128& o) noexcept {
        uint64_t new_lo = lo_bits + o.lo_bits;
        uint64_t carry = (new_lo < lo_bits) ? 1ull : 0ull;
        lo_bits = new_lo;
        hi_bits += o.hi_bits + carry;
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b128 operator+(const uint64_t& o) const noexcept {
        uint64_t new_lo = lo_bits + o;
        uint64_t carry = (new_lo < lo_bits) ? 1ull : 0ull; // overflow detection
        uint64_t new_hi = hi_bits + carry;
        return b128(new_hi, new_lo);
    }

    NT_ALWAYS_INLINE constexpr b128& operator+=(const uint64_t& o) noexcept {
        uint64_t new_lo = lo_bits + o;
        uint64_t carry = (new_lo < lo_bits) ? 1ull : 0ull;
        lo_bits = new_lo;
        hi_bits += carry;
        return *this;
    }



    NT_ALWAYS_INLINE constexpr b128& operator++() noexcept {
        constexpr b128 b128_1(1);
        return *this += b128_1;
    }

    NT_ALWAYS_INLINE constexpr b128 operator++(int){
        b128 temp(*this);
        ++*this;
        return temp;
    }

    // subtract (b128 - b128)
    NT_ALWAYS_INLINE constexpr b128 operator-(const b128& o) const noexcept {
        uint64_t new_lo = lo_bits - o.lo_bits;
        uint64_t borrow = (lo_bits < o.lo_bits) ? 1ull : 0ull;
        uint64_t new_hi = hi_bits - o.hi_bits - borrow;
        return b128(new_hi, new_lo);
    }
    
    NT_ALWAYS_INLINE constexpr b128& operator-=(const b128& o) noexcept {
        uint64_t new_lo = lo_bits - o.lo_bits;
        uint64_t borrow = (lo_bits < o.lo_bits) ? 1ull : 0ull;
        uint64_t new_hi = hi_bits - o.hi_bits - borrow;
        hi_bits = new_hi;
        lo_bits = new_lo;
        return *this;
    }
    
    NT_ALWAYS_INLINE constexpr b128 operator-(uint64_t b) const noexcept {
        uint64_t new_lo = lo_bits - b;
        uint64_t borrow = (lo_bits < b) ? 1 : 0;
        return b128(hi_bits - borrow, new_lo);
    }

    NT_ALWAYS_INLINE constexpr b128& operator-=(uint64_t b) {
        uint64_t new_lo = lo_bits - b;
        uint64_t borrow = (lo_bits < b) ? 1 : 0;
        hi_bits -= borrow;
        lo_bits = new_lo;
        return *this;
    }
    
    NT_ALWAYS_INLINE constexpr b128 operator-(int64_t b) const noexcept { return (b < 0) ? (*this + (uint64_t(-b))) : (*this - (uint64_t)b);}
    NT_ALWAYS_INLINE constexpr b128& operator-=(int64_t b) noexcept { return (b < 0) ? (*this += (uint64_t(-b))) : (*this -= (uint64_t)b);}
    NT_ALWAYS_INLINE constexpr b128 operator+(const int64_t& b) const noexcept {return (b < 0) ? (*this - uint64_t(-b)) : (*this + uint64_t(b));}
    NT_ALWAYS_INLINE constexpr b128& operator+=(const int64_t& b) noexcept {return (b < 0) ? (*this -= uint64_t(-b)) : (*this += uint64_t(b));}

    NT_ALWAYS_INLINE constexpr b128 operator<<(const int64_t& s) const noexcept {
        if (s == 0) return *this;
        if (s >= 128) return b128(0, 0);

        if (s >= 64) {
            // lo_bits moves entirely into hi_bits
            return b128(lo_bits << (s - 64), 0);
        }

        // s in [0..63]
        uint64_t new_hi = (hi_bits << s) | (lo_bits >> (64 - s));
        uint64_t new_lo = lo_bits << s;
        return b128(new_hi, new_lo);
    }

    NT_ALWAYS_INLINE constexpr b128 operator>>(const int64_t& s) const noexcept {
        if (s == 0) return *this;
        if (s >= 128) return b128(0, 0);

        if (s >= 64) {
            // hi_bits moves entirely into lo_bits
            return b128(0, hi_bits >> (s - 64));
        }

        // s in [0..63]
        uint64_t new_hi = hi_bits >> s;
        uint64_t new_lo = (lo_bits >> s) | (hi_bits << (64 - s));
        return b128(new_hi, new_lo);
    }

    NT_ALWAYS_INLINE constexpr b128& operator<<=(const int64_t& s) noexcept {
        if (s == 0) return *this;
        if (s >= 128){
            hi_bits = 0;
            lo_bits = 0;
            return *this;
        }
        if (s >= 64) {
            // lo_bits moves entirely into hi_bits
            hi_bits = (lo_bits << (s-64));
            lo_bits = 0;
            return *this;
        }

        // s in [0..63]
        hi_bits = (hi_bits << s) | (lo_bits >> (64 - s));
        lo_bits <<= s;
        return *this; 
    }

    NT_ALWAYS_INLINE constexpr b128& operator>>=(const int64_t& s) noexcept {
        if (s == 0) return *this;
        if (s >= 128){
            hi_bits = 0;
            lo_bits = 0;
            return *this;
        }
        if (s >= 64) {
            // hi_bits moves entirely into lo_bits
            lo_bits = hi_bits >> (s-64);
            hi_bits = 0;
            return *this;
        }
        else{
            // s in [0..63]
            lo_bits = (lo_bits >> s) | (hi_bits << (64-s));
            hi_bits >>= s;
        }
        return *this;
    }
    
    NT_B128_SHIFT_OPERATORS_(uint8_t)
    NT_B128_SHIFT_OPERATORS_(int8_t)
    NT_B128_SHIFT_OPERATORS_(uint16_t)
    NT_B128_SHIFT_OPERATORS_(int16_t)
    NT_B128_SHIFT_OPERATORS_(uint32_t)
    NT_B128_SHIFT_OPERATORS_(int32_t)
    NT_B128_SHIFT_OPERATORS_(uint64_t)
    NT_B128_SHIFT_OPERATORS_(size_t)


    NT_ALWAYS_INLINE constexpr b128 operator&(const b128& rhs) const noexcept {
        return b128(hi_bits & rhs.hi_bits, lo_bits & rhs.lo_bits);
    }

    NT_ALWAYS_INLINE constexpr b128& operator&=(const b128& rhs) noexcept {
        hi_bits &= rhs.hi_bits;
        lo_bits &= rhs.lo_bits;
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b128 operator&(const uint64_t& rhs) const noexcept { return *this & b128(rhs); }
    NT_ALWAYS_INLINE constexpr b128& operator&=(const uint64_t& rhs) noexcept { return *this &= b128(rhs); }
    NT_ALWAYS_INLINE constexpr b128 operator&(const int64_t& rhs) const noexcept { 
        return b128(hi_bits, lo_bits & rhs);
    }
    NT_ALWAYS_INLINE constexpr b128& operator&=(const int64_t& rhs) noexcept { 
        lo_bits &= rhs;
        return *this;
    }



    NT_ALWAYS_INLINE constexpr b128 operator|(const b128 & rhs) const noexcept {
        return b128(hi_bits | rhs.hi_bits, lo_bits | rhs.lo_bits);
    }

    NT_ALWAYS_INLINE constexpr b128& operator|=(const b128& rhs) noexcept {
        hi_bits |= rhs.hi_bits;
        lo_bits |= rhs.lo_bits;
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b128 operator|(const uint64_t& rhs) const noexcept { 
        return b128(hi_bits, lo_bits | rhs);
    }
    NT_ALWAYS_INLINE constexpr b128& operator|=(const uint64_t& rhs) noexcept { 
        lo_bits |= rhs;
        return *this;
    }


    NT_ALWAYS_INLINE constexpr b128 operator|(const int64_t& rhs) const noexcept { 
        return b128(hi_bits, lo_bits | rhs);
    }
    NT_ALWAYS_INLINE constexpr b128& operator|=(const int64_t& rhs) noexcept { 
        lo_bits |= rhs;
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b128 operator^(const b128& rhs) const noexcept {
        return b128(hi_bits ^ rhs.hi_bits, lo_bits ^ rhs.lo_bits);
    }

    NT_ALWAYS_INLINE constexpr b128& operator^=(const b128& rhs) noexcept {
        hi_bits ^= rhs.hi_bits;
        lo_bits ^= rhs.lo_bits;
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b128 operator^(const uint64_t& rhs) const noexcept { 
        return b128(hi_bits, lo_bits ^ rhs);
    }
    NT_ALWAYS_INLINE constexpr b128& operator^=(const uint64_t& rhs) noexcept { 
        lo_bits |= rhs;
        return *this;
    }


    NT_ALWAYS_INLINE constexpr b128 operator^(const int64_t& rhs) const noexcept { 
        return b128(hi_bits, lo_bits ^ rhs);
    }
    NT_ALWAYS_INLINE constexpr b128& operator^=(const int64_t& rhs) noexcept { 
        lo_bits |= rhs;
        return *this;
    }


    
    NT_ALWAYS_INLINE constexpr b128 operator~() const noexcept {
        return b128(~hi_bits, ~lo_bits);
    }
    NT_ALWAYS_INLINE constexpr operator bool() const noexcept {
        return (bool) (hi_bits | lo_bits);
    }
    

   NT_ALWAYS_INLINE constexpr b128 operator-() const noexcept {
        uint64_t nlo = ~lo_bits + 1;
        uint64_t nhi = ~hi_bits + (nlo == 0 ? 1 : 0); 
        return b128(nhi, nlo);
    }

    // returns the number of bits that are occupied
    inline constexpr uint8_t bits() const noexcept {
        uint8_t out = 0;
        if(hi_bits){
            out = 64;
            uint64_t hi = hi_bits;
            while(hi){
                ++out;
                hi >>= 1;
            }
        }else{
            uint64_t lo = lo_bits;
            while(lo){
                ++out;
                lo >>= 1;
            }
        }
        return out;
    }

    NT_ALWAYS_INLINE constexpr bool bit(int pos) const noexcept {
        if (pos < 0 || pos >= 128) {
            return false; // or assert
        }
        if (pos < 64) {
            return (lo_bits & (uint64_t(1) << pos)) != 0;
        }
        return (hi_bits & (uint64_t(1) << (pos - 64))) != 0;
    }

    
    // the clz function returns the number of bits that are 0 before the first leading 1 bit
    NT_ALWAYS_INLINE static constexpr uint8_t clz64(uint64_t x) noexcept {
        if (x == 0) return 64;
        uint8_t n = 0;
        uint64_t mask = 1ull << 63;
        while ((x & mask) == 0) {
            ++n;
            mask >>= 1;
        }
        return n;
    }

    NT_ALWAYS_INLINE constexpr uint8_t clz() const noexcept {
        if (hi_bits != 0)
            return clz64(hi_bits);

        if (lo_bits != 0)
            return 64 + clz64(lo_bits);

        return 128; // value == 0
    }

    inline static constexpr std::pair<b128, b128> divmod(const b128& lhs, const b128& rhs){
        constexpr b128 b128_0 = b128(0);
        constexpr b128 b128_1 = b128(1);
        // Save some calculations /////////////////////
        if (rhs == b128_0){
            throw std::domain_error("Error: division or modulus by 0");
        }
        else if (rhs == b128_1){
            return std::pair <b128, b128> (lhs, b128_0);
        }
        else if (lhs == rhs){
            return std::pair <b128, b128> (b128_1, b128_0);
        }
        else if ((lhs == b128_0) || (lhs < rhs)){
            return std::pair <b128, b128> (b128_0, lhs);
        }

        std::pair <b128, b128> qr (b128_0, b128_0);
        for(uint8_t x = lhs.bits(); x > 0; --x){
            qr.first  <<= 1;
            qr.second <<= 1;

            if ((lhs >> (x - 1U)) & 1){
                ++qr.second;
            }

            if (qr.second >= rhs){
                qr.second -= rhs;
                ++qr.first;
            }
        }
        return qr;
    }

    NT_ALWAYS_INLINE constexpr b128 operator/(const b128& rhs) const {
        return divmod(*this, rhs).first;
    }

    NT_ALWAYS_INLINE constexpr b128& operator/=(const b128& rhs) {
        *this = *this / rhs;
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b128 operator/(const uint64_t& rhs) const {
        return *this / b128(rhs);
    }
    
    
    NT_ALWAYS_INLINE constexpr b128 operator/(int64_t v) const {
        if (v == 0) return b128(0, 0);

        bool neg = (v < 0);

        // Compute absolute value in uint64_t safely
        uint64_t mag = neg
            ? (uint64_t)(-(uint64_t)v)   // safe: negation happens in uint64_t
            : (uint64_t)v;

        b128 out = (*this) / mag;

        return neg ? -out : out;
    }

    NT_ALWAYS_INLINE constexpr b128& operator/=(const uint64_t& rhs) {
        *this = (*this / rhs);
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b128& operator/=(const int64_t& rhs) {
        *this = (*this / rhs);
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b128 operator%(const b128& rhs) const {
        return divmod(*this, rhs).second;
    }

    NT_ALWAYS_INLINE constexpr b128& operator%=(const b128& rhs) {
        *this = *this % rhs;
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b128 operator%(const uint64_t& rhs) const {
        return *this % b128(rhs);
    }
    
    
    NT_ALWAYS_INLINE constexpr b128 operator%(int64_t v) const {
        bool neg = (v < 0);

        // Compute absolute value in uint64_t safely
        uint64_t mag = neg
            ? (uint64_t)(-(uint64_t)v)   // safe: negation happens in uint64_t
            : (uint64_t)v;

        b128 out = (*this) % mag;
        return out;
    }

    NT_ALWAYS_INLINE constexpr b128& operator%=(const uint64_t& rhs) {
        *this = (*this % rhs);
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b128& operator%=(const int64_t& rhs) {
        *this = (*this % rhs);
        return *this;
    }


    inline constexpr b128 operator*(const b128& rhs) const noexcept {
        // split values into 4 32-bit parts
        uint64_t top[4] = {hi_bits >> 32, hi_bits & 0xffffffff, lo_bits >> 32, lo_bits & 0xffffffff};
        uint64_t bottom[4] = {rhs.hi_bits >> 32, rhs.hi_bits & 0xffffffff, rhs.lo_bits >> 32, rhs.lo_bits & 0xffffffff};
        uint64_t products[4][4] {
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        };

        // multiply each component of the values
        for(int y = 3; y > -1; y--){
            for(int x = 3; x > -1; x--){
                products[3 - x][y] = top[x] * bottom[y];
            }
        }

        // first row
        uint64_t fourth32 = (products[0][3] & 0xffffffff);
        uint64_t third32  = (products[0][2] & 0xffffffff) + (products[0][3] >> 32);
        uint64_t second32 = (products[0][1] & 0xffffffff) + (products[0][2] >> 32);
        uint64_t first32  = (products[0][0] & 0xffffffff) + (products[0][1] >> 32);

        // second row
        third32  += (products[1][3] & 0xffffffff);
        second32 += (products[1][2] & 0xffffffff) + (products[1][3] >> 32);
        first32  += (products[1][1] & 0xffffffff) + (products[1][2] >> 32);

        // third row
        second32 += (products[2][3] & 0xffffffff);
        first32  += (products[2][2] & 0xffffffff) + (products[2][3] >> 32);

        // fourth row
        first32  += (products[3][3] & 0xffffffff);

        // move carry to next digit
        third32  += fourth32 >> 32;
        second32 += third32  >> 32;
        first32  += second32 >> 32;

        // remove carry from current digit
        fourth32 &= 0xffffffff;
        third32  &= 0xffffffff;
        second32 &= 0xffffffff;
        first32  &= 0xffffffff;

        // combine components
        return b128((first32 << 32) | second32, (third32 << 32) | fourth32);
    }

    inline static constexpr std::pair<uint64_t, uint64_t> multiply_64x64(uint64_t a, uint64_t b) {
        // Splits a*b into two 64-bit results: high and low
        
        // Low 32 bits of a and b
        uint64_t al = a & 0xFFFFFFFF;
        uint64_t bl = b & 0xFFFFFFFF;

        // High 32 bits of a and b
        uint64_t ah = a >> 32;
        uint64_t bh = b >> 32;

        // Four partial 64-bit products
        uint64_t p0 = al * bl;             // L*L (bits 0-63)
        uint64_t p1 = al * bh;             // L*H (bits 32-95)
        uint64_t p2 = ah * bl;             // H*L (bits 32-95)
        uint64_t p3 = ah * bh;             // H*H (bits 64-127)

        // Intermediate carry (p1 and p2 contribute to the middle 64 bits)
        uint64_t carry = p1 + p2;
        
        // Low result (64 bits)
        uint64_t low = p0 + (carry << 32); 
        
        // High result (64 bits)
        uint64_t high = p3 + (carry >> 32) + (low < p0 ? 1 : 0); // Check for overflow from low sum
        
        return {high, low};
    }

    // Function that returns a pair of b128: {High 128 bits, Low 128 bits}
    inline static constexpr std::pair<b128, b128> multiply_128x128(const b128& a, const b128& b) noexcept {
        // 1. Split a and b into two 64-bit limbs:
        // a = [A1, A0], b = [B1, B0]
        uint64_t A1 = a.hi_bits;
        uint64_t A0 = a.lo_bits;
        uint64_t B1 = b.hi_bits;
        uint64_t B0 = b.lo_bits;

        // 2. Compute the four 64x64 products (total 4 products, each 128 bits):
        // P0 = A0 * B0 (Lowest 128 bits)
        // P1 = A1 * B0
        // P2 = A0 * B1
        // P3 = A1 * B1 (Highest 128 bits)

        std::pair<uint64_t, uint64_t> P0 = multiply_64x64(A0, B0); // P0.first (high 64), P0.second (low 64)
        std::pair<uint64_t, uint64_t> P1 = multiply_64x64(A1, B0);
        std::pair<uint64_t, uint64_t> P2 = multiply_64x64(A0, B1);
        std::pair<uint64_t, uint64_t> P3 = multiply_64x64(A1, B1);

        // 3. Sum the partial products with carries (Long Addition):
        // Result R is split into 4 64-bit limbs: R3 | R2 | R1 | R0

        // R0 is simply the low part of P0
        uint64_t R0 = P0.second; 

        // R1 = P0.high + P1.low + P2.low
        uint64_t R1_carry = 0;
        
        // Start with P0.first
        uint64_t R1 = P0.first; 
        
        // Add P1.second (low 64 of A1*B0)
        R1 += P1.second;
        if (R1 < P1.second) R1_carry = 1; // Check for overflow after R1 += P1.second

        // Add P2.second (low 64 of A0*B1)
        uint64_t temp_R1 = R1;
        R1 += P2.second;
        if (R1 < temp_R1) R1_carry += 1; // Check for overflow after R1 += P2.second

        // R2 = P3.low + P1.high + P2.high + R1_carry
        uint64_t R2_carry = 0;
        
        // Start with P3.second (low 64 of A1*B1)
        uint64_t R2 = P3.second; 
        
        // Add P1.first
        uint64_t temp_R2 = R2;
        R2 += P1.first;
        if (R2 < temp_R2) R2_carry = 1;
        
        // Add P2.first
        temp_R2 = R2;
        R2 += P2.first;
        if (R2 < temp_R2) R2_carry += 1;
        
        // Add R1_carry
        R2 += R1_carry;
        if (R2 < R1_carry) R2_carry += 1;
        
        // R3 = P3.high + R2_carry
        uint64_t R3 = P3.first + R2_carry;
        
        // 4. Combine into b128 pair:
        b128 R_high(R3, R2);
        b128 R_low(R1, R0);

        return {R_high, R_low};
    }

    NT_ALWAYS_INLINE static constexpr std::pair<b128, b128> shift_256_right(
        const b128& a, const b128& b, const int64_t& s) noexcept {
        if (s == 0) return std::pair<b128, b128>(a, b);
        if (s >= 256) return std::pair<b128, b128>(b128(0,0), b128(0, 0));
        if (s >= 128) {
            // hi_bits moves entirely into lo_bits
            return std::pair<b128, b128>(b128(0, 0), a >> (s - 128));
        }
        b128 new_hi = a >> s;
        b128 new_lo = (b >> s) | (a << (128 - s));
        return {new_hi, new_lo};
    }



    inline constexpr b128 operator*(const uint64_t& rhs) const noexcept {
        // split values into 4 32-bit parts
        uint64_t top[4] = {hi_bits >> 32, hi_bits & 0xffffffff, lo_bits >> 32, lo_bits & 0xffffffff};
        uint64_t bottom[4] = {0, 0, rhs >> 32, rhs & 0xffffffff};
        uint64_t products[4][4] = {
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        };


        // multiply each component of the values
        for(int y = 3; y > -1; y--){
            for(int x = 3; x > -1; x--){
                products[3 - x][y] = top[x] * bottom[y];
            }
        }

        // first row
        uint64_t fourth32 = (products[0][3] & 0xffffffff);
        uint64_t third32  = (products[0][2] & 0xffffffff) + (products[0][3] >> 32);
        uint64_t second32 = (products[0][1] & 0xffffffff) + (products[0][2] >> 32);
        uint64_t first32  = (products[0][0] & 0xffffffff) + (products[0][1] >> 32);

        // second row
        third32  += (products[1][3] & 0xffffffff);
        second32 += (products[1][2] & 0xffffffff) + (products[1][3] >> 32);
        first32  += (products[1][1] & 0xffffffff) + (products[1][2] >> 32);

        // third row
        second32 += (products[2][3] & 0xffffffff);
        first32  += (products[2][2] & 0xffffffff) + (products[2][3] >> 32);

        // fourth row
        first32  += (products[3][3] & 0xffffffff);

        // move carry to next digit
        third32  += fourth32 >> 32;
        second32 += third32  >> 32;
        first32  += second32 >> 32;

        // remove carry from current digit
        fourth32 &= 0xffffffff;
        third32  &= 0xffffffff;
        second32 &= 0xffffffff;
        first32  &= 0xffffffff;

        // combine components
        return b128((first32 << 32) | second32, (third32 << 32) | fourth32);
    }
    
    NT_ALWAYS_INLINE constexpr b128& operator*=(b128 v){
        *this = (*this * v);
        return *this;
    } 

    NT_ALWAYS_INLINE constexpr b128& operator*=(uint64_t v){
        *this = (*this * v);
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b128 operator*(int64_t v) const {
        if (v == 0) return b128(0, 0);

        bool neg = (v < 0);

        // Compute absolute value in uint64_t safely
        uint64_t mag = neg
            ? (uint64_t)(-(uint64_t)v)   // safe: negation happens in uint64_t
            : (uint64_t)v;

        b128 out = (*this) * mag;

        return neg ? -out : out;
    }

    NT_ALWAYS_INLINE constexpr b128& operator*=(int64_t v) {
        *this = (*this * v);
        return *this;
    }

    NT_BIT_OPERATORS_UNSIGNED_(uint8_t)
    NT_BIT_OPERATORS_UNSIGNED_(uint16_t)
    NT_BIT_OPERATORS_UNSIGNED_(uint32_t)
    NT_BIT_OPERATORS_SIGNED_(int8_t)
    NT_BIT_OPERATORS_SIGNED_(int16_t)
    NT_BIT_OPERATORS_SIGNED_(int32_t)


    // pack IEEE-754 binary128 fields into b128
    // sign: 0/1, exp: 15-bit raw biased, frac_hi: high 48? / frac_low: low 64 bits
    // fraction must be 112-bit as hi: upper 48 bits placed into hi_bits low bits, low 64 into lo_bits
    NT_ALWAYS_INLINE static constexpr b128 pack_ieee128(uint16_t sign, uint16_t exp_raw, uint64_t frac_hi48, uint64_t frac_lo64) noexcept {
        // hi_bits layout: [sign(1) at bit127][exp(15) bits at 126..112][frac high 48 bits at 111..64]
        // lo_bits: frac low 64 bits at 63..0
        uint64_t hi = 0;
        // place sign into hi at bit 63 of hi (since hi_bits corresponds to bits [127..64])
        if (sign) hi |= (1ull << 63);
        // place exponent in bits 62..48 of hi (that's bits 126..112 overall)
        hi |= (uint64_t(exp_raw & 0x7FFFu) << 48);
        // place top 48 of fraction into low 48 bits of hi (bits 47..0 of hi -> overall bits 111..64)
        hi |= (frac_hi48 & 0x0000FFFFFFFFFFFFull);
        uint64_t lo = frac_lo64;
        return b128(hi, lo);
    }
};

#undef NT_B128_SHIFT_OPERATORS_ 
#undef NT_BIT_OPERATORS_UNSIGNED_
#undef NT_BIT_OPERATORS_SIGNED_


}


inline std::ostream& operator<<(std::ostream& os, const nt::b128 i){
  std::ostream::sentry s(os);
  if (s) {
    nt::b128 tmp = i;
    char buffer[128];
    char *d = std::end(buffer);
    do {
      --d;
      *d = "0123456789"[(tmp % 10).lo_bits];
      tmp /= 10;
    } while (tmp != 0);
    int len = std::end(buffer) - d;
    if (os.rdbuf()->sputn(d, len) != len) {
      os.setstate(std::ios_base::badbit);
    }
  }
  return os;
}


#endif
