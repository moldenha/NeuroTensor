#ifndef NT_TYPES_128_BIT_TYPES_B256_H__
#define NT_TYPES_128_BIT_TYPES_B256_H__ 
// This is a struct which acts in many ways like a uint256

#include "../../utils/always_inline_macro.h"
#include "b128.h"
#include <cstddef>
#include <vector>
#include <exception>
#include <iostream>


namespace nt{

#define NT_B256_SHIFT_OPERATORS_(type)\
 NT_ALWAYS_INLINE constexpr b256 operator>>(const type& s) const noexcept {return *this >> (int64_t)s;}\
 NT_ALWAYS_INLINE constexpr b256 operator<<(const type& s) const noexcept {return *this << (int64_t)s;}\
 NT_ALWAYS_INLINE constexpr b256& operator>>=(const type& s) noexcept {return *this >>= (int64_t)s;}\
 NT_ALWAYS_INLINE constexpr b256& operator<<=(const type& s) noexcept {return *this <<= (int64_t)s;}\

#define NT_BIT_OPERATORS_UNSIGNED_(type)\
    NT_ALWAYS_INLINE constexpr b256 operator&(const type& s) const noexcept {return *this & (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256& operator&=(const type& s) noexcept {return *this &= (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256 operator^(const type& s) const noexcept {return *this ^ (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256& operator^=(const type& s) noexcept {return *this ^= (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256 operator|(const type& s) const noexcept {return *this | (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256& operator|=(const type& s) noexcept {return *this |= (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256 operator-(const type& s) const noexcept {return *this - (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256& operator-=(const type& s) noexcept {return *this -= (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256 operator+(const type& s) const noexcept {return *this + (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256& operator+=(const type& s) noexcept {return *this += (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256 operator*(const type& s) const noexcept {return *this * (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256& operator*=(const type& s) noexcept {return *this *= (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256 operator/(const type& s) const noexcept {return *this / (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256& operator/=(const type& s) noexcept {return *this /= (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256 operator%(const type& s) const noexcept {return *this % (uint64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256& operator%=(const type& s) noexcept {return *this %= (uint64_t)(s);}\

#define NT_BIT_OPERATORS_SIGNED_(type)\
    NT_ALWAYS_INLINE constexpr b256 operator&(const type& s) const noexcept {return *this & (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256& operator&=(const type& s) noexcept {return *this &= (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256 operator^(const type& s) const noexcept {return *this ^ (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256& operator^=(const type& s) noexcept {return *this ^= (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256 operator|(const type& s) const noexcept {return *this | (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256& operator|=(const type& s) noexcept {return *this |= (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256 operator-(const type& s) const noexcept {return *this - (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256 operator-=(const type& s) noexcept {return *this -= (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256 operator+(const type& s) const noexcept {return *this + (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256 operator+=(const type& s) noexcept {return *this += (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256 operator*(const type& s) const noexcept {return *this * (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256 operator*=(const type& s) noexcept {return *this *= (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256 operator/(const type& s) const noexcept {return *this / (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256 operator/=(const type& s) noexcept {return *this /= (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256 operator%(const type& s) const noexcept {return *this % (int64_t)(s);}\
    NT_ALWAYS_INLINE constexpr b256 operator%=(const type& s) noexcept {return *this %= (int64_t)(s);}\


struct b256{
    b128 hi_bits, lo_bits;
    constexpr b256() noexcept : hi_bits(0), lo_bits(0) {}
    constexpr b256(b128 h, b128 l) noexcept : hi_bits(h), lo_bits(l) {}
    constexpr b256(uint64_t ha, uint64_t hb, uint64_t la, uint64_t lb) noexcept : hi_bits(ha, hb), lo_bits(la, lb) {}
    constexpr explicit b256(b128 l) noexcept : hi_bits(0), lo_bits(l) {}
    constexpr explicit b256(uint64_t l) noexcept : hi_bits(0), lo_bits(l) {}
    constexpr b256(const b256&) noexcept = default;
    constexpr b256(b256&&) noexcept = default;
    NT_ALWAYS_INLINE constexpr b256& operator=(const b256&) = default;
    NT_ALWAYS_INLINE constexpr b256& operator=(b256&&) = default;


    // == helpers ==
    NT_ALWAYS_INLINE constexpr bool is_zero() const noexcept { return hi_bits.is_zero() && lo_bits.is_zero(); }
    NT_ALWAYS_INLINE constexpr b256& zero() noexcept { hi_bits.zero(); lo_bits.zero(); return *this; }

    // compare
    NT_ALWAYS_INLINE constexpr int cmp(const b256& o) const noexcept {
        if (hi_bits < o.hi_bits) return -1;
        if (hi_bits > o.hi_bits) return  1;
        if (lo_bits < o.lo_bits) return -1;
        if (lo_bits > o.lo_bits) return  1;
        return 0;
    }

    NT_ALWAYS_INLINE constexpr bool operator==(const b256& o) const noexcept { return hi_bits == o.hi_bits && lo_bits == o.lo_bits; }
    NT_ALWAYS_INLINE constexpr bool operator!=(const b256& o) const noexcept { return !(*this == o); }
    NT_ALWAYS_INLINE constexpr bool operator<(const b256& o) const noexcept { return cmp(o) < 0; }
    NT_ALWAYS_INLINE constexpr bool operator<=(const b256& o) const noexcept { return cmp(o) <= 0; }
    NT_ALWAYS_INLINE constexpr bool operator>(const b256& o) const noexcept { return cmp(o) > 0; }
    NT_ALWAYS_INLINE constexpr bool operator>=(const b256& o) const noexcept { return cmp(o) >= 0; }

    NT_ALWAYS_INLINE constexpr int bit_at(int i) const noexcept {
        return (i < 128) ? lo_bits.bit_at(i) : hi_bits.bit_at(i-128);
    }


    // add (b256 + b256)
    NT_ALWAYS_INLINE constexpr b256 operator+(const b256& o) const noexcept {
        b128 new_lo = lo_bits + o.lo_bits;
        b128 carry = (new_lo < lo_bits) ? b128(1ull) : b128(0ull); // overflow detection
        b128 new_hi = hi_bits + o.hi_bits + carry;
        return b256(new_hi, new_lo);
    }


    NT_ALWAYS_INLINE constexpr b256& operator+=(const b256& o) noexcept {
        *this = (*this + o);
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b256 operator+(const b128& o) const noexcept {
        b128 new_lo = lo_bits + o;
        b128 carry = (new_lo < lo_bits) ? b128(1ull) : b128(0ull); // overflow detection
        b128 new_hi = hi_bits + carry;
        return b256(new_hi, new_lo);
    }

    NT_ALWAYS_INLINE constexpr b256& operator+=(const b128& o) noexcept {
        *this = (*this + o);
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b256 operator+(uint64_t b) const noexcept {
        return *this + b128(b);
    }

    NT_ALWAYS_INLINE constexpr b256& operator+=(uint64_t b) {
        return *this += b128(b);
    }



    NT_ALWAYS_INLINE constexpr b256& operator++() noexcept {
        constexpr b256 b256_1(1);
        return *this += b256_1;
    }

    NT_ALWAYS_INLINE constexpr b256 operator++(int){
        b256 temp(*this);
        ++*this;
        return temp;
    }

    // subtract (b256 - b256)
    NT_ALWAYS_INLINE constexpr b256 operator-(const b256& o) const noexcept {
        b128 new_lo = lo_bits - o.lo_bits;
        b128 borrow = (lo_bits < o.lo_bits) ? b128(1ull) : b128(0ull);
        b128 new_hi = hi_bits - o.hi_bits - borrow;
        return b256(new_hi, new_lo);
    }
    
    NT_ALWAYS_INLINE constexpr b256& operator-=(const b256& o) noexcept {
        b128 new_lo = lo_bits - o.lo_bits;
        b128 borrow = (lo_bits < o.lo_bits) ? b128(1ull) : b128(0ull);
        b128 new_hi = hi_bits - o.hi_bits - borrow;
        hi_bits = new_hi;
        lo_bits = new_lo;
        return *this;
    }
    
    NT_ALWAYS_INLINE constexpr b256 operator-(b128 b) const noexcept {
        b128 new_lo = lo_bits - b;
        b128 borrow = (lo_bits < b) ? b128(1) : b128(0);
        return b256(hi_bits - borrow, new_lo);
    }

    NT_ALWAYS_INLINE constexpr b256& operator-=(b128 b) {
        b128 new_lo = lo_bits - b;
        b128 borrow = (lo_bits < b) ? b128(1) : b128(0);
        hi_bits -= borrow;
        lo_bits = new_lo;
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b256 operator-(uint64_t b) const noexcept {
        return *this - b128(b);
    }

    NT_ALWAYS_INLINE constexpr b256& operator-=(uint64_t b) {
        return *this -= b128(b);
    }
    
    NT_ALWAYS_INLINE constexpr b256 operator-(int64_t b) const noexcept { return (b < 0) ? (*this + (uint64_t(-b))) : (*this - (uint64_t)b);}
    NT_ALWAYS_INLINE constexpr b256& operator-=(int64_t b) noexcept { return (b < 0) ? (*this += (uint64_t(-b))) : (*this -= (uint64_t)b);}
    NT_ALWAYS_INLINE constexpr b256 operator+(const int64_t& b) const noexcept {return (b < 0) ? (*this - uint64_t(-b)) : (*this + uint64_t(b));}
    NT_ALWAYS_INLINE constexpr b256& operator+=(const int64_t& b) noexcept {return (b < 0) ? (*this -= uint64_t(-b)) : (*this += uint64_t(b));}


    NT_ALWAYS_INLINE constexpr b256 operator>>(const int64_t& s) const noexcept {
        if (s == 0) return b256(hi_bits, lo_bits);
        if (s >= 256) return b256(0);
        if (s >= 128) {
            // hi_bits moves entirely into lo_bits
            return b256(b128(0), hi_bits >> (s - 128));
        }
        b128 new_hi = hi_bits >> s;
        b128 new_lo = (lo_bits >> s) | (hi_bits << (128 - s));
        return {new_hi, new_lo};
    }

    NT_ALWAYS_INLINE constexpr b256 operator<<(const int64_t& s) const noexcept {
        if (s == 0) return *this;
        if (s >= 256) return b256(0);

        if (s >= 128) {
            // lo_bits moves entirely into hi_bits
            return b256(lo_bits << (s - 128), b128(0));
        }

        // s in [0..127]
        b128 new_hi = (hi_bits << s) | (lo_bits >> (128 - s));
        b128 new_lo = lo_bits << s;
        return b256(new_hi, new_lo);
    }



    NT_ALWAYS_INLINE constexpr b256& operator<<=(const int64_t& s) noexcept {
        if (s == 0) return *this;
        if (s >= 256){
            hi_bits = b128(0);
            lo_bits = b128(0);
            return *this;
        }
        if (s >= 128) {
            // lo_bits moves entirely into hi_bits
            hi_bits = (lo_bits << (s-128));
            lo_bits = b128(0);
            return *this;
        }

        // s in [0..127]
        hi_bits = (hi_bits << s) | (lo_bits >> (128 - s));
        lo_bits <<= s;
        return *this; 
    }

    NT_ALWAYS_INLINE constexpr b256& operator>>=(const int64_t& s) noexcept {
        if (s == 0) return *this;
        if (s >= 256){
            hi_bits = b128(0);
            lo_bits = b128(0);
            return *this;
        }
        if (s >= 128) {
            // hi_bits moves entirely into lo_bits
            lo_bits = hi_bits >> (s-128);
            hi_bits = b128(0);
        }

        // s in [0..63]
        lo_bits = (lo_bits >> s) | (hi_bits << (128-s));
        hi_bits >>= s;
        return *this;
    }
    
    NT_B256_SHIFT_OPERATORS_(uint8_t)
    NT_B256_SHIFT_OPERATORS_(int8_t)
    NT_B256_SHIFT_OPERATORS_(uint16_t)
    NT_B256_SHIFT_OPERATORS_(int16_t)
    NT_B256_SHIFT_OPERATORS_(uint32_t)
    NT_B256_SHIFT_OPERATORS_(int32_t)
    NT_B256_SHIFT_OPERATORS_(uint64_t)
    NT_B256_SHIFT_OPERATORS_(size_t)


    NT_ALWAYS_INLINE constexpr b256 operator&(const b256& rhs) const noexcept {
        return b256(hi_bits & rhs.hi_bits, lo_bits & rhs.lo_bits);
    }

    NT_ALWAYS_INLINE constexpr b256& operator&=(const b256& rhs) noexcept {
        hi_bits &= rhs.hi_bits;
        lo_bits &= rhs.lo_bits;
        return *this;
    }
    NT_ALWAYS_INLINE constexpr b256 operator&(const b128& rhs) const noexcept { return *this & b256(rhs); }
    NT_ALWAYS_INLINE constexpr b256 operator&(const uint64_t& rhs) const noexcept { return *this & b128(rhs); }
    NT_ALWAYS_INLINE constexpr b256& operator&=(const b128& rhs) noexcept { return *this &= b256(rhs); }
    NT_ALWAYS_INLINE constexpr b256& operator&=(const uint64_t& rhs) noexcept { return *this &= b128(rhs); }
    NT_ALWAYS_INLINE constexpr b256 operator&(const int64_t& rhs) const noexcept { 
        return b256(hi_bits.hi_bits, hi_bits.lo_bits, lo_bits.hi_bits, lo_bits.lo_bits & rhs);
    }
    NT_ALWAYS_INLINE constexpr b256& operator&=(const int64_t& rhs) noexcept { 
        lo_bits.lo_bits &= rhs;
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b256 operator|(const b256& rhs) const noexcept {
        return b256(hi_bits | rhs.hi_bits, lo_bits | rhs.lo_bits); 
    }

    NT_ALWAYS_INLINE constexpr b256 operator|(const b128 & rhs) const noexcept {
        return b256(hi_bits.hi_bits, hi_bits.lo_bits, lo_bits.hi_bits | rhs.hi_bits, lo_bits.lo_bits | rhs.lo_bits); 
    }

    NT_ALWAYS_INLINE constexpr b256& operator|=(const b256& rhs) noexcept {
        hi_bits |= rhs.hi_bits;
        lo_bits |= rhs.lo_bits;
        return *this;
    }
    NT_ALWAYS_INLINE constexpr b256& operator|=(const b128& rhs) noexcept {
        lo_bits.hi_bits |= rhs.hi_bits;
        lo_bits.lo_bits |= rhs.lo_bits;
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b256 operator|(const uint64_t& rhs) const noexcept { 
        return b256(hi_bits.hi_bits, hi_bits.lo_bits, lo_bits.hi_bits, lo_bits.lo_bits | rhs); 
    }

    NT_ALWAYS_INLINE constexpr b256& operator|=(const uint64_t& rhs) noexcept { 
        lo_bits.lo_bits |= rhs;
        return *this;
    }


    NT_ALWAYS_INLINE constexpr b256 operator|(const int64_t& rhs) const noexcept { 
        return b256(hi_bits.hi_bits, hi_bits.lo_bits, lo_bits.hi_bits, lo_bits.lo_bits | rhs);
    }
    NT_ALWAYS_INLINE constexpr b256& operator|=(const int64_t& rhs) noexcept { 
        lo_bits.lo_bits |= rhs;
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b256 operator^(const b256& rhs) const noexcept {
        return b256(hi_bits ^ rhs.hi_bits, lo_bits ^ rhs.lo_bits);
    }
    
    NT_ALWAYS_INLINE constexpr b256 operator^(const b128& rhs) const noexcept {
        return b256(hi_bits, lo_bits ^ rhs);
    }

    NT_ALWAYS_INLINE constexpr b256 operator^(const int64_t& rhs) const noexcept {
        return b256(hi_bits.hi_bits, hi_bits.lo_bits, lo_bits.hi_bits, lo_bits.lo_bits ^ rhs);
    }

    NT_ALWAYS_INLINE constexpr b256 operator^(const uint64_t& rhs) const noexcept {
        return b256(hi_bits.hi_bits, hi_bits.lo_bits, lo_bits.hi_bits, lo_bits.lo_bits ^ rhs);
    }

    NT_ALWAYS_INLINE constexpr b256& operator^=(const b256& rhs) noexcept {
        hi_bits ^= rhs.hi_bits;
        lo_bits ^= rhs.lo_bits;
        return *this;
    }
    
    NT_ALWAYS_INLINE constexpr b256& operator^=(const b128& rhs) noexcept {
        lo_bits ^= rhs;
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b256& operator^=(const int64_t& rhs) noexcept {
        lo_bits.lo_bits ^= rhs;
        return *this;
    }


    NT_ALWAYS_INLINE constexpr b256& operator^=(const uint64_t& rhs) noexcept {
        lo_bits.lo_bits ^= rhs;
        return *this;
    }

    
    NT_ALWAYS_INLINE constexpr b256 operator~() const noexcept {
        return b256(~hi_bits, ~lo_bits);
    }
    NT_ALWAYS_INLINE constexpr operator bool() const noexcept {
        return bool(hi_bits | lo_bits);
    }
    

   NT_ALWAYS_INLINE constexpr b256 operator-() const noexcept {
        b128 nlo = ~lo_bits + 1;
        b128 nhi = ~hi_bits + (nlo.is_zero() ? b128(1) : b128(0)); 
        return b256(nhi, nlo);
    }

    // returns the number of bits that are occupied
    NT_ALWAYS_INLINE constexpr uint16_t bits() const noexcept {
        if(hi_bits){
            return uint16_t(128) + uint16_t(hi_bits.bits());
        }else{
            return uint16_t(lo_bits.bits());
        }
    }

    // returns true if there is a bit in that position
    NT_ALWAYS_INLINE constexpr bool bit(int pos) const noexcept {
        if(pos < 128){
            return lo_bits.bit(pos);
        }
        return hi_bits.bit(pos - 128);
    }

    
    // the clz function returns the number of bits that are 0 before the first leading 1 bit
    NT_ALWAYS_INLINE constexpr uint16_t clz() const noexcept {
        if (hi_bits != b128(0))
            return uint16_t(hi_bits.clz());

        if (lo_bits != 0)
            return uint16_t(128) + uint16_t(lo_bits.clz());

        return uint16_t(256); // value == 0
    }

    // slightly faster, useful in some cases
    inline static constexpr std::pair<b256, b256> divmod_noexcept(const b256& lhs, const b256& rhs) noexcept {
        constexpr b256 b256_0 = b256(0);
        constexpr b256 b256_1 = b256(1);
        // Save some calculations
        if(rhs == b256_0){
            return std::pair<b256, b256>(b256_0, b256_0);
        }
        else if (rhs == b256_1){
            return std::pair<b256, b256>(lhs, b256_0);
        }
        else if (lhs == rhs){
            return std::pair<b256, b256>(b256_1, b256_0);
        }
        else if ((lhs == b256_0) || (lhs < rhs)){
            return std::pair <b256, b256> (b256_0, lhs);
        }
        std::pair <b256, b256> qr (b256_0, b256_0);
        for(uint16_t x = lhs.bits(); x > 0; --x){
            qr.first  <<= 1;
            qr.second <<= 1;

            if(lhs.bit(x-1)){
                ++qr.second;
            }

            if (qr.second >= rhs){
                qr.second -= rhs;
                ++qr.first;
            }
        }
        return qr;
    }

    inline static constexpr std::pair<b256, b256>
    divmod(const b256& lhs, const b256& rhs) {
        constexpr b256 b256_0(0);
        constexpr b256 b256_1(1);

        if (rhs == b256_0) {
            throw std::domain_error("division by zero");
        }

        if (lhs < rhs) {
            return { b256_0, lhs };
        }

        if (rhs == b256_1) {
            return { lhs, b256_0 };
        }

        if (lhs == rhs) {
            return { b256_1, b256_0 };
        }

        b256 q(0);
        b256 r(0);

        const int nbits = lhs.bits(); // MUST be msb+1

        for (int i = nbits; i > 0; --i) {
            q <<= 1;
            r <<= 1;

            if (lhs.bit(i - 1)) {
                ++r;
            }

            if (r >= rhs) {
                r -= rhs;
                ++q;
            }
        }

        return { q, r };
    }

    NT_ALWAYS_INLINE constexpr b256 operator/(const b256& rhs) const {
        return divmod(*this, rhs).first;
    }

    NT_ALWAYS_INLINE constexpr b256& operator/=(const b256& rhs) {
        *this = *this / rhs;
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b256 operator/(const b128& rhs) const {
        return *this / b256(rhs);
    }

    NT_ALWAYS_INLINE constexpr b256 operator/(const uint64_t& rhs) const {
        return *this / b256(rhs);
    }
    
    
    NT_ALWAYS_INLINE constexpr b256 operator/(int64_t v) const {
        if (v == 0) return b256(0, 0, 0, 0);

        bool neg = (v < 0);

        // Compute absolute value in uint64_t safely
        uint64_t mag = neg
            ? (uint64_t)(-(uint64_t)v)   // safe: negation happens in uint64_t
            : (uint64_t)v;

        b256 out = (*this) / mag;

        return neg ? -out : out;
    }

    NT_ALWAYS_INLINE constexpr b256& operator/=(const uint64_t& rhs) {
        *this = (*this / rhs);
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b256& operator/=(const int64_t& rhs) {
        *this = (*this / rhs);
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b256 operator%(const b256& rhs) const {
        return divmod(*this, rhs).second;
    }

    NT_ALWAYS_INLINE constexpr b256 operator%(const b128& rhs) const {
        return divmod(*this, b256(rhs)).second;
    }

    NT_ALWAYS_INLINE constexpr b256& operator%=(const b256& rhs) {
        *this = *this % rhs;
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b256& operator%=(const b128& rhs) {
        *this = *this % rhs;
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b256 operator%(const uint64_t& rhs) const {
        return *this % b256(0, 0, 0, rhs);
    }
    
    
    NT_ALWAYS_INLINE constexpr b256 operator%(int64_t v) const {
        bool neg = (v < 0);

        // Compute absolute value in uint64_t safely
        uint64_t mag = neg
            ? (uint64_t)(-(uint64_t)v)   // safe: negation happens in uint64_t
            : (uint64_t)v;

        b256 out = (*this) % mag;
        return out;
    }

    NT_ALWAYS_INLINE constexpr b256& operator%=(const uint64_t& rhs) {
        *this = (*this % rhs);
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b256& operator%=(const int64_t& rhs) {
        *this = (*this % rhs);
        return *this;
    }

    // not completely verified, will verify when needed
    inline constexpr b256 operator*(const b256& rhs) const noexcept {
        // split values into 8 32-bit parts
        uint64_t top[8] = {
                hi_bits.hi_bits >> 32, hi_bits.hi_bits & 0xffffffff, hi_bits.lo_bits >> 32, hi_bits.lo_bits & 0xffffffff, 
                lo_bits.hi_bits >> 32, lo_bits.hi_bits & 0xffffffff, lo_bits.lo_bits >> 32, lo_bits.lo_bits & 0xffffffff};

        uint64_t bottom[8] = {
                rhs.hi_bits.hi_bits >> 32, rhs.hi_bits.hi_bits & 0xffffffff, rhs.hi_bits.lo_bits >> 32, rhs.hi_bits.lo_bits & 0xffffffff,
                rhs.lo_bits.hi_bits >> 32, rhs.lo_bits.hi_bits & 0xffffffff, rhs.lo_bits.lo_bits >> 32, rhs.lo_bits.lo_bits & 0xffffffff};

        uint64_t products[8][8] {
            {0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0}
        };

        // multiply each component of the values
        for(int y = 7; y > -1; y--){
            for(int x = 7; x > -1; x--){
                products[7 - x][y] = top[x] * bottom[y];
            }
        }

        // first row
        uint64_t eighth32 = (products[0][7] & 0xffffffff);
        uint64_t seventh32= (products[0][6] & 0xffffffff) + (products[0][7] >> 32);
        uint64_t sixth32  = (products[0][5] & 0xffffffff) + (products[0][6] >> 32);
        uint64_t fifth32  = (products[0][4] & 0xffffffff) + (products[0][5] >> 32);
        uint64_t fourth32 = (products[0][3] & 0xffffffff) + (products[0][4] >> 32);
        uint64_t third32  = (products[0][2] & 0xffffffff) + (products[0][3] >> 32);
        uint64_t second32 = (products[0][1] & 0xffffffff) + (products[0][2] >> 32);
        uint64_t first32  = (products[0][0] & 0xffffffff) + (products[0][1] >> 32);

        // second row
        seventh32+= (products[1][7] & 0xffffffff);
        sixth32  += (products[1][6] & 0xffffffff) + (products[1][7] >> 32);
        fifth32  += (products[1][5] & 0xffffffff) + (products[1][6] >> 32);
        fourth32 += (products[1][4] & 0xffffffff) + (products[1][5] >> 32);
        third32  += (products[1][3] & 0xffffffff) + (products[1][4] >> 32);
        second32 += (products[1][2] & 0xffffffff) + (products[1][3] >> 32);
        first32  += (products[1][1] & 0xffffffff) + (products[1][2] >> 32);

        // third row
        sixth32  += (products[2][7] & 0xffffffff);
        fifth32  += (products[2][6] & 0xffffffff) + (products[2][7] >> 32);
        fourth32 += (products[2][5] & 0xffffffff) + (products[2][6] >> 32);
        third32  += (products[2][4] & 0xffffffff) + (products[2][5] >> 32);
        second32 += (products[2][3] & 0xffffffff) + (products[2][4] >> 32);
        first32  += (products[2][2] & 0xffffffff) + (products[2][3] >> 32);

        // fourth row
        fifth32  += (products[3][7] & 0xffffffff);
        fourth32 += (products[3][6] & 0xffffffff) + (products[3][7] >> 32);
        third32  += (products[3][5] & 0xffffffff) + (products[3][6] >> 32);
        second32 += (products[3][4] & 0xffffffff) + (products[3][5] >> 32);
        first32  += (products[3][3] & 0xffffffff) + (products[3][4] >> 32);

        // fifth row
        fourth32 += (products[4][7] & 0xffffffff);
        third32  += (products[4][6] & 0xffffffff) + (products[4][7] >> 32);
        second32 += (products[4][5] & 0xffffffff) + (products[4][6] >> 32);
        first32  += (products[4][4] & 0xffffffff) + (products[4][5] >> 32); 

        // sixth row
        third32  += (products[5][7] & 0xffffffff);
        second32 += (products[5][6] & 0xffffffff) + (products[5][7] >> 32);
        first32  += (products[5][5] & 0xffffffff) + (products[5][6] >> 32); 
        
        // seventh row
        second32 += (products[6][7] & 0xffffffff);
        first32  += (products[6][6] & 0xffffffff) + (products[6][7] >> 32); 

        // eight row
        first32  += (products[7][7] & 0xffffffff); 

        // move carry to next digit
        seventh32+= eighth32 >> 32;
        sixth32  += seventh32 >> 32;
        fifth32  += sixth32 >> 32;
        fourth32 += fifth32 >> 32;
        third32  += fourth32 >> 32;
        second32 += third32  >> 32;
        first32  += second32 >> 32;

        // remove carry from current digit
        eighth32 &= 0xffffffff;
        seventh32&= 0xffffffff;
        sixth32  &= 0xffffffff;
        fifth32  &= 0xffffffff;
        third32  &= 0xffffffff;
        second32 &= 0xffffffff;
        first32  &= 0xffffffff;

        // combine components
        return b256((first32 << 32) | second32, (third32 << 32) | fourth32, (fifth32 << 32) | sixth32, (seventh32 << 32) | eighth32);
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

    // not completely verified, will verify when needed
    inline constexpr b256 operator*(const b128& rhs) const noexcept {
        return *this * b256(rhs);
    }

    inline constexpr b256 operator*(const uint64_t& rhs) const noexcept {
        return *this * b256(rhs);
    }

    NT_ALWAYS_INLINE constexpr b256& operator*=(b256 v){
        *this = (*this * v);
        return *this;
    }
    
    NT_ALWAYS_INLINE constexpr b256& operator*=(b128 v){
        *this = (*this * v);
        return *this;
    } 

    NT_ALWAYS_INLINE constexpr b256& operator*=(uint64_t v){
        *this = (*this * v);
        return *this;
    }

    NT_ALWAYS_INLINE constexpr b256 operator*(int64_t v) const {
        if (v == 0) return b256(0, 0, 0, 0);

        bool neg = (v < 0);

        // Compute absolute value in uint64_t safely
        uint64_t mag = neg
            ? (uint64_t)(-(uint64_t)v)   // safe: negation happens in uint64_t
            : (uint64_t)v;

        b256 out = (*this) * mag;

        return neg ? -out : out;
    }

    NT_ALWAYS_INLINE constexpr b256& operator*=(int64_t v) {
        *this = (*this * v);
        return *this;
    }

    NT_BIT_OPERATORS_UNSIGNED_(uint8_t)
    NT_BIT_OPERATORS_UNSIGNED_(uint16_t)
    NT_BIT_OPERATORS_UNSIGNED_(uint32_t)
    NT_BIT_OPERATORS_SIGNED_(int8_t)
    NT_BIT_OPERATORS_SIGNED_(int16_t)
    NT_BIT_OPERATORS_SIGNED_(int32_t)

};

#undef NT_B256_SHIFT_OPERATORS_ 
#undef NT_BIT_OPERATORS_UNSIGNED_
#undef NT_BIT_OPERATORS_SIGNED_


}


inline std::ostream& operator<<(std::ostream& os, const nt::b256 i){
  std::ostream::sentry s(os);
  if (s) {
    nt::b256 tmp = i;
    char buffer[256];
    char *d = std::end(buffer);
    do {
      --d;
      *d = "0123456789"[(tmp % 10).lo_bits.lo_bits];
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
