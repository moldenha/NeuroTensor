/*
 * This contains the nt::bitset class 
 * This class is meant to allow the user to specify the integer type, and the number of bits
 *  - MUST BE AN INTEGER
 *  - If wanting a float look at float_bits.h
 * the way it works is for example:
 * I have an integer 
 *  uint16_t(0) == 0000000000000000
 *
 * and I want the 5th bit to be 1, that would be:
 * constexpr nt::bitset<16, uint16_t> bits(uint16_t(0));
 * bits.set(4, true);
 * constexpr uint16_t out = bits.lo_type();
 *
 * and the bits of out would be:
 *    0000010000000000 
 *
*/


#ifndef NT_BIT_BITSET_H__
#define NT_BIT_BITSET_H__

#include "../utils/always_inline_macro.h"
#include "../utils/type_traits.h"
#include <utility>
#include <cstdint>
#include <cstddef>
#include <iostream>

namespace nt{

template<std::size_t N, class Type = uint32_t>
class bitset{
    static_assert(type_traits::is_integral_v<Type>, "Error bitset must use an integer type");
    static constexpr std::size_t BITS_PER_WORD = sizeof(Type) * CHAR_BIT;
    static constexpr std::size_t WORDS = (N + BITS_PER_WORD - 1) / BITS_PER_WORD;
    Type data[WORDS]{};
public:

    constexpr bitset(){
        for(std::size_t i = 0; i < WORDS; ++i){
            data[i] = Type(0);
        }
    }
    constexpr bitset(Type low_word)
    {
        if constexpr (WORDS > 0){
            data[0] = low_word;
        }
    }
    
    inline constexpr bool test(std::size_t pos) const {
        std::size_t word = pos / BITS_PER_WORD;
        std::size_t n_pos = pos - (word * BITS_PER_WORD);
        std::size_t bit  = BITS_PER_WORD - (n_pos+1);
        return (data[word] >> bit) & Type(1);
    }
    
    inline constexpr bool operator[](std::size_t pos) const {
        return test(pos);
    }

    inline constexpr void set(std::size_t pos, bool value) {
        std::size_t word = pos / BITS_PER_WORD;
        std::size_t n_pos = pos - (word * BITS_PER_WORD);
        std::size_t bit  = BITS_PER_WORD - (n_pos+1);
        Type mask = Type(1) << bit;
        if (value)
            data[word] |= mask;
        else
            data[word] &= ~mask;
    }

    inline constexpr void reset(std::size_t pos) {
        set(pos, false);
    }

    inline constexpr bitset operator<<(std::size_t shift) const {
        bitset result;

        if (shift == 0)
            return *this;

        if (shift >= N)
            return result; // everything becomes zero

        const std::size_t word_shift = shift / BITS_PER_WORD;
        const std::size_t bit_shift  = shift % BITS_PER_WORD;

        for (std::size_t i = WORDS; i-- > 0; ) {
            Type v = 0;

            // source word
            if (i >= word_shift) {
                v = data[i - word_shift] << bit_shift;

                // carry from next lower word
                if (bit_shift != 0 && i >= word_shift + 1) {
                    v |= (data[i - word_shift - 1] >>
                          (BITS_PER_WORD - bit_shift));
                }
            }

            result.data[i] = v;
        }

        // Clear unused high bits (beyond N)
        if constexpr ((N % BITS_PER_WORD) != 0) {
            const std::size_t high_bits = N % BITS_PER_WORD;
            const Type mask = (Type(1) << high_bits) - 1;
            result.data[WORDS - 1] &= mask;
        }

        return result;
    }

    inline constexpr bitset operator>>(std::size_t shift) const {
        bitset result;

        if (shift == 0)
            return *this;

        if (shift >= N)
            return result; // all zero

        const std::size_t word_shift = shift / BITS_PER_WORD;
        const std::size_t bit_shift  = shift % BITS_PER_WORD;

        for (std::size_t i = 0; i < WORDS; ++i) {
            Type v = 0;

            if (i + word_shift < WORDS) {
                v = data[i + word_shift] >> bit_shift;

                if (bit_shift != 0 &&
                    i + word_shift + 1 < WORDS) {
                    v |= (data[i + word_shift + 1] <<
                          (BITS_PER_WORD - bit_shift));
                }
            }

            result.data[i] = v;
        }

        return result;
    }

    inline constexpr Type lo_type() const {
        static_assert(WORDS >= 1, "bitset too small");
        return data[0];
    }
    
    inline constexpr Type hi_type() const {
        static_assert(WORDS >= 2, "bitset must be >= 128 bits");
        return data[1];
    }

    inline constexpr const Type* raw() const { return data; }

};


}

template<std::size_t N, typename Type>
inline std::ostream& operator<<(std::ostream& os, const nt::bitset<N, Type>& b){
    for(std::size_t i = 0; i < N; ++i){
        os << (b[i] ? 1 : 0);
    }
    return os;
}


#endif
