/*
 * This contains the nt::float_bitset class 
 * This class is meant to allow the user to specify the floating type, and the number of bits
 * the way it works is for example:
 * I have an float 
 *  float16_t(0) == 0000000000000000
 *
 * and I want the 5th bit to be 1, that would be:
 * constexpr nt::float_bitset<16, float16_t> bits(float16_t(0));
 * bits.set(4, true);
 * constexpr float16_t out = bits.lo_type();
 *
 * and the bits of out would be:
 *    0000010000000000 
 *
*/


#ifndef NT_BIT_FLOAT_BITSET_H__
#define NT_BIT_FLOAT_BITSET_H__

#include "../utils/always_inline_macro.h"
#include "../utils/type_traits.h"
#include "float_bits.h"
#include <utility>
#include <cstdint>
#include <cstddef>
#include <iostream>

namespace nt{

template<std::size_t N, class Type>
class float_bitset{
    static_assert(type_traits::is_floating_point_v<Type>, "Error bitset must use an integer type");
    using FloatingType = float_bits<Type>;
    static constexpr std::size_t BITS_PER_WORD = FloatingType::NUM_BITS;
    static constexpr std::size_t WORDS = (N + BITS_PER_WORD - 1) / BITS_PER_WORD;
    FloatingType data[WORDS]{};
public:

    constexpr float_bitset(){
        for(std::size_t i = 0; i < WORDS; ++i){
            data[i].zero();
        }
    }
    constexpr float_bitset(Type low_word)
    {
        if constexpr (WORDS > 0){
            data[0] = FloatingType(low_word);
        }
    }
    
    inline constexpr bool test(std::size_t pos) const {
        std::size_t word = pos / BITS_PER_WORD;
        std::size_t n_pos = pos - (word * BITS_PER_WORD);
        return (data[word][n_pos]);
    }
    
    inline constexpr bool operator[](std::size_t pos) const {
        return test(pos);
    }

    inline constexpr void set(std::size_t pos, bool value) {
        std::size_t word = pos / BITS_PER_WORD;
        std::size_t n_pos = pos - (word * BITS_PER_WORD);
        data[word].set(n_pos, value);
    }

    inline constexpr void reset(std::size_t pos) {
        set(pos, false);
    }

    inline constexpr Type lo_type() const {
        static_assert(WORDS >= 1, "bitset too small");
        return data[0].get();
    }
    
    inline constexpr Type hi_type() const {
        static_assert(WORDS >= 2, "bitset must be >= 128 bits");
        return data[1].get();
    }

    inline constexpr const Type* raw() const { return data; }

};


}

#endif
