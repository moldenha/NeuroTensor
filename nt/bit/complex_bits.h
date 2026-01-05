/*
 * This contains the nt::complex_bits class 
 *  This is just the nt::float_bits class for complex type
 *  It just takes advantage of the nt::float_bitset and nt::float_bits to make this

 *
*/


#ifndef NT_BIT_COMPLEX_BITS_H__
#define NT_BIT_COMPLEX_BITS_H__

#include "../utils/always_inline_macro.h"
#include "../utils/type_traits.h"
#include "../types/complex.h"
#include "float_bits.h"
#include "float_bitset.h"
#include <utility>
#include <cstdint>
#include <cstddef>
#include <iostream>

namespace nt{

template<typename Type>
class complex_bits{
    static_assert(type_traits::is_complex_v<Type>, "Error, complex bits only works with nt::my_complex<T>");
    using floating_point = typename Type::value_type;
    using FloatingType = float_bits<floating_point>;
    static constexpr std::size_t BITS_PER_WORD = FloatingType::NUM_BITS;
    static constexpr std::size_t WORDS = 2;
    static constexpr std::size_t NUM_BITS = BITS_PER_WORD * 2;
    FloatingType data[WORDS]{};

public:

    constexpr complex_bits(){
        data[0].zero();
        data[1].zero();
    }

    constexpr complex_bits(Type complex_num){
        data[0] = FloatingType(complex_num.real());
        data[1] = FloatingType(complex_num.imag());
    }

    NT_ALWAYS_INLINE constexpr bool test(std::size_t pos) const {
        std::size_t word = pos / BITS_PER_WORD;
        std::size_t n_pos = pos - (word * BITS_PER_WORD);
        return (data[word][n_pos]);

    }

    NT_ALWAYS_INLINE constexpr bool operator[](std::size_t pos) const {
        return test(pos);
    }

    NT_ALWAYS_INLINE constexpr void set(std::size_t pos, bool value) {
        std::size_t word = pos / BITS_PER_WORD;
        std::size_t n_pos = pos - (word * BITS_PER_WORD);
        data[word].set(n_pos, value);
    }

    NT_ALWAYS_INLINE constexpr void reset(std::size_t pos) {
        set(pos, false);
    }
    
    NT_ALWAYS_INLINE constexpr Type get() const {
        return Type(data[0].get(), data[1].get());
    }

};


}

#endif
