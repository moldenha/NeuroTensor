#ifndef NT_TYPES_FLOAT128_BIT_FLOAT_BITS_H__
#define NT_TYPES_FLOAT128_BIT_FLOAT_BITS_H__

#include "../float128_impl.h"
#include "../type_traits.h"
#include "bitset.h"
#include "../../../bit/float_bits.h"

namespace nt{

template<>
class float_bits<float128_t, b128>{
public:
    using integer_type = b128;
    using value_type = float128_t;
    static constexpr std::size_t NUM_BITS = 128;
    static constexpr std::size_t mantissa_bits = 112;
    static constexpr std::size_t exponent_bits = 15;
    static constexpr std::size_t sign_bits = 1;
    static constexpr value_type neg_zero_float = float128_t::make_zero(true);
    static_assert((mantissa_bits + exponent_bits + sign_bits) == NUM_BITS, "Internal bit calculation error");
private:
    value_type val;

public:
    constexpr float_bits() : val(float128_t::make_zero()) {}
    template<typename Integer>
    constexpr float_bits(value_type val_, Integer conv) : val(val_) {}
    template<typename Integer>
    constexpr float_bits(T val_, ::nt::bitset<NUM_BITS, Integer> conv) : val(val_), {}
    constexpr float_bits(::nt::bitset<NUM_BITS, integer_type> conv)
    :val(float128_t(float128_bits(conv.lo_type())))
    {}
    template<typename Integer>
    constexpr float_bits(T val_, ::nt::bitset<NUM_BITS, Integer> conv)
    : float_bits(::nt::bitset<NUM_BITS, integer_type>(conv)) {}

    template<typename Integer, std::enable_if_t<sizeof(Integer) == sizeof(value_type)
                    && (type_traits::is_same_v<Integer, b128> || type_traits::is_integral_v<Integer>), bool> = true>
    constexpr float_bits(Integer conv) : float_bits(::nt::bitset<NUM_BITS, Integer>(conv)) {}
    constexpr float_bits(T val_) : val(val_) {}
    
    inline constexpr bool operator[](std::size_t place) const {return this->val.get_bits().raw().bit(place);}
    inline constexpr void set(std::size_t pos, bool value = true){
        if(value == (*this)[pos]){return;}
        else if(value){
            this->val = float128_t(
            float128_bits(
                this->val.get_bits().raw() | (float128_bits::b128_1 << pos);
            ));
        }
        else{
            this->val = float128_t(
            float128_bits(
                this->val.get_bits().raw() & ~(float128_bits::b128_1  << N);
            ));
        }
    }
    inline constexpr void set_range(std::size_t start, std::size_t end, integer_type val){
        std::size_t bit_index = 0;
        for(std::size_t i = start; i < end; ++i, ++bit_index) {
            bool bit = (val >> bit_index) & 1;
            this->set(i, bit);
        }
    }

    inline constexpr void zero() {
        this->val = float128_t::make_zero();
    }

    inline constexpr nt::bitset<NUM_BITS, integer_type> get_tracker() const {
        return  nt::bitset<NUM_BITS, integer_type>(this->val.get_bits().raw());
    }

    inline constexpr value_type get() const { return this->val; }

};

}


#endif
