/*
 * This contains the nt::float_bits class 
 * This class is meant to allow the user to specify the floating type, and constexpr manipulation of bits
 *  - MUST BE A floating type
 *  - If wanting a float bitset look at float_bitset.h
 * the way it works is for example:
 * I have an float16: 
 *  float16_t(0) == 0000000000000000
 *
 * and I want the 5th bit to be 1, that would be:
 * constexpr nt::float_bits<float16_t> bits(float16_t(0));
 * bits.set(4, true);
 * constexpr float16_t out = bits.get();
 *
 * and the bits of out would be:
 *    0000010000000000 
 *
*/


#ifndef NT_BIT_FLOAT_BITS_H__
#define NT_BIT_FLOAT_BITS_H__

#include "../utils/always_inline_macro.h"
#include "../utils/type_traits.h"
#ifndef NT_KMATH_NO_INCLUDE_F128_
#define NT_KMATH_NO_INCLUDE_F128_
#include "../math/kmath/ldexp.hpp"
#include "../math/kmath/abs.hpp"
#include "../math/kmath/frexp.hpp"
#include "../math/kmath/copysign.hpp"
#undef NT_KMATH_NO_INCLUDE_F128_
#else
#include "../math/kmath/ldexp.hpp"
#include "../math/kmath/abs.hpp"
#include "../math/kmath/frexp.hpp"
#include "../math/kmath/copysign.hpp"
#endif

#include "bitset.h"
#include <utility>
#include <cstdint>
#include <cstddef>
#include <iostream>

namespace nt{

// template<typename Real>
// inline static constexpr Real clear_mantissa_under2(Real x){

//     using integer_type = type_traits::make_unsigned_t<Real>;
//     using signed_type = type_traits::make_signed_t<integer_type>;
//     static constexpr std::size_t NUM_BITS = sizeof(Real) * CHAR_BIT;
//     static constexpr std::size_t mantissa_bits = type_traits::numeric_num_digits_v<Real> - 1;
//     static constexpr std::size_t exponent_bits = NUM_BITS - (type_traits::numeric_num_digits_v<Real>);
//     static constexpr std::size_t sign_bits = 1;
//     static constexpr value_type neg_zero_float = value_type(-0.0);
//     static constexpr signed_type max_exponent_code = signed_type(1) << (exponent_bits-1); // (2^(exponent_bits-1))
//     if(x == 0) return x;

//     Real ax = ::nt::math::kmath::abs(x);
//     Real s  = ::nt::math::kmath::copysign(Real(1.0), x);


//     // Find exponent e such that ax = m * 2^e with m in [1,2)
//     signed_type e = 0;
    
//     // Scale ax into [1,2)
//     while (ax < Real(1.0)) {
//         ax *= Real(2.0);
//         e -= 1;
//     }
//     while (ax >= Real(2.0)) {
//         ax /= Real(2.0);
//         e += 1;
//     }
    
//     // Now ax is in [1,2); clearing mantissa -> exactly 1.0
//     // So result is s * 2^e
//     return s * ::nt::math::kmath::ldexp(Real(1.0), e);


// }

// template<typename Real>
// inline static constexpr Real clear_mantissa(Real x)
// {
//     using integer_type = type_traits::make_unsigned_t<Real>;
//     using signed_type = type_traits::make_signed_t<integer_type>;
//     static constexpr std::size_t NUM_BITS = sizeof(Real) * CHAR_BIT;
//     static constexpr std::size_t mantissa_bits = type_traits::numeric_num_digits_v<Real> - 1;
//     static constexpr std::size_t exponent_bits = NUM_BITS - (type_traits::numeric_num_digits_v<Real>);
//     static constexpr std::size_t sign_bits = 1;
//     static constexpr value_type neg_zero_float = value_type(-0.0);
//     static constexpr signed_type max_exponent_code = signed_type(1) << (exponent_bits-1); // (2^(exponent_bits-1))

//     if (x == 0) return x;

//     if (::nt::math::kmath::abs(x) < ::nt::math::kmath::ldexp(Real(1), 1 - max_exponent_code)) {
//         return 0; // truly no exponent bits (subnormal)
//     }
    
//     if(::nt::math::kmath::abs(x) == 2.0) return 2.0;
//     if(::nt::math::kmath::abs(x) < 2.0){
//         return clear_mantissa_under2(x);
//     }

//     int e = 0;
//     Real m = ::nt::math::kmath::frexp(x, &e);   // x = m * 2^e


//     // Convert to IEEE normalized exponent:   exponent = e - 1
//     int E = e - 1;
    

//     Real sign = ::nt::math::kmath::copysign(Real(1), x);
//     return ::nt::math::kmath::ldexp(sign, E);  // produces ±2^E
// }


template<typename T, typename Int = type_traits::make_unsigned_t<T>>
class float_bits{
public:
    using integer_type = Int;
    using signed_type = type_traits::make_signed_t<Int>;
private:
    static_assert(type_traits::is_integral_v<Int>, "Error, Int type needs to be an integer");
    static_assert(type_traits::is_unsigned_v<Int>, "Error, Int type needs to be unsigned for float bits");
    static_assert(type_traits::is_floating_point_v<T>, "Error, value_type needs to be a floating point for float_bits");
    inline static constexpr signed_type integer_ldexp(signed_type arg, signed_type exp){
        constexpr signed_type two = 2;
        while(exp > 0)
        {
            arg *= two;
            --exp;
        }
        return arg;
    }
public:
    using value_type = T;
    static constexpr std::size_t NUM_BITS = sizeof(T) * CHAR_BIT;
    static constexpr std::size_t mantissa_bits = type_traits::numeric_num_digits_v<T> - 1;
    static constexpr std::size_t exponent_bits = NUM_BITS - (type_traits::numeric_num_digits_v<T>);
    static constexpr std::size_t sign_bits = 1;
    static constexpr value_type neg_zero_float = value_type(-0.0);
    static_assert((mantissa_bits + exponent_bits + sign_bits) == NUM_BITS, "Internal bit calculation error");

    static constexpr signed_type max_exponent_code = signed_type(1) << (exponent_bits-1); // (2^(exponent_bits-1))
    static constexpr signed_type exponent_negative = -max_exponent_code + 2;
    static constexpr signed_type exponent_bias = max_exponent_code - 1;// (2^(exponent_bits-1)) - 1
    
    static constexpr value_type first_exponent_bit = ::nt::math::kmath::ldexp(T(1.0), 1);
    static constexpr value_type last_exponent_bit = ::nt::math::kmath::ldexp(T(2.0), exponent_negative);
    static constexpr value_type first_mantissa_bit = last_exponent_bit / 2.0;
    static constexpr value_type last_mantissa_bit = ::nt::math::kmath::ldexp(first_mantissa_bit, -(mantissa_bits));
    
    struct exponent_table_builder {
        T table[exponent_bits];

        constexpr exponent_table_builder() : table{} {
            table[0] = first_exponent_bit;
            table[exponent_bits-1] = last_exponent_bit;
            for (int i = 1; i < (exponent_bits-1); ++i) {
                int exponent_code = 1 << (exponent_bits - (i+1));
                int exponent_convert = exponent_negative + exponent_code - 1;
                table[i] = ::nt::math::kmath::ldexp(T(1.0), exponent_convert);
            }
        }
    };
    
    static constexpr exponent_table_builder exponent_bit_table{};

    static constexpr T exponent_bit_at(int place){
        return exponent_bit_table.table[place];
    }

    struct mantissa_table_builder {
        T table[mantissa_bits];

        constexpr mantissa_table_builder() : table{} {
            table[0] = first_mantissa_bit;
            table[mantissa_bits-1] = last_mantissa_bit;
            for (int i = 1; i < (mantissa_bits-1); ++i) {
                table[i] = ::nt::math::kmath::ldexp(first_mantissa_bit, -(i+1));
            }
        }
    };
    
    static constexpr mantissa_table_builder mantissa_bit_table{};

    static constexpr T mantissa_bit_at(int place){
        return mantissa_bit_table.table[place]; 
    }


// private:
    
    value_type val;
    ::nt::bitset<NUM_BITS, Int> bit_tracker;
    
    inline static constexpr int num_mantissa_bits(::nt::bitset<NUM_BITS, Int> tracker) {
        int cntr = 0;
        for (int i = 0; i < mantissa_bits; ++i) {
            if(tracker[i])
                ++cntr;
        }
        return cntr;
    }
    inline static constexpr int num_exponent_bits(::nt::bitset<NUM_BITS, Int> tracker) {
        int cntr = 0;
        // exponent bits come right after the sign bit at the MSB side
        int start = mantissa_bits;
        int end = mantissa_bits + exponent_bits;
        for (int i = start; i < end; ++i) {
            if (tracker[i])
                ++cntr;
        }
        return cntr;    
    }

    inline static constexpr signed_type get_exponent_value_under2(value_type x){
        if(x == 0) return 0;

        value_type ax = ::nt::math::kmath::abs(x);
        // value_type s  = ::nt::math::kmath::copysign(value_type(1.0), x);


        // Find exponent e such that ax = m * 2^e with m in [1,2)
        signed_type e = 0;
        
        // Scale ax into [1,2)
        while (ax < value_type(1.0)) {
            ax *= value_type(2.0);
            e -= 1;
        }
        while (ax >= value_type(2.0)) {
            ax /= value_type(2.0);
            e += 1;
        }
        
        return e;
    }
    
    inline static constexpr signed_type get_exponent_value(value_type x){
        if(x == 0)
            return 0;
        if (::nt::math::kmath::abs(x) < ::nt::math::kmath::ldexp(value_type(1), 1 - max_exponent_code)){
            return 0;
        }

        if(::nt::math::kmath::abs(x) == 2.0) return 1;
        if(::nt::math::kmath::abs(x) < 2.0){
            return get_exponent_value_under2(x);
        }

        int e = 0;
        T m = ::nt::math::kmath::frexp(x, &e);   // x = m * 2^e


        // Convert to IEEE normalized exponent:   exponent = e - 1
        signed_type E = e - 1;
        return E;
    }

    template<typename Real>
    inline static constexpr Real clear_mantissa_under2(Real x){
        if(x == 0) return x;

        Real ax = ::nt::math::kmath::abs(x);
        Real s  = ::nt::math::kmath::copysign(Real(1.0), x);


        // Find exponent e such that ax = m * 2^e with m in [1,2)
        signed_type e = 0;
        
        // Scale ax into [1,2)
        while (ax < value_type(1.0)) {
            ax *= value_type(2.0);
            e -= 1;
        }
        while (ax >= value_type(2.0)) {
            ax /= value_type(2.0);
            e += 1;
        }
        
        // Now ax is in [1,2); clearing mantissa -> exactly 1.0
        // So result is s * 2^e
        return s * ::nt::math::kmath::ldexp(value_type(1.0), e);


    }

    template<typename Real>
    inline static constexpr Real clear_mantissa(Real x)
    {
        if (x == 0) return x;

        if (::nt::math::kmath::abs(x) < ::nt::math::kmath::ldexp(Real(1), 1 - max_exponent_code)) {
            return 0; // truly no exponent bits (subnormal)
        }
        
        if(::nt::math::kmath::abs(x) == 2.0) return 2.0;
        if(::nt::math::kmath::abs(x) < 2.0){
            return clear_mantissa_under2(x);
        }

        int e = 0;
        Real m = ::nt::math::kmath::frexp(x, &e);   // x = m * 2^e


        // Convert to IEEE normalized exponent:   exponent = e - 1
        int E = e - 1;
        

        Real sign = ::nt::math::kmath::copysign(Real(1), x);
        return ::nt::math::kmath::ldexp(sign, E);  // produces ±2^E
    }

    inline static constexpr value_type get_mantissa_value(value_type x){
        value_type exponent = clear_mantissa(x); // if exponent is E, this is sign * 2^E
        // to get the integer exponent it would be log_2(exponent)
        // the above also preserves the sign
        // so to get the mantissa bits, its just:
        // Resulting float = sign * 2^(E) * mantissa
        // -> mantissa = (Resulting float) / (sign * 2^E);
        if(exponent == 0){
            // true exponent for subnormals: E = 1 - bias
            // constexpr int32_t bias = type_traits::numeric_limits<value_type>::max_exponent - 1;
            int32_t E = 1 - exponent_bias;

            // mantissa = x / 2^E, no “-1” because there is NO implicit leading 1
            return ::nt::math::kmath::abs( x / ::nt::math::kmath::ldexp(value_type(1), E) );
        }
        return ::nt::math::kmath::abs(x / exponent) - 1.0;
    }
    
    inline static constexpr void get_mantissa_bits(value_type f, nt::bitset<NUM_BITS, Int>& out){
        value_type value = get_mantissa_value(f);
        if(value == 0){ return; }
        // the first mantissa bit is 9 (on float32 IEE_757)
        size_t current = (exponent_bits + sign_bits); // for example on float32 IEE_757 (1 + e)
        
        value_type start = 0.5;
        while(current < NUM_BITS){
            if(start < value){
                out.set(current, true);
                value -= start;
            }else if(start == value){
                out.set(current, true);
                return;
            }
            start /= 2.0;
            ++current;
        }
    }

    inline static constexpr nt::bitset<NUM_BITS, Int> get_mantissa_bits(value_type f){
        nt::bitset<NUM_BITS, Int> out(0);
        value_type value = get_mantissa_value(f);
        if(value == 0){ return out; }
        // the first mantissa bit is 9 (on float32 IEE_757)
        size_t current = (exponent_bits + sign_bits); // for example on float32 IEE_757 (1 + e)
        
        value_type start = 0.5;
        while(current < NUM_BITS){
            if(start < value){
                out.set(current, true);
                value -= start;
            }else if(start == value){
                out.set(current, true);
                return out;
            }
            start /= 2.0;
            ++current;
        }
        return out;
    }

    inline static constexpr nt::bitset<NUM_BITS, Int> get_exponent_bits(value_type f){
        signed_type E_value = get_exponent_value(f);
        if(E_value == 0){
            // set all bits except the first 2:
            // bitset<NUM_BITS, Int> out(0);
            // for(int i = 2; i < (exponent_bits + 1); ++i){
            //     out.set(i, true);
            // }
            // return out;
            return nt::bitset<NUM_BITS, Int>(exponent_bias << mantissa_bits);
        }
        signed_type biased = E_value + exponent_bias;
        return nt::bitset<NUM_BITS, Int>((Int)biased) << mantissa_bits;
    }
    
    inline static constexpr nt::bitset<NUM_BITS, Int> get_all_bits(value_type f){
        nt::bitset<NUM_BITS, Int> out_bits = float_bits::get_exponent_bits(f);
        // if(value_type(f) == neg_zero_float){
        //     out_bits.set(0, true);
        // }
        if(f < 0){
            out_bits.set(0, true);
        }
        float_bits::get_mantissa_bits(f, out_bits);
        return out_bits;
    }

// public:
    constexpr float_bits() : val(0) {}
    template<typename Integer>
    constexpr float_bits(T val_, Integer conv) : val(val_), bit_tracker(conv) { static_assert(sizeof(Integer) == sizeof(T));}
    constexpr float_bits(T val_, nt::bitset<NUM_BITS, Int> conv) : val(val_), bit_tracker(conv) {}
    template<typename Integer>
    constexpr float_bits(nt::bitset<NUM_BITS, Integer> conv)
    :val(0)
    {
        for(int i = 0; i < NUM_BITS; ++i){
            if(conv[i]){this->set(i, true);}
        }
    }
    template<typename Integer, std::enable_if_t<sizeof(Integer) == sizeof(T) && type_traits::is_integral_v<Integer> , bool> = true>
    constexpr float_bits(Integer conv) : float_bits(nt::bitset<NUM_BITS, Integer>(conv)) {}
    constexpr float_bits(T val_) : val(val_), bit_tracker(float_bits::get_all_bits(val_)) {}

    inline constexpr int count_mantissa_bits() const { return num_mantissa_bits(this->bit_tracker); }
    inline constexpr int count_exponent_bits() const { return num_exponent_bits(this->bit_tracker); }
    // inline static constexpr T exponent_bit_at(int place){
    //     signed_type exponent_code = integer_ldexp(1, exponent_bits-(place+1));
    //     signed_type exponent_convert = exponent_negative + exponent_code - 1;
    //     return ::nt::math::kmath::ldexp(T(1.0), exponent_convert);
    // }

    // static constexpr T mantissa_bit_at(int place){
    //     return ::nt::math::kmath::ldexp(first_mantissa_bit, -(place+1));
    // }

    inline constexpr bool operator[](std::size_t place) const {return this->bit_tracker[place];}
    inline constexpr void set(std::size_t pos, bool value = true){
        if(value == (*this)[pos]){return;}
        this->bit_tracker.set(pos, value);
        if(pos == 0){
            return;
        }
        if(pos > 0 && pos < (exponent_bits+1)){
            if (value){
                val += exponent_bit_at(pos-1);
            }else{
                val -= exponent_bit_at(pos-1);
            }
            return;
        }
        if (value){
            val += mantissa_bit_at(pos - (1 + exponent_bits));
        }else{
            val -= mantissa_bit_at(pos - (1 + exponent_bits));
        }
        return;
    }
    
    inline constexpr void set_range(std::size_t start, std::size_t end, integer_type val){
        std::size_t bit_index = 0;
        for(std::size_t i = start; i < end; ++i, ++bit_index) {
            bool bit = (val >> bit_index) & 1;
            this->set(i, bit);
        }
    }

    inline constexpr void zero() {
        this->val = 0;
        this->bit_tracker = nt::bitset<NUM_BITS, Int>(0);
    }
    inline constexpr nt::bitset<NUM_BITS, Int> get_tracker() const {return bit_tracker;}

    inline constexpr T get() const {
        if(this->bit_tracker[0]){
            if(val == 0){
                return neg_zero_float;
            }else{
                return -val;
            }
        }
        return ::nt::math::kmath::abs(val);
    }


};

}

#endif
