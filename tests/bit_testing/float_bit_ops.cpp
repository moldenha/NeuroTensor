#include <iostream>
#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <cmath>
#include <bitset>
#include "../../nt/bit/bit.h"
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time() as a seed
#include <random>
#include <limits>
#include <iomanip> // Required for setprecision

// constexpr math to mimic things like a bit shift
namespace ccmath{

template <typename Type>
inline constexpr Type ldexp(Type arg, int exp) noexcept
{
    constexpr Type two = 2;
    while(exp > 0)
    {
        arg *= two;
        --exp;
    }
    while(exp < 0)
    {
        arg /= two;
        ++exp;
    }

    return arg;
}


// Bit shifts used to mimic:
// ldexp(x, +1) -> x << 1
// ldexp(x, -1) -> x >> 1

template <typename T>
inline constexpr T floor_pos_impl(T arg) noexcept
{
    constexpr auto max_comp_val = T(1) / ::nt::type_traits::numeric_limits<T>::epsilon();

    if (arg >= max_comp_val)
    {
        return arg;
    }

    T result = 1;

    if(result < arg)
    {
        while(result < arg)
        {
            result *= 2;
        }
        while(result > arg)
        {
            --result;
        }

        return result;
    }
    else
    {
        return T(0);
    }
}

template <typename T>
inline constexpr T floor_neg_impl(T arg) noexcept
{
    T result = -1;

    if(result > arg)
    {
        while(result > arg)
        {
            result *= 2;
        }
        while(result < arg)
        {
            ++result;
        }
        if(result != arg)
        {
            --result;
        }
    }

    return result;
}

template <typename T>
inline constexpr T floor(T arg) noexcept
{
    if(arg > 0)
    {
        return floor_pos_impl(arg);
    }
    else
    {
        return floor_neg_impl(arg);
    }
}


template <typename T>
inline constexpr T ceil(T arg) noexcept
{
    T result = floor(arg);

    if(result == arg)
    {
        return result;
    }
    else
    {
        return result + 1;
    }
}


template <typename T>
inline constexpr T trunc(T arg) noexcept
{
    return (arg > 0) ? floor(arg) : ceil(arg);
}

template<typename T>
inline constexpr T abs(T arg) noexcept {
    return (arg > 0) ? arg : -arg;
}


template <typename Real>
inline constexpr Real frexp_real_out(Real arg)
{
    const bool negative_arg = (arg < Real(0));
    
    Real f = negative_arg ? -arg : arg;
    int e2 = 0;
    constexpr Real two_pow_32 = Real(4294967296);

    while (f >= two_pow_32)
    {
        f = f / two_pow_32;
        e2 += 32;
    }

    while(f >= Real(1))
    {
        f = f / Real(2);
        ++e2;
    }
    
    // if(exp != nullptr)
    // {
    //     *exp = e2;
    // }

    return !negative_arg ? f : -f;
}

template <typename Real>
inline constexpr Real frexp_exp_out(Real arg)
{
    const bool negative_arg = (arg < Real(0));
    
    Real f = negative_arg ? -arg : arg;
    int e2 = 0;
    constexpr Real two_pow_32 = Real(4294967296);

    while (f >= two_pow_32)
    {
        f = f / two_pow_32;
        e2 += 32;
    }

    while(f >= Real(1))
    {
        f = f / Real(2);
        ++e2;
    }
    
    return e2;
}

template<typename Real>
inline constexpr Real frexp(Real arg, int* exp = nullptr){
    const bool negative_arg = (arg < Real(0));
    
    Real f = negative_arg ? -arg : arg;
    int e2 = 0;
    constexpr Real two_pow_32 = Real(4294967296);

    while (f >= two_pow_32)
    {
        f = f / two_pow_32;
        e2 += 32;
    }

    while(f >= Real(1))
    {
        f = f / Real(2);
        ++e2;
    }
    
    if(exp != nullptr)
    {
        *exp = e2;
    }

    return !negative_arg ? f : -f;

}


template<class T>
inline constexpr T ulp_at(T x) {
    // if (!(std::isfinite(x))) return T(0);              // NaN/Inf -> ill-defined
    x = abs(x);
    if (x == T(0)) return ::nt::type_traits::numeric_limits<T>::denorm_min();

    // b = total significand bits (including implicit leading 1 for normal numbers)
    int mantissa_bits = ::nt::type_traits::numeric_limits<T>::digits - 1; // explicit stored bits
    int b = mantissa_bits + 1;

    int e = frexp_exp_out(x);           // frexp returns m in [0.5,1) and exponent in e such that x = m * 2^e
    // e is e', so E = e-1
    // ulp = 2^( (e-1) - (b-1) ) = 2^( e - b )
    return ldexp(T(1), e - b - 8);
}

template<typename Real>
constexpr bool signbit(const Real val) noexcept {
    if(val == Real(-0.0)){
        return true;
    }else if(val == Real(0.0)){
        return false;
    }else{
        return val < Real(0.0);
    }
}

template <typename Real>
constexpr Real copysign(const Real mag, const Real sgn) noexcept
{
    if(signbit(sgn)){
        return -abs(mag); 
    }else{
        return abs(mag);
    }
}


}




namespace nt{


/*

Resulting float = sign * 2^(exponent) * mantissa

converting back to float; (sign already pre-known)

log(2)(f (known) / mantissa) = exponent


[note without mantissa, the hidden bit is 0, so it would come out to log_2(f (known)) = exponent]


 */

// template<typename Real>
// constexpr bool has_exponent_bits(Real x){
//     static constexpr size_t NUM_BITS = sizeof(Real) * CHAR_BIT;
//     static constexpr size_t exponent_bits = NUM_BITS - (::nt::type_traits::numeric_limits<Real>::digits);
//     // bias = 2^(k−1) − 1 (k = number of exponent bits)
//     float bias = ccmath::ldexp(1.0, 
    
// }

template<typename Real>
constexpr Real clear_mantissa_under2(Real x){
    if(x == 0) return x;

    Real ax = ccmath::abs(x);
    Real s  = ccmath::copysign(Real(1.0), x);


    // Find exponent e such that ax = m * 2^e with m in [1,2)
    int e = 0;
    
    // Scale ax into [1,2)
    while (ax < 1.0f) {
        ax *= 2.0f;
        e -= 1;
    }
    while (ax >= 2.0f) {
        ax /= 2.0f;
        e += 1;
    }
    
    // Now ax is in [1,2); clearing mantissa -> exactly 1.0
    // So result is s * 2^e
    return s * ccmath::ldexp(1.0f, e);


}

template<typename Real>
constexpr Real clear_mantissa(Real x)
{
    if (x == 0) return x;

    if (ccmath::abs(x) < ccmath::ldexp(Real(1), 1 - ::nt::type_traits::numeric_limits<Real>::max_exponent)) {
        return 0; // truly no exponent bits (subnormal)
    }
    
    if(ccmath::abs(x) == 2.0) return 2.0;
    if(ccmath::abs(x) < 2.0){
        return clear_mantissa_under2(x);
    }

    int e = 0;
    Real m = ccmath::frexp(x, &e);   // x = m * 2^e


    // Convert to IEEE normalized exponent:   exponent = e - 1
    int E = e - 1;
    

    Real sign = ccmath::copysign(Real(1), x);
    return ccmath::ldexp(sign, E);  // produces ±2^E
}

template<typename Real>
constexpr int get_exponent_value_under2(Real x){
    if(x == 0) return x;

    Real ax = ccmath::abs(x);
    // Real s  = ccmath::copysign(Real(1.0), x);


    // Find exponent e such that ax = m * 2^e with m in [1,2)
    int e = 0;
    
    // Scale ax into [1,2)
    while (ax < 1.0f) {
        ax *= 2.0f;
        e -= 1;
    }
    while (ax >= 2.0f) {
        ax /= 2.0f;
        e += 1;
    }
    
    return e;

}


template<typename Real>
constexpr int get_exponent_value(Real x){
    if(x == 0)
        return 0;
    if (ccmath::abs(x) < ccmath::ldexp(Real(1), 1 - ::nt::type_traits::numeric_limits<Real>::max_exponent)){
        return 0;
    }

    if(ccmath::abs(x) == 2.0) return 1;
    if(ccmath::abs(x) < 2.0){
        return get_exponent_value_under2(x);
    }

    int e = 0;
    Real m = ccmath::frexp(x, &e);   // x = m * 2^e


    // Convert to IEEE normalized exponent:   exponent = e - 1
    int E = e - 1;
    return E;
}


// this currently only works for values that have an exponent value
// but, this currently represents 1.f 
// where f = (b_1 * 2^-1 + b_2 * 2^-2 + ...)
template<typename Real>
constexpr Real get_mantissa_value(Real x){
    Real exponent = clear_mantissa(x); // if exponent is E, this is sign * 2^E
    // to get the integer exponent it would be log_2(exponent)
    // the above also preserves the sign
    // so to get the mantissa bits, its just:
    // Resulting float = sign * 2^(E) * mantissa
    // -> mantissa = (Resulting float) / (sign * 2^E);
    if(exponent == 0){
        // true exponent for subnormals: E = 1 - bias
        constexpr int bias = ::nt::type_traits::numeric_limits<Real>::max_exponent - 1;
        int E = 1 - bias;

        // mantissa = x / 2^E, no “-1” because there is NO implicit leading 1
        return ccmath::abs( x / ccmath::ldexp(Real(1), E) );
    }
    return ccmath::abs(x / exponent) - 1.0;
    
}

// not needed to extract the bits really at all
// Obviously, given all the functions created this could be made
//  but no reason to bloat the code base for a function that would never get used
// template<typename Real>
// constexpr Real clear_exponent(Real x){
//     return get_mantissa_value(x);
// }



// this is a test run
constexpr nt::bitset<32, uint32_t> get_mantissa_bits(float f){
    float value = get_mantissa_value(f);
    if(value == 0){
        return nt::bitset<32, uint32_t>(0);
    }
    // the first mantissa bit is 9
    nt::bitset<32, uint32_t> out(0);
    int current = 9;

    float start = 0.5;
    while(current < 32){
        if(start < value){
            out.set<true>(current);
            value -= start;
        }else if(start == value){
            out.set<true>(current);
            return out;
        }
        start /= 2.0;
        ++current;
    }
    return out;
}



constexpr nt::bitset<32, uint32_t> get_exponent_bits(float f){
    int E_value = get_exponent_value(f);
    if(E_value == 0){
        return nt::bitset<32, uint32_t>(0);
    }
    int biased = E_value + 127;
    return nt::bitset<32, uint32_t>((uint32_t)biased) << 23;
}

// for the floats used by neurotensor, 2.0 is currently the first bit of the exponent for all types
template<typename T, typename Int = type_traits::make_unsigned_t<T>>
class FloatShift{
    using signed_type = type_traits::make_signed_t<Int>;
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
    static constexpr size_t NUM_BITS = sizeof(T) * CHAR_BIT;
    static constexpr size_t mantissa_bits = ::nt::type_traits::numeric_num_digits_v<T> - 1;
    static constexpr size_t exponent_bits = NUM_BITS - (::nt::type_traits::numeric_num_digits_v<T>);
    static constexpr size_t sign_bits = 1;

    // the reason this is a floating point and not T:
    //          - 
    static constexpr signed_type max_exponent_code = integer_ldexp(signed_type(1), (exponent_bits-1)); // (2^(exponent_bits-1))
    static constexpr signed_type exponent_negative = -max_exponent_code + 2;
    static constexpr signed_type exponent_bias = max_exponent_code - 1;// (2^(exponent_bits-1)) - 1
    static constexpr T first_exponent_bit = ccmath::ldexp(T(1.0), 1);
    static constexpr T last_exponent_bit = ccmath::ldexp(T(2.0), exponent_negative);
    static constexpr T first_mantissa_bit = last_exponent_bit / 2.0;
    static constexpr T last_mantissa_bit = ccmath::ldexp(first_mantissa_bit, -(mantissa_bits));
    
    static constexpr auto build_exponent_table() noexcept {
        std::array<T, exponent_bits> table;
        table[0] = first_exponent_bit;
        table[exponent_bits-1] = last_exponent_bit;
        for(size_t i = 1; i < (exponent_bits-1); ++i){
            table[i] = ccmath::ldexp(T(1.0),     
                                     exponent_negative + (1 << (exponent_bits - bit - 1)) - 1);
        }

    }
    
    static constexpr auto exponent_bit_table = build_exponent_table();

    static constexpr T exponent_bit_at(int place){
        return exponent_bit_table[place];
    }
    // static constexpr T exponent_bit_at(int place){
    //     int exponent_code = ccmath::ldexp(1, exponent_bits-(place+1));
    //     int exponent_convert = FloatShift<T>::exponent_negative + exponent_code - 1;
    //     return ccmath::ldexp(T(1.0), exponent_convert);
    // }

private:
    T val;

    nt::bitset<NUM_BITS, Int> bit_tracker;
    static_assert((mantissa_bits + exponent_bits + sign_bits) == NUM_BITS);
    static_assert(std::is_floating_point_v<T>);
    
    constexpr bool sign_bit(nt::bitset<NUM_BITS, Int> tracker){
        return tracker[NUM_BITS-1];
    }

    constexpr T ensure_sign(T cur, bool negative) const {
        if(negative){
            if(cur < 0) return cur;
            return -cur;
        }
        if(cur < 0) return -cur;
        return cur;
    }

    constexpr int num_mantissa_bits(nt::bitset<NUM_BITS, Int> tracker) const {
        int cntr = 0;
        for (int i = 0; i < mantissa_bits; ++i) {
            if(tracker[i])
                ++cntr;
        }
        return cntr;
    }
    constexpr int num_exponent_bits(nt::bitset<NUM_BITS, Int> tracker) const {
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

    inline constexpr signed_type get_exponent_value_under2(value_type x){
        if(x == 0) return x;

        value_type ax = ccmath::abs(x);
        // value_type s  = ccmath::copysign(value_type(1.0), x);


        // Find exponent e such that ax = m * 2^e with m in [1,2)
        signed_type e = 0;
        
        // Scale ax into [1,2)
        while (ax < 1.0f) {
            ax *= 2.0f;
            e -= 1;
        }
        while (ax >= 2.0f) {
            ax /= 2.0f;
            e += 1;
        }
        
        return e;
    }


    inline constexpr signed_type get_exponent_value(value_type x){
        if(x == 0)
            return 0;
        if (ccmath::abs(x) < ccmath::ldexp(value_type(1), 1 - max_exponent_code)){
            return 0;
        }

        if(ccmath::abs(x) == 2.0) return 1;
        if(ccmath::abs(x) < 2.0){
            return get_exponent_value_under2(x);
        }

        signed_type e = 0;
        T m = ccmath::frexp(x, &e);   // x = m * 2^e


        // Convert to IEEE normalized exponent:   exponent = e - 1
        int32_t E = e - 1;
        return E;
    }
    
    constexpr value_type get_mantissa_value(value_type x){
        value_type exponent = clear_mantissa(x); // if exponent is E, this is sign * 2^E
        // to get the integer exponent it would be log_2(exponent)
        // the above also preserves the sign
        // so to get the mantissa bits, its just:
        // Resulting float = sign * 2^(E) * mantissa
        // -> mantissa = (Resulting float) / (sign * 2^E);
        if(exponent == 0){
            // true exponent for subnormals: E = 1 - bias
            // constexpr int32_t bias = ::nt::type_traits::numeric_limits<T>::max_exponent;
            signed_type E = 1 - exponent_bias;

            // mantissa = x / 2^E, no “-1” because there is NO implicit leading 1
            return ccmath::abs( x / ccmath::ldexp(value_type(1), E) );
        }
        return ccmath::abs(x / exponent) - 1.0;
    }
    

    constexpr void get_mantissa_bits(value_type f, nt::bitset<NUM_BITS, Int>& out){
        value_type value = get_mantissa_value(f);
        if(value == 0){ return; }
        // the first mantissa bit is 9
        int current = (exponent_bits + sign_bits); // for example on float32 IEE_757 (1 + e)
        
        value_type start = 0.5;
        while(current < NUM_BITS){
            if(start < value){
                out.template set<true>(current);
                value -= start;
            }else if(start == value){
                out.template set<true>(current);
                return;
            }
            start /= 2.0;
            ++current;
        }
    }

    constexpr nt::bitset<NUM_BITS, Int> get_exponent_bits(value_type f){
        signed_type E_value = get_exponent_value(f);
        if(E_value == 0){
            return bitset<NUM_BITS, Int>(0);
        }
        signed_type biased = E_value + exponent_bias;
        return nt::bitset<NUM_BITS, Int>((Int)biased) << mantissa_bits;
    }
    static constexpr float neg_zero_float = T(-0.0);
    
    constexpr nt::bitset<NUM_BITS, Int> get_all_bits(value_type f){
        nt::bitset<NUM_BITS, Int> out_bits = get_exponent_bits(f);
        if(value_type(f) == neg_zero_float){
            out_bits.template set<true>(0);
        }else if(f < 0){
            out_bits.template set<true>(0);
        }
        get_mantissa_bits(f, out_bits);
        return out_bits;
    }


public:
    constexpr FloatShift() : val(0) {}
    template<typename Integer>
    constexpr FloatShift(T val_, Integer conv) : val(val_), bit_tracker(conv) { static_assert(sizeof(Integer) == sizeof(T));}
    constexpr FloatShift(T val_, nt::bitset<NUM_BITS, Int> conv) : val(val_), bit_tracker(conv) {}
    template<typename Integer>
    constexpr FloatShift(nt::bitset<NUM_BITS, Integer> conv)
    :val(0)
    {
        for(int i = 0; i < NUM_BITS; ++i){
            if(conv[i]){this->set<true>(i);}
        }
    }
    constexpr FloatShift(T val_) : val(val_), bit_tracker(get_all_bits(val_)) {}



    constexpr int num_mantissa_bits() const { return num_mantissa_bits(this->bit_tracker); }
    constexpr int num_exponent_bits() const { return num_exponent_bits(this->bit_tracker); }
    
    constexpr bool operator[](std::size_t place) const {return this->bit_tracker[place];}
    


    static constexpr T mantissa_shift_up(T val, int shifts) {
        return ccmath::ldexp(val, shifts);
    }



    static constexpr T mantissa_bit_at(int place){
        return ccmath::ldexp(FloatShift<T>::first_mantissa_bit, -(place+1));
    }
    
    template<bool value>
    inline constexpr void set(std::size_t pos){
        if(value == (*this)[pos]){return;}
        this->bit_tracker.template set<value>(pos);
        if(pos == 0){
            return;
        }
        if(pos > 0 && pos < (exponent_bits+1)){
            if constexpr (value){
                val += exponent_bit_at(pos-1);
            }else{
                val -= exponent_bit_at(pos-1);
            }
            return;
        }
        if constexpr (value){
            val += mantissa_bit_at(pos - (1 + exponent_bits));
        }else{
            val -= mantissa_bit_at(pos - (1 + exponent_bits));
        }
        return;
    }

    // static constexpr T exponent_shift_up(T val, int shifts) {
    //     return val * ccmath::ldexp(1.0, shifts);
    // }

    constexpr FloatShift operator>>(int shifts) const {
        return FloatShift(ccmath::ldexp(val, -shifts), bit_tracker >> shifts);
    }
    constexpr FloatShift operator<<(int shifts) const {
        int cur_mantissa = num_mantissa_bits();
        int cur_exponent = num_exponent_bits();
        nt::bitset<NUM_BITS, uint32_t> n_tracker = bit_tracker << shifts;
        int n_mantissa = num_mantissa_bits(n_tracker);
        int n_exponent = num_exponent_bits(n_tracker);
        if(cur_mantissa == n_mantissa && cur_exponent == n_exponent)
            return FloatShift(ccmath::ldexp(val, shifts), bit_tracker << shifts);
        
        T exponent = ccmath::trunc(val);
        T mantissa = val - exponent;

        T exponent_shift = (exponent == 0) ? 0 : ccmath::ldexp(exponent, shifts);
        
        return FloatShift(ccmath::ldexp(val, shifts), bit_tracker << shifts);
    }

    constexpr nt::bitset<NUM_BITS, Int> get_tracker() const {return bit_tracker;}

    constexpr T get() const {
        if(this->bit_tracker[0]){
            if(val == 0){
                return neg_zero_float;
            }else{
                return -val;
            }
        }
        return val;
    }

    T uk_get() const {
        if(this->bit_tracker[0]){
            if(val == 0){
                std::cout << "val is 0" << std::endl;
                return neg_zero_float;
            }else{
                std::cout << "val is not 0" << std::endl;
                return -val;
            }
        }
        std::cout << "bit tracker shows no negatives" << std::endl;
        return val;
    }

};

}

template<typename T>
std::ostream& operator<<(std::ostream& os, const nt::FloatShift<T>& fs){
    // soon it is going to be {return os << fs.get_tracker();}
    if constexpr (nt::type_traits::is_same_v<T, float>){
        return os << nt::bitset<32, uint32_t>(nt::bit_cast<uint32_t>(fs.get()));
    }else if constexpr (nt::type_traits::is_same_v<T, double>){
        return os << nt::bitset<64, uint64_t>(nt::bit_cast<uint64_t>(fs.get()));
    }
}


void check_num(nt::FloatShift<float> f){
    float n = f.get();

    std::cout << n << ' ' << nt::bit_cast<uint32_t>(n) << ' ' << std::bitset<32>(nt::bit_cast<uint32_t>(n)) << std::endl;
    std::cout << f.get_tracker() << std::endl;
    std::cout << f.num_mantissa_bits() << ' ' << f.num_exponent_bits() << std::endl;
    std::bitset<32> printing(nt::bit_cast<uint32_t>(n));

}



constexpr float make_my_float_ex(int m){
    nt::FloatShift<float> f;
    for(int i = 0; i < m; ++i){
        f.set<true>(i * 3);
    }
    return f.get();
}

constexpr float ex_f = make_my_float_ex(4);

constexpr float make_my_float_ex2(int m){
    nt::FloatShift<float> f;
    std::cout << f.uk_get() << std::endl;
    for(int i = 0; i < m; ++i){
        f.set<true>(i * 6);
        std::cout << "set " << i*6 << std::endl;
        std::cout << f << std::endl;
        std::cout << f.uk_get() << std::endl;
        std::cout << f.get_tracker() << std::endl;
    }
    return f.get();
}


int check_integer_setting(int num){
    std::cout << "checking "<<num<<" integers to floats" << std::endl;
    uint32_t min = 0;
    uint32_t max = ::nt::type_traits::numeric_limits<uint32_t>::max();
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint32_t> distrib(0, ::nt::type_traits::numeric_limits<uint32_t>::max());


    for(int i = 0; i < num; ++i){
        uint32_t number = distrib(gen);
        nt::FloatShift<float> f((nt::bitset<32, uint32_t>(number)));
        float nf = f.get();
        uint32_t conv = nt::bit_cast<uint32_t>(number);
        if(number != conv){
            std::cout << "Error, "<<number<<" != "<<conv<<std::endl;
            return -1;
        }
    }
    std::cout << "All numbers worked" << std::endl;
    return 0;
}


int check_clear_mantissa_setting(int num){
    std::cout << "checking "<<num<<" floating clear mantissa" << std::endl;

    for(int i = 0; i < num; ++i){
        float number = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/(2.3e-22 - 3.8e-38));
        float cleared = nt::clear_mantissa(number);
        uint32_t conv = nt::bit_cast<uint32_t>(cleared);
        nt::bitset<32, uint32_t> bits(conv);
        // if(number < 1.0) std::cout << number << std::endl;
        // if(cleared < 1.0){
        //     std::cout << cleared << std::endl;
        // }
        if(cleared > number){
            std::cout << "Error, "<< cleared <<" from "<<number <<" is not cleared of mantissa (>)"<<std::endl;
            return -1;
        }
        for(int i = nt::FloatShift<float>::exponent_bits+1; i < 32; ++i){
            if(bits[i] != false){
                std::cout << "Error, "<< std::setprecision(20) << cleared <<" from "<<number <<" is not cleared of mantissa"<<std::endl;
                std::cout << nt::bitset<32, uint32_t>(nt::bit_cast<uint32_t>(cleared)) << std::endl;
                return -1;
            }
        }
    }

    for(int i = 0; i < num; ++i){
        float number = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/(2.35e+38 - 3.8e+22));
        float cleared = nt::clear_mantissa(number);
        uint32_t conv = nt::bit_cast<uint32_t>(cleared);
        nt::bitset<32, uint32_t> bits(conv);
        // if(number < 1.0) std::cout << number << std::endl;
        // if(cleared < 1.0){
        //     std::cout << cleared << std::endl;
        // }
        if(cleared > number){
            std::cout << "Error, "<< cleared <<" from "<<number <<" is not cleared of mantissa (>)"<<std::endl;
            return -1;
        }
        for(int i = nt::FloatShift<float>::exponent_bits+1; i < 32; ++i){
            if(bits[i] != false){
                std::cout << "Error, "<< std::setprecision(20) << cleared <<" from "<<number <<" is not cleared of mantissa"<<std::endl;
                std::cout << nt::bitset<32, uint32_t>(nt::bit_cast<uint32_t>(cleared)) << std::endl;
                return -1;
            }
        }
    }

    for(int i = 0; i < num; ++i){
        float number = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/(2.35e+38 - 3.8e-22));
        float cleared = nt::clear_mantissa(number);
        uint32_t conv = nt::bit_cast<uint32_t>(cleared);
        nt::bitset<32, uint32_t> bits(conv);
        // if(number < 1.0) std::cout << number << std::endl;
        // if(cleared < 1.0){
        //     std::cout << cleared << std::endl;
        // }
        if(cleared > number){
            std::cout << "Error, "<< cleared <<" from "<<number <<" is not cleared of mantissa (>)"<<std::endl;
            return -1;
        }
        for(int i = nt::FloatShift<float>::exponent_bits+1; i < 32; ++i){
            if(bits[i] != false){
                std::cout << "Error, "<< std::setprecision(20) << cleared <<" from "<<number <<" is not cleared of mantissa"<<std::endl;
                std::cout << nt::bitset<32, uint32_t>(nt::bit_cast<uint32_t>(cleared)) << std::endl;
                return -1;
            }
        }
    }
    std::cout << "All numbers worked" << std::endl;
    return 0;
}


int check_floats_to_bits(int num){
    std::cout << "checking "<<num<<" floating get_bits" << std::endl;

    for(int i = 0; i < num; ++i){
        float number = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/(2.3e-22 - 3.8e-38));
        nt::FloatShift<float, uint32_t> f(number);
        nt::bitset<32, uint32_t> bs(nt::bit_cast<uint32_t>(number));
        if(f.get_tracker().lo_type() != bs.lo_type()){
            std::cout << "number "<<number << " did not get properly bitcasted" << std::endl;
        }
    }


    for(int i = 0; i < num; ++i){
        float number = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/(2.35e+38 - 3.8e+22));
        nt::FloatShift<float, uint32_t> f(number);
        nt::bitset<32, uint32_t> bs(nt::bit_cast<uint32_t>(number));
        if(f.get_tracker().lo_type() != bs.lo_type()){
            std::cout << "number "<<number << " did not get properly bitcasted" << std::endl;
        }
    }

    for(int i = 0; i < num; ++i){
        float number = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/(2.35e+38 - 3.8e-22));
        nt::FloatShift<float, uint32_t> f(number);
        nt::bitset<32, uint32_t> bs(nt::bit_cast<uint32_t>(number));
        if(f.get_tracker().lo_type() != bs.lo_type()){
            std::cout << "number "<<number << " did not get properly bitcasted" << std::endl;
        }
    }
    std::cout << "All numbers worked" << std::endl;
    return 0;
}

int main(){
    constexpr float original = 1.4013e-45;
    
    constexpr nt::FloatShift f(original, uint32_t(1));
    /* this just tests to make sure constexpr compatibility */
    constexpr nt::FloatShift f2 = f << 3;
    constexpr float f3 = f2.get();
    std::cout << f3 << " is constexpr" << std::endl;
    /* End constexpr test */

    std::cout << nt::bit_cast<uint32_t>(nt::FloatShift<float>::last_mantissa_bit) << ' ' << nt::FloatShift<float>::last_mantissa_bit << std::endl;
    std::cout << nt::bit_cast<uint64_t>(nt::FloatShift<double>::last_mantissa_bit) << std::endl;

    std::cout << original << std::endl;
    for(int i = 0; i < 32; ++i){
        std::cout << "(i == " << i<<"): ";
        check_num(f << i);
    }
    // constexpr nt::FloatShift nf = f << 30;
    // constexpr nt::FloatShift nft = nf >> 1;
    // constexpr float n = nf.get();
    // constexpr float n_t = nft.get();
    // std::cout << original << ' ' << n << ' ' << nt::bit_cast<uint32_t>(n) << std::endl;
    // std::cout << original << ' ' << n_t << ' ' << nt::bit_cast<uint32_t>(n_t) << std::endl;
    // uint32_t v = 1;
    // std::cout << nt::bit_cast<float>(v) << std::endl;
    //
    std::cout << std::bitset<32>(nt::bit_cast<uint32_t>(nt::FloatShift<float>::first_exponent_bit)) << std::endl;
    std::cout << std::bitset<64>(nt::bit_cast<uint64_t>(nt::FloatShift<double>::first_exponent_bit)) << std::endl;
    std::cout << std::bitset<32>(nt::bit_cast<uint32_t>(nt::FloatShift<float>::last_exponent_bit)) << std::endl;
    std::cout << std::bitset<64>(nt::bit_cast<uint64_t>(nt::FloatShift<double>::last_exponent_bit)) << std::endl;
    std::cout << std::bitset<32>(nt::bit_cast<uint32_t>(nt::FloatShift<float>::first_mantissa_bit)) << std::endl;
    std::cout << std::bitset<64>(nt::bit_cast<uint64_t>(nt::FloatShift<double>::first_mantissa_bit)) << std::endl;
    std::cout << std::bitset<32>(nt::bit_cast<uint32_t>(nt::FloatShift<float>::last_mantissa_bit)) << std::endl;
    std::cout << std::bitset<64>(nt::bit_cast<uint64_t>(nt::FloatShift<double>::last_mantissa_bit)) << std::endl;
    

    std::cout << std::bitset<32>(nt::bit_cast<uint32_t>(nt::FloatShift<float>::mantissa_shift_up(nt::FloatShift<float>::last_mantissa_bit, 10))) << std::endl;
    std::cout << std::bitset<64>(nt::bit_cast<uint64_t>(nt::FloatShift<double>::mantissa_shift_up(nt::FloatShift<double>::last_mantissa_bit, 10))) << std::endl;
    // std::cout << std::bitset<32>(nt::bit_cast<uint32_t>(nt::FloatShift<float>::exponent_shift_up(nt::FloatShift<float>::last_exponent_bit, 3))) << std::endl;
    // std::cout << std::bitset<64>(nt::bit_cast<uint64_t>(nt::FloatShift<double>::exponent_shift_up(nt::FloatShift<double>::last_exponent_bit, 3))) << std::endl;
    
    std::cout << nt::FloatShift<float>::last_exponent_bit << ' ' <<  nt::FloatShift<float>::first_mantissa_bit << std::endl;
    std::cout << nt::FloatShift<double>::exponent_negative << ' ' << nt::FloatShift<float>::exponent_negative << std::endl; 
    std::cout << nt::FloatShift<double>::mantissa_bits << ' ' << nt::FloatShift<float>::mantissa_bits << std::endl; 
    std::cout << nt::FloatShift<double>::exponent_bits << ' ' << nt::FloatShift<float>::exponent_bits << std::endl; 
    for(int i = 0; i < 8; ++i){
        std::cout << std::bitset<32>(nt::bit_cast<uint32_t>(nt::FloatShift<float>::exponent_bit_at(i))) << ' ' << nt::FloatShift<float>::exponent_bit_at(i) << std::endl;
        // std::cout << ccmath::ldexp(1.0, 8 - (i+1)) << std::endl;
    }
    std::cout << "mantissa bits:" << std::endl;
    for(int i = 0; i < 23; ++i){
        std::cout << std::bitset<32>(nt::bit_cast<uint32_t>(nt::FloatShift<float>::mantissa_bit_at(i))) << ' ' << nt::FloatShift<float>::mantissa_bit_at(i) << std::endl;
        // std::cout << ccmath::ldexp(1.0, 8 - (i+1)) << std::endl;
    }
    std::cout << nt::FloatShift<float>::exponent_negative << std::endl; 
    std::cout << ex_f << ' ' << std::bitset<32>(nt::bit_cast<uint32_t>(make_my_float_ex2(3))) << std::endl;
    std::cout << ex_f << ' ' << std::bitset<32>(nt::bit_cast<uint32_t>(make_my_float_ex2(3))) << std::endl;
    nt::bitset<32, uint32_t> example(0);
    example.set<true>(0);
    std::cout << example << std::endl;
   
    constexpr float neg_zero_float = float(-0.0);
    std::cout << nt::bitset<32, uint32_t>(nt::bit_cast<uint32_t>(neg_zero_float)) << std::endl;
    constexpr float val = 5.75f;
    constexpr float exponent = nt::clear_mantissa(val);
    std::cout << val << ',' << exponent << std::endl;


    check_integer_setting(1000);
    check_clear_mantissa_setting(1000);
    int e = 0;
    float v2 = 5.75;
    float m = std::frexp(v2, &e);
    std::cout << v2 << ' ' << e << ' ' << m << std::endl;
    float sign = std::copysign(1.0f, m);
    std::cout << std::ldexp(sign, e) << std::endl;

    constexpr float random_float = 2.0056955e-37;
    constexpr float random_float_exponent = nt::clear_mantissa(random_float);
    std::cout << random_float << ',' << random_float_exponent << std::endl;
    std::cout << nt::get_mantissa_bits (random_float) << std::endl;
    std::cout << nt::bitset<32, uint32_t>(nt::bit_cast<uint32_t>(random_float)) << std::endl;
    std::cout << nt::get_mantissa_value(7.80654e-40f) << std::endl;
    std::cout << nt::bitset<32, uint32_t>(nt::bit_cast<uint32_t>(7.80654e-40f)) << std::endl;
    std::cout << nt::get_mantissa_bits(7.80654e-40f) << std::endl;
    std::cout << nt::get_exponent_value(random_float) << std::endl;
    std::cout << nt::get_exponent_value(7.80654e-40f) << std::endl;
    std::cout << nt::get_exponent_value(-23.451) << std::endl;
    std::cout << nt::get_exponent_bits(random_float) << std::endl;
    std::cout << nt::get_exponent_bits(7.80654e-40f) << std::endl;
    std::cout << nt::get_exponent_bits(-23.451) << std::endl;
    std::cout << nt::FloatShift<float, uint32_t>(random_float) << std::endl;
    std::cout << nt::bitset<32, uint32_t>(nt::bit_cast<uint32_t>(random_float)) << std::endl;
    std::cout << nt::FloatShift<float, uint32_t>(random_float).get() << std::endl;
    std::cout << nt::bitset<32, uint32_t>(nt::bit_cast<uint32_t>(nt::FloatShift<float, uint32_t>(random_float).get())) << std::endl;
    check_floats_to_bits(1000);
    std::cout << nt::bitset<16, uint16_t>(0x0400) << std::endl;
    std::cout << nt::FloatShift<float, uint32_t>::max_exponent_code << ' ' << nt::FloatShift<double, uint64_t>::max_exponent_code << std::endl;
    return 0;
}
