#ifndef NT_TYPES_FLOAT128_TO_STRING_HPP_
#define NT_TYPES_FLOAT128_TO_STRING_HPP_
#include "float128_impl.h"
#include "b128.h"
#include "b256.h"
#include "kmath/trunc.hpp"
#include "kmath/abs.hpp"
#include "kmath/signbit.hpp"
#include <iostream>
#include <string>
#include <cmath>

namespace nt::float128_func{




inline std::string to_string(float128_t f){
    constexpr float128_t div_by_ten = float128_t::from_double(0.1);
    constexpr float128_t mul_by_ten(10);
    int64_t int_part__ = int64_t(f);
    std::string integer_part = std::to_string(int_part__);
    int_part__ = std::abs(int_part__);
    f = math::kmath::abs(f);
    f = f - int_part__;
    if(f == float128_t(0))
        return integer_part;
    integer_part += '.';
    int i = 0;
    while(i < 4967){
        f = f * mul_by_ten;
        int64_t decimal = int64_t(f);
        integer_part += std::to_string(decimal);
        f = f - decimal;
        ++i;
        if(f.get_bits().is_zero())
            return std::move(integer_part);
    }
    return std::move(integer_part);
}

// struct fp_parts {
//     bool sign;
//     int32_t exp2;   // power of two
//     b128 mant;      // integer mantissa
// };

// constexpr fp_parts unpack(float128_t x) {
//     float128_bits b = x.get_bits();

//     constexpr int EXP_BIAS = 16383;
//     constexpr int M_BITS   = 112;

//     fp_parts p{};
//     p.sign = b.sign();
//     p.mant = b.mantissa_with_hidden_bit();
//     p.exp2 = int32_t(b.exponent()) - EXP_BIAS - M_BITS;

//     return p;
// }

// struct decimal_parts {
//     b256 int_part;
//     b256 frac_part;
//     b256 frac_scale; // denominator
// };

// constexpr decimal_parts to_fixed(const fp_parts& p) {
//     decimal_parts d{};

//     if (p.exp2 >= 0) {
//         d.int_part = b256(p.mant) << p.exp2;
//         d.frac_part = b256(0);
//         d.frac_scale = b256(1);
//     } else {
//         int shift = -p.exp2;
//         d.int_part = b256(p.mant) >> shift;

//         b256 mask = (b256(1) << shift) - 1;
//         d.frac_part = b256(p.mant) & mask;
//         d.frac_scale = b256(1) << shift;
//     }
//     return d;
// }

// std::string to_decimal(b256 v) {
//     if (v.is_zero()) return "0";

//     std::string out;
//     while (!v.is_zero()) {
//         auto [q, r] = b256::divmod(v, b256(10));
//         out.push_back(char('0' + int(r.lo_bits.lo_bits)));
//         v = q;
//     }
//     std::reverse(out.begin(), out.end());
//     return out;
// }

// std::string frac_to_decimal(b256 frac, b256 scale, int max_digits = 40) {
//     std::string out;
//     for (int i = 0; i < max_digits && !frac.is_zero(); ++i) {
//         frac *= 10;
//         auto [digit, rem] = b256::divmod(frac, scale);
//         out.push_back(char('0' + int(digit.lo_bits.lo_bits)));
//         frac = rem;
//     }
//     return out;
// }

// std::string to_string(float128_t x, int max_frac_digits = 40) {
//     float128_bits b = x.get_bits();

//     if (b.is_nan())  return "nan";
//     if (b.is_inf())  return b.sign() ? "-inf" : "inf";
//     if (b.is_zero()) return b.sign() ? "-0" : "0";
    
//     std::string s;
//     bool sign = b.sign();
//     if(sign) s.push_back('-');



//     fp_parts p = unpack(x);
//     decimal_parts d = to_fixed(p);


//     s += to_decimal(d.int_part);

//     if (!d.frac_part.is_zero()) {
//         s.push_back('.');
//         s += frac_to_decimal(d.frac_part, d.frac_scale, max_frac_digits);
//     }

//     return s;
// }

}


#endif
