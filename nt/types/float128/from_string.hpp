#ifndef NT_TYPES_FLOAT128_FROM_STRING_HPP_
#define NT_TYPES_FLOAT128_FROM_STRING_HPP_
#include "float128_impl.h"
#include "b128.h"
#include "b256.h"
#include "kmath/trunc.hpp"
#include "kmath/abs.hpp"
#include "kmath/signbit.hpp"
#include <iostream>
#include <string>
#include <string_view>
#include <cmath>

namespace nt::float128_func{

NT_ALWAYS_INLINE constexpr float128_t char_to_num(char val){
    constexpr float128_t a(0);
    constexpr float128_t b(1);
    constexpr float128_t c(2);
    constexpr float128_t d(3);
    constexpr float128_t e(4);
    constexpr float128_t f(5);
    constexpr float128_t g(6);
    constexpr float128_t h(7);
    constexpr float128_t i(8);
    constexpr float128_t j(9);

    switch(val){
        case '0': return a;
        case '1': return b;
        case '2': return c;
        case '3': return d;
        case '4': return e;
        case '5': return f;
        case '6': return g;
        case '7': return h;
        case '8': return i;
        case '9': return j;
        // case 10: return k;
        default: return float128_t::make_nan();
    }
}

inline constexpr int64_t parseInt(std::string_view str, int& success) {
    
    // Skip leading whitespace (simple version)
    size_t index = 0;
    while (index < str.size() && (str[index] == ' ' || str[index] == '\t')) {
        ++index;
    }

    if (index == str.size()) return 0; // Handle empty or only whitespace string

    int64_t sign = 1;
    if (str[index] == '-' || str[index] == '+') {
        sign = (str[index] == '-') ? -1 : 1;
        ++index;
    }

    int64_t val = 0;
    for (; index < str.size(); ++index) {
        char ch = str[index];
        if (ch >= '0' && ch <= '9') {
            // Basic check to prevent simple overflow in the intermediate step.
            // For production code, consider more robust overflow checks.
            // if (val > (INT_MAX / 10) || (val == (INT_MAX / 10) && (ch - '0') > (INT_MAX % 10))) {
                // In a constexpr context in C++17, you cannot throw an exception.
                // You might need a way to signal error if necessary (e.g. static_assert, or return a special value).
                // For simplicity here, we assume valid input.
            // }
            val = val * 10 + (ch - '0');
        } else {
            success = -1;
            break; // Stop at non-digit characters
        }
    }

    return sign * val;
}

inline constexpr float128_t from_string(std::string_view f){
    if(f == "NaN" || f == "nan"){
        return float128_t::make_nan();
    }
    if(f == "inf"){
        return float128_t::make_inf(false);
    }
    if(f == "-inf"){
        return float128_t::make_inf(true);
    }
    if(f == "0"){
        return float128_t::make_zero();
    }
    if(f == "-0"){
        return float128_t::make_zero(true);
    }
    std::size_t decimal = f.find('.');
    if(decimal == std::string_view::npos){
        int suc = 0;
        int64_t integer = parseInt(f, suc);
        if(suc == -1)
            return float128_t::make_nan();
        return float128_t(integer);
    }
    int suc__ = 0;
    int64_t integer__ = parseInt(f.substr(0, decimal), suc__);
    if(suc__ == -1)
        return float128_t::make_nan();
    float128_t start(integer__); 
    std::string_view decimal_str = f.substr(decimal+1);
    float128_t mul(1);
    while(decimal_str.size() > 0){
        mul = mul / 10;
        float128_t integer = char_to_num(decimal_str[0]);
        decimal_str = decimal_str.substr(1);
        start = start + (integer * mul);
    }
    return start;
}

}

constexpr nt::float128_t operator "" _f128(const char* s) {
    return nt::float128_func::from_string(std::string_view(s));
}

constexpr nt::float128_t operator "" _f128(const char* __str, std::size_t __len) {
    return nt::float128_func::from_string(std::basic_string_view{__str, __len});
}

#endif
