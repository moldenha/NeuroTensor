#ifndef NT_TYPES_FLOAT128_PRINT_HPP_
#define NT_TYPES_FLOAT128_PRINT_HPP_
#include "float128_impl.h"
#include "to_string.hpp"
#include <iostream>

namespace nt{

inline std::ostream& operator<<(std::ostream& os, const float128_t& x){
    return os << double(x);
    // return os << nt::float128_func::to_string(x);
}

}

#endif
