//This is a header file that ensures that nt::float128_t is included
#ifndef NT_TYPES_FLOAT128_ENSURE_H__
#define NT_TYPES_FLOAT128_ENSURE_H__


/* ********************************************************************************************************************************* */
/*       On certain systems that have long double, it is technically 128 bits, but it is really 80 bits and padded to 128 bits       */
/* This would not work with the sign bit being where it is, and certiain functions would definitely break, therefore it is taken out */
/* Unfortunately long doubles are faster, they just also don't work for my use case of needing a true 128 bit floating point (not 80)*/
/*                    Therefore, I developed a constexpr NeuroTensor float128 you can see under nt/types/float128                    */
/* ********************************************************************************************************************************* */

// this holds the float128_t class
#include "float128/float128_impl.h"
// #ifndef NT_FLOAT128_HEADER_ONLY__
// #define NT_FLOAT128_HEADER_ONLY__
// #endif
// this holds the ability to go from strings and do something like:
// constexpr float128_t f = 1.235867284617643_f128;
#include "float128/from_string.hpp"
// this holds the ability to go to a string:
// std::string str = nt::float128_func::to_string(f);
#include "float128/to_string.hpp"
// this holds the ability to print a float128_t:
// std::cout << f << std::endl;
#include "float128/print.hpp"
// this holds numeric limits, make_unsigned, is_floating_point
#include "float128/type_traits.h"


#endif //_NT_TYPES_FLOAT128_ENSURE_H_ 
