//This is a header file that ensures that nt::float128_t is included
#ifndef NT_TYPES_FLOAT128_ENSURE_H__
#define NT_TYPES_FLOAT128_ENSURE_H__

#include "../utils/type_traits.h"

#if defined(__SIZEOF_LONG_DOUBLE__) && __SIZEOF_LONG_DOUBLE__ == 16
namespace nt{
  using float128_t = long double;
    static_assert(type_traits::is_floating_point_v<float128_t>,
                  "Error, type traits floating point for long double not implemented");
  #define _128_FLOAT_SUPPORT_
  #define NT_128_FLOAT_LONG_DOUBLE_TYPE_DEFINED__
}

#elif defined(__GNUC__) && !defined(__APPLE__) && defined(__SIZEOF_FLOAT128__)
  // GCC on Linux usually supports __float128
namespace nt{
  using float128_t = __float128;
  std::ostream& operator<<(std::ostream& os, const float128_t& val);
  #define _128_FLOAT_SUPPORT_
}

#elif defined(__GNUC__) && defined(__FP_FAST_F128)
  // fallback for __fp128 on some platforms (rare)
namespace nt{
  using float128_t = __fp128;
  std::ostream& operator<<(std::ostream& os, const float128_t& val);
  #define _128_FLOAT_SUPPORT_
}

#else
  #define _NO_128_SUPPORT_
#endif

#ifdef _NO_128_SUPPORT_
#ifndef BOOST_MP_STANDALONE
    #define BOOST_MP_STANDALONE
#endif // BOOST_MP_STANDALONE 
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/number.hpp>
#include <cmath>  // for std::signbit
namespace nt{
using float128_t = boost::multiprecision::cpp_bin_float_quad;
#define _128_FLOAT_SUPPORT_ 
}

#undef _NO_128_SUPPORT_
#endif // _NO_128_SUPPORT_


#ifndef NT_128_FLOAT_LONG_DOUBLE_TYPE_DEFINED__ 
namespace nt::type_traits{

template<>
struct is_floating_point<float128_t> : true_type {};
template<>
struct is_floating_point<const float128_t> : true_type {};
template<>
struct is_floating_point<const volatile float128_t> : true_type {};
template<>
struct is_floating_point<volatile float128_t> : true_type {};

}

#else
#undef NT_128_FLOAT_LONG_DOUBLE_TYPE_DEFINED__
#endif


#endif //_NT_TYPES_FLOAT128_ENSURE_H_ 
