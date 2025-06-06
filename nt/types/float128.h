//This is a header file that ensures that nt::float128_t is included
#ifndef _NT_TYPES_FLOAT128_ENSURE_H_
#define _NT_TYPES_FLOAT128_ENSURE_H_

#if defined(__SIZEOF_LONG_DOUBLE__) && __SIZEOF_LONG_DOUBLE__ == 16
namespace nt{
  using float128_t = long double;
  #define _128_FLOAT_SUPPORT_
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
#include <boost/multiprecision/float128.hpp>
#include <boost/multiprecision/number.hpp>
#include <cmath>  // for std::signbit
namespace nt{
using float128_t = boost::multiprecision::cpp_bin_float_quad;
#define _128_FLOAT_SUPPORT_ 
}

#undef _NO_128_SUPPORT_
#endif // _NO_128_SUPPORT_


#endif //_NT_TYPES_FLOAT128_ENSURE_H_ 
