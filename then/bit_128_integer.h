// This is a header file that ensures that nt::uint128_t is included
// Currently, the suport of nt::int128_t is platform-dependent
#ifndef _NT_TYPES_BIT_128_INTEGER_ENSURE_H_
#define _NT_TYPES_BIT_128_INTEGER_ENSURE_H_

#ifdef __SIZEOF_INT128__
namespace nt{
using uint128_t = __uint128_t;
using int128_t = __int128_t;
std::ostream& operator<<(std::ostream& os, const __int128_t i);
std::ostream& operator<<(std::ostream& os, const __uint128_t i);
}

#else

//currently has library for uint128 support that is cross platform
//will be adding int128 support that is cross platform soon
#include <uint128_t.h>
namespace nt{
using uint128_t = uint128_t;
}

#ifndef BOOST_MP_STANDALONE
    #define BOOST_MP_STANDALONE
#endif // BOOST_MP_STANDALONE 
#include <boost/multiprecision/cpp_bin_float.hpp>
namespace nt{
using int128_t = boost::multiprecision::int128_t; 
}
#endif //__SIZEOF_INT128__


#ifndef __SIZEOF_INT128__ 
namespace std{
template<>
struct hash<::nt::uint128_t>{
    std::size_t operator()(const ::nt::uint128_t& x) const {
        return std::hash<uint64_t>()(static_cast<uint64_t>(x)) ^
               std::hash<uint64_t>()(static_cast<uint64_t>(x >> 64));
    }
};
}
#endif // __SIZEOF_INT128__ 

#endif // _NT_TYPES_BIT_128_INTEGER_ENSURE_H_ 
