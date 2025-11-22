// This is a header file that ensures that nt::uint128_t and nt::int128_t is included
#ifndef NT_TYPES_BIT_128_INTEGER_ENSURE_H__
#define NT_TYPES_BIT_128_INTEGER_ENSURE_H__

#ifdef __SIZEOF_INT128__
namespace nt{
using uint128_t = __uint128_t;
using int128_t = __int128_t;
std::ostream& operator<<(std::ostream& os, const __int128_t i);
std::ostream& operator<<(std::ostream& os, const __uint128_t i);
namespace type_traits{
constexpr bool system_int128 = true;
}
}

#else

namespace nt::type_traits{
constexpr bool system_int128 = false;
}

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
    NT_ALWAYS_INLINE std::size_t operator()(const ::nt::uint128_t& x) const {
        return std::hash<uint64_t>()(static_cast<uint64_t>(x)) ^
               std::hash<uint64_t>()(static_cast<uint64_t>(x >> 64));
    }
};

template<>
struct hash<::nt::int128_t>{
    NT_ALWAYS_INLINE std::size_t operator()(const ::nt::int128_t& s) const noexcept{
        return std::hash<int64_t>()(int64_t(x)) ^
               std::hash<int64_t>()(int64_t(x >> 64));
    }
};
}
#endif // __SIZEOF_INT128__ 

#include "../utils/type_traits.h"

static_assert(nt::type_traits::is_same_v<int, int>);

namespace nt::type_traits{


// template<>
// struct is_integer<::nt::uint128_t> : true_type{};
// template<>
// struct is_integer<const ::nt::uint128_t> : true_type{};
// template<>
// struct is_integer<volatile const ::nt::uint128_t> : true_type{};
// template<>
// struct is_integer<volatile ::nt::uint128_t> : true_type{};

template<>
struct is_integral<::nt::uint128_t>: true_type {};
template<>
struct is_integral<const ::nt::uint128_t>: true_type {};
template<>
struct is_integral<volatile ::nt::uint128_t>: true_type {};
template<>
struct is_integral<volatile const ::nt::uint128_t>: true_type {};

// template<>
// struct is_integer<::nt::int128_t> : true_type{};
// template<>
// struct is_integer<const ::nt::int128_t> : true_type{};
// template<>
// struct is_integer<volatile ::nt::int128_t> : true_type{};
// template<>
// struct is_integer<volatile const ::nt::int128_t> : true_type{};

template<>
struct is_integral<::nt::int128_t>: true_type {};
template<>
struct is_integral<const ::nt::int128_t>: true_type {};
template<>
struct is_integral<volatile ::nt::int128_t>: true_type {};
template<>
struct is_integral<volatile const ::nt::int128_t>: true_type {};
}

#endif // _NT_TYPES_BIT_128_INTEGER_ENSURE_H_ 
