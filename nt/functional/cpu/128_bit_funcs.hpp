#ifndef NT_FUNCTIONAL_CPU_128_BIT_FUNCS_HPP__
#define NT_FUNCTIONAL_CPU_128_BIT_FUNCS_HPP__

#include "../../types/Types.h"
#include "../../convert/std_convert.h"
#include <cmath>
#include <math.h>

//boost 128 bit float functions
#ifdef BOOST_MP_STANDALONE
namespace std{


#define NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(func)\
inline ::nt::float128_t func(const ::nt::float128_t& x){\
    double _x = ::nt::convert::convert<double, ::nt::float128_t>(x);\
    double _r = func(_x);\
    return ::nt::convert::convert<::nt::float128_t, double>(_r);\
}\

NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(exp);
NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(log);
NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(sqrt);
NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(abs);

NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(tanh);
NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(cosh);
NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(sinh);
NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(asinh);
NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(acosh);
NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(atanh);
NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(atan);
NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(asin);
NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(acos);
NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(tan);
NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(sin);
NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(cos);


#undef NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE

inline ::nt::float128_t pow(const ::nt::float128_t& a, const ::nt::float128_t& b){
    double _a = ::nt::convert::convert<double, ::nt::float128_t>(a);
    double _b = ::nt::convert::convert<double, ::nt::float128_t>(b);
    double _r = pow(_a, _b);
    return ::nt::convert::convert<::nt::float128_t, double>(_r);
}

}
#endif //BOOST_MP_STANDALONE




namespace std{
//making of specific types



#define __NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, func_name)\
inline type func_name(type t){\
    ::nt::float128_t _t = ::nt::convert::convert<::nt::float128_t, type>(t);\
    ::nt::float128_t _r = func_name(_t);\
    return ::nt::convert::convert<type, ::nt::float128_t>(_r);\
}

#define NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, log)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, exp)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, sqrt)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, tanh)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, cosh)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, sinh)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, asinh)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, acosh)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, atanh)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, atan)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, asin)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, acos)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, tan)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, sin)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, cos)\


#ifdef __SIZEOF_INT128__
NT_MAKE_LARGE_STD_FUNCTION_ROUTE(::nt::int128_t)
inline ::nt::int128_t pow(::nt::int128_t a, ::nt::int128_t b){
    long double _a = static_cast<long double>(::nt::convert::convert<int64_t>(a));
    long double _b = static_cast<long double>(::nt::convert::convert<int64_t>(b));
    long double _r = powl(_a, _b);
    int64_t __r(_r);
    return ::nt::convert::convert<::nt::int128_t>(__r);
}

NT_MAKE_LARGE_STD_FUNCTION_ROUTE(uint128_t)
inline ::nt::uint128_t pow(::nt::uint128_t a, ::nt::uint128_t b){
    long double _a = static_cast<long double>(::nt::convert::convert<int64_t>(a));
    long double _b = static_cast<long double>(::nt::convert::convert<int64_t>(b));
    long double _r = powl(_a, _b);
    int64_t __r(_r);
    return ::nt::convert::convert<::nt::uint128_t>(__r);
}

#endif

// #undef NT_MAKE_STD_FUNCTION_ROUTE_LOG
// #undef NT_MAKE_STD_FUNCTION_ROUTE_EXP
#undef __NT_MAKE_LARGE_STD_FUNCTION_ROUTE 
#undef NT_MAKE_LARGE_STD_FUNCTION_ROUTE 

}

#endif
