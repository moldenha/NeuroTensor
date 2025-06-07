#include "../../mp/simde_traits.h"
#include "../../mp/simde_traits/simde_traits_iterators.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include "../../utils/numargs_macro.h"
#include <algorithm>
#include <cmath>
#include <math.h>
#include "../../convert/Convert.h"
#include "../../types/Types.h"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


//if this is defined
//this means that for float128_t boost's 128 bit floating point is used
#ifdef BOOST_MP_STANDALONE
namespace std{

#define NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(func)\
inline ::nt::float128_t func(const ::nt::float128_t& x){return func(x);}

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
NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(sqrt);

#undef NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE

inline ::nt::float128_t pow(const ::nt::float128_t& a, const ::nt::float128_t& b){
    return pow(a, b);
}

}
#endif //BOOST_MP_STANDALONE



namespace std{
//making of specific types


#define NT_STD_FUNCTIONAL_OUT_CONVERSION_LARGE(type, val)\
if constexpr (std::is_same_v<::nt::float128_t, long double>){\
    return ::nt::convert::convert<type, ::nt::float128_t>(val);\
}else{\
    return static_cast<type>(val);\
}\



#define __NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, to, func_name)\
inline type func_name(type t){NT_STD_FUNCTIONAL_OUT_CONVERSION_LARGE(type, func_name##l(static_cast<to>(t)));}

#define NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, to)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, to, tanh)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, to, cosh)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, to, sinh)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, to, asinh)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, to, acosh)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, to, atanh)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, to, atan)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, to, asin)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, to, acos)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, to, tan)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, to, sin)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, to, cos)

#ifdef __SIZEOF_INT128__
NT_MAKE_LARGE_STD_FUNCTION_ROUTE(::nt::int128_t, int64_t)
inline ::nt::int128_t pow(::nt::int128_t a, ::nt::int128_t b){return static_cast<::nt::int128_t>(std::pow(static_cast<long double>(a), static_cast<long double>(b)));}
#endif
NT_MAKE_LARGE_STD_FUNCTION_ROUTE(::nt::uint128_t, uint64_t)
inline ::nt::uint128_t pow(::nt::uint128_t a, ::nt::uint128_t b){return static_cast<::nt::uint128_t>(std::pow(static_cast<long double>(a), static_cast<long double>(b)));}


// #undef NT_MAKE_STD_FUNCTION_ROUTE_LOG
// #undef NT_MAKE_STD_FUNCTION_ROUTE_EXP
#undef NT_STD_FUNCTIONAL_OUT_CONVERSION_LARGE 
#undef __NT_MAKE_LARGE_STD_FUNCTION_ROUTE 
#undef NT_MAKE_LARGE_STD_FUNCTION_ROUTE 

}


namespace std{


#define NT_MAKE_FUNCTION_COMPATIBILITY(func_name)\
inline ::nt::my_complex<float> func_name(::nt::my_complex<float> num) noexcept {\
    return ::nt::my_complex<float>(func_name(num.real()), func_name(num.imag()));\
}\
inline ::nt::my_complex<double> func_name(::nt::my_complex<double> num) noexcept {\
    return ::nt::my_complex<double>(func_name(num.real()), func_name(num.imag()));\
}\
inline ::nt::float16_t func_name(::nt::float16_t num) noexcept {\
    using namespace ::nt;\
    return _NT_FLOAT32_TO_FLOAT16_(func_name(_NT_FLOAT16_TO_FLOAT32_(num)));\
}\
inline ::nt::my_complex<::nt::float16_t> func_name(::nt::my_complex<::nt::float16_t> num) noexcept {\
    return ::nt::my_complex<::nt::float16_t>(func_name(num.real()), func_name(num.imag()));\
}\
\

NT_MAKE_FUNCTION_COMPATIBILITY(asinh);
NT_MAKE_FUNCTION_COMPATIBILITY(acosh);
NT_MAKE_FUNCTION_COMPATIBILITY(atanh);

}

namespace nt {
namespace mp {

#define _NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(func_name, simde_op, transform_op) \
template<typename T, typename O>\
inline void func_name(T begin, T end, O out){\
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<O>>, "Expected to get base types the same for simde optimized routes");\
	using base_type = utils::IteratorBaseType_t<T>;\
	if constexpr (simde_svml_supported_v<base_type>){\
		static constexpr size_t pack_size = pack_size_v<base_type>;\
		for(; begin + pack_size <= end; begin += pack_size, out += pack_size){\
			simde_type<base_type> a = it_loadu(begin);\
			simde_type<base_type> c = SimdTraits<base_type>::simde_op(a);\
			it_storeu(out, c);\
		}\
		std::transform(begin, end, out,\
				[](base_type x) { return transform_op(x); });\
	}\
	else{\
		std::transform(begin, end, out,\
				[](base_type x) { return transform_op(x); });\
	}\
}\
\

#define _NT_MAKE_INV_INLINE_FUNC_(operation, name)\
template<typename T>\
inline T _nt_##name(T element) noexcept {return T(1)/operation(element);}


_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(tanh, tanh, std::tanh);
_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(tan, tan, std::tan);
_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(atan, atan, std::atan);
_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(atanh, atanh, std::atanh);
_NT_MAKE_INV_INLINE_FUNC_(std::tanh, cotanh)
_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(cotanh, cotanh, _nt_cotanh);
_NT_MAKE_INV_INLINE_FUNC_(std::tan, cotan)
_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(cotan, cotan, _nt_cotanh);

_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(sinh, sinh, std::sinh);
_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(sin, sin, std::sin);
_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(asin, asin, std::asin);
_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(asinh, asinh, std::asinh);
_NT_MAKE_INV_INLINE_FUNC_(std::sinh, csch)
_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(csch, csch, _nt_csch);
_NT_MAKE_INV_INLINE_FUNC_(std::sin, csc)
_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(csc, csc, _nt_csc);

_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(cosh, cosh, std::cosh);
_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(cos, cos, std::cos);
_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(acos, acos, std::acos);
_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(acosh, acosh, std::acosh);
_NT_MAKE_INV_INLINE_FUNC_(std::cosh, sech)
_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(sech, sech, _nt_sech);
_NT_MAKE_INV_INLINE_FUNC_(std::cos, sec)
_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(sec, sec, _nt_sec);


#undef _NT_MAKE_INV_INLINE_FUNC_
#undef _NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_


template<typename T, typename U>
inline void dtan(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> twos = SimdTraits<base_type>::set1(base_type(2.0));
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> current = it_loadu(begin);
					      current = SimdTraits<base_type>::sec(current);
					      current = SimdTraits<base_type>::pow(current, twos);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out)
			*out = (base_type(1.0) / (std::pow(std::cos(*begin), 2)));
	}else{
        for(;begin != end; ++begin, ++out){
            *out = (base_type(1.0) / (std::pow(std::cos(*begin), 2)));
        }
	}
}


template<typename T, typename U>
inline void dtanh(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> twos = SimdTraits<base_type>::set1(base_type(2.0));
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> current = it_loadu(begin);
					      current = SimdTraits<base_type>::sech(current);
					      current = SimdTraits<base_type>::pow(current, twos);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out)
			*out = (1 / (std::pow(std::cosh(*begin), 2)));
	}else{
		for(;begin != end; ++begin, ++out){
			*out = (1 / (std::pow(std::cosh(*begin), 2)));
		}
	}
}


template<typename T, typename U>
inline void dcos(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> current = it_loadu(begin);
					      current = SimdTraits<base_type>::sin(current);
					      current = SimdTraits<base_type>::negative(current);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out)
			*out = -std::sin(*begin);
	}else{
        for(;begin != end; ++begin, ++out){
			*out = -std::sin(*begin);
        }
	}
}

template<typename T, typename U>
inline void dasin(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
    base_type base__one(1);
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> ones = SimdTraits<base_type>::set1(base__one);

		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> current = it_loadu(begin);
					      current = SimdTraits<base_type>::multiply(current, current);
					      current = SimdTraits<base_type>::subtract(ones, current);
					      current = SimdTraits<base_type>::invsqrt(current);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out)
			*out = base__one / std::sqrt(base__one - (*begin * *begin));
	}else{
        for(;begin != end; ++begin, ++out){
			*out = base__one / std::sqrt(base__one - (*begin * *begin));
        }
	}
}


template<typename T, typename U>
inline void dacos(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
    base_type base__one(1);
    base_type base__NGone(-1);
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> ones = SimdTraits<base_type>::set1(base__one);

		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> current = it_loadu(begin);
					      current = SimdTraits<base_type>::multiply(current, current);
					      current = SimdTraits<base_type>::subtract(ones, current);
					      current = SimdTraits<base_type>::invsqrt(current);
					      current = SimdTraits<base_type>::negative(current);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out)
			*out = base__NGone / std::sqrt(base__one - (*begin * *begin));
	}else{
        for(;begin != end; ++begin, ++out){
			*out = base__NGone / std::sqrt(base__one - (*begin * *begin));
        }
	}
}

template<typename T, typename U>
inline void datan(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
    base_type base__one(1);
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> ones = SimdTraits<base_type>::set1(base__one);

		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> current = it_loadu(begin);
					      current = SimdTraits<base_type>::multiply(current, current);
					      current = SimdTraits<base_type>::add(ones, current);
					      current = SimdTraits<base_type>::reciprical(current);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out)
			*out = base__one / (base__one + (*begin * *begin));
	}else{
        for(;begin != end; ++begin, ++out){
			*out = base__one / (base__one + (*begin * *begin));
        }
	}
}

template<typename T, typename U>
inline void dasinh(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
    base_type base__one(1);
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> ones = SimdTraits<base_type>::set1(base__one);

		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> current = it_loadu(begin);
					      current = SimdTraits<base_type>::multiply(current, current);
					      current = SimdTraits<base_type>::add(current, ones);
					      current = SimdTraits<base_type>::invsqrt(current);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out)
			*out = base__one / std::sqrt((*begin * *begin) + base__one);
	}else{
        for(;begin != end; ++begin, ++out){
			*out = base__one / std::sqrt((*begin * *begin) + base__one);
        }
	}
}


template<typename T, typename U>
inline void dacosh(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
    base_type base__one(1)
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> ones = SimdTraits<base_type>::set1(base__one);

		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> current = it_loadu(begin);
					      current = SimdTraits<base_type>::multiply(current, current);
					      current = SimdTraits<base_type>::subtract(current, ones);
					      current = SimdTraits<base_type>::invsqrt(current);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out)
			*out = base__one / std::sqrt((*begin * *begin) - base__one);
	}else{
        for(;begin != end; ++begin, ++out){
			*out = base__one / std::sqrt((*begin * *begin) - base__one);
        }
	}
}


template<typename T, typename U>
inline void datanh(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
    base_type base__one(1);
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> ones = SimdTraits<base_type>::set1(base__one);

		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> current = it_loadu(begin);
					      current = SimdTraits<base_type>::multiply(current, current);
					      current = SimdTraits<base_type>::subtract(ones, current);
					      current = SimdTraits<base_type>::reciprical(current);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out)
			*out = base__one / (base__one - (*begin * *begin));
	}else{
        for(;begin != end; ++begin, ++out){
			*out = base__one / (base__one - (*begin * *begin));
        }
	}
}

template<typename T, typename U>
inline void dcotan(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;

		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){

			simde_type<base_type> current = it_loadu(begin);
                          current = SimdTraits<base_type>::csc(current);
					      current = SimdTraits<base_type>::multiply(current, current);
					      current = SimdTraits<base_type>::negative(current);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out){
		    base_type val = _nt_csc(*begin);
            *out = -(val*val);
        }
	}else{
        for(;begin != end; ++begin, ++out){
		    base_type val = _nt_csc(*begin);
            *out = -(val*val);
        }
	}
}


template<typename T, typename U>
inline void dcotanh(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;

		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){

			simde_type<base_type> current = it_loadu(begin);
                          current = SimdTraits<base_type>::csch(current);
					      current = SimdTraits<base_type>::multiply(current, current);
					      current = SimdTraits<base_type>::negative(current);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out){
		    base_type val = _nt_csch(*begin);
            *out = -(val*val);
        }
	}else{
        for(;begin != end; ++begin, ++out){
		    base_type val = _nt_csch(*begin);
            *out = -(val*val);
        }
	}
}


template<typename T, typename U>
inline void dcsc(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;

		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){

			simde_type<base_type> current = it_loadu(begin);
                          current = SimdTraits<base_type>::multiply(
                                    SimdTraits<base_type>::csc(current),
                                    SimdTraits<base_type>::cotan(current));
					      current = SimdTraits<base_type>::negative(current);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out){
		    *out = -(_nt_csc(*begin) * _nt_cotan(*begin));
        }
	}else{
        for(;begin != end; ++begin, ++out){
		    *out = -(_nt_csc(*begin) * _nt_cotan(*begin));
        }
	}
}

template<typename T, typename U>
inline void dcsch(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;

		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){

			simde_type<base_type> current = it_loadu(begin);
                          current = SimdTraits<base_type>::multiply(
                                    SimdTraits<base_type>::csch(current),
                                    SimdTraits<base_type>::cotanh(current));
					      current = SimdTraits<base_type>::negative(current);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out){
		    *out = -(_nt_csch(*begin) * _nt_cotanh(*begin));
        }
	}else{
        for(;begin != end; ++begin, ++out){
		    *out = -(_nt_csch(*begin) * _nt_cotanh(*begin));
        }
	}
}



template<typename T, typename U>
inline void dsec(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;

		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){

			simde_type<base_type> current = it_loadu(begin);
                          current = SimdTraits<base_type>::multiply(
                                    SimdTraits<base_type>::sec(current),
                                    SimdTraits<base_type>::tan(current));
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out){
		    *out = (_nt_sec(*begin) * std::tan(*begin));
        }
	}else{
        for(;begin != end; ++begin, ++out){
		    *out = (_nt_sec(*begin) * std::tan(*begin));
        }
	}
}



template<typename T, typename U>
inline void dsech(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;

		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){

			simde_type<base_type> current = it_loadu(begin);
                          current = SimdTraits<base_type>::multiply(
                                    SimdTraits<base_type>::sech(current),
                                    SimdTraits<base_type>::tanh(current));
					      current = SimdTraits<base_type>::negative(current);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out){
		    *out = -(_nt_sech(*begin) * std::tanh(*begin));
        }
	}else{
        for(;begin != end; ++begin, ++out){
		    *out = -(_nt_sech(*begin) * std::tanh(*begin));
        }
	}
}

} // namespace mp
} // namespace nt

namespace nt {
namespace functional {
namespace cpu {

#define ADD_UNDERSCORE(name) _##name
#define ADD_DOUBLE_UNDERSCORE(name) _##name##_

#define NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(func_name)\
void ADD_UNDERSCORE(func_name)(const ArrayVoid& a, ArrayVoid& out){\
    if(a.dtype == DType::Bool){\
        throw std::logic_error("Cannot run" \
                                #func_name \
                                "on bool data type"); \
    }\
    out = a.clone();\
    out.execute_function_chunk<WRAP_DTYPES<NumberTypesL> >([](auto begin, auto end){\
        mp::func_name(begin, end, begin);\
    });\
}\
\
\
void ADD_DOUBLE_UNDERSCORE(func_name)(ArrayVoid& out){\
   if(out.dtype == DType::Bool){\
        throw std::logic_error("Cannot run" \
                                #func_name \
                                "on bool data type"); \
    }\
    out.execute_function_chunk<WRAP_DTYPES<NumberTypesL> >([](auto begin, auto end){\
        mp::func_name(begin, end, begin);\
    });\
}\
 

NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(tanh);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(tan);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(dtanh);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(dtan);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(atan);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(atanh);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(datan);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(datanh);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(cotanh);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(cotan);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(dcotanh);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(dcotan);

NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(sinh);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(sin);
//d(sin) = cos
//d(sinh) = cosh
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(asin);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(dasin);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(asinh);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(dasinh);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(csch);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(csc);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(dcsch);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(dcsc);

NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(cosh);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(cos);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(dcos);
//d(cosh) = sinh
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(acos);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(dacos);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(acosh);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(dacosh);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(sech);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(sec);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(dsech);
NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_(dsec);


#undef NT_MAKE_ACCESSIBLE_TRIG_FUNCTION_ 
} // namespace cpu
} // namespace functional
} // namespace nt
