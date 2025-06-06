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

NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(exp);
NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(log);
NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(sqrt);
NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(tanh);

#undef NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE

inline ::nt::float128_t pow(const ::nt::float128_t& a, const ::nt::float128_t& b){
    return boost::multiprecision::powq(a, b);
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
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, to, log)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, to, exp)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, to, sqrt)\
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, to, tanh)\

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
inline T _nt_##name(T element) noexcept {return 1.0/operation(element);}


_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(sqrt, sqrt, std::sqrt);
_NT_MAKE_INV_INLINE_FUNC_(std::sqrt, invsqrt)
_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(invsqrt, invsqrt, _nt_invsqrt);

#undef _NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_
#undef _NT_MAKE_INV_INLINE_FUNC_ 

template<typename T, typename U>
inline void sigmoid(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> ones = SimdTraits<base_type>::set1(base_type(1.0));
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> current = it_loadu(begin);
					      current = SimdTraits<base_type>::negative(current);
					      current = SimdTraits<base_type>::exp(current);
					      current = SimdTraits<base_type>::add(ones, current);
					      current = SimdTraits<base_type>::reciprical(current);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out)
			*out = (base_type(1.0) / (base_type(1.0) + std::exp(-(*begin))));
	}else{
        for(;begin != end; ++begin, ++out){
            *out = (base_type(1.0) / (base_type(1.0) + std::exp(-(*begin))));
        }
	}
}


template<typename T, typename U>
inline void dsigmoid(T begin, T end, U out, bool apply_sigmoid){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	if(apply_sigmoid){
		sigmoid<T, U>(begin, end, out);
		dsigmoid(out, out + (end-begin), out, false);
	}
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> ones = SimdTraits<base_type>::set1(base_type(1.0));
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> current = it_loadu(begin);
			simde_type<base_type> current_m = SimdTraits<base_type>::subtract(ones, current);
					      current = SimdTraits<base_type>::multiply(current, current_m);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out)
			*out = *begin * (base_type(1.0) - *begin);
	}else{
        for(;begin != end; ++begin, ++out){
            *out = *begin * (base_type(1.0) - *begin);
        }
	}
}

//x * sigmoid(x)
template<typename T, typename U>
inline void silu(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> ones = SimdTraits<base_type>::set1(base_type(1.0));
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> current = it_loadu(begin);
                          current = SimdTraits<base_type>::multiply(current, 
                                        SimdTraits<base_type>::reciprical(
                                        SimdTraits<base_type>::add(ones, 
                                        SimdTraits<base_type>::exp(
                                        SimdTraits<base_type>::negative(current)))));
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out)
			*out = *begin * (base_type(1.0) / (base_type(1.0) + std::exp(-(*begin))));
	}else{
        for(;begin != end; ++begin, ++out){
            *out = *begin * (base_type(1.0) / (base_type(1.0) + std::exp(-(*begin))));
        }
	}
}

template<typename T, typename U>
inline void dsilu(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> ones = SimdTraits<base_type>::set1(base_type(1.0));
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
            simde_type<base_type> current = it_loadu(begin);
            simde_type<base_type> sigmoid_x = SimdTraits<base_type>::reciprical(
                                                SimdTraits<base_type>::add(ones, 
                                                SimdTraits<base_type>::exp(
                                                SimdTraits<base_type>::negative(current))));
            current = SimdTraits<base_type>::multiply(current,
                                            SimdTraits<base_type>::subtract(ones, sigmoid_x));
            current = SimdTraits<base_type>::add(ones, current);
            current = SimdTraits<base_type>::multiply(sigmoid_x, current);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out){
			base_type sigmoid_x = (base_type(1.0) / (base_type(1.0) + std::exp(-(*begin))));
            *out = sigmoid_x * (1 + *begin * (1 - sigmoid_x));
        }
	}else{
        for(;begin != end; ++begin, ++out){
			base_type sigmoid_x = (base_type(1.0) / (base_type(1.0) + std::exp(-(*begin))));
            *out = sigmoid_x * (1 + *begin * (1 - sigmoid_x));
        }
	}
}


//Scalar sqrt_2_pi = std::sqrt(2.0 / M_PI);
//0.5 * x * (1.0 + tanh(sqrt_2_pi * (x + 0.044715 * std::pow(x, 3))));
template<typename T, typename U>
inline void gelu(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
    base_type sqrt_2_pi = static_cast<base_type>(std::sqrt(2.0 / M_PI));

	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> three = SimdTraits<base_type>::set1(base_type(3.0));
		simde_type<base_type> ones = SimdTraits<base_type>::set1(base_type(1.0));
		simde_type<base_type> half = SimdTraits<base_type>::set1(base_type(0.5));
		simde_type<base_type> weird_num = SimdTraits<base_type>::set1(base_type(0.044715));
		simde_type<base_type> pi_num = SimdTraits<base_type>::set1(sqrt_2_pi);

		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> current = it_loadu(begin);
            simde_type<base_type> pow = SimdTraits<base_type>::pow(current, three);
            simde_type<base_type> first = SimdTraits<base_type>::add(current, 
                                          SimdTraits<base_type>::multiply(weird_num, pow)); //(x + 0.044715 * std::pow(x, 3))

            simde_type<base_type> second = SimdTraits<base_type>::tanh( 
                                          SimdTraits<base_type>::multiply(pi_num, first)); //tanh(sqrt_2_pi * (x + 0.044715 * std::pow(x, 3)))
            current = SimdTraits<base_type>::multiply(current,
                        SimdTraits<base_type>::add(ones, second));
            current = SimdTraits<base_type>::multiply(current, half);
      		it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out)
            *out= base_type(0.5) * *begin * (base_type(1.0) + std::tanh(sqrt_2_pi * ( *begin + base_type(0.044715) * std::pow(*begin, 3))));
	}else{
        for(;begin != end; ++begin, ++out){
            *out= base_type(0.5) * *begin * (base_type(1.0) + std::tanh(sqrt_2_pi * ( *begin + base_type(0.044715) * std::pow(*begin, 3))));
        }
	}
}



//Scalar sqrt_2_pi = std::sqrt(2.0 / M_PI);
//const Scalar c(0.044715)
//z = sqrt_2_pi * (x + c * std::pow(x, 3));
//z = tanh(z)
//tanh_derivative = 1 - (z * z)
//dz_dx = sqrt_2_pi * (1 + 3 * c.to<double>() * x * x);
//0.5 * (1 + z) + 0.5 * x * tanh_derivative * dz_dx
template<typename T, typename U>
inline void dgelu(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
    base_type sqrt_2_pi = static_cast<base_type>(std::sqrt(2.0 / M_PI));
    base_type c  = static_cast<base_type>(0.044715);
    base_type cm  = static_cast<base_type>(3 * 0.044715);

	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> three = SimdTraits<base_type>::set1(base_type(3.0));
		simde_type<base_type> ones = SimdTraits<base_type>::set1(base_type(1.0));
		simde_type<base_type> half = SimdTraits<base_type>::set1(base_type(0.5));
		simde_type<base_type> weird_num = SimdTraits<base_type>::set1(c);
		simde_type<base_type> weird_num_M = SimdTraits<base_type>::set1(cm);
		simde_type<base_type> pi_num = SimdTraits<base_type>::set1(sqrt_2_pi);

		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> current = it_loadu(begin);
            simde_type<base_type> z = SimdTraits<base_type>::tanh(
                                        SimdTraits<base_type>::multiply(pi_num,
                                        SimdTraits<base_type>::add(current,
                                        SimdTraits<base_type>::multiply(weird_num,
                                        SimdTraits<base_type>::pow(current, three)))));
            simde_type<base_type> tanh_derivative = SimdTraits<base_type>::subtract(ones,
                                                        SimdTraits<base_type>::multiply(z, z));
            simde_type<base_type> dz_dx = SimdTraits<base_type>::add(ones,
                                            SimdTraits<base_type>::multiply(weird_num_M, 
                                            SimdTraits<base_type>::multiply(current, current)));
            current = SimdTraits<base_type>::add(
                        SimdTraits<base_type>::multiply(half, SimdTraits<base_type>::add(ones, z)),
                        SimdTraits<base_type>::multiply(
                            half,
                            SimdTraits<base_type>::multiply(
                                current,
                                SimdTraits<base_type>::multiply(tanh_derivative, dz_dx)
                            )
                        )
                    );
      		it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out){
            base_type z = sqrt_2_pi * (*begin + c * std::pow(*begin, 3));
            z = std::tanh(z);
            base_type tanh_derivative = 1 - (z * z);
            base_type dz_dx = sqrt_2_pi * (1 + cm * *begin * *begin);
            *out = 0.5 * (1 + z) + 0.5 * *begin * tanh_derivative * dz_dx;
        }
	}else{
        for(;begin != end; ++begin, ++out){
            base_type z = sqrt_2_pi * (*begin + c * std::pow(*begin, 3));
            z = std::tanh(z);
            base_type tanh_derivative = 1 - (z * z);
            base_type dz_dx = sqrt_2_pi * (1 + cm * *begin * *begin);
            *out = 0.5 * (1 + z) + 0.5 * *begin * tanh_derivative * dz_dx;
        }
	}
}

//it is just 0.5 * invsqrt(x)
template<typename T, typename U>
inline void dsqrt(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> to_mult = SimdTraits<base_type>::set1(base_type(0.5));
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> current = it_loadu(begin);
					      current = SimdTraits<base_type>::invsqrt(current);
					      current = SimdTraits<base_type>::multiply(current, to_mult);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out)
			*out = (1 / (2 * (std::sqrt(*begin))));
	}else{
		for(;begin != end; ++begin, ++out){
			*out = (-1 / (2 * (std::sqrt(*begin))));
		}
	}
}

template<typename T, typename U>
inline void dinvsqrt(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> to_pow = SimdTraits<base_type>::set1(base_type(3.0));
		simde_type<base_type> to_mult = SimdTraits<base_type>::set1(base_type(-0.5));
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> current = it_loadu(begin);
					      current = SimdTraits<base_type>::pow(current, to_pow);
					      current = SimdTraits<base_type>::invsqrt(current);
					      current = SimdTraits<base_type>::multiply(current, to_mult);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out)
			*out = (-1 / (2 * (std::sqrt(std::pow(*begin, 3)))));
	}else{
		for(;begin != end; ++begin, ++out){
			*out = (-1 / (2 * (std::sqrt(std::pow(*begin, 3)))));
		}
	}
}

template<typename T, typename O>
inline void pow(T begin, T end, O out, utils::IteratorBaseType_t<T> num){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<O>>, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> to_pow = SimdTraits<base_type>::set1(num);
		for(; begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> a = it_loadu(begin);
			simde_type<base_type> c = SimdTraits<base_type>::pow(a, to_pow);
			it_storeu(out, c);
		}
		std::transform(begin, end, out,
				[&](base_type x) { return std::pow(x, num); });
	}
	else{
		std::transform(begin, end, out,
				[&](base_type x) { return std::pow(x, num); });
	}

}

} // namespace mp
} // namespace nt

namespace nt {
namespace functional {
namespace cpu {

#define ADD_UNDERSCORE(name) _##name
#define ADD_DOUBLE_UNDERSCORE(name) _##name##_

#define NT_MAKE_ACCESSIBLE_ACTIVATION_FUNCTION_(func_name)\
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
\





NT_MAKE_ACCESSIBLE_ACTIVATION_FUNCTION_(sigmoid);
NT_MAKE_ACCESSIBLE_ACTIVATION_FUNCTION_(sqrt);
NT_MAKE_ACCESSIBLE_ACTIVATION_FUNCTION_(dsqrt);
NT_MAKE_ACCESSIBLE_ACTIVATION_FUNCTION_(invsqrt);
NT_MAKE_ACCESSIBLE_ACTIVATION_FUNCTION_(dinvsqrt);
NT_MAKE_ACCESSIBLE_ACTIVATION_FUNCTION_(silu);
NT_MAKE_ACCESSIBLE_ACTIVATION_FUNCTION_(dsilu);
NT_MAKE_ACCESSIBLE_ACTIVATION_FUNCTION_(gelu);
NT_MAKE_ACCESSIBLE_ACTIVATION_FUNCTION_(dgelu);





void _dsigmoid(const ArrayVoid& a, ArrayVoid& out, const bool& apply_sigmoid){
    if(a.dtype == DType::Bool){
        throw std::logic_error("Cannot run dsigmoid on bool data type");
    }
    out = a.clone();
    out.execute_function_chunk<WRAP_DTYPES<NumberTypesL> >([&apply_sigmoid](auto begin, auto end){
        mp::dsigmoid(begin, end, begin, apply_sigmoid);
    });
}

void _dsigmoid_(ArrayVoid& a, const bool& apply_sigmoid){
    if(a.dtype == DType::Bool){
        throw std::logic_error("Cannot run exp on bool data type");
    }
    a.execute_function_chunk<WRAP_DTYPES<NumberTypesL> >([&apply_sigmoid](auto begin, auto end){
        mp::dsigmoid(begin, end, begin, apply_sigmoid);
    });
}

void _pow_(ArrayVoid& a, Scalar p){
    if(p.isNegative()){_pow_(a.inverse_(), -p); return;}
	if(p.isZero()){ a.fill_(1); return;}
    a.execute_function_chunk<WRAP_DTYPES<RealNumberTypesL, ComplexTypesL> >(
        [&p](auto a_begin, auto a_end){
            using value_t = utils::IteratorBaseType_t<decltype(a_begin)>;
            if(p.to<value_t>() == 1) return;
            mp::pow(a_begin, a_end, a_begin, p.to<value_t>());
    });
}

void _pow(const ArrayVoid& a, ArrayVoid& out, Scalar p){
    out = a.clone();
    _pow_(out, p);
}

void _abs_(ArrayVoid& a){
    if(!DTypeFuncs::is_complex(a.dtype)){
        a.execute_function_chunk<WRAP_DTYPES< FloatingTypesL, SignedTypesL> >([](auto begin, auto end){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
#ifdef __SIZEOF_INT128__
            if constexpr (std::is_same_v<value_t, int128_t>){
                for(;begin != end; ++begin)
                    *begin = static_cast<int128_t>(std::abs(static_cast<int64_t>(*begin)));
                    
            }
            else{
                for(;begin != end; ++begin)
                    *begin = std::abs(*begin);
            }
#else
            for(;begin != end; ++begin)
                *begin = std::abs(*begin);
#endif
        });
    }
    if(DTypeFuncs::is_complex(a.dtype)){
        a.execute_function_chunk<WRAP_DTYPES<ComplexTypesL> >([](auto begin, auto end){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            for(;begin != end; ++begin)
                *begin = value_t(std::abs(std::get<0>(*begin)), std::abs(std::get<1>(*begin)));
        });
    }
}


void _abs(const ArrayVoid& a, ArrayVoid& out){
    out = a.clone();
    _abs_(out);
}

#undef NT_MAKE_ACCESSIBLE_ACTIVATION_FUNCTION_ 
#undef ADD_UNDERSCORE 
#undef ADD_DOUBLE_UNDERSCORE 

} // namespace cpu
} // namespace functional
} // namespace nt
