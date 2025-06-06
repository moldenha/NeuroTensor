#include "softmax.h"
// tensors need to be included in this one
#include "../../mp/simde_traits.h"
#include "../../mp/simde_traits/simde_traits_iterators.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include <algorithm>

#include "../../convert/Convert.h"
#include "../../types/Types.h"

//if this is defined
//this means that for float128_t boost's 128 bit floating point is used
#ifdef BOOST_MP_STANDALONE
namespace std{

#define NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(func)\
inline ::nt::float128_t func(const ::nt::float128_t& x){return boost::multiprecition::func##q(x);}

NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(exp);
NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE(log);

#undef NT_MAKE_BOOST_FLOAT128_FUNCTION_ROUTE

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
__NT_MAKE_LARGE_STD_FUNCTION_ROUTE(type, to, exp)

#ifdef __SIZEOF_INT128__
NT_MAKE_LARGE_STD_FUNCTION_ROUTE(::nt::int128_t, int64_t)
#endif
NT_MAKE_LARGE_STD_FUNCTION_ROUTE(::nt::uint128_t, uint64_t)


// #undef NT_MAKE_STD_FUNCTION_ROUTE_LOG
// #undef NT_MAKE_STD_FUNCTION_ROUTE_EXP
#undef NT_STD_FUNCTIONAL_OUT_CONVERSION_LARGE 
#undef __NT_MAKE_LARGE_STD_FUNCTION_ROUTE 
#undef NT_MAKE_LARGE_STD_FUNCTION_ROUTE 

}



namespace nt {
namespace mp {


template<typename T>
inline utils::IteratorBaseType_t<T> accumulate(T begin, T end, utils::IteratorBaseType_t<T> init){
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		for(;begin + pack_size <= end; begin += pack_size){
			init += SimdTraits<base_type>::sum(it_loadu(begin));
		}
		for(;begin < end; ++begin)
			init += *begin;
		return init;
	}else{
		return std::accumulate(begin, end, init);
	}
}


template<typename T, typename O>
inline void exp(T begin, T end, O out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<O>>, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		for(; begin + pack_size <= end; begin += pack_size, out += pack_size){\
			simde_type<base_type> a = it_loadu(begin);
			simde_type<base_type> c = SimdTraits<base_type>::exp(a);
			it_storeu(out, c);
		}
		std::transform(begin, end, out,
				[](base_type x) { return std::exp(x); });
	}
	else{
		std::transform(begin, end, out,
				[](base_type x) { return std::exp(x); });
	}
}

template<typename T, typename O>
inline void log(T begin, T end, O out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<O>>, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		for(; begin + pack_size <= end; begin += pack_size, out += pack_size){\
			simde_type<base_type> a = it_loadu(begin);
			simde_type<base_type> c = SimdTraits<base_type>::log(a);
			it_storeu(out, c);
		}
		std::transform(begin, end, out,
				[](base_type x) { return std::log(x); });
	}
	else{
		std::transform(begin, end, out,
				[](base_type x) { return std::log(x); });
	}
}



} // namespace mp
} // namespace nt

namespace nt {
namespace functional {
namespace cpu {

void _exp(const ArrayVoid& a, ArrayVoid& out){
    if(a.dtype == DType::Bool){
        throw std::logic_error("Cannot run exp on bool data type");
    }
    out = a.clone();
    out.execute_function_chunk<WRAP_DTYPES<NumberTypesL> >([](auto begin, auto end){
        mp::exp(begin, end, begin);
    });
}

void _log(const ArrayVoid& a, ArrayVoid& out){
    if(a.dtype == DType::Bool){
        throw std::logic_error("Cannot run log on bool data type");
    }
    out = a.clone();
    out.execute_function_chunk<WRAP_DTYPES<NumberTypesL> >([](auto begin, auto end){
        mp::log(begin, end, begin);
    });
}

void _exp_(ArrayVoid& a){
    if(a.dtype == DType::Bool){
        throw std::logic_error("Cannot run exp on bool data type");
    }
    a.execute_function_chunk<WRAP_DTYPES<NumberTypesL> >([](auto begin, auto end){
        mp::exp(begin, end, begin);
    });

}
void _log_(ArrayVoid& a){
    if(a.dtype == DType::Bool){
        throw std::logic_error("Cannot run log on bool data type");
    }
    a.execute_function_chunk<WRAP_DTYPES<NumberTypesL> >([](auto begin, auto end){
        mp::log(begin, end, begin);
    });
 
}


Scalar _accumulate(const ArrayVoid& a, Scalar initial){
    if(a.dtype == DType::Bool){
        throw std::logic_error("Cannot run accumulate on bool data type");
    }
    return a.cexecute_function<WRAP_DTYPES<NumberTypesL> >([&initial](auto begin, auto end) -> Scalar{
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        value_t val = initial.to<value_t>();
        return mp::accumulate(begin, end, val);
    });
}

void _sum_every(const ArrayVoid& a, ArrayVoid& out, int64_t size){
    if(a.dtype == DType::Bool){
        throw std::logic_error("Cannot run log on bool data type");
    }
    if(a.Size() % size != 0){
        throw std::logic_error("Error with sum every and size division modulous");
    }
    if(a.Size() / size != out.Size()){
        throw std::logic_error("Error with sum every and size division");
    }

    a.cexecute_function<WRAP_DTYPES<NumberTypesL>>([&out, &size](auto begin, auto end){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        value_t* out_p = reinterpret_cast<value_t*>(out.data_ptr());
        for(;begin != end; begin += size, ++out_p){
            *out_p = mp::accumulate(begin, begin+size, *out_p);
        }
    });



}

} // namespace cpu
} // namespace functional
} // namespace nt
