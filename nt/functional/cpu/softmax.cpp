#include "softmax.h"
// tensors need to be included in this one
#include "../../mp/simde_traits.h"
#include "../../mp/simde_traits/simde_traits_iterators.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include <algorithm>

#include "128_bit_funcs.hpp"


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


template<typename T, typename U>
inline void divide_num(T begin, T end, U out, utils::IteratorBaseType_t<T> num){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> nums = SimdTraits<base_type>::set1(num);
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> a = it_loadu(begin);
			simde_type<base_type> c = SimdTraits<base_type>::divide(a, nums);
			it_storeu(out, c);
		}
		for(;begin < end; ++begin, ++out)
			*out = *begin / num;
	}else{
		for(; begin != end; ++begin, ++out)
			*out = *begin / num;
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


//softmax function
template<typename T, typename U>
inline void softmax(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	U end_out = out + (end - begin);
	exp(begin, end, out);
	base_type total = accumulate(out, end_out, base_type(0.0));
	divide_num(out, end_out, out, total);
}

template<typename T, typename U>
inline void subtract_num(T begin, T end, U out, utils::IteratorBaseType_t<T> num){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> nums = SimdTraits<base_type>::set1(num);
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> a = it_loadu(begin);
			simde_type<base_type> c = SimdTraits<base_type>::subtract(a, nums);
			it_storeu(out, c);
		}
		for(;begin < end; ++begin, ++out)
			*out = *begin - num;
	}else{
		for(; begin != end; ++begin, ++out)
			*out = *begin - num;
	}	
}


template<typename T, typename U>
inline void softmax_stable(T begin, T end, U out, utils::IteratorBaseType_t<T> max){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	U end_out = out + (end - begin);
	exp(begin, end, out);
    subtract_num(out, end_out, out, max);
	base_type total = accumulate(out, end_out, base_type(0.0));
	divide_num(out, end_out, out, total);
}




//derivative of softmax functions:
template<typename T, typename U>
inline utils::IteratorBaseType_t<T> inner_product(T begin, T end, U begin2, utils::IteratorBaseType_t<T> init){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> initial = SimdTraits<base_type>::zero();
		for(;begin + pack_size <= end; begin += pack_size, begin2 += pack_size){
			SimdTraits<base_type>::fmadd(it_loadu(begin), it_loadu(begin2), initial);
		}
		init += SimdTraits<base_type>::sum(initial);
		for(;begin < end; ++begin, ++begin2)
			init += (*begin * *begin2);
		return init;
	}else{
		return std::inner_product(begin, end, begin2, init);
	}	
}



template<typename T, typename U, typename O>
inline void multiply(T begin, T end, U begin2, O out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, 
    utils::IteratorBaseType_t<U> > && 
    std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<O>>, 
    "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		for(; begin + pack_size <= end; begin += pack_size, begin2 += pack_size, out += pack_size){
			simde_type<base_type> a = it_loadu(begin);
			simde_type<base_type> b = it_loadu(begin2);
			simde_type<base_type> c = SimdTraits<base_type>::multiply(a, b);
			it_storeu(out, c);
		}
		std::transform(begin, end, begin2, out, std::multiplies<>{});
	}
	else{
		std::transform(begin, end, begin2, out, std::multiplies<>{});
	}
}


template<typename T, typename U, typename V>
inline void Dsoftmax(T softmax_output, T softmax_output_end,
                     U dL_dY_begin, U dL_dY_end,
                     V out ){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >
        && std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<V> >, 
        "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	V end_out = out + (softmax_output - softmax_output_end);
    base_type dot(0);
    dot = inner_product(softmax_output, softmax_output_end,  dL_dY_begin, dot);
    subtract_num(dL_dY_begin, dL_dY_end, out, dot);
    multiply(softmax_output,softmax_output_end, out, out); 
}


template<typename base_type>
NT_ALWAYS_INLINE simde_type<base_type> gumbel_clamp(const simde_type<base_type>& u){
    static simde_type<base_type> min_v = SimdTraits<base_type>::set1(base_type(-3));
    static simde_type<base_type> max_v = SimdTraits<base_type>::set1(base_type(3));
    return SimdTraits<base_type>::min(SimdTraits<base_type>::max(u, min_v), max_v);
}

template<typename T>
NT_ALWAYS_INLINE T gumbel_clamp_r(const T& u){
    return std::clamp(u, T(-3), T(3));
}


template<typename base_type>
NT_ALWAYS_INLINE simde_type<base_type> sample_gumbel_noise(const simde_type<base_type>& u){
    static simde_type<base_type> simd_scalar = SimdTraits<base_type>::set1(base_type(1e-10));
    return gumbel_clamp<base_type>( SimdTraits<base_type>::negative(SimdTraits<base_type>::log(SimdTraits<base_type>::add(
        SimdTraits<base_type>::negative(
            SimdTraits<base_type>::log(SimdTraits<base_type>::add(u, simd_scalar))
        ), simd_scalar)
    ) ) );
}

template<typename T>
NT_ALWAYS_INLINE T sample_gumbel_noise_r(const T& u){
    return gumbel_clamp_r(-std::log(-std::log(u + T(1e-10)) + T(1e-10)));
}



// begin is the input logits
// begin2 is the random uniform noise
template<typename T, typename U, typename O>
inline void gumbel_algorithm(T begin, T end, U begin2, O out, utils::IteratorBaseType_t<T> tau){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, 
    utils::IteratorBaseType_t<U> > && 
    std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<O>>, 
    "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
        simde_type<base_type> simd_tau = SimdTraits<base_type>::set1(tau);
		static constexpr size_t pack_size = pack_size_v<base_type>;
		for(; begin + pack_size <= end; begin += pack_size, begin2 += pack_size, out += pack_size){
			simde_type<base_type> a = it_loadu(begin);
			simde_type<base_type> b = it_loadu(begin2);
            simde_type<base_type> noise = sample_gumbel_noise<base_type>(b);
			simde_type<base_type> c = SimdTraits<base_type>::divide(SimdTraits<base_type>::add(a, noise), simd_tau);
			it_storeu(out, c);
		}
        for(;begin != end; ++begin, ++begin2, ++out){
            base_type noise = sample_gumbel_noise_r(*begin2);
            *out = (*begin + noise) / tau;
        }
	}
	else{
        for(;begin != end; ++begin, ++begin2, ++out){
            base_type noise = sample_gumbel_noise_r(*begin2);
            *out = (*begin + noise) / tau;
        }
	}
}


} // namespace mp
} // namespace nt

namespace nt {
namespace functional {
namespace cpu {

void _softmax(ArrayVoid& in, ArrayVoid& out){
    in.execute_function<WRAP_DTYPES<NumberTypesL> >([](auto begin, auto end, auto begin2){
        mp::softmax(begin, end, begin2);
    }, out);
}

void _softmax_stable(ArrayVoid& in, ArrayVoid& out, Scalar max){
     in.execute_function<WRAP_DTYPES<NumberTypesL> >([&max](auto begin, auto end, auto begin2){
        using base_type = utils::IteratorBaseType_t<decltype(begin)>;
        base_type val = max.to<base_type>();
        mp::softmax_stable(begin, end, begin2, val);
    }, out);

}

void _gumbel_algorithm_(ArrayVoid& in_o, ArrayVoid& noise, Scalar tau){
    in_o.execute_function<WRAP_DTYPES<NumberTypesL> >([&tau](auto begin, auto end, auto begin2){
        using base_type = utils::IteratorBaseType_t<decltype(begin)>;
        mp::gumbel_algorithm(begin, end, begin2, begin, tau.to<base_type>());
    }, noise); 
}



void _dsoftmax(const ArrayVoid& softmax_output, const ArrayVoid& dL_dY, ArrayVoid& out){
    if(softmax_output.dtype() != dL_dY.dtype() && softmax_output.dtype() != out.dtype()){
        throw std::invalid_argument("got different dtypes for softmax derivative");
    }
    if(!out.is_contiguous()){
        softmax_output.cexecute_function<WRAP_DTYPES<NumberTypesL> >([&out](auto begin, auto end, auto begin2){
            using base_type = utils::IteratorBaseType_t<decltype(begin)>;
            // constexpr DType dt = DTypeFuncs::type_to_dtype<base_type>;
            // have to have this format for MSVC :/
            out.execute_function<WRAP_DTYPES<DTypeEnum<DTypeFuncs::type_to_dtype<utils::IteratorBaseType_t<decltype(begin)>>>>>([&](auto out_b, auto out_e){
                mp::Dsoftmax(begin, end, begin2, begin2 + (end-begin), out_b);
            });
            
        }, dL_dY);
        return;
    }
    softmax_output.cexecute_function<WRAP_DTYPES<NumberTypesL> >([&out](auto begin, auto end, auto begin2){
        using base_type = utils::IteratorBaseType_t<decltype(begin)>;
        base_type* out_ptr = reinterpret_cast<base_type*>(out.data_ptr());
        mp::Dsoftmax(begin, end, begin2, begin2 + (end-begin), out_ptr);
        
    }, dL_dY);
}



} // namespace cpu
} // namespace functional
} // namespace nt
