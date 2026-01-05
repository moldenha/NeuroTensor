#include "operators.h"
// tensors need to be included in this one
#include "../../mp/simde_traits.h"
#include "../../mp/simde_traits/simde_traits_iterators.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include "../../convert/Convert.h"
#include <algorithm>
#include <cmath>


#include "../../utils/always_inline_macro.h"
#include "../../math/functional/floor.hpp"
#include "../../math/functional/fmod.hpp"
#include "../../math/functional/trunc.hpp"



namespace nt {
namespace mp {

NT_ALWAYS_INLINE float128_t euclidean_remainder(float128_t a, float128_t b){
    float128_t r = ::nt::math::fmod(a, b);
    return ((r < 0 && b > 0 )|| (r > 0 && b < 0)) ? r + b : r;
}

NT_ALWAYS_INLINE double euclidean_remainder(double a, double b) {
    double r = ::nt::math::fmod(a, b);
    return ((r < 0 && b > 0 )|| (r > 0 && b < 0)) ? r + b : r;
}

NT_ALWAYS_INLINE float euclidean_remainder(float a, float b){
    float r = ::nt::math::fmod(a, b);
    return ((r < 0 && b > 0 )|| (r > 0 && b < 0)) ? r + b : r;
}

NT_ALWAYS_INLINE float16_t euclidean_remainder(float16_t a, float16_t b){
    float16_t r = ::nt::math::fmod(a, b);
    return ((r < 0 && b > 0 )|| (r > 0 && b < 0)) ? r + b : r;
}


NT_ALWAYS_INLINE complex_128 euclidean_remainder(complex_128 a, complex_128 b){
    return complex_128(euclidean_remainder(a.real(), b.real()), euclidean_remainder(a.imag(), b.imag()));
}

NT_ALWAYS_INLINE complex_64 euclidean_remainder(complex_64 a, complex_64 b){
    return complex_64(euclidean_remainder(a.real(), b.real()), euclidean_remainder(a.imag(), b.imag()));
}

NT_ALWAYS_INLINE complex_32 euclidean_remainder(complex_32 a, complex_32 b){
    return complex_32(euclidean_remainder(a.real(), b.real()), euclidean_remainder(a.imag(), b.imag()));
}


#define NT_SIMDE_OP_TRANSFORM_EQUIVALENT_TWO_(func_name, simde_op,            \
                                               transform_op)                   \
    template <typename T, typename U, typename O>                              \
    inline void func_name(T begin, T end, U begin2, O out) {                   \
        static_assert(                                                         \
            std::is_same_v<utils::IteratorBaseType_t<T>,                       \
                           utils::IteratorBaseType_t<U>> &&                    \
                std::is_same_v<utils::IteratorBaseType_t<T>,                   \
                               utils::IteratorBaseType_t<O>>,                  \
            "Expected to get base types the same for simde optimized routes"); \
        using base_type = utils::IteratorBaseType_t<T>;                        \
        if constexpr (simde_supported_v<base_type>) {                          \
            static constexpr size_t pack_size = pack_size_v<base_type>;        \
            for (; begin + pack_size <= end;                                   \
                 begin += pack_size, begin2 += pack_size, out += pack_size) {  \
                simde_type<base_type> a = it_loadu(begin);                     \
                simde_type<base_type> b = it_loadu(begin2);                    \
                simde_type<base_type> c =                                      \
                    SimdTraits<base_type>::simde_op(a, b);                     \
                it_storeu(out, c);                                             \
            }                                                                  \
            std::transform(begin, end, begin2, out, transform_op<>{});         \
        } else {                                                               \
            std::transform(begin, end, begin2, out, transform_op<>{});         \
        }                                                                      \
    }

NT_SIMDE_OP_TRANSFORM_EQUIVALENT_TWO_(add, add, std::plus);
NT_SIMDE_OP_TRANSFORM_EQUIVALENT_TWO_(subtract, subtract, std::minus);
NT_SIMDE_OP_TRANSFORM_EQUIVALENT_TWO_(multiply, multiply, std::multiplies);
NT_SIMDE_OP_TRANSFORM_EQUIVALENT_TWO_(divide, divide, std::divides);

#undef NT_SIMDE_OP_TRANSFORM_EQUIVALENT_TWO_

template<typename T, typename U>
inline void add_num(T begin, T end, U out, utils::IteratorBaseType_t<T> num){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> nums = SimdTraits<base_type>::set1(num);
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> a = it_loadu(begin);
			simde_type<base_type> c = SimdTraits<base_type>::add(a, nums);
			it_storeu(out, c);
		}
		for(;begin < end; ++begin, ++out)
			*out = *begin + num;
	}else{
		for(; begin != end; ++begin, ++out)
			*out = *begin + num;
	}	
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
inline void num_subtract(T begin, T end, U out, utils::IteratorBaseType_t<T> num){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> nums = SimdTraits<base_type>::set1(num);
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> a = it_loadu(begin);
			simde_type<base_type> c = SimdTraits<base_type>::subtract(nums, a);
			it_storeu(out, c);
		}
		for(;begin < end; ++begin, ++out)
			*out = num - *begin;
	}else{
		for(; begin != end; ++begin, ++out)
			*out = num - *begin;
	}	
}

template<typename T, typename U>
inline void multiply_num(T begin, T end, U out, utils::IteratorBaseType_t<T> num){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> nums = SimdTraits<base_type>::set1(num);
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> a = it_loadu(begin);
			simde_type<base_type> c = SimdTraits<base_type>::multiply(a, nums);
			it_storeu(out, c);
		}
		for(;begin < end; ++begin, ++out)
			*out = *begin * num;
	}else{
		for(; begin != end; ++begin, ++out)
			*out = *begin * num;
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

template<typename T, typename U>
inline void num_divide(T begin, T end, U out, utils::IteratorBaseType_t<T> num){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> nums = SimdTraits<base_type>::set1(num);
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> a = it_loadu(begin);
			simde_type<base_type> c = SimdTraits<base_type>::divide(nums, a);
			it_storeu(out, c);
		}
		for(;begin < end; ++begin, ++out)
			*out = num / *begin;
	}else{
		for(; begin != end; ++begin, ++out)
			*out = num / *begin;
	}	
}


//use this to make a cpu::_dot in the future
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


//
template<typename T, typename U, typename V, typename W>
inline void fmod_backward(T begin, T end, U begin2, V grad_begin, W out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >
               && std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<V> >
               && std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<W> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		for(;begin + pack_size <= end; begin += pack_size, begin2 += pack_size, grad_begin += pack_size, out += pack_size){
            simde_type<base_type> current_a = it_loadu(begin);
            simde_type<base_type> current_b = it_loadu(begin2);
            simde_type<base_type> current_grad = it_loadu(grad_begin);
            
            simde_type<base_type> current = SimdTraits<base_type>::negative(current_grad);
            simde_type<base_type> div = SimdTraits<base_type>::divide(current_a, current_b);
            simde_type<base_type> floored = SimdTraits<base_type>::round(div, SIMDE_MM_FROUND_TO_ZERO);
            current = SimdTraits<base_type>::multiply(current, floored);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++begin2, ++grad_begin, ++out)
            *out = -(*grad_begin) * ::nt::math::trunc(base_type(*begin / *begin2));
	}else{
		for(;begin < end; ++begin, ++begin2, ++grad_begin, ++out)
            *out = -(*grad_begin) * ::nt::math::trunc(*begin / *begin2);
	}	
}


template<typename T, typename U, typename V, typename W>
inline void fmod_backward_scalar(T begin, T end, U grad_begin, V out, W s){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >
               && std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<V> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
    static_assert(std::is_same_v<std::decay_t<W>, base_type>, "Error expected scalar to be the same type");
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> current_a = SimdTraits<base_type>::set1(s);

		for(;begin + pack_size <= end; begin += pack_size, grad_begin += pack_size, out += pack_size){
            // simde_type<base_type> current_a = it_loadu(begin);
            simde_type<base_type> current_b = it_loadu(begin);
            simde_type<base_type> current_grad = it_loadu(grad_begin);
            
            simde_type<base_type> current = SimdTraits<base_type>::negative(current_grad);
            simde_type<base_type> div = SimdTraits<base_type>::divide(current_a, current_b);
            simde_type<base_type> floored = SimdTraits<base_type>::round(div, SIMDE_MM_FROUND_TO_ZERO);
            current = SimdTraits<base_type>::multiply(current, floored);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++grad_begin, ++out)
            *out = -(*grad_begin) * ::nt::math::trunc(s / *begin);
	}else{
		for(;begin < end; ++begin, ++grad_begin, ++out)
            *out = -(*grad_begin) * ::nt::math::trunc(s / *begin);
	}	
}


template<typename T, typename U, typename V, typename W>
inline void remainder_backward(T begin, T end, U begin2, V grad_begin, W out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >
               && std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<V> >
               && std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<W> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		for(;begin + pack_size <= end; begin += pack_size, begin2 += pack_size, grad_begin += pack_size, out += pack_size){
            simde_type<base_type> current_a = it_loadu(begin);
            simde_type<base_type> current_b = it_loadu(begin2);
            simde_type<base_type> current_grad = it_loadu(grad_begin);
            
            simde_type<base_type> current = SimdTraits<base_type>::negative(current_grad);
            simde_type<base_type> div = SimdTraits<base_type>::divide(current_a, current_b);
            simde_type<base_type> floored = SimdTraits<base_type>::floor(div);
            current = SimdTraits<base_type>::multiply(current, floored);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++begin2, ++grad_begin, ++out)
            *out = -(*grad_begin) * ::nt::math::floor(base_type(*begin / *begin2));
	}else{
		for(;begin < end; ++begin, ++begin2, ++grad_begin, ++out)
            *out = -(*grad_begin) * ::nt::math::floor(*begin / *begin2);
	}	
}


template<typename T, typename U, typename V, typename W>
inline void remainder_backward_scalar(T begin, T end, U grad_begin, V out, W s){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >
               && std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<V> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
    static_assert(std::is_same_v<std::decay_t<W>, base_type>, "Error expected scalar to be the same type");
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> current_a = SimdTraits<base_type>::set1(s);

		for(;begin + pack_size <= end; begin += pack_size, grad_begin += pack_size, out += pack_size){
            // simde_type<base_type> current_a = it_loadu(begin);
            simde_type<base_type> current_b = it_loadu(begin);
            simde_type<base_type> current_grad = it_loadu(grad_begin);
            
            simde_type<base_type> current = SimdTraits<base_type>::negative(current_grad);
            simde_type<base_type> div = SimdTraits<base_type>::divide(current_a, current_b);
            simde_type<base_type> floored = SimdTraits<base_type>::floor(div);
            current = SimdTraits<base_type>::multiply(current, floored);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++grad_begin, ++out)
            *out = -(*grad_begin) * ::nt::math::floor(s / *begin);
	}else{
		for(;begin < end; ++begin, ++grad_begin, ++out)
            *out = -(*grad_begin) * ::nt::math::floor(s / *begin);
	}	
}


template<typename T, typename U, typename V>
inline void remainder(T begin, T end, U begin2, V out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >
               && std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<V> > , "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		for(;begin + pack_size <= end; begin += pack_size, begin2 += pack_size, out += pack_size){
            simde_type<base_type> current_a = it_loadu(begin);
            simde_type<base_type> current_b = it_loadu(begin2);
            simde_type<base_type> current = SimdTraits<base_type>::remainder(current_a, current_b);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++begin2, ++out)
            *out = euclidean_remainder(*begin, *begin2);
	}else{
		for(;begin < end; ++begin, ++begin2, ++out)
            *out = euclidean_remainder(*begin, *begin2);
	}	
}


template<typename T, typename U, typename V>
inline void remainder_scalar_second(T begin, T end, U scalar, V out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<V> > , "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
    static_assert(std::is_same_v<std::decay_t<U>, base_type>, "Error expected scalar to be the same type");
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> current_b = SimdTraits<base_type>::set1(scalar);
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
            simde_type<base_type> current_a = it_loadu(begin);
            simde_type<base_type> current = SimdTraits<base_type>::remainder(current_a, current_b);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out)
            *out = euclidean_remainder(*begin, scalar);
	}else{
		for(;begin < end; ++begin, ++out)
            *out = euclidean_remainder(*begin, scalar);
	}	
}

template<typename T, typename U, typename V>
inline void remainder_scalar_first(T begin, T end, U scalar, V out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<V> > , "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
    static_assert(std::is_same_v<std::decay_t<U>, base_type>, "Error expected scalar to be the same type");
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> current_a = SimdTraits<base_type>::set1(scalar);
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
            simde_type<base_type> current_b = it_loadu(begin);
            simde_type<base_type> current = SimdTraits<base_type>::remainder(current_a, current_b);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out)
            *out = euclidean_remainder(scalar, *begin);
	}else{
		for(;begin < end; ++begin, ++out)
            *out = euclidean_remainder(scalar, *begin);
	}	
}


template<typename T, typename U, typename V>
inline void fmod(T begin, T end, U begin2, V out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >
               && std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<V> > , "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		for(;begin + pack_size <= end; begin += pack_size, begin2 += pack_size, out += pack_size){
            simde_type<base_type> current_a = it_loadu(begin);
            simde_type<base_type> current_b = it_loadu(begin2);
            simde_type<base_type> current = SimdTraits<base_type>::fmod(current_a, current_b);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++begin2, ++out)
            *out = ::nt::math::fmod(*begin, *begin2);
	}else{
		for(;begin < end; ++begin, ++begin2, ++out)
            *out = ::nt::math::fmod(*begin, *begin2);
	}	
}


template<typename T, typename U, typename V>
inline void fmod_scalar_second(T begin, T end, U scalar, V out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<V> > , "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
    static_assert(std::is_same_v<std::decay_t<U>, base_type>, "Error expected scalar to be the same type");
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> current_b = SimdTraits<base_type>::set1(scalar);
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
            simde_type<base_type> current_a = it_loadu(begin);
            simde_type<base_type> current = SimdTraits<base_type>::fmod(current_a, current_b);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out)
            *out = ::nt::math::fmod(*begin, scalar);
	}else{
		for(;begin < end; ++begin, ++out)
            *out = ::nt::math::fmod(*begin, scalar);
	}	
}

template<typename T, typename U, typename V>
inline void fmod_scalar_first(T begin, T end, U scalar, V out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<V> > , "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
    static_assert(std::is_same_v<std::decay_t<U>, base_type>, "Error expected scalar to be the same type");
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> current_a = SimdTraits<base_type>::set1(scalar);
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
            simde_type<base_type> current_b = it_loadu(begin);
            simde_type<base_type> current = SimdTraits<base_type>::fmod(current_a, current_b);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out)
            *out = ::nt::math::fmod(scalar, *begin);
	}else{
		for(;begin < end; ++begin, ++out)
            *out = ::nt::math::fmod(scalar, *begin);
	}	
}



#define NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(func_name, simde_op, transform_op) \
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

NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(reciprical, reciprical, 1.0/ );
#undef NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_

} // namespace mp
} // namespace nt

namespace nt {
namespace functional {
namespace cpu {

// multiply divide subtract add
void _operator_mdsa(const ArrayVoid &a, const ArrayVoid &b, ArrayVoid &o,
                    int op) {
    if (op == 0) {
        // multiply
        a.cexecute_function<WRAP_DTYPES<NumberTypesL>>(
            [](auto begin, auto end, auto begin2, void *out_p) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                value_t *out = reinterpret_cast<value_t *>(out_p);
                mp::multiply(begin, end, begin2, out);
            },
            b, o.data_ptr());
    } else if (op == 2) {
        // subtract
        a.cexecute_function<WRAP_DTYPES<NumberTypesL>>(
            [](auto begin, auto end, auto begin2, void *out_p) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                value_t *out = reinterpret_cast<value_t *>(out_p);
                mp::subtract(begin, end, begin2, out);
            },
            b, o.data_ptr());
    } else if (op == 1) {
        // divide
        a.cexecute_function<WRAP_DTYPES<NumberTypesL>>(
            [](auto begin, auto end, auto begin2, void *out_p) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                value_t *out = reinterpret_cast<value_t *>(out_p);
                mp::divide(begin, end, begin2, out);
            },
            b, o.data_ptr());
    } else if (op == 3) {
        // add
        a.cexecute_function<WRAP_DTYPES<NumberTypesL>>(
            [](auto begin, auto end, auto begin2, void *out_p) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                value_t *out = reinterpret_cast<value_t *>(out_p);
                mp::add(begin, end, begin2, out);
            },
            b, o.data_ptr());
    }
}

// multiply divide subtract add
void _operator_mdsa_(ArrayVoid &a, const ArrayVoid &b, int op) {
    if (op == 0) {
        // multiply
        a.execute_function<WRAP_DTYPES<NumberTypesL>>(
            [](auto begin, auto end, auto begin2) {
                mp::multiply(begin, end, begin2, begin);
            },
            const_cast<ArrayVoid &>(b));
    } else if (op == 2) {
        // subtract
        a.execute_function<WRAP_DTYPES<NumberTypesL>>(
            [](auto begin, auto end, auto begin2) {
                mp::subtract(begin, end, begin2, begin);
            },
            const_cast<ArrayVoid &>(b));
    } else if (op == 1) {
        // divide
        a.execute_function<WRAP_DTYPES<NumberTypesL>>(
            [](auto begin, auto end, auto begin2) {
                mp::divide(begin, end, begin2, begin);
            },
            const_cast<ArrayVoid &>(b));
    } else if (op == 3) {
        // add
        a.execute_function<WRAP_DTYPES<NumberTypesL>>(
            [](auto begin, auto end, auto begin2) {
                mp::add(begin, end, begin2, begin);
            },
            const_cast<ArrayVoid &>(b));
    }
}


void _operator_mdsa_scalar(const ArrayVoid& in, ArrayVoid& out, Scalar s, int op){
   if (op == 0) {
        // multiply
        const_cast<ArrayVoid &>(in).execute_function<WRAP_DTYPES<NumberTypesL>>(
            [&s](auto begin, auto end, auto begin2) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                auto v = s.to<value_t>();
                mp::multiply_num(begin, end, begin2, v);
            },
            out);
    } else if (op == 2) {
        // subtract
        const_cast<ArrayVoid &>(in).execute_function<WRAP_DTYPES<NumberTypesL>>(
            [&s](auto begin, auto end, auto begin2) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                auto v = s.to<value_t>();
                mp::subtract_num(begin, end, begin2, v);
            },
            out);
    } else if (op == 1) {
        // divide
        const_cast<ArrayVoid &>(in).execute_function<WRAP_DTYPES<NumberTypesL>>(
            [&s](auto begin, auto end, auto begin2) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                auto v = s.to<value_t>();
                mp::divide_num(begin, end, begin2, v);
            },
            out);
    } else if (op == 3) {
        // add
        const_cast<ArrayVoid &>(in).execute_function<WRAP_DTYPES<NumberTypesL>>(
            [&s](auto begin, auto end, auto begin2) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                auto v = s.to<value_t>();
                mp::add_num(begin, end, begin2, v);
            },
            out);
    }
}

//the previous one would have been in - s for example
//this one would be s - in
void _operator_mdsa_scalar_first(const ArrayVoid& in, ArrayVoid& out, Scalar s, int op){
    if(op == 0 || op == 3){
        _operator_mdsa_scalar(in, out, s, op);
        return;
    }

    if (op == 2) {
        // subtract
        const_cast<ArrayVoid &>(in).execute_function<WRAP_DTYPES<NumberTypesL>>(
            [&s](auto begin, auto end, auto begin2) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                auto v = s.to<value_t>();
                mp::num_subtract(begin, end, begin2, v);
            },
            out);
    } else if (op == 1) {
        // divide
        const_cast<ArrayVoid &>(in).execute_function<WRAP_DTYPES<NumberTypesL>>(
            [&s](auto begin, auto end, auto begin2) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                auto v = s.to<value_t>();
                mp::num_divide(begin, end, begin2, v);
            },
            out);
    } 
}


void _operator_mdsa_scalar_(ArrayVoid& out, Scalar s, int op){
   if (op == 0) {
        // multiply
        out.execute_function_chunk<WRAP_DTYPES<NumberTypesL>>(
            [&s](auto begin, auto end) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                auto v = s.to<value_t>();
                mp::multiply_num(begin, end, begin, v);
            });
    } else if (op == 2) {
        // subtract
        out.execute_function_chunk<WRAP_DTYPES<NumberTypesL>>(
            [&s](auto begin, auto end) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                auto v = s.to<value_t>();
                mp::subtract_num(begin, end, begin, v);
            });
    } else if (op == 1) {
        // divide
        out.execute_function_chunk<WRAP_DTYPES<NumberTypesL>>(
            [&s](auto begin, auto end) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                auto v = s.to<value_t>();
                mp::divide_num(begin, end, begin, v);
            });
    } else if (op == 3) {
        // add
        out.execute_function_chunk<WRAP_DTYPES<NumberTypesL>>(
            [&s](auto begin, auto end) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                auto v = s.to<value_t>();
                mp::add_num(begin, end, begin, v);
            });
    }
}

ArrayVoid _inverse(const ArrayVoid& in){
    DType dtype = in.dtype();
    uint64_t size = in.Size();
    if(dtype == DType::LongLong){
		ArrayVoid output(size, DType::Double);
		in.transform_function<DType::LongLong>([](const auto& inp) -> double {return 1.0/((double)inp);}, reinterpret_cast<double*>(output.data_ptr()));
		return std::move(output);
	}
	if(dtype == DType::int128 || dtype == DType::uint128){
		ArrayVoid output(size, DType::Float128);
		in.transform_function<DType::uint128, DType::int128>([](const auto& inp) -> float128_t 
				{return 1.0/(::nt::convert::convert<DType::Float128>(inp));}, reinterpret_cast<float128_t*>(output.data_ptr()));
        return std::move(output);
	}
	if(dtype == DType::Integer || dtype == DType::Long || dtype == DType::uint8 || dtype == DType::int8 || dtype == DType::uint16 || dtype == DType::int16){
		if(dtype == DType::int64){
			return _inverse(in.to(DType::Float64)); //double is slower with the registers but works
		}else{
			return _inverse(in.to(DType::Float32));
		}
	}
    ArrayVoid output(size, dtype);
	//all svml compatible types
	in.cexecute_function<WRAP_DTYPES<FloatingTypesL,ComplexTypesL> >(
			[&output](auto begin, auto end){
		using value_t = utils::IteratorBaseType_t<decltype(begin)>;
		mp::reciprical(begin, end, output.get_bucket().begin_contiguous<value_t>());
	});
    return std::move(output);
}

void _inverse_(ArrayVoid& arr){
    DType dtype = arr.dtype();
    if(dtype != DType::TensorObj && !DTypeFuncs::is_floating(dtype) && !DTypeFuncs::is_complex(dtype)){
		_inverse_(arr.floating_());
        return;
    }
    arr.execute_function_chunk<WRAP_DTYPES<FloatingTypesL,ComplexTypesL> >([](auto begin, auto end){
		mp::reciprical(begin, end, begin);
	});

}


void _fmod_(ArrayVoid& arr, Scalar num){
    DType dtype = arr.dtype();
    if(DTypeFuncs::is_floating(dtype) || DTypeFuncs::is_complex(dtype)){
        arr.execute_function_chunk<WRAP_DTYPES<FloatingTypesL, ComplexTypesL>>([&num](auto begin, auto end){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            value_t val = num.to<value_t>();
            ::nt::mp::fmod_scalar_second(begin, end, val, begin);
        });
    }
    else{
        arr.execute_function_chunk<WRAP_DTYPES<IntegerTypesL>>([&num](auto begin, auto end){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            value_t val = num.to<value_t>();
            for(;begin != end; ++begin){
                *begin = *begin % val;
            }
        });
    }
}

void _fmod_first_scalar_(ArrayVoid& arr, Scalar num){
    DType dtype = arr.dtype();
    if(DTypeFuncs::is_floating(dtype) || DTypeFuncs::is_complex(dtype)){
        arr.execute_function_chunk<WRAP_DTYPES<FloatingTypesL, ComplexTypesL>>([&num](auto begin, auto end){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            value_t val = num.to<value_t>();
            ::nt::mp::fmod_scalar_first(begin, end, val, begin);
        });
    }
    else{
        arr.execute_function_chunk<WRAP_DTYPES<IntegerTypesL>>([&num](auto begin, auto end){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            value_t val = num.to<value_t>();
            for(;begin != end; ++begin){
                *begin = val % *begin;
            }
        });
    }
}

void _fmod_array_(ArrayVoid& arr, const ArrayVoid& other){
    DType dtype = arr.dtype();
    if(DTypeFuncs::is_floating(dtype)){
        arr.execute_function<WRAP_DTYPES<FloatingTypesL, ComplexTypesL>>([](auto begin, auto end, auto begin2){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            ::nt::mp::fmod(begin, end, begin2, begin);
        }, const_cast<ArrayVoid&>(other));
    }
    else{
        arr.execute_function<WRAP_DTYPES<IntegerTypesL>>([](auto begin, auto end, auto begin2){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            for(;begin != end; ++begin, ++begin2){
                *begin = *begin % *begin2;
            }
        }, const_cast<ArrayVoid&>(other));
    }
}


void _fmod_backward(const ArrayVoid& a, const ArrayVoid& b, const ArrayVoid& grad, ArrayVoid& out){
    if(a.dtype() != b.dtype() || b.dtype() != grad.dtype() || grad.dtype() != out.dtype()){
        throw std::logic_error("dtypes must be the same for fmod backward");
    }
    if(!DTypeFuncs::is_floating(a.dtype())){
        throw std::logic_error("Error. dtype for fmod backward must be floating");
    }
    // if(!out.is_contiguous()){
    //     throw std::logic_error("fmod backward output should be contiguous");
    // }
    a.cexecute_function<WRAP_DTYPES<FloatingTypesL>>([&grad, &out](auto begin, auto end, auto begin2){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        const_cast<ArrayVoid&>(grad).execute_function<WRAP_DTYPES<DTypeEnum<DTypeFuncs::type_to_dtype<value_t>>>>([&](auto grad_begin, auto grad_end, auto out_begin){
            ::nt::mp::fmod_backward(begin, end, begin2, grad_begin, out_begin);

        }, out);
    }, b);
}

void _fmod_backward(const Scalar& a, const ArrayVoid& b, const ArrayVoid& grad, ArrayVoid& out){
    if(b.dtype() != grad.dtype() || grad.dtype() != out.dtype()){
        throw std::logic_error("dtypes must be the same for fmod backward");
    }
    if(!DTypeFuncs::is_floating(grad.dtype())){
        throw std::logic_error("Error. dtype for fmod backward must be floating");
    }
    // if(!out.is_contiguous()){
    //     throw std::logic_error("fmod backward output should be contiguous");
    // }
    b.cexecute_function<WRAP_DTYPES<FloatingTypesL>>([&a, &out](auto begin, auto end, auto grad_begin){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        out.execute_function<WRAP_DTYPES<DTypeEnum<DTypeFuncs::type_to_dtype<value_t>>>>([&](auto out_begin, auto out_end){
            ::nt::mp::fmod_backward_scalar(begin, end, grad_begin, out_begin, a.to<value_t>());
        });
    }, grad);
}

void _remainder_(ArrayVoid& arr, Scalar num){
    DType dtype = arr.dtype();
    if(DTypeFuncs::is_floating(dtype) || DTypeFuncs::is_complex(dtype)){
        arr.execute_function_chunk<WRAP_DTYPES<FloatingTypesL, ComplexTypesL>>([&num](auto begin, auto end){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            value_t val = num.to<value_t>();
            ::nt::mp::remainder_scalar_second(begin, end, val, begin);
        });
    }
    else{
        arr.execute_function_chunk<WRAP_DTYPES<IntegerTypesL>>([&num](auto begin, auto end){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            value_t val = num.to<value_t>();
            for(;begin != end; ++begin){
                *begin = *begin % val;
            }
        });
    }
}

void _remainder_first_scalar_(ArrayVoid& arr, Scalar num){
    DType dtype = arr.dtype();
    if(DTypeFuncs::is_floating(dtype) || DTypeFuncs::is_complex(dtype)){
        arr.execute_function_chunk<WRAP_DTYPES<FloatingTypesL, ComplexTypesL>>([&num](auto begin, auto end){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            value_t val = num.to<value_t>();
            ::nt::mp::remainder_scalar_first(begin, end, val, begin);
        });
    }
    else{
        arr.execute_function_chunk<WRAP_DTYPES<IntegerTypesL>>([&num](auto begin, auto end){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            value_t val = num.to<value_t>();
            for(;begin != end; ++begin){
                *begin = val % *begin;
            }
        });
    }
}

void _remainder_array_(ArrayVoid& arr, const ArrayVoid& other){
    DType dtype = arr.dtype();
    if(DTypeFuncs::is_floating(dtype)){
        arr.execute_function<WRAP_DTYPES<FloatingTypesL, ComplexTypesL>>([](auto begin, auto end, auto begin2){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            ::nt::mp::remainder(begin, end, begin2, begin);
        }, const_cast<ArrayVoid&>(other));
    }
    else{
        arr.execute_function<WRAP_DTYPES<IntegerTypesL>>([](auto begin, auto end, auto begin2){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            for(;begin != end; ++begin, ++begin2){
                *begin = *begin % *begin2;
            }
        }, const_cast<ArrayVoid&>(other));
    }
}


void _remainder_backward(const ArrayVoid& a, const ArrayVoid& b, const ArrayVoid& grad, ArrayVoid& out){
    if(a.dtype() != b.dtype() || b.dtype() != grad.dtype() || grad.dtype() != out.dtype()){
        throw std::logic_error("dtypes must be the same for remainder backward");
    }
    if(!(DTypeFuncs::is_floating(a.dtype()) || DTypeFuncs::is_complex(a.dtype()))){
        throw std::logic_error("Error. dtype for remainder backward must be floating or complex");
    }
    // if(!out.is_contiguous()){
    //     throw std::logic_error("remainder backward output should be contiguous");
    // }
    a.cexecute_function<WRAP_DTYPES<FloatingTypesL, ComplexTypesL>>([&grad, &out](auto begin, auto end, auto begin2){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        const_cast<ArrayVoid&>(grad).execute_function<WRAP_DTYPES<DTypeEnum<DTypeFuncs::type_to_dtype<value_t>>>>([&](auto grad_begin, auto grad_end, auto out_begin){
            ::nt::mp::remainder_backward(begin, end, begin2, grad_begin, out_begin);

        }, out);
    }, b);
}

void _remainder_backward(const Scalar& a, const ArrayVoid& b, const ArrayVoid& grad, ArrayVoid& out){
    if(b.dtype() != grad.dtype() || grad.dtype() != out.dtype()){
        throw std::logic_error("dtypes must be the same for remainder backward");
    }
    if(!(DTypeFuncs::is_floating(grad.dtype()) || DTypeFuncs::is_complex(grad.dtype()))){
        throw std::logic_error("Error. dtype for remainder backward must be floating or complex");
    }
    // if(!out.is_contiguous()){
    //     throw std::logic_error("remainder backward output should be contiguous");
    // }
    b.cexecute_function<WRAP_DTYPES<FloatingTypesL, ComplexTypesL>>([&a, &out](auto begin, auto end, auto grad_begin){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        out.execute_function<WRAP_DTYPES<DTypeEnum<DTypeFuncs::type_to_dtype<value_t>>>>([&](auto out_begin, auto out_end){
            ::nt::mp::remainder_backward_scalar(begin, end, grad_begin, out_begin, a.to<value_t>());
        });
    }, grad);
}





} // namespace cpu
} // namespace functional
} // namespace nt
