//this is a group of functions that use simde to accelerate their use
//they accept any iterator type
//obviously, the contiguous one will always be faster
//but the option is still here when a contiguous version is not available
//which is nice

#ifndef _NT_SIMDE_OPS_H_
#define _NT_SIMDE_OPS_H_
#include "simde_traits.h"
#include "simde_traits/simde_traits_iterators.h"
#include <algorithm>
#include <random>

namespace nt{
namespace mp{

//this is generally the only one specified like this
//for a pointer specifically, this is basically just a faster std::memset(begin, end, 0)
inline void fill_zero(void* begin, void* end) noexcept{
	simde_type<int8_t> zero = SimdTraits<int8_t>::zero();
	int8_t* i_begin = reinterpret_cast<int8_t*>(begin);
	int8_t* i_end = reinterpret_cast<int8_t*>(end);
	static constexpr size_t pack_size = pack_size_v<int8_t>;
	for(;i_begin + pack_size <= i_end; i_begin += pack_size){
		SimdTraits<int8_t>::storeu(reinterpret_cast<mask_type*>(i_begin), zero);
	}
	for(;i_begin < i_end; ++i_begin){
		*i_begin = 0;
	}
}

template<typename T>
inline void fill_zero(BucketIterator_blocked<T>& begin, BucketIterator_blocked<T>& end) noexcept {
	if constexpr (simde_supported_v<T>){
	simde_type<T> zero = SimdTraits<T>::zero();
	static constexpr size_t pack_size = pack_size_v<T>;
	for(;begin + pack_size <= end; begin += pack_size){
		it_storeu(begin, zero);
	}
	for(;begin < end; ++begin){
		*begin = 0;
	}
	}else{
		std::fill(begin, end, T(0));	
	}
}
template<typename T>
inline void fill_zero(BucketIterator_list<T>& begin, BucketIterator_list<T>& end) noexcept {
	if constexpr (simde_supported_v<T>){
	simde_type<T> zero = SimdTraits<T>::zero();
	static constexpr size_t pack_size = pack_size_v<T>;
	for(;begin + pack_size <= end; begin += pack_size){
		it_storeu(begin, zero);
	}
	for(;begin < end; ++begin){
		*begin = 0;
	}
	}else{
		std::fill(begin, end, T(0));
	}
}

template<typename T, typename U>
inline void fill(T begin, T end, const U& value){
	using base_type = utils::IteratorBaseType_t<T>;
	static_assert(std::is_same_v<base_type, U>, "Need to be same for fill");
	if constexpr (simde_supported_v<base_type>){
		simde_type<base_type> val = SimdTraits<base_type>::set1(value);
		static constexpr size_t pack_size = pack_size_v<base_type>;
		for(;begin + pack_size <= end; begin += pack_size){
			it_storeu(begin, val);
		}
		for(;begin < end; ++begin)
			*begin = value;
	}else{
		std::fill(begin, end, value);
	}
}



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


template<typename T>
inline void iota(T begin, T end, utils::IteratorBaseType_t<T> value = 0){
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		base_type loading[pack_size];
        utils::IteratorBaseType_t<T> value_cpy = value;
		for(size_t i = 0; i < pack_size; ++i, ++value_cpy){
			loading[i] = value_cpy;
		}
		simde_type<base_type> val;
		simde_type<base_type> add = SimdTraits<base_type>::set1(static_cast<base_type>(pack_size));

		if constexpr (std::is_integral<base_type>::value || std::is_unsigned<base_type>::value){
			val = SimdTraits<base_type>::loadu(reinterpret_cast<const simde_type<base_type>*>(loading));
		}else{
			val = SimdTraits<base_type>::loadu(loading);
		}

		for(;begin + pack_size <= end; begin += pack_size, value += pack_size){
			it_storeu(begin, val);
			val = SimdTraits<base_type>::add(val, add);
		}
		for(;begin < end; ++begin, ++value)
			*begin = value;
	}else{
		std::iota(begin, end, value);
	}	
}


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

////this uses fused multiply add functionality to make this operation much faster
////for operations such as LU and rank reduction
////note to self: make an operation such that it doesn't automatically go into c
//template<typename T, typename U, typename V>
//inline void fused_multiply_add(T begin_a, T end_a, U begin_b, V begin_c){
//	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> > && 
//                    std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<V>>
//                , "Expected to get base types the same for simde optimized routes");
//	using base_type = utils::IteratorBaseType_t<T>;
//	if constexpr (simde_supported_v<base_type>){
//		static constexpr size_t pack_size = pack_size_v<base_type>;
//		for(;begin_a + pack_size <= end_a; begin_a += pack_size, begin_b += pack_size, begin_c += pack_size){
//			simde_type<base_type> a = it_loadu(begin_a);
//			simde_type<base_type> b = it_loadu(begin_b);
//			simde_type<base_type> c = it_loadu(begin_c);
//            SimdTraits<base_type>::fmadd(a, b, c);
//			it_storeu(begin_c, c);
//		}
//		for(; begin_a != end_a; ++begin_a, ++begin_b, ++begin_c)
//			*begin_c += *begin_a * *begin_b;
//	}else{
//		for(; begin_a != end_a; ++begin_a, ++begin_b, ++begin_c)
//			*begin_c += *begin_a * *begin_b;
//	}
    
//}

//template<typename T, typename U>
//inline void fused_multiply_add_scalar(T begin_a, T end_a, U begin_c, utils::IteratorBaseType_t<T> num){
//	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> > 
//                , "Expected to get base types the same for simde optimized routes");
//	using base_type = utils::IteratorBaseType_t<T>;
//	if constexpr (simde_supported_v<base_type>){
//		static constexpr size_t pack_size = pack_size_v<base_type>;
//		simde_type<base_type> nums = SimdTraits<base_type>::set1(num);
//		for(;begin_a + pack_size <= end_a; begin_a += pack_size, begin_c += pack_size){
//			simde_type<base_type> a = it_loadu(begin_a);
//			simde_type<base_type> c = it_loadu(begin_c);
//            SimdTraits<base_type>::fmadd(a, nums, c);
//			it_storeu(begin_c, c);
//		}
//		for(; begin_a != end_a; ++begin_a, ++begin_c)
//			*begin_c += *begin_a * num;
//	}else{
//		for(; begin_a != end_a; ++begin_a, ++begin_c)
//			*begin_c += *begin_a * num;
//	}
    
//}

////this uses fused multiply subtract functionality to make this operation much faster
////for operations such as LU and rank reduction
//template<typename T, typename U, typename V, typename O>
//inline void fused_multiply_subtract(T begin_a, T end_a, U begin_b, V begin_c, O out){
//	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> > && 
//                    std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<V>>
//                , "Expected to get base types the same for simde optimized routes");
//	using base_type = utils::IteratorBaseType_t<T>;
//	if constexpr (simde_supported_v<base_type>){
//		static constexpr size_t pack_size = pack_size_v<base_type>;
//		for(;begin_a + pack_size <= end_a; begin_a += pack_size, begin_b += pack_size, begin_c += pack_size, out += pack_size){
//			simde_type<base_type> a = it_loadu(begin_a);
//			simde_type<base_type> b = it_loadu(begin_b);
//			simde_type<base_type> c = it_loadu(begin_c);
//            simde_type<base_type> o = SimdTraits<base_type>::fmsub(a, b, c);
//			it_storeu(out, o);
//		}
//		for(; begin_a != end_a; ++begin_a, ++begin_b, ++begin_c, ++out)
//			*out = *begin_c - (*begin_a * *begin_b);
//	}else{
//		for(; begin_a != end_a; ++begin_a, ++begin_b, ++begin_c, ++out)
//			*out = *begin_c - (*begin_a * *begin_b);
//	}
    
//}



//template<typename T, typename U, typename O>
//inline void fused_multiply_subtract_scalar(T begin_a, T end_a, U begin_c, O out, utils::IteratorBaseType_t<T> num){
//	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> > && 
//                    std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<O>>
//                , "Expected to get base types the same for simde optimized routes");
//	using base_type = utils::IteratorBaseType_t<T>;
//	if constexpr (simde_supported_v<base_type>){
//		static constexpr size_t pack_size = pack_size_v<base_type>;
//		simde_type<base_type> nums = SimdTraits<base_type>::set1(num);
//		for(;begin_a + pack_size <= end_a; begin_a += pack_size, begin_c += pack_size, out += pack_size){
//			simde_type<base_type> a = it_loadu(begin_a);
//			simde_type<base_type> c = it_loadu(begin_c);
//            simde_type<base_type> o = SimdTraits<base_type>::fmsub(a, nums, c);
//			it_storeu(out, o);
//		}
//		for(; begin_a != end_a; ++begin_a, ++begin_c, ++out)
//			*out = *begin_c - (*begin_a * num);
//	}else{
//		for(; begin_a != end_a; ++begin_a, ++begin_c, ++out)
//			*out = *begin_c - (*begin_a * num);
//	}
    
//}


// #define _NT_SIMDE_OP_TRANSFORM_EQUIVALENT_TWO_(func_name, simde_op, transform_op) \
// template<typename T, typename U, typename O>\
// inline void func_name(T begin, T end, U begin2, O out){\
// 	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> > && std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<O>>, "Expected to get base types the same for simde optimized routes");\
// 	using base_type = utils::IteratorBaseType_t<T>;\
// 	if constexpr (simde_supported_v<base_type>){\
// 		static constexpr size_t pack_size = pack_size_v<base_type>;\
// 		for(; begin + pack_size <= end; begin += pack_size, begin2 += pack_size, out += pack_size){\
// 			simde_type<base_type> a = it_loadu(begin);\
// 			simde_type<base_type> b = it_loadu(begin2);\
// 			simde_type<base_type> c = SimdTraits<base_type>::simde_op(a, b);\
// 			it_storeu(out, c);\
// 		}\
// 		std::transform(begin, end, begin2, out, transform_op<>{});\
// 	}\
// 	else{\
// 		std::transform(begin, end, begin2, out, transform_op<>{});\
// 	}\
// }\
// \



// _NT_SIMDE_OP_TRANSFORM_EQUIVALENT_TWO_(add, add, std::plus);
// _NT_SIMDE_OP_TRANSFORM_EQUIVALENT_TWO_(subtract, subtract, std::minus);
// _NT_SIMDE_OP_TRANSFORM_EQUIVALENT_TWO_(multiply, multiply, std::multiplies);
// _NT_SIMDE_OP_TRANSFORM_EQUIVALENT_TWO_(divide, divide, std::divides);

// #undef _NT_SIMDE_OP_TRANSFORM_EQUIVALENT_TWO_ 



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

_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(reciprical, reciprical, 1.0/ );
_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(exp, exp, std::exp);

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

_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(sqrt, sqrt, std::sqrt);
_NT_MAKE_INV_INLINE_FUNC_(std::sqrt, invsqrt)
_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(invsqrt, invsqrt, _nt_invsqrt);
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
_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(log, log, std::log);


template<typename T, typename U>
inline void sigmoid(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> ones = SimdTraits<base_type>::set1(1.0);
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
		if constexpr (std::is_same_v<Tensor, base_type>){
			for(;begin != end; ++begin, ++out){
				*out = (1 / (1 + std::exp(-(*begin))));
			}
		}else{
			for(;begin != end; ++begin, ++out){
				*out = (base_type(1.0) / (base_type(1.0) + std::exp(-(*begin))));
			}
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
		simde_type<base_type> ones = SimdTraits<base_type>::set1(1.0);
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> current = it_loadu(begin);
			simde_type<base_type> current_m = SimdTraits<base_type>::subtract(ones, current);
					      current = SimdTraits<base_type>::multiply(current, current_m);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out)
			*out = *begin * (base_type(1.0) - *begin);
	}else{
		if constexpr (std::is_same_v<Tensor, base_type>){
			for(;begin != end; ++begin, ++out){
				*out = *begin * (base_type(1.0) - *begin);
			}
		}
		else{
			for(;begin != end; ++begin, ++out){
				*out = *begin * (base_type(1.0) - *begin);
			}
		}
	}
}



template<typename T, typename U>
inline void dtan(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> twos = SimdTraits<base_type>::set1(2.0);
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> current = it_loadu(begin);
					      current = SimdTraits<base_type>::sec(current);
					      current = SimdTraits<base_type>::pow(current, twos);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out)
			*out = (base_type(1.0) / (std::pow(std::cos(*begin), 2)));
	}else{
		if constexpr (std::is_same_v<Tensor, base_type>){
			for(;begin != end; ++begin, ++out){
				*out = (1 / (std::pow(std::cos(*begin), 2)));
			}
	
		}else{
			for(;begin != end; ++begin, ++out){
				*out = (base_type(1.0) / (std::pow(std::cos(*begin), 2)));
			}
		}
	}
}


template<typename T, typename U>
inline void dtanh(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> twos = SimdTraits<base_type>::set1(2.0);
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
inline void dinvsqrt(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> to_pow = SimdTraits<base_type>::set1(3.0);
		simde_type<base_type> to_mult = SimdTraits<base_type>::set1(-0.5);
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

/* template<typename T, typename U, typename O> */
/* inline void add(T begin, T end, U begin2, O out){ */
/* 	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> > && std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<O>>, */ 
/* 			"Expected to get base types the same for simde optimized routes"); */
/* 	using base_type = utils::IteratorBaseType_t<T>; */
/* 	if constexpr (simde_supported_v<base_type>){ */
/* 		static constexpr size_t pack_size = pack_size_v<base_type>; */
/* 		simde_type<base_type> initial = SimdTraits<base_type>::zero(); */
/* 		for(;begin + pack_size <= end; begin += pack_size, begin2 += pack_size){ */
/* 			SimdTraits<base_type>::fmadd(it_loadu(begin), it_loadu(begin2), initial); */
/* 		} */
/* 		init += SimdTraits<base_type::sum(initial); */
/* 		for(;begin < end; ++begin, ++begin2) */
/* 			init += (*begin * *begin2); */
/* 		return init; */
/* 	}else{ */
/* 		std::transform(begin, end, begin2, out, std::plus<>{}) */
/* 	} */	
/* } */


}} //nt::mp::

#endif //_NT_SIMDE_OPS_H_
