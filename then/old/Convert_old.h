#ifndef _MY_CONVERT_DTYPE_H
#define _MY_CONVERT_DTYPE_H

#include "compatible/DType_compatible.h"
#include "DType.h"
#include "DType_enum.h"
#include "compatible/DType_compatible_all.h"
#include <type_traits>

namespace nt{
namespace convert{

template<DType T, typename A, std::enable_if_t<std::is_same_v<::nt::DTypeFuncs::dtype_to_type_t<T>, A>, bool> = true>
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& v){return v;}

template<DType T, typename A, std::enable_if_t<
	::nt::DTypeFuncs::is_dtype_complex_v<T> && ::nt::DTypeFuncs::is_dtype_complex_v<::nt::DTypeFuncs::type_to_dtype<A>>
	&& !std::is_same_v<::nt::DTypeFuncs::dtype_to_type_t<T>, A>
	, bool> = true>
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& v){
	return ::nt::DTypeFuncs::dtype_to_type_t<T>(v); 
}

template<DType T, typename A, std::enable_if_t<
	::nt::DTypeFuncs::is_dtype_floating_v<T> && ::nt::DTypeFuncs::is_dtype_floating_v<::nt::DTypeFuncs::type_to_dtype<A>>
	&& !std::is_same_v<::nt::DTypeFuncs::dtype_to_type_t<T>, A>
	, bool> = true> 
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& v){
	return static_cast<::nt::DTypeFuncs::dtype_to_type_t<T>>(v); 
}

template<DType T, typename A, std::enable_if_t<
		::nt::DTypeFuncs::is_dtype_integer_v<T> && ::nt::DTypeFuncs::is_dtype_integer_v<::nt::DTypeFuncs::type_to_dtype<A>>
		&& !std::is_same_v<::nt::DTypeFuncs::dtype_to_type_t<T>, A>
		, bool> = true> 
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& v){
	return static_cast<::nt::DTypeFuncs::dtype_to_type_t<T>>(v); 
}

template<DType T, typename A, std::enable_if_t<
		::nt::DTypeFuncs::is_dtype_floating_v<T> && ::nt::DTypeFuncs::is_dtype_integer_v<::nt::DTypeFuncs::type_to_dtype<A>>
		&& !std::is_same_v<::nt::DTypeFuncs::dtype_to_type_t<T>, A>
		, bool> = true> 
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& v){
#if defined(_HALF_FLOAT_SUPPORT_) and defined(__SIZEOF_INT128__)
	if constexpr (T == DType::Float16 && (std::is_same_v<A, uint128_t> || std::is_same_v<A, int128_t>)){
		return static_cast<float16_t>(float(v));
	}else{
		return ::nt::DTypeFuncs::dtype_to_type_t<T>(v); 
	}
#else
	return ::nt::DTypeFuncs::dtype_to_type_t<T>(v);
#endif
}

template<DType T, typename A, std::enable_if_t<
		::nt::DTypeFuncs::is_dtype_integer_v<T> && ::nt::DTypeFuncs::is_dtype_floating_v<::nt::DTypeFuncs::type_to_dtype<A>>
		&& !std::is_same_v<::nt::DTypeFuncs::dtype_to_type_t<T>, A>
		, bool> = true> 
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& v){
#if defined(_HALF_FLOAT_SUPPORT_) and defined(__SIZEOF_INT128__)
	if constexpr ((T == DType::uint128 || T == DType::int128) && std::is_same_v<A, float16_t>){
		return static_cast<::nt::DTypeFuncs::dtype_to_type_t<T>>(static_cast<float>(v));
	}
	else{
		return static_cast<::nt::DTypeFuncs::dtype_to_type_t<T>>(v); 
	}
#else
	return static_cast<::nt::DTypeFuncs::dtype_to_type_t<T>>(v); 
#endif
}

template<DType T, typename A, std::enable_if_t<
		::nt::DTypeFuncs::is_dtype_complex_v<T> && ::nt::DTypeFuncs::is_dtype_real_num_v<::nt::DTypeFuncs::type_to_dtype<A>>
		&& !std::is_same_v<::nt::DTypeFuncs::dtype_to_type_t<T>, A>
		, bool> = true> 
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& v){
	if constexpr(T == DType::Complex64){
		return complex_64(convert<DType::Float>(v), 0);
		
	}
	else if constexpr(T == DType::Complex128){
		return complex_128(convert<DType::Double>(v), 0);
		
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(T == DType::Complex32){
		return complex_32(convert<DType::Float16>(v), 0);
	}
#endif
}


template<DType T, typename A, std::enable_if_t<
		::nt::DTypeFuncs::is_dtype_real_num_v<T> && ::nt::DTypeFuncs::is_dtype_complex_v<::nt::DTypeFuncs::type_to_dtype<A>>
		&& !std::is_same_v<::nt::DTypeFuncs::dtype_to_type_t<T>, A>
		, bool> = true> 
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& v){
	return convert<T>(v.real());
}

template<DType T, typename A, std::enable_if_t<
			T == DType::TensorObj && ::nt::DTypeFuncs::type_to_dtype<A> != DType::TensorObj
			&& !std::is_same_v<::nt::DTypeFuncs::dtype_to_type_t<T>, A>
			, bool> = true> 
inline Tensor convert(const A& v){
	Tensor output({1}, ::nt::DTypeFuncs::type_to_dtype<A>);
	output = v;
	return std::move(output);
}

template<DType T, typename A, std::enable_if_t<
		T != DType::TensorObj && std::is_same_v<A, Tensor>
		&& !std::is_same_v<::nt::DTypeFuncs::dtype_to_type_t<T>, A>
		, bool> = true>
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& v){
	Scalar s = v.toScalar();
	return s.to<::nt::DTypeFuncs::dtype_to_type_t<T> >();
}

template<DType T, typename A, std::enable_if_t<
		T == DType::Bool && ::nt::DTypeFuncs::is_dtype_real_num_v<::nt::DTypeFuncs::type_to_dtype<A>>
		&& !std::is_same_v<::nt::DTypeFuncs::dtype_to_type_t<T>, A>
		, bool> = true>
inline uint_bool_t convert(const A& v){
	return uint_bool_t(v == 1);
}


template<DType T, typename A, std::enable_if_t<
		T == DType::Bool && ::nt::DTypeFuncs::is_dtype_complex_v<::nt::DTypeFuncs::type_to_dtype<A>>
		&& !std::is_same_v<::nt::DTypeFuncs::dtype_to_type_t<T>, A>
		, bool> = true>
inline uint_bool_t convert(const A& v){
	return uint_bool_t(v.real() == 1);
}

template<DType T, typename A, std::enable_if_t<
		::nt::DTypeFuncs::is_dtype_num_v<T> && std::is_same_v<A, uint_bool_t>
		&& !std::is_same_v<::nt::DTypeFuncs::dtype_to_type_t<T>, A>
		, bool> = true>
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& v){
	return convert<T>(v.value);
}


template<typename T, typename A>
inline T convert(const A& val){return convert<::nt::DTypeFuncs::type_to_dtype<T>>(val);}



}
}

#endif
