#ifndef NT_CONVERT_DTYPE_H__
#define NT_CONVERT_DTYPE_H__

#include "compatible/DType_compatible.h"
#include "DType.h"
#include "DType_enum.h"
#include "compatible/DType_compatible_all.h"
#include <type_traits>

namespace nt{
namespace convert{


template<DType T, typename A, std::enable_if_t<std::is_same_v<::nt::DTypeFuncs::dtype_to_type_t<T>, A>, bool> = true>
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& val){return val;}

template<DType T, typename A, std::enable_if_t<::nt::DTypeFuncs::is_dtype_complex_v<T> && ::nt::DTypeFuncs::is_dtype_real_num_v<::nt::DTypeFuncs::type_to_dtype<A>>, bool> = true>
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& val){
	if(T == DType::Complex32)
		return complex_32(static_cast<float16_t>(val), 0);
	if(T == DType::cdouble)
		return ::nt::DTypeFuncs::dtype_to_type_t<T>(static_cast<double>(val), 0);
	return ::nt::DTypeFuncs::dtype_to_type_t<T>(static_cast<float>(val), 0);
}

template<DType T, typename A, std::enable_if_t<::nt::DTypeFuncs::is_dtype_complex_v<T> && std::is_same_v<A, uint_bool_t>, bool> = true>
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& val){
	if(T == DType::Complex32)
		return complex_32(static_cast<float16_t>(val.value), 0);
	if(T == DType::cdouble)
		return ::nt::DTypeFuncs::dtype_to_type_t<T>(static_cast<double>(val.value), 0);
	return ::nt::DTypeFuncs::dtype_to_type_t<T>(static_cast<float>(val.value), 0);
}


template<DType T, typename A, std::enable_if_t<::nt::DTypeFuncs::is_dtype_complex_v<T> && std::is_same_v<A, bool>, bool> = true>
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& val){
	if(T == DType::Complex32)
		return complex_32(static_cast<float16_t>(val), 0);
	if(T == DType::cdouble)
		return ::nt::DTypeFuncs::dtype_to_type_t<T>(static_cast<double>(val), 0);
	return ::nt::DTypeFuncs::dtype_to_type_t<T>(static_cast<float>(val), 0);
}


template<DType T, typename A, std::enable_if_t<
	::nt::DTypeFuncs::is_dtype_complex_v<T> && ::nt::DTypeFuncs::is_dtype_complex_v<::nt::DTypeFuncs::type_to_dtype<A>> && !std::is_same_v<::nt::DTypeFuncs::dtype_to_type_t<T>, A>, bool> = true>
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& val){
	return ::nt::DTypeFuncs::dtype_to_type_t<T>(val);
}

template<DType T, typename A, std::enable_if_t<::nt::DTypeFuncs::is_dtype_real_num_v<T> && ::nt::DTypeFuncs::is_dtype_real_num_v<::nt::DTypeFuncs::type_to_dtype<A>>, bool> = true>
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& val){
	return static_cast<::nt::DTypeFuncs::dtype_to_type_t<T>>(val);
}

template<DType T, typename A, std::enable_if_t<::nt::DTypeFuncs::is_dtype_real_num_v<T> && ::nt::DTypeFuncs::is_dtype_complex_v<::nt::DTypeFuncs::type_to_dtype<A>>, bool> = true>
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& val){
	return static_cast<::nt::DTypeFuncs::dtype_to_type_t<T>>(val.real());
}

template<DType T, typename A, std::enable_if_t<::nt::DTypeFuncs::is_dtype_real_num_v<T> && std::is_same_v<A, uint_bool_t>, bool> = true>
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& val){
	return static_cast<::nt::DTypeFuncs::dtype_to_type_t<T>>(val.value);
}

template<DType T, typename A, std::enable_if_t<::nt::DTypeFuncs::is_dtype_real_num_v<T> && std::is_same_v<A, bool>, bool> = true>
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& val){
	return static_cast<::nt::DTypeFuncs::dtype_to_type_t<T>>(val);
}

template<typename T, typename A, std::enable_if_t<std::is_same_v<bool, T> && std::is_same_v<uint_bool_t, A>, bool> = true>
inline T convert(const A& val){
	return (val.value == 1);
}

template<DType T, typename A, std::enable_if_t<T == DType::Bool && std::is_same_v<bool, A>, bool> = true>
inline uint_bool_t convert(const A& val){
	return uint_bool_t(val);
}


template<typename T, typename A, std::enable_if_t<std::is_same_v<bool, T> && ::nt::DTypeFuncs::is_dtype_real_num_v<::nt::DTypeFuncs::type_to_dtype<A>>, bool> = true>
inline T convert(const A& val){
	return (val != 0);
}

template<DType T, typename A, std::enable_if_t<T == DType::Bool && ::nt::DTypeFuncs::is_dtype_real_num_v<::nt::DTypeFuncs::type_to_dtype<A>>, bool> = true>
inline uint_bool_t convert(const A& val){
	return uint_bool_t(val != 0);
}

template<typename T, typename A, std::enable_if_t<std::is_same_v<bool, T> && ::nt::DTypeFuncs::is_dtype_complex_v<::nt::DTypeFuncs::type_to_dtype<A>>, bool> = true>
inline T convert(const A& val){
	return (val.real() != 0);
}

template<DType T, typename A, std::enable_if_t<T == DType::Bool && ::nt::DTypeFuncs::is_dtype_complex_v<::nt::DTypeFuncs::type_to_dtype<A>>, bool> = true>
inline uint_bool_t convert(const A& val){
	return uint_bool_t(val.real() != 0);
}

template<DType T, typename A, std::enable_if_t<T == DType::TensorObj, bool> = true>
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& val){
	::nt::DTypeFuncs::dtype_to_type_t<T> outp({1}, ::nt::DTypeFuncs::type_to_dtype<A>);
	outp = val;
	return std::move(outp);
}

/* template<DType T, typename A, std::enable_if_t<std::is_same_v<A, Tensor>, bool> = true> */
/* inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& val){ */
/* 	return val.item<::nt::DTypeFuncs::dtype_to_type_t<T>>(); */
/* } */



template<typename A, std::enable_if_t<::nt::DTypeFuncs::is_dtype_floating_v<::nt::DTypeFuncs::type_to_dtype<A>>, bool> = true>
inline float16_t to_float_16_from_floating(const A& val){
	return static_cast<float16_t>(val);
}

template<typename A, std::enable_if_t<::nt::DTypeFuncs::is_dtype_complex_v<::nt::DTypeFuncs::type_to_dtype<A>>, bool> = true>
inline float16_t to_float_16_from_complex(const A& val){
	return static_cast<float16_t>(val.real());
}

template<typename A, std::enable_if_t<::nt::DTypeFuncs::is_dtype_integer_v<::nt::DTypeFuncs::type_to_dtype<A>>, bool> = true>
inline float16_t to_float_16_from_int(const A& val){
	if constexpr(std::is_same_v<A, uint128_t> || std::is_same_v<A, int128_t>){
		return float16_t(int(val));
	}
	return static_cast<float16_t>(val);
}


template<DType T, typename A, std::enable_if_t<T == DType::Float16, bool> = true>
inline float16_t convert(const A& val){
	if constexpr(::nt::DTypeFuncs::is_dtype_complex_v<::nt::DTypeFuncs::type_to_dtype<A>>)
		return to_float_16_from_complex(val);
	else if constexpr(::nt::DTypeFuncs::is_dtype_floating_v<::nt::DTypeFuncs::type_to_dtype<A>>)
		return to_float_16_from_floating(val);
	else if constexpr(::nt::DTypeFuncs::is_dtype_integer_v<::nt::DTypeFuncs::type_to_dtype<A>>)
		return to_float_16_from_int(val);
	else if constexpr(::nt::DTypeFuncs::type_to_dtype<A> == DType::Bool)
		return float16_t(val.value);
	/* else if constexpr(::nt::DTypeFuncs::type_to_dtype<A> == DType::TensorObj) */
	/* 	return val.item<float16_t>(); */
	return float16_t(0);
}


template<DType T>
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert_float16_to_complex(const float16_t& val){
	if constexpr(T == DType::Complex32)
		return complex_32(val, 0);
	else if constexpr(T == DType::Complex64)
		return complex_64(static_cast<float>(val), 0);
	else if constexpr(T == DType::Complex128)
		return complex_128(static_cast<double>(val), 0);
	return ::nt::DTypeFuncs::dtype_to_type_t<T>(0, 0);
}




template<DType T, typename A, std::enable_if_t<::nt::DTypeFuncs::type_to_dtype<A> == DType::Float16, bool> = true>
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& val){
	if constexpr(::nt::DTypeFuncs::is_dtype_complex_v<T>)
		return convert_float16_to_complex(val);
	else if(::nt::DTypeFuncs::is_dtype_floating_v<T>)
		return static_cast<::nt::DTypeFuncs::dtype_to_type_t<T>>(val);
	else if(::nt::DTypeFuncs::is_dtype_integer_v<::nt::DTypeFuncs::type_to_dtype<A>>){
		if constexpr(T == DType::int128 || T == DType::uint128)
			return ::nt::DTypeFuncs::dtype_to_type_t<T>(static_cast<int>(val));
		return static_cast<::nt::DTypeFuncs::dtype_to_type_t<T>>(val);
	}
	else if(T == DType::Bool)
		return uint_bool_t(val != 0);
	else if(T == DType::TensorObj){
		::nt::DTypeFuncs::dtype_to_type_t<T> output({1}, DType::Float16);
		output = val;
		return std::move(output);
	}
	else
		return ::nt::DTypeFuncs::dtype_to_type_t<T>(0);
}



template<typename A, std::enable_if_t<::nt::DTypeFuncs::is_dtype_floating_v<::nt::DTypeFuncs::type_to_dtype<A>>, bool> = true>
inline float128_t to_float_128_from_floating(const A& val){
	return static_cast<float128_t>(val);
}

template<typename A, std::enable_if_t<::nt::DTypeFuncs::is_dtype_complex_v<::nt::DTypeFuncs::type_to_dtype<A>>, bool> = true>
inline float128_t to_float_128_from_complex(const A& val){
	return static_cast<float128_t>(val.real());
}

template<typename A, std::enable_if_t<::nt::DTypeFuncs::is_dtype_integer_v<::nt::DTypeFuncs::type_to_dtype<A>>, bool> = true>
inline float128_t to_float_128_from_int(const A& val){
	if constexpr(std::is_same_v<A, uint128_t> || std::is_same_v<A, int128_t>){
		return float128_t(val);
	}
	return static_cast<float128_t>(val);
}


template<DType T, typename A, std::enable_if_t<T == DType::Float128, bool> = true>
inline float128_t convert(const A& val){
	if constexpr(::nt::DTypeFuncs::is_dtype_complex_v<::nt::DTypeFuncs::type_to_dtype<A>>)
		return to_float_128_from_complex(val);
	else if constexpr(::nt::DTypeFuncs::is_dtype_floating_v<::nt::DTypeFuncs::type_to_dtype<A>>)
		return to_float_128_from_floating(val);
	else if constexpr(::nt::DTypeFuncs::is_dtype_integer_v<::nt::DTypeFuncs::type_to_dtype<A>>)
		return to_float_128_from_int(val);
	else if constexpr(::nt::DTypeFuncs::type_to_dtype<A> == DType::Bool)
		return float128_t(val.value);
	/* else if constexpr(::nt::DTypeFuncs::type_to_dtype<A> == DType::TensorObj) */
	/* 	return val.item<float128_t>(); */
	return float128_t(0);
}


template<DType T>
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert_float128_to_complex(const float128_t& val){
	if constexpr(T == DType::Complex32)
		return complex_32(static_cast<float16_t>(val), 0);
	else if constexpr(T == DType::Complex64)
		return complex_64(static_cast<float>(val), 0);
	else if constexpr(T == DType::Complex128)
		return complex_128(static_cast<double>(val), 0);
	return ::nt::DTypeFuncs::dtype_to_type_t<T>(0, 0);
}




template<DType T, typename A, std::enable_if_t<::nt::DTypeFuncs::type_to_dtype<A> == DType::Float128, bool> = true>
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& val){
	if constexpr(::nt::DTypeFuncs::is_dtype_complex_v<T>)
		return convert_float128_to_complex(val);
	else if constexpr(::nt::DTypeFuncs::is_dtype_floating_v<T>)
		return static_cast<::nt::DTypeFuncs::dtype_to_type_t<T>>(val);
	else if constexpr(::nt::DTypeFuncs::is_dtype_integer_v<::nt::DTypeFuncs::type_to_dtype<A>>)
		return static_cast<::nt::DTypeFuncs::dtype_to_type_t<T>>(val);
	else if constexpr(T == DType::Bool)
		return uint_bool_t(val != 0);
	else if constexpr(T == DType::TensorObj){
		::nt::DTypeFuncs::dtype_to_type_t<T> output({1}, DType::Float128);
		output = val;
		return std::move(output);
	}
	else
		return ::nt::DTypeFuncs::dtype_to_type_t<T>(0);
}


template<typename A, std::enable_if_t<::nt::DTypeFuncs::is_dtype_floating_v<::nt::DTypeFuncs::type_to_dtype<A>>, bool> = true>
inline int128_t to_int128_from_floating(const A& val){
	return static_cast<int128_t>(val);
}

template<typename A, std::enable_if_t<::nt::DTypeFuncs::is_dtype_complex_v<::nt::DTypeFuncs::type_to_dtype<A>>, bool> = true>
inline int128_t to_int128_from_complex(const A& val){
	return static_cast<int128_t>(val.real());
}

template<typename A, std::enable_if_t<::nt::DTypeFuncs::is_dtype_integer_v<::nt::DTypeFuncs::type_to_dtype<A>>, bool> = true>
inline int128_t to_int128_from_int(const A& val){
	return static_cast<int128_t>(val);
}

template<typename A, std::enable_if_t<::nt::DTypeFuncs::is_dtype_floating_v<::nt::DTypeFuncs::type_to_dtype<A>>, bool> = true>
inline uint128_t to_uint128_from_floating(const A& val){
	return static_cast<uint128_t>(val);
}

template<typename A, std::enable_if_t<::nt::DTypeFuncs::is_dtype_complex_v<::nt::DTypeFuncs::type_to_dtype<A>>, bool> = true>
inline uint128_t to_uint128_from_complex(const A& val){
	return static_cast<uint128_t>(val.real());
}

template<typename A, std::enable_if_t<::nt::DTypeFuncs::is_dtype_integer_v<::nt::DTypeFuncs::type_to_dtype<A>>, bool> = true>
inline uint128_t to_uint128_from_int(const A& val){
	return static_cast<uint128_t>(val);
}


template<DType T, typename A, std::enable_if_t<T == DType::int128, bool> = true>
inline int128_t convert(const A& val){
	if constexpr(::nt::DTypeFuncs::is_dtype_complex_v<::nt::DTypeFuncs::type_to_dtype<A>>)
		return to_int128_from_complex(val);
	else if constexpr(::nt::DTypeFuncs::is_dtype_floating_v<::nt::DTypeFuncs::type_to_dtype<A>>)
		return to_int128_from_floating(val);
	else if constexpr(::nt::DTypeFuncs::is_dtype_integer_v<::nt::DTypeFuncs::type_to_dtype<A>>)
		return to_int128_from_int(val);
	else if constexpr(::nt::DTypeFuncs::type_to_dtype<A> == DType::Bool)
		return int128_t(val.value);
	/* else if constexpr(::nt::DTypeFuncs::type_to_dtype<A> == DType::TensorObj) */
	/* 	return val.item<int128_t>(); */
	return int128_t(0);
}


template<DType T, typename A, std::enable_if_t<T == DType::uint128, bool> = true>
inline uint128_t convert(const A& val){
	if constexpr(::nt::DTypeFuncs::is_dtype_complex_v<::nt::DTypeFuncs::type_to_dtype<A>>)
		return to_uint128_from_complex(val);
	else if constexpr(::nt::DTypeFuncs::is_dtype_floating_v<::nt::DTypeFuncs::type_to_dtype<A>>)
		return to_uint128_from_floating(val);
	else if constexpr(::nt::DTypeFuncs::is_dtype_integer_v<::nt::DTypeFuncs::type_to_dtype<A>>)
		return to_uint128_from_int(val);
	else if constexpr(::nt::DTypeFuncs::type_to_dtype<A> == DType::Bool)
		return uint128_t(val.value);
	/* else if constexpr(::nt::DTypeFuncs::type_to_dtype<A> == DType::TensorObj) */
	/* 	return val.item<uint128_t>(); */
	return uint128_t(0);
}


template<DType T>
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert_int128_to_complex(const int128_t& val){
	if constexpr(T == DType::Complex32)
		return complex_32(static_cast<float16_t>(val), 0);
	else if constexpr(T == DType::Complex64)
		return complex_64(static_cast<float>(val), 0);
	else if constexpr(T == DType::Complex128)
		return complex_128(static_cast<double>(val), 0);
	return ::nt::DTypeFuncs::dtype_to_type_t<T>(0, 0);
}


template<DType T>
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert_uint128_to_complex(const uint128_t& val){
	if constexpr(T == DType::Complex32)
		return complex_32(static_cast<float16_t>(val), 0);
	else if constexpr(T == DType::Complex64)
		return complex_64(static_cast<float>(val), 0);
	else if constexpr(T == DType::Complex128)
		return complex_128(static_cast<double>(val), 0);
	return ::nt::DTypeFuncs::dtype_to_type_t<T>(0, 0);
}



template<DType T, typename A, std::enable_if_t<::nt::DTypeFuncs::type_to_dtype<A> == DType::int128, bool> = true>
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& val){
	if constexpr(::nt::DTypeFuncs::is_dtype_complex_v<T>)
		return convert_int128_to_complex(val);
	else if constexpr(::nt::DTypeFuncs::is_dtype_floating_v<T>)
		return static_cast<::nt::DTypeFuncs::dtype_to_type_t<T>>(val);
	else if constexpr(::nt::DTypeFuncs::is_dtype_integer_v<::nt::DTypeFuncs::type_to_dtype<A>>)
		return static_cast<::nt::DTypeFuncs::dtype_to_type_t<T>>(val);
	else if constexpr(T == DType::Bool)
		return uint_bool_t(val > 0);
	else if constexpr(T == DType::TensorObj){
		::nt::DTypeFuncs::dtype_to_type_t<T> output({1}, DType::int128);
		output = val;
		return std::move(output);
	}
	else
		return ::nt::DTypeFuncs::dtype_to_type_t<T>(0);
}

template<DType T, typename A, std::enable_if_t<::nt::DTypeFuncs::type_to_dtype<A> == DType::uint128, bool> = true>
inline ::nt::DTypeFuncs::dtype_to_type_t<T> convert(const A& val){
	if constexpr(::nt::DTypeFuncs::is_dtype_complex_v<T>)
		return convert_uint128_to_complex(val);
	else if constexpr(::nt::DTypeFuncs::is_dtype_floating_v<T>)
		return static_cast<::nt::DTypeFuncs::dtype_to_type_t<T>>(val);
	else if constexpr(::nt::DTypeFuncs::is_dtype_integer_v<::nt::DTypeFuncs::type_to_dtype<A>>)
		return static_cast<::nt::DTypeFuncs::dtype_to_type_t<T>>(val);
	else if constexpr(T == DType::Bool)
		return uint_bool_t(val > 0);
	else if constexpr(T == DType::TensorObj){
		::nt::DTypeFuncs::dtype_to_type_t<T> output({1}, DType::uint128);
		output = val;
		return std::move(output);
	}
	else
		return ::nt::DTypeFuncs::dtype_to_type_t<T>(0);
}


template<typename T, typename A>
inline T convert(const A& val){return convert<::nt::DTypeFuncs::type_to_dtype<T>>(val);}

}
}

#endif
