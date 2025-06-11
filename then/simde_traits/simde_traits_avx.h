//AVX only traits for using simde
//DONE
#ifndef _NT_SIMDE_TRAITS_AVX_H_
#define _NT_SIMDE_TRAITS_AVX_H_
#include "../../types/Types.h"
#include <simde/x86/avx.h>
#include <simde/x86/fma.h>  // only for FMA if supported
#include <simde/x86/sse.h>  
#include <simde/x86/f16c.h>
#include <simde/x86/svml.h> // for functions such as simde_mm_exp_ps 
#include <cstddef>
#include <cstddef>
#include <type_traits>



namespace nt{
namespace mp{


//types that are supported for avx instructions
template<typename T>
struct simde_supported_avx{
	static constexpr bool value = false;
};

template<>
struct simde_supported_avx<float>{
	static constexpr bool value = true;
};

template<>
struct simde_supported_avx<double>{
	static constexpr bool value = true;
};


template<>
struct simde_supported_avx<uint8_t>{
	static constexpr bool value = true;
};

template<>
struct simde_supported_avx<int8_t>{
	static constexpr bool value = true;
};

template<>
struct simde_supported_avx<uint16_t>{
	static constexpr bool value = true;
};

template<>
struct simde_supported_avx<int16_t>{
	static constexpr bool value = true;
};

template<>
struct simde_supported_avx<uint32_t>{
	static constexpr bool value = true;
};

template<>
struct simde_supported_avx<int32_t>{
	static constexpr bool value = true;
};

template<>
struct simde_supported_avx<uint64_t>{
	static constexpr bool value = true;
};

template<>
struct simde_supported_avx<int64_t>{
	static constexpr bool value = true;
};



template<>
struct simde_supported_avx<float16_t >{
	static constexpr bool value = true;
};




template<>
struct simde_supported_avx<complex_128>{
	static constexpr bool value = true;
};

template<>
struct simde_supported_avx<complex_64>{
	static constexpr bool value = true;
};


template<>
struct simde_supported_avx<complex_32>{
	static constexpr bool value = true;
};


//these are the types that are specifically supported by the svml header file
//these are going to be the cos, sin, tan, exp, pow, etc functions
//its going to basically just be the floating point numbers (including complex types
template<typename T>
struct simde_svml_supported_avx{
	static constexpr bool value = false;
};

template<>
struct simde_svml_supported_avx<float>{
	static constexpr bool value = true;
};

template<>
struct simde_svml_supported_avx<double>{
	static constexpr bool value = true;
};


template<>
struct simde_svml_supported_avx<float16_t>{
	static constexpr bool value = true;
};

template<>
struct simde_svml_supported_avx<complex_64>{
	static constexpr bool value = true;
};

template<>
struct simde_svml_supported_avx<complex_32>{
	static constexpr bool value = true;
};

template<>
struct simde_svml_supported_avx<complex_128>{
	static constexpr bool value = true;
};



template<typename T>
inline constexpr bool simde_svml_supported_avx_v = simde_svml_supported_avx<T>::value;

template<typename T>
inline constexpr bool simde_supported_avx_v = simde_supported_avx<T>::value;

template<typename T>
struct SimdTraits_avx;

//float traits
template <>
struct SimdTraits_avx<float> {
	using Type = simde__m128;
	static constexpr size_t pack_size = 4;
	static constexpr auto load = simde_mm_load_ps;
	static constexpr auto loadu = simde_mm_loadu_ps;
	static constexpr auto load_masked = simde_mm_maskload_ps;
	static constexpr auto set = simde_mm_set_ps;
	static constexpr auto set1 = simde_mm_set_ps1;
	static constexpr auto broadcast = simde_mm_broadcast_ss;
	static constexpr auto store = simde_mm_store_ps;
	static constexpr auto storeu = simde_mm_storeu_ps;
	static constexpr auto store_masked = simde_mm_maskstore_ps;
	static constexpr auto zero = simde_mm_setzero_ps;
	//svml exponent functions
	static constexpr auto reciprical = simde_mm_rcp_ps;
	static constexpr auto exp = simde_mm_exp_ps;
	static constexpr auto pow = simde_mm_pow_ps;
	static constexpr auto sqrt = simde_mm_sqrt_ps;
	static constexpr auto invsqrt = simde_mm_invsqrt_ps;

        //trig svml functions
	static constexpr auto tanh = simde_mm_tanh_ps;
	static constexpr auto tan = simde_mm_tan_ps;
	static constexpr auto atanh = simde_mm_atanh_ps;
	static constexpr auto atan = simde_mm_atan_ps;
	inline static constexpr auto cotanh = [](const Type& a) noexcept -> Type { return reciprical(tanh(a));};
	inline static constexpr auto cotan = [](const Type& a) noexcept -> Type { return reciprical(tan(a));};
	static constexpr auto sinh = simde_mm_sinh_ps;
	static constexpr auto sin = simde_mm_sin_ps;
	static constexpr auto asinh = simde_mm_asinh_ps;
	static constexpr auto asin = simde_mm_asin_ps;
	inline static constexpr auto csch = [](const Type& a) noexcept -> Type { return reciprical(sinh(a));};
	inline static constexpr auto csc = [](const Type& a) noexcept -> Type { return reciprical(sin(a));};
	static constexpr auto cosh = simde_mm_cosh_ps;
	static constexpr auto cos = simde_mm_cos_ps;
	static constexpr auto acosh = simde_mm_acosh_ps;
	static constexpr auto acos = simde_mm_acos_ps;
	inline static constexpr auto sech = [](const Type& a) noexcept -> Type { return reciprical(cosh(a));};
	inline static constexpr auto sec = [](const Type& a) noexcept -> Type { return reciprical(cos(a));};
	static constexpr auto log = simde_mm_log_ps;

	
	static constexpr auto subtract = simde_mm_sub_ps;
	static constexpr auto divide = simde_mm_div_ps;
	static constexpr auto add = simde_mm_add_ps;
	static constexpr auto multiply = simde_mm_mul_ps;
	/* inline static constexpr auto modulo = [](const Type& dividend_c, const Type& divisor_c ) -> Type{ //dividend % divisor */
	/* 	simde__m128i dividend = simde_mm_cvtps_epi32(dividend_c); */
	/* 	simde__m128i divisor = simde_mm_cvtps_epi32(divisor); */
	/* 	return simde_mm_cvtepi32_ps(simde_mm_rem_epi32(dividend, divisor)); */
	/* }; */
	inline static constexpr auto negative = [](const Type& a) noexcept -> Type{
		return subtract(zero(), a);
	}; //returns negative value of current type
    
    static constexpr auto fmsub = simde_mm_fmsub_ps;
	inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c){
#if defined(__FMA__) || defined(SIMDE_X86_FMA)
		c = simde_mm_fmadd_ps(a, b, c);
#else
		c = simde_mm_add_ps(simde_mm_mul_ps(a,b),c);
#endif //defined(__FMA__) || defined(SIMDE_X86_FMA)
	};
	inline static constexpr auto sum = [](const Type& x) -> float{
#ifdef SIMDE_ARCH_SSE3
		simde__m128 hadd = simde_mm_hadd_ps(x, x);  // Horizontal add: vec[0] + vec[1], vec[2] + vec[3]
		hadd = simde_mm_hadd_ps(hadd, hadd);       // Add the two results together
		return simde_mm_cvtss_f32(hadd);           // Extract the sum from the resulting __m128
#else
		// x = ( x3, x2, x1, x0 )
		// loDual = ( -, -, x1, x0 )
		const simde__m128 loDual = x;
		// hiDual = ( -, -, x3, x2)
		const simde__m128 hiDual = simde_mm_movehl_ps(x, x);
		// sumDual = ( -, -, x1 + x3, x0 + x2)
		const simde__m128 sumDual = simde_mm_add_ps(loDual, hiDual);
		// lo = ( -, -, -, x0 + x2 )
		const simde__m128 lo = sumDual;
		// hi = ( -, -, -, x1 + x3 )
		const simde__m128 hi = simde_mm_shuffle_ps(sumDual, sumDual, 0x1);
		// sum = ( -, -, -, x0 + x1 + x2 + x3 )
		const simde__m128 sum = simde_mm_add_ss(lo, hi);
		return simde_mm_cvtss_f32(sum);

#endif
	};
};


//double traits
template <>
struct SimdTraits_avx<double> {
	using Type = simde__m128d;
	static constexpr size_t pack_size = 2;
	static constexpr auto load = simde_mm_load_pd;
	static constexpr auto loadu = simde_mm_loadu_pd;
	static constexpr auto load_masked = simde_mm_maskload_pd;
	static constexpr auto set = simde_mm_set_pd;
	static constexpr auto set1 = simde_mm_set1_pd;
	inline static constexpr auto broadcast = [](const double* arr) { return set1(*arr);};
	static constexpr auto store = simde_mm_store_pd;
	static constexpr auto storeu = simde_mm_storeu_pd;
	static constexpr auto store_masked = simde_mm_maskstore_pd;
	static constexpr auto zero = simde_mm_setzero_pd;
	static constexpr auto divide = simde_mm_div_pd;

	//svml exponent functions
	inline static constexpr auto reciprical = [](Type a){return divide(set1(1.0), a);};
	static constexpr auto exp = simde_mm_exp_pd;
	static constexpr auto pow = simde_mm_pow_pd;
	static constexpr auto sqrt = simde_mm_sqrt_pd;
	static constexpr auto invsqrt = simde_mm_invsqrt_pd;

        //trig svml functions
	static constexpr auto tanh = simde_mm_tanh_pd;
	static constexpr auto tan = simde_mm_tan_pd;
	static constexpr auto atanh = simde_mm_atanh_pd;
	static constexpr auto atan = simde_mm_atan_pd;
	inline static constexpr auto cotanh = [](const Type& a) noexcept -> Type { return reciprical(tanh(a));};
	inline static constexpr auto cotan = [](const Type& a) noexcept -> Type { return reciprical(tan(a));};
	static constexpr auto sinh = simde_mm_sinh_pd;
	static constexpr auto sin = simde_mm_sin_pd;
	static constexpr auto asinh = simde_mm_asinh_pd;
	static constexpr auto asin = simde_mm_asin_pd;
	inline static constexpr auto csch = [](const Type& a) noexcept -> Type { return reciprical(sinh(a));};
	inline static constexpr auto csc = [](const Type& a) noexcept -> Type { return reciprical(sin(a));};
	static constexpr auto cosh = simde_mm_cosh_pd;
	static constexpr auto cos = simde_mm_cos_pd;
	static constexpr auto acosh = simde_mm_acosh_pd;
	static constexpr auto acos = simde_mm_acos_pd;
	inline static constexpr auto sech = [](const Type& a) noexcept -> Type { return reciprical(cosh(a));};
	inline static constexpr auto sec = [](const Type& a) noexcept -> Type { return reciprical(cos(a));};
	static constexpr auto log = simde_mm_log_pd;

	static constexpr auto subtract = simde_mm_sub_pd;
	static constexpr auto add = simde_mm_add_pd;
	static constexpr auto multiply = simde_mm_mul_pd;
	/* inline static constexpr auto modulo = [](const Type& dividend_c, const Type& divisor_c ) -> Type{ //dividend % divisor */
	/* 	simde__m128i dividend = simde_mm_cvtpd_epi64(dividend_c); */
	/* 	simde__m128i divisor = simde_mm_cvtpd_epi64(divisor); */
	/* 	return simde_mm_cvtepi64_pd(simde_mm_rem_epi64(dividend, divisor)); */
	/* }; */
	inline static constexpr auto negative = [](const Type& a) noexcept -> Type{
		return subtract(zero(), a);
	}; //returns negative value of current type

    static constexpr auto fmsub = simde_mm_fmsub_pd;
	inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c){
#if defined(__FMA__) || defined(SIMDE_X86_FMA)
		c = simde_mm_fmadd_pd(a, b, c);
#else
		c = simde_mm_add_pd(simde_mm_mul_pd(a,b),c);
#endif
	};
	inline static constexpr auto sum = [](const Type& x) -> double {
		//x is low
		simde__m128d high64 = simde_mm_unpackhi_pd(x, x);
		return simde_mm_cvtsd_f64(simde_mm_add_sd(high64, x));
	};

};

//int64_t traits
//%s/fmadd = [](const Type& a, const Type& b, Type& c)/fmadd = \[\](const Type\&\& a, const Type\&\& b, Type\& c)/g
template <>
struct SimdTraits_avx<int64_t> {
    using Type = simde__m128i;
    static constexpr size_t pack_size = 2;
    static constexpr auto load = simde_mm_load_si128;
    static constexpr auto loadu = simde_mm_loadu_si128;
    inline static constexpr auto load_masked = [](const int64_t* data, const simde__m128i& mask) noexcept -> Type{
	simde__m128i loaded_data = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data));
	simde__m128i result = simde_mm_and_si128(loaded_data, mask);
	return result;
    };    static constexpr auto set = simde_mm_set_epi64x;
    static constexpr auto set1 = simde_mm_set1_epi64x;
    static constexpr auto store = simde_mm_store_si128;
    static constexpr auto storeu = simde_mm_storeu_si128;
    inline static constexpr auto store_masked = [](int64_t* data, const simde__m128i& mask_data, const simde__m128i& vector){
	simde__m128i data_vector = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data));
	simde__m128i result_data = simde_mm_and_si128(data_vector, simde_mm_xor_si128(mask_data, simde_mm_cmpeq_epi64(mask_data,mask_data)));
	simde__m128i result_vector = simde_mm_and_si128(vector, mask_data);
	             result_data = simde_mm_add_epi64(result_data, result_vector);
	simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(data), result_data);
    };
    static constexpr auto zero = simde_mm_setzero_si128;
    static constexpr auto subtract = simde_mm_sub_epi64;
    static constexpr auto divide = simde_mm_div_epi64;
    static constexpr auto add = simde_mm_add_epi64;
    /* static constexpr auto modulo = simde_mm_rem_epi64; */
    inline static constexpr auto multiply = [](const Type& a, const Type& b) -> Type{
	return simde_mm_set_epi64x(simde_mm_extract_epi64(a, 0) * simde_mm_extract_epi64(b, 0), simde_mm_extract_epi64(a, 1) * simde_mm_extract_epi64(b, 1));
    };
	inline static constexpr auto negative = [](const Type& a) noexcept -> Type{
		return subtract(zero(), a);
	}; //returns negative value of current type
    inline static constexpr auto broadcast = [](const int64_t* arr) -> Type {return simde_mm_set1_epi64x(*arr);};
    inline static constexpr auto fmsub = [](const Type& a, const Type& b, Type& c) -> Type {return subtract(c, multiply(a, b));};
    inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c){
	c = simde_mm_add_epi64(SimdTraits_avx<int64_t>::multiply(a, b), c);
    };
    inline static constexpr auto sum = [](const Type& x) -> int64_t {
	return simde_mm_extract_epi64(x, 0) + simde_mm_extract_epi64(x, 1);
    };

};


//uint64_t traits
template <>
struct SimdTraits_avx<uint64_t> {
    using Type = simde__m128i;
    static constexpr size_t pack_size = 2;
    static constexpr auto load = simde_mm_load_si128;
    static constexpr auto loadu = simde_mm_loadu_si128;
    inline static constexpr auto load_masked = [](const uint64_t* data, const simde__m128i& mask) noexcept -> Type{
	simde__m128i loaded_data = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data));
	simde__m128i result = simde_mm_and_si128(loaded_data, mask);
	return result;
    };    static constexpr auto set = simde_mm_set_epi64x;
    static constexpr auto set1 = simde_mm_set_epi64x;
    static constexpr auto store = simde_mm_store_si128;
    static constexpr auto storeu = simde_mm_storeu_si128;
    inline static constexpr auto store_masked = [](uint64_t* data, const simde__m128i& mask_data, const simde__m128i& vector){
	simde__m128i data_vector = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data));
	simde__m128i result_data = simde_mm_and_si128(data_vector, simde_mm_xor_si128(mask_data, simde_mm_cmpeq_epi64(mask_data,mask_data)));
	simde__m128i result_vector = simde_mm_and_si128(vector, mask_data);
	             result_data = simde_mm_add_epi64(result_data, result_vector);
	simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(data), result_data);
    };
    static constexpr auto zero = simde_mm_setzero_si128;
    static constexpr auto subtract = simde_mm_sub_epi64;
    static constexpr auto divide = simde_mm_div_epu64;
    static constexpr auto add = simde_mm_add_epi64;
    inline static constexpr auto multiply = [](const Type& a, const Type& b) -> Type{
	uint64_t res_a[2];
	uint64_t res_b[2];
	simde_mm_storeu_si128((simde__m128i*)res_a, a);
	simde_mm_storeu_si128((simde__m128i*)res_b, b);
	return simde_mm_set_epi64x(res_a[0] * res_b[0], res_a[1] * res_b[1]);
    };
   
    /* static constexpr auto modulo = simde_mm_rem_epu64; */
    inline static constexpr auto broadcast = [](const uint64_t* arr) noexcept -> Type {return simde_mm_set1_epi64x(*arr);};
    inline static constexpr auto fmsub = [](const Type& a, const Type& b, Type& c) -> Type {return subtract(c, multiply(a, b));};
    inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c) noexcept {
	c = simde_mm_add_epi64(SimdTraits_avx<uint64_t>::multiply(a, b), c);
    };
    inline static constexpr auto sum = [](const Type& x) -> int64_t {
	return simde_mm_extract_epi64(x, 0) + simde_mm_extract_epi64(x, 1);
    };

};


//int32_t traits
template <>
struct SimdTraits_avx<int32_t> {
    using Type = simde__m128i;
    static constexpr size_t pack_size = 4;
    static constexpr auto load = simde_mm_load_si128;
    static constexpr auto loadu = simde_mm_loadu_si128;
    inline static constexpr auto load_masked = [](const int32_t* data, const simde__m128i& mask) noexcept -> Type{
	simde__m128i loaded_data = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data));
	simde__m128i result = simde_mm_and_si128(loaded_data, mask);
	return result;
    };
    static constexpr auto set = simde_mm_set_epi32;
    static constexpr auto set1 = simde_mm_set1_epi32;
    static constexpr auto store = simde_mm_store_si128;
    static constexpr auto storeu = simde_mm_storeu_si128;
    inline static constexpr auto store_masked = [](int32_t* data, const simde__m128i& mask_data, const simde__m128i& vector){
	simde__m128i data_vector = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data));
	simde__m128i result_data = simde_mm_and_si128(data_vector, simde_mm_xor_si128(mask_data, simde_mm_cmpeq_epi32(mask_data,mask_data)));
	simde__m128i result_vector = simde_mm_and_si128(vector, mask_data);
	             result_data = simde_mm_add_epi32(result_data, result_vector);
	simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(data), result_data);
    };
    static constexpr auto zero = simde_mm_setzero_si128;
    static constexpr auto subtract = simde_mm_sub_epi32;
    static constexpr auto divide = simde_mm_div_epi32;
    static constexpr auto add = simde_mm_add_epi32;
    static constexpr auto multiply = simde_mm_mullo_epi32;
    /* static constexpr auto modulo = simde_mm_rem_epi32; */

	inline static constexpr auto negative = [](const Type& a) noexcept -> Type{
		return subtract(zero(), a);
	}; //returns negative value of current type
 
	inline static constexpr auto broadcast = [](const int32_t* arr) -> Type {return SimdTraits_avx<int32_t>::set1(*arr);};
    inline static constexpr auto fmsub = [](const Type& a, const Type& b, Type& c) -> Type {return subtract(c, multiply(a, b));};
    inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c){
	c = simde_mm_add_epi32(simde_mm_mullo_epi32(a, b), c);
    };
    inline static constexpr auto sum = [](const Type& x) -> int32_t{
	simde__m128i sum64 = simde_mm_hadd_epi32(x, x);
	return simde_mm_extract_epi32(sum64, 0) + simde_mm_extract_epi32(sum64, 1);
    };

};


//uint32_t traits
template <>
struct SimdTraits_avx<uint32_t> {
    using Type = simde__m128i;
    static constexpr size_t pack_size = 4;
    static constexpr auto load = simde_mm_load_si128;
    static constexpr auto loadu = simde_mm_loadu_si128;
    inline static constexpr auto load_masked = [](const void* data, const simde__m128i& mask) noexcept -> Type{
	simde__m128i loaded_data = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data));
	simde__m128i result = simde_mm_and_si128(loaded_data, mask);
	return result;
    };
    static constexpr auto set = simde_mm_set_epi32;
    static constexpr auto set1 = simde_mm_set1_epi32;
    static constexpr auto store = simde_mm_store_si128;
    static constexpr auto storeu = simde_mm_storeu_si128;
    inline static constexpr auto store_masked = [](void* data, const simde__m128i& mask_data, const simde__m128i& vector){
	simde__m128i data_vector = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data));
	simde__m128i result_data = simde_mm_and_si128(data_vector, simde_mm_xor_si128(mask_data, simde_mm_cmpeq_epi32(mask_data,mask_data)));
	simde__m128i result_vector = simde_mm_and_si128(vector, mask_data);
	             result_data = simde_mm_add_epi32(result_data, result_vector);
	simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(data), result_data);
    };
    static constexpr auto zero = simde_mm_setzero_si128;
    static constexpr auto subtract = simde_mm_sub_epi32;
    static constexpr auto divide = simde_mm_div_epu32;
    static constexpr auto add = simde_mm_add_epi32;
    static constexpr auto multiply = simde_mm_mul_epu32;
    /* static constexpr auto modulo = simde_mm_rem_epu32; */
	inline static constexpr auto negative = [](const Type& a) noexcept -> Type{
		return subtract(zero(), a);
	}; //returns negative value of current type
    inline static constexpr auto broadcast = [](const uint32_t* arr) -> Type {return SimdTraits_avx<uint32_t>::set1(*arr);};
    inline static constexpr auto fmsub = [](const Type& a, const Type& b, Type& c) -> Type {return subtract(c, multiply(a, b));};
    inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c){
	c = simde_mm_add_epi32(simde_mm_mullo_epi32(a, b), c);
    };
    inline static constexpr auto sum = [](const Type& x) -> uint32_t{
	simde__m128i sum64 = simde_mm_hadd_epi32(x, x);
	return simde_mm_extract_epi32(sum64, 0) + simde_mm_extract_epi32(sum64, 1);
    };

};



//int8_t traits




template <>
struct SimdTraits_avx<int8_t> {
    using Type = simde__m128i;
    static constexpr size_t pack_size = 16;
    static constexpr auto load = simde_mm_load_si128;
    static constexpr auto loadu = simde_mm_loadu_si128;
    inline static constexpr auto load_masked = [](const int8_t* data, const simde__m128i& mask) noexcept -> Type{
	simde__m128i loaded_data = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data));
	simde__m128i result = simde_mm_and_si128(loaded_data, mask);
	return result;
    };
    static constexpr auto set = simde_mm_set_epi8;
    static constexpr auto set1 = simde_mm_set1_epi8;
    static constexpr auto store = simde_mm_store_si128;
    static constexpr auto storeu = simde_mm_storeu_si128;
    inline static constexpr auto store_masked = [](int8_t* data, const simde__m128i& mask_data, const simde__m128i& vector){
	simde__m128i data_vector = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data));
	simde__m128i result_data = simde_mm_and_si128(data_vector, simde_mm_xor_si128(mask_data, simde_mm_cmpeq_epi8(mask_data,mask_data)));
	simde__m128i result_vector = simde_mm_and_si128(vector, mask_data);
	             result_data = simde_mm_add_epi8(result_data, result_vector);
	simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(data), result_data);
    };
    static constexpr auto subtract = simde_mm_sub_epi8;
    static constexpr auto divide = simde_mm_div_epi8;
    static constexpr auto zero = simde_mm_setzero_si128;
    static constexpr auto add = simde_mm_add_epi8;
	inline static constexpr auto negative = [](const Type& a) noexcept -> Type{
		return subtract(zero(), a);
	}; //returns negative value of current type
    //https://github.com/lemire/vectorclass/blob/master/vectori128.h#L299
    inline static constexpr auto multiply = [](simde__m128i a, simde__m128i b) -> Type{
	    simde__m128i aodd = simde_mm_srli_epi16(a, 8);
	    simde__m128i bodd = simde_mm_srli_epi16(b, 8);
	    simde__m128i muleven = simde_mm_mullo_epi16(a, b);
	    simde__m128i mulodd = simde_mm_mullo_epi16(aodd, bodd);
		         mulodd = simde_mm_slli_epi16(mulodd, 8); // put odd numbered elements back in place
	    simde__m128i mask = simde_mm_set1_epi32(0x00FF00FF);
	    return simde_mm_blendv_epi8 (mulodd, muleven, mask);
    };
    /* static constexpr auto modulo = simde_mm_rem_epi8; */

    inline static constexpr auto broadcast = [](const int8_t* arr) -> Type {return simde_mm_set1_epi8(*arr);};
    inline static constexpr auto fmsub = [](const Type& a, const Type& b, Type& c) -> Type {return subtract(c, multiply(a, b));};
    inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c){
	c = simde_mm_add_epi8(SimdTraits_avx<int8_t>::multiply(a, b), c);
    };
    //slightly more complicated
    //https://github.com/lemire/vectorclass/blob/master/vectori128.h#L20 for the following specific function
    inline static constexpr auto sum = [](const Type& a) noexcept -> int32_t {
	simde__m128i even = simde_mm_slli_epi16(a,8); // extract even numbers
		     even = simde_mm_srai_epi16(even,8); //sign extend
	simde__m128i odd =  simde_mm_srai_epi16(a,8);
	simde__m128i sum  = simde_mm_add_epi16(even,odd);             
	simde__m128i sum2  = simde_mm_shuffle_epi32(sum,0x0E);
	simde__m128i sum3  = simde_mm_add_epi16(sum,sum2);
	simde__m128i sum4  = simde_mm_shuffle_epi32(sum3,0x01);
	simde__m128i sum5  = simde_mm_add_epi16(sum3,sum4);
	simde__m128i sum6  = simde_mm_shufflelo_epi16(sum5,0x01);
	simde__m128i sum7  = simde_mm_add_epi16(sum5,sum6);
	int16_t sum8 = simde_mm_cvtsi128_si32(sum7); //16 bit sum
	return sum8;
    };

};


//uint8 traits
template <>
struct SimdTraits_avx<uint8_t> {
    using Type = simde__m128i;
    static constexpr size_t pack_size = 16;
    static constexpr auto load = simde_mm_load_si128;
    static constexpr auto loadu = simde_mm_loadu_si128;
    inline static constexpr auto load_masked = [](const void* data, const simde__m128i& mask) noexcept -> Type{
	simde__m128i loaded_data = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data));
	simde__m128i result = simde_mm_and_si128(loaded_data, mask);
	return result;
    };
    static constexpr auto set = simde_mm_set_epi8;
    static constexpr auto set1 = simde_mm_set1_epi8;
    static constexpr auto store = simde_mm_store_si128;
    static constexpr auto storeu = simde_mm_storeu_si128;
    inline static constexpr auto store_masked = [](void* data, const simde__m128i& mask_data, const simde__m128i& vector){
	simde__m128i data_vector = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data));
	simde__m128i result_data = simde_mm_and_si128(data_vector, simde_mm_xor_si128(mask_data, simde_mm_cmpeq_epi8(mask_data,mask_data)));
	simde__m128i result_vector = simde_mm_and_si128(vector, mask_data);
	             result_data = simde_mm_add_epi8(result_data, result_vector);
	simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(data), result_data);
    };
    static constexpr auto zero = simde_mm_setzero_si128;
    static constexpr auto subtract = simde_mm_sub_epi8;
    static constexpr auto divide = simde_mm_div_epu8;
    static constexpr auto add = simde_mm_add_epi8;
    inline static constexpr auto multiply = [](simde__m128i a, simde__m128i b) -> Type{
	    simde__m128i aodd = simde_mm_srli_epi16(a, 8);
	    simde__m128i bodd = simde_mm_srli_epi16(b, 8);
	    simde__m128i muleven = simde_mm_mullo_epi16(a, b);
	    simde__m128i mulodd = simde_mm_mullo_epi16(aodd, bodd);
		         mulodd = simde_mm_slli_epi16(mulodd, 8); // put odd numbered elements back in place
	    simde__m128i mask = simde_mm_set1_epi32(0x00FF00FF);
	    return simde_mm_blendv_epi8 (mulodd, muleven, mask);
    };
    /* static constexpr auto modulo = simde_mm_rem_epu8; */

    inline static constexpr auto broadcast = [](const uint8_t* arr) -> Type {return simde_mm_set1_epi8(*arr);};
    inline static constexpr auto fmsub = [](const Type& a, const Type& b, Type& c) -> Type {return subtract(c, multiply(a, b));};
    inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c){
	c = simde_mm_add_epi8(SimdTraits_avx<uint8_t>::multiply(a, b), c);
    };
    inline static constexpr auto sum = [](const Type& a) noexcept -> int32_t {
	simde__m128i sum = simde_mm_sad_epu8(a, simde_mm_setzero_si128());
	sum = simde_mm_add_epi32(sum, simde_mm_shuffle_epi32(sum, 0x4e));
	return simde_mm_cvtsi128_si32(sum);
    };

};



//int16 traits


template <>
struct SimdTraits_avx<int16_t> {
    using Type = simde__m128i;
    static constexpr size_t pack_size = 8;
    static constexpr auto load = simde_mm_load_si128;
    static constexpr auto loadu = simde_mm_loadu_si128;
    inline static constexpr auto load_masked = [](const int16_t* data, const simde__m128i& mask) noexcept -> Type{
	simde__m128i loaded_data = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data));
	simde__m128i result = simde_mm_and_si128(loaded_data, mask);
	return result;
    };
    static constexpr auto set = simde_mm_set_epi16;
    static constexpr auto set1 = simde_mm_set1_epi16;
    static constexpr auto store = simde_mm_store_si128;
    static constexpr auto storeu = simde_mm_storeu_si128;
    inline static constexpr auto store_masked = [](int16_t* data, const simde__m128i& mask_data, const simde__m128i& vector) {
	//probably this would fix it:
	//load data into a register
	//result_data = data_vector & (~mask_data);
	//result_vector = simde_mm_and_si128(vector, mask_data);
	//result_data = simde_mm_add_epi16(result_data, result_vector);
	//simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(data), result_data);
	simde__m128i data_vector = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data));
	simde__m128i result_data = simde_mm_and_si128(data_vector, simde_mm_xor_si128(mask_data, simde_mm_cmpeq_epi16(mask_data,mask_data)));
	simde__m128i result_vector = simde_mm_and_si128(vector, mask_data);
	             result_data = simde_mm_add_epi16(result_data, result_vector);
	simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(data), result_data);
    };
    static constexpr auto zero = simde_mm_setzero_si128;
    static constexpr auto subtract = simde_mm_sub_epi16;
    static constexpr auto divide = simde_mm_div_epi16;
    static constexpr auto add = simde_mm_add_epi16;
    static constexpr auto multiply = simde_mm_mullo_epi16;
    /* static constexpr auto modulo = simde_mm_rem_epi16; */


	inline static constexpr auto negative = [](const Type& a) noexcept -> Type{
		return subtract(zero(), a);
	}; //returns negative value of current type
    inline static constexpr auto broadcast = [](const int16_t* arr) -> Type {return simde_mm_set1_epi16(*arr);};
    inline static constexpr auto fmsub = [](const Type& a, const Type& b, Type& c) -> Type {return subtract(c, multiply(a, b));};
    inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c){
	c = simde_mm_add_epi16(simde_mm_mullo_epi16(a, b), c);
    };
    //the following is from https://github.com/lemire/vectorclass/blob/master/vectori128.h#L1529
    inline static constexpr auto sum = [](const Type& a) noexcept -> int32_t{
	simde__m128i sum1  = simde_mm_shuffle_epi32(a,0x0E);
	simde__m128i sum2  = simde_mm_add_epi16(a,sum1);
	simde__m128i sum3  = simde_mm_shuffle_epi32(sum2,0x01);
	simde__m128i sum4  = simde_mm_add_epi16(sum2,sum3);
	simde__m128i sum5  = simde_mm_shufflelo_epi16(sum4,0x01);
	simde__m128i sum6  = simde_mm_add_epi16(sum4,sum5);
	int16_t sum7  = simde_mm_cvtsi128_si32(sum6);
	return  sum7;
    };

};

//uint16 traits


template <>
struct SimdTraits_avx<uint16_t> {
    using Type = simde__m128i;
    static constexpr size_t pack_size = 8;
    static constexpr auto load = simde_mm_load_si128;
    static constexpr auto loadu = simde_mm_loadu_si128;
    inline static constexpr auto load_masked = [](const void* data, const simde__m128i& mask) noexcept -> Type{
	simde__m128i loaded_data = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data));
	simde__m128i result = simde_mm_and_si128(loaded_data, mask);
	return result;
    };
    static constexpr auto set = simde_mm_set_epi16;
    static constexpr auto set1 = simde_mm_set1_epi16;
    static constexpr auto store = simde_mm_store_si128;
    static constexpr auto storeu = simde_mm_storeu_si128;
    inline static constexpr auto store_masked = [](void* data, const simde__m128i& mask_data, const simde__m128i& vector) {
	//probably this would fix it:
	//load data into a register
	//result_data = data_vector & (~mask_data);
	//result_vector = simde_mm_and_si128(vector, mask_data);
	//result_data = simde_mm_add_epi16(result_data, result_vector);
	//simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(data), result_data);
	simde__m128i data_vector = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data));
	simde__m128i result_data = simde_mm_and_si128(data_vector, simde_mm_xor_si128(mask_data, simde_mm_cmpeq_epi16(mask_data,mask_data)));
	simde__m128i result_vector = simde_mm_and_si128(vector, mask_data);
	             result_data = simde_mm_add_epi16(result_data, result_vector);
	simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(data), result_data);
    };
    static constexpr auto zero = simde_mm_setzero_si128;
    static constexpr auto subtract = simde_mm_sub_epi16;
    static constexpr auto divide = simde_mm_div_epu16;
    static constexpr auto add = simde_mm_add_epi16;
    static constexpr auto multiply = simde_mm_mullo_epi16;
    /* static constexpr auto modulo = simde_mm_rem_epu16; */
 

    inline static constexpr auto broadcast = [](const uint16_t* arr) -> Type {return simde_mm_set1_epi16(*arr);};
    inline static constexpr auto fmsub = [](const Type& a, const Type& b, Type& c) -> Type {return subtract(c, multiply(a, b));};
    inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c){
	c = simde_mm_add_epi16(simde_mm_mullo_epi16(a, b), c);
    };
    //the following is from https://github.com/lemire/vectorclass/blob/master/vectori128.h#L1529
    inline static constexpr auto sum = [](const Type& a) noexcept -> int32_t{
	simde__m128i sum1  = simde_mm_shuffle_epi32(a,0x0E);
	simde__m128i sum2  = simde_mm_add_epi16(a,sum1);
	simde__m128i sum3  = simde_mm_shuffle_epi32(sum2,0x01);
	simde__m128i sum4  = simde_mm_add_epi16(sum2,sum3);
	simde__m128i sum5  = simde_mm_shufflelo_epi16(sum4,0x01);
	simde__m128i sum6  = simde_mm_add_epi16(sum4,sum5);
	int16_t sum7  = simde_mm_cvtsi128_si32(sum6);
	return  sum7;
    };

};



//complex<float>
template <>
struct SimdTraits_avx<complex_64> {
	using Type = simde__m128;
	static constexpr size_t pack_size = 2;  // AVX can handle 4 floats
	inline static constexpr auto load = [](const complex_64* arr) noexcept -> Type {return simde_mm_load_ps(reinterpret_cast<const float*>(arr));};
	inline static constexpr auto loadu = [](const complex_64* arr) noexcept -> Type {return simde_mm_loadu_ps(reinterpret_cast<const float*>(arr));};
	inline static constexpr auto load_masked = [](const complex_64* arr, const simde__m128i& mask) noexcept -> Type{
		return  simde_mm_maskload_ps(reinterpret_cast<const float*>(arr), mask);
	};//look at the generate mask function to see the difference
	inline static constexpr auto set = [](const complex_64& comp3, const complex_64& comp4) noexcept -> Type {
	//(comp4.re, comp4.im, comp3.re, comp3.im) 
		return simde_mm_set_ps(
		std::get<1>(static_cast<const my_complex<float>&>(comp3)), std::get<0>(static_cast<const my_complex<float>&>(comp3)),
		std::get<1>(static_cast<const my_complex<float>&>(comp4)), std::get<0>(static_cast<const my_complex<float>&>(comp4)));
	};
	inline static constexpr auto set1 = [](const complex_64& comp) noexcept -> Type {return SimdTraits_avx<complex_64>::set(comp, comp);};
	inline static constexpr auto broadcast = [](const complex_64* comp) noexcept -> Type {return SimdTraits_avx<complex_64>::set1(*comp);};
	inline static constexpr auto store = [](complex_64* comp_arr, const Type& vec) noexcept {simde_mm_store_ps(reinterpret_cast<float*>(comp_arr), vec);};
	inline static constexpr auto storeu = [](complex_64* comp_arr, const Type& vec) noexcept {simde_mm_storeu_ps(reinterpret_cast<float*>(comp_arr), vec);};
	inline static constexpr auto store_masked = [](complex_64* comp_arr, const simde__m128i& mask, const Type& vec) noexcept {
	simde_mm_maskstore_ps(reinterpret_cast<float*>(comp_arr), mask, vec);
	};
	//svml exponent functions
	static constexpr auto reciprical = simde_mm_rcp_ps;
	static constexpr auto exp = simde_mm_exp_ps;
	static constexpr auto pow = simde_mm_pow_ps;
	static constexpr auto sqrt = simde_mm_sqrt_ps;
	static constexpr auto invsqrt = simde_mm_invsqrt_ps;

        //trig svml functions
	static constexpr auto tanh = simde_mm_tanh_ps;
	static constexpr auto tan = simde_mm_tan_ps;
	static constexpr auto atanh = simde_mm_atanh_ps;
	static constexpr auto atan = simde_mm_atan_ps;
	inline static constexpr auto cotanh = [](const Type& a) noexcept -> Type { return reciprical(tanh(a));};
	inline static constexpr auto cotan = [](const Type& a) noexcept -> Type { return reciprical(tan(a));};
	static constexpr auto sinh = simde_mm_sinh_ps;
	static constexpr auto sin = simde_mm_sin_ps;
	static constexpr auto asinh = simde_mm_asinh_ps;
	static constexpr auto asin = simde_mm_asin_ps;
	inline static constexpr auto csch = [](const Type& a) noexcept -> Type { return reciprical(sinh(a));};
	inline static constexpr auto csc = [](const Type& a) noexcept -> Type { return reciprical(sin(a));};
	static constexpr auto cosh = simde_mm_cosh_ps;
	static constexpr auto cos = simde_mm_cos_ps;
	static constexpr auto acosh = simde_mm_acosh_ps;
	static constexpr auto acos = simde_mm_acos_ps;
	inline static constexpr auto sech = [](const Type& a) noexcept -> Type { return reciprical(cosh(a));};
	inline static constexpr auto sec = [](const Type& a) noexcept -> Type { return reciprical(cos(a));};
	static constexpr auto log = simde_mm_log_ps;

	
	static constexpr auto subtract = simde_mm_sub_ps;
	static constexpr auto divide = simde_mm_div_ps;
	static constexpr auto add = simde_mm_add_ps;
	static constexpr auto multiply = simde_mm_mul_ps;
	static constexpr auto zero = simde_mm_setzero_ps;
	/* inline static constexpr auto modulo = [](const Type& dividend_c, const Type& divisor_c ) -> Type{ //dividend % divisor */
	/* 	simde__m128i dividend = simde_mm_cvtps_epi32(dividend_c); */
	/* 	simde__m128i divisor = simde_mm_cvtps_epi32(divisor); */
	/* 	return simde_mm_cvtepi32_ps(simde_mm_rem_epi32(dividend, divisor)); */
	/* }; */

	inline static constexpr auto negative = [](const Type& a) noexcept -> Type{
		return subtract(zero(), a);
	}; //returns negative value of current type
    static constexpr auto fmsub = simde_mm_fmsub_ps;
	inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c){
#if defined(__FMA__) || defined(SIMDE_X86_FMA)
		c = simde_mm_fmadd_ps(a, b, c);
#else
		c = simde_mm_add_ps(simde_mm_mul_ps(a,b),c);
#endif //defined(__FMA__) || defined(SIMDE_X86_FMA)
	};
	inline static constexpr auto sum = [](Type x) -> complex_64{
		// x = ( x1.re, x1.im, x0.re, x0.im )
		//loDual = (- , -, x0.re, x0.im) [x]
		//hiDual = (-, -, x1.re, x1.im)
		const simde__m128 hiDual = simde_mm_movehl_ps(x, x);
		const simde__m128 sumDual = simde_mm_add_ps(x, hiDual);
		complex_64 result[2];
		simde_mm_storeu_ps(reinterpret_cast<float*>(result), sumDual);
		return result[0];

	};
};

#define NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res)\
Type ilow_res = simde_mm_cvtps_ph(low_res, 0);\
Type ihigh_res = simde_mm_cvtps_ph(high_res, 0);\
return simde_mm_unpacklo_epi64(ilow_res, ihigh_res);\

template<>
struct SimdTraits_avx<float16_t>{
	using Type = simde__m128i; // going to hold it as a list of int16_t, less efficient, but will avoid segmentation faults from processing
	static constexpr size_t pack_size = 8;
        inline static constexpr auto load = [](const float16_t* arr) noexcept -> Type {
		return simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(arr));
	};
	inline static constexpr auto loadu = [](const float16_t* arr) noexcept -> Type {
		return simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(arr));
	};
	inline static constexpr auto load_masked = [](const float16_t* data, const simde__m128i& mask) noexcept -> Type {
		simde__m128i loaded_data = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data));
		return simde_mm_and_si128(loaded_data, mask);
	};
	inline static constexpr auto set = [](float16_t ele1, float16_t ele2,float16_t ele3, float16_t ele4, float16_t ele5, float16_t ele6, float16_t ele7, float16_t ele8)
		noexcept -> Type {
			return  simde_mm_set_epi16(
					_NT_FLOAT16_TO_INT16_(ele1),
					_NT_FLOAT16_TO_INT16_(ele2),
					_NT_FLOAT16_TO_INT16_(ele3),
					_NT_FLOAT16_TO_INT16_(ele4),
					_NT_FLOAT16_TO_INT16_(ele5),
					_NT_FLOAT16_TO_INT16_(ele6),
					_NT_FLOAT16_TO_INT16_(ele7),
					_NT_FLOAT16_TO_INT16_(ele8));
	};
	inline static constexpr auto set1 = [](float16_t ele) noexcept -> Type {return simde_mm_set1_epi16(_NT_FLOAT16_TO_INT16_(ele));};
	inline static constexpr auto store = [](float16_t* arr, const Type& vec) noexcept {
		simde_mm_store_si128(reinterpret_cast<simde__m128i*>(arr), vec);
	};
	inline static constexpr auto storeu = [](float16_t* arr, const Type& vec) noexcept {
		simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(arr), vec);
	};
	inline static constexpr auto store_masked = [](float16_t* data, const simde__m128i& mask_data, const Type& vector) noexcept {
		simde__m128i data_vector = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data));
		simde__m128i result_data = simde_mm_and_si128(data_vector, simde_mm_xor_si128(mask_data, simde_mm_cmpeq_epi16(mask_data,mask_data)));
		simde__m128i result_vector = simde_mm_and_si128(vector, mask_data);
			     result_data = simde_mm_add_epi16(result_data, result_vector);
		simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(data), result_data);
	};
	static constexpr auto zero = simde_mm_setzero_si128;
	//trig svml functions
	inline static constexpr auto tanh = [](const Type& a) noexcept -> Type {
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_res = simde_mm_tanh_ps(low_a);
		simde__m128 high_res = simde_mm_tanh_ps(high_a);
        NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res);
	};
	inline static constexpr auto tan = [](const Type& a) noexcept -> Type {
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_res = simde_mm_tan_ps(low_a);
		simde__m128 high_res = simde_mm_tan_ps(high_a);
        NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res);
	};
	inline static constexpr auto atan = [](const Type& a) noexcept -> Type {
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_res = simde_mm_atan_ps(low_a);
		simde__m128 high_res = simde_mm_atan_ps(high_a);
        NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res);
	};
	inline static constexpr auto atanh = [](const Type& a) noexcept -> Type {
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_res = simde_mm_atanh_ps(low_a);
		simde__m128 high_res = simde_mm_atanh_ps(high_a);
        NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res);
	};
	inline static constexpr auto cotanh = [](const Type& a) noexcept -> Type {
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_res = simde_mm_rcp_ps(simde_mm_tanh_ps(low_a));
		simde__m128 high_res = simde_mm_rcp_ps(simde_mm_tanh_ps(high_a));
        NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res);
	};
	inline static constexpr auto cotan = [](const Type& a) noexcept -> Type {
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_res = simde_mm_rcp_ps(simde_mm_tan_ps(low_a));
		simde__m128 high_res = simde_mm_rcp_ps(simde_mm_tan_ps(high_a));
        NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res);
	};
	inline static constexpr auto sinh = [](const Type& a) noexcept -> Type {
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_res = simde_mm_sinh_ps(low_a);
		simde__m128 high_res = simde_mm_sinh_ps(high_a);
        NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res);
	};
	inline static constexpr auto sin = [](const Type& a) noexcept -> Type {
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_res = simde_mm_sin_ps(low_a);
		simde__m128 high_res = simde_mm_sin_ps(high_a);
        NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res);
	};
	inline static constexpr auto asinh = [](const Type& a) noexcept -> Type {
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_res = simde_mm_asinh_ps(low_a);
		simde__m128 high_res = simde_mm_asinh_ps(high_a);
        NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res);
	};
	inline static constexpr auto asin = [](const Type& a) noexcept -> Type {
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_res = simde_mm_asin_ps(low_a);
		simde__m128 high_res = simde_mm_asin_ps(high_a);
        NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res);
	};
	inline static constexpr auto csch = [](const Type& a) noexcept -> Type {
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_res = simde_mm_rcp_ps(simde_mm_sinh_ps(low_a));
		simde__m128 high_res = simde_mm_sinh_ps(high_a);
        NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res);
	};
	inline static constexpr auto csc = [](const Type& a) noexcept -> Type {
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_res = simde_mm_rcp_ps(simde_mm_sin_ps(low_a));
		simde__m128 high_res = simde_mm_rcp_ps(simde_mm_sin_ps(high_a));
        NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res);
	};


	inline static constexpr auto cosh = [](const Type& a) noexcept -> Type {
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_res = simde_mm_cosh_ps(low_a);
		simde__m128 high_res = simde_mm_cosh_ps(high_a);
        NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res);
	};
	inline static constexpr auto cos = [](const Type& a) noexcept -> Type {
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_res = simde_mm_cos_ps(low_a);
		simde__m128 high_res = simde_mm_cos_ps(high_a);
        NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res);
	};
	inline static constexpr auto acosh = [](const Type& a) noexcept -> Type {
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_res = simde_mm_acosh_ps(low_a);
		simde__m128 high_res = simde_mm_acosh_ps(high_a);
        NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res);
	};
	inline static constexpr auto acos = [](const Type& a) noexcept -> Type {
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_res = simde_mm_acos_ps(low_a);
		simde__m128 high_res = simde_mm_acos_ps(high_a);
        NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res);
	};
	inline static constexpr auto sech = [](const Type& a) noexcept -> Type {
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_res = simde_mm_rcp_ps(simde_mm_cosh_ps(low_a));
		simde__m128 high_res = simde_mm_cosh_ps(high_a);
        NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res);
	};
	inline static constexpr auto sec = [](const Type& a) noexcept -> Type {
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_res = simde_mm_rcp_ps(simde_mm_cos_ps(low_a));
		simde__m128 high_res = simde_mm_rcp_ps(simde_mm_cos_ps(high_a));
        NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res);
	};


	//svml exponent functions
	inline static constexpr auto reciprical = [](const Type& a) noexcept -> Type {
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_res = simde_mm_rcp_ps(low_a);
		simde__m128 high_res = simde_mm_rcp_ps(high_a);
        NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res);
	
	};
	inline static constexpr auto exp = [](const Type& a) noexcept -> Type {
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_res = simde_mm_exp_ps(low_a);
		simde__m128 high_res = simde_mm_exp_ps(high_a);
        NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res);
	};
	inline static constexpr auto pow = [](const Type& a, const Type& nums) noexcept -> Type { // raises a to the power of nums
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_nums = simde_mm_cvtph_ps(nums);
		simde__m128 high_nums = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(nums, nums));
		simde__m128 low_pow = simde_mm_pow_ps(low_a, low_nums);
		simde__m128 high_pow = simde_mm_pow_ps(high_a, high_nums);
		simde__m128i low_res = simde_mm_cvtps_ph(low_pow, 0);
		simde__m128i high_res = simde_mm_cvtps_ph(high_pow, 0);
		return simde_mm_unpacklo_epi64(low_res, high_res);

	};

	inline static constexpr auto sqrt = [](const Type& a) noexcept -> Type {
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_res = simde_mm_sqrt_ps(low_a);
		simde__m128 high_res = simde_mm_sqrt_ps(high_a);
        NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res);
	};
	inline static constexpr auto invsqrt = [](const Type& a) noexcept -> Type { //important for self-attention and useful for other computations
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_res = simde_mm_invsqrt_ps(low_a);
		simde__m128 high_res = simde_mm_invsqrt_ps(high_a);
        NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res);
	};

	inline static constexpr auto log = [](const Type& a) noexcept -> Type { //important for self-attention and useful for other computations
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_res = simde_mm_log_ps(low_a);
		simde__m128 high_res = simde_mm_log_ps(high_a);
        NT_TWO_TYPE_FP32_ONE_FP16(low_res, high_res);
	};
	//regular
	inline static constexpr auto subtract = [](const Type& a, const Type& b) noexcept -> Type{
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_b = simde_mm_cvtph_ps(b);
		simde__m128 high_b = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(b, b));
		simde__m128 low_add = simde_mm_sub_ps(low_a, low_b);
		simde__m128 high_add = simde_mm_sub_ps(high_a, high_b);
		simde__m128i low_res = simde_mm_cvtps_ph(low_add, 0);
		simde__m128i high_res = simde_mm_cvtps_ph(high_add, 0);
		return simde_mm_unpacklo_epi64(low_res, high_res);
	};
	inline static constexpr auto divide = [](const Type& a, const Type& b) noexcept -> Type{
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_b = simde_mm_cvtph_ps(b);
		simde__m128 high_b = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(b, b));
		simde__m128 low_add = simde_mm_div_ps(low_a, low_b);
		simde__m128 high_add = simde_mm_div_ps(high_a, high_b);
		simde__m128i low_res = simde_mm_cvtps_ph(low_add, 0);
		simde__m128i high_res = simde_mm_cvtps_ph(high_add, 0);
		return simde_mm_unpacklo_epi64(low_res, high_res);
	};
	inline static constexpr auto add = [](const Type& a, const Type& b) noexcept -> Type {
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_b = simde_mm_cvtph_ps(b);
		simde__m128 high_b = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(b, b));
		simde__m128 low_add = simde_mm_add_ps(low_a, low_b);
		simde__m128 high_add = simde_mm_add_ps(high_a, high_b);
		simde__m128i low_res = simde_mm_cvtps_ph(low_add, 0);
		simde__m128i high_res = simde_mm_cvtps_ph(high_add, 0);
		return simde_mm_unpacklo_epi64(low_res, high_res);

	};
	inline static constexpr auto multiply = [](const Type& a, const Type& b) noexcept -> Type {
		simde__m128 low_a = simde_mm_cvtph_ps(a);
		simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(a, a));
		simde__m128 low_b = simde_mm_cvtph_ps(b);
		simde__m128 high_b = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(b, b));
		simde__m128 low_add = simde_mm_mul_ps(low_a, low_b);
		simde__m128 high_add = simde_mm_mul_ps(high_a, high_b);
		simde__m128i low_res = simde_mm_cvtps_ph(low_add, 0);
		simde__m128i high_res = simde_mm_cvtps_ph(high_add, 0);
		return simde_mm_unpacklo_epi64(low_res, high_res);

	};
	/* inline static constexpr auto modulo = [](const Type& dividend_c, const Type& divisor_c ) -> Type{ //dividend % divisor */
	/* 	simde__m128 loDividend = simde_mm_cvtph_ps(dividend_c); */
	/* 	simde__m128 hiDividend = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(dividend_c, dividend_c)); */
	/* 	simde__m128 loDivisor = simde_mm_cvtph_ps(divisor_c); */
	/* 	simde__m128 hiDivisor = simde_mm_cvtph_ps(simde_mm_unpackhi_epi64(divisor_c, divisor_c)); */
	/* 	return simde_mm_unpacklo_epi64(simde_mm_cvtps_ph(SimdTraits_avx<float>::modulo(loDividend, loDivisor)), */
	/* 					simde_mm_cvtps_ph(SimdTraits_avx<float>::modulo(hiDividend, hiDivisor))); */
	/* }; */
	inline static constexpr auto negative = [](const Type& a) noexcept -> Type{
		return subtract(zero(), a);
	};
	inline static constexpr auto broadcast = [](const float16_t* a) noexcept -> Type {return SimdTraits_avx<float16_t>::set1(*a);};
	inline static constexpr auto fmsub = [](const Type& a, const Type& b, Type& c) noexcept {return subtract(c, multiply(a, b));};
    inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c) noexcept {
		c = SimdTraits_avx<float16_t>::add(SimdTraits_avx<float16_t>::multiply(a,b),c);
    };
	inline static constexpr auto sum_float = [](const simde__m128& x){
#ifdef SIMDE_ARCH_SSE3
		simde__m128 hadd = simde_mm_hadd_ps(x, x);  // Horizontal add: vec[0] + vec[1], vec[2] + vec[3]
		hadd = simde_mm_hadd_ps(hadd, hadd);       // Add the two results together
		return simde_mm_cvtss_f32(hadd);           // Extract the sum from the resulting __m128
#else
		// x = ( x3, x2, x1, x0 )
		// loDual = ( -, -, x1, x0 )
		const simde__m128 loDual = x;
		// hiDual = ( -, -, x3, x2)
		const simde__m128 hiDual = simde_mm_movehl_ps(x, x);
		// sumDual = ( -, -, x1 + x3, x0 + x2)
		const simde__m128 sumDual = simde_mm_add_ps(loDual, hiDual);
		// lo = ( -, -, -, x0 + x2 )
		const simde__m128 lo = sumDual;
		// hi = ( -, -, -, x1 + x3 )
		const simde__m128 hi = simde_mm_shuffle_ps(sumDual, sumDual, 0x1);
		// sum = ( -, -, -, x0 + x1 + x2 + x3 )
		const simde__m128 sum = simde_mm_add_ss(lo, hi);
		return simde_mm_cvtss_f32(sum);
#endif
	};
    inline static constexpr auto sum = [](const Type& x) -> float16_t{
	    simde__m128 low_a = simde_mm_cvtph_ps(x);
	    simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_shuffle_epi32(x, SIMDE_MM_SHUFFLE(1, 0, 3, 2)));
	    return _NT_FLOAT32_TO_FLOAT16_(SimdTraits_avx<float16_t>::sum_float(low_a) + SimdTraits_avx<float16_t>::sum_float(high_a));
    };


};


#undef NT_TWO_TYPE_FP32_ONE_FP16 

template<>
struct SimdTraits_avx<complex_32>{
	using Type = simde__m128i; // going to hold it as a list of int16_t, less efficient, but will avoid segmentation faults from processing
	static constexpr size_t pack_size = 4;
        inline static constexpr auto load = [](const complex_32* arr) noexcept -> Type {
		return simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(arr));
	};
	inline static constexpr auto loadu = [](const complex_32* arr) noexcept -> Type {
		return simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(arr));
	};
	inline static constexpr auto load_masked = [](const complex_32* data, const simde__m128i& mask) noexcept -> Type {
		simde__m128i loaded_data = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data));
		return simde_mm_and_si128(loaded_data, mask);
	};
	inline static constexpr auto set = [](complex_32 ele1, complex_32 ele2, complex_32 ele3, complex_32 ele4)
		noexcept -> Type {
			return  simde_mm_set_epi16(
				_NT_FLOAT16_TO_INT16_(std::get<1>(static_cast<const complex_32&>(ele1))), _NT_FLOAT16_TO_INT16_(std::get<0>(static_cast<const complex_32&>(ele1))),
				_NT_FLOAT16_TO_INT16_(std::get<1>(static_cast<const complex_32&>(ele2))), _NT_FLOAT16_TO_INT16_(std::get<0>(static_cast<const complex_32&>(ele2))),
				_NT_FLOAT16_TO_INT16_(std::get<1>(static_cast<const complex_32&>(ele3))), _NT_FLOAT16_TO_INT16_(std::get<0>(static_cast<const complex_32&>(ele3))),
				_NT_FLOAT16_TO_INT16_(std::get<1>(static_cast<const complex_32&>(ele4))), _NT_FLOAT16_TO_INT16_(std::get<0>(static_cast<const complex_32&>(ele4))));
	};
	inline static constexpr auto set1 = [](complex_32 ele) noexcept -> Type {return SimdTraits_avx<complex_32>::set(ele, ele, ele, ele);};
	inline static constexpr auto store = [](complex_32* arr, const Type& vec) noexcept {
		simde_mm_store_si128(reinterpret_cast<simde__m128i*>(arr), vec);
	};
	inline static constexpr auto storeu = [](complex_32* arr, const Type& vec) noexcept {
		simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(arr), vec);
	};
	inline static constexpr auto store_masked = [](complex_32* data, const simde__m128i& mask_data, const Type& vector) noexcept {
		simde__m128i data_vector = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data));
		simde__m128i result_data = simde_mm_and_si128(data_vector, simde_mm_xor_si128(mask_data, simde_mm_cmpeq_epi16(mask_data,mask_data)));
		simde__m128i result_vector = simde_mm_and_si128(vector, mask_data);
			     result_data = simde_mm_add_epi16(result_data, result_vector);
		simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(data), result_data);
	};
	static constexpr auto zero = simde_mm_setzero_si128;
	//trig svml functions
	static constexpr auto tanh = SimdTraits_avx<float16_t>::tanh;
	static constexpr auto tan = SimdTraits_avx<float16_t>::tan;
	static constexpr auto atanh = SimdTraits_avx<float16_t>::atanh;
	static constexpr auto atan = SimdTraits_avx<float16_t>::atan;
	static constexpr auto cotanh = SimdTraits_avx<float16_t>::cotanh;
	static constexpr auto cotan = SimdTraits_avx<float16_t>::cotan;
	static constexpr auto sinh = SimdTraits_avx<float16_t>::sinh;
	static constexpr auto sin = SimdTraits_avx<float16_t>::sin;
	static constexpr auto asinh = SimdTraits_avx<float16_t>::asinh;
	static constexpr auto asin = SimdTraits_avx<float16_t>::asin;
	static constexpr auto csch = SimdTraits_avx<float16_t>::csch;
	static constexpr auto csc = SimdTraits_avx<float16_t>::csc;
	static constexpr auto cosh = SimdTraits_avx<float16_t>::cosh;
	static constexpr auto cos = SimdTraits_avx<float16_t>::cos;
	static constexpr auto acosh = SimdTraits_avx<float16_t>::acosh;
	static constexpr auto acos = SimdTraits_avx<float16_t>::acos;
	static constexpr auto sech = SimdTraits_avx<float16_t>::sech;
	static constexpr auto sec = SimdTraits_avx<float16_t>::sec;
	static constexpr auto log = SimdTraits_avx<float16_t>::log;


	//svml exponent functions
	static constexpr auto reciprical = SimdTraits_avx<float16_t>::reciprical;
	static constexpr auto exp = SimdTraits_avx<float16_t>::exp;
	static constexpr auto pow = SimdTraits_avx<float16_t>::pow;
	static constexpr auto sqrt = SimdTraits_avx<float16_t>::sqrt;
	static constexpr auto invsqrt = SimdTraits_avx<float16_t>::invsqrt;


	//regular 
	static constexpr auto divide = SimdTraits_avx<float16_t>::divide;
	static constexpr auto subtract = SimdTraits_avx<float16_t>::subtract;
	static constexpr auto add = SimdTraits_avx<float16_t>::add;
	static constexpr auto multiply = SimdTraits_avx<float16_t>::multiply;
	/* static constexpr auto modulo = SimdTraits_avx<flaot16_t>::modulo; */
	inline static constexpr auto negative = [](const Type& a) noexcept -> Type{
		return subtract(zero(), a);
	}; //returns negative value of current type
	inline static constexpr auto broadcast = [](const complex_32* a) noexcept -> Type {return SimdTraits_avx<complex_32>::set1(*a);};
	inline static constexpr auto fmsub = [](const Type& a, const Type& b, Type& c) noexcept {return subtract(c, multiply(a, b));};
	inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c) noexcept {
		c = add(multiply(a,b),c);
        };
	inline static constexpr auto sum_floatsC = [](const simde__m128& x) -> complex_32{
		// x = ( x1.re, x1.im, x0.re, x0.im )
		//loDual = (- , -, x0.re, x0.im) [x]
		//hiDual = (-, -, x1.re, x1.im)
		const simde__m128 hiDual = simde_mm_movehl_ps(x, x);
		const simde__m128 sumDual = simde_mm_add_ps(x, hiDual);
		complex_32 result[2];
		simde_mm_storeu_ps(reinterpret_cast<float*>(result), sumDual);
		return result[0];
	
	};
    inline static constexpr auto sum = [](const Type& x) -> complex_32{
	    simde__m128 low_a = simde_mm_cvtph_ps(x);
	    simde__m128 high_a = simde_mm_cvtph_ps(simde_mm_shuffle_epi32(x, SIMDE_MM_SHUFFLE(1, 0, 3, 2)));
	    return SimdTraits_avx<complex_32>::sum_floatsC(low_a) + SimdTraits_avx<complex_32>::sum_floatsC(high_a);
       };


};


template <>
struct SimdTraits_avx<complex_128> {
	using Type = simde__m128d;
	static constexpr size_t pack_size = 1;
	inline static constexpr auto load = [](const complex_128* arr) noexcept -> Type {return simde_mm_load_pd(reinterpret_cast<const double*>(arr));};
	inline static constexpr auto loadu = [](const complex_128* arr) noexcept -> Type {return simde_mm_loadu_pd(reinterpret_cast<const double*>(arr));};
	inline static constexpr auto load_masked = [](const complex_128* arr, const simde__m128i& mask) noexcept -> Type {return load(arr);};
	inline static constexpr auto set = [](const complex_128& comp3) noexcept -> Type {
	//(comp4.re, comp4.im, comp3.re, comp3.im) 
		return simde_mm_set_pd(
		std::get<1>(static_cast<const my_complex<double>&>(comp3)), std::get<0>(static_cast<const my_complex<double>&>(comp3)));
	};
	inline static constexpr auto set1 = [](const complex_128& comp) noexcept -> Type {return SimdTraits_avx<complex_128>::set(comp);};
	inline static constexpr auto broadcast = [](const complex_128* comp) noexcept -> Type {return SimdTraits_avx<complex_128>::set1(*comp);};
	inline static constexpr auto store = [](complex_128* comp_arr, const Type& vec) noexcept {simde_mm_store_pd(reinterpret_cast<double*>(comp_arr), vec);};
	inline static constexpr auto storeu = [](complex_128* comp_arr, const Type& vec) noexcept {simde_mm_storeu_pd(reinterpret_cast<double*>(comp_arr), vec);};
	inline static constexpr auto store_masked = [](complex_128* comp_arr, const simde__m128i& mask, const Type& vec) noexcept {
	simde_mm_maskstore_pd(reinterpret_cast<double*>(comp_arr), mask, vec);
	};

    inline static constexpr auto grab = [](const Type& x) noexcept -> complex_128{
        complex_128 out;
        simde_mm_store_pd(reinterpret_cast<double*>(&out), x);
        return out;
    };

	static constexpr auto zero = simde_mm_setzero_pd;
	static constexpr auto divide = simde_mm_div_pd;

	//svml exponent functions
	inline static constexpr auto reciprical = [](Type a){return divide(set1(1.0), a);};
	static constexpr auto exp = simde_mm_exp_pd;
	static constexpr auto pow = simde_mm_pow_pd;
	static constexpr auto sqrt = simde_mm_sqrt_pd;
	static constexpr auto invsqrt = simde_mm_invsqrt_pd;

        //trig svml functions
	static constexpr auto tanh = simde_mm_tanh_pd;
	static constexpr auto tan = simde_mm_tan_pd;
	static constexpr auto atanh = simde_mm_atanh_pd;
	static constexpr auto atan = simde_mm_atan_pd;
	inline static constexpr auto cotanh = [](const Type& a) noexcept -> Type { return reciprical(tanh(a));};
	inline static constexpr auto cotan = [](const Type& a) noexcept -> Type { return reciprical(tan(a));};
	static constexpr auto sinh = simde_mm_sinh_pd;
	static constexpr auto sin = simde_mm_sin_pd;
	static constexpr auto asinh = simde_mm_asinh_pd;
	static constexpr auto asin = simde_mm_asin_pd;
	inline static constexpr auto csch = [](const Type& a) noexcept -> Type { return reciprical(sinh(a));};
	inline static constexpr auto csc = [](const Type& a) noexcept -> Type { return reciprical(sin(a));};
	static constexpr auto cosh = simde_mm_cosh_pd;
	static constexpr auto cos = simde_mm_cos_pd;
	static constexpr auto acosh = simde_mm_acosh_pd;
	static constexpr auto acos = simde_mm_acos_pd;
	inline static constexpr auto sech = [](const Type& a) noexcept -> Type { return reciprical(cosh(a));};
	inline static constexpr auto sec = [](const Type& a) noexcept -> Type { return reciprical(cos(a));};
	static constexpr auto log = simde_mm_log_pd;

	static constexpr auto subtract = simde_mm_sub_pd;
	static constexpr auto add = simde_mm_add_pd;
	static constexpr auto multiply = simde_mm_mul_pd;
	/* inline static constexpr auto modulo = [](const Type& dividend_c, const Type& divisor_c ) -> Type{ //dividend % divisor */
	/* 	simde__m128i dividend = simde_mm_cvtpd_epi64(dividend_c); */
	/* 	simde__m128i divisor = simde_mm_cvtpd_epi64(divisor); */
	/* 	return simde_mm_cvtepi64_pd(simde_mm_rem_epi64(dividend, divisor)); */
	/* }; */
	inline static constexpr auto negative = [](const Type& a) noexcept -> Type{
		return subtract(zero(), a);
	}; //returns negative value of current type

    static constexpr auto fmsub = simde_mm_fmsub_pd;
	inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c){
#if defined(__FMA__) || defined(SIMDE_X86_FMA)
		c = simde_mm_fmadd_pd(a, b, c);
#else
		c = simde_mm_add_pd(simde_mm_mul_pd(a,b),c);
#endif
	};
	inline static constexpr auto sum = [](const Type& x) -> complex_128 {return grab(x);};

};

//max pack size is 32, so it needs to account for all of it
alignas(64) static const int8_t mask_data_avx[64] = { 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	0,  0,  0,  0,  0,  0,  0,  0, 
	0,  0,  0,  0,  0,  0,  0,  0,
	0,  0,  0,  0,  0,  0,  0,  0, 
	0,  0,  0,  0,  0,  0,  0,  0
};

alignas(64) static const int16_t mask_data_16_avx[64] = { 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	0,  0,  0,  0,  0,  0,  0,  0, 
	0,  0,  0,  0,  0,  0,  0,  0,
	0,  0,  0,  0,  0,  0,  0,  0, 
	0,  0,  0,  0,  0,  0,  0,  0
};

alignas(64) static const int64_t mask_data_64_avx[16] = { 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	0,  0,  0,  0,  0,  0,  0,  0, 
};

alignas(64) static const int32_t mask_data_32_avx[32] = { 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	0,  0,  0,  0,  0,  0,  0,  0, 
	0,  0,  0,  0,  0,  0,  0,  0 
};


using mask_type_avx = simde__m128i;
template<typename T>
using simde_type_avx = typename SimdTraits_avx<T>::Type;
template<typename T>
inline constexpr size_t pack_size_avx_v = simde_supported_avx_v<T> ? SimdTraits_avx<T>::pack_size : 0;

template<typename T, size_t N>
inline constexpr mask_type_avx Kgenerate_mask_avx() noexcept {
	constexpr size_t pack_size = pack_size_avx_v<T>;
	static_assert(N <= pack_size, "N cannot exceed the number of elements in the SIMD register.");
	if constexpr (std::is_same_v<T, my_complex<float> >){ //special case to handle complex numbers
		return SimdTraits_avx<int32_t>::loadu(reinterpret_cast<const mask_type_avx*>(&mask_data_32_avx[16-(N*2)]));
	}
	else if constexpr(std::is_same_v<T, float16_t>){
		return SimdTraits_avx<int16_t>::loadu(reinterpret_cast<const mask_type_avx*>(&mask_data_16_avx[32-N]));
	}
	else if constexpr (std::is_same_v<T, complex_32>){
		return SimdTraits_avx<int16_t>::loadu(reinterpret_cast<const mask_type_avx*>(&mask_data_16_avx[32-(N*2)]));
	}
	else if constexpr (pack_size == pack_size_avx_v<int32_t>){
		return SimdTraits_avx<int32_t>::loadu(reinterpret_cast<const mask_type_avx*>(&mask_data_32_avx[16-N]));
	}
	else if constexpr (pack_size == pack_size_avx_v<int64_t>){
		return SimdTraits_avx<int64_t>::loadu(reinterpret_cast<const mask_type_avx*>(&mask_data_64_avx[8-N]));
	}
	else if constexpr (pack_size == pack_size_avx_v<int8_t>){
		return SimdTraits_avx<int8_t>::loadu(reinterpret_cast<const mask_type_avx*>(&mask_data_avx[32-N]));
	}else{
		static_assert(pack_size == pack_size_avx_v<int16_t>, "Unexpected mask pack size");
		return SimdTraits_avx<int16_t>::loadu(reinterpret_cast<const mask_type_avx*>(&mask_data_16_avx[32-N]));
	}
}

template<typename T>
inline constexpr mask_type_avx generate_mask_avx(size_t N) {
	constexpr size_t pack_size = pack_size_avx_v<T>;
	if constexpr (std::is_same_v<T, my_complex<float> >){ //special case to handle complex numbers
		return SimdTraits_avx<int32_t>::loadu(reinterpret_cast<const mask_type_avx*>(&mask_data_32_avx[16-(N*2)]));
	}
	else if constexpr(std::is_same_v<T, float16_t>){
		return SimdTraits_avx<int16_t>::loadu(reinterpret_cast<const mask_type_avx*>(&mask_data_16_avx[32-N]));
	}
	else if constexpr (std::is_same_v<T, complex_32>){
		return SimdTraits_avx<int16_t>::loadu(reinterpret_cast<const mask_type_avx*>(&mask_data_16_avx[32-(N*2)]));
	}
	else if constexpr (pack_size == pack_size_avx_v<int32_t>){
		return SimdTraits_avx<int32_t>::loadu(reinterpret_cast<const mask_type_avx*>(&mask_data_32_avx[16-N]));
	}
	else if constexpr (pack_size == pack_size_avx_v<int64_t>){
		return SimdTraits_avx<int64_t>::loadu(reinterpret_cast<const mask_type_avx*>(&mask_data_64_avx[8-N]));
	}
	else if constexpr (pack_size == pack_size_avx_v<int8_t>){
		return SimdTraits_avx<int8_t>::loadu(reinterpret_cast<const mask_type_avx*>(&mask_data_avx[32-N]));
	}else{
		return SimdTraits_avx<int16_t>::loadu(reinterpret_cast<const mask_type_avx*>(&mask_data_16_avx[32-N]));
	}
}

}} //nt::mp::


#endif //_NT_SIMDE_TRAITS_AVX_H_ 
