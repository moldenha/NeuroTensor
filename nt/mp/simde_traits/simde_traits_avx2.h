#ifndef NT_SIMDE_TRAITS_AVX2_H__
#define NT_SIMDE_TRAITS_AVX2_H__
#include "../../types/Types.h"
#include "../../utils/always_inline_macro.h"
#include <simde/x86/avx.h>
#include <simde/x86/fma.h>  // only for FMA if supported
#include <cstddef>
#include <cstddef>
#include <type_traits>
#include <simde/x86/avx2.h>
#include "simde_traits_avx.h"


#define NT_MP_AVX2_CONCAT(a, b) a##b

namespace nt{
namespace mp{

template<typename T>
struct simde_supported_avx2{
	static constexpr bool value = false;
};

template<>
struct simde_supported_avx2<float>{
	static constexpr bool value = true;
};

template<>
struct simde_supported_avx2<double>{
	static constexpr bool value = true;
};

template<>
struct simde_supported_avx2<float16_t>{
	static constexpr bool value = true;
};

template<>
struct simde_supported_avx2<uint8_t>{
	static constexpr bool value = true;
};

template<>
struct simde_supported_avx2<int8_t>{
	static constexpr bool value = true;
};

template<>
struct simde_supported_avx2<uint16_t>{
	static constexpr bool value = true;
};

template<>
struct simde_supported_avx2<int16_t>{
	static constexpr bool value = true;
};

template<>
struct simde_supported_avx2<uint32_t>{
	static constexpr bool value = true;
};

template<>
struct simde_supported_avx2<int32_t>{
	static constexpr bool value = true;
};

template<>
struct simde_supported_avx2<uint64_t>{
	static constexpr bool value = true;
};

template<>
struct simde_supported_avx2<int64_t>{
	static constexpr bool value = true;
};

template<>
struct simde_supported_avx2<my_complex<float> >{
	static constexpr bool value = true;
};

template<>
struct simde_supported_avx2<my_complex<double> >{
	static constexpr bool value = true;
};

template<>
struct simde_supported_avx2<my_complex<float16_t> >{
	static constexpr bool value = true;
};


//these are the types that are specifically supported by the svml header file
//these are going to be the cos, sin, tan, exp, pow, etc functions
//its going to basically just be the floating point numbers (including complex types
template<typename T>
struct simde_svml_supported_avx2{
	static constexpr bool value = false;
};

template<>
struct simde_svml_supported_avx2<float>{
	static constexpr bool value = true;
};

template<>
struct simde_svml_supported_avx2<double>{
	static constexpr bool value = true;
};


template<>
struct simde_svml_supported_avx2<float16_t>{
	static constexpr bool value = true;
};

template<>
struct simde_svml_supported_avx2<complex_128>{
	static constexpr bool value = true;
};


template<>
struct simde_svml_supported_avx2<complex_64>{
	static constexpr bool value = true;
};

template<>
struct simde_svml_supported_avx2<complex_32>{
	static constexpr bool value = true;
};

template<typename T>
inline constexpr bool simde_svml_supported_avx2_v = simde_svml_supported_avx2<T>::value;



template<typename T>
inline constexpr bool simde_supported_avx2_v = simde_supported_avx2<T>::value;


template <typename T>
struct SimdTraits_avx2;

//this is a helper function to join together 2 m128i's
//https://github.com/lemire/vectorclass/blob/master/vectori256.h#L67
#define join_m128i(lo, hi) simde_mm256_inserti128_si256(simde_mm256_castsi128_si256(lo),(hi),1)
//float avx2
//for sum functions, is adapted from https://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally
#define NT_FLOAT_COMPARE_OPS(type)\
    inline static constexpr auto less_than_equal = [](const Type& a, const Type& b) noexcept -> Type {return SimdTraits_avx2<type>::compare(a, b, SIMDE_CMP_LE_OQ);};\
    inline static constexpr auto compare_equal = [](const Type& a, const Type& b) noexcept -> Type {return SimdTraits_avx2<type>::compare(a, b, SIMDE_CMP_EQ_OQ);};\
    inline static constexpr auto compare_not_equal = [](const Type& a, const Type& b) noexcept -> Type {return SimdTraits_avx2<type>::compare(a, b, SIMDE_CMP_NEQ_UQ);};\
    inline static constexpr auto store_compare_equal = [](const Type& a, const Type& b, bool* out_bool){\
        int mask = SimdTraits_avx2<type>::move_mask(SimdTraits_avx2<type>::compare(a, b, SIMDE_CMP_EQ_OQ));\
        for (int i = 0; i < SimdTraits_avx2<type>::pack_size; ++i) {\
            out_bool[i] = (mask >> i) & 1;\
        }\
    };\
    inline static constexpr auto store_compare_not_equal = [](const Type& a, const Type& b, bool* out_bool){\
        int mask = SimdTraits_avx2<type>::move_mask(SimdTraits_avx2<type>::compare(a, b, SIMDE_CMP_NEQ_UQ));\
        for (int i = 0; i < SimdTraits_avx2<type>::pack_size; ++i) {\
            out_bool[i] = (mask >> i) & 1;\
        }\
    };\
    inline static constexpr auto store_compare_less_than_equal = [](const Type& a, const Type& b, bool* out_bool){\
        int mask = SimdTraits_avx2<type>::move_mask(SimdTraits_avx2<type>::compare(a, b, SIMDE_CMP_LE_OQ));\
        for (int i = 0; i < SimdTraits_avx2<type>::pack_size; ++i) {\
            out_bool[i] = (mask >> i) & 1;\
        }\
    };\



template <>
struct SimdTraits_avx2<float> {
	using Type = simde__m256;
	static constexpr size_t pack_size = 8;  // AVX2 can handle 8 floats
	static constexpr auto load = simde_mm256_load_ps; //takes aligned version, valid due to alginment in packed memory
	static constexpr auto loadu = simde_mm256_loadu_ps; //takes aligned version, valid due to alginment in packed memory
	static constexpr auto load_masked = simde_mm256_maskload_ps;
	static constexpr auto set = simde_mm256_set_ps;
	static constexpr auto set1 = simde_mm256_set1_ps;
	static constexpr auto broadcast = simde_mm256_broadcast_ss;
	static constexpr auto store = simde_mm256_store_ps;
	static constexpr auto storeu = simde_mm256_storeu_ps;
	static constexpr auto store_masked = simde_mm256_maskstore_ps;
	static constexpr auto zero = simde_mm256_setzero_ps;
	
	//svml exponent functions
	static constexpr auto reciprical = simde_mm256_rcp_ps;
	static constexpr auto exp = simde_mm256_exp_ps;
	static constexpr auto pow = simde_mm256_pow_ps;
	static constexpr auto sqrt = simde_mm256_sqrt_ps;
	static constexpr auto invsqrt = simde_mm256_invsqrt_ps;

        //trig svml functions
	static constexpr auto tanh = simde_mm256_tanh_ps;
	static constexpr auto tan = simde_mm256_tan_ps;
	static constexpr auto atanh = simde_mm256_atanh_ps;
	static constexpr auto atan = simde_mm256_atan_ps;
	inline static constexpr auto cotanh = [](const Type& a) noexcept -> Type { return reciprical(tanh(a));};
	inline static constexpr auto cotan = [](const Type& a) noexcept -> Type { return reciprical(tan(a));};
	static constexpr auto sinh = simde_mm256_sinh_ps;
	static constexpr auto sin = simde_mm256_sin_ps;
	static constexpr auto asinh = simde_mm256_asinh_ps;
	static constexpr auto asin = simde_mm256_asin_ps;
	inline static constexpr auto csch = [](const Type& a) noexcept -> Type { return reciprical(sinh(a));};
	inline static constexpr auto csc = [](const Type& a) noexcept -> Type { return reciprical(sin(a));};
	static constexpr auto cosh = simde_mm256_cosh_ps;
	static constexpr auto cos = simde_mm256_cos_ps;
	static constexpr auto acosh = simde_mm256_acosh_ps;
	static constexpr auto acos = simde_mm256_acos_ps;
	inline static constexpr auto sech = [](const Type& a) noexcept -> Type { return reciprical(cosh(a));};
	inline static constexpr auto sec = [](const Type& a) noexcept -> Type { return reciprical(cos(a));};
	static constexpr auto log = simde_mm256_log_ps;
    
    // svml round functions
    static constexpr auto round = simde_mm256_round_ps;
    static constexpr auto floor = simde_mm256_floor_ps;
    static constexpr auto ceil = simde_mm256_ceil_ps;

	static constexpr auto subtract = simde_mm256_sub_ps;
	static constexpr auto divide = simde_mm256_div_ps;
	static constexpr auto add = simde_mm256_add_ps;
	static constexpr auto multiply = simde_mm256_mul_ps;
    static constexpr auto compare = simde_mm256_cmp_ps;
    static constexpr auto move_mask = simde_mm256_movemask_ps;
    NT_FLOAT_COMPARE_OPS(float);
	inline static constexpr auto negative = [](const Type& a) noexcept -> Type{
		return subtract(zero(), a);
	};

    inline static constexpr auto remainder(const Type& a, const Type& b){
        return subtract(a, multiply(floor(divide(a, b)), b));
    }
    inline static constexpr auto fmod(const Type& a, const Type& b){
        return subtract(a, multiply(round(divide(a, b), SIMDE_MM_FROUND_TO_ZERO), b));
    }
	/* inline static constexpr auto modulo = [](const Type& divisor_c, const Type& dividend_c) noexcept -> Type{ */
	/* 	/1* static constexpr auto modulo = simde_mm_rem_epi64; *1/ */
	/* 	simde__m256i divisor = simde_mm256_cvtps_epi32(divisor_c); */
	/* 	simde__m256i dividend = simde_mm256_cvtps_epi32(dividend_c); */
	/* 	return simde_mm256_cvtepi32_ps(simde_mm256_rem_epi32(divisor, dividend)); */
	/* } */

	static constexpr auto fmsub = simde_mm256_fmsub_ps;
    inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c){
#if defined(__FMA__) || defined(SIMDE_X86_FMA)
		c = simde_mm256_fmadd_ps(a, b, c);
#else
		c = simde_mm256_add_ps(simde_mm256_mul_ps(a,b),c);
#endif //defined(__FMA__) || defined(SIMDE_X86_FMA)
	};
	inline static constexpr auto sum = [](Type x) -> float{
#ifdef SIMDE_ARCH_SSE3
		x = simde_mm256_hadd_ps(x, x);
		x = simde_mm256_hadd_ps(x, x);
		return simde_mm_cvtss_f32(simde_mm256_castps256_ps128(x));
#else
		// hiQuad = ( x7, x6, x5, x4 )
		const simde__m128 hiQuad = simde_mm256_extractf128_ps(x, 1);
		// loQuad = ( x3, x2, x1, x0 )
		const simde__m128 loQuad = simde_mm256_castps256_ps128(x);
		// sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
		const simde__m128 sumQuad = simde_mm_add_ps(loQuad, hiQuad);
		// loDual = ( -, -, x1 + x5, x0 + x4 )
		const simde__m128 loDual = sumQuad;
		// hiDual = ( -, -, x3 + x7, x2 + x6 )
		const simde__m128 hiDual = simde_mm_movehl_ps(sumQuad, sumQuad);
		// sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
		const simde__m128 sumDual = simde_mm_add_ps(loDual, hiDual);
		// lo = ( -, -, -, x0 + x2 + x4 + x6 )
		const simde__m128 lo = sumDual;
		// hi = ( -, -, -, x1 + x3 + x5 + x7 )
		const simde__m128 hi = simde_mm_shuffle_ps(sumDual, sumDual, 0x1);
		// sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
		const simde__m128 sum = simde_mm_add_ss(lo, hi);
		return simde_mm_cvtss_f32(sum);

#endif

	};

    static constexpr auto min = simde_mm256_min_ps;
    static constexpr auto max = simde_mm256_max_ps;
};


//double avx2
template <>
struct SimdTraits_avx2<double> {
	using Type = simde__m256d;
	static constexpr size_t pack_size = 4;
	static constexpr auto load = simde_mm256_load_pd;
	static constexpr auto loadu = simde_mm256_loadu_pd;
	static constexpr auto load_masked = simde_mm256_maskload_pd;
	static constexpr auto broadcast = simde_mm256_broadcast_sd;
	static constexpr auto set = simde_mm256_set_pd;
	static constexpr auto set1 = simde_mm256_set1_pd; 
	static constexpr auto store = simde_mm256_store_pd;
	static constexpr auto storeu = simde_mm256_storeu_pd;
	static constexpr auto store_masked = simde_mm256_maskstore_pd;
	static constexpr auto zero = simde_mm256_setzero_pd;
	static constexpr auto divide = simde_mm256_div_pd;
	static constexpr auto subtract = simde_mm256_sub_pd;
	static constexpr auto add = simde_mm256_add_pd;
	static constexpr auto multiply = simde_mm256_mul_pd;
	inline static constexpr auto negative = [](const Type& a) noexcept -> Type{
		return subtract(zero(), a);
	};

	//svml exponent functions
	inline static constexpr auto reciprical = [](Type a) noexcept -> Type {return divide(set1(1.0), a);};
	static constexpr auto exp = simde_mm256_exp_pd;
	static constexpr auto pow = simde_mm256_pow_pd;
	static constexpr auto sqrt = simde_mm256_sqrt_pd;
	static constexpr auto invsqrt = simde_mm256_invsqrt_pd;

        //trig svml functions
	static constexpr auto tanh = simde_mm256_tanh_pd;
	static constexpr auto tan = simde_mm256_tan_pd;
	static constexpr auto atanh = simde_mm256_atanh_pd;
	static constexpr auto atan = simde_mm256_atan_pd;
	inline static constexpr auto cotanh = [](const Type& a) noexcept -> Type { return reciprical(tanh(a));};
	inline static constexpr auto cotan = [](const Type& a) noexcept -> Type { return reciprical(tan(a));};
	static constexpr auto sinh = simde_mm256_sinh_pd;
	static constexpr auto sin = simde_mm256_sin_pd;
	static constexpr auto asinh = simde_mm256_asinh_pd;
	static constexpr auto asin = simde_mm256_asin_pd;
	inline static constexpr auto csch = [](const Type& a) noexcept -> Type { return reciprical(sinh(a));};
	inline static constexpr auto csc = [](const Type& a) noexcept -> Type { return reciprical(sin(a));};
	static constexpr auto cosh = simde_mm256_cosh_pd;
	static constexpr auto cos = simde_mm256_cos_pd;
	static constexpr auto acosh = simde_mm256_acosh_pd;
	static constexpr auto acos = simde_mm256_acos_pd;
	inline static constexpr auto sech = [](const Type& a) noexcept -> Type { return reciprical(cosh(a));};
	inline static constexpr auto sec = [](const Type& a) noexcept -> Type { return reciprical(cos(a));};
	static constexpr auto log = simde_mm256_log_pd;

    // svml round functions
    static constexpr auto round = simde_mm256_round_pd;
    static constexpr auto floor = simde_mm256_floor_pd;
    static constexpr auto ceil = simde_mm256_ceil_pd;

    inline static constexpr auto remainder(const Type& a, const Type& b){
        return subtract(a, multiply(floor(divide(a, b)), b));
    }
    inline static constexpr auto fmod(const Type& a, const Type& b){
        return subtract(a, multiply(round(divide(a, b), SIMDE_MM_FROUND_TO_ZERO), b));
    }
	/* inline static constexpr auto modulo = [](const Type& divisor_c, const Type& dividend_c) noexcept -> Type{ */
	/* 	/1* static constexpr auto modulo = simde_mm_rem_epi64; *1/ */
	/* 	simde__m256i divisor = simde_mm256_cvtpd_epi64(divisor_c); */
	/* 	simde__m256i dividend = simde_mm256_cvtpd_epi64(dividend_c); */
	/* 	return simde_mm256_cvtepi64_pd(simde_mm256_rem_epi64(divisor, dividend)); */
	/* } */

    static constexpr auto compare = simde_mm256_cmp_pd;
    static constexpr auto move_mask = simde_mm256_movemask_pd;
    NT_FLOAT_COMPARE_OPS(double);

	static constexpr auto fmsub = simde_mm256_fmsub_pd;
	inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c){
#if defined(__FMA__) || defined(SIMDE_X86_FMA)
		c = simde_mm256_fmadd_pd(a, b, c);
#else
		c = simde_mm256_add_pd(simde_mm256_mul_pd(a,b),c);
#endif 
	};
	//https://stackoverflow.com/questions/49941645/get-sum-of-values-stored-in-m256d-with-sse-avx
	inline static constexpr auto sum = [](const Type& x) -> double {
		simde__m128d low = simde_mm256_castpd256_pd128(x);
		simde__m128d high = simde_mm256_extractf128_pd(x, 1);
			     low = simde_mm_add_pd(low, high);
		simde__m128d high64 = simde_mm_unpackhi_pd(low, low);
		return simde_mm_cvtsd_f64(simde_mm_add_sd(low, high64));
	};
    static constexpr auto min = simde_mm256_min_pd;
    static constexpr auto max = simde_mm256_max_pd;
};



//example use: NT_INTEGER_SIMDE_AVX256_STORE_COMPARE_EQUAL(int32_t, epi32) 
#define NT_INTEGER_SIMDE_AVX256_STORE_COMPARE_EQUAL(type, code)\
    static constexpr auto compare_equal = NT_MP_AVX2_CONCAT(simde_mm256_cmpeq_, code);\
    static constexpr auto compare_greater_than = NT_MP_AVX2_CONCAT(simde_mm256_cmpgt_, code);\
    inline static constexpr auto store_compare_equal = [](const Type& a, const Type& b, bool* out) noexcept {\
        type vals[SimdTraits_avx2<type>::pack_size];\
        simde_mm256_storeu_si256((simde__m256i*)vals, SimdTraits_avx2<type>::compare_equal(a, b));\
        for(int i = 0; i < SimdTraits_avx2<type>::pack_size; ++i){\
            out[i] = (vals[i] == -1);\
        }\
    };\
    inline static constexpr auto store_compare_not_equal = [](const Type& a, const Type& b, bool* out) noexcept {\
        type vals[SimdTraits_avx2<type>::pack_size];\
        simde_mm256_storeu_si256((simde__m256i*)vals, SimdTraits_avx2<type>::compare_equal(a, b));\
        for(int i = 0; i < SimdTraits_avx2<type>::pack_size; ++i){\
            out[i] = (vals[i] != -1);\
        }\
    };\
    inline static constexpr auto store_compare_less_than_equal = [](const Type& a, const Type& b, bool* out) noexcept {\
        auto eq_mask = SimdTraits_avx2<type>::compare_equal(a, b); \
        auto lt_mask = SimdTraits_avx2<type>::compare_greater_than(b, a);\
        auto le_mask = simde_mm256_or_si256(eq_mask, lt_mask);\
        type vals[SimdTraits_avx2<type>::pack_size];\
        simde_mm256_storeu_si256((simde__m256i*)vals, le_mask);\
        for(int i = 0; i < SimdTraits_avx2<type>::pack_size; ++i){\
            out[i] = (vals[i] == -1);\
        }\
    };\

//int32 avx2
template <>
struct SimdTraits_avx2<int32_t> {
    using Type = simde__m256i;
    static constexpr size_t pack_size = 8;
    static constexpr auto load = simde_mm256_load_si256;
    static constexpr auto loadu = simde_mm256_loadu_si256;
    static constexpr auto load_masked = simde_mm256_maskload_epi32;
    static constexpr auto set = simde_mm256_set_epi32;
    static constexpr auto set1 = simde_mm256_set1_epi32;
    static constexpr auto store = simde_mm256_store_si256;
    static constexpr auto storeu = simde_mm256_storeu_si256;
    static constexpr auto store_masked = simde_mm256_maskstore_epi32;
    static constexpr auto zero = simde_mm256_setzero_si256;
    static constexpr auto add = simde_mm256_add_epi32;
    static constexpr auto multiply = simde_mm256_mullo_epi32;
    static constexpr auto divide = simde_mm256_div_epi32;
    static constexpr auto subtract = simde_mm256_sub_epi32;
    // static constexpr auto compare_equal = simde_mm256_cmpeq_epi32;
    NT_INTEGER_SIMDE_AVX256_STORE_COMPARE_EQUAL(int32_t, epi32)
 
    inline static constexpr auto negative = [](const Type& a) noexcept -> Type{
	return subtract(zero(), a);
    };

    /* static constexpr auto modulo = simde_mm256_rem_epi32; */
    inline static constexpr auto broadcast = [](const int32_t* arr) -> Type {return SimdTraits_avx2<int32_t>::set1(*arr);};
    inline static constexpr auto fmsub = [](const Type& a, const Type& b, Type& c){return subtract(c, multiply(a, b));};
    inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c){
	c = simde_mm256_add_epi32(simde_mm256_mullo_epi32(a, b), c);
    };
    inline static constexpr auto sum = [](const Type& x) -> int32_t{
	simde__m128i low = simde_mm256_castsi256_si128(x);
	simde__m128i high = simde_mm256_extractf128_si256(x, 1);
	simde__m128i sum128 = simde_mm_add_epi32(low, high);
	simde__m128i sum64 = simde_mm_hadd_epi32(sum128, sum128);
	return simde_mm_extract_epi32(sum64, 0) + simde_mm_extract_epi32(sum64, 1);
    };
    
    static constexpr auto min = simde_mm256_min_epi32;
    static constexpr auto max = simde_mm256_max_epi32;

};


//uint32 avx2
template <>
struct SimdTraits_avx2<uint32_t> {
    using Type = simde__m256i;
    static constexpr size_t pack_size = 8;
    static constexpr auto load = simde_mm256_load_si256;
    static constexpr auto loadu = simde_mm256_loadu_si256;
    static constexpr auto load_masked = simde_mm256_maskload_epi32;
    static constexpr auto set = simde_mm256_set_epi32;
    static constexpr auto set1 = simde_mm256_set1_epi32;
    static constexpr auto store = simde_mm256_store_si256;
    static constexpr auto storeu = simde_mm256_storeu_si256;
    static constexpr auto store_masked = simde_mm256_maskstore_epi32;
    static constexpr auto zero = simde_mm256_setzero_si256;
    static constexpr auto add = simde_mm256_add_epi32;
    static constexpr auto multiply = simde_mm256_mullo_epi32;
    static constexpr auto divide = simde_mm256_div_epu32;
    static constexpr auto subtract = simde_mm256_sub_epi32;
    NT_INTEGER_SIMDE_AVX256_STORE_COMPARE_EQUAL(int32_t, epi32);
    /* static constexpr auto modulo = simde_mm256_rem_epu32; */
    inline static constexpr auto broadcast = [](const uint32_t* arr) -> Type {return SimdTraits_avx2<uint32_t>::set1(*arr);};
    inline static constexpr auto fmsub = [](const Type& a, const Type& b, Type& c){return subtract(c, multiply(a, b));};
    inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c){
	c = simde_mm256_add_epi32(simde_mm256_mullo_epi32(a, b), c);
    };
    inline static constexpr auto sum = [](const Type& x) -> uint32_t{
	simde__m128i low = simde_mm256_castsi256_si128(x);
	simde__m128i high = simde_mm256_extractf128_si256(x, 1);
	simde__m128i sum128 = simde_mm_add_epi32(low, high);
	simde__m128i sum64 = simde_mm_hadd_epi32(sum128, sum128);
	return simde_mm_extract_epi32(sum64, 0) + simde_mm_extract_epi32(sum64, 1);
    };
    static constexpr auto min = simde_mm256_min_epu32;
    static constexpr auto max = simde_mm256_max_epu32;

};


template <>
struct SimdTraits_avx2<int64_t> {
    using Type = simde__m256i;
    static constexpr size_t pack_size = 4;
    static constexpr auto load = simde_mm256_load_si256;
    static constexpr auto loadu = simde_mm256_loadu_si256;
    static constexpr auto load_masked = simde_mm256_maskload_epi64;
    static constexpr auto set = simde_mm256_set_epi64x;
    static constexpr auto set1 = simde_mm256_set1_epi64x;
    static constexpr auto store = simde_mm256_store_si256;
    static constexpr auto storeu = simde_mm256_storeu_si256;
    static constexpr auto store_masked = simde_mm256_maskstore_epi64;
    static constexpr auto zero = simde_mm256_setzero_si256;
    static constexpr auto add = simde_mm256_add_epi64;
    static constexpr auto divide = simde_mm256_div_epi64;
    static constexpr auto subtract = simde_mm256_sub_epi64;
    // static constexpr auto compare_equal = simde_mm256_cmpeq_epi64;
    NT_INTEGER_SIMDE_AVX256_STORE_COMPARE_EQUAL(int64_t, epi64)
    inline static constexpr auto negative = [](const Type& a) noexcept -> Type{
	return subtract(zero(), a);
    };

    //gotten from https://stackoverflow.com/questions/76436053/simd-intrinsics-avx-tried-to-use-mm256-mullo-epi64-but-got-0xc000001d-illega
    //mull64 haswell
    inline static constexpr auto multiply = [](const Type& a, const Type& b) -> Type{
        alignas(32) int64_t a_vals[4], b_vals[4], result[4];
        simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(a_vals), a);
        simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(b_vals), b);
        result[0] = a_vals[0] * b_vals[0];
        result[1] = a_vals[1] * b_vals[1];
        result[2] = a_vals[2] * b_vals[2];
        result[3] = a_vals[3] * b_vals[3];

        return simde_mm256_load_si256(reinterpret_cast<const simde__m256i*>(result));
    };

    /* static constexpr auto modulo = simde_mm256_rem_epi64; */
    inline static constexpr auto broadcast = [](const int64_t* arr) -> Type {return simde_mm256_set1_epi64x(*arr);};
    inline static constexpr auto fmsub = [](const Type& a, const Type& b, Type& c){return subtract(c, multiply(a, b));};
    inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c){
	c = simde_mm256_add_epi64(SimdTraits_avx2<int64_t>::multiply(a, b), c);
    };
    inline static constexpr auto sum = [](const Type& x) -> int64_t {
	simde__m128i sum128 = simde_mm_add_epi64(simde_mm256_extracti128_si256(x, 0), simde_mm256_extracti128_si256(x, 1));
	return simde_mm_extract_epi64(sum128, 0) + simde_mm_extract_epi64(sum128, 1);
    };
    inline static constexpr auto min = [](const Type& a, const Type& b) noexcept -> Type{
        __m256i mask = _mm256_cmpgt_epi64(a, b);
        return _mm256_blendv_epi8(a, b, mask);
    };
    inline static constexpr auto max = [](const Type& a, const Type& b) noexcept -> Type{
        __m256i mask = _mm256_cmpgt_epi64(a, b);
        return _mm256_blendv_epi8(b, a, mask);
    };

};

template <>
struct SimdTraits_avx2<uint64_t> {
    using Type = simde__m256i;
    static constexpr size_t pack_size = 4;
    static constexpr auto load = simde_mm256_load_si256;
    static constexpr auto loadu = simde_mm256_loadu_si256;
    static constexpr auto load_masked = simde_mm256_maskload_epi64;
    static constexpr auto set = simde_mm256_set_epi64x;
    static constexpr auto set1 = simde_mm_set1_epi64x;
    static constexpr auto store = simde_mm256_store_si256;
    static constexpr auto storeu = simde_mm256_storeu_si256;
    static constexpr auto store_masked = simde_mm256_maskstore_epi64;
    static constexpr auto zero = simde_mm256_setzero_si256;
    static constexpr auto add = simde_mm_add_epi64;
    // static constexpr auto compare_equal = simde_mm256_cmpeq_epi64;
    NT_INTEGER_SIMDE_AVX256_STORE_COMPARE_EQUAL(int64_t, epi64)
    inline static constexpr auto multiply = [](const Type& a, const Type& b) noexcept -> Type{
	    simde__m256i bswap   = simde_mm256_shuffle_epi32(b,0xB1);
	    simde__m256i prodlh  = simde_mm256_mullo_epi32(a,bswap);

	    simde__m256i prodlh2 = simde_mm256_srli_epi64(prodlh, 32);
	    simde__m256i prodlh3 = simde_mm256_add_epi32(prodlh2, prodlh);
	    simde__m256i prodlh4 = simde_mm256_and_si256(prodlh3, simde_mm256_set1_epi64x(0x00000000FFFFFFFF));

	    simde__m256i prodll  = simde_mm256_mul_epu32(a,b);
	    simde__m256i prod    = simde_mm256_add_epi64(prodll,prodlh4);
	    return  prod;
    };
    static constexpr auto divide = simde_mm256_div_epu64;
    static constexpr auto subtract = simde_mm256_sub_epi64;

    /* static constexpr auto modulo = simde_mm256_rem_epu64; */
    inline static constexpr auto broadcast = [](const uint64_t* arr) -> Type {return simde_mm256_set1_epi64x(*arr);};
    inline static constexpr auto fmsub = [](const Type& a, const Type& b, Type& c){return subtract(c, multiply(a, b));};
    inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c){
	c = simde_mm256_add_epi64(SimdTraits_avx2<uint64_t>::multiply(a, b), c);
    };
    inline static constexpr auto sum = [](const Type& x) -> uint64_t {
	simde__m128i sum128 = simde_mm_add_epi64(simde_mm256_extracti128_si256(x, 0), simde_mm256_extracti128_si256(x, 1));
	return (uint64_t)simde_mm_extract_epi64(sum128, 0) + (uint64_t)simde_mm_extract_epi64(sum128, 1);
    };
    inline static constexpr auto min = [](const Type& a, const Type& b) noexcept -> Type{
        __m256i mask = _mm256_cmpgt_epi64(a, b);
        return _mm256_blendv_epi8(a, b, mask);
    };
    inline static constexpr auto max = [](const Type& a, const Type& b) noexcept -> Type{
        __m256i mask = _mm256_cmpgt_epi64(a, b);
        return _mm256_blendv_epi8(b, a, mask);
    };

};

template <>
struct SimdTraits_avx2<int8_t> {
    using Type = simde__m256i;
    static constexpr size_t pack_size = 32;
    static constexpr auto load = simde_mm256_load_si256;
    static constexpr auto loadu = simde_mm256_loadu_si256;
    inline static constexpr auto load_masked = [](const int8_t* data, const simde__m256i& mask) noexcept -> Type{
	int8_t cpy_data[pack_size]; //make sure there is not a segmentation fault
	int8_t mask_data[pack_size];
	simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(mask_data), mask);
	for(size_t i = 0; i < pack_size; ++i){
		if(mask_data[i] != 0){//this ensires no segmentation faults if data is smaller than 32
			cpy_data[i] = data[i];
		}
	}
	//now mask it
	simde__m256i loaded_data = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(cpy_data));
	return simde_mm256_and_si256(loaded_data, mask);
    };

    static constexpr auto set = simde_mm256_set_epi8;
    static constexpr auto set1 = simde_mm256_set1_epi8;
    static constexpr auto store = simde_mm256_store_si256;
    static constexpr auto storeu = simde_mm256_storeu_si256;
    inline static constexpr auto store_masked = [](int8_t* data, const simde__m256i& mask_data, const Type& vector) noexcept {
	//pretty much have to do this to avoid segmentation faults
	int8_t values_arr[pack_size];
	simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(values_arr), vector);
	int8_t mask_arr[pack_size];
	simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(mask_arr), mask_data);
	for(size_t i = 0; i < pack_size; ++i){
		if(mask_arr[i] != 0){
			data[i] = values_arr[i];
		}
	}
    };

    static constexpr auto zero = simde_mm256_setzero_si256;
    static constexpr auto add = simde_mm256_add_epi8;
    // static constexpr auto compare_equal = simde_mm256_cmpeq_epi8;
    NT_INTEGER_SIMDE_AVX256_STORE_COMPARE_EQUAL(int8_t, epi8)
    //https://github.com/lemire/vectorclass/blob/master/vectori256.h#L284
    inline static constexpr auto multiply = [](simde__m256i a, simde__m256i b) noexcept -> Type{
	simde__m256i aodd = simde_mm256_srli_epi16(a,8);
	simde__m256i bodd = simde_mm256_srli_epi16(b,8);
	simde__m256i muleven = simde_mm256_mullo_epi16(a,b); //product of even number elements
	simde__m256i mulodd = simde_mm256_mullo_epi16(aodd, bodd);
		     mulodd = simde_mm256_slli_epi16(mulodd, 8); // put odd numbered elements back in place
	simde__m256i mask    = simde_mm256_set1_epi32(0x00FF00FF);
	return simde_mm256_blendv_epi8(mulodd, muleven, mask);
    };
    static constexpr auto divide = simde_mm256_div_epi8;
    static constexpr auto subtract = simde_mm256_sub_epi8;
    inline static constexpr auto negative = [](const Type& a) noexcept -> Type{
	return subtract(zero(), a);
    };

    /* static constexpr auto modulo = simde_mm256_rem_epi8; */
    inline static constexpr auto broadcast = [](const int8_t* arr) -> Type {return simde_mm256_set1_epi8(*arr);};
    inline static constexpr auto fmsub = [](const Type& a, const Type& b, Type& c){return subtract(c, multiply(a, b));};
    inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c){
	c = simde_mm256_add_epi8(SimdTraits_avx2<int8_t>::multiply(a, b), c);
    };
    //https://github.com/lemire/vectorclass/blob/master/vectori256.h#L731 for the following specific function
    inline static constexpr auto sum = [](const Type& a) noexcept -> int32_t {
	simde__m256i sum1 = simde_mm256_sad_epu8(a,simde_mm256_setzero_si256());
	simde__m256i sum2 = simde_mm256_shuffle_epi32(sum1,2);
	simde__m256i sum3 = simde_mm256_add_epi16(sum1,sum2);
	simde__m128i sum4 = simde_mm256_extracti128_si256(sum3,1);
	simde__m128i sum5 = simde_mm_add_epi16(simde_mm256_castsi256_si128(sum3),sum4);
	int8_t  sum6 = (int8_t)_mm_cvtsi128_si32(sum5); //truncate
	return sum6;
    };
    static constexpr auto min = simde_mm256_min_epi8;
    static constexpr auto max = simde_mm256_max_epi8;

};


template <>
struct SimdTraits_avx2<uint8_t> {
    using Type = simde__m256i;
    static constexpr size_t pack_size = 32;
    static constexpr auto load = simde_mm256_load_si256;
    static constexpr auto loadu = simde_mm256_loadu_si256;
    static constexpr auto load_masked = SimdTraits_avx2<int8_t>::load_masked;
    /* inline static constexpr auto load_masked = [](const uint8_t* data, const simde__m256i& mask) noexcept -> Type{ */
	/* return SimdTraits_avx2<int8_t>::load_masked(reinterpret_cast<const int8_t*>(data), mask); */
	/* /1* simde__m256i loaded_data = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(data)); *1/ */
	/* /1* simde__m256i result = simde_mm256_and_si256(loaded_data, mask); *1/ */
	/* /1* return result; *1/ */
    /* }; */

    static constexpr auto set = simde_mm256_set_epi8;
    static constexpr auto set1 = simde_mm256_set1_epi8;
    static constexpr auto store = simde_mm256_store_si256;
    static constexpr auto storeu = simde_mm256_storeu_si256;
    static constexpr auto store_masked = SimdTraits_avx2<int8_t>::store_masked;
    // static constexpr auto compare_equal = simde_mm256_cmpeq_epi8;
    NT_INTEGER_SIMDE_AVX256_STORE_COMPARE_EQUAL(int8_t, epi8)
    /* inline static constexpr auto store_masked = [](uint8_t* data, const simde__m256i& mask_data, const Type& vector) noexcept { */
	/* SimdTraits_avx2<int8_t>::store_masked(reinterpret_cast<int8_t*>(data), mask_data, vector); */
	/* /1* simde__m256i data_vector = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(data)); *1/ */
	/* /1* simde__m256i result_data = simde_mm256_and_si256(data_vector, simde_mm256_xor_si256(mask_data, simde_mm256_cmpeq_epi8(mask_data,mask_data))); *1/ */
	/* /1* simde__m256i result_vector = simde_mm256_and_si256(vector, mask_data); *1/ */
	/* /1*              result_data = simde_mm256_add_epi8(result_data, result_vector); *1/ */
	/* /1* simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(data), result_data); *1/ */
    /* }; */

    static constexpr auto zero = simde_mm256_setzero_si256;
    static constexpr auto add = simde_mm256_add_epi8;
    //https://github.com/lemire/vectorclass/blob/master/vectori256.h#L284
    inline static constexpr auto multiply = [](simde__m256i a, simde__m256i b) noexcept -> Type{
	simde__m256i aodd = simde_mm256_srli_epi16(a,8);
	simde__m256i bodd = simde_mm256_srli_epi16(b,8);
	simde__m256i muleven = simde_mm256_mullo_epi16(a,b); //product of even number elements
	simde__m256i mulodd = simde_mm256_mullo_epi16(aodd, bodd);
		     mulodd = simde_mm256_slli_epi16(mulodd, 8); // put odd numbered elements back in place
	simde__m256i mask    = simde_mm256_set1_epi32(0x00FF00FF);
	return simde_mm256_blendv_epi8(mulodd, muleven, mask);
    };
    static constexpr auto divide = simde_mm256_div_epu8;
    static constexpr auto subtract = simde_mm256_sub_epi8;


    /* static constexpr auto modulo = simde_mm256_rem_epu8; */
    inline static constexpr auto broadcast = [](const uint8_t* arr) -> Type {return simde_mm256_set1_epi8(*arr);};
    inline static constexpr auto fmsub = [](const Type& a, const Type& b, Type& c){return subtract(c, multiply(a, b));};
    inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c){
	c = simde_mm256_add_epi8(SimdTraits_avx2<uint8_t>::multiply(a, b), c);
    };

    static constexpr auto sum = SimdTraits_avx2<int8_t>::sum;
    static constexpr auto min = simde_mm256_min_epu8;
    static constexpr auto max = simde_mm256_max_epu8;

};


template <>
struct SimdTraits_avx2<int16_t> {
    using Type = simde__m256i;
    static constexpr size_t pack_size = 16;
    static constexpr auto load = simde_mm256_load_si256;
    static constexpr auto loadu = simde_mm256_loadu_si256;
    inline static constexpr auto load_masked = [](const int16_t* data, const simde__m256i& mask) noexcept -> Type{
	int16_t cpy_data[pack_size]; //make sure there is not a segmentation fault
	int16_t mask_data[pack_size];
	simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(mask_data), mask);
	for(size_t i = 0; i < pack_size; ++i){
		if(mask_data[i] != 0){//this ensires no segmentation faults if data is smaller than 32
			cpy_data[i] = data[i];
		}
	}
	//now mask it
	simde__m256i loaded_data = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(cpy_data));
	return simde_mm256_and_si256(loaded_data, mask);
    };

    /* inline static constexpr auto load_masked = [](const int16_t* data, const simde__m256i& mask) noexcept -> Type{ */
	/* simde__m256i loaded_data = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(data)); */
	/* simde__m256i result = simde_mm256_and_si256(loaded_data, mask); */
	/* return result; */
    /* }; */
    static constexpr auto set = simde_mm256_set_epi16;
    static constexpr auto set1 = simde_mm256_set1_epi16;
    static constexpr auto store = simde_mm256_store_si256;
    static constexpr auto storeu = simde_mm256_storeu_si256;
    // static constexpr auto compare_equal = simde_mm256_cmpeq_epi16;
    NT_INTEGER_SIMDE_AVX256_STORE_COMPARE_EQUAL(int16_t, epi16)
    inline static constexpr auto store_masked = [](int16_t* data, const simde__m256i& mask_data, const Type& vector) noexcept {
	//pretty much have to do this to avoid segmentation faults
	int16_t values_arr[pack_size];
	simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(values_arr), vector);
	int16_t mask_arr[pack_size];
	simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(mask_arr), mask_data);
	for(size_t i = 0; i < pack_size; ++i){
		if(mask_arr[i] != 0){
			data[i] = values_arr[i];
		}
	}
    };

    /* inline static constexpr auto store_masked = [](int16_t* data, const simde__m256i& mask_data, const Type& vector) noexcept { */
	/* simde__m256i data_vector = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(data)); */
	/* simde__m256i result_data = simde_mm256_and_si256(data_vector, simde_mm256_xor_si256(mask_data, simde_mm256_cmpeq_epi16(mask_data,mask_data))); */
	/* simde__m256i result_vector = simde_mm256_and_si256(vector, mask_data); */
	             /* result_data = simde_mm256_add_epi16(result_data, result_vector); */
	/* simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(data), result_data); */
    /* }; */
    static constexpr auto zero = simde_mm256_setzero_si256;
    static constexpr auto add = simde_mm256_add_epi16;
    static constexpr auto multiply = simde_mm256_mullo_epi16;
    static constexpr auto divide = simde_mm256_div_epi16;
    static constexpr auto subtract = simde_mm256_sub_epi16;
    inline static constexpr auto negative = [](const Type& a) noexcept -> Type{
	return subtract(zero(), a);
    };

    /* static constexpr auto modulo = simde_mm256_rem_epi16; */
    inline static constexpr auto broadcast = [](const int16_t* arr) -> Type {return simde_mm256_set1_epi16(*arr);};
    inline static constexpr auto fmsub = [](const Type& a, const Type& b, Type& c){return subtract(c, multiply(a, b));};
    inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c){
	c = simde_mm256_add_epi16(simde_mm256_mullo_epi16(a, b), c);
    };
    //the following is from https://github.com/lemire/vectorclass/blob/master/vectori128.h#L1529
    inline static constexpr auto sum_avx = [](const simde__m128i& a) noexcept -> int32_t{
	simde__m128i sum1  = simde_mm_shuffle_epi32(a,0x0E);
	simde__m128i sum2  = simde_mm_add_epi16(a,sum1);
	simde__m128i sum3  = simde_mm_shuffle_epi32(sum2,0x01);
	simde__m128i sum4  = simde_mm_add_epi16(sum2,sum3);
	simde__m128i sum5  = simde_mm_shufflelo_epi16(sum4,0x01);
	simde__m128i sum6  = simde_mm_add_epi16(sum4,sum5);
	int16_t sum7  = simde_mm_cvtsi128_si32(sum6);
	return  sum7;
    };
    inline static constexpr auto sum = [](const Type& vector) noexcept -> int32_t {
	simde__m128i lo8 = simde_mm256_castsi256_si128(vector);
	simde__m128i hi8 = simde_mm256_extracti128_si256(vector, 1);
	return SimdTraits_avx2<int16_t>::sum_avx(lo8) + SimdTraits_avx2<int16_t>::sum_avx(hi8);
    };

    static constexpr auto min = simde_mm256_min_epi16;
    static constexpr auto max = simde_mm256_max_epi16;

};

template <>
struct SimdTraits_avx2<uint16_t> {
    using Type = simde__m256i;
    static constexpr size_t pack_size = 16;
    static constexpr auto load = simde_mm256_load_si256;
    static constexpr auto loadu = simde_mm256_loadu_si256;
    static constexpr auto load_masked = SimdTraits_avx2<int16_t>::load_masked;
    /* inline static constexpr auto load_masked = [](const uint16_t* data, const simde__m256i& mask) noexcept -> Type{ */
	/* return SimdTraits_avx2<int16_t>::load_masked(reinterpret_cast<const int16_t*>(data), mask); */	
    /* }; */
    static constexpr auto set = simde_mm256_set_epi16;
    static constexpr auto set1 = simde_mm256_set1_epi16;
    static constexpr auto store = simde_mm256_store_si256;
    static constexpr auto storeu = simde_mm256_storeu_si256;
    static constexpr auto store_masked = SimdTraits_avx2<int16_t>::store_masked;
    /* inline static constexpr auto store_masked = [](uint16_t* data, const simde__m256i& mask_data, const Type& vector) noexcept { */
	    /* SimdTraits_avx2<int16_t>::store_masked(reinterpret_cast<int16_t*>(data), mask_data, vector); */
    /* }; */
    static constexpr auto zero = simde_mm256_setzero_si256;
    static constexpr auto add = simde_mm256_add_epi16;
    static constexpr auto multiply = simde_mm256_mullo_epi16;
    static constexpr auto divide = simde_mm256_div_epu16;
    static constexpr auto subtract = simde_mm256_sub_epi16;
    // static constexpr auto compare_equal = simde_mm256_cmpeq_epi16;
    NT_INTEGER_SIMDE_AVX256_STORE_COMPARE_EQUAL(int16_t, epi16)
    /* static constexpr auto modulo = simde_mm256_rem_epu16; */
    inline static constexpr auto broadcast = [](const uint16_t* arr) -> Type {return simde_mm256_set1_epi16(*arr);};
    inline static constexpr auto fmsub = [](const Type& a, const Type& b, Type& c){return subtract(c, multiply(a, b));};
    inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c){
	c = simde_mm256_add_epi16(simde_mm256_mullo_epi16(a, b), c);
    };
    //the following is from https://github.com/lemire/vectorclass/blob/master/vectori128.h#L1529
    inline static constexpr auto sum_avx = [](const simde__m128i& a) noexcept -> int32_t{
	simde__m128i sum1  = simde_mm_shuffle_epi32(a,0x0E);
	simde__m128i sum2  = simde_mm_add_epi16(a,sum1);
	simde__m128i sum3  = simde_mm_shuffle_epi32(sum2,0x01);
	simde__m128i sum4  = simde_mm_add_epi16(sum2,sum3);
	simde__m128i sum5  = simde_mm_shufflelo_epi16(sum4,0x01);
	simde__m128i sum6  = simde_mm_add_epi16(sum4,sum5);
	uint16_t sum7  = simde_mm_cvtsi128_si32(sum6);
	return  sum7;
    };
    inline static constexpr auto sum = [](const Type& vector) noexcept -> int32_t {
	simde__m128i lo8 = simde_mm256_castsi256_si128(vector);
	simde__m128i hi8 = simde_mm256_extracti128_si256(vector, 1);
	return SimdTraits_avx2<uint16_t>::sum_avx(lo8) + SimdTraits_avx2<uint16_t>::sum_avx(hi8);
    };
    
    static constexpr auto min = simde_mm256_min_epu16;
    static constexpr auto max = simde_mm256_max_epu16;

};


#undef NT_INTEGER_SIMDE_AVX256_STORE_COMPARE_EQUAL

#define NT_COMPLEX_COMPARE_OPS(type)\
    inline static constexpr auto less_than_equal = [](const Type& a, const Type& b) noexcept -> Type {return SimdTraits_avx2<type>::compare(a, b, SIMDE_CMP_LE_OQ);};\
    inline static constexpr auto compare_equal = [](const Type& a, const Type& b) noexcept -> Type {return SimdTraits_avx2<type>::compare(a, b, SIMDE_CMP_EQ_OQ);};\
    inline static constexpr auto compare_not_equal = [](const Type& a, const Type& b) noexcept -> Type {return SimdTraits_avx2<type>::compare(a, b, SIMDE_CMP_NEQ_UQ);};\
    inline static constexpr auto store_compare_equal = [](const Type& a, const Type& b, bool* out_bool){\
        int mask = SimdTraits_avx2<type>::move_mask(SimdTraits_avx2<type>::compare(a, b, SIMDE_CMP_EQ_OQ));\
        for(int i = 0; i < SimdTraits_avx2<type>::pack_size; ++i){\
            out_bool[i] = ((mask >> (i * 2)) & 1) && ((mask >> (i * 2 + 1)) & 1);\
        }\
    };\
    inline static constexpr auto store_compare_not_equal = [](const Type& a, const Type& b, bool* out_bool){\
        int mask = SimdTraits_avx2<type>::move_mask(SimdTraits_avx2<type>::compare(a, b, SIMDE_CMP_NEQ_UQ));\
        for(int i = 0; i < SimdTraits_avx2<type>::pack_size; ++i){\
            out_bool[i] = ((mask >> (i * 2)) & 1) || ((mask >> (i * 2 + 1)) & 1);\
        }\
    };\
    inline static constexpr auto store_compare_less_than_equal = [](const Type& a, const Type& b, bool* out_bool){\
        int mask = SimdTraits_avx2<type>::move_mask(SimdTraits_avx2<type>::compare(a, b, SIMDE_CMP_LE_OQ));\
        for(int i = 0; i < SimdTraits_avx2<type>::pack_size; ++i){\
            out_bool[i] = ((mask >> (i * 2)) & 1) && ((mask >> (i * 2 + 1)) & 1);\
        }\
    };\

template <>
struct SimdTraits_avx2<complex_64> {
using Type = simde__m256;
	static constexpr size_t pack_size = 4;  // AVX2 can handle 8 floats -> 4 complex floats
	inline static constexpr auto load = [](const complex_64* arr) noexcept -> Type {
	    return simde_mm256_load_ps(reinterpret_cast<const float*>(arr));
	};
	inline static constexpr auto loadu = [](const complex_64* arr) noexcept -> Type {
	    return simde_mm256_loadu_ps(reinterpret_cast<const float*>(arr));
	};
	inline static constexpr auto load_masked = [](const complex_64* arr, const simde__m256i& mask) noexcept -> Type {
	    return simde_mm256_maskload_ps(reinterpret_cast<const float*>(arr), mask);
	};
	//(comp4.re, comp4.im, comp3.re, comp3.im, comp2.re, comp2.im, comp1.re, comp1.im) 
	inline static constexpr auto set = [](const complex_64& comp1, const complex_64& comp2, const complex_64& comp3, const complex_64& comp4) 
	    noexcept -> Type {
		return simde_mm256_set_ps(
			std::get<1>(static_cast<const my_complex<float>&>(comp1)), std::get<0>(static_cast<const my_complex<float>&>(comp1)),
			std::get<1>(static_cast<const my_complex<float>&>(comp2)), std::get<0>(static_cast<const my_complex<float>&>(comp2)),
			std::get<1>(static_cast<const my_complex<float>&>(comp3)), std::get<0>(static_cast<const my_complex<float>&>(comp3)),
			std::get<1>(static_cast<const my_complex<float>&>(comp4)), std::get<0>(static_cast<const my_complex<float>&>(comp4)));
	};
	inline static constexpr auto set1 = [](const complex_64& comp) noexcept -> Type {return SimdTraits_avx2<complex_64>::set(comp, comp, comp, comp);};
	inline static constexpr auto broadcast = [](const complex_64* comp_arr) noexcept -> Type { return SimdTraits_avx2<complex_64>::set1(*comp_arr); }; 
	inline static constexpr auto store = [](complex_64* comp_arr, const Type& vec) noexcept { simde_mm256_store_ps(reinterpret_cast<float*>(comp_arr), vec); };
	inline static constexpr auto storeu = [](complex_64* comp_arr, const Type& vec) noexcept { simde_mm256_storeu_ps(reinterpret_cast<float*>(comp_arr), vec); };
	inline static constexpr auto store_masked = [](complex_64* comp_arr, const simde__m256i& mask, const Type& vec) noexcept {simde_mm256_maskstore_ps(reinterpret_cast<float*>(comp_arr), mask, vec);};
	static constexpr auto zero = simde_mm256_setzero_ps; //same
	//svml exponent functions
	static constexpr auto reciprical = simde_mm256_rcp_ps;
	static constexpr auto exp = simde_mm256_exp_ps;
	static constexpr auto pow = simde_mm256_pow_ps;
	static constexpr auto sqrt = simde_mm256_sqrt_ps;
	static constexpr auto invsqrt = simde_mm256_invsqrt_ps;

        //trig svml functions
	static constexpr auto tanh = simde_mm256_tanh_ps;
	static constexpr auto tan = simde_mm256_tan_ps;
	static constexpr auto atanh = simde_mm256_atanh_ps;
	static constexpr auto atan = simde_mm256_atan_ps;
	inline static constexpr auto cotanh = [](const Type& a) noexcept -> Type { return reciprical(tanh(a));};
	inline static constexpr auto cotan = [](const Type& a) noexcept -> Type { return reciprical(tan(a));};
	static constexpr auto sinh = simde_mm256_sinh_ps;
	static constexpr auto sin = simde_mm256_sin_ps;
	static constexpr auto asinh = simde_mm256_asinh_ps;
	static constexpr auto asin = simde_mm256_asin_ps;
	inline static constexpr auto csch = [](const Type& a) noexcept -> Type { return reciprical(sinh(a));};
	inline static constexpr auto csc = [](const Type& a) noexcept -> Type { return reciprical(sin(a));};
	static constexpr auto cosh = simde_mm256_cosh_ps;
	static constexpr auto cos = simde_mm256_cos_ps;
	static constexpr auto acosh = simde_mm256_acosh_ps;
	static constexpr auto acos = simde_mm256_acos_ps;
	inline static constexpr auto sech = [](const Type& a) noexcept -> Type { return reciprical(cosh(a));};
	inline static constexpr auto sec = [](const Type& a) noexcept -> Type { return reciprical(cos(a));};
	static constexpr auto log = simde_mm256_log_ps;

    // svml round functions
    static constexpr auto round = simde_mm256_round_ps;
    static constexpr auto floor = simde_mm256_floor_ps;
    static constexpr auto ceil = simde_mm256_ceil_ps;


	static constexpr auto subtract = simde_mm256_sub_ps;
	static constexpr auto divide = simde_mm256_div_ps;
	static constexpr auto add = simde_mm256_add_ps;
	static constexpr auto multiply = simde_mm256_mul_ps;
	inline static constexpr auto negative = [](const Type& a) noexcept -> Type{
		return subtract(zero(), a);
	};

    inline static constexpr auto remainder(const Type& a, const Type& b){
        return subtract(a, multiply(floor(divide(a, b)), b));
    }
    inline static constexpr auto fmod(const Type& a, const Type& b){
        return subtract(a, multiply(round(divide(a, b), SIMDE_MM_FROUND_TO_ZERO), b));
    }

    static constexpr auto compare = simde_mm256_cmp_ps;
    static constexpr auto move_mask = simde_mm256_movemask_ps;
    NT_COMPLEX_COMPARE_OPS(complex_64);

    static constexpr auto fmsub = simde_mm256_fmsub_ps;
    inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c){ //same
#if defined(__FMA__) || defined(SIMDE_X86_FMA)
		c = simde_mm256_fmadd_ps(a, b, c);
#else
		c = simde_mm256_add_ps(simde_mm256_mul_ps(a,b),c);
#endif //defined(__FMA__) || defined(SIMDE_X86_FMA)
	};
	inline static constexpr auto sum = [](simde__m256 x) -> complex_64{
		// x = ( x3.re, x3.im, x2.re, x2.im, x1.re, x1.im, x0.re, x0.im )
		// hiQuad = ( x3.re, x3.im, x2.re, x2.im )
		const simde__m128 hiQuad = simde_mm256_extractf128_ps(x, 1);
		// loQuad = ( x1.re, x1.im, x0.re, x0.im )
		const simde__m128 loQuad = simde_mm256_castps256_ps128(x);
		// sumQuad = ( x3.re + x1.re, x3.im + x1.im, x2.re + x0.re, x2.im + x0.im )
		const simde__m128 sumQuad = simde_mm_add_ps(loQuad, hiQuad);
		// loDual = ( -, -, x2.re + x0.re, x2.im + x0.im )
		const simde__m128 loDual = sumQuad;
		// hiDual = ( -, -, x3.re + x1.re, x3.im + x1.im )
		const simde__m128 hiDual = simde_mm_movehl_ps(sumQuad, sumQuad);
		// sumDual = ( -, -, x3.re + x1.re + x2.re + x0.re, x3.im + x1.im + x2.im + x0.im)
		const simde__m128 sumDual = simde_mm_add_ps(hiDual, loDual);
		complex_64 result[2];
		simde_mm_storeu_ps(reinterpret_cast<float*>(result), sumDual);
		return result[0];

	};
    static constexpr auto min = simde_mm256_min_ps;
    static constexpr auto max = simde_mm256_max_ps;

};


template <>
struct SimdTraits_avx2<complex_128> {
	using Type = simde__m256d;
	static constexpr size_t pack_size = 2;
	inline static constexpr auto load = [](const complex_128* arr) noexcept -> Type {return simde_mm256_load_pd(reinterpret_cast<const double*>(arr));};
	inline static constexpr auto loadu = [](const complex_128* arr) noexcept -> Type {return simde_mm256_loadu_pd(reinterpret_cast<const double*>(arr));};
	inline static constexpr auto load_masked = [](const complex_128* arr, const simde__m256i& mask) noexcept -> Type {
		return simde_mm256_maskload_pd(reinterpret_cast<const double*>(arr), mask);
	};
	inline static constexpr auto set = [](const complex_128& comp1, const complex_128& comp2) noexcept -> Type{
		return simde_mm256_set_pd(
			std::get<1>(static_cast<const my_complex<double>&>(comp1)), std::get<0>(static_cast<const my_complex<double>&>(comp1)),
			std::get<1>(static_cast<const my_complex<double>&>(comp2)), std::get<0>(static_cast<const my_complex<double>&>(comp2))
		);
	};
	inline static constexpr auto set1 = [](const complex_128& comp) noexcept -> Type {return SimdTraits_avx2<complex_128>::set(comp, comp);};
	inline static constexpr auto broadcast = [](const complex_128* comps) noexcept -> Type {return SimdTraits_avx2<complex_128>::set1(*comps);};
	inline static constexpr auto store = [](complex_128* comp_arr, const Type& vec) noexcept {  simde_mm256_store_pd(reinterpret_cast<double*>(comp_arr), vec); };
	inline static constexpr auto storeu = [](complex_128* comp_arr, const Type& vec) noexcept {  simde_mm256_storeu_pd(reinterpret_cast<double*>(comp_arr), vec); };
	static constexpr auto store_masked = [](complex_128* comp_arr, const simde__m256i& mask, const Type& vec) noexcept {
		simde_mm256_maskstore_pd(reinterpret_cast<double*>(comp_arr), mask, vec);
	};
	static constexpr auto zero = simde_mm256_setzero_pd;
	static constexpr auto divide = simde_mm256_div_pd;
	static constexpr auto subtract = simde_mm256_sub_pd;
	static constexpr auto add = simde_mm256_add_pd;
	static constexpr auto multiply = simde_mm256_mul_pd;
	inline static constexpr auto negative = [](const Type& a) noexcept -> Type{
		return subtract(zero(), a);
	};
	/* static constexpr auto modulo = SimdTraits_avx2<double>::modulo; */

	//svml exponent functions
	inline static constexpr auto reciprical = [](Type a) noexcept -> Type {return divide(set1(complex_128(1.0, 1.0)), a);};
	static constexpr auto exp = simde_mm256_exp_pd;
	static constexpr auto pow = simde_mm256_pow_pd;
	static constexpr auto sqrt = simde_mm256_sqrt_pd;
	static constexpr auto invsqrt = simde_mm256_invsqrt_pd;

        //trig svml functions
	static constexpr auto tanh = simde_mm256_tanh_pd;
	static constexpr auto tan = simde_mm256_tan_pd;
	static constexpr auto atanh = simde_mm256_atanh_pd;
	static constexpr auto atan = simde_mm256_atan_pd;
	inline static constexpr auto cotanh = [](const Type& a) noexcept -> Type { return reciprical(tanh(a));};
	inline static constexpr auto cotan = [](const Type& a) noexcept -> Type { return reciprical(tan(a));};
	static constexpr auto sinh = simde_mm256_sinh_pd;
	static constexpr auto sin = simde_mm256_sin_pd;
	static constexpr auto asinh = simde_mm256_asinh_pd;
	static constexpr auto asin = simde_mm256_asin_pd;
	inline static constexpr auto csch = [](const Type& a) noexcept -> Type { return reciprical(sinh(a));};
	inline static constexpr auto csc = [](const Type& a) noexcept -> Type { return reciprical(sin(a));};
	static constexpr auto cosh = simde_mm256_cosh_pd;
	static constexpr auto cos = simde_mm256_cos_pd;
	static constexpr auto acosh = simde_mm256_acosh_pd;
	static constexpr auto acos = simde_mm256_acos_pd;
	inline static constexpr auto sech = [](const Type& a) noexcept -> Type { return reciprical(cosh(a));};
	inline static constexpr auto sec = [](const Type& a) noexcept -> Type { return reciprical(cos(a));};
	static constexpr auto log = simde_mm256_log_pd;

    // svml round functions
    static constexpr auto round = simde_mm256_round_pd;
    static constexpr auto floor = simde_mm256_floor_pd;
    static constexpr auto ceil = simde_mm256_ceil_pd;


    inline static constexpr auto remainder(const Type& a, const Type& b){
        return subtract(a, multiply(floor(divide(a, b)), b));
    }
    inline static constexpr auto fmod(const Type& a, const Type& b){
        return subtract(a, multiply(round(divide(a, b), SIMDE_MM_FROUND_TO_ZERO), b));
    }

    static constexpr auto compare = simde_mm256_cmp_pd;
    static constexpr auto move_mask = simde_mm256_movemask_pd;
    NT_COMPLEX_COMPARE_OPS(complex_128);


	static constexpr auto fmsub = simde_mm256_fmsub_pd;
	inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c){
#if defined(__FMA__) || defined(SIMDE_X86_FMA)
		c = simde_mm256_fmadd_pd(a, b, c);
#else
		c = simde_mm256_add_pd(simde_mm256_mul_pd(a,b),c);
#endif 
	};
	//https://stackoverflow.com/questions/49941645/get-sum-of-values-stored-in-m256d-with-sse-avx
	inline static constexpr auto sum = [](Type x) noexcept -> complex_128 {
		simde__m128d low = simde_mm256_castpd256_pd128(x);
		simde__m128d high = simde_mm256_extractf128_pd(x, 1);
			     low = simde_mm_add_pd(low, high);
		complex_128 out;
		simde_mm_storeu_pd(reinterpret_cast<double*>(&out), low);
		return out;
	};
    static constexpr auto min = simde_mm256_min_pd;
    static constexpr auto max = simde_mm256_max_pd;

};


template<>
struct SimdTraits_avx2<float16_t>{
	using Type = simde__m256; // going to hold it as a list of floats, more 
	static constexpr size_t pack_size = 8; //takes only 8 instead of 16 because no direct functions for float16
        inline static constexpr auto load = [](const float16_t* arr) noexcept -> Type {
		return simde_mm256_cvtph_ps(simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(arr)));
	};
	inline static constexpr auto loadu = [](const float16_t* arr) noexcept -> Type {
		return simde_mm256_cvtph_ps(simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(arr)));
	};
	inline static constexpr auto load_masked = [](const float16_t* data, const simde__m256i& mask) noexcept -> Type {
		return simde_mm256_cvtph_ps(SimdTraits_avx<int16_t>::load_masked(reinterpret_cast<const int16_t*>(data), simde_mm256_castsi256_si128(mask)));
		/* simde__m128i low_mask  = simde_mm256_castsi256_si128(mask);  // Lower 128-bits of mask */
		/* simde__m128i loaded_data = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data)); */
		/* simde__m128i result = simde_mm_and_si128(loaded_data, low_mask); */
		/* return simde_mm256_cvtph_ps(result); */
	};
	inline static constexpr auto set = [](const float16_t& ele1, const float16_t& ele2, const float16_t& ele3, 
					const float16_t& ele4, const float16_t& ele5, const float16_t& ele6, const float16_t& ele7, const float16_t& ele8)
		noexcept -> Type {
			return  simde_mm256_set_ps(
					_NT_FLOAT16_TO_FLOAT32_(ele1),
					_NT_FLOAT16_TO_FLOAT32_(ele2),
					_NT_FLOAT16_TO_FLOAT32_(ele3),
					_NT_FLOAT16_TO_FLOAT32_(ele4),
					_NT_FLOAT16_TO_FLOAT32_(ele5),
					_NT_FLOAT16_TO_FLOAT32_(ele6),
					_NT_FLOAT16_TO_FLOAT32_(ele7),
					_NT_FLOAT16_TO_FLOAT32_(ele8));
		};
	inline static constexpr auto set1 = [](const float16_t& ele) noexcept -> Type {return simde_mm256_set1_ps(_NT_FLOAT16_TO_FLOAT32_(ele));};
	inline static constexpr auto store = [](float16_t* arr, const Type& vec) noexcept {
		simde_mm_store_si128(reinterpret_cast<simde__m128i*>(arr), simde_mm256_cvtps_ph(vec, 0));
	};
	inline static constexpr auto storeu = [](float16_t* arr, const Type& vec) noexcept {
		simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(arr), simde_mm256_cvtps_ph(vec, 0));
	};
	inline static constexpr auto store_masked = [](float16_t* data, const simde__m256i& mask_data, const Type& vector) noexcept {
		SimdTraits_avx<int16_t>::store_masked(reinterpret_cast<int16_t*>(data), simde_mm256_castsi256_si128(mask_data), simde_mm256_cvtps_ph(vector, 0));
		/* simde__m256i data_vector = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(data)); */
		/* simde__m256i result_data = simde_mm256_and_si256(data_vector, simde_mm256_xor_si256(mask_data, simde_mm256_cmpeq_epi16(mask_data,mask_data))); */
		/* simde__m256i result_vector = simde_mm256_and_si256(vector, mask_data); */
		/* 	     result_data = simde_mm256_add_epi16(result_data, result_vector); */
		/* simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(data), result_data); */
	};
	static constexpr auto zero = simde_mm256_setzero_ps;
	//svml exponent functions
	static constexpr auto reciprical = simde_mm256_rcp_ps;
	static constexpr auto exp = simde_mm256_exp_ps;
	static constexpr auto pow = simde_mm256_pow_ps;
	static constexpr auto sqrt = simde_mm256_sqrt_ps;
	static constexpr auto invsqrt = simde_mm256_invsqrt_ps;

        //trig svml functions
	static constexpr auto tanh = simde_mm256_tanh_ps;
	static constexpr auto tan = simde_mm256_tan_ps;
	static constexpr auto atanh = simde_mm256_atanh_ps;
	static constexpr auto atan = simde_mm256_atan_ps;
	inline static constexpr auto cotanh = [](const Type& a) noexcept -> Type { return reciprical(tanh(a));};
	inline static constexpr auto cotan = [](const Type& a) noexcept -> Type { return reciprical(tan(a));};
	static constexpr auto sinh = simde_mm256_sinh_ps;
	static constexpr auto sin = simde_mm256_sin_ps;
	static constexpr auto asinh = simde_mm256_asinh_ps;
	static constexpr auto asin = simde_mm256_asin_ps;
	inline static constexpr auto csch = [](const Type& a) noexcept -> Type { return reciprical(sinh(a));};
	inline static constexpr auto csc = [](const Type& a) noexcept -> Type { return reciprical(sin(a));};
	static constexpr auto cosh = simde_mm256_cosh_ps;
	static constexpr auto cos = simde_mm256_cos_ps;
	static constexpr auto acosh = simde_mm256_acosh_ps;
	static constexpr auto acos = simde_mm256_acos_ps;
	inline static constexpr auto sech = [](const Type& a) noexcept -> Type { return reciprical(cosh(a));};
	inline static constexpr auto sec = [](const Type& a) noexcept -> Type { return reciprical(cos(a));};
	static constexpr auto log = simde_mm256_log_ps;

    // svml round functions
    static constexpr auto round = simde_mm256_round_ps;
    static constexpr auto floor = simde_mm256_floor_ps;
    static constexpr auto ceil = simde_mm256_ceil_ps;


	static constexpr auto subtract = simde_mm256_sub_ps;
	static constexpr auto divide = simde_mm256_div_ps;
	static constexpr auto add = simde_mm256_add_ps;
	static constexpr auto multiply = simde_mm256_mul_ps;
	inline static constexpr auto negative = [](const Type& a) noexcept -> Type{
		return subtract(zero(), a);
	};

    inline static constexpr auto remainder(const Type& a, const Type& b){
        return subtract(a, multiply(floor(divide(a, b)), b));
    }
    inline static constexpr auto fmod(const Type& a, const Type& b){
        return subtract(a, multiply(round(divide(a, b), SIMDE_MM_FROUND_TO_ZERO), b));
    }
	/* static constexpr auto modulo = SimdTraits_avx2<float>::modulo; */
    static constexpr auto compare = simde_mm256_cmp_ps;
    static constexpr auto move_mask = simde_mm256_movemask_ps;
    NT_FLOAT_COMPARE_OPS(float16_t);

    inline static constexpr auto broadcast = [](const float16_t* a) noexcept -> Type {return SimdTraits_avx2<float16_t>::set1(*a);};
	static constexpr auto fmsub = simde_mm256_fmsub_ps;
	inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c) noexcept {
#if defined(__FMA__) || defined(SIMDE_X86_FMA)
		c = simde_mm256_fmadd_ps(a, b, c);
#else
		c = simde_mm256_add_ps(simde_mm256_mul_ps(a,b),c);
#endif //defined(__FMA__) || defined(SIMDE_X86_FMA)
       };
	inline static constexpr auto sum = [](Type x) noexcept -> float16_t{
#ifdef SIMDE_ARCH_SSE3
		x = simde_mm256_hadd_ps(x, x);
		x = simde_mm256_hadd_ps(x, x);
		return _NT_FLOAT32_TO_FLOAT16_(simde_mm_cvtss_f32(simde_mm256_castps256_ps128(x)));
#else
		// hiQuad = ( x7, x6, x5, x4 )
		const simde__m128 hiQuad = simde_mm256_extractf128_ps(x, 1);
		// loQuad = ( x3, x2, x1, x0 )
		const simde__m128 loQuad = simde_mm256_castps256_ps128(x);
		// sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
		const simde__m128 sumQuad = simde_mm_add_ps(loQuad, hiQuad);
		// loDual = ( -, -, x1 + x5, x0 + x4 )
		const simde__m128 loDual = sumQuad;
		// hiDual = ( -, -, x3 + x7, x2 + x6 )
		const simde__m128 hiDual = simde_mm_movehl_ps(sumQuad, sumQuad);
		// sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
		const simde__m128 sumDual = simde_mm_add_ps(loDual, hiDual);
		// lo = ( -, -, -, x0 + x2 + x4 + x6 )
		const simde__m128 lo = sumDual;
		// hi = ( -, -, -, x1 + x3 + x5 + x7 )
		const simde__m128 hi = simde_mm_shuffle_ps(sumDual, sumDual, 0x1);
		// sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
		const simde__m128 sum = simde_mm_add_ss(lo, hi);
		return _NT_FLOAT32_TO_FLOAT16_(simde_mm_cvtss_f32(sum));

#endif
	};

    static constexpr auto min = simde_mm256_min_ps;
    static constexpr auto max = simde_mm256_max_ps;


};

template<>
struct SimdTraits_avx2<complex_32>{
	using Type = simde__m256; // going to hold it as a list of floats, more 
	static constexpr size_t pack_size = 4; //takes only 4 instead of 8 because no direct functions for float16
        inline static constexpr auto load = [](const complex_32* arr) noexcept -> Type {
		return simde_mm256_cvtph_ps(simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(arr)));
	};
	inline static constexpr auto loadu = [](const complex_32* arr) noexcept -> Type {
		return simde_mm256_cvtph_ps(simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(arr)));
	};
	inline static constexpr auto load_masked = [](const complex_32* data, const simde__m256i& mask) noexcept -> Type {
		return simde_mm256_cvtph_ps(SimdTraits_avx<int16_t>::load_masked(reinterpret_cast<const int16_t*>(data), simde_mm256_castsi256_si128(mask)));
	};
	inline static constexpr auto set = [](const complex_32& comp1, const complex_32& comp2, const complex_32& comp3, const complex_32& comp4) 
	    noexcept -> Type {
		return simde_mm256_set_ps(
			_NT_FLOAT16_TO_FLOAT32_(std::get<1>(static_cast<const my_complex<float16_t>&>(comp1))), _NT_FLOAT16_TO_FLOAT32_(std::get<0>(static_cast<const my_complex<float16_t>&>(comp1))),
			_NT_FLOAT16_TO_FLOAT32_(std::get<1>(static_cast<const my_complex<float16_t>&>(comp2))), _NT_FLOAT16_TO_FLOAT32_(std::get<0>(static_cast<const my_complex<float16_t>&>(comp2))),
			_NT_FLOAT16_TO_FLOAT32_(std::get<1>(static_cast<const my_complex<float16_t>&>(comp3))), _NT_FLOAT16_TO_FLOAT32_(std::get<0>(static_cast<const my_complex<float16_t>&>(comp3))),
			_NT_FLOAT16_TO_FLOAT32_(std::get<1>(static_cast<const my_complex<float16_t>&>(comp4))), _NT_FLOAT16_TO_FLOAT32_(std::get<0>(static_cast<const my_complex<float16_t>&>(comp4))));
	};

	inline static constexpr auto set1 = [](const complex_32& ele) noexcept -> Type {return set(ele, ele, ele, ele);};
	inline static constexpr auto store = [](complex_32* arr, const Type& vec) noexcept {
		simde_mm_store_si128(reinterpret_cast<simde__m128i*>(arr), simde_mm256_cvtps_ph(vec, 0));
	};
	inline static constexpr auto storeu = [](complex_32* arr, const Type& vec) noexcept {
		simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(arr), simde_mm256_cvtps_ph(vec, 0));
	};
	inline static constexpr auto store_masked = [](complex_32* data, const simde__m256i& mask_data, const Type& vector) noexcept {
		SimdTraits_avx<int16_t>::store_masked(reinterpret_cast<int16_t*>(data), simde_mm256_castsi256_si128(mask_data), simde_mm256_cvtps_ph(vector, 0));
		/* simde__m256i data_vector = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(data)); */
		/* simde__m256i result_data = simde_mm256_and_si256(data_vector, simde_mm256_xor_si256(mask_data, simde_mm256_cmpeq_epi16(mask_data,mask_data))); */
		/* simde__m256i result_vector = simde_mm256_and_si256(vector, mask_data); */
		/* 	     result_data = simde_mm256_add_epi16(result_data, result_vector); */
		/* simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(data), result_data); */
	};
	static constexpr auto zero = simde_mm256_setzero_ps;
	//svml exponent functions
	static constexpr auto reciprical = simde_mm256_rcp_ps;
	static constexpr auto exp = simde_mm256_exp_ps;
	static constexpr auto pow = simde_mm256_pow_ps;
	static constexpr auto sqrt = simde_mm256_sqrt_ps;
	static constexpr auto invsqrt = simde_mm256_invsqrt_ps;

        //trig svml functions
	static constexpr auto tanh = simde_mm256_tanh_ps;
	static constexpr auto tan = simde_mm256_tan_ps;
	static constexpr auto atanh = simde_mm256_atanh_ps;
	static constexpr auto atan = simde_mm256_atan_ps;
	inline static constexpr auto cotanh = [](const Type& a) noexcept -> Type { return reciprical(tanh(a));};
	inline static constexpr auto cotan = [](const Type& a) noexcept -> Type { return reciprical(tan(a));};
	static constexpr auto sinh = simde_mm256_sinh_ps;
	static constexpr auto sin = simde_mm256_sin_ps;
	static constexpr auto asinh = simde_mm256_asinh_ps;
	static constexpr auto asin = simde_mm256_asin_ps;
	inline static constexpr auto csch = [](const Type& a) noexcept -> Type { return reciprical(sinh(a));};
	inline static constexpr auto csc = [](const Type& a) noexcept -> Type { return reciprical(sin(a));};
	static constexpr auto cosh = simde_mm256_cosh_ps;
	static constexpr auto cos = simde_mm256_cos_ps;
	static constexpr auto acosh = simde_mm256_acosh_ps;
	static constexpr auto acos = simde_mm256_acos_ps;
	inline static constexpr auto sech = [](const Type& a) noexcept -> Type { return reciprical(cosh(a));};
	inline static constexpr auto sec = [](const Type& a) noexcept -> Type { return reciprical(cos(a));};
	static constexpr auto log = simde_mm256_log_ps;

    // svml round functions
    static constexpr auto round = simde_mm256_round_ps;
    static constexpr auto floor = simde_mm256_floor_ps;
    static constexpr auto ceil = simde_mm256_ceil_ps;


	static constexpr auto subtract = simde_mm256_sub_ps;
	static constexpr auto divide = simde_mm256_div_ps;
	static constexpr auto add = simde_mm256_add_ps;
	static constexpr auto multiply = simde_mm256_mul_ps;
	inline static constexpr auto negative = [](const Type& a) noexcept -> Type{
		return subtract(zero(), a);
	};

    inline static constexpr auto remainder(const Type& a, const Type& b){
        return subtract(a, multiply(floor(divide(a, b)), b));
    }
    inline static constexpr auto fmod(const Type& a, const Type& b){
        return subtract(a, multiply(round(divide(a, b), SIMDE_MM_FROUND_TO_ZERO), b));
    }
    
    static constexpr auto compare = simde_mm256_cmp_ps;
    static constexpr auto move_mask = simde_mm256_movemask_ps;
    NT_COMPLEX_COMPARE_OPS(complex_32);


	/* static constexpr auto modulo = SimdTraits_avx2<float16_t>::modulo; */
	inline static constexpr auto broadcast = [](const complex_32* a) noexcept -> Type {return set1(*a);};
	static constexpr auto fmsub = simde_mm256_fmsub_ps;
	inline static constexpr auto fmadd = [](const Type& a, const Type& b, Type& c) noexcept {
#if defined(__FMA__) || defined(SIMDE_X86_FMA)
		c = simde_mm256_fmadd_ps(a, b, c);
#else
		c = simde_mm256_add_ps(simde_mm256_mul_ps(a,b),c);
#endif //defined(__FMA__) || defined(SIMDE_X86_FMA)
       };
	inline static constexpr auto sum = [](Type x) noexcept -> complex_32{
#ifdef SIMDE_ARCH_SSE3
		x = simde_mm256_hadd_ps(x, x);
		x = simde_mm256_hadd_ps(x, x);
		return _NT_FLOAT32_TO_FLOAT16_(simde_mm_cvtss_f32(simde_mm256_castps256_ps128(x)));
#else
		// hiQuad = ( x3.im, x3.re, x2.im, x2.re )
		const simde__m128 hiQuad = simde_mm256_extractf128_ps(x, 1);
		// loQuad = ( x1.im, x1.re, x0.im, x0.re )
		const simde__m128 loQuad = simde_mm256_castps256_ps128(x);
		// sumQuad = ( x3.im + x1.im, x3.re + x1.re, x2.im + x0.im, x2.re + x0.re )
		const simde__m128 sumQuad = simde_mm_add_ps(loQuad, hiQuad);
		// loDual = ( -, -, x2.im + x0.im, x2.re + x0.re )
		const simde__m128 loDual = sumQuad;
		// hiDual = ( -, -, x3.im + x1.im, x3.re + x1.re )
		const simde__m128 hiDual = simde_mm_movehl_ps(sumQuad, sumQuad);
		// sumDual = ( -, -, x2.im + x0.im + x3.im + x1.im, x2.re + x0.re + x3.re + x1.re )
		const simde__m128 sumDual = simde_mm_add_ps(loDual, hiDual);
		float result[4];
		simde_mm_storeu_ps(result, sumDual);
		return complex_32(_NT_FLOAT32_TO_FLOAT16_(result[0]), _NT_FLOAT32_TO_FLOAT16_(result[1]));
#endif
	};
    static constexpr auto min = simde_mm256_min_ps;
    static constexpr auto max = simde_mm256_max_ps;

};



#undef NT_FLOAT_COMPARE_OPS
#undef NT_COMPLEX_COMPARE_OPS

using mask_type_avx2 = simde__m256i;
template<typename T>
using simde_type_avx2 = typename SimdTraits_avx2<T>::Type;
template<typename T>
inline constexpr size_t pack_size_avx2_v = simde_supported_avx2_v<T> ? SimdTraits_avx2<T>::pack_size : 0;

//max pack size is 32, so it needs to account for all of it
alignas(64) static const int8_t mask_data_avx2[64] = { 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	0,  0,  0,  0,  0,  0,  0,  0, 
	0,  0,  0,  0,  0,  0,  0,  0,
	0,  0,  0,  0,  0,  0,  0,  0, 
	0,  0,  0,  0,  0,  0,  0,  0
};

alignas(64) static const int16_t mask_data_16_avx2[64] = { 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	0,  0,  0,  0,  0,  0,  0,  0, 
	0,  0,  0,  0,  0,  0,  0,  0,
	0,  0,  0,  0,  0,  0,  0,  0, 
	0,  0,  0,  0,  0,  0,  0,  0
};

alignas(64) static const int64_t mask_data_64_avx2[16] = { 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	0,  0,  0,  0,  0,  0,  0,  0, 
};

alignas(64) static const int32_t mask_data_32_avx2[32] = { 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	0,  0,  0,  0,  0,  0,  0,  0, 
	0,  0,  0,  0,  0,  0,  0,  0 
};


template<typename T, size_t N>
NT_ALWAYS_INLINE constexpr mask_type_avx2 Kgenerate_mask_avx2() noexcept {
	constexpr size_t pack_size = pack_size_avx2_v<T>;
	static_assert(N <= pack_size, "N cannot exceed the number of elements in the SIMD register.");
	if constexpr (std::is_same_v<T, complex_64>){ //special case to handle complex numbers
		return SimdTraits_avx2<int32_t>::loadu(reinterpret_cast<const mask_type_avx2*>(&mask_data_32_avx2[16-(N*2)]));
	}
	else if constexpr(std::is_same_v<T, complex_128>){
		return SimdTraits_avx2<int64_t>::loadu(reinterpret_cast<const mask_type_avx2*>(&mask_data_64_avx2[8-(N*2)]));
	}
	else if constexpr(std::is_same_v<T, float16_t>){
		return SimdTraits_avx2<int16_t>::loadu(reinterpret_cast<const mask_type_avx2*>(&mask_data_16_avx2[32-N]));
	}
	else if constexpr (std::is_same_v<T, complex_32>){
		return SimdTraits_avx2<int16_t>::loadu(reinterpret_cast<const mask_type_avx2*>(&mask_data_16_avx2[32-(N*2)]));
	}
	else if constexpr (pack_size == pack_size_avx2_v<int32_t>){
		return SimdTraits_avx2<int32_t>::loadu(reinterpret_cast<const mask_type_avx2*>(&mask_data_32_avx2[16-N]));
	}
	else if constexpr (pack_size == pack_size_avx2_v<int64_t>){
		return SimdTraits_avx2<int64_t>::loadu(reinterpret_cast<const mask_type_avx2*>(&mask_data_64_avx2[8-N]));
	}
	else if constexpr (pack_size == pack_size_avx2_v<int8_t>){
		return SimdTraits_avx2<int8_t>::loadu(reinterpret_cast<const mask_type_avx2*>(&mask_data_avx2[32-N]));
	}else{
		static_assert(pack_size == pack_size_avx2_v<int16_t>, "Unexpected mask pack size");
		return SimdTraits_avx2<int16_t>::loadu(reinterpret_cast<const mask_type_avx2*>(&mask_data_16_avx2[32-N]));
	}
}

template<typename T>
NT_ALWAYS_INLINE constexpr mask_type_avx2 generate_mask_avx2(size_t N) {
	constexpr size_t pack_size = pack_size_avx2_v<T>;
	if constexpr (std::is_same_v<T, complex_64>){ //special case to handle complex numbers
		return SimdTraits_avx2<int32_t>::loadu(reinterpret_cast<const mask_type_avx2*>(&mask_data_32_avx2[16-(N*2)]));
	}
	else if constexpr(std::is_same_v<T, complex_128>){
		return SimdTraits_avx2<int64_t>::loadu(reinterpret_cast<const mask_type_avx2*>(&mask_data_64_avx2[8-(N*2)]));
	}
	else if constexpr(std::is_same_v<T, float16_t>){
		return SimdTraits_avx2<int16_t>::loadu(reinterpret_cast<const mask_type_avx2*>(&mask_data_16_avx2[32-N]));
	}
	else if constexpr (std::is_same_v<T, complex_32>){
		return SimdTraits_avx2<int16_t>::loadu(reinterpret_cast<const mask_type_avx2*>(&mask_data_16_avx2[32-(N*2)]));
	}
	else if constexpr (pack_size == pack_size_avx2_v<int32_t>){
		return SimdTraits_avx2<int32_t>::loadu(reinterpret_cast<const mask_type_avx2*>(&mask_data_32_avx2[16-N]));
	}
	else if constexpr (pack_size == pack_size_avx2_v<int64_t>){
		return SimdTraits_avx2<int64_t>::loadu(reinterpret_cast<const mask_type_avx2*>(&mask_data_64_avx2[8-N]));
	}
	else if constexpr (pack_size == pack_size_avx2_v<int8_t>){
		return SimdTraits_avx2<int8_t>::loadu(reinterpret_cast<const mask_type_avx2*>(&mask_data_avx2[32-N]));
	}else{
		return SimdTraits_avx2<int16_t>::loadu(reinterpret_cast<const mask_type_avx2*>(&mask_data_16_avx2[32-N]));
	}
}


}} // nt::mp::


#undef NT_MP_AVX2_CONCAT 

#endif // _NT_SIMDE_TRAITS_AVX2_H
