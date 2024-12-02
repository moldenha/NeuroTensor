#ifndef _NT_MATMULT_SIMDE_AVX_HPP_
#define _NT_MATMULT_SIMDE_AVX_HPP_
//types fully working:
//float
//double
//int32
//uint32
//int64
//uint64
//int8
//uint8
//int16
//uint16
//types working on:
//types to add:
//float16


#include <simde/x86/avx.h>
#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>  // only for FMA if supported
#include <cstddef>
#include <cstddef>
#include <array>
#include <type_traits>


namespace nt{
namespace functional{
namespace std_functional{

template<typename T>
struct simde_supported{
	static constexpr bool value = false;
};

template<>
struct simde_supported<float>{
	static constexpr bool value = true;
};

template<>
struct simde_supported<double>{
	static constexpr bool value = true;
};


template<>
struct simde_supported<uint8_t>{
	static constexpr bool value = true;
};

template<>
struct simde_supported<int8_t>{
	static constexpr bool value = true;
};

template<>
struct simde_supported<uint16_t>{
	static constexpr bool value = true;
};

template<>
struct simde_supported<int16_t>{
	static constexpr bool value = true;
};

template<>
struct simde_supported<uint32_t>{
	static constexpr bool value = true;
};

template<>
struct simde_supported<int32_t>{
	static constexpr bool value = true;
};

template<>
struct simde_supported<uint64_t>{
	static constexpr bool value = true;
};

template<>
struct simde_supported<int64_t>{
	static constexpr bool value = true;
};





template<typename T>
inline constexpr bool simde_supported_v = simde_supported<T>::value;

template <typename T>
struct SimdTraits;

#ifdef SIMDE_ARCH_X86_AVX2 
// Specialization for AVX2 if available
template <>
struct SimdTraits<float> {
    using Type = simde__m256;
    static constexpr size_t pack_size = 8;  // AVX2 can handle 8 floats
    static constexpr size_t tile_size = 16; 
    static constexpr auto load = simde_mm256_load_ps; //takes aligned version, valid due to alginment in packed memory
    static constexpr auto loadu = simde_mm256_loadu_ps; //takes aligned version, valid due to alginment in packed memory
    static constexpr auto load_masked = simde_mm256_maskload_ps;
    static constexpr auto set = simde_mm256_set_ps;
    static constexpr auto broadcast = simde_mm256_broadcast_ss;
    static constexpr auto store = simde_mm256_store_ps;
    static constexpr auto storeu = simde_mm256_storeu_ps;
    static constexpr auto store_masked = simde_mm256_maskstore_ps;
    static constexpr auto zero = simde_mm256_setzero_ps;
    inline static constexpr auto dot = [](const simde__m256& a, const simde__m256& b, simde__m256& c){
#if defined(__FMA__) || defined(SIMDE_X86_FMA)
	c = simde_mm256_fmadd_ps(a, b, c);
#else
	c = simde_mm256_add_ps(simde_mm256_mullo_ps(a,b),c);
#endif //defined(__FMA__) || defined(SIMDE_X86_FMA)
    };
};

#else


template <>
struct SimdTraits<float> {
    using Type = simde__m128;
    static constexpr size_t pack_size = 4;
    static constexpr size_t tile_size = 8;
    static constexpr auto load = simde_mm_load_ps;
    static constexpr auto loadu = simde_mm_loadu_ps;
    static constexpr auto load_masked = simde_mm_maskload_ps;
    static constexpr auto set = simde_mm_set_ps;
    static constexpr auto broadcast = simde_mm_broadcast_ss;
    static constexpr auto store = simde_mm_store_ps;
    static constexpr auto storeu = simde_mm_storeu_ps;
    static constexpr auto store_masked = simde_mm_maskstore_ps;
    static constexpr auto zero = simde_mm_setzero_ps;
    inline static constexpr auto dot = [](const simde__m128& a, const simde__m128& b, simde__m128& c){
#if defined(__FMA__) || defined(SIMDE_X86_FMA)
	c = simde_mm_fmadd_ps(a, b, c);
#else
	c = simde_mm_add_ps(simde_mm256_mullo_ps(a,b),c);
#endif //defined(__FMA__) || defined(SIMDE_X86_FMA)
    };
};
#endif




#ifdef SIMDE_ARCH_X86_AVX2
template <>
struct SimdTraits<double> {
    using Type = simde__m256d;
    static constexpr size_t pack_size = 4;
    static constexpr size_t tile_size = 8;
    static constexpr auto load = simde_mm256_load_pd;
    static constexpr auto loadu = simde_mm256_loadu_pd;
    static constexpr auto load_masked = simde_mm256_maskload_pd;
    static constexpr auto broadcast = simde_mm256_broadcast_sd;
    static constexpr auto set = simde_mm256_set_pd;
    static constexpr auto store = simde_mm256_store_pd;
    static constexpr auto storeu = simde_mm256_storeu_pd;
    static constexpr auto store_masked = simde_mm256_maskstore_pd;
    static constexpr auto zero = simde_mm256_setzero_pd;
    inline static constexpr auto dot = [](const Type& a, const Type& b, Type& c){
#if defined(__FMA__) || defined(SIMDE_X86_FMA)
	c = simde_mm256_fmadd_pd(a, b, c);
#else
	c = simde_mm256_add_pd(simde_mm256_mullo_pd(a,b),c);
#endif 
    };
};

#else
template <>
struct SimdTraits<double> {
    using Type = simde__m128d;
    static constexpr size_t pack_size = 2;
    static constexpr size_t tile_size = 4;
    static constexpr auto load = simde_mm_load_pd;
    static constexpr auto loadu = simde_mm_loadu_pd;
    static constexpr auto load_masked = simde_mm_maskload_pd;
    static constexpr auto broadcast = simde_mm_broadcast_sd;
    static constexpr auto set = simde_mm_set_pd;
    static constexpr auto store = simde_mm_store_pd;
    static constexpr auto storeu = simde_mm_storeu_pd;
    static constexpr auto store_masked = simde_mm_maskstore_pd;
    static constexpr auto zero = simde_mm_setzero_pd;
    inline static constexpr auto dot = [](const Type& a, const Type& b, Type& c){
#if defined(__FMA__) || defined(SIMDE_X86_FMA)
	c = simde_mm_fmadd_pd(a, b, c);
#else
	c = simde_mm_add_pd(simde_mm_mullo_pd(a,b),c);
#endif
    };
};

#endif

#ifdef SIMDE_ARCH_X86_AVX2



template <>
struct SimdTraits<int32_t> {
    using Type = simde__m256i;
    static constexpr size_t pack_size = 8;
    static constexpr size_t tile_size = 16;
    static constexpr auto load = simde_mm256_load_si256;
    static constexpr auto loadu = simde_mm256_loadu_si256;
    static constexpr auto load_masked = simde_mm256_maskload_epi32;
    static constexpr auto set = simde_mm256_set_epi32;
    static constexpr auto store = simde_mm256_store_si256;
    static constexpr auto storeu = simde_mm256_storeu_si256;
    static constexpr auto store_masked = simde_mm256_maskstore_epi32;
    static constexpr auto zero = simde_mm256_setzero_si256;
    inline static constexpr auto broadcast = [](const int32_t* arr) -> Type {return simde_mm256_set1_epi32(*arr);};
    inline static constexpr auto dot = [](const Type& a, const Type& b, Type& c){
	c = simde_mm256_add_epi32(simde_mm256_mullo_epi32(a, b), c);
    };

};
#else

template <>
struct SimdTraits<int32_t> {
    using Type = simde__m128i;
    static constexpr size_t pack_size = 4;
    static constexpr size_t tile_size = 8;
    static constexpr auto load = simde_mm_load_si128;
    static constexpr auto loadu = simde_mm_loadu_si128;
    static constexpr auto load_masked = simde_mm_maskload_epi32;
    static constexpr auto set = simde_mm_set_epi32;
    static constexpr auto store = simde_mm_store_si256;
    static constexpr auto storeu = simde_mm_storeu_si256;
    static constexpr auto store_masked = simde_mm_maskstore_epi32;
    static constexpr auto zero = simde_mm_setzero_si128;
    inline static constexpr auto broadcast = [](const int32_t* arr) -> Type {return simde_mm_set1_epi32(*arr);};
    inline static constexpr auto dot = [](const Type& a, const Type& b, Type& c){
	c = simde_mm_add_epi32(simde_mm_mullo_epi32(a, b), c);
    };

};

#endif



#ifdef SIMDE_ARCH_X86_AVX2


template <>
struct SimdTraits<uint32_t> {
    using Type = simde__m256i;
    static constexpr size_t pack_size = 8;
    static constexpr size_t tile_size = 16;
    static constexpr auto load = simde_mm256_load_si256;
    static constexpr auto loadu = simde_mm256_loadu_si256;
    static constexpr auto load_masked = simde_mm256_maskload_epi32;
    static constexpr auto set = simde_mm256_set_epi32;
    static constexpr auto store = simde_mm256_store_si256;
    static constexpr auto storeu = simde_mm256_storeu_si256;
    static constexpr auto store_masked = simde_mm256_maskstore_epi32;
    static constexpr auto zero = simde_mm256_setzero_si256;
    inline static constexpr auto broadcast = [](const uint32_t* arr) -> Type {return simde_mm256_set1_epi32(*arr);};
    inline static constexpr auto dot = [](const Type& a, const Type& b, Type& c){
	c = simde_mm256_add_epi32(simde_mm256_mullo_epi32(a, b), c);
    };

};
#else

template <>
struct SimdTraits<uint32_t> {
    using Type = simde__m128i;
    static constexpr size_t pack_size = 4;
    static constexpr size_t tile_size = 8;
    static constexpr auto load = simde_mm_load_si128;
    static constexpr auto loadu = simde_mm_loadu_si128;
    static constexpr auto load_masked = simde_mm_maskload_epi32;
    static constexpr auto set = simde_mm_set_epi32;
    static constexpr auto store = simde_mm_store_si256;
    static constexpr auto storeu = simde_mm_storeu_si256;
    static constexpr auto store_masked = simde_mm_maskstore_epi32;
    static constexpr auto zero = simde_mm_setzero_si128;
    inline static constexpr auto broadcast = [](const uint32_t* arr) -> Type {return simde_mm_set1_epi32(*arr);};
    inline static constexpr auto dot = [](const Type& a, const Type& b, Type& c){
	c = simde_mm_add_epi32(simde_mm_mullo_epi32(a, b), c);
    };

};

#endif

#ifdef SIMDE_ARCH_X86_AVX2

//gotten from https://stackoverflow.com/questions/76436053/simd-intrinsics-avx-tried-to-use-mm256-mullo-epi64-but-got-0xc000001d-illega
simde__m256i mul64_haswell (simde__m256i a, simde__m256i b) {
    simde__m256i bswap   = simde_mm256_shuffle_epi32(b,0xB1);
    simde__m256i prodlh  = simde_mm256_mullo_epi32(a,bswap);

    simde__m256i prodlh2 = simde_mm256_srli_epi64(prodlh, 32);
    simde__m256i prodlh3 = simde_mm256_add_epi32(prodlh2, prodlh);
    simde__m256i prodlh4 = simde_mm256_and_si256(prodlh3, _mm256_set1_epi64x(0x00000000FFFFFFFF));

    simde__m256i prodll  = simde_mm256_mul_epu32(a,b);
    simde__m256i prod    = simde_mm256_add_epi64(prodll,prodlh4);
    return  prod;
}


//from https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx
//this may come in handy
__m256d int64_to_double_fast_precise(const __m256i v)
/* Optimized full range int64_t to double conversion           */
/* Emulate _mm256_cvtepi64_pd()                                */
{
    __m256i magic_i_lo   = _mm256_set1_epi64x(0x4330000000000000);                /* 2^52               encoded as floating-point  */
    __m256i magic_i_hi32 = _mm256_set1_epi64x(0x4530000080000000);                /* 2^84 + 2^63        encoded as floating-point  */
    __m256i magic_i_all  = _mm256_set1_epi64x(0x4530000080100000);                /* 2^84 + 2^63 + 2^52 encoded as floating-point  */
    __m256d magic_d_all  = _mm256_castsi256_pd(magic_i_all);

    __m256i v_lo         = _mm256_blend_epi32(magic_i_lo, v, 0b01010101);         /* Blend the 32 lowest significant bits of v with magic_int_lo                                                   */
    __m256i v_hi         = _mm256_srli_epi64(v, 32);                              /* Extract the 32 most significant bits of v                                                                     */
            v_hi         = _mm256_xor_si256(v_hi, magic_i_hi32);                  /* Flip the msb of v_hi and blend with 0x45300000                                                                */
    __m256d v_hi_dbl     = _mm256_sub_pd(_mm256_castsi256_pd(v_hi), magic_d_all); /* Compute in double precision:                                                                                  */
    __m256d result       = _mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo));    /* (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition !!                        */
            return result;                                                        /* With gcc use -O3, then -fno-associative-math is default. Do not use -Ofast, which enables -fassociative-math! */
                                                                                  /* With icc use -fp-model precise                                                                                */
}


__m256d uint64_to_double_fast_precise(const __m256i v)                    
/* Optimized full range uint64_t to double conversion          */
/* This code is essentially identical to Mysticial's solution. */
/* Emulate _mm256_cvtepu64_pd()                                */
{
    __m256i magic_i_lo   = _mm256_set1_epi64x(0x4330000000000000);                /* 2^52        encoded as floating-point  */
    __m256i magic_i_hi32 = _mm256_set1_epi64x(0x4530000000000000);                /* 2^84        encoded as floating-point  */
    __m256i magic_i_all  = _mm256_set1_epi64x(0x4530000000100000);                /* 2^84 + 2^52 encoded as floating-point  */
    __m256d magic_d_all  = _mm256_castsi256_pd(magic_i_all);

    __m256i v_lo         = _mm256_blend_epi32(magic_i_lo, v, 0b01010101);         /* Blend the 32 lowest significant bits of v with magic_int_lo                                                   */
    __m256i v_hi         = _mm256_srli_epi64(v, 32);                              /* Extract the 32 most significant bits of v                                                                     */
            v_hi         = _mm256_xor_si256(v_hi, magic_i_hi32);                  /* Blend v_hi with 0x45300000                                                                                    */
    __m256d v_hi_dbl     = _mm256_sub_pd(_mm256_castsi256_pd(v_hi), magic_d_all); /* Compute in double precision:                                                                                  */
    __m256d result       = _mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo));    /* (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition !!                        */
            return result;                                                        /* With gcc use -O3, then -fno-associative-math is default. Do not use -Ofast, which enables -fassociative-math! */
                                                                                  /* With icc use -fp-model precise                                                                                */
}

template <>
struct SimdTraits<int64_t> {
    using Type = simde__m256i;
    static constexpr size_t pack_size = 4;
    static constexpr size_t tile_size = 8;
    static constexpr auto load = simde_mm256_load_si256;
    static constexpr auto loadu = simde_mm256_loadu_si256;
    static constexpr auto load_masked = simde_mm256_maskload_epi64;
    static constexpr auto set = simde_mm256_set_epi64x;
    static constexpr auto store = simde_mm256_store_si256;
    static constexpr auto storeu = simde_mm256_storeu_si256;
    static constexpr auto store_masked = simde_mm256_maskstore_epi64;
    static constexpr auto zero = simde_mm256_setzero_si256;
    inline static constexpr auto broadcast = [](const int64_t* arr) -> Type {return simde_mm256_set1_epi64x(*arr);};
    inline static constexpr auto dot = [](const Type& a, const Type& b, Type& c){
	c = simde_mm256_add_epi64(mul64_haswell(a, b), c);
    };

};
#else
simde__m128i mult_int64s(const simde__m128i& a,const simde__m128i& b){
	int64_t res_a[2];
	int64_t res_b[2];
	simde_mm_storeu_si256((simde__m128i*)res_a, a);
	simde_mm_storeu_si256((simde__m128i*)res_b, b);
	return simde_mm_set_epi64x(res_a[0] * res_b[0], res_a[1] * res_b[1]);
}

simde__m128d uint64_to_double_full(__m128i x){
    simde__m128i xH = simde_mm_srli_epi64(x, 32);
    xH = simde_mm_or_si128(xH, simde_mm_castpd_si128(simde_mm_set1_pd(19342813113834066795298816.)));          //  2^84
    simde__m128i xL = simde_mm_blend_epi16(x, simde_mm_castpd_si128(simde_mm_set1_pd(0x0010000000000000)), 0xcc);   //  2^52
    simde__m128d f = simde_mm_sub_pd(simde_mm_castsi128_pd(xH), _mm_set1_pd(19342813118337666422669312.));     //  2^84 + 2^52
    return simde_mm_add_pd(f, simde_mm_castsi128_pd(xL));
}

simde__m128d int64_to_double_full(__m128i x){
    __m128i xH = _mm_srai_epi32(x, 16);
    xH = _mm_blend_epi16(xH, _mm_setzero_si128(), 0x33);
    xH = _mm_add_epi64(xH, _mm_castpd_si128(_mm_set1_pd(442721857769029238784.)));              //  3*2^67
    __m128i xL = _mm_blend_epi16(x, _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)), 0x88);   //  2^52
    __m128d f = _mm_sub_pd(_mm_castsi128_pd(xH), _mm_set1_pd(442726361368656609280.));          //  3*2^67 + 2^52
    return _mm_add_pd(f, _mm_castsi128_pd(xL));
}



template <>
struct SimdTraits<int64_t> {
    using Type = simde__m128i;
    static constexpr size_t pack_size = 2;
    static constexpr size_t tile_size = 4;
    static constexpr auto load = simde_mm_load_si256;
    static constexpr auto loadu = simde_mm_loadu_si256;
    static constexpr auto load_masked = simde_mm_maskload_epi64;
    static constexpr auto set = simde_mm_set_epi64x;
    static constexpr auto store = simde_mm_store_si256;
    static constexpr auto storeu = simde_mm_storeu_si256;
    static constexpr auto store_masked = simde_mm_maskstore_epi64;
    static constexpr auto zero = simde_mm_setzero_si256;
    inline static constexpr auto broadcast = [](const int64_t* arr) -> Type {return simde_mm_set1_epi64x(*arr);};
    inline static constexpr auto dot = [](const Type& a, const Type& b, Type& c){
	c = simde_mm_add_epi64(mult_int64s(a, b), c);
    };

};

#endif




#ifdef SIMDE_ARCH_X86_AVX2
template <>
struct SimdTraits<uint64_t> {
    using Type = simde__m256i;
    static constexpr size_t pack_size = 4;
    static constexpr size_t tile_size = 8;
    static constexpr auto load = simde_mm256_load_si256;
    static constexpr auto loadu = simde_mm256_loadu_si256;
    static constexpr auto load_masked = simde_mm256_maskload_epi64;
    static constexpr auto set = simde_mm256_set_epi64x;
    static constexpr auto store = simde_mm256_store_si256;
    static constexpr auto storeu = simde_mm256_storeu_si256;
    static constexpr auto store_masked = simde_mm256_maskstore_epi64;
    static constexpr auto zero = simde_mm256_setzero_si256;
    inline static constexpr auto broadcast = [](const uint64_t* arr) -> Type {return simde_mm256_set1_epi64x(*arr);};
    inline static constexpr auto dot = [](const Type& a, const Type& b, Type& c){
	c = simde_mm256_add_epi64(mul64_haswell(a, b), c);
    };

};
#else

simde__m128i mult_int64s(const simde__m128i& a,const simde__m128i& b){
	uint64_t res_a[2];
	uint64_t res_b[2];
	simde_mm_storeu_si256((simde__m128i*)res_a, a);
	simde_mm_storeu_si256((simde__m128i*)res_b, b);
	return simde_mm_set_epi64x(res_a[0] * res_b[0], res_a[1] * res_b[1]);
}

template <>
struct SimdTraits<uint64_t> {
    using Type = simde__m128i;
    static constexpr size_t pack_size = 2;
    static constexpr size_t tile_size = 4;
    static constexpr auto load = simde_mm_load_si256;
    static constexpr auto loadu = simde_mm_loadu_si256;
    static constexpr auto load_masked = simde_mm_maskload_epi64;
    static constexpr auto set = simde_mm_set_epi64x;
    static constexpr auto store = simde_mm_store_si256;
    static constexpr auto storeu = simde_mm_storeu_si256;
    static constexpr auto store_masked = simde_mm_maskstore_epi64;
    static constexpr auto zero = simde_mm_setzero_si256;
    inline static constexpr auto broadcast = [](const uint64_t* arr) -> Type {return simde_mm_set1_epi64x(*arr);};
    inline static constexpr auto dot = [](const Type& a, const Type& b, Type& c){
	c = simde_mm_add_epi64(mult_int64s(a, b), c);
    };

};
#endif

// SimdTraits for int8_t
#ifdef SIMDE_ARCH_X86_AVX2

simde__m256i load_8bit_from_mask(const int8_t* data, const simde__m256i& mask){
	simde__m256i loaded_data = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(data));
	simde__m256i result = simde_mm256_and_si256(loaded_data, mask);
	return result;
}

// Function to store 8-bit integers from __m256i to an array using a mask
/* void store_8bit_with_mask(int8_t* data, const simde__m256i& vector, simde__m256i mask_data) { */
/* 	// Unpack the 8-bit values from the __m256i vector (expand to 32-bit integers) */
/* 	simde__m256i mask_low = simde_mm256_and_si256(mask_data, simde_mm256_set1_epi32(0x000000FF)); */
/* 	simde__m256i mask_high = simde_mm256_srli_epi32(mask_data, 8); */ 
/* 	simde_mm256_maskstore_epi32(reinterpret_cast<int32_t*>(data), mask_low, vector); */
/* 	simde_mm256_maskstore_epi32(reinterpret_cast<int32_t*>(data) + 4, mask_high, vector); */
/* } */

//I can't find anything that I can do this with in a non-manual fashion so this will have to do

void store_8bit_with_mask(int8_t* data, const simde__m256i& mask_data, const simde__m256i& vector) {
	int8_t values_arr[32];
	simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(values_arr), vector);

	int8_t mask_arr[32];
	simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(mask_arr), mask_data);

	for (int i = 0; i < 32; i++) {
		if (mask_arr[i] != 0) {
			data[i] = values_arr[i];
		}
	}
}

//unfortunately simde does not have built in simde_mm256_mullo_epi8 support yet :/
/* simde__m256i simde_mm256_mullo_epi8(simde__m256i a, simde__m256i b) { */
/*   // Saturate the result to 8-bit range */
/*   return simde_mm256_and_si256(simde_mm256_mullo_epi16(a, b), simde_mm256_set1_epi16(0xFF)); */
/* } */
//TODO: look into converting a simde__m256i of 16 bits into a simde__m128i of 8 bits without saturation
//currently, there is a way to do it, but it goes (value, 0) continued
//maybe theres a way to permute it, but it still doesn't hold all the values correctly
simde__m256i simde_mm256_mullo_epi8_custom(simde__m256i a, simde__m256i b) {
    // Step 1: Unpack the 8-bit integers to 16-bit integers (low and high).
    simde__m128i a_low  = simde_mm256_castsi256_si128(a);  // Lower 128-bits of a
    simde__m128i a_high = simde_mm256_extractf128_si256(a, 1);  // Upper 128-bits of a
    simde__m128i b_low  = simde_mm256_castsi256_si128(b);  // Lower 128-bits of b
    simde__m128i b_high = simde_mm256_extractf128_si256(b, 1);  // Upper 128-bits of b

    // Unpack the 8-bit integers to 16-bit integers (sign-extended)
    simde__m256i a_low_16 = simde_mm256_cvtepi8_epi16(a_low);
    simde__m256i a_high_16 = simde_mm256_cvtepi8_epi16(a_high);
    simde__m256i b_low_16 = simde_mm256_cvtepi8_epi16(b_low);
    simde__m256i b_high_16 = simde_mm256_cvtepi8_epi16(b_high);

    // Step 2: Multiply the 16-bit parts.
    simde__m256i result_low_16  = simde_mm256_mullo_epi16(a_low_16, b_low_16);
    simde__m256i result_high_16 = simde_mm256_mullo_epi16(a_high_16, b_high_16);

    int16_t res_low[16];
    int16_t res_high[16];
    simde_mm256_storeu_si256((simde__m256i*)res_low, result_low_16);
    simde_mm256_storeu_si256((simde__m256i*)res_high, result_high_16);
    int8_t out[32];
    for(int i = 0; i < 16; ++i)
	    out[i] = static_cast<int8_t>(res_low[i]);
    for(int i = 0; i < 16; ++i)
	    out[i+16] = static_cast<int8_t>(res_high[i]);

    return simde_mm256_loadu_si256((simde__m256i*)out);
    // Step 4: Combine the two 128-bit results into a 256-bit result
    /* return simde_mm256_set_m128i(result_high_8, result_low_8); */
}

template <>
struct SimdTraits<int8_t> {
    using Type = simde__m256i;
    static constexpr size_t pack_size = 32;
    static constexpr size_t tile_size = 64;
    static constexpr auto load = simde_mm256_load_si256;
    static constexpr auto loadu = simde_mm256_loadu_si256;
    static constexpr auto load_masked = load_8bit_from_mask;
    static constexpr auto set = simde_mm256_set_epi8;
    static constexpr auto store = simde_mm256_store_si256;
    static constexpr auto storeu = simde_mm256_storeu_si256;
    static constexpr auto store_masked = store_8bit_with_mask;
    static constexpr auto zero = simde_mm256_setzero_si256;
    inline static constexpr auto broadcast = [](const int8_t* arr) -> Type {return simde_mm256_set1_epi8(*arr);};
    inline static constexpr auto dot = [](const Type& a, const Type& b, Type& c){
	c = simde_mm256_add_epi8(simde_mm256_mullo_epi8_custom(a, b), c);
    };

};
#else

simde__m128i load_8bit_from_mask(const int8_t* data, const simde__m128i& mask){
	simde__m128i loaded_data = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data));
	simde__m128i result = simde_mm_and_si128(loaded_data, mask);
	return result;
}

void store_8bit_with_mask(int8_t* data, const simde__m128i& mask_data, const simde__m128i& vector) {
	int8_t values_arr[16];
	simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(values_arr), vector);

	int8_t mask_arr[16];
	simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(mask_arr), mask_data);

	for (int i = 0; i < 16; i++) {
		if (mask_arr[i] != 0) {
			data[i] = values_arr[i];
		}
	}
}


//need to test this further
simde__m128i mullo_epi8_simde_custom(simde__m128i a, simde__m128i b) {
    // Multiply pairs of 16-bit integers
    simde__m128i product = simde_mm_mullo_epi16(a, b);

    // Pack the 16-bit products into 8-bit integers, truncating the higher 8 bits
    return simde_mm_packus_epi16(product, product);
}




template <>
struct SimdTraits<int8_t> {
    using Type = simde__m128i;
    static constexpr size_t pack_size = 16;
    static constexpr size_t tile_size = 32;
    static constexpr auto load = simde_mm_load_si256;
    static constexpr auto loadu = simde_mm_loadu_si256;
    static constexpr auto load_masked = load_8bit_from_mask;
    static constexpr auto set = simde_mm_set_epi8;
    static constexpr auto store = simde_mm_store_si256;
    static constexpr auto storeu = simde_mm_storeu_si256;
    static constexpr auto store_masked = store_8bit_with_mask;
    static constexpr auto zero = simde_mm_setzero_si256;
    inline static constexpr auto broadcast = [](const int8_t* arr) -> Type {return simde_mm_set1_epi8(*arr);};
    inline static constexpr auto dot = [](const Type& a, const Type& b, Type& c){
	c = simde_mm_add_epi64(mullo_epi8_simde_custom(a, b), c);
    };

};
#endif


#ifdef SIMDE_ARCH_X86_AVX2

simde__m256i load_u8bit_from_mask(const int8_t* data, const simde__m256i& mask){
	simde__m256i loaded_data = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(data));
	simde__m256i result = simde_mm256_and_si256(loaded_data, mask);
	return result;
}

// Function to store 8-bit integers from __m256i to an array using a mask
/* void store_8bit_with_mask(int8_t* data, const simde__m256i& vector, simde__m256i mask_data) { */
/* 	// Unpack the 8-bit values from the __m256i vector (expand to 32-bit integers) */
/* 	simde__m256i mask_low = simde_mm256_and_si256(mask_data, simde_mm256_set1_epi32(0x000000FF)); */
/* 	simde__m256i mask_high = simde_mm256_srli_epi32(mask_data, 8); */ 
/* 	simde_mm256_maskstore_epi32(reinterpret_cast<int32_t*>(data), mask_low, vector); */
/* 	simde_mm256_maskstore_epi32(reinterpret_cast<int32_t*>(data) + 4, mask_high, vector); */
/* } */

//I can't find anything that I can do this with in a non-manual fashion so this will have to do

void store_u8bit_with_mask(int8_t* data, const simde__m256i& mask_data, const simde__m256i& vector) {
	uint8_t values_arr[32];
	simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(values_arr), vector);

	int8_t mask_arr[32];
	simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(mask_arr), mask_data);

	for (int i = 0; i < 32; i++) {
		if (mask_arr[i] != 0) {
			data[i] = values_arr[i];
		}
	}
}

//unfortunately simde does not have built in simde_mm256_mullo_epi8 support yet :/
/* simde__m256i simde_mm256_mullo_epi8(simde__m256i a, simde__m256i b) { */
/*   // Saturate the result to 8-bit range */
/*   return simde_mm256_and_si256(simde_mm256_mullo_epi16(a, b), simde_mm256_set1_epi16(0xFF)); */
/* } */
//TODO: look into converting a simde__m256i of 16 bits into a simde__m128i of 8 bits without saturation
//currently, there is a way to do it, but it goes (value, 0) continued
//maybe theres a way to permute it, but it still doesn't hold all the values correctly
simde__m256i simde_mm256_mullo_epui8_custom(simde__m256i a, simde__m256i b) {
    // Step 1: Unpack the 8-bit integers to 16-bit integers (low and high).
    simde__m128i a_low  = simde_mm256_castsi256_si128(a);  // Lower 128-bits of a
    simde__m128i a_high = simde_mm256_extractf128_si256(a, 1);  // Upper 128-bits of a
    simde__m128i b_low  = simde_mm256_castsi256_si128(b);  // Lower 128-bits of b
    simde__m128i b_high = simde_mm256_extractf128_si256(b, 1);  // Upper 128-bits of b

    // Unpack the 8-bit integers to 16-bit integers (sign-extended)
    simde__m256i a_low_16 = simde_mm256_cvtepi8_epi16(a_low);
    simde__m256i a_high_16 = simde_mm256_cvtepi8_epi16(a_high);
    simde__m256i b_low_16 = simde_mm256_cvtepi8_epi16(b_low);
    simde__m256i b_high_16 = simde_mm256_cvtepi8_epi16(b_high);

    // Step 2: Multiply the 16-bit parts.
    simde__m256i result_low_16  = simde_mm256_mullo_epi16(a_low_16, b_low_16);
    simde__m256i result_high_16 = simde_mm256_mullo_epi16(a_high_16, b_high_16);

    uint16_t res_low[16];
    uint16_t res_high[16];
    simde_mm256_storeu_si256((simde__m256i*)res_low, result_low_16);
    simde_mm256_storeu_si256((simde__m256i*)res_high, result_high_16);
    uint8_t out[32];
    for(int i = 0; i < 16; ++i)
	    out[i] = static_cast<uint8_t>(res_low[i]);
    for(int i = 0; i < 16; ++i)
	    out[i+16] = static_cast<uint8_t>(res_high[i]);

    return simde_mm256_loadu_si256((simde__m256i*)out);
    // Step 4: Combine the two 128-bit results into a 256-bit result
    /* return simde_mm256_set_m128i(result_high_8, result_low_8); */
}

template <>
struct SimdTraits<uint8_t> {
    using Type = simde__m256i;
    static constexpr size_t pack_size = 32;
    static constexpr size_t tile_size = 64;
    static constexpr auto load = simde_mm256_load_si256;
    static constexpr auto loadu = simde_mm256_loadu_si256;
    static constexpr auto load_masked = load_u8bit_from_mask;
    static constexpr auto set = simde_mm256_set_epi8;
    static constexpr auto store = simde_mm256_store_si256;
    static constexpr auto storeu = simde_mm256_storeu_si256;
    static constexpr auto store_masked = store_u8bit_with_mask;
    static constexpr auto zero = simde_mm256_setzero_si256;
    inline static constexpr auto broadcast = [](const uint8_t* arr) -> Type {return simde_mm256_set1_epi8(*arr);};
    inline static constexpr auto dot = [](const Type& a, const Type& b, Type& c){
	c = simde_mm256_add_epi8(simde_mm256_mullo_epui8_custom(a, b), c);
    };

};
#else

simde__m128i load_u8bit_from_mask(const int8_t* data, const simde__m128i& mask){
	simde__m128i loaded_data = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data));
	simde__m128i result = simde_mm_and_si128(loaded_data, mask);
	return result;
}

void store_u8bit_with_mask(int8_t* data, const simde__m128i& mask_data, const simde__m128i& vector) {
	uint8_t values_arr[16];
	simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(values_arr), vector);

	int8_t mask_arr[16];
	simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(mask_arr), mask_data);

	for (int i = 0; i < 16; i++) {
		if (mask_arr[i] != 0) {
			data[i] = values_arr[i];
		}
	}
}


//need to test this further
simde__m128i mullo_epui8_simde_custom(simde__m128i a, simde__m128i b) {
    // Multiply pairs of 16-bit integers
    simde__m128i product = simde_mm_mullo_epi16(a, b);

    // Pack the 16-bit products into 8-bit integers, truncating the higher 8 bits
    return simde_mm_packus_epi16(product, product);
}




template <>
struct SimdTraits<uint8_t> {
    using Type = simde__m128i;
    static constexpr size_t pack_size = 16;
    static constexpr size_t tile_size = 32;
    static constexpr auto load = simde_mm_load_si256;
    static constexpr auto loadu = simde_mm_loadu_si256;
    static constexpr auto load_masked = load_u8bit_from_mask;
    static constexpr auto set = simde_mm_set_epi8;
    static constexpr auto store = simde_mm_store_si256;
    static constexpr auto storeu = simde_mm_storeu_si256;
    static constexpr auto store_masked = store_u8bit_with_mask;
    static constexpr auto zero = simde_mm_setzero_si256;
    inline static constexpr auto broadcast = [](const uint8_t* arr) -> Type {return simde_mm_set1_epi8(*arr);};
    inline static constexpr auto dot = [](const Type& a, const Type& b, Type& c){
	c = simde_mm_add_epi64(mullo_epui8_simde_custom(a, b), c);
    };

};
#endif



#ifdef SIMDE_ARCH_X86_AVX2


simde__m256i load_16bit_from_mask(const int16_t* data, const simde__m256i& mask) {
    // Load 16 int16_t elements from memory into a 256-bit vector
    simde__m256i loaded_data = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(data));
    
    // Perform bitwise AND with the mask to zero out unwanted elements
    simde__m256i result = simde_mm256_and_si256(loaded_data, mask);

    return result;
}


void store_16bit_with_mask(int16_t* data, const simde__m256i& mask_data, const simde__m256i& vector) {
	int16_t values_arr[16];
	simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(values_arr), vector);

	int16_t mask_arr[16];
	simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(mask_arr), mask_data);

	for (int i = 0; i < 16; i++) {
		if (mask_arr[i] != 0) {
			data[i] = values_arr[i];
		}
	}
}

template <>
struct SimdTraits<int16_t> {
    using Type = simde__m256i;
    static constexpr size_t pack_size = 16;
    static constexpr size_t tile_size = 32;
    static constexpr auto load = simde_mm256_load_si256;
    static constexpr auto loadu = simde_mm256_loadu_si256;
    static constexpr auto load_masked = load_16bit_from_mask;
    static constexpr auto set = simde_mm256_set_epi16;
    static constexpr auto store = simde_mm256_store_si256;
    static constexpr auto storeu = simde_mm256_storeu_si256;
    static constexpr auto store_masked = store_16bit_with_mask;
    static constexpr auto zero = simde_mm256_setzero_si256;
    inline static constexpr auto broadcast = [](const int16_t* arr) -> Type {return simde_mm256_set1_epi16(*arr);};
    inline static constexpr auto dot = [](const Type& a, const Type& b, Type& c){
	c = simde_mm256_add_epi16(simde_mm256_mullo_epi16(a, b), c);
    };

};
#else

simde__m128i load_16bit_from_mask(const int16_t* data, const simde__m128i& mask){
	simde__m128i loaded_data = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(data));
	simde__m128i result = simde_mm_and_si128(loaded_data, mask);
	return result;
}

void store_16bit_with_mask(int16_t* data, const simde__m128i& mask_data, const simde__m128i& vector) {
	int16_t values_arr[8];
	simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(values_arr), vector);

	int16_t mask_arr[8];
	simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(mask_arr), mask_data);

	for (int i = 0; i < 8; i++) {
		if (mask_arr[i] != 0) {
			data[i] = values_arr[i];
		}
	}
}

template <>
struct SimdTraits<int16_t> {
    using Type = simde__m128i;
    static constexpr size_t pack_size = 16;
    static constexpr size_t tile_size = 32;
    static constexpr auto load = simde_mm_load_si128;
    static constexpr auto loadu = simde_mm_loadu_si128;
    static constexpr auto load_masked = load_16bit_from_mask;
    static constexpr auto set = simde_mm_set_epi16;
    static constexpr auto store = simde_mm_store_si256;
    static constexpr auto storeu = simde_mm_storeu_si256;
    static constexpr auto store_masked = store_16bit_with_mask;
    static constexpr auto zero = simde_mm_setzero_si128;
    inline static constexpr auto broadcast = [](const int16_t* arr) -> Type {return simde_mm_set1_epi16(*arr);};
    inline static constexpr auto dot = [](const Type& a, const Type& b, Type& c){
	c = simde_mm_add_epi16(simde_mm_mullo_epi16(a, b), c);
    };

};

#endif


#ifdef SIMDE_ARCH_X86_AVX2



void store_u16bit_with_mask(int16_t* data, const simde__m256i& mask_data, const simde__m256i& vector) {
	uint16_t values_arr[16];
	simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(values_arr), vector);

	int16_t mask_arr[16];
	simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(mask_arr), mask_data);

	for (int i = 0; i < 16; i++) {
		if (mask_arr[i] != 0) {
			data[i] = values_arr[i];
		}
	}
}

template <>
struct SimdTraits<uint16_t> {
    using Type = simde__m256i;
    static constexpr size_t pack_size = 16;
    static constexpr size_t tile_size = 32;
    static constexpr auto load = simde_mm256_load_si256;
    static constexpr auto loadu = simde_mm256_loadu_si256;
    static constexpr auto load_masked = load_16bit_from_mask;
    static constexpr auto set = simde_mm256_set_epi16;
    static constexpr auto store = simde_mm256_store_si256;
    static constexpr auto storeu = simde_mm256_storeu_si256;
    static constexpr auto store_masked = store_u16bit_with_mask;
    static constexpr auto zero = simde_mm256_setzero_si256;
    inline static constexpr auto broadcast = [](const uint16_t* arr) -> Type {return simde_mm256_set1_epi16(*arr);};
    inline static constexpr auto dot = [](const Type& a, const Type& b, Type& c){
	c = simde_mm256_add_epi16(simde_mm256_mullo_epi16(a, b), c);
    };

};

#elif defined(SIMDE_ARCH_X86_AVX)


void store_u16bit_with_mask(int16_t* data, const simde__m128i& mask_data, const simde__m128i& vector) {
	uint16_t values_arr[8];
	simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(values_arr), vector);

	int16_t mask_arr[8];
	simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(mask_arr), mask_data);

	for (int i = 0; i < 8; i++) {
		if (mask_arr[i] != 0) {
			data[i] = values_arr[i];
		}
	}
}

template <>
struct SimdTraits<uint16_t> {
    using Type = simde__m128i;
    static constexpr size_t pack_size = 16;
    static constexpr size_t tile_size = 32;
    static constexpr auto load = simde_mm_load_si128;
    static constexpr auto loadu = simde_mm_loadu_si128;
    static constexpr auto load_masked = load_16bit_from_mask;
    static constexpr auto set = simde_mm_set_epi16;
    static constexpr auto store = simde_mm_store_si256;
    static constexpr auto storeu = simde_mm_storeu_si256;
    static constexpr auto store_masked = store_u16bit_with_mask;
    static constexpr auto zero = simde_mm_setzero_si128;
    inline static constexpr auto broadcast = [](const uint16_t* arr) -> Type {return simde_mm_set1_epi16(*arr);};
    inline static constexpr auto dot = [](const Type& a, const Type& b, Type& c){
	c = simde_mm_add_epi16(simde_mm_mullo_epi16(a, b), c);
    };

};
#endif





template<typename T>
inline constexpr size_t tile_size_v = simde_supported_v<T> ? SimdTraits<T>::tile_size : 0;

template<typename T>
inline constexpr size_t pack_size_v = simde_supported_v<T> ? SimdTraits<T>::pack_size : 0;


template<typename T>
using simde_type = typename SimdTraits<T>::Type;


//max pack size is 32, so it needs to account for all of it
alignas(64) static const int8_t mask_data[64] = { 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	0,  0,  0,  0,  0,  0,  0,  0, 
	0,  0,  0,  0,  0,  0,  0,  0,
	0,  0,  0,  0,  0,  0,  0,  0, 
	0,  0,  0,  0,  0,  0,  0,  0
};

alignas(64) static const int16_t mask_data_16[64] = { 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	0,  0,  0,  0,  0,  0,  0,  0, 
	0,  0,  0,  0,  0,  0,  0,  0,
	0,  0,  0,  0,  0,  0,  0,  0, 
	0,  0,  0,  0,  0,  0,  0,  0
};

alignas(64) static const int64_t mask_data_64[16] = { 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	0,  0,  0,  0,  0,  0,  0,  0, 
};

alignas(64) static const int32_t mask_data_32[32] = { 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, 
	0,  0,  0,  0,  0,  0,  0,  0, 
	0,  0,  0,  0,  0,  0,  0,  0 
};



template<typename T>
void print_simde_m128i(const simde__m128i& mask) {
    // Create a buffer to store the elements (256 bits / 32 bits per element = 8 elements)
#ifdef SIMDE_ARCH_X86_AVX2
    constexpr size_t pack_size = pack_size_v<T> / 2;
#else
    constexpr size_t pack_size = pack_size_v<T>;
#endif
    T elements[pack_size];
    
    // Store the mask's content into the buffer
    simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(elements), mask);

    // Print the elements
    std::cout << "simde__m128i: { ";
    for (int i = 0; i < pack_size; ++i) {
	if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>){
		std::cout << (int)elements[i];
	}else{
		std::cout << elements[i];
	}
        if (i < (pack_size-1)) std::cout << ", ";
    }
    std::cout << " }" << std::endl;

}

#ifdef SIMDE_ARCH_X86_AVX2
template<typename T>
void print_simde_m256i(const simde__m256i& mask) {
    // Create a buffer to store the elements (256 bits / 32 bits per element = 8 elements)
    constexpr size_t pack_size = pack_size_v<T>;
    T elements[pack_size];
    
    // Store the mask's content into the buffer
    simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(elements), mask);

    // Print the elements
    std::cout << "simde__m256i mask: { ";
    for (int i = 0; i < pack_size; ++i) {
	if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>){
		std::cout << (int)elements[i];
	}else{
		std::cout << elements[i];
	}
        if (i < (pack_size-1)) std::cout << ", ";
    }
    std::cout << " }" << std::endl;

}

using mask_type = simde__m256i;
#else
using mask_type = simde__m128i;
#endif
template<typename T, size_t N>
inline constexpr simde__m256i generate_mask() noexcept {
	constexpr size_t pack_size = pack_size_v<T>;
	static_assert(N <= pack_size, "N cannot exceed the number of elements in the SIMD register.");
	if constexpr (pack_size == pack_size_v<int32_t>){
		return SimdTraits<int32_t>::loadu(reinterpret_cast<const mask_type*>(&mask_data_32[16-N]));
	}
	else if constexpr (pack_size == pack_size_v<int64_t>){
		return SimdTraits<int64_t>::loadu(reinterpret_cast<const mask_type*>(&mask_data_64[8-N]));
	}
	else if constexpr (pack_size == pack_size_v<int8_t>){
		return SimdTraits<int8_t>::loadu(reinterpret_cast<const mask_type*>(&mask_data[32-N]));
	}else{
		static_assert(pack_size == pack_size_v<int16_t>, "Unexpected mask pack size");
		return SimdTraits<int16_t>::loadu(reinterpret_cast<const mask_type*>(&mask_data_16[32-N]));
	}
}

//addition is the number of cols in a packed block
template<typename T, size_t ADDITION, size_t... Indices>
inline constexpr std::array<simde_type<T>, sizeof...(Indices)> load_threaded_row_elements (
    const T* A, std::index_sequence<Indices...>
) noexcept {
    constexpr size_t ratio_tile_pack = tile_size_v<T> / pack_size_v<T>;
    if constexpr (std::is_integral<T>::value || std::is_unsigned<T>::value){
	return { SimdTraits<T>::load((simde_type<T>*)&A[(pack_size_v<T> * (Indices % ratio_tile_pack)) + (ADDITION * (Indices / ratio_tile_pack))])... };
    }else{
	return { SimdTraits<T>::load(&A[(pack_size_v<T> * (Indices % ratio_tile_pack)) + (ADDITION * (Indices / ratio_tile_pack))])... };
    }
}


//in this case,
//skip is the number of "Packs" to skip
template<typename T, size_t ADDITION, size_t skip, size_t... Indices>
inline constexpr std::array<simde_type<T>, sizeof...(Indices)> load_threaded_row_elements_skip (
    const T* A, std::index_sequence<Indices...>
) noexcept {
    
	constexpr size_t ratio_tile_pack = tile_size_v<T> / pack_size_v<T>;
	constexpr size_t skip_ratio = ratio_tile_pack - skip;
	if constexpr (std::is_integral<T>::value || std::is_unsigned<T>::value){
		return { SimdTraits<T>::load((simde_type<T>*)&A[(pack_size_v<T> * (Indices % skip_ratio)) + (ADDITION * (Indices / skip_ratio))])... };
	}else{
		return { SimdTraits<T>::load(&A[(pack_size_v<T> * (Indices % skip_ratio)) + (ADDITION * (Indices / skip_ratio))])... };
	}
}


template<typename T, size_t per_row, size_t... Indices>
inline constexpr void load_c_elements_2(
		T* C, size_t src_c_cols, simde_type<T>* arr, std::index_sequence<Indices...>
) noexcept {
	if constexpr (std::is_integral<T>::value || std::is_unsigned<T>::value){
		((arr[Indices] =  SimdTraits<T>::loadu((simde_type<T>*)&C[(pack_size_v<T> * (Indices % per_row)) + (src_c_cols * (Indices / per_row))])), ...);

	}else{
		((arr[Indices] =  SimdTraits<T>::loadu(&C[(pack_size_v<T> * (Indices % per_row)) + (src_c_cols * (Indices / per_row))])), ...);
	}
}




template<typename T, size_t per_row, size_t... Indices>
inline constexpr void store_c_elements (
		T* C, const size_t& src_c_cols, simde_type<T>* rowCs, std::index_sequence<Indices...>
) noexcept {
	if constexpr (std::is_integral<T>::value || std::is_unsigned<T>::value){
	(SimdTraits<T>::storeu((simde_type<T>*)&C[(pack_size_v<T> * (Indices % per_row)) + (src_c_cols * (Indices / per_row))], rowCs[Indices]), ...);
	}else{
	(SimdTraits<T>::storeu(&C[(pack_size_v<T> * (Indices % per_row)) + (src_c_cols * (Indices / per_row))], rowCs[Indices]), ...);
	}
}


template<typename T, size_t... Indices>
inline constexpr void load_c_elements_masked(
		T* C, const size_t& src_c_cols, simde__m256i& mask, simde_type<T>* arr, std::index_sequence<Indices...>
) noexcept {
	if constexpr (std::is_unsigned<T>::value){
	((arr[Indices] = SimdTraits<T>::load_masked(reinterpret_cast<std::make_signed_t<T>*>(&C[src_c_cols * Indices]), mask)), ...);
	}else{
	((arr[Indices] = SimdTraits<T>::load_masked(&C[src_c_cols * Indices], mask)), ...);
	}
}

template<typename T, size_t... Indices>
inline constexpr void load_c_elements_masked_2(
		T* C, const size_t& src_c_cols, simde__m256i& mask, simde_type<T>* arr, std::index_sequence<Indices...>
) noexcept {
	if constexpr (std::is_integral<T>::value || std::is_unsigned<T>::value){
		((arr[Indices*2] = SimdTraits<T>::loadu((simde_type<T>*)&C[src_c_cols * Indices])), ...);
	}else{
		((arr[Indices*2] = SimdTraits<T>::loadu(&C[src_c_cols * Indices])), ...);
	}
	if constexpr (std::is_unsigned<T>::value){
	((arr[Indices*2+1] = SimdTraits<T>::load_masked(reinterpret_cast<std::make_signed_t<T>*>(&C[pack_size_v<T>  + (src_c_cols * Indices)]), mask)), ...);
	}else{
	((arr[Indices*2+1] = SimdTraits<T>::load_masked(&C[pack_size_v<T>  + (src_c_cols * Indices)], mask)), ...);
	}
}


template<typename T, size_t... Indices>
inline constexpr void store_c_elements_masked(
		T* C, const size_t& src_c_cols, simde__m256i& mask, simde_type<T>* arr, std::index_sequence<Indices...>
) noexcept {

	/* if constexpr (std::is_integral<T>::value){ */
	/* (SimdTraits<T>::store_masked(static_cast<simde_type<T>*>(&C[src_c_cols * Indices]), mask, arr[Indices]), ...); */
	/* }else{ */
	if constexpr (std::is_unsigned<T>::value){
	(SimdTraits<T>::store_masked(reinterpret_cast<std::make_signed_t<T>*>(&C[src_c_cols * Indices]), mask, arr[Indices]), ...);
	}else{
	(SimdTraits<T>::store_masked(&C[src_c_cols * Indices], mask, arr[Indices]), ...);
	}
	/* } */
}

template<typename T, size_t... Indices>
inline constexpr void store_c_elements_masked_2(
		T* C, const size_t& src_c_cols, simde__m256i& mask, simde_type<T>* arr, std::index_sequence<Indices...>
) noexcept {
	if constexpr(std::is_integral<T>::value || std::is_unsigned<T>::value){
	(SimdTraits<T>::storeu((simde_type<T>*)&C[src_c_cols * Indices], arr[Indices*2]), ...);
	}else{
	(SimdTraits<T>::storeu(&C[src_c_cols * Indices], arr[Indices*2]), ...);
	}
	if constexpr (std::is_unsigned<T>::value){
	(SimdTraits<T>::store_masked(reinterpret_cast<std::make_signed_t<T>*>(&C[pack_size_v<T>  + (src_c_cols * Indices)]), mask, arr[Indices*2+1]), ...);
	}else{
	(SimdTraits<T>::store_masked(&C[pack_size_v<T>  + (src_c_cols * Indices)], mask, arr[Indices*2+1]), ...);
	}
}


//this is only to be run if A_COLS > pack_size_v<T>
//tile size is never more than 2 times pack_size_v<T>
template<typename T>
inline constexpr void fused_product_2(simde_type<T>& aVec, const T* A, simde_type<T>& C0, simde_type<T>& C1, const simde_type<T>& B0, const simde_type<T>& B1) noexcept{
	aVec = SimdTraits<T>::broadcast(A);
	SimdTraits<T>::dot(aVec, B0, C0);
	SimdTraits<T>::dot(aVec, B1, C1);
}

template<typename T>
inline constexpr void fused_product_1(simde_type<T>& aVec, const T* A, simde_type<T>& C0, const simde_type<T>& B0) noexcept{
	aVec = SimdTraits<T>::broadcast(A);
	SimdTraits<T>::dot(aVec, B0, C0);

}

template<typename T, size_t total_row_elements, size_t... colIndices>
inline constexpr void second_loop_direct_2(simde_type<T>& aVec, const T* A, simde_type<T>& C0, simde_type<T>& C1,
		const std::array<simde_type<T>, total_row_elements>& rowBs,
		std::index_sequence<colIndices...>) noexcept{
	(fused_product_2(aVec, A + colIndices, C0, C1, rowBs[colIndices * 2], rowBs[colIndices * 2 + 1]), ...);
}

template<typename T, size_t total_row_elements, size_t... colIndices>
inline constexpr void second_loop_direct_1(simde_type<T>& aVec, const T* A, simde_type<T>& C0, 
		const std::array<simde_type<T>, total_row_elements>& rowBs,
		std::index_sequence<colIndices...>) noexcept{
	(fused_product_1(aVec, A + colIndices, C0, rowBs[colIndices]), ...);
}

//rowIndices corresponds to the number of rows in A
template<typename T, size_t per_row, size_t total_row_elements, size_t A_COLS, size_t... rowIndices>
inline constexpr void run_loops_directly(simde_type<T>& aVec, const T* A,
				const std::array<simde_type<T>, total_row_elements>& rowBs,
				simde_type<T>* rowCs,
				std::index_sequence<rowIndices...>) noexcept {
	if constexpr (per_row == 2){
		//the amount of collumns in A packed is tile_size_v<T>
		(second_loop_direct_2(aVec, A + (tile_size_v<T> * rowIndices), rowCs[rowIndices * 2], rowCs[rowIndices * 2 + 1], rowBs, std::make_index_sequence<A_COLS>{}), ...);
	}else if(per_row == 1){
		(second_loop_direct_1(aVec, A + (tile_size_v<T> * rowIndices), rowCs[rowIndices], rowBs, std::make_index_sequence<A_COLS>{}), ...);
		
	}
	
	
}



//addition is the number of collumns in B packed
template<typename T, size_t A_ROWS, size_t A_COLS, size_t B_ROWS, size_t B_COLS, size_t ADDITION>
void kmatmult_simdeT_directly_threaded(const T* A, const T* B, T* C, const size_t& src_c_cols){
	static_assert(A_COLS == B_ROWS, "Expected A_COLS to be the same as B_ROWS");
	//going to load all the row elements from B into vectors and store them in an array
	constexpr size_t rowB_size = B_COLS / pack_size_v<T>;
	constexpr size_t total_row_elements = B_ROWS * rowB_size;
	const std::array<simde_type<T>, total_row_elements> rowBs = load_threaded_row_elements_skip<T, ADDITION,
									(B_COLS == pack_size_v<T> ? 1 : 0)>(B, std::make_index_sequence<total_row_elements>{});

	//now I need to load all of the rows of C into vectors
	constexpr size_t per_row = B_COLS / pack_size_v<T>; // the amount of vectors per row
	static_assert(per_row == 1 || per_row == 2, "Error with per row logic!");
	static_assert(B_COLS % pack_size_v<T> == 0, "Error, directly simdeT does not handle masking, use masked version");
	constexpr size_t total_c_row_elements = per_row * A_ROWS;
	simde_type<T> rowCs[total_c_row_elements];
	load_c_elements_2<T, per_row>(C, src_c_cols, rowCs, std::make_index_sequence<total_c_row_elements>{});
	
	//an element to broadcast A to
	simde_type<T> aVector;

	//run the appropriate dot products
	run_loops_directly<T, per_row, total_row_elements, A_COLS>(aVector, A, rowBs, rowCs, std::make_index_sequence<A_ROWS>{});


	//now to store rowCs back into C:
	store_c_elements<T, per_row>(C, src_c_cols, rowCs, std::make_index_sequence<total_c_row_elements>{});
}



template<typename T, size_t A_ROWS, size_t A_COLS, size_t B_ROWS, size_t B_COLS, size_t ADDITION>
void kmatmult_simdeT_masked_threaded(const T* A, const T* B, T* C, const size_t& src_c_cols){
	static_assert(A_COLS == B_ROWS, "Expected A_COLS to be the same as B_ROWS");
	static_assert(B_COLS % pack_size_v<T> != 0, "B cols should not be divisible by pack size, inefficiency error"); 
	//going to load all the row elements from B into vectors and store them in an array
	constexpr size_t rowB_size =  B_COLS / pack_size_v<T> + 1;
	constexpr size_t total_row_elements = B_ROWS * rowB_size;
	const std::array<simde_type<T>, total_row_elements> rowBs = load_threaded_row_elements_skip<T, ADDITION, 
	      B_COLS < pack_size_v<T> ? 1 : 0>(B, std::make_index_sequence<total_row_elements>{});
	
	//now I need to load all of the rows of C into vectors
	mask_type mask = generate_mask<T, B_COLS < pack_size_v<T> ? B_COLS : B_COLS - pack_size_v<T>>();
	constexpr size_t per_row = (B_COLS < pack_size_v<T> ? 1 : 2);// the amount of vectors per row
 
	constexpr size_t total_c_row_elements = per_row * A_ROWS;// the amount of vectors per row  
	simde_type<T> rowCs[total_c_row_elements];
	if constexpr (B_COLS < pack_size_v<T>){
		load_c_elements_masked<T>(C, src_c_cols, mask, rowCs, std::make_index_sequence<total_c_row_elements>{});	
	}else{
		load_c_elements_masked_2<T>(C, src_c_cols, mask, rowCs, std::make_index_sequence<total_c_row_elements/2>{});	
	}
	

	//an element to broadcast A to
	simde_type<T> aVector;

	//run the appropriate dot products
	run_loops_directly<T, per_row, total_row_elements, A_COLS>(aVector, A, rowBs, rowCs, std::make_index_sequence<A_ROWS>{});





	//now to store rowCs back into C:
	if constexpr (B_COLS < pack_size_v<T>){
		store_c_elements_masked<T>(C, src_c_cols, mask, rowCs, std::make_index_sequence<total_c_row_elements>{});	
	}else{
		store_c_elements_masked_2<T>(C, src_c_cols, mask, rowCs, std::make_index_sequence<total_c_row_elements/2>{});	
	}
}

template<typename T, size_t A_ROWS, size_t A_COLS, size_t B_ROWS, size_t B_COLS, size_t ADDITION>
void kmatmult_simdeT_threaded_fma(const T* A, const T* B, T* C, const size_t& src_c_cols){
	if constexpr (B_COLS % pack_size_v<T> == 0){
		kmatmult_simdeT_directly_threaded<T, A_ROWS, A_COLS, B_ROWS, B_COLS, ADDITION>(A, B, C, src_c_cols);
	}else{
		kmatmult_simdeT_masked_threaded<T, A_ROWS, A_COLS, B_ROWS, B_COLS, ADDITION>(A, B, C, src_c_cols);
	}
}


}}} //nt::functional::std_functional::


#endif //_NT_MATMULT_SIMDE_AVX_HPP_ 
