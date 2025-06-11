//this is for a standardization of functions and function names using the SIMD Everywhere library
//the use of the SIMD Everywhere is beacuse it is a header-only cross platform library that makes 
//distrabution a lot easier
//basically the point is that a template type can be used to access any of these functions, and it will be specific to 
//any of the supported types
//this really just makes using simd intrinsics a lot easier throughout all of NeuroTensor
//currently, making simd traits structs for each different type of instruction
//done: avx, avx2
//next: avx512, arm_neon

#ifndef _NT_SIMDE_TRAITS_H_
#define _NT_SIMDE_TRAITS_H_
#include <cstddef>
#include <cstddef>
#include <array>
#include <type_traits>
#include "../types/Types.h"
#include <simde/simde-common.h>
#ifndef SIMDE_ARCH_X86_AVX
#include "simde_traits/simde_traits_avx.h" //by default include 2
#elif defined(SIMDE_ARCH_X86_AVX2)
#include "simde_traits/simde_traits_avx2.h"
#else
#include "simde_traits/simde_traits_avx.h"
#endif

//#if defined(SIMDE_ARCH_X86_AVX2) || defined(SIMDE_ARCH_X86_AVX)  


namespace nt{
namespace mp{

#ifdef SIMDE_ARCH_X86_AVX2
//avx2 instructions
template<typename T>
using SimdTraits = SimdTraits_avx2<T>;

template<typename T>
using simde_supported = simde_supported_avx2<T>;

template<typename T>
using simde_svml_supported = simde_svml_supported_avx2<T>;

using mask_type = mask_type_avx2;
template<typename T>
using simde_type = simde_type_avx2<T>;

template<typename T, size_t N>
inline constexpr auto Kgenerate_mask() noexcept {
    return Kgenerate_mask_avx2<T, N>();
}

template<typename T>
inline constexpr auto generate_mask(size_t N) noexcept {
	return generate_mask_avx2<T>(N);
}

template<typename T>
inline constexpr size_t pack_size_v = pack_size_avx2_v<T>;


#elif defined(SIMDE_ARCH_X86_AVX)
//avx instructions
template<typename T>
using SimdTraits = SimdTraits_avx<T>;

template<typename T>
using simde_supported = simde_supported_avx<T>;

template<typename T>
using simde_svml_supported = simde_svml_supported_avx<T>;

using mask_type = mask_type_avx;

template<typename T>
using simde_type = simde_type_avx<T>;



template<typename T, size_t N>
inline constexpr auto Kgenerate_mask() noexcept {
    return Kgenerate_mask_avx<T, N>();
}

template<typename T>
inline constexpr auto generate_mask(size_t N) noexcept {
	return generate_mask_avx<T>(N);
}

template<typename T>
inline constexpr size_t pack_size_v = pack_size_avx_v<T>;

#else
//avx instructions
template<typename T>
using SimdTraits = SimdTraits_avx<T>;

template<typename T>
using simde_supported = simde_supported_avx<T>;

template<typename T>
using simde_svml_supported = simde_svml_supported_avx<T>;

using mask_type = mask_type_avx;

template<typename T>
using simde_type = simde_type_avx<T>;



template<typename T, size_t N>
inline constexpr auto Kgenerate_mask() noexcept {
    return Kgenerate_mask_avx<T, N>();
}

template<typename T>
inline constexpr auto generate_mask(size_t N) noexcept {
	return generate_mask_avx<T>(N);
}

template<typename T>
inline constexpr size_t pack_size_v = pack_size_avx_v<T>;
#endif

template<typename T>
inline constexpr bool simde_supported_v = simde_supported<T>::value;

template<typename T>
inline constexpr bool simde_svml_supported_v = simde_svml_supported<T>::value;

}} //nt::mp::

#endif //_NT_SIMDE_TRAITS_H_
