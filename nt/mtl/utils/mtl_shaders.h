// this is a file for mtl shader utilities that can be used with every .metal shader function
#ifndef NT_MTL_UTILS_MTL_SHADERS_H__
#define NT_MTL_UTILS_MTL_SHADERS_H__

#include "../../utils/type_traits.hpp"

#define NT_MTL_TYPES_GET_(func)\
    func(float)\
    func(half)\
    func(char)\
    func(uchar)\
    func(short)\
    func(ushort)\
    func(int)\
    func(uint)\
    func(long)\
    func(float2)\
    func(half2)\
    func(bool)

#define NT_MTL_TYPES_FLOAT_GET_(func)\
    func(float)\
    func(half)\

#define NT_MTL_TYPES_COMPLEX_GET_(func)\
    func(float2)\
    func(half2)\


#define NT_MTL_TYPES_SIGNED_GET_(func)\
    func(char)\
    func(short)\
    func(int)\
    func(long)\

#define NT_MTL_TYPES_UNSIGNED_GET_(func)\
    func(uchar)\
    func(ushort)\
    func(uint)\
    func(ulong)\

// types:
//  - float, float16, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, complex<float>, complex<float16>, bool

// Note: in memory, complex<float> is stored as float2, and complex<float16> is stored as half2
// 

#define NT_MTL_VECTOR_TYPES_GET_(func) \
    func(float4) \
    func(half4)

inline ulong tid_to_gid(uint3 tid, uint3 gridSize) noexcept {
    return (ulong)tid.z * gridSize.y * gridSize.x +
           (ulong)tid.y * gridSize.x +
           (ulong)tid.x;
}


namespace nt::utils{

template<class Out, class In>
inline Out convert(const In& val){
    if constexpr (nt::type_traits::is_same_v<Out, In>){
        return val;
    }else if constexpr (nt::type_traits::is_in_v<Out, half4, float4> && !nt::type_traits::is_in_v<In, half4, float4>){
        using singular = typename nt::type_traits::conditional_t<nt::type_traits::is_same_v<Out, float4>, float, half>;
        Out o_val;
        o_val.x = singular(val);
        o_val.y = singular(val);
        o_val.z = singular(val);
        o_val.w = singular(val);
        return o_val;
    }else if constexpr(nt::type_traits::is_in_v<In, half4, float4> && !nt::type_traits::is_in_v<Out, half4, float4>){
        return Out(val.x);
    }else if constexpr(nt::type_traits::is_in_v<In, half4, float4> && nt::type_traits::is_in_v<Out, half4, float4>){
        using singular = typename nt::type_traits::conditional_t<nt::type_traits::is_same_v<Out, float4>, float, half>;
        Out o_val;
        o_val.x = singular(val);
        o_val.y = singular(val);
        o_val.z = singular(val);
        o_val.w = singular(val);
        return o_val;
    }else if constexpr (nt::type_traits::is_in_v<Out, half2, float2> && !nt::type_traits::is_in_v<In, half2, float2>){
        using singular = typename nt::type_traits::conditional_t<nt::type_traits::is_same_v<Out, float2>, float, half>;
        Out o_val;
        o_val.x = singular(val);
        o_val.y = singular(val);
    }else if constexpr(nt::type_traits::is_in_v<In, half2, float2> && !nt::type_traits::is_in_v<Out, half2, float2>){
        return Out(val.x);
    }else if constexpr(nt::type_traits::is_in_v<In, half2, float2> && nt::type_traits::is_in_v<Out, half2, float2){
        using singular = typename nt::type_traits::conditional_t<nt::type_traits::is_same_v<Out, float2>, float, half>;
        Out o_val;
        o_val.x = singular(val);
        o_val.y = singular(val);
        return o_val;
    } 
    else{
        return Out(val);
    }
}

}

#endif
