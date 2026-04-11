// think of this as the same thing as std::iota
// Only type discluded is booleans

//memory/mtl/shaders/iota_kernel.metal

#include <metal_stdlib>
#include "../utils/mtl_shaders.h"
using namespace metal;

#define NT_MAKE_CONTIGUOUS_IOTA_KERNEL(Type)\
kernel void iota_contiguous_kernel_##Type(\
    device Type*    out       [[ buffer(0) ]], \
    constant Type&  start     [[ buffer(1) ]], \
    constant uint3& gris_size [[ buffer(2) ]], \
    uint3 tid [[ thread_position_in_grid ]]){\
    ulong id = tid_to_grid(tid, grid_size); \
    out[id] = (start + nt::utils::convert<Type>(id));\
}\

#define NT_MAKE_CONTIGUOUS_IOTA_KERNEL_VEC(Type)\
kernel void iota_contiguous_kernel_##Type(\
    device Type*    out       [[ buffer(0) ]], \
    constant Type&  start     [[ buffer(1) ]], \
    constant uint3& gris_size [[ buffer(2) ]], \
    uint3 tid [[ thread_position_in_grid ]]){\
    ulong id = tid_to_grid(tid, grid_size); \
    ulong mul = id * 4; \
    Type adding(mul, mul+1, mul+2, mul+3);\
    out[id] = (start + adding);\
}\

NT_MTL_TYPES_FLOAT_GET_(NT_MAKE_CONTIGUOUS_IOTA_KERNEL);
NT_MTL_TYPES_COMPLEX_GET_(NT_MAKE_CONTIGUOUS_IOTA_KERNEL);
NT_MTL_TYPES_SIGNED_GET_(NT_MAKE_CONTIGUOUS_IOTA_KERNEL);
NT_MTL_TYPES_UNSIGNED_GET_(NT_MAKE_CONTIGUOUS_IOTA_KERNEL);
NT_MTL_VECTOR_TYPES_GET_(NT_MAKE_CONTIGUOUS_IOTA_KERNEL_VEC); 
#undef NT_MAKE_CONTIGUOUS_IOTA_KERNEL
#undef NT_MAKE_CONTIGUOUS_IOTA_KERNEL_VEC

#define NT_MAKE_IOTA_AFFINE_KERNEL(Type)\
kernel void iota_affine_kernel_##Type(\
    device Type*        out         [[buffer(0)]],         \
    constant uint&      ndim        [[buffer(1)]],         \
    constant long*      sizes       [[buffer(2)]],         \
    constant long*      strides     [[buffer(3)]],         \
    constant long&      numel       [[buffer(4)]],         \
    constant Type&      start       [[buffer(5)]],         \
    constant uint3& grid_size [[ buffer(6) ]],             \
    uint3 tid [[thread_position_in_grid]]                  \
) {                                                        \
    ulong gid = tid_to_gid(tid, grid_size);                \
    if (gid >= numel) return;                              \
                                                           \
    ulong idx = gid;                                       \
    long in_offset = 0;                                    \
                                                           \
    for (int d = ndim - 1; d >= 0; --d) {                  \
        ulong coord = idx % sizes[d];                      \
        idx /= sizes[d];                                   \
        in_offset += coord * strides[d];                   \
    }                                                      \
    if(offset >= numel) return;                            \
    out[offset] = nt::utils::conver<Type>(offset) + start; \
}                                                          \


NT_MTL_TYPES_FLOAT_GET_(NT_MAKE_AFFINE_IOTA_KERNEL);
NT_MTL_TYPES_COMPLEX_GET_(NT_MAKE_AFFINE_IOTA_KERNEL);
NT_MTL_TYPES_SIGNED_GET_(NT_MAKE_AFFINE_IOTA_KERNEL);
NT_MTL_TYPES_UNSIGNED_GET_(NT_MAKE_AFFINE_IOTA_KERNEL);

#undef NT_MAKE_IOTA_AFFINE_KERNEL

#define NT_MAKE_IOTA_STRIDED_KERNEL(Type)        \
kernel void clone_strided_kernel_##Type(         \
    device Type*         out      [[buffer(0)]], \
    device const long*   indices  [[buffer(1)]], \
    constant long&       nnz      [[buffer(2)]], \
    constant Type&       start    [[buffer(3)]], \
    constant uint3& grid_size [[ buffer(4) ]],   \
    uint3 tid [[thread_position_in_grid]]        \
) {                                              \
    ulong gid = tid_to_gid(tid, grid_size); \
    if (gid < nnz) {                             \
        out[indices[gid]] = nt::utils::convert<Type>(gid) + start;  \
    }                                            \
}                                                \

NT_MTL_TYPES_FLOAT_GET_(NT_MAKE_STRIDED_IOTA_KERNEL);
NT_MTL_TYPES_COMPLEX_GET_(NT_MAKE_STRIDED_IOTA_KERNEL);
NT_MTL_TYPES_SIGNED_GET_(NT_MAKE_STRIDED_IOTA_KERNEL);
NT_MTL_TYPES_UNSIGNED_GET_(NT_MAKE_STRIDED_IOTA_KERNEL);

#undef NT_MAKE_IOTA_STRIDED_KERNEL
