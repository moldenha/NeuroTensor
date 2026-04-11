//memory/mtl/shaders/clone_kernel.metal

#include <metal_stdlib>
#include "../utils/mtl_shaders.h"
using namespace metal;

#define NT_MAKE_CLONE_CONTIGUOUS_KERNEL(Type)\
kernel void clone_contiguous_kernel_##Type(\
    device const Type* in [[ buffer(0) ]],\
    device Type* out [[ buffer(1) ]],\
    constant long& numel [[ buffer(2) ]],\
    constant uint3& grid_size [[ buffer(3) ]], \
    uint3 tid [[ thread_position_in_grid ]]){\
    ulong id = tid_to_gid(tid, grid_size); \
    if(id < numel) out[id] = in[id];\
}\

NT_MTL_TYPES_GET_(NT_MAKE_CLONE_CONTIGUOUS_KERNEL);
NT_MTL_VECTOR_TYPES_GET_(NT_MAKE_CLONE_CONTIGUOUS_KERNEL); 
#undef NT_MAKE_CLONE_CONTIGUOUS_KERNEL

// the for loop inside is a row-major unravel
#define NT_MAKE_CLONE_AFFINE_KERNEL(Type)\
kernel void clone_affine_kernel_##Type(\
    device const Type*  in          [[buffer(0)]],\
    device Type*        out         [[buffer(1)]],\
    constant uint&      ndim        [[buffer(2)]],\
    constant long*      sizes       [[buffer(3)]],\
    constant long*      strides     [[buffer(4)]],\
    constant long&     numel       [[buffer(5)]], \
    constant uint3& grid_size [[ buffer(6) ]],    \
    uint3 tid [[thread_position_in_grid]]         \
) {                                               \
    ulong gid = tid_to_gid(tid, grid_size);       \
    if (gid >= numel) return;                     \
                                                  \
    ulong idx = gid;                              \
    long in_offset = 0;                           \
                                                  \
    for (int d = ndim - 1; d >= 0; --d) {         \
        ulong coord = idx % sizes[d];             \
        idx /= sizes[d];                          \
        in_offset += coord * strides[d];          \
    }                                             \
                                                  \
    out[gid] = in[in_offset];                     \
}                                                 \

NT_MTL_TYPES_GET_(NT_MAKE_CLONE_AFFINE_KERNEL);
#undef NT_MAKE_CLONE_AFFINE_KERNEL


#define NT_MAKE_CLONE_STRIDED_KERNEL(Type)       \
kernel void clone_strided_kernel_##Type(         \
    device const Type*   in       [[buffer(0)]], \
    device Type*         out      [[buffer(1)]], \
    device const long*   indices  [[buffer(2)]], \
    constant long&       nnz      [[buffer(3)]], \
    constant uint3& grid_size [[ buffer(4) ]],   \
    uint3 tid [[thread_position_in_grid]]         \
) {                                              \
    ulong gid = tid_to_gid(tid, grid_size); \
    if (gid < nnz) {                             \
        out[gid] = in[indices[gid]];             \
    }                                            \
}                                                \


NT_MTL_TYPES_GET_(NT_MAKE_CLONE_STRIDED_KERNEL);
#undef NT_MAKE_CLONE_STRIDED_KERNEL

