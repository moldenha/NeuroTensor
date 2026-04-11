// this is a header file that is dedicated to basically converting indexes to the appropriate index 
// based on whether it is coming from a contiguous, affine, or strided memory

//memory/mtl/shaders/indexes_convert_kernel.metal

#include <metal_stdlib>
#include "../utils/mtl_shaders.h"
using namespace metal;

/*
For concatenated indexes converting contiguous
The minus needs to be the addition of all the previous buffers' numels

think of it this way:
a user will use the index [1000] for maybe a concatenated tensor
but that 1000 corresponds to the first index of this contiguous buffer, and then just add the offset
*/

kernel void indexes_convert_contiguous_kernel(
    device long* out_idxs [[ buffer(0) ]],
    constant long& offset [[ buffer(1) ]],
    constant long& minus  [[ buffer(2) ]]
    constant uint3& grid_size [[ buffer(3) ]],
    uint3 tid [[ thread_position_in_grid ]]){
    ulong id = tid_to_gid(tid, grid_size);
    out_idxs[id] += (offset - minus);
}

kernel void indexes_convert_affine_kernel(
    device long* out_idxs [[ buffer(0) ]],
    constant long&      base_offset [[buffer(1)]],
    constant uint&      ndim        [[buffer(2)]],
    constant long*      sizes       [[buffer(3)]],
    constant long*      strides     [[buffer(4)]],
    constant long&      numel       [[buffer(5)]],
    constant long&      minus       [[buffer(6)]],
    constant uint3& grid_size [[ buffer(7) ]],
    uint3 tid [[thread_position_in_grid]]          ){
    
    ulong gid = tid_to_gid(tid, grid_size);
    if(gid >= numel) return;
    ulong idx = out_idxs[gid] - minus;
    if(idx >= numel){
        out_idxs[gid] = 0;
        return;
    }
    long in_offset = base_offset;

    for (int d = ndim - 1; d >= 0; --d) {
        ulong coord = idx % sizes[d];
        idx /= sizes[d];
        in_offset += coord * strides[d];
    }

    out_idxs[gid] = in_offset;
}

kernel void indexes_convert_strided_kernel(
    device long*       out_idxs  [[ buffer(0) ]],
    device const long* in        [[ buffer(1) ]],
    constant long&     nnz       [[ buffer(2) ]],
    constant long&     minus     [[ buffer(3) ]],
    constant uint3&    grid_size [[ buffer(4) ]],
    uint3 tid [[ thread_position_in_grid ]]){
    ulong gid = tid_to_gid(tid, grid_size);
    constant long index = out_idx[gid] - minus;
    out_idxs[gid] = index < nnz ? in[index] : 0;
}




