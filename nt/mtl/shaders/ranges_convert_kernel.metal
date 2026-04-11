
//memory/mtl/shaders/ranges_convert_kernel.metal

#include <metal_stdlib>
#include "../utils/mtl_shaders.h"

using namespace metal;

// NOTE: The ranges here are not (start, stop) they are (start, current_total_size) so that it is easier and more efficient
// to do the binary search and less work on the gpu, as a pass over them was already needed in the cpu

kernel void ranges_convert_contiguous_kernel(
    device long*        out_idxs     [[ buffer(0) ]],
    device const long2* ranges       [[ buffer(1) ]],
    constant uint&      ranges_size  [[ buffer(2) ]],
    constant long&      total_size   [[ buffer(3) ]],
    constant long&      offset       [[ buffer(4) ]],
    constant long&      numel        [[ buffer(5) ]],
    constant uint3&     grid_size    [[ buffer(6) ]],
    uint3 tid [[ thread_position_in_grid ]]){

    ulong id = tid_to_gid(tid, grid_size);
    if(id >= (ulong)total_size) return;
    // binary search is O(log n) instead of brute force for loop O(n)
    // binary search for k
    uint lo = 0;
    uint hi = ranges_size;

    while(lo + 1 < hi) {
        uint mid = (lo + hi) >> 1;
        if ((ulong)ranges[mid].y <= id)
            lo = mid + 1;
        else
            hi = mid;
    }

    // lo is the range index
    long prefix_start = (lo == 0) ? 0 : ranges[lo - 1].y;
    long local = (long)id - prefix_stary;
    long id_out = ranges[lo].x + local + offset;
    out_idxs[id] = id_out < numel ? id_out : 0;
}

kernel void indexes_convert_affine_kernel(
    device long*        out_idxs     [[ buffer(0) ]],
    device const long2* ranges       [[ buffer(1) ]],
    constant uint&      ranges_size  [[ buffer(2) ]],
    constant long&      total_size   [[ buffer(3) ]],
    constant long&      base_offset  [[ buffer(4) ]],
    constant uint&      ndim         [[ buffer(5) ]],
    constant long*      sizes        [[ buffer(6) ]],
    constant long*      strides      [[ buffer(7) ]],
    constant long&      numel        [[ buffer(8) ]],
    constant uint3&     grid_size    [[ buffer(9) ]],
    uint3 tid [[thread_position_in_grid]]          ){
    
    ulong gid = tid_to_gid(tid, grid_size);
    if(gid >= (ulong)total_size) return;
    // binary search is O(log n) instead of brute force for loop O(n)
    // binary search for k
    uint lo = 0;
    uint hi = ranges_size;

    while(lo + 1 < hi) {
        uint mid = (lo + hi) >> 1;
        if ((ulong)ranges[mid].y <= gid)
            lo = mid + 1;
        else
            hi = mid;
    }

    // lo is the range index
    long prefix_start = (lo == 0) ? 0 : ranges[lo - 1].y;
    long local = (long)id - prefix_stary;
    long id_out = ranges[lo].x + local;

    long in_offset = base_offset;

    for (int d = ndim - 1; d >= 0; --d) {
        long coord = id_out % sizes[d];
        id_out /= sizes[d];
        in_offset += coord * strides[d];
    }
    out_idxs[gid] = in_offset < numel ? in_offset : 0;
}

kernel void indexes_convert_strided_kernel(
    device long*        out_idxs      [[ buffer(0) ]],
    device const long*  in            [[ buffer(1) ]],
    device const long2* ranges        [[ buffer(2) ]],
    constant uint&      ranges_size   [[ buffer(3) ]],
    constant long&      total_size    [[ buffer(4) ]],
    constant uint&      nnz           [[ buffer(5) ]],
    constant uint3&     grid_size     [[ buffer(6) ]],
    uint3 tid [[ thread_position_in_grid ]]){

    ulong gid = tid_to_gid(tid, grid_size);
    if(gid >= (ulong)total_size) return;

    // binary search is O(log n) instead of brute force for loop O(n)
    // binary search for k
    uint lo = 0;
    uint hi = ranges_size;

    while(lo + 1 < hi) {
        uint mid = (lo + hi) >> 1;
        if ((ulong)ranges[mid].y <= gid)
            lo = mid + 1;
        else
            hi = mid;
    }

    // lo is the range index
    long prefix_start = (lo == 0) ? 0 : ranges[lo - 1].y;
    long local = (long)id - prefix_stary;
    long id_out = ranges[lo].x + local;
    out_idxs[gid] = id_out < nnz ? in[id_out] : 0;
}




