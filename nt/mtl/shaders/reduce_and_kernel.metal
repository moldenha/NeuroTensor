//memory/mtl/shaders/clone_kernel.metal

#include <metal_stdlib>
#include "../utils/mtl_shaders.h"
using namespace metal;

// This is an example of how to predicate a kernel and generate flags:
/* kernel void predicate_kernel(
    device const float* data [[ buffer(0) ]],
    device uint*        flags [[ buffer(1) ]],
    constant uint&      N     [[ buffer(2) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= N) return;
    flags[tid] = (data[tid] > 0.0f) ? 1u : 0u;
} */

kernel void reduce_and_kernel_uint(
    device const uint* in  [[ buffer(0) ]],
    device uint*       out [[ buffer(1) ]],
    constant uint&     N   [[ buffer(2) ]],
    uint tid  [[ thread_position_in_grid ]],
    uint ltid [[ thread_index_in_threadgroup ]]
) {
    threadgroup uint shared[256];
    long idx = tid;

    shared[ltid] = (idx < N) ? in[idx] : 1u;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (ltid < stride) {
            shared[ltid] &= shared[ltid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (ltid == 0) {
        out[tid / 256] = shared[0];
    }
}

