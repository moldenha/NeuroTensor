// This is a structure to hold metal tensors


/* How the gpu tensor needs to be initialized internally:
 * ELEMENT WISE INSTRUCTIONS:
 * For Tensors where the function does not care about dimensionality, and it is basically just treated as a pointer (like relu element wise)
 * For contiguous:
 *      NtGPUTensor<(data dtype)>{
 *          data, nullptr, 0,
 *          0, nullptr, nullptr, numel, offset
 *      };
 *  for affine:
 *      NtGPUTensor<(data dtype)>{
 *          data, nullptr, 1,
 *          ndims, sizes, strides, numel, offset
 *      }
 *  for strided:
 *      NtGPUTensor<(data type)>{
 *          data, indices, 2,
 *          0, nullptr, nullptr, numel, offset
 *      }
 * MULTI DIM TENSOR INSTRUCTIONS:
 * This is for tensors where you are going to care about them dimensionality-wise (for example a tensor of 3 dims)
 *  - NOTE: This example will have a dimensionality of 3, but that is not the limit
 *
 *  - Important: Make sure that you flatten or take into account exactly how the tensor should be handled
 *  - notice how the ndim will match the constexpr Dim
 *  - Contiguous:
 *     - sizes and strides can be generated based on the original tensor
 *     NtGPUTensor<(data type)>{
 *          data, nullptr, 0,
 *          3, sizes, sizes, numel, offset
 *     };
 *  - Affine:
 *      NtGPUTensor<(data dtype)>{
 *          data, nullptr, 1,
 *          3, sizes, strides, numel, offset
 *      }
 *  - Strided:
 *      - the strides and sizes would be the same as if it were contiguous
 *      NtGPUTensor<(data type){
 *          data, indices, 2,
 *          3, sizes, strides, numel, offset
 *      }
 *      
 */


// dispatch params should be in every single kernel
struct DispatchParams{
    long start, end;
};

template<typename T>
struct NtGPUTensor {
    device T* data                  [[ id(0) ]];
    device const int64_t* indices   [[ id(1) ]];   // null if not indexed

    int layout                      [[ id(2) ]];  // 0=contig,1=affine,2=indexed
    int ndim                        [[ id(3) ]];

    int64_t numel                   [[ id(4) ]];
    int64_t offset                  [[ id(5) ]];
    int64_t sizes[8];
    int64_t strides[8];
};


template<typename T>
inline long compute_offset(const NtGPUTensor<T> t, long idx) {
    switch(t.layout){
        case 0: // contiguous
            return idx + t.offset;
        case 1:{ // affine
            long in_offset = t.offset;
            for (int d = t.ndim - 1; d >= 0; --d) {
                ulong coord = idx % t.sizes[d];
                idx /= t.sizes[d];
                in_offset += coord * t.strides[d];
            }
            return in_offset;
        }
        case 2: // stried
            return t.indices[idx + t.offset];
        default:
            return 0;
    }
}

// this assumes that the correct number of dimensions has been taken into account already
template<typename T>
inline long compute_offset(const NtGPUTensor<T> t, long idx_0, long idx_1) {
    if(t.ndim != 2) return 0;
    // on contiguous and strided strides[1] will just be 1
    switch(t.layout){
        case 0: // contiguous
        case 1: // affine
            return ((idx_0 * t.strides[0]) + 
                    (idx_1 * t.strides[1])) + 
                    t.offset;
        case 2: // strided
            return t.indices[((idx_0 * t.strides[0]) + idx_1) + t.offset];
    }
}

template<typename T>
inline long compute_offset(const NtGPUTensor<T> t, long idx_0, long idx_1, long idx_2) {
    if(t.ndim != 3) return 0;
    // on contiguous and strided strides[1] will just be 1
    switch(t.layout){
        case 0: // contiguous
        case 1: // affine
            return ((idx_0 * t.strides[0]) + 
                    (idx_1 * t.strides[1]) + 
                    (idx_2 * t.strides[2])) + 
                    t.offset;
        case 2: // strided
            return t.indices[((idx_0 * t.strides[0]) + (idx_1 * t.strides[1]) + idx_2) + t.offset];
    }
}

template<typename T>
inline long compute_offset(const NtGPUTensor<T> t, long idx_0, long idx_1, long idx_2, long idx_3) {
    if(t.ndim != 4) return 0;
    // on contiguous and strided strides[1] will just be 1
    switch(t.layout){
        case 0: // contiguous
        case 1: // affine
            return ((idx_0 * t.strides[0]) + 
                    (idx_1 * t.strides[1]) + 
                    (idx_2 * t.strides[2]) +
                    (idx_3 * t.strides[3])) +
                    t.offset;
        case 2: // strided
            return t.indices[((idx_0 * t.strides[0]) + (idx_1 * t.strides[1]) + (idx_2 * t.strides[2]) + idx_3) + t.offset];
    }
}

template<typename T>
inline long compute_offset(const NtGPUTensor<T> t, long idx_0, long idx_1, long idx_2, long idx_3, long idx_4) {
    if(t.ndim != 5) return 0;
    // on contiguous and strided strides[1] will just be 1
    switch(t.layout){
        case 0: // contiguous
        case 1: // affine
            return ((idx_0 * t.strides[0]) + 
                    (idx_1 * t.strides[1]) + 
                    (idx_2 * t.strides[2]) +
                    (idx_3 * t.strides[3]) +
                    (idx_4 * t.strides[4])) +
                    t.offset;
        case 2: // strided
            return t.indices[((idx_0 * t.strides[0]) + 
                    (idx_1 * t.strides[1]) + 
                    (idx_2 * t.strides[2]) + 
                    (idx_3 * t.strides[3]) + 
                    idx_4) + 
                    t.offset];
    }
}

template<typename T>
inline long compute_offset(const NtGPUTensor<T> t, long idx_0, long idx_1, long idx_2, long idx_3, long idx_4, long idx_5) {
    if(t.ndim != 6) return 0;
    // on contiguous and strided strides[1] will just be 1
    switch(t.layout){
        case 0: // contiguous
        case 1: // affine
            return ((idx_0 * t.strides[0]) + 
                    (idx_1 * t.strides[1]) + 
                    (idx_2 * t.strides[2]) +
                    (idx_3 * t.strides[3]) +
                    (idx_4 * t.strides[4]) +
                    (idx_5 * t.strides[5])) +
                    t.offset;
        case 2: // strided
            return t.indices[((idx_0 * t.strides[0]) + 
                    (idx_1 * t.strides[1]) + 
                    (idx_2 * t.strides[2]) + 
                    (idx_3 * t.strides[3]) + 
                    (idx_4 * t.strides[4]) +
                    idx_5) + 
                    t.offset];
    }
}

template<typename T>
inline long compute_offset(const NtGPUTensor<T> t, long idx_0, long idx_1, long idx_2, long idx_3, long idx_4, long idx_5,
                                                    long idx_6) {
    if(t.ndim != 7) return 0;
    // on contiguous and strided strides[1] will just be 1
    switch(t.layout){
        case 0: // contiguous
        case 1: // affine
            return ((idx_0 * t.strides[0]) + 
                    (idx_1 * t.strides[1]) + 
                    (idx_2 * t.strides[2]) +
                    (idx_3 * t.strides[3]) +
                    (idx_4 * t.strides[4]) +
                    (idx_5 * t.strides[5]) +
                    (idx_6 * t.strides[6])) +
                    t.offset;
        case 2: // strided
            return t.indices[((idx_0 * t.strides[0]) + 
                    (idx_1 * t.strides[1]) + 
                    (idx_2 * t.strides[2]) + 
                    (idx_3 * t.strides[3]) + 
                    (idx_4 * t.strides[4]) +
                    (idx_5 * t.strides[5]) + 
                    idx_6) + 
                    t.offset];
    }
}


template<typename T>
inline long compute_offset(const NtGPUTensor<T> t, long idx_0, long idx_1, long idx_2, long idx_3, long idx_4, long idx_5,
                                                    long idx_6, long idx_7) {
    if(t.ndim != 8) return 0;
    // on contiguous and strided strides[1] will just be 1
    switch(t.layout){
        case 0: // contiguous
        case 1: // affine
            return ((idx_0 * t.strides[0]) + 
                    (idx_1 * t.strides[1]) + 
                    (idx_2 * t.strides[2]) +
                    (idx_3 * t.strides[3]) +
                    (idx_4 * t.strides[4]) +
                    (idx_5 * t.strides[5]) +
                    (idx_6 * t.strides[6]) +
                    (idx_7 * t.strides[7])) +
                    t.offset;
        case 2: // strided
            return t.indices[((idx_0 * t.strides[0]) + 
                    (idx_1 * t.strides[1]) + 
                    (idx_2 * t.strides[2]) + 
                    (idx_3 * t.strides[3]) + 
                    (idx_4 * t.strides[4]) +
                    (idx_5 * t.strides[5]) + 
                    (idx_6 * t.strides[6]) +
                    idx_7) + 
                    t.offset];
    }
}
