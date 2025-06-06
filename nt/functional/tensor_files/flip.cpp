#include "flip.h"
#include "exceptions.hpp"
#include "../../mp/Threading.h"
#include <simde/x86/avx2.h>
#include <simde/x86/avx512.h>
#include <type_traits>

namespace nt{
namespace functional{

namespace details{

void reverse_manual(const std::uintptr_t* input, std::uintptr_t* output, int64_t size){
    threading::preferential_parallel_for(threading::block_ranges<1>(0, size),
    [&](threading::blocked_range<1> range){
    for(int64_t i = range.begin[0]; i != range.end[0]; ++i){
        output[size-1-i] = input[i];
    }});
}


#if defined(SIMDE_X86_AVX512F_NATIVE) || defined(__arm64__)
void reverse_intrinsics(const int64_t* input, int64_t* output, int64_t size) {
    int64_t i = 0;
    constexpr int64_t simd_width = 8;
    static simde__m512i idx = simde_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
    // Reverse in chunks of 4
    for (; i + simd_width <= size; i += simd_width) {
        simde__m512i vec = simde_mm512_loadu_si512(reinterpret_cast<const simde__m512i*>(input + i));
        simde__m512i reversed = simde_mm512_permutexvar_epi64(idx, vec);
        simde_mm512_storeu_si512(reinterpret_cast<simde__m512i*>(output + size - i - simd_width), reversed);
    }

    // Handle the tail (if size not divisible by 8)
    for (size_t j = 0; j < size % simd_width; ++j) {
        output[j] = input[size - 1 - j];
    }
}

#else 
void reverse_intrinsics(const int64_t* input, int64_t* output, int64_t size) {
    int64_t i = 0;
    constexpr int64_t simd_width = 4;
    static simde__m256i idx = simde_mm256_setr_epi64x(3, 2, 1, 0);
    // Reverse in chunks of 4
    for (; i + simd_width <= size; i += simd_width) {
        simde__m256i vec = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(input + i));
        simde__m256i reversed = simde_mm256_permutexvar_epi64(idx, vec);
        simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(output + size - i - simd_width), reversed);
    }

    // Handle the tail (if size not divisible by 8)
    for (size_t j = 0; j < size % simd_width; ++j) {
        output[j] = input[size - 1 - j];
    }
}

#endif

void reverse_view(void** _in, void** _out, int64_t numel){

    if constexpr (sizeof(int64_t) == sizeof(void*)){
        reverse_intrinsics(reinterpret_cast<const int64_t*>(_in), reinterpret_cast<int64_t*>(_out), numel);
    } else{
        reverse_manual(reinterpret_cast<const std::uintptr_t*>(_in), reinterpret_cast<std::uintptr_t*>(_out), numel);
    }
}

void reverse_strided_unrolled(const std::uintptr_t* __input, std::uintptr_t* __output,
                                 int64_t size, int64_t stride,
                                 int64_t inner_batches, int64_t outter_batches,
                                 int64_t next) {
    int64_t stride_size = size * stride;
    
    threading::preferential_parallel_for(
    threading::block_ranges<1>(0, outter_batches),
    [&](threading::blocked_range<1> range){
    const std::uintptr_t* input;
    std::uintptr_t* output;
    for (int64_t ob = range.begin[0]; ob != range.end[0]; ++ob) {
        input = __input + (next * ob);
        output = __output + (next * ob);
        for (int64_t b = 0; b < inner_batches; ++b) {
            int64_t i = 0;
            for (; i + 4 * stride <= stride_size; i += 4 * stride) {
                output[stride_size - stride - (i + 3 * stride)] = input[i + 3 * stride];
                output[stride_size - stride - (i + 2 * stride)] = input[i + 2 * stride];
                output[stride_size - stride - (i + 1 * stride)] = input[i + 1 * stride];
                output[stride_size - stride - i] = input[i];
            }

            // Handle tail manually
            for (; i < stride_size; i += stride) {
                output[stride_size - stride - i] = input[i];
            }

            ++input;
            ++output;
        }
    }});
}

void reverse_strided_from(void** in, void** out, const SizeRef& shape, int64_t dim, const int64_t& numel){
    dim = dim < 0 ? dim + shape.size(): dim;
    utils::throw_exception(dim > 0 && dim < shape.size(), "Cannot flip dimension $ for tensor with shape $", 
        (dim < 0 ? dim - shape.size() : dim), shape);
    int64_t next = 1;
    for(int64_t i = dim; i < shape.size(); ++i){
        next *= shape[i];
    }
    int64_t inner_batches = next / shape[dim];
    int64_t outter_batches = numel / next;
    reverse_strided_unrolled(reinterpret_cast<std::uintptr_t*>(in), reinterpret_cast<std::uintptr_t*>(out),
                           /*size=*/shape[dim], /*stride = */ inner_batches, 
                           /*inner_batches=*/ inner_batches, /*outter_batches = */ outter_batches, 
                           /*next = */ next);

}

}

Tensor flip_view(const Tensor& t, utils::optional_list list){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    ArrayVoid in_vals = t.arr_void().get_bucket().is_strided()
                                 ? t.arr_void()
                                 : t.arr_void().bucket_all_indices();
    void **_in = in_vals.stride_begin();
    ArrayVoid out_vals = t.arr_void().new_strides(t.numel());
    void **_out = out_vals.stride_begin();
    if(!list){
        details::reverse_view(_in, _out, t.numel());
        Tensor out(out_vals, t.shape());
        out.set_mutability(t.is_mutable());
        return std::move(out);
    }
    for (auto begin = list->cbegin(); begin != list->cend(); ++begin) {
        details::reverse_strided_from(_in, _out, t.shape(), *begin, t.numel());
    }
    Tensor out(out_vals, t.shape());
    out.set_mutability(t.is_mutable());
    return std::move(out);
}

Tensor flip(const Tensor& t, utils::optional_list list){return flip_view(t, list).clone();}

}
}
