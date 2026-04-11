// Very much non-optimized
// currently adapted from: https://github.com/krrish94/chamferdist/blob/master/chamferdist/knn_cpu.cpp
// There is currently no exposing .h file because it is not integrated into the framework yet
// Will probably use a KD-Tree for the forward pass on future versions for higher optimization
// Also needs to be adapted to return all K values, makes running a Hodge Laplacian/Persistent Homology much faster
//  and will reduce computational times/memory usage

#include "../../../dtype/ArrayVoid.h"
#include "../../../dtype/ArrayVoid.h"
#include "../../../Tensor.h"
#include "../../../functional/functional.h"
#include "../../../functional/TensorAccessor.h"
#include "../../../utils/always_inline_macro.h"
#include <queue>
#include <tuple>

namespace nt::tda::nn{

namespace details{
template<typename T>
NT_ALWAYS_INLINE T* get_contiguous_tensor_dataptr__(const Tensor& t){return reinterpret_cast<T*>(t.data_ptr());}
}

// will adapt for any-K
// This was the optimization used in the old_tda for the KD-Tree
// Will also be adapted for the vector-rips complex
std::tuple<Tensor, Tensor> KNearestNeighborIdxCpu(
        const Tensor& p1,
        const Tensor& p2,
        const Tensor& lengths1,
        const Tensor& lengths2,
        int64_t K){
    const int64_t& N = p1.shape()[0];
    const int64_t& P1 = p1.shape()[1];
    const int64_t& D = p1.shape()[2];
    
    using paccessor_type = TensorAccessor_iter<const float, 3, const float*>;
    using laccessor_type = get_contiguous_tensor_dataptr__<const int64_t>;

    Tensor idxs = functional::zeros({N, P1, K}, DType::Int64);
    Tensor dists = functional::zeros({N, P1, K}, p1.dtype());

    utils::throw_exception(p1.is_contiguous() && p2.is_contiguous() && lengths1.is_contiguous() && lengths2.is_contiguous(),
            "Error: only contiguous tensors accepted for KNearestNeighbor (CPU)");

    utils::throw_exception(p1.dims() == 3 && p2.dims() == 3 && lengths1.dims() == 1 && lengths2.dims() == 1,
            "Error: only 3 dimensions tensors accepted for KNearestNeighbor p vals, and 1 for lengths");
    
    utils::throw_exception(p1.dtype() == DType::Float32 && p2.dtype() == DType::Float32 && lengths1.dtype() == DType::int64 && lengths2.dtype() == DType::int64,
            "Error: lengths must have dtypes of int64 and p1 and p2 must have dtypes of floats");


    auto p1_a = paccessor_type(p1);
    auto p2_a = paccessor_type(p2);
    auto lengths1_a = laccessor_type(lenghts1);
    auto lengths2_a = laccessor_type(lenghts2);
    auto idxs_a = TensorAccessor_iter<int64_t, 3, int64_t*>(idxs);
    auto dists_a = TensorAccessor_iter<float, 3, float*>(dists);

    for (int64_t n = 0; n < N; ++n) {
        const int64_t& length1 = lengths1_a[n];
        const int64_t& length2 = lengths2_a[n];
        for (int64_t i1 = 0; i1 < length1; ++i1) {
            // Use a priority queue to store (distance, index) pairs.
            std::priority_queue<std::pair<float, int64_t>> q;
            for (int64_t i2 = 0; i2 < length2; ++i2) {
                float dist = 0;
                for (int64_t d = 0; d < D; ++d) {
                    float diff = p1_a[n][i1][d] - p2_a[n][i2][d];
                    dist += diff * diff;
                }
                // int size = static_cast<int>(q.size());
                if (q.size() < K || dist < q.top().first) {
                    q.emplace(dist, i2);
                    if (q.size() >= K) {
                        q.pop();
                    }
                }
            }
            while (!q.empty()) {
                const auto& t = q.top();
                q.pop();
                // const int k = q.size();
                dists_a[n][i1][q.size()] = t.first;
                idxs_a[n][i1][q.size()] = t.second;
            }
        }
    } 
    return std::make_tuple(idxs, dists);
    
}
std::tuple<Tensor, Tensor> KNearestNeighborBackwardCpu(
        const Tensor& p1,
        const Tensor& p2,
        const Tensor& lengths1,
        const Tensor& lengths2,
        const Tensor& idxs,
        const Tensor& grad_dists){
    const int64_t& N = p1.shape()[0];
    const int64_t& P1 = p1.shape()[1];
    const int64_t& D = p1.shape()[2];
    const int64_t& P2 = p2.shape()[1];
    const int64_t& K = idxs.shape()[2];

    using paccessor_type = TensorAccessor_iter<const float, 3, const float*>;
    using laccessor_type = get_contiguous_tensor_dataptr__<const int64_t>;

    Tensor grad_p1 = functional::zeros({N, P1, D}, p1.dtype());
    Tensor grad_p2 = functional::zeros({N, P1, D}, p2.dtype());
  
    utils::throw_exception(p1.is_contiguous() && p2.is_contiguous() && lengths1.is_contiguous() && lengths2.is_contiguous() && idxs.is_contiguous() && grad_dists.is_contiguous(),
            "Error: only contiguous tensors accepted for KNearestNeighborBackward (CPU)");

    utils::throw_exception(p1.dims() == 3 && p2.dims() == 3 && lengths1.dims() == 1 && lengths2.dims() == 1 && idxs.dims() == 3 && grad_dists.dims() == 3,
            "Error: only 3 dimensions tensors accepted for KNearestNeighborBackward p vals, and 1 for lengths");
    
    utils::throw_exception(p1.dtype() == DType::Float32 && p2.dtype() == DType::Float32 
                            && lengths1.dtype() == DType::int64 && lengths2.dtype() == DType::int64
                            && idxs.dtype() == DType::int64 && grad_dists.dtype() == DType::Float32,
            "Error: lengths must have dtypes of int64 and p1 and p2 must have dtypes of floats");
    
    auto p1_a = paccessor_type(p1);
    auto p2_a = paccessor_type(p2);
    auto lengths1_a = laccessor_type(lengths1);
    auto lengths2_a = laccessor_type(lengths2);
    auto idxs_a = TensorAccessor_iter<const int64_t, 3, const int64_t*>(idxs);
    auto grad_dists_a = paccessor_type(grad_dists);
    auto grad_p1_a = TensorAccessor_iter<float, 3, float*>(grad_p1);
    auto grad_p2_a = TensorAccessor_iter<float, 3, float*>(grad_p2);

    for (int64_t n = 0; n < N; ++n) {
        const int64_t& length1 = lengths1_a[n];
        const int64_t& length2 = (lengths2_a[n] < K ? lengths2_a[n] : K);
        for (int64_t i1 = 0; i1 < length1; ++i1) {
            for (int64_t k = 0; k < length2; ++k) {
                const int64_t& i2 = idxs_a[n][i1][k];
                for (int64_t d = 0; d < D; ++d) {
                    const float diff =
                         2.0f * grad_dists_a[n][i1][k] * (p1_a[n][i1][d] - p2_a[n][i2][d]);
                    grad_p1_a[n][i1][d] += diff;
                    grad_p2_a[n][i2][d] += -1.0f * diff;
                }
            }
        }
    }
     return std::make_tuple(grad_p1, grad_p2);

}


}




