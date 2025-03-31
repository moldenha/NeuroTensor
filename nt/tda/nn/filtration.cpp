#include "filtration.h"
#include "../SimplexConstruct.h"
#include "../../nn/TensorGrad.hpp"

namespace nt{
namespace tda{

std::tuple<Tensor, TensorGrad> VRfiltration(const TensorGrad& dist_matrix, int64_t k){
    utils::throw_exception(k >= 1, "Cannot get simplexes with less than 1 point, but got k of $", k);
    if(dist_matrix.dtype == DType::Float32){
        utils::throw_exception(dist_matrix.dims() == 2, 
                               "Expected dist_matrix to be a square matrix but got shape $", dist_matrix.shape());
        utils::throw_exception(dist_matrix.shape()[0] == dist_matrix.shape()[1],
                               "Expected dist_matrix to be a square matrix but got shape $", dist_matrix.shape());
        std::pair<Tensor, Tensor> complex_gr =  find_all_simplicies(k, dist_matrix.shape()[0], dist_matrix.tensor);
        auto [simplex_complex, radii] = get<2>(complex_gr.first.contiguous());
        if(!dist_matrix.do_track_grad){
            return {simplex_complex, TensorGrad(radii, false)};
        }
        Tensor grad_indices = complex_gr.second.contiguous();
        TensorGrad out_grad = TensorGrad::make_view_grad(radii, dist_matrix,
                                [grad_indices](Tensor& grad){
                                    return grad.flatten(0, -1)[grad_indices];
                                });
        return std::make_tuple(simplex_complex, out_grad);
        
    }
    utils::throw_exception(dist_matrix.dtype == DType::TensorObj,
                           "filtration can only happen from a distance matrix of float32 or from a tensor of distance matrices");

    //will make this happen over multiple threads in future
    Tensor out_complex = Tensor::makeNullTensorArray(dist_matrix.numel());
    Tensor* oc_access = reinterpret_cast<Tensor*>(out_complex.data_ptr());
    TensorGrad out_grad = TensorGrad::makeNullTensorArray(dist_matrix.numel());
    for(int64_t i = 0; i < dist_matrix.numel(); ++i){
        auto [complex, grad] = VRfiltration(dist_matrix[i], k);
        oc_access[i] = complex;
        out_grad[i] = grad;
    }
    return std::make_tuple(out_complex, out_grad);
}

}
}
