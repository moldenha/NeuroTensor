#include "filtration.h"
#include "../SimplexConstruct.h"
#include "../../nn/TensorGrad.hpp"

namespace nt{
namespace tda{

std::tuple<Tensor, TensorGrad> VRfiltration(const TensorGrad& dist_matrix, int64_t k, double max_radi, bool sort){
    utils::throw_exception(k >= 1, "Cannot get simplexes with less than 1 point, but got k of $", k);
    if(dist_matrix.dtype() == DType::Float32){
        utils::throw_exception(dist_matrix.dims() == 2, 
                               "Expected dist_matrix to be a square matrix but got shape $", dist_matrix.shape());
        utils::throw_exception(dist_matrix.shape()[0] == dist_matrix.shape()[1],
                               "Expected dist_matrix to be a square matrix but got shape $", dist_matrix.shape());
        if(k == 1){
            Tensor simplex_complex = functional::arange(dist_matrix.shape()[1], DType::int64).view(-1, 1);
            Tensor radii = functional::zeros({dist_matrix.shape()[1]}, DType::Float32);
            return {simplex_complex, TensorGrad(radii, true)};
        }
        std::pair<Tensor, Tensor> complex_gr =  find_all_simplicies(k, dist_matrix.shape()[0], dist_matrix.detach(), max_radi, sort);
        auto [simplex_complex, radii] = get<2>(complex_gr.first.contiguous());
        if(!dist_matrix.track_grad()){
            return {simplex_complex, TensorGrad(radii, false)};
        }
        Tensor grad_indices = complex_gr.second.contiguous();
        TensorGrad out_grad = TensorGrad::make_view_grad(radii, dist_matrix,
                                [grad_indices](Tensor& grad){
                                    return grad.flatten(0, -1)[grad_indices];
                                });
        return std::make_tuple(simplex_complex, out_grad);
        
    }
    utils::throw_exception(dist_matrix.dtype() == DType::TensorObj,
                           "filtration can only happen from a distance matrix of float32 or from a tensor of distance matrices");

    //will make this happen over multiple threads in future
    Tensor out_complex = Tensor::makeNullTensorArray(dist_matrix.numel());
    Tensor* oc_access = reinterpret_cast<Tensor*>(out_complex.data_ptr());
    TensorGrad out_grad = TensorGrad::makeNullTensorArray(dist_matrix.numel());
    for(int64_t i = 0; i < dist_matrix.numel(); ++i){
        auto [complex, grad] = VRfiltration(dist_matrix[i], k, max_radi, sort);
        oc_access[i] = complex;
        out_grad[i] = grad;
    }
    return std::make_tuple(out_complex, out_grad);
}


std::tuple<Tensor, TensorGrad, TensorGrad> VRfiltration(const TensorGrad& dist_matrix1, const TensorGrad& dist_matrix2, int64_t k, double max_radi, bool sort){
    utils::throw_exception(k >= 1, "Cannot get simplexes with less than 1 point, but got k of $", k);
    utils::throw_exception(dist_matrix2.dtype() == dist_matrix1.dtype(),
                           "Expected distance matrices to have the same dtype ($) != ($)", dist_matrix1.dtype() , dist_matrix2.dtype());
    utils::throw_exception(dist_matrix1.shape() == dist_matrix2.shape(), 
                           "Expected distance matrices 1 ($) and 2 ($) to have the same shape", 
                           dist_matrix1.shape(), dist_matrix2.shape()); 
    if(dist_matrix1.dtype() == DType::Float32){
        utils::throw_exception(dist_matrix1.dims() == 2, 
                               "Expected dist_matrix to be a square matrix but got shape $", dist_matrix1.shape());
        utils::throw_exception(dist_matrix1.shape()[0] == dist_matrix1.shape()[1],
                               "Expected dist_matrix to be a square matrix but got shape $", dist_matrix1.shape());
        if(k == 1){
            Tensor simplex_complex = functional::arange(dist_matrix1.shape()[1], DType::int64).view(-1, 1);
            Tensor radii1 = functional::ones({dist_matrix1.shape()[1]}, DType::Float32);
            return {simplex_complex, TensorGrad(radii1.clone(), true), TensorGrad(radii1, true)};
        }
        std::pair<Tensor, Tensor> complex_gr =  find_all_simplicies(k, dist_matrix1.shape()[0], dist_matrix1.detach(), max_radi, sort);
        auto [simplex_complex, radii1] = get<2>(complex_gr.first.contiguous());
        if(!dist_matrix1.track_grad()){
            Tensor radii2 = dist_matrix2.detach().flatten(0, -1)[complex_gr.second].contiguous();
            return {simplex_complex, TensorGrad(radii1, false), TensorGrad(radii2, false)};
        }
        TensorGrad radii2 = dist_matrix2.flatten(0, -1)[complex_gr.second].contiguous();
        Tensor grad_indices = complex_gr.second.contiguous();
        TensorGrad out_grad = TensorGrad::make_view_grad(radii1, dist_matrix1,
                                [grad_indices](Tensor& grad){
                                    return grad.flatten(0, -1)[grad_indices];
                                });
        return std::make_tuple(simplex_complex, out_grad, radii2);
        
    }

    utils::throw_exception(dist_matrix1.dtype() == DType::TensorObj,
                           "filtration can only happen from a distance matrix of float32 or from a tensor of distance matrices");

    //will make this happen over multiple threads in future
    Tensor out_complex = Tensor::makeNullTensorArray(dist_matrix1.numel());
    Tensor* oc_access = reinterpret_cast<Tensor*>(out_complex.data_ptr());
    TensorGrad out_grad1 = TensorGrad::makeNullTensorArray(dist_matrix1.numel());
    TensorGrad out_grad2 = TensorGrad::makeNullTensorArray(dist_matrix2.numel());
    for(int64_t i = 0; i < dist_matrix1.numel(); ++i){
        auto [complex, grad1, grad2] = VRfiltration(dist_matrix1[i], dist_matrix2[i], k, max_radi, sort);
        oc_access[i] = complex;
        out_grad1[i] = grad1;
        out_grad2[i] = grad2;
    }
    return std::make_tuple(out_complex, out_grad1, out_grad2);


}

}
}
