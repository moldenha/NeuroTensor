// functions to make a boundary matrix differentiable in discrete steps
#ifndef __NT_TDA_NN_PH_BOUNDARIES_H__
#define __NT_TDA_NN_PH_BOUNDARIES_H__

#include "../../Tensor.h"
#include "../../nn/TensorGrad.h"

namespace nt {
namespace tda {

TensorGrad BoundaryMatrix(Tensor simplex_complex_kp1, Tensor simplex_complex_k,
                          TensorGrad radi_kp1, TensorGrad radi_k);

TensorGrad BoundaryMatrix(Tensor simplex_complex_kp1, Tensor simplex_complex_k,
                          TensorGrad radi_kp1, TensorGrad radi_k, 
                          TensorGrad radi_kp1_2, TensorGrad radi_k_2);

}
} // namespace nt

#endif
