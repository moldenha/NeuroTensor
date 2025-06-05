#ifndef _NT_NN_LAYER_LOSS_H_
#define _NT_NN_LAYER_LOSS_H_

#include "TensorGrad.h"
#include "Layer.h"
#include "layers.h"
#include "ScalarGrad.h"

namespace nt{
namespace loss{


ScalarGrad raw_error(const TensorGrad&, const Tensor&); //target - output
ScalarGrad MSE(const TensorGrad&, const Tensor&); //std::pow(output.tensor - target, 2) / target.numel();

}} //nt::loss::


#endif //_NT_LAYER_LOSS_H_
