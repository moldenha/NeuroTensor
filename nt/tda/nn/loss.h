//this are loss functions specific to TDA filtrations and distance matrices
//when it comes to the gradient for the filtration or distance matrix
//the gradient of a single point is:
//∑([point[i] - point[j]] / distance[i][j]) * grad[i][j]  | i != j 
//if the gradient in negative, points move further (r increases)
//if the gradient is positive, points move close (r decreases)
//for |grad values| < 1 it is the opposite
#ifndef __NT_TDA_NN_PH_LOSS_H__
#define __NT_TDA_NN_PH_LOSS_H__
#include "../../nn/Loss.h"
namespace nt{
namespace tda{
namespace loss{

TensorGrad filtration_loss(const TensorGrad& output, const Tensor& target, Scalar epsilon = 1e-5);
}
}
}
#endif
