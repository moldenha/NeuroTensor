#ifndef __NT_TDA_NN_PH_DISTANCE_H__
#define __NT_TDA_NN_PH_DISTANCE_H__
#include "../../nn/TensorGrad.h"
namespace nt{
namespace tda{

//creates a learnable distance matrix from a point cloud
TensorGrad cloudToDist(const TensorGrad& _cloud, Scalar threshold, Scalar grad_lr=0.7, int64_t dims=-1);
//creates a learnable distance matrix from a tensor of points with shape {N,D}
TensorGrad coordsToDist(const TensorGrad&);

Tensor cloudToDist(const Tensor& _cloud, Scalar threshold, int64_t dims = -1);
Tensor coordsToDist(const Tensor&);


}
}

#endif
