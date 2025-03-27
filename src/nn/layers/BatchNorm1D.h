#ifndef _NT_LAYERS_BATCH_NORM_1D_H_
#define _NT_LAYERS_BATCH_NORM_1D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"

namespace nt {
namespace layers {

// Input: (N,C) or (N,C,L) N = batch_size, C = channels (num_features), L =
// sequence lenght Output: same as input
class BatchNorm1D : public Module {
  public:
    int64_t num_features;
    Scalar epsilon, momentum; // ensures stable division, momentum for running
                              // mean and variance
    bool affine, track_running_stats;
    Tensor running_mean, running_var; // tracking stats
    TensorGrad gamma, beta;           // learnable parameters; scale, shift
    BatchNorm1D(int64_t num_features, double epsilon = 1e-5,
                double momentum = 0.1, bool affine = true,
                bool track_running_stats = true);

    TensorGrad forward(TensorGrad x);
    Tensor eval(Tensor x);
};

} // namespace layers
} // namespace nt

#endif
