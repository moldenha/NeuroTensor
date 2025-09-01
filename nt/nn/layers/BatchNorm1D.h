#ifndef NT_LAYERS_BATCH_NORM_1D_H_
#define NT_LAYERS_BATCH_NORM_1D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"

namespace nt {
namespace layers {

// Input: (N,C) or (N,C,L) N = batch_size, C = channels (num_features), L =
// sequence lenght Output: same as input
class NEUROTENSOR_API BatchNorm1D : public Module {
  public:
    int64_t num_features;
    Scalar epsilon, momentum; // ensures stable division, momentum for running
                              // mean and variance
    bool affine, track_running_stats;
    Tensor running_mean, running_var; // tracking stats
    TensorGrad gamma, beta;           // learnable parameters; scale, shift
    NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(BatchNorm1D,
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(num_features, epsilon, momentum, affine, track_running_stats),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(1e-5,0.1,true,true),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(int64_t, double, double, bool, bool));
    // BatchNorm1D(int64_t num_features, double epsilon = 1e-5,
    //             double momentum = 0.1, bool affine = true,
    //             bool track_running_stats = true);

    TensorGrad forward(TensorGrad x);
    Tensor eval(Tensor x);
};

} // namespace layers
} // namespace nt

#endif
