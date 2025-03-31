#include "BatchNorm1D.h"
#include "../../functional/functional.h"
#include "../functional.h"
#include "../layer_reflect/layer_registry.hpp"
#include "../layer_reflect/reflect_macros.h"

namespace nt {
namespace layers {

BatchNorm1D::BatchNorm1D(int64_t num_features, double epsilon, double momentum,
                         bool affine, bool track_running_stats)
    : num_features(num_features), epsilon(epsilon), momentum(momentum),
      affine(affine), track_running_stats(track_running_stats),
      running_mean(track_running_stats ? functional::zeros({num_features})
                                       : Tensor::Null()),
      running_var(track_running_stats ? functional::ones({num_features})
                                      : Tensor::Null()),
      gamma(affine ? functional::ones({num_features}) : Tensor::Null()),
      beta(affine ? functional::zeros({num_features}) : Tensor::Null()) {}

TensorGrad BatchNorm1D::forward(
    TensorGrad x) { // output shape is the same as the input shape
    utils::throw_exception(x.dims() == 2 || x.dims() == 3,
                           "Expected to have a shape of (N, C, L) or (N, C) "
                           "for BatchNorm1D but got $ shape",
                           x.shape());
    int64_t dim = (-1 * x.dims()) + 1;
    utils::throw_exception(x.shape()[dim] == this->num_features,
                           "Expected to have dim $ of shape $ be $ "
                           "(num_features) but got $ for BatchNorm1D",
                           dim, x.shape(), this->num_features, x.shape()[dim]);
    TensorGrad batch_mean = x.mean(dim, true); // the dim, and keep the dims
    TensorGrad batch_var =
        functional::var(x, dim, 1,
                        true); // variance along the dim, with a correction
                               // of 1, and keep dims
    TensorGrad x_normalized =
        (x - batch_mean) *
        functional::invsqrt(batch_var +
                            this->epsilon); // faster than 1/functional::sqrt
    if (this->track_running_stats) {
        // then the shape needs to be expanded on the bottom
        if (dim == -2) {
            Tensor run_m = this->running_mean.view(this->num_features, 1);
            Tensor run_v = this->running_var.view(this->num_features, 1);
            run_m *= ((this->momentum * batch_mean.detach()) +
                      (1.0 - this->momentum.to<double>()));
            run_v *= ((this->momentum * batch_var.detach()) +
                      (1.0 - this->momentum.to<double>()));
        } else {
            running_mean *= ((this->momentum * batch_mean.detach()) +
                             (1.0 - this->momentum.to<double>()));
            running_var *= ((this->momentum * batch_var.detach()) +
                            (1.0 - this->momentum.to<double>()));
        }
    }
    if (this->affine) {
        if (dim == -1) {
            return (x_normalized * this->gamma) +
                   this->beta; // x_normalized first ensures the shape out
                               // is the same as x
        } else {
            return (x_normalized * this->gamma.view(this->num_features, 1)) +
                   this->beta.view(this->num_features,
                                   1); // x_normalized first ensures the
                                       // shape out is the same as x
        }
    }
    return std::move(x_normalized);
}

Tensor BatchNorm1D::eval(Tensor x) {
    utils::throw_exception(x.dims() == 2 || x.dims() == 3,
                           "Expected to have a shape of (N, C, L) or (N, C) "
                           "for BatchNorm1D but got $ shape",
                           x.shape());
    int64_t dim = x.dims() == 2 ? -1 : -2;
    utils::throw_exception(x.shape()[dim] == this->num_features,
                           "Expected to have dim $ of shape $ be $ "
                           "(num_features) but got $ for BatchNorm1D",
                           dim, x.shape(), this->num_features, x.shape()[dim]);
    Tensor mean =
        this->track_running_stats ? this->running_mean : x.mean(dim, true);
    Tensor var = this->track_running_stats ? this->running_var
                                           : functional::var(x, dim, 1, true);
    if (this->track_running_stats && dim == -2) {
        mean = mean.view(this->num_features, 1);
        var = var.view(this->num_features, 1);
    }
    Tensor x_normalized = (x - mean) * functional::invsqrt(var + this->epsilon);
    if (this->affine) {
        Tensor g = (dim == -2)
                       ? this->gamma.detach().view(this->num_features, 1)
                       : this->gamma.detach();
        Tensor b = (dim == -2) ? this->beta.detach().view(this->num_features, 1)
                               : this->beta.detach();
        return (x_normalized * g) + b;
    }
    return std::move(x_normalized);
}

} // namespace layers
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::BatchNorm1D, nt__layers__BatchNorm1D,
                               num_features, epsilon, momentum, affine,
                               track_running_stats, running_mean, running_var,
                               gamma, beta)
