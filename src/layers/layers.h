#ifndef _NT_LAYERS_H_
#define _NT_LAYERS_H_

#include "../Tensor.h"

#include "Module.h"
#include "TensorGrad.h"
#include "functional.h"
#include "layer_reflect/layer_registry.hpp"
#include "layer_reflect/reflect_macros.h"
#include <cstdint>
#include <functional>
#include <sys/_types/_int64_t.h>
#include <tuple>
#include <type_traits>
#include <variant>

namespace nt {
namespace layers {
// the linear layer:

class Linear : public Module {
public:
  bool use_bias;
  TensorGrad Weight, Bias;
  Linear(int64_t in_dims, int64_t out_dims, bool use_bias = true)
      : Weight(functional::randn({in_dims, out_dims})),
        Bias(use_bias ? functional::randn({out_dims}) : Tensor::Null()),
        use_bias(use_bias) {}
  inline TensorGrad forward(TensorGrad x) {
    nt::TensorGrad out = functional::matmult(x, Weight);
    if (use_bias) {
      out += Bias;
    }
    return std::move(out);
  }
};

class Identity : public Module {
public:
  Identity() = default;
};

class Conv1D : public Module{
public:
    bool use_bias;
    int64_t groups, in_channels, out_channels;
    int64_t stride, padding, dilation;
    TensorGrad Weight, Bias;
    Conv1D(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
           int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1,
           int64_t groups = 1, bool use_bias = true)
        :use_bias(use_bias), groups(groups), in_channels(in_channels),
        out_channels(out_channels), stride(stride), padding(padding),
        dilation(dilation),
        Weight(functional::randn({out_channels, in_channels / groups, kernel_size})),
        Bias(use_bias ? functional::randn({out_channels, 1})
                : Tensor::Null())
        {
    utils::THROW_EXCEPTION(out_channels % groups == 0,
                           "Expected in channels to be divisible by groups");
    utils::THROW_EXCEPTION(in_channels % groups == 0,
                           "Expected in channels to be divisible by groups");
    }

    inline TensorGrad forward(TensorGrad x){
        utils::THROW_EXCEPTION(
            x.shape()[-2] == in_channels,
            "Expected input tensor to have channel size of $ but got $",
            in_channels, x.shape());
        TensorGrad outp = 
            functional::conv1d(x, Weight, stride, padding, dilation, groups);
        if(use_bias){return outp + Bias;}
        return std::move(outp);
    }
};

class Conv2D : public Module {
public:
  bool use_bias;
  int64_t groups, in_channels, out_channels;
  utils::my_tuple stride, padding, dilation;
  TensorGrad Weight, Bias;
  Conv2D(int64_t in_channels, int64_t out_channels, utils::my_tuple kernel_size,
         utils::my_tuple stride = 1, utils::my_tuple padding = 0,
         utils::my_tuple dilation = 1, int64_t groups = 1, bool use_bias = true)
      : use_bias(use_bias), groups(groups), in_channels(in_channels),
        out_channels(out_channels), stride(stride), padding(padding),
        dilation(dilation),
        Weight(functional::randn({out_channels, in_channels / groups,
                                  kernel_size[0], kernel_size[1]})),
        Bias(use_bias ? functional::randn({out_channels, 1, 1})
                      : Tensor::Null()) {
    utils::THROW_EXCEPTION(out_channels % groups == 0,
                           "Expected in channels to be divisible by groups");
    utils::THROW_EXCEPTION(in_channels % groups == 0,
                           "Expected in channels to be divisible by groups");
  }

  inline TensorGrad forward(TensorGrad x) {
    utils::THROW_EXCEPTION(
        x.shape()[-3] == in_channels,
        "Expected input tensor to have channel size of $ but got $",
        in_channels, x.shape());
    TensorGrad outp =
        functional::conv2d(x, Weight, stride, padding, dilation, groups);
    if (!use_bias) {
      return outp;
    }
    return outp + Bias;
  }
};

class Conv3D : public Module{
public:
    bool use_bias;
    int64_t groups, in_channels, out_channels;
    utils::my_n_tuple<3> stride, padding, dilation;
    TensorGrad Weight, Bias;
    Conv3D(int64_t in_channels, int64_t out_channels, utils::my_n_tuple<3> kernel_size,
           utils::my_n_tuple<3> stride = 1, utils::my_n_tuple<3> padding = 0, utils::my_n_tuple<3> dilation = 1,
           int64_t groups = 1, bool use_bias = true)
        :use_bias(use_bias), groups(groups), in_channels(in_channels),
        out_channels(out_channels), stride(stride), padding(padding),
        dilation(dilation),
        Weight(functional::randn({out_channels, in_channels / groups, kernel_size[0], kernel_size[1], kernel_size[3]})),
        Bias(use_bias ? functional::randn({out_channels, 1, 1, 1})
                : Tensor::Null())
        {
    utils::THROW_EXCEPTION(out_channels % groups == 0,
                           "Expected in channels to be divisible by groups");
    utils::THROW_EXCEPTION(in_channels % groups == 0,
                           "Expected in channels to be divisible by groups");
    }

    inline TensorGrad forward(TensorGrad x){
        utils::THROW_EXCEPTION(
            x.shape()[-4] == in_channels,
            "Expected input tensor to have channel size of $ but got $",
            in_channels, x.shape());
        TensorGrad outp = 
            functional::conv3d(x, Weight, stride, padding, dilation, groups);
        if(use_bias){return outp + Bias;}
        return std::move(outp);
    }
};


class ConvTranspose1D : public Module{
public:
    bool use_bias;
    int64_t groups, in_channels, out_channels;
    int64_t stride, padding, output_padding, dilation;
    TensorGrad Weight, Bias;
    ConvTranspose1D(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
           int64_t stride = 1, int64_t padding = 0, int64_t output_padding = 0, int64_t dilation = 1,
           int64_t groups = 1, bool use_bias = true)
        :use_bias(use_bias), groups(groups), in_channels(in_channels),
        out_channels(out_channels), stride(stride), padding(padding),
        output_padding(output_padding), dilation(dilation),
        Weight(functional::randn({in_channels, out_channels / groups, kernel_size})),
        Bias(use_bias ? functional::randn({out_channels, 1})
                : Tensor::Null())
        {
    utils::THROW_EXCEPTION(out_channels % groups == 0,
                           "Expected in channels to be divisible by groups");
    utils::THROW_EXCEPTION(in_channels % groups == 0,
                           "Expected in channels to be divisible by groups");
    }

    inline TensorGrad forward(TensorGrad x){
        utils::THROW_EXCEPTION(
            x.shape()[-2] == in_channels,
            "Expected input tensor to have channel size of $ but got $",
            in_channels, x.shape());
        TensorGrad outp = 
            functional::conv_transpose1d(x, Weight, stride, padding, output_padding, dilation, groups);
        if(use_bias){return outp + Bias;}
        return std::move(outp);
    }
};

class ConvTranspose2D : public Module {
public:
  bool use_bias;
  int64_t groups, in_channels, out_channels;
  utils::my_tuple stride, padding, output_padding, dilation;
  TensorGrad Weight, Bias;
  ConvTranspose2D(int64_t in_channels, int64_t out_channels, utils::my_tuple kernel_size,
         utils::my_tuple stride = 1, utils::my_tuple padding = 0, utils::my_tuple output_padding = 0,
         utils::my_tuple dilation = 1, int64_t groups = 1, bool use_bias = true)
      : use_bias(use_bias), groups(groups), in_channels(in_channels),
        out_channels(out_channels), stride(stride), padding(padding),
        output_padding(output_padding), dilation(dilation),
        Weight(functional::randn({in_channels, out_channels / groups,
                                  kernel_size[0], kernel_size[1]})),
        Bias(use_bias ? functional::randn({out_channels, 1, 1})
                      : Tensor::Null()) {
    utils::THROW_EXCEPTION(out_channels % groups == 0,
                           "Expected in channels to be divisible by groups");
    utils::THROW_EXCEPTION(in_channels % groups == 0,
                           "Expected in channels to be divisible by groups");
  }

  inline TensorGrad forward(TensorGrad x) {
    utils::THROW_EXCEPTION(
        x.shape()[-3] == in_channels,
        "Expected input tensor to have channel size of $ but got $",
        in_channels, x.shape());
    TensorGrad outp =
        functional::conv_transpose2d(x, Weight, stride, padding, output_padding, dilation, groups);
    if (!use_bias) {
      return outp;
    }
    return outp + Bias;
  }
};

class ConvTranspose3D : public Module{
public:
    bool use_bias;
    int64_t groups, in_channels, out_channels;
    utils::my_n_tuple<3> stride, padding, output_padding, dilation;
    TensorGrad Weight, Bias;
    ConvTranspose3D(int64_t in_channels, int64_t out_channels, utils::my_n_tuple<3> kernel_size,
           utils::my_n_tuple<3> stride = 1, utils::my_n_tuple<3> padding = 0, utils::my_n_tuple<3> output_padding = 0,
           utils::my_n_tuple<3> dilation = 1,
           int64_t groups = 1, bool use_bias = true)
        :use_bias(use_bias), groups(groups), in_channels(in_channels),
        out_channels(out_channels), stride(stride), padding(padding), output_padding(output_padding),
        dilation(dilation),
        Weight(functional::randn({in_channels, out_channels / groups, kernel_size[0], kernel_size[1], kernel_size[3]})),
        Bias(use_bias ? functional::randn({out_channels, 1, 1, 1})
                : Tensor::Null())
        {
    utils::THROW_EXCEPTION(out_channels % groups == 0,
                           "Expected in channels to be divisible by groups");
    utils::THROW_EXCEPTION(in_channels % groups == 0,
                           "Expected in channels to be divisible by groups");
    }

    inline TensorGrad forward(TensorGrad x){
        utils::THROW_EXCEPTION(
            x.shape()[-4] == in_channels,
            "Expected input tensor to have channel size of $ but got $",
            in_channels, x.shape());
        TensorGrad outp = 
            functional::conv_transpose3d(x, Weight, stride, padding, output_padding, dilation, groups);
        if(use_bias){return outp + Bias;}
        return std::move(outp);
    }
};

class Softplus : public Module {
public:
  Scalar beta, threshold;
  Softplus(Scalar beta = 1.0, Scalar threshold = 20.0)
      : beta(beta), threshold(threshold) {}

  inline TensorGrad forward(TensorGrad x) {
    return functional::softplus(x, beta, threshold);
  }
};

class Dropout : public Module {
public:
  double p;
  Dropout(Scalar s) : p(s.to<double>()) {
    utils::throw_exception(p >= 0 && p <= 1,
                           "Expected p for dropout to be in [0, 1] but got $",
                           p);
  }
  Dropout(double s) : p(s) {
    utils::throw_exception(p >= 0 && p <= 1,
                           "Expected p for dropout to be in [0, 1] but got $",
                           p);
  }

  inline TensorGrad forward(TensorGrad x) { return functional::dropout(x, p); }
};

class Unfold2D : public Module {
public:
  utils::my_tuple kernel_size, dilation, padding, stride;
  bool transpose_out;
  Unfold2D(utils::my_tuple kernel_size, utils::my_tuple dilation = 1,
           utils::my_tuple padding = 0, utils::my_tuple stride = 1,
           bool transpose_out = true)
      : kernel_size(kernel_size), dilation(dilation), padding(padding),
        stride(stride), transpose_out(transpose_out) {}

  inline TensorGrad forward(TensorGrad x) {
    return functional::unfold(x, kernel_size, dilation, padding, stride,
                              transpose_out);
  }
};

class Fold : public Module {
public:
  utils::my_tuple output_size, kernel_size, dilation, padding, stride;
  Fold(utils::my_tuple output_size, utils::my_tuple kernel_size,
       utils::my_tuple dilation = 1, utils::my_tuple padding = 0,
       utils::my_tuple stride = 1)
      : output_size(output_size), kernel_size(kernel_size), dilation(dilation),
        padding(padding), stride(stride) {}
  inline TensorGrad forward(TensorGrad x) {
    return functional::fold(x, output_size, kernel_size, dilation, padding,
                            stride);
  }
};

class Unfold1D : public Module {
public:
  Tensor::size_value_t kernel_size, dilation, padding, stride;
  bool transpose_out;
  Unfold1D(Tensor::size_value_t kernel_size, Tensor::size_value_t dilation = 1,
           Tensor::size_value_t padding = 0, Tensor::size_value_t stride = 1,
           bool transpose_out = true)
      : kernel_size(kernel_size), dilation(dilation), padding(padding),
        stride(stride), transpose_out(transpose_out) {}

  inline TensorGrad forward(TensorGrad x) {
    return functional::unfold1d(x, kernel_size, dilation, padding, stride,
                                transpose_out);
  }
};

class Unfold3D : public Module {
public:
  utils::my_n_tuple<3> kernel_size, dilation, padding, stride;
  bool transpose_out;
  Unfold3D(utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> dilation = 1,
           utils::my_n_tuple<3> padding = 0, utils::my_n_tuple<3> stride = 1,
           bool transpose_out = true)
      : kernel_size(kernel_size), dilation(dilation), padding(padding),
        stride(stride), transpose_out(transpose_out) {}

  inline TensorGrad forward(TensorGrad x) {
    return functional::unfold3d(x, kernel_size, dilation, padding, stride,
                                transpose_out);
  }
};

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
              bool track_running_stats = true)
      : num_features(num_features), epsilon(epsilon), momentum(momentum),
        affine(affine), track_running_stats(track_running_stats),
        running_mean(track_running_stats ? functional::zeros({num_features})
                                         : Tensor::Null()),
        running_var(track_running_stats ? functional::ones({num_features})
                                        : Tensor::Null()),
        gamma(affine ? functional::ones({num_features}) : Tensor::Null()),
        beta(affine ? functional::zeros({num_features}) : Tensor::Null()) {}

  inline TensorGrad
  forward(TensorGrad x) { // output shape is the same as the input shape
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

  inline Tensor eval(Tensor x) {
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
      Tensor g = (dim == -2) ? this->gamma.detach().view(this->num_features, 1)
                             : this->gamma.detach();
      Tensor b = (dim == -2) ? this->beta.detach().view(this->num_features, 1)
                             : this->beta.detach();
      return (x_normalized * g) + b;
    }
    return std::move(x_normalized);
  }
};


} // namespace layers
} // namespace nt

#define _NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(name, function)                      \
  namespace nt {                                                               \
  namespace layers {                                                           \
  class name : public Module {                                                 \
  public:                                                                      \
    name() { ; }                                                               \
    inline TensorGrad forward(TensorGrad x) {                                  \
      return functional::function(x);                                          \
    }                                                                          \
  };                                                                           \
  }                                                                            \
  }                                                                            \
  _NT_REGISTER_LAYER_NAMESPACED_(nt::layers::name, nt__layers__##name)

_NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(ReLU, relu)
_NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(SiLU, silu)
_NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(GELU, gelu)
_NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(Sigmoid, sigmoid)
_NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(Tanh, tanh)
_NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(Tan, tan)
_NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(Log, log)

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Linear, nt__layers__Linear, use_bias,
                               Weight, Bias)
_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::BatchNorm1D, nt__layers__BatchNorm1D,
                               num_features, epsilon, momentum, affine,
                               track_running_stats, running_mean, running_var,
                               gamma, beta)

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Identity, nt__layers__Identity)
_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Conv1D, nt__layers__Conv1D, use_bias,
                               groups, in_channels, out_channels, Weight, Bias)
_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Conv2D, nt__layers__Conv2D, use_bias,
                               groups, in_channels, out_channels, Weight, Bias)
_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Conv3D, nt__layers__Conv3D, use_bias,
                               groups, in_channels, out_channels, Weight, Bias)
_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::ConvTranspose1D, nt__layers__ConvTranspose1D, use_bias,
                               groups, in_channels, out_channels, Weight, Bias)
_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::ConvTranspose2D, nt__layers__ConvTranspose2D, use_bias,
                               groups, in_channels, out_channels, Weight, Bias)
_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::ConvTranspose3D, nt__layers__ConvTranspose3D, use_bias,
                               groups, in_channels, out_channels, Weight, Bias)
_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Softplus, nt__layers__Softplus, beta,
                               threshold)
_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Dropout, nt__layers__Dropout, p)
_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Unfold2D, nt__layers__Unfold2D,
                               kernel_size, dilation, padding, stride,
                               transpose_out)
_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Fold, nt__layers__Fold, output_size,
                               kernel_size, dilation, padding, stride)
_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Unfold1D, nt__layers__Unfold1D,
                               kernel_size, dilation, padding, stride,
                               transpose_out)
_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Unfold3D, nt__layers__Unfold3D,
                               kernel_size, dilation, padding, stride,
                               transpose_out)

#endif
