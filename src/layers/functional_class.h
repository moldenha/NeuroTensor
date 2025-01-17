// this is a friend class to TensorGrad that holds the functional functions
// dedicated to tensorgrad this is meant to re-create the functional.h that is
// dedicated to the Tensor class
#ifndef _NT_TENSORGRAD_FUNCTIONAL_CLASS_H_
#define _NT_TENSORGRAD_FUNCTIONAL_CLASS_H_

namespace nt {
namespace functional {
class TensorGrad_Functional_Class; // forward declaration

} // namespace functional
} // namespace nt


#include "TensorGrad.h"
namespace nt {
namespace functional {
class TensorGrad_Functional_Class {
  public:
    TensorGrad_Functional_Class() = default;
    static TensorGrad matmult(const TensorGrad &, const TensorGrad &);
    static TensorGrad matmult(const Tensor &, const TensorGrad &);
    static TensorGrad matmult(const TensorGrad &a, const Tensor &b);
    static TensorGrad unfold1d(const TensorGrad &, Tensor::size_value_t,
                               Tensor::size_value_t, Tensor::size_value_t,
                               Tensor::size_value_t, bool);
    static TensorGrad unfold(const TensorGrad &, utils::my_tuple,
                             utils::my_tuple, utils::my_tuple, utils::my_tuple,
                             bool);
    static TensorGrad unfold3d(const TensorGrad &, utils::my_n_tuple<3>,
                               utils::my_n_tuple<3>, utils::my_n_tuple<3>,
                               utils::my_n_tuple<3>, bool);
    static TensorGrad fold(const TensorGrad &, utils::my_tuple, utils::my_tuple,
                           utils::my_tuple, utils::my_tuple, utils::my_tuple);
    // image, kernel, stride, padding, dilation, groups
    static TensorGrad conv2d(const TensorGrad &, const TensorGrad &,
                                 utils::my_tuple, utils::my_tuple,
                                 utils::my_tuple, int64_t);
    static TensorGrad conv2d(const Tensor &, const TensorGrad &,
                                 utils::my_tuple, utils::my_tuple,
                                 utils::my_tuple, int64_t);
    static TensorGrad conv2d(const TensorGrad &, const Tensor &,
                                 utils::my_tuple, utils::my_tuple,
                                 utils::my_tuple, int64_t);
    static TensorGrad sigmoid(const TensorGrad &);
    static TensorGrad clamp(const TensorGrad &, std::optional<int64_t>,
                            std::optional<int64_t>);
    static TensorGrad relu(const TensorGrad &);
    static TensorGrad var(const TensorGrad &, utils::optional_list, int64_t,
                          bool);
    static TensorGrad invsqrt(const TensorGrad &);
    static TensorGrad silu(const TensorGrad &);
    static TensorGrad gelu(const TensorGrad &);
    static TensorGrad tanh(const TensorGrad &);
    static TensorGrad tan(const TensorGrad &);
    static TensorGrad cat(std::vector<TensorGrad>, int64_t);
    static TensorGrad cat(TensorGrad, int64_t);
    static TensorGrad stack(TensorGrad, int64_t);
    static TensorGrad stack(std::vector<TensorGrad>, int64_t);
    static TensorGrad chunk(TensorGrad, typename Tensor::size_value_t,
                            int64_t); // splits into that many chunks
    static TensorGrad split(TensorGrad input, typename Tensor::size_value_t split_size, int64_t);
    static TensorGrad split(TensorGrad input, std::vector<typename Tensor::size_value_t> split_sections, int64_t);
    static TensorGrad log(const TensorGrad &);

    static TensorGrad dropout(const TensorGrad &, double);
    static TensorGrad abs(const TensorGrad&);
    static TensorGrad softplus(const TensorGrad&, Scalar, Scalar);

}; // TensorGrad_Functional_Class
}
}


#endif
