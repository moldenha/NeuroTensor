#include "FractionalMaxPool3D.h"
#include "../functional.h"
#include "../../functional/functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"
namespace nt{
namespace layers{

FractionalMaxPool3D::FractionalMaxPool3D(utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> output_size, 
                                         std::variant<double, std::tuple<double, double, double>> output_ratio, bool return_indices)
        : kernel_size(kernel_size), output_size(output_size), output_ratio(output_ratio), return_indices(return_indices)

{}

TensorGrad FractionalMaxPool3D::forward(TensorGrad x) {
    return functional::fractional_max_pool3d(x, this->kernel_size, this->output_size, this->output_ratio, this->return_indices);
}

}
}


_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::FractionalMaxPool3D, nt__layers__FractionalMaxPool3D, kernel_size,
                               output_size, output_ratio, return_indices)

