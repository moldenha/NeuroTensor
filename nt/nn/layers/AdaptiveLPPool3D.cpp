#include "AdaptiveLPPool3D.h"
#include "../functional.h"
#include "../../functional/functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"
namespace nt{
namespace layers{

AdaptiveLPPool3D::AdaptiveLPPool3D(Scalar power, utils::my_n_tuple<3> output_size)
        : power(power), output_size(output_size)
{}

TensorGrad AdaptiveLPPool3D::forward(TensorGrad x) {
    return functional::adaptive_lp_pool3d(x,this->output_size, this->power);
}

}
}


_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::AdaptiveLPPool3D, nt__layers__AdaptiveLPPool3D, power, output_size)

