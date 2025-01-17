#ifndef _NT_LAYERS_NCPS_CFC_CELL_H_
#define _NT_LAYERS_NCPS_CFC_CELL_H_

#include "../../../utils/type_traits.h"
#include "../../Layer.h"
#include "../../Module.h"
#include "../../Sequential.h"
#include "../../layers.h"
#include "../wiring/wiring.h"
#include <stdexcept>

namespace nt {
namespace ncps {

class LeCun : public Module {
  public:
    LeCun() {}
    inline TensorGrad forward(TensorGrad x) {
        x = 1.7159 * functional::tanh(0.666 * x);
        return std::move(x);
    }

}; // LeCun
// a closed form continuous time cell

class CfCCell : public Module {
    static Layer make_backbone_activation(std::string backbone_activation); 

    void init_weights();

  public:
    int64_t input_size, hidden_size, backbone_layers;
    std::string mode;
    Tensor sparsity_mask;
    Layer backbone, time_a, time_b;
    TensorGrad ff1_weight, ff1_bias, ff2_weight, ff2_bias, w_tau, A;
    CfCCell(int64_t input_size, int64_t hidden_size,
            std::string mode = "default",
            std::string backbone_activation = "lecu_tanh",
            int64_t backbone_units = 128, int64_t backbone_layers = 1,
            double backbone_dropout = 0.0,
            Tensor sparsity_mask = Tensor::Null());

    TensorGrad forward(TensorGrad input, TensorGrad hx, const Tensor &ts, TensorGrad& hx_out); 
};

} // namespace ncps
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::ncps::CfCCell, nt__ncps__CfCCell, input_size,
                               hidden_size, backbone_layers, mode,
                               sparsity_mask, backbone,
                               time_a, time_b, ff1_weight, ff1_bias, ff2_weight,
                               ff2_bias, w_tau, A)
_NT_REGISTER_LAYER_NAMESPACED_(nt::ncps::LeCun, nt__ncps__LeCun)

#endif //_NT_LAYERS_NCPS_CFC_CELL_H_
