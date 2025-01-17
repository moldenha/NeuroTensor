#include "cfc_cell.h"
#include <stdexcept>

namespace nt{
namespace ncps{



Layer CfCCell::make_backbone_activation(std::string backbone_activation) {
    if (backbone_activation == "silu")
        return Layer(layers::SiLU());
    else if (backbone_activation == "relu")
        return Layer(layers::ReLU());
    else if (backbone_activation == "tanh")
        return Layer(layers::Tanh());
    else if (backbone_activation == "gelu")
        return Layer(layers::GELU());
    else if (backbone_activation == "lecun_tanh")
        return Layer(LeCun());
    else {
        throw std::invalid_argument("unknown activation ");
        return Layer(layers::Identity());
    }

}

void CfCCell::init_weights() {
    functional::xavier_uniform_(ff1_weight.tensor);
    functional::xavier_uniform_(ff1_bias.tensor);
    functional::xavier_uniform_(ff2_weight.tensor);
    functional::xavier_uniform_(ff2_bias.tensor);
    if (!A.is_null()) {
        functional::xavier_uniform_(this->A.tensor);
    }
    if (!w_tau.is_null()) {
        functional::xavier_uniform_(w_tau.tensor);
    }

}


CfCCell::CfCCell(int64_t input_size, int64_t hidden_size,
        std::string mode,
        std::string backbone_activation,
        int64_t backbone_units , int64_t backbone_layers,
        double backbone_dropout,
        Tensor sparsity_mask)
    : input_size(input_size), hidden_size(hidden_size),
      mode(std::move(mode)),
      sparsity_mask(sparsity_mask.is_null()
                        ? sparsity_mask
                        : sparsity_mask.to(DType::Float32)),
      // backbone_activation(
          // CfCCell::make_backbone_activation(backbone_activation)),
      backbone_layers(backbone_layers),
      time_a(this->mode == "pure"
                 ? Layer(layers::Linear((backbone_layers == 0)
                                            ? this->hidden_size + input_size
                                            : backbone_units,
                                        hidden_size))
                 : Layer(layers::Identity())),
      time_b(this->mode == "pure"
                 ? Layer(layers::Linear((backbone_layers == 0)
                                            ? this->hidden_size + input_size
                                            : backbone_units,
                                        hidden_size))
                 : Layer(layers::Identity())),

      ff1_weight(functional::randn({(backbone_layers == 0)
                                        ? this->hidden_size + input_size
                                        : backbone_units,
                                    hidden_size},
                                   DType::Float32)),

      ff1_bias(functional::randn({1, hidden_size}, DType::Float32)),
      ff2_weight(functional::randn({(backbone_layers == 0)
                                        ? this->hidden_size + input_size
                                        : backbone_units,
                                    hidden_size},
                                   DType::Float32)),
      ff2_bias(functional::randn({1, hidden_size}, DType::Float32)),
      w_tau(this->mode == "pure" ? functional::zeros({1, hidden_size})
                                 : Tensor::Null()),
      A(this->mode == "pure" ? functional::ones({1, hidden_size})
                             : Tensor::Null()),
      backbone(nt::layers::Identity())

{
    utils::THROW_EXCEPTION(
        this->mode == "default" || this->mode == "pure" ||
            this->mode == "no_gate",
        "unknown mode for CfCCell, expected [default, pure, no_gate, got $",
        this->mode);
    
    if (backbone_layers > 0) {
        std::vector<Layer> layer_list = {Layer(layers::Linear(
                                             input_size + hidden_size,
                                             backbone_units)),
                                         CfCCell::make_backbone_activation(backbone_activation)};

        for (int64_t i = 1; i < backbone_layers; ++i) {
            layer_list.emplace_back(
                layers::Linear(backbone_units, backbone_units)
            );
            layer_list.emplace_back(CfCCell::make_backbone_activation(backbone_activation));
            if (backbone_dropout > 0.0) {
                layer_list.emplace_back(
                    layers::Dropout(backbone_dropout)
                );
            }
        }
        this->backbone = Layer(layers::Sequential(std::move(layer_list)));

    }
    this->init_weights();
}

TensorGrad CfCCell::forward(TensorGrad input, TensorGrad hx, const Tensor &ts, TensorGrad& hx_out) {

    TensorGrad x_list = functional::list(input, hx);
    TensorGrad x = functional::cat(functional::list(input, hx), 1);
    if (this->backbone_layers > 0) {
        x = this->backbone(x);
    }

    TensorGrad ff1 = ((!this->sparsity_mask.is_null())
                          ? functional::matmult(x, this->ff1_weight *
                                                       this->sparsity_mask)
                          : functional::matmult(x, this->ff1_weight)) +
                     this->ff1_bias;

    if (this->mode == "pure") {
        TensorGrad new_hidden =
            (-this->A * (-ts * (functional::abs(w_tau) +
                                    functional::abs(ff1))).exp() * ff1 + this->A);
        hx_out = new_hidden;
        return new_hidden;
    }
    // Cfc
    TensorGrad ff2 = ((!this->sparsity_mask.is_null())
                          ? functional::matmult(x, this->ff2_weight *
                                                       this->sparsity_mask)
                          : functional::matmult(x, this->ff2_weight)) +
                     this->ff2_bias;
    ff1 = functional::tanh(ff1);
    ff2 = functional::tanh(ff2);
    TensorGrad t_a = this->time_a(ff1);
    TensorGrad t_b = this->time_a(ff2);
    TensorGrad t_interp = functional::sigmoid(t_a * ts + t_b);
    TensorGrad new_hidden =
        (this->mode == "no_gate" ? ff1 + t_interp * ff2
                                 : ff1 * (1.0 - t_interp) + t_interp * ff2);
    hx_out = new_hidden;
    return new_hidden;
}

}
}




