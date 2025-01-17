#include "wired_cfc_cell.h"

namespace nt{
namespace ncps{

WiredCfCCell::WiredCfCCell(int64_t input_size, intrusive_ptr<Wiring> wiring,
             std::string mode)
    : _wiring(wiring), num_layers(0) {
    std::cout << "building..."<<std::endl;
    _wiring->build(input_size);
    std::cout << "built"<<std::endl;
    utils::THROW_EXCEPTION(_wiring->is_built(),
                           "Expected wiring to be built in WiredCfCCell");
    int64_t in_features = this->_wiring->get_input_dim();
    this->num_layers = this->_wiring->num_layers();
    this->_layers.reserve(num_layers);
    for (int64_t l = 0; l < this->num_layers; ++l) {
        Tensor hidden_units = this->_wiring->get_neurons_of_layer(l);
        Tensor input_sparsity(nullptr);
        if (l == 0) {
            input_sparsity = this->_wiring->get_sensory_adjacency_matrix()
                                 .transpose(0, 1)[hidden_units].transpose(0,1)
                                 .to(DType::Float32);
        } else {
            Tensor prev_layer_neurons =
                this->_wiring->get_neurons_of_layer(l - 1);
            input_sparsity = this->_wiring->get_sensory_adjacency_matrix()
                                 .transpose(0, 1)[hidden_units].transpose(0, 1);
            input_sparsity =
                input_sparsity[prev_layer_neurons].to(DType::Float32);
        }
        input_sparsity = functional::cat(functional::list(
            input_sparsity, functional::ones({hidden_units.numel(),
                                              hidden_units.numel()})), 0);
        Layer rcnn_cell(CfCCell(in_features, hidden_units.numel(), mode,
                                "lecun_tanh", 0, 0, 0.0,
                                std::move(input_sparsity)));
        this->_layers.push_back(std::move(rcnn_cell));
        this->register_module("layer_" + std::to_string(l),
                              this->_layers.back());
        in_features = hidden_units.numel();
    }
}

WiredCfCCell::WiredCfCCell(WiredCfCCell&& wi)
    :_wiring(std::move(wi._wiring)),
    _layers(std::move(wi._layers)),
    num_layers(wi.num_layers)
{}

WiredCfCCell::WiredCfCCell(const WiredCfCCell& wi)
    :_wiring(wi._wiring),
    _layers(wi._layers),
    num_layers(wi.num_layers)
{}

WiredCfCCell& WiredCfCCell::operator=(WiredCfCCell&& wi){
    _wiring = std::move(wi._wiring);
    _layers = std::move(wi._layers);
    num_layers = wi.num_layers;
    return *this;
}

WiredCfCCell& WiredCfCCell::operator=(const WiredCfCCell& wi){
    _wiring = wi._wiring;
    _layers = wi._layers;
    num_layers = wi.num_layers;
    return *this;
}


std::vector<int64_t> WiredCfCCell::layer_sizes() {
    std::vector<int64_t> out(this->num_layers);
    for (int64_t i = 0; i < this->num_layers; ++i) {
        out[i] = this->_wiring->get_number_neurons_of_layer(i);
    }
    return std::move(out);
}



TensorGrad WiredCfCCell::forward(TensorGrad input, TensorGrad hx, Tensor timespans, TensorGrad& hx_out) {
    TensorGrad h_state = functional::split(hx, this->layer_sizes(), 1);

    std::vector<TensorGrad> new_h_state(this->num_layers, TensorGrad(Tensor::Null()));
    TensorGrad inputs = input;
    for (int64_t i = 0; i < this->num_layers; ++i) {
        inputs = this->_layers[i](inputs, h_state[i], timespans, new_h_state[i]);
    }
    hx_out = functional::cat(std::move(new_h_state), 1);
    //this prints true, indicating that the actual memory in hx is modified
    //this makes doing a concatenation just an operation that would take extra time
    //std::cout << std::boolalpha << (hx.data_ptr() == h_state[0].tensor.data_ptr()) << 
    //    std::noboolalpha << std::endl;
    //the shape state that is outputted from a cfc_cell stays the same
    //therefore there is no reason to do a concatenation or anything like that
    return std::move(inputs);
}

}
}
