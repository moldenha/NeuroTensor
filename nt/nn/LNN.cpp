#include "Layer.h"
#include "layers.h"
#include "LNN.h"
#include "../reflection/layer_reflect/reflect_macros.h"
#include "ncps/ltc/ltc.h"
#include "ncps/cfc/cfc.h"



namespace nt{
namespace layers{

// class LNN : public Module{
//     intrusive_ptr<Wiring> _wiring;
//     Tensor construct_hidden(int64_t batch_size);
// 	public:
//         TensorGrad hidden_state;
//         Layer lnn;
// 		LNN(int64_t input_size, int64_t output_size, int64_t neurons, LNNOptions option = LNNOptions::LTC);
// 		TensorGrad forward(TensorGrad);
// };


Tensor LNN::construct_hidden(int64_t batch_size){
    return functional::zeros({batch_size, this->_wiring->get_units()});
}

intrusive_ptr<ncps::Wiring> 
    LNN::build_wiring(int64_t input_size, int64_t output_size, int64_t neurons){
    intrusive_ptr<ncps::Wiring> out = make_intrusive<ncps::AutoNCP>(neurons, output_size);
    out->build(input_size);
    out->set_output_dim(output_size);
    std::cout << "built wiring..."<<std::endl;
    return std::move(out);
}

LNN::LNN(int64_t input_size, int64_t output_size, int64_t neurons, LNNOptions option, bool return_sequences)
    :_wiring(LNN::build_wiring(input_size, output_size, neurons)), //units == neurons, output_size from lnn
    hidden_state(nullptr),
    lnn(option == LNNOptions::LTC ? Layer(ncps::LTC(input_size, this->_wiring, return_sequences))
                                : Layer(ncps::CfC(input_size, this->_wiring, -1, return_sequences)))
{}

TensorGrad LNN::forward(TensorGrad x){
    int64_t batch_size = x.dims() == 3 ? x.shape()[0] : 1;
    if(hidden_state.is_null() || hidden_state.shape()[0] != batch_size){
        hidden_state = TensorGrad(this->construct_hidden(batch_size));
    }
    // Timespans: (batch_size, sequence_length)
    // Example: Varying time intervals between elements in the sequence
    // timespans = torch.detach()([[1, 2, 1, 0.5, 1, 1] for _ in range(batch_size)], dtype=torch.float32)
    return this->lnn(x, hidden_state, Tensor::Null(), hidden_state);


}

}}

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::LNN, nt__layers__LNN, hidden_state, lnn)


