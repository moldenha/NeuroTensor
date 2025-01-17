#ifndef _NT_LAYERS_NCPS_LTC_H_
#define _NT_LAYERS_NCPS_LTC_H_

#include "../../../utils/type_traits.h"
#include "../../Layer.h"
#include "../../Module.h"
#include "../../layers.h"
#include "../lstm_cell.h"
#include "../wiring/wiring.h"
#include "ltc_cell.h"

namespace nt {
namespace ncps {

// forward parameters:
//   Param: inputs: (a TensorGrad) of shape (L,C) in batchless mode, or (B,L,C)
//   if batch_first was set to true and (L,B,C) if batch_first is false
//  Param: hx:       (can be TensorGrad(nullptr)), Initial hidden state of the
//  RNN of shape (B,H) if mixed_memory is false and a list of tensors
//  {(B,H),(B,H)} if mixed_memory is true. If nullptr, the hidden states are
//  initialized with all zeros. Param: timespan: a tensor representing the
//  current time sequence (can be Tensor::Null()) Param: hx_out: a reference
//  TensorGrad to store the final hidden state Output: readout: final out state
//  of the RNN
class LTC : public Module {
    intrusive_ptr<Wiring> _wiring;
    int64_t input_size;
    bool batch_first, return_sequences, use_mixed;
    template <typename T,
              std::enable_if_t<std::is_base_of_v<Wiring, T>, bool> = true>
    inline intrusive_ptr<Wiring> make_wiring(T val) {
        return make_intrusive<Wiring>(std::move(val));
    }
    inline intrusive_ptr<Wiring> make_wiring(intrusive_ptr<Wiring> val) {
        return val;
    }

    inline intrusive_ptr<Wiring> make_wiring(Scalar val) {
        return make_intrusive<FullyConnected>(static_cast<int64_t>(val.to<int64_t>()));
    }
    template <typename T,
              std::enable_if_t<utils::is_scalar_value_v<T> && !std::is_same_v<T, Scalar>, bool> = true>
    inline intrusive_ptr<Wiring> make_wiring(T val) {
        return make_intrusive<FullyConnected>(static_cast<int64_t>(val));
    }

  public:
    Layer lstm, rnn_cell;
    int64_t state_size, sensory_size, motor_size, output_size, synapse_count,
        sensory_synapse_count;
    template <typename T, std::enable_if_t<std::is_base_of_v<Wiring, T> ||
                                               utils::is_scalar_value_v<T> ||
                                               std::is_same_v<T, Scalar> ||
                                               std::is_same_v<T, intrusive_ptr<Wiring> >,
                                           bool> = true>
    LTC(int64_t input_size, T units, bool return_sequences = true,
        bool batch_first = true, bool mixed_memory = false,
        std::string input_mapping = "affine",
        std::string output_mapping = "affine", int64_t ode_unfolds = 6,
        double epsilon = 1e-8, bool implicit_param_constraints = true)
        : _wiring(this->make_wiring(units)), input_size(input_size),
          batch_first(batch_first), return_sequences(return_sequences),
          use_mixed(mixed_memory),
          lstm(mixed_memory ? Layer(LSTMCell(this->input_size,
                                             this->_wiring->get_units()))
                            : Layer(layers::Identity())),
          rnn_cell(LTCCell(this->_wiring, this->input_size, input_mapping,
                           output_mapping, ode_unfolds, epsilon,
                           implicit_param_constraints)),
          state_size(_wiring->get_units()),
          sensory_size(_wiring->get_input_dim()),
          motor_size(_wiring->get_output_dim()),
          output_size(_wiring->get_output_dim()),
          synapse_count(_wiring->synapse_count()),
          sensory_synapse_count(_wiring->sensory_synapse_count()) {}

    TensorGrad forward(TensorGrad input, const TensorGrad& hx, Tensor timespan,
                       TensorGrad &hx_out); 
};

} // namespace ncps
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::ncps::LTC, nt__ncps__LTC, lstm, rnn_cell,
                               state_size, sensory_size, motor_size,
                               output_size, synapse_count,
                               sensory_synapse_count)

#endif //_NT_LAYERS_NCPS_LTC_H_
