#ifndef _NT_LAYERS_NCPS_CFC_H_
#define _NT_LAYERS_NCPS_CFC_H_

#include "../../../utils/type_traits.h"
#include "../../Layer.h"
#include "../../Module.h"
#include "../../layers.h"
#include "../lstm_cell.h"
#include "../wiring/wiring.h"
#include "cfc_cell.h"
#include "wired_cfc_cell.h"

namespace nt {
namespace ncps {

class CfC : public Module{
    intrusive_ptr<Wiring> _wiring;
    int64_t units;
    template <typename T,
              std::enable_if_t<std::is_base_of_v<Wiring, T>, bool> = true>
    inline static intrusive_ptr<Wiring> make_wiring(T val) {
        return make_intrusive<Wiring>(std::move(val));
    }
    inline static intrusive_ptr<Wiring> make_wiring(intrusive_ptr<Wiring> val) {
        return val;
    }

    inline static intrusive_ptr<Wiring> make_wiring(Scalar val) {
        return intrusive_ptr<Wiring>(nullptr);
    }
    template <typename T,
              std::enable_if_t<utils::is_scalar_value_v<T> && !std::is_same_v<T, Scalar>, bool> = true>
    inline static intrusive_ptr<Wiring> make_wiring(T val) {
        return intrusive_ptr<Wiring>(nullptr);
    }
    
    inline static int64_t make_units(Scalar val){
        return val.to<int64_t>();
    }

    template<typename T>
    inline static int64_t make_units(T val){
        if constexpr (std::is_base_of_v<Wiring, T> || std::is_same_v<Wiring, T> 
                || std::is_same_v<intrusive_ptr<Wiring>, T>){
            return -1;
        }else if (std::is_same_v<T, Scalar>){
            return static_cast<Scalar>(val).to<int64_t>();
        }else{
            return static_cast<int64_t>(val);
        }
    }


  public:
    int64_t input_size, proj_size;
    bool batch_first, return_sequences, wired_mode;
    Layer rnn_cell, lstm, fc;
    int64_t state_size, output_size;
    bool use_mixed;
    template <typename T, std::enable_if_t<std::is_base_of_v<Wiring, T> ||
                                               utils::is_scalar_value_v<T> ||
                                               std::is_same_v<T, Scalar> ||
                                                std::is_same_v<T, intrusive_ptr<Wiring> >,
                                           bool> = true>
    CfC(int64_t input_size, T units, int64_t proj_size = -1, bool return_sequences = true,
        bool batch_first = true, bool mixed_memory = false, std::string mode = "default",
        std::string activation = "lecun_tanh", int64_t backbone_units = 128,
        int64_t backbone_layers = 1, double backbone_dropout = 0.0)
    :input_size(input_size),
    _wiring(CfC::make_wiring(units)),
    units(CfC::make_units(units)),
    proj_size(proj_size),
    batch_first(batch_first),
    return_sequences(return_sequences),
    wired_mode(bool(this->_wiring)),
    rnn_cell(this->wired_mode ? Layer(WiredCfCCell(input_size, this->_wiring, mode))
                :Layer(CfCCell(input_size, this->units, mode, activation, backbone_units, backbone_layers, backbone_dropout))),
    state_size(this->wired_mode ? this->_wiring->get_units() : this->units),
    output_size(this->wired_mode ? this->_wiring->get_output_dim() : this->units),
    use_mixed(mixed_memory),
    lstm(mixed_memory ? Layer(LSTMCell(this->input_size, this->wired_mode ? this->_wiring->get_units() : this->units))
                            : Layer(layers::Identity())),
    fc(proj_size > 0 ? Layer(layers::Linear(this->wired_mode ? this->_wiring->get_units() : this->units, this->proj_size))
                    : Layer(layers::Identity()))
    {;}

    TensorGrad forward(TensorGrad input, const TensorGrad& hx, Tensor timespan, TensorGrad &hx_out); 
    
};

} // namespace ncps
} // namespace nt


_NT_REGISTER_LAYER_NAMESPACED_(nt::ncps::CfC, nt__ncps__CfC, 
                                input_size, proj_size,
                                batch_first, return_sequences, wired_mode,
                                rnn_cell, lstm, fc,
                                state_size, output_size,
                                use_mixed)

#endif //!_NT_LAYERS_NCPS_CFC_H_
