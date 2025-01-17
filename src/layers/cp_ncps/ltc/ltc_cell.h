#ifndef _NT_LAYERS_NCPS_LTC_CELL_H_
#define _NT_LAYERS_NCPS_LTC_CELL_H_

#include "../../../utils/type_traits.h"
#include "../../Layer.h"
#include "../../Module.h"
#include "../../layers.h"
#include "../wiring/wiring.h"

namespace nt {
namespace ncps {

class LTCCell : public Module {
    intrusive_ptr<Wiring> _wiring;
    Layer make_positive_fn, _clip;
    std::string _input_mapping, _output_mapping;
    int64_t _ode_unfolds;
    double _epsilon;
    bool _implicit_param_constraints;
    std::map<std::string, std::tuple<double, double>> _init_ranges;
    std::map<std::string, TensorGrad> _params;
    inline static std::map<std::string, std::tuple<double, double>> _make_init_ranges() {
        return {
            {"gleak", {0.001, 1.0}},
            {"vleak", {-0.2, 0.2}},
            {"cm", {0.4, 0.6}},
            {"w", {0.001, 1.0}},
            {"sigma", {3.0, 8.0}},
            {"mu", {0.3, 0.8}},
            {"sensory_w", {0.001, 1.0}},
            {"sensory_sigma", {3.0, 8.0}},
            {"sensory_mu", {0.3, 0.8}},
        };
    }

    template <typename T>
    inline static intrusive_ptr<Wiring> _make_wiring(T val) {
        if constexpr (std::is_same_v < T, intrusive_ptr<Wiring>>) {
            return val;
        } else if constexpr (utils::is_intrusive_ptr_v<T>) {
            return intrusive_ptr<Wiring>(val);
        } else if constexpr (std::is_same_v<T, Wiring>) {
            return make_intrusive<Wiring>(val);
        } else if constexpr (std::is_base_of_v<Wiring, T>) {
            return intrusive_ptr<Wiring>(make_intrusive<T>(val));
        } else {
            static_assert(
                std::is_base_of_v<Wiring, T>,
                "Expected to get a val that is an intrusive_ptr or that is"
                "a base of wiring");
            return intrusive_ptr<Wiring>(make_intrusive<T>(val));
        }
    }

  public:
    Tensor sparsity_mask,
        sensory_sparsity_mask; // always don't track the gradient
    template <typename T>
    LTCCell(T wiring, std::optional<int64_t> in_features = std::nullopt,
            std::string input_mapping = "affine",
            std::string output_mapping = "affine", int64_t ode_unfolds = 6,
            double epsilon = 1e-8, bool implicit_param_constraints = false)
        : _wiring(LTCCell::_make_wiring(wiring)),
          make_positive_fn(implicit_param_constraints ? Layer(layers::Softplus())
                                                      : Layer(layers::Identity())),
          _clip(layers::ReLU()), _ode_unfolds(ode_unfolds), _epsilon(epsilon),
          _implicit_param_constraints(implicit_param_constraints),
          _init_ranges(LTCCell::_make_init_ranges()),
          sparsity_mask(std::abs(this->_wiring->get_adjacency_matrix())),
          sensory_sparsity_mask(
              std::abs(this->_wiring->get_sensory_adjacency_matrix()))
    {
        this->_allocate_parameters();
    }

    inline int64_t state_size(){
        return this->_wiring->get_units();
    }

    inline int64_t sensory_size(){
        return this->_wiring->get_input_dim();
    }

    inline int64_t motor_size(){
        return this->_wiring->get_output_dim();
    }

    inline int64_t output_size(){
        return this->_wiring->get_output_dim();
    }

    inline int64_t synapse_count(){
        return this->_wiring->synapse_count();
    }

    inline int64_t sensory_synapse_count(){
        return this->_wiring->sensory_synapse_count();
    }
    
    LTCCell(LTCCell&&);
    LTCCell(const LTCCell&);
    LTCCell& operator=(LTCCell&&);
    LTCCell& operator=(const LTCCell&);

    void add_weight(std::string name, Tensor init_value,
                    bool requires_grad = true);

    Tensor _get_init_value(SizeRef shape, std::string param_name);

    void _allocate_parameters();

    TensorGrad _sigmoid(TensorGrad v_pre, const TensorGrad &mu,
                        const TensorGrad &sigma);

    TensorGrad _ode_solver(TensorGrad inputs, const TensorGrad &state,
                           const Tensor &elapsed_time);

    void _map_inputs(TensorGrad &inputs);

    TensorGrad _map_outputs(const TensorGrad &state);

    void apply_weight_constraints();

    TensorGrad forward(TensorGrad inputs, TensorGrad &states,
                       Tensor elapsed_time);

};

} // namespace ncps
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::ncps::LTCCell, nt__ncps__LTCCell,
                               sparsity_mask, sensory_sparsity_mask)

#endif //_NT_LAYERS_NCPS_LTC_CELL_H_
