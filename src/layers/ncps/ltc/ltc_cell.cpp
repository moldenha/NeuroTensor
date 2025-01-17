#include "ltc_cell.h"
#include <utility> //std::exchange

namespace nt {
namespace ncps {


LTCCell::LTCCell(LTCCell&& ltc)
    :_wiring(std::move(ltc._wiring)),
    make_positive_fn(std::move(ltc.make_positive_fn)),
    _clip(std::move(ltc._clip)),
    _input_mapping(std::move(ltc._input_mapping)),
    _output_mapping(std::move(ltc._output_mapping)),
    _ode_unfolds(std::exchange(ltc._ode_unfolds, 0)),
    _epsilon(std::exchange(ltc._epsilon, 0.0)),
    _implicit_param_constraints(ltc._implicit_param_constraints),
    _init_ranges(std::move(ltc._init_ranges)),
    _params(std::move(ltc._params))
{}


LTCCell::LTCCell(const LTCCell& ltc)
    :_wiring(ltc._wiring),
    make_positive_fn(ltc.make_positive_fn),
    _clip(ltc._clip),
    _input_mapping(ltc._input_mapping),
    _output_mapping(ltc._output_mapping),
    _ode_unfolds(ltc._ode_unfolds),
    _epsilon(ltc._epsilon),
    _implicit_param_constraints(ltc._implicit_param_constraints),
    _init_ranges(ltc._init_ranges),
    _params(ltc._params)
{}


LTCCell& LTCCell::operator=(LTCCell&& ltc){
    _wiring = std::move(ltc._wiring);
    make_positive_fn = std::move(ltc.make_positive_fn);
    _clip = std::move(ltc._clip);
    _input_mapping = std::move(ltc._input_mapping);
    _output_mapping = std::move(ltc._output_mapping);
    _epsilon = std::exchange(ltc._epsilon, 0.0);
    _ode_unfolds = std::exchange(ltc._ode_unfolds, 0);
    _implicit_param_constraints = ltc._implicit_param_constraints;
    _init_ranges = std::move(ltc._init_ranges);
    _params = std::move(ltc._params);
    return *this;
}

LTCCell& LTCCell::operator=(const LTCCell& ltc){
    _wiring = ltc._wiring;
    make_positive_fn = ltc.make_positive_fn;
    _clip = ltc._clip;
    _input_mapping = ltc._input_mapping;
    _output_mapping = ltc._output_mapping;
    _epsilon = ltc._epsilon;
    _ode_unfolds = ltc._ode_unfolds, 0;
    _implicit_param_constraints = ltc._implicit_param_constraints;
    _init_ranges = ltc._init_ranges;
    _params = ltc._params;
    return *this;
}


void LTCCell::add_weight(std::string name, Tensor init_value,
                         bool requires_grad) {
    TensorGrad adding(std::move(init_value.to(DType::Float32)), requires_grad);
    auto result = this->_params.insert({name, std::move(adding)});
    utils::THROW_EXCEPTION(result.second, "Trying to add weight of name $ but already added to LTCCell", name);
    this->register_parameter(name, _params.at(name));
}

Tensor LTCCell::_get_init_value(SizeRef shape, std::string param_name) {
    auto &[minval, maxval] = this->_init_ranges[param_name];
    if (minval == maxval)
        return Tensor(shape).fill_(minval);
    return functional::rand(minval, maxval, std::move(shape));
}

void LTCCell::_allocate_parameters() {
    this->_params.clear();
    this->add_weight("gleak",
                     this->_get_init_value({this->state_size()}, "gleak"));
    this->add_weight("vleak",
                     this->_get_init_value({this->state_size()}, "vleak"));
    this->add_weight("cm", this->_get_init_value({this->state_size()}, "cm"));
    this->add_weight(
        "sigma",
        this->_get_init_value({this->state_size(), this->state_size()}, "sigma"));
    this->add_weight("mu", this->_get_init_value(
                               {this->state_size(), this->state_size()}, "mu"));
    this->add_weight(
        "w", this->_get_init_value({this->state_size(), this->state_size()}, "w"));
    this->add_weight("erev", this->_wiring->erev_initializer());
    this->add_weight("sensory_sigma", this->_get_init_value({this->sensory_size(),
                                                             this->state_size()},
                                                            "sensory_sigma"));
    this->add_weight("sensory_mu",
                     this->_get_init_value(
                         {this->sensory_size(), this->state_size()}, "sensory_mu"));
    this->add_weight("sensory_w",
                     this->_get_init_value(
                         {this->sensory_size(), this->state_size()}, "sensory_w"));
    this->add_weight("sensory_erev", this->_wiring->sensory_erev_initializer());
    if (this->_input_mapping == "affine" || this->_input_mapping == "linear") {
        this->add_weight("input_w", functional::ones({this->sensory_size()}));
        if (this->_input_mapping == "affine") {
            this->add_weight("input_b",
                             functional::zeros({this->sensory_size()}));
        }
    }
    if (this->_output_mapping == "affine" ||
        this->_output_mapping == "linear") {
        this->add_weight("output_w", functional::ones({this->motor_size()}));
        if (this->_output_mapping == "affine") {
            this->add_weight("output_b", functional::zeros({this->motor_size()}));
        }
    }
}


TensorGrad LTCCell::_sigmoid(TensorGrad v_pre, const TensorGrad &mu,
                             const TensorGrad &sigma) {
    v_pre = v_pre.unsqueeze(-1);
    TensorGrad mues = v_pre - mu;
    TensorGrad x = sigma * mues;
    return functional::sigmoid(x);
}

TensorGrad LTCCell::_ode_solver(TensorGrad inputs, const TensorGrad &state,
                                const Tensor &elapsed_time) {
    TensorGrad v_pre = state;
    
    // pre-compute the effects of the sensory neurons

    TensorGrad sensory_w_activation =
        this->make_positive_fn(this->_params.at("sensory_w")) *
        this->_sigmoid(inputs, this->_params.at("sensory_mu"),
                       this->_params.at("sensory_sigma"));

    sensory_w_activation *= this->sensory_sparsity_mask;

    TensorGrad sensory_rev_activation =
        sensory_w_activation * this->_params.at("sensory_erev");

    // Reduce over dimension 1 (=source sensory neurons)
    TensorGrad w_numerator_sensory = sensory_rev_activation.sum(1);
    TensorGrad w_denominator_sensory = sensory_w_activation.sum(1);
    // cm/t is loop invariant
    TensorGrad cm_t = this->make_positive_fn(this->_params.at("cm")) /
                      (elapsed_time / (double)this->_ode_unfolds);

    // Unfold the multiply ODE multiple times into one RNN step
    TensorGrad w_param = this->make_positive_fn(
        this->_params.at("w")); 
    for (int64_t t = 0; t < this->_ode_unfolds; ++t) {
        TensorGrad w_activation =
            w_param *
            this->_sigmoid(v_pre, this->_params.at("mu"), this->_params.at("sigma"));
        
        w_activation *= this->sparsity_mask;
        TensorGrad rev_activation = w_activation * this->_params.at("erev");


        // Reduce over dimension 1 (=source neurons)
        
        TensorGrad w_numerator = rev_activation.sum(1) + w_numerator_sensory;
        TensorGrad w_denominator = w_activation.sum(1) + w_denominator_sensory;
        TensorGrad gleak = this->make_positive_fn(this->_params.at("gleak"));

        TensorGrad numerator =
            cm_t * v_pre + gleak * this->_params.at("vleak") + w_numerator;
        TensorGrad denominator = cm_t + gleak + w_denominator;
        // the epsilon is to not divide by 0
        TensorGrad v_pre = numerator / (denominator + this->_epsilon);
    }
    return v_pre;
}

void LTCCell::_map_inputs(TensorGrad &inputs) {
    if (this->_input_mapping == "affine" || this->_input_mapping == "linear") {
        inputs *= this->_params.at("input_w");
        if (this->_input_mapping == "affine") {
            inputs += this->_params.at("input_b");
        }
    }
}

TensorGrad LTCCell::_map_outputs(const TensorGrad &state) {
    TensorGrad output = state;
    if (this->motor_size() < this->state_size()) {
        output = output.transpose(0, 1)[my_range(0, this->motor_size())].transpose(0, 1).contiguous();
    }
    if (this->_output_mapping == "affine" ||
        this->_output_mapping == "linear") {
        output = output * this->_params.at("output_w");
        if (this->_output_mapping == "affine") {
            output = output + this->_params.at("output_b");
        }
    }
    return std::move(output);
}

void LTCCell::apply_weight_constraints() {
    if (!this->_implicit_param_constraints) {
        // softplus when in implicit mode
        this->_params.at("w").tensor = functional::relu(this->_params.at("w").tensor);
        this->_params.at("sensory_w").tensor =
            functional::relu(this->_params.at("sensory_w").tensor);
        this->_params.at("cm").tensor =
            functional::relu(this->_params.at("cm").tensor);
        this->_params.at("gleak").tensor =
            functional::relu(this->_params.at("gleak").tensor);
    }
}

TensorGrad LTCCell::forward(TensorGrad inputs, TensorGrad &states,
                            Tensor elapsed_time) {
    this->_map_inputs(inputs);
    TensorGrad next_state = this->_ode_solver(inputs, states, elapsed_time);
    TensorGrad outputs = this->_map_outputs(next_state);
    states = next_state;
    return std::move(outputs);
}

} // namespace ncps
} // namespace nt
