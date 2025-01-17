#include "wiring.h"

namespace nt {
namespace ncps {

void Wiring::build(int64_t _input_dim){
    utils::THROW_EXCEPTION(!this->input_dim.has_value()
            || this->input_dim.value() == _input_dim,
            "Conflicting input dimensions provided. setting input dim was called with $ but actual input has dimension $", _input_dim, this->input_dim.value_or(0));
    if(!this->input_dim.has_value()){
        this->set_input_dim(_input_dim);
    }
}

void Wiring::set_input_dim(int64_t _input_dim){
    this->input_dim = _input_dim;
    this->sensory_adjacency_matrix = 
        functional::zeros({_input_dim, this->units}, DType::int64);
}


void Wiring::add_synapse(int64_t src, int64_t dest, int64_t polarity){
    utils::THROW_EXCEPTION(src > 0 && src < this->units, "Cannot add synapse originating in $ if cell has only $ units", src, this->units);
    utils::THROW_EXCEPTION(dest > 0 && dest < this->units, "Cannot add synapse feeding into $ if cell has only $ units", dest, this->units);
    utils::THROW_EXCEPTION(polarity == -1 || polarity == 1,
            "Cannot add synapse with polarity $ (expected -1 or +1)", polarity);
    reinterpret_cast<int64_t*>(this->adjacency_matrix.data_ptr())[src * this->units + dest] = polarity;
}

void Wiring::add_sensory_synapse(int64_t src, int64_t dest, int64_t polarity){
    utils::THROW_EXCEPTION(this->input_dim.has_value(),
            "Cannot add a sensory synapse before build is called");
    utils::THROW_EXCEPTION(src >= 0 && src < this->input_dim.value(), 
                           "Cannot add synapse originating in $ if cell has only $ units", 
                           src, this->input_dim.value());
    utils::THROW_EXCEPTION(dest >= 0 && dest < this->units, 
                           "Cannot add synapse feeding into $ if cell has only $ units", 
                           dest, this->units);
    utils::THROW_EXCEPTION(polarity == -1 || polarity == 1,
            "Cannot add synapse with polarity $ (expected -1 or +1)", polarity);
    reinterpret_cast<int64_t*>(this->sensory_adjacency_matrix.data_ptr())[src * this->units + dest] = polarity;
}


Wiring Wiring::from_config(std::map<std::string, Tensor> config){
    Wiring out(config["units"].toScalar().to<int64_t>());
    if(!config["adjacency_matrix"].is_null()){
        out.adjacency_matrix = config["adjacency_matrix"];
    }
    if(!config["sensory_adjacency_matrix"].is_null()){
        out.sensory_adjacency_matrix = config["sensory_adjacency_matrix"];
    }
    if(!config["input_dim"].is_null()){
        out.input_dim = config["input_dim"].toScalar().to<int64_t>();
    }
    if(!config["output_dim"].is_null()){
        out.output_dim = config["output_dim"].toScalar().to<int64_t>();
    }
    return out;	
}


//xnetwork::DiGraph Wiring::get_graph(bool include_sensory_neurons ) const {
//    utils::THROW_EXCEPTION(this->is_built(),
//        "Wiring is not built yet.\n This is probably because the input shape is not known yet.\n Consider calling the model.build(...) method using the shape of the inputs.");

//    xnetwork::DiGraph DG;

//    // add neurons
//    for (int64_t i = 0; i < this->units; ++i) {
//        std::string neuron_name = "neuron_" + std::to_string(i);
//        std::string neuron_type = this->get_type_of_neuron(i);
//        DG.add_node(neuron_name, {{"neuron_type", neuron_type}});
//    }

//    // add sensory neurons if required
//    if (include_sensory_neurons && this->input_dim.has_value()) {
//        for (int64_t i = 0; i < this->input_dim.value(); ++i) {
//            std::string sensory_name = "sensory_" + std::to_string(i);
//            DG.add_node(sensory_name, {{"neuron_type", "sensory"}});
//        }
//    }

//    // add edges from sensory adjacency matrix
//    if (include_sensory_neurons && this->input_dim.has_value()) {
//        for (int64_t src = 0; src < this->input_dim.value(); ++src) {
//            for (int64_t dest = 0; dest < this->units; ++dest) {
//            if (this->sensory_adjacency_matrix[src][dest].item<int64_t>() != 0) {
//                std::string polarity = this->sensory_adjacency_matrix[src][dest].item<int64_t>() >= 0
//                    ? "excitatory"
//                    : "inhibitory";
//                 DG.add_edge("sensory_" + std::to_string(src),
//                        "neuron_" + std::to_string(dest),
//                        {{"polarity", polarity}});
//            }
//            }
//        }
//    }

//    //add edges from adjacency matrix
//    for (int64_t src = 0; src < this->units; ++src) {
//        for (int64_t dest = 0; dest < this->units; ++dest) {
//            if (this->adjacency_matrix[src][dest].item<int64_t>()  != 0) {
//                std::string polarity = this->adjacency_matrix[src][dest].item<int64_t>() >= 0
//                        ? "excitatory"
//                        : "inhibitory";
//                DG.add_edge("neuron_" + std::to_string(src),
//                        "neuron_" + std::to_string(dest),
//                        {{"polarity", polarity}});
//            }
//        }
//    }

//    return DG;
//}

FullyConnected::FullyConnected(int64_t units, std::optional<int64_t> output_dim, bool self_connections, uint32_t erev_init_seed)
    :Wiring(units),
    self_connections(self_connections),
    _erev_init_seed(erev_init_seed),
    _rng(std::mt19937(erev_init_seed))

{
    this->set_output_dim(output_dim.value_or(units));
    std::vector<int64_t> choices({-1, 1, 1});
    std::uniform_int_distribution<std::size_t> dist(0, choices.size() - 1);
    for(int64_t src = 0; src < units; ++src){
        for(int64_t dst = 0; dst < units; ++dst){
            if(src == dst && !this->self_connections){continue;}
            int64_t polarity = choices[dist(this->_rng)];
            this->add_synapse(src, dst, polarity);
        }
    }
}

void FullyConnected::build(int64_t input_dim){ 
    Wiring::build(input_dim);
    std::vector<int64_t> choices({-1, 1, 1});
    std::uniform_int_distribution<std::size_t> dist(0, choices.size() - 1);
    for(int64_t src = 0; src < this->get_input_dim(); ++src){
        for(int64_t dst = 0; dst < this->units; ++dst){
            int64_t polarity = choices[dist(this->_rng)];
            this->add_sensory_synapse(src, dst, polarity);
        }
        
    }
}

FullyConnected from_config(std::map<std::string, Tensor> config){
    return FullyConnected(config["units"].toScalar().to<int64_t>(),
                  config["output_dim"].is_null() ? 
                std::optional<int64_t>(std::nullopt) :
                std::optional<int64_t>(config["output_dim"].toScalar().to<int64_t>()),
                  bool(config["self_connections"].toScalar().to<bool>()),
                  config["erev_init_seed"].toScalar().to<uint32_t>());
}

Random::Random(int64_t units, std::optional<int64_t> output_dim, double sparsity_level, uint32_t random_seed) 
    :Wiring(units),
    _random_seed(random_seed),
    sparsity_level(sparsity_level),
    _rng(std::mt19937(random_seed)) 
{
    utils::THROW_EXCEPTION(sparsity_level >= 0 && sparsity_level < 1.0,
            "Invalid sparsity level, supposed to be in range (0,1], but got $",
            sparsity_level);

    this->set_output_dim(output_dim.value_or(units));
    
    int64_t number_of_synapses = static_cast<int64_t>(
            std::round(static_cast<double>(units * units) * (1.0 - sparsity_level)));
    std::vector<std::tuple<int64_t, int64_t> > all_synapses(units * units);
    auto begin = all_synapses.begin();
    for(int64_t src = 0; src < units; ++src){
        for(int64_t dst = 0; dst < units; ++dst, ++begin){
            *begin = std::make_tuple(src, dst);
        }
    }
    std::vector<std::tuple<int64_t, int64_t>> used_synapses;
    used_synapses.reserve(number_of_synapses);
    std::sample(all_synapses.begin(), all_synapses.end(),
            std::back_inserter(used_synapses),
            number_of_synapses, this->_rng);
    std::vector<int64_t> choices({-1, 1, 1});
    std::uniform_int_distribution<std::size_t> dist(0, choices.size() - 1);
    for(const auto& xy : used_synapses){
        const auto [src, dest] = xy;
        int64_t polarity = choices[dist(this->_rng)];
        this->add_synapse(src, dest, polarity);
    }
}

void Random::build(int64_t input_dim) {
    Wiring::build(input_dim);
    int64_t number_of_sensory_synapses = static_cast<int64_t>(
            std::round(static_cast<double>(this->get_input_dim() * this->units) * 
                    (1.0 - this->sparsity_level)));
    std::vector<std::tuple<int64_t, int64_t> > all_sensory_synapses(units * units);
    auto begin = all_sensory_synapses.begin();
    for(int64_t src = 0; src < this->get_input_dim(); ++src){
        for(int64_t dst = 0; dst < units; ++dst, ++begin){
            *begin = std::make_tuple(src, dst);
        }
    }
    std::vector<std::tuple<int64_t, int64_t>> used_sensory_synapses;
    used_sensory_synapses.reserve(number_of_sensory_synapses);
    std::sample(all_sensory_synapses.begin(), all_sensory_synapses.end(),
            std::back_inserter(used_sensory_synapses),
            number_of_sensory_synapses, this->_rng);

    std::vector<int64_t> choices({-1, 1, 1});
    std::uniform_int_distribution<std::size_t> dist(0, choices.size() - 1);
    for(const auto& xy : used_sensory_synapses){
        const auto [src, dest] = xy;
        int64_t polarity = choices[dist(this->_rng)];
        this->add_sensory_synapse(src, dest, polarity);
        
    }
    
}

Random Random::from_config(std::map<std::string, Tensor> config){
    return Random(config["units"].toScalar().to<int64_t>(),
                  config["output_dim"].is_null() ? 
                std::optional<int64_t>(std::nullopt) :
                std::optional<int64_t>(config["output_dim"].toScalar().to<int64_t>()),
                  config["sparsity_level"].toScalar().to<double>(),
                  config["random_seed"].toScalar().to<uint32_t>());
}

NCP::NCP(int64_t inter_neurons,
        int64_t command_neurons,
        int64_t motor_neurons,
        int64_t sensory_fanout,
        int64_t inter_fanout,
        int64_t recurrent_command_synapses,
        int64_t motor_fanin,
        uint32_t seed)
    :Wiring(inter_neurons + command_neurons + motor_neurons),
    _rng(std::mt19937(seed)),
    _seed(seed),
    _num_inter_neurons(inter_neurons),
    _num_command_neurons(command_neurons),
    _num_motor_neurons(motor_neurons),
    _sensory_fanout(sensory_fanout),
    _inter_fanout(inter_fanout),
    _recurrent_command_synapses(recurrent_command_synapses),
    _motor_fanin(motor_fanin),
    _motor_neurons(functional::arange(motor_neurons, DType::int64, 0)),
    _command_neurons(functional::arange(command_neurons, DType::int64, motor_neurons)),
    _inter_neurons(functional::arange(inter_neurons, DType::int64, motor_neurons + command_neurons)),
    _num_sensory_neurons(0),
    _sensory_neurons(nullptr)
{
    utils::THROW_EXCEPTION(this->_motor_fanin <= this->_num_command_neurons,
            "Error: Motor fanin parameter is $ but there are only $ command neurons",
            this->_motor_fanin, this->_num_command_neurons);
    utils::THROW_EXCEPTION(this->_sensory_fanout <= this->_num_inter_neurons,
            "Error: Sensory fanout parameter is $ but there are only $ inter neurons",
            this->_sensory_fanout, this->_num_inter_neurons);
    utils::THROW_EXCEPTION(this->_inter_fanout <= this->_num_command_neurons,
            "Error: Inter fanout parameter is $ but there are only $ command neurons",
            this->_inter_fanout, this->_num_command_neurons);

}

void NCP::_build_sensory_to_inter_layer(){
    utils::THROW_EXCEPTION(!_sensory_neurons.is_null(), "Expected Sensory Neurons to be initialized to build sensory to inter layer");

    //Randomly connects each sensory neuron to exactly _sensory_fanout number of interneurons
    const int64_t* inter_begin = reinterpret_cast<const int64_t*>(this->_inter_neurons.data_ptr());
    const int64_t* inter_end = reinterpret_cast<const int64_t*>(this->_inter_neurons.data_ptr_end());
    std::vector<int64_t> unreachable_inter_neurons(inter_begin, inter_end);

    const int64_t* sensory_begin = reinterpret_cast<const int64_t*>(this->_sensory_neurons.data_ptr());
    const int64_t* sensory_end = reinterpret_cast<const int64_t*>(this->_sensory_neurons.data_ptr_end());
    std::vector<int64_t> choices({-1, 1});
    std::uniform_int_distribution<std::size_t> dist(0, choices.size() - 1);
    for(const int64_t* src = inter_begin; src != inter_end; ++src){
        std::vector<int64_t> destinations;
        destinations.reserve(this->_sensory_fanout);
        std::sample(inter_begin, inter_end,
            std::back_inserter(destinations),
            this->_sensory_fanout, this->_rng);

        for(const auto& dest : destinations){
            auto found = std::find(unreachable_inter_neurons.begin(), 
                    unreachable_inter_neurons.end(), dest);
            if(found != unreachable_inter_neurons.end()){
                unreachable_inter_neurons.erase(found);
            }
            int64_t polarity = choices[dist(this->_rng)];
            this->add_sensory_synapse(*src, dest, polarity);
        }
    }

    //if it happens that some interneurons are not connected, connect them now
    int64_t mean_inter_neuron_fanin = static_cast<int64_t>(
        this->_num_sensory_neurons * this->_sensory_fanout / this->_num_inter_neurons
        );
    //connect "forgotten" inter neuron by at least 1 and at most all sensory neuron
    int64_t mean_inter_neuo_fanin_lo = 1;
    mean_inter_neuron_fanin = std::clamp(mean_inter_neuron_fanin,
            mean_inter_neuo_fanin_lo, this->_num_sensory_neurons);
    for(const auto& dest : unreachable_inter_neurons){
        std::vector<int64_t> sources;
        sources.reserve(mean_inter_neuron_fanin);
        std::sample(sensory_begin, sensory_end, std::back_inserter(sources),
                mean_inter_neuron_fanin, this->_rng);
        for(const auto& src : sources){
            int64_t polarity = choices[dist(this->_rng)];
            this->add_sensory_synapse(src, dest, polarity);
            
        }
        
    }
}


void NCP::_build_inter_to_command_layer(){
    //Randomly connects each sensory neuron to exactly _sensory_fanout number of interneurons
    const int64_t* inter_begin = reinterpret_cast<const int64_t*>(this->_inter_neurons.data_ptr());
    const int64_t* inter_end = reinterpret_cast<const int64_t*>(this->_inter_neurons.data_ptr_end());


    const int64_t* command_begin = reinterpret_cast<const int64_t*>(this->_command_neurons.data_ptr());
    const int64_t* command_end = reinterpret_cast<const int64_t*>(this->_command_neurons.data_ptr_end());
    std::vector<int64_t> unreachable_command_neurons(command_begin, command_end);

    std::vector<int64_t> choices({-1, 1});
    std::uniform_int_distribution<std::size_t> dist(0, choices.size() - 1);
    for(const int64_t* src = inter_begin; src != inter_end; ++src){
        std::vector<int64_t> destinations;
        destinations.reserve(this->_inter_fanout);
        std::sample(command_begin, command_end,
            std::back_inserter(destinations),
            this->_inter_fanout, this->_rng);

        for(const auto& dest : destinations){
            auto found = std::find(unreachable_command_neurons.begin(), 
                    unreachable_command_neurons.end(), dest);
            if(found != unreachable_command_neurons.end()){
                unreachable_command_neurons.erase(found);
            }
            int64_t polarity = choices[dist(this->_rng)];
            this->add_synapse(*src, dest, polarity);
        }
    }

    //if it happens that some interneurons are not connected, connect them now
    int64_t mean_command_neuron_fanin = static_cast<int64_t>(
        this->_num_inter_neurons * this->_inter_fanout / this->_num_command_neurons
        );
    //connect "forgotten" inter neuron by at least 1 and at most all sensory neuron
    int64_t mean_cmd_neuro_fanin_lo = 1;
    mean_command_neuron_fanin = std::clamp(mean_command_neuron_fanin,
            mean_cmd_neuro_fanin_lo, this->_num_command_neurons);
    for(const auto& dest : unreachable_command_neurons){
        std::vector<int64_t> sources;
        sources.reserve(mean_command_neuron_fanin);
        std::sample(inter_begin, inter_end, std::back_inserter(sources),
                mean_command_neuron_fanin, this->_rng);
        for(const auto& src : sources){
            int64_t polarity = choices[dist(this->_rng)];
            this->add_synapse(src, dest, polarity);
            
        }
        
    }
}

void NCP::_build_recurrent_command_layer(){
    std::vector<int64_t> polarity_choices({-1, 1});
    utils::THROW_EXCEPTION(!this->_command_neurons.is_null(),
                           "Logic error command neurons is null");
    const int64_t* command_begin = reinterpret_cast<const int64_t*>(this->_command_neurons.data_ptr());
    const int64_t* command_end = reinterpret_cast<const int64_t*>(this->_command_neurons.data_ptr_end());

    const int64_t command_size = command_end - command_begin; 
    std::uniform_int_distribution<std::size_t> dist_polarity(0, polarity_choices.size() - 1);
    std::uniform_int_distribution<std::size_t> dist_commands(0, command_size-1);
    for(int64_t i = 0; i < _recurrent_command_synapses; ++i){
        int64_t src = command_begin[dist_commands(this->_rng)];
        int64_t dest = command_begin[dist_commands(this->_rng)];
        int64_t polarity = polarity_choices[dist_polarity(this->_rng)];
        this->add_synapse(src, dest, polarity);
    }
}

void NCP::_build_command_to_motor_layer(){
    //Randomly connect command neurons to motor neurons
    //Randomly connects each sensory neuron to exactly _sensory_fanout number of interneurons
    const int64_t* motor_begin = reinterpret_cast<const int64_t*>(this->_motor_neurons.data_ptr());
    const int64_t* motor_end = reinterpret_cast<const int64_t*>(this->_motor_neurons.data_ptr_end());


    const int64_t* command_begin = reinterpret_cast<const int64_t*>(this->_command_neurons.data_ptr());
    const int64_t* command_end = reinterpret_cast<const int64_t*>(this->_command_neurons.data_ptr_end());
    std::vector<int64_t> unreachable_command_neurons(command_begin, command_end);

    std::vector<int64_t> choices({-1, 1});
    std::uniform_int_distribution<std::size_t> dist(0, choices.size() - 1);
    for(const int64_t* dest = command_begin; dest != command_end; ++dest){
        
        std::vector<int64_t> sources;
        sources.reserve(this->_motor_fanin);
        std::sample(command_begin, command_end,
            std::back_inserter(sources),
            this->_motor_fanin, this->_rng);

        for(const auto& src : sources){
            auto found = std::find(unreachable_command_neurons.begin(), 
                    unreachable_command_neurons.end(), src);
            if(found != unreachable_command_neurons.end()){
                unreachable_command_neurons.erase(found);
            }
            int64_t polarity = choices[dist(this->_rng)];
            this->add_synapse(src, *dest, polarity);
        }
    }

    //if it happens that some command neurons are not connected, connect them now
    int64_t mean_command_fanout = static_cast<int64_t>(
        this->_num_motor_neurons * this->_motor_fanin / this->_num_command_neurons
        );
    //connect "forgotten" command neuron by at least 1 and at most all motor neurons
    int64_t mean_cmd_low = 1;
    mean_command_fanout = std::clamp(mean_command_fanout,
            mean_cmd_low, this->_num_motor_neurons);
    for(const auto& src : unreachable_command_neurons){
        std::vector<int64_t> destinations;
        destinations.reserve(mean_command_fanout);
        std::sample(motor_begin, motor_end, std::back_inserter(destinations),
                mean_command_fanout, this->_rng);
        for(const auto& dest : destinations){
            int64_t polarity = choices[dist(this->_rng)];
            this->add_synapse(src, dest, polarity);
            
        }
        
    }
}


void NCP::build(int64_t input_dim){ 
    Wiring::build(input_dim);
    this->_num_sensory_neurons = this->get_input_dim();
    this->_sensory_neurons = functional::arange(this->_num_sensory_neurons, DType::int64);

    this->_build_sensory_to_inter_layer();
    this->_build_inter_to_command_layer();
    this->_build_recurrent_command_layer();
    this->_build_command_to_motor_layer();
}

NCP NCP::from_config(std::map<std::string, Tensor> config){
    /* NCP(int64_t inter_neurons, */
    /* int64_t command_neurons, */
    /* int64_t motor_neurons, */
    /* int64_t sensory_fanout, */
    /* int64_t inter_fanout, */
    /* int64_t recurrent_command_synapses, */
    /* int64_t motor_fanin, */
    /* uint32_t seed=22222) */

    NCP out(config["inter_neurons"].numel(),
            config["command_neurons"].numel(),
            config["motor_neurons"].numel(),
            config["sensory_fanout"].toScalar().to<int64_t>(),
            config["inter_fanout"].toScalar().to<int64_t>(),
            config["recurrent_command_synapses"].toScalar().to<int64_t>(),
            config["motor_fanin"].toScalar().to<int64_t>(),
            config["seed"].toScalar().to<uint32_t>());
    if(!config["input_dim"].is_null()){
        out.build(config["input_dim"].toScalar().to<int64_t>());
    }
    out._inter_neurons = config["inter_neurons"];
    out._command_neurons = config["command_neurons"];
    out._motor_neurons = config["motor_neurons"];


    return out;
}


NCP AutoNCP::generate_ncp(int64_t units, int64_t output_size, double sparsity_level, uint32_t seed){
    utils::THROW_EXCEPTION(output_size < (units-2),
        "Output size must be less than the number of units-2"
        "(given $ units, $ output size)", units, output_size);
    utils::THROW_EXCEPTION(sparsity_level > 0.1 && sparsity_level < 1.0, 
            "Expected to have sparsity level in (0.1, 1.0) but got $", sparsity_level);
    double density_level = 1.0 - sparsity_level;
    int64_t inter_and_command_neurons = units - output_size;
    int64_t command_neurons = std::max(static_cast<int64_t>(0.4 * (double)inter_and_command_neurons),  static_cast<int64_t>(1));
    int64_t inter_neurons = inter_and_command_neurons - command_neurons;

    int64_t sensory_fanout = std::max(static_cast<int64_t>(density_level * double(inter_neurons)),  static_cast<int64_t>(1));
    int64_t inter_fanout = std::max(static_cast<int64_t>(density_level * double(command_neurons)),  static_cast<int64_t>(1));
    int64_t recurrent_command_synapses = std::max(static_cast<int64_t>(double(command_neurons) * density_level * 2.0),  static_cast<int64_t>(1));
    int64_t motor_fanin = std::max(static_cast<int64_t>(double(command_neurons) * density_level), static_cast<int64_t>(1));
    return NCP(inter_neurons, command_neurons, output_size,
            sensory_fanout, inter_fanout, recurrent_command_synapses,
            motor_fanin, seed);


}

}
}
