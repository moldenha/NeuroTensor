#ifndef _NT_LAYERS_NCPS_WIRING_H_
#define _NT_LAYERS_NCPS_WIRING_H_

#include "../../../Tensor.h"
#include "../../../intrusive_ptr/intrusive_ptr.hpp"
// #include <xnetwork/classes/digraphs.hpp>
#include <optional>
#include <map>
#include <cstdint>
#include <random>
#include <tuple>
#include <algorithm>

//look at:
//https://github.com/mlech26l/ncps/blob/master/ncps/wirings/wirings.py
//this is basically a copy of the above
//but in c++, and adapted specifically for NeuroTensor
//

namespace nt{
namespace ncps{
class Wiring : public intrusive_ptr_target{
		Tensor adjacency_matrix, sensory_adjacency_matrix;
	protected:
		std::optional<int64_t> input_dim, output_dim;
		int64_t units;
	public:
		Wiring(int64_t units)
			:units(units),
			adjacency_matrix(functional::zeros({units, units}, DType::int64)),
			sensory_adjacency_matrix(nullptr),
			input_dim(std::nullopt),
			output_dim(std::nullopt)
			{}

		virtual inline int64_t num_layers() const noexcept {return 1;}
		virtual inline Tensor get_neurons_of_layer(int64_t layer_id) const {return functional::arange(this->units, DType::int64); }
        virtual inline int64_t get_number_neurons_of_layer(int64_t layer_id) const {return this->units;}
		inline bool is_built() const noexcept {return this->input_dim.has_value();}
		virtual void build(int64_t _input_dim);

		inline Tensor erev_initializer(std::optional<SizeRef> shape = std::nullopt, 
                                 std::optional<DType> dtype = std::nullopt){
			return this->adjacency_matrix.clone();
		}
		inline Tensor sensory_erev_initializer(std::optional<SizeRef> shape = std::nullopt, 
                                  std::optional<DType> dtype = std::nullopt){
			return this->sensory_adjacency_matrix.clone();
		}
		void set_input_dim(int64_t _input_dim);

		inline void set_output_dim(int64_t _output_dim){
			this->output_dim = _output_dim;
		}

		virtual inline std::string get_type_of_neuron(int64_t neuron_id) const {
			utils::THROW_EXCEPTION(this->output_dim.has_value(), "Expected output dim to have a value in order to get the neuron type");
			return (neuron_id < this->output_dim.value()) ? std::string("motor") : std::string("inter");
		}

		void add_synapse(int64_t src, int64_t dest, int64_t polarity);
        void add_sensory_synapse(int64_t src, int64_t dest, int64_t polarity);
		virtual inline std::map<std::string, Tensor> get_config() const{
			return { {"units", Tensor(Scalar(this->units))},
				 {"adjacency_matrix", this->adjacency_matrix},
				 {"sensory_adjacency_matrix", this->sensory_adjacency_matrix},
				 {"input_dim", this->input_dim.has_value() ? Tensor(Scalar(this->input_dim.value())) 
					 : Tensor::Null()},
				 {"output_dim", this->output_dim.has_value() ? Tensor(Scalar(this->output_dim.value())) 
					 : Tensor::Null()}
			};
		}

		static Wiring from_config(std::map<std::string, Tensor> config);

		// xnetwork::DiGraph get_graph(bool include_sensory_neurons = true) const; 
		//Counts the number of synapses between internal neurons of the model
		inline int64_t synapse_count() const {
			return std::abs(this->adjacency_matrix).sum().toScalar().to<int64_t>();
		}
		//Counts the number of synapses from the inputs (sensory neurons) to the internal neurons of the model
		inline int64_t sensory_synapse_count() const {
            utils::THROW_EXCEPTION(!this->sensory_adjacency_matrix.is_null(),
                                   "Error, sensory adjacency matrix is null, not built");
			return std::abs(this->sensory_adjacency_matrix).sum().toScalar().to<int64_t>();
			
		}
		inline int64_t get_input_dim() const noexcept {return this->input_dim.value_or(0);}
		inline int64_t get_output_dim() const noexcept {return this->output_dim.value_or(0);}
		inline int64_t get_units() const noexcept {return this->units;}
		inline const Tensor& get_adjacency_matrix() const noexcept { return this->adjacency_matrix;}
		inline const Tensor& get_sensory_adjacency_matrix() const noexcept { return this->sensory_adjacency_matrix;}

		//TODO: make a draw graph to draw the graph (display it in a gui) <- this is a maybe


};

class FullyConnected final : public Wiring{
	bool self_connections;
	uint32_t _erev_init_seed;
	std::mt19937 _rng;
	public:
		FullyConnected(int64_t units, std::optional<int64_t> output_dim = std::nullopt, bool self_connections=true, uint32_t erev_init_seed=1111);

		void build(int64_t input_dim) override; 

		inline std::map<std::string, Tensor> get_config() const override{
			return { {"units", Tensor(Scalar(this->units))},
				 {"self_connections", Tensor(Scalar(this->self_connections)) },
				 {"erev_init_seed", Tensor(Scalar(this->_erev_init_seed)) },
				 {"output_dim", this->output_dim.has_value() ? Tensor(Scalar(this->output_dim.value())) 
					 : Tensor::Null()}
			};
		}

		static FullyConnected from_config(std::map<std::string, Tensor> config);
	
};


class Random final : public Wiring{
	uint32_t _random_seed;
	double sparsity_level;
	std::mt19937 _rng;
	public:
		Random(int64_t units, std::optional<int64_t> output_dim = std::nullopt, double sparsity_level = 0.0, uint32_t random_seed = 1111);

		void build(int64_t input_dim) override; 
		
		inline std::map<std::string, Tensor> get_config() const override{
			return { {"units", Tensor(Scalar(this->units))},
				 {"random_seed", Tensor(Scalar(this->_random_seed)) },
				 {"sparsity_level", Tensor(Scalar(this->sparsity_level)) },
				 {"output_dim", this->output_dim.has_value() ? Tensor(Scalar(this->output_dim.value())) 
					 : Tensor::Null()}
			};
		}

		static Random from_config(std::map<std::string, Tensor> config);
		
};

/*
https://github.com/mlech26l/ncps/blob/master/ncps/wirings/wirings.py
Line 408:
Creates a Neural Circuit Policies wiring.
The total number of neurons (= state size of the RNN) is given by the sum of inter, command, and motor neurons.

inter_neurons: The number of inter neurons (layer 2)
command_neurons: The number of command neurons (layer 3)
motor_neurons: The number of motor neurons (layer 4 = number of outputs)
sensory_fanout: The average number of outgoing synapses from the sensory to the inter neurons
inter_fanout: The average number of outgoing synapses from the inter to the command neurons
recurrent_command_synapses: The average number of recurrent connections in the command neuron layer
motor_fanin: The average number of incoming synapses of the motor neurons from the command neurons
seed: The random seed used to generate the wiring

 */

class NCP : public Wiring{
	std::mt19937 _rng;
	int64_t _num_inter_neurons, _num_command_neurons, _num_motor_neurons, _sensory_fanout,
		_inter_fanout, _recurrent_command_synapses, _motor_fanin;
	Tensor _motor_neurons, _command_neurons, _inter_neurons;
	int64_t _num_sensory_neurons;
	Tensor _sensory_neurons;
	protected:
	uint32_t _seed;
    void _build_sensory_to_inter_layer();
    void _build_inter_to_command_layer();
    void _build_recurrent_command_layer();
    void _build_command_to_motor_layer();


	public:
		NCP(int64_t inter_neurons,
		    int64_t command_neurons,
		    int64_t motor_neurons,
		    int64_t sensory_fanout,
		    int64_t inter_fanout,
		    int64_t recurrent_command_synapses,
		    int64_t motor_fanin,
		    uint32_t seed=22222);
        NCP(const NCP &val) = default;
		inline int64_t num_layers()  const noexcept override {return 3;}
		inline Tensor get_neurons_of_layer(int64_t layer_id) const override {
			utils::THROW_EXCEPTION(layer_id >= 0 && layer_id <= 2,
					"Error: Expected Layer id for NCP to be in [0, 2] but got $", layer_id);
			if(layer_id == 0)
				return this->_inter_neurons;
			else if(layer_id == 1)
				return this->_command_neurons;
			else if(layer_id == 2)
				return this->_motor_neurons;
			else
				return Tensor();
		}
        inline int64_t get_number_neurons_of_layer(int64_t layer_id) const override { 
            utils::THROW_EXCEPTION(layer_id >= 0 && layer_id <= 2,
					"Error: Expected Layer id for NCP to be in [0, 2] but got $", layer_id);
			if(layer_id == 0)
				return this->_inter_neurons.numel();
			else if(layer_id == 1)
				return this->_command_neurons.numel();
			else if(layer_id == 2)
				return this->_motor_neurons.numel();
			else
				return 0;
        }
		inline std::string get_type_of_neuron(int64_t neuron_id) const override{
			if (neuron_id < this->_num_motor_neurons)
				return "motor";
			if (neuron_id < (this->_num_motor_neurons + this->_num_command_neurons))
				return "command";
			return "inter";
		}


		void build(int64_t input_dim) override; 


		inline std::map<std::string, Tensor> get_config() const override{
			return { {"inter_neurons", this->_inter_neurons},
				 {"command_neurons", this->_command_neurons },
				 {"motor_neurons", this->_motor_neurons },
				 {"sensory_fanout", Tensor(Scalar(this->_sensory_fanout)) },
				 {"inter_fanout", Tensor(Scalar(this->_inter_fanout)) },
				 {"recurrent_command_synapses", Tensor(Scalar(this->_recurrent_command_synapses)) },
				 {"motor_fanin", Tensor(Scalar(this->_motor_fanin)) },
				 {"seed", Tensor(Scalar(this->_seed)) },
				 {"input_dim", this->is_built() ? Tensor(Scalar(this->_num_sensory_neurons)) 
					 : Tensor::Null()}
			};
		}

		static NCP from_config(std::map<std::string, Tensor> config);

};

class AutoNCP final : public NCP{
	int64_t _output_size;
	double _sparsity_level;
	
	static NCP generate_ncp(int64_t units, int64_t output_size, double sparsity_level, uint32_t seed);

	public:
		AutoNCP(int64_t units, int64_t output_size, double sparsity_level = 0.5, uint32_t seed = 22222)
			:NCP(AutoNCP::generate_ncp(units, output_size, sparsity_level, seed)),
			_output_size(output_size),
			_sparsity_level(sparsity_level)
		{}

		inline std::map<std::string, Tensor> get_config() const override{
			return { {"units", Tensor(Scalar(this->units))},
				 {"seed", Tensor(Scalar(this->_seed)) },
				 {"sparsity_level", Tensor(Scalar(this->_sparsity_level)) },
				 {"output_size", Tensor(Scalar(this->_output_size)) } 
			};

		}
		
		inline static AutoNCP from_config(std::map<std::string, Tensor> config){
			return AutoNCP(config["units"].toScalar().to<int64_t>(),
					config["output_size"].toScalar().to<int64_t>(),
					      config["sparsity_level"].toScalar().to<double>(),
					      config["seed"].toScalar().to<uint32_t>());	
		}
};

}} //nt::ncps::

#endif //_NT_LAYERS_NCPS_WIRING_H_ 
