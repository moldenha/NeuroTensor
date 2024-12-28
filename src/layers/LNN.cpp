#include "Layer.h"
#include "layers.h"
#include "LNN.h"
#include "layer_reflect/reflect_macros.h"



namespace nt{
namespace layers{


//this is going to be kind of the default
class LiquidTimeConstantLayer : public Module{
	int64_t _channels, _input_size, _hidden_size;
	TensorGrad hidden;
	public:
		Layer W_in;
		Layer W_state;
		TensorGrad tau;
		TensorGrad time_constants;
		LiquidTimeConstantLayer(int64_t channels, int64_t input_size, int64_t hidden_size)
			:_channels(channels),
			_input_size(input_size),
			_hidden_size(hidden_size),
			W_in(Linear(input_size, hidden_size, false)),
			W_state(Linear(hidden_size, hidden_size, false)),
			tau(functional::ones({hidden_size})),
			time_constants(functional::randn({hidden_size})),
			hidden(nullptr)
		{time_constants.tensor *= 0.1;}

		TensorGrad forward(const TensorGrad& x) override{
			TensorGrad pre_activation = W_in(x) + W_state(this->hidden) + this->tau;
			TensorGrad h_new = this->hidden + this->time_constants * (functional::tanh(pre_activation) - this->hidden);
			return std::move(h_new);
		}
		void register_hidden(TensorGrad x){hidden = std::move(x);}
		void delete_hidden(){hidden.nullify();}
};

_NT_REGISTER_LAYER_(LiquidTimeConstantLayer, W_in, W_state, tau, time_constants);


void register_hidden(Layer& l, Tensor& h0){
	if(auto layer = l.is_layer<LiquidTimeConstantLayer>()){
		layer->register_hidden(TensorGrad(h0));
	}
}

void delete_hidden(Layer& l){
	if(auto layer = l.is_layer<LiquidTimeConstantLayer>()){
		layer->delete_hidden();
	}
}

LNN::LNN(int64_t channels, int64_t input_size, int64_t hidden_size, int64_t output_size, LNNOptions option)
	:_channels(channels), _input_size(input_size), _hidden_size(hidden_size), _output_size(output_size),
	ODE(LiquidTimeConstantLayer(channels, input_size, hidden_size)),
	W_out(Linear(hidden_size, output_size))
{}

TensorGrad LNN::forward(const TensorGrad& x){
	TensorGrad _x = x.view(-1, x.shape()[-2], x.shape()[-1]);
	const int64_t sequence_length = _x.shape()[-2];
	const int64_t batch_size = _x.shape()[-3];
	_x = _x.transpose(0, 1);
	TensorGrad split = _x.split_axis(0);
	Tensor h = functional::zeros({batch_size, _hidden_size}, x.tensor.dtype);
	std::vector<TensorGrad> outputs;
	outputs.reserve(sequence_length);
	for(int64_t i = 0; i < sequence_length; ++i){
		register_hidden(this->ODE, h);
		outputs.push_back(std::move(ODE(split[i])));
		delete_hidden(this->ODE);
		h = outputs.back().tensor;
	}
	TensorGrad output = functional::cat(outputs).view(sequence_length, batch_size, -1).transpose(0,1);
	return this->W_out(output);
}

}}


