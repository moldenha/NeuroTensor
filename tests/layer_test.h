#include "../src/layers/Layer.h"
#include "../src/layers/LNN.h"
#include "../src/layers/Optimizers.h"
#include <iostream>
#include <sstream>




//these are just examples
//not anything really usable
class CustomLayer : public nt::Module{
	public:
		nt::Layer l1;
		nt::Layer b1;
		nt::TensorGrad parameter;
		bool normalize;
		CustomLayer(int64_t hidden_features, int64_t out_features, bool normal = true)
			:l1(nt::layers::Linear(hidden_features, out_features)),
			b1(normal ? nt::Layer(nt::layers::BatchNorm1D(out_features)) : nt::Layer(nt::layers::Identity())),
			parameter(nullptr),
			normalize(normal)
			{}

		inline nt::TensorGrad forward(nt::TensorGrad x){
			std::cout << "starting custom layer"<<std::endl;
			nt::TensorGrad out = l1(x);
			std::cout << "doing batch normalization"<<std::endl;
			std::cout << out.shape() << std::endl;
			out = b1(out);
			std::cout << "checking for null parameter..."<<std::endl;
			if(parameter.is_null()){
				parameter = nt::functional::zeros({x.shape()[-2], 1}, x.tensor.dtype);
			}
			return out + parameter;
		}

};

_NT_REGISTER_LAYER_(CustomLayer, l1, b1, parameter, normalize)

class CustomModule : public nt::Module{
	public:
		nt::Layer l1, l2, l3, o1;
		CustomModule(int64_t in_features, int64_t out_features)
			:l1(CustomLayer(in_features, in_features + 2)),
			l2(CustomLayer(in_features+2, in_features+4, false)),
			l3(CustomLayer(in_features+4, in_features+6)),
			o1(CustomLayer(in_features+6, out_features, false))
		{}

		inline nt::TensorGrad forward(nt::TensorGrad x){
			nt::TensorGrad a = l1(x);
			nt::TensorGrad b = l2(nt::functional::tanh(a));
			nt::TensorGrad c = l3(nt::functional::sigmoid(b));
			nt::TensorGrad d = o1(c);
			return nt::functional::sigmoid(d);
		}

};

_NT_REGISTER_LAYER_(CustomModule, l1, l2, l3, o1)

void test_layers(){
	auto critereon = nt::loss::MSE;
	nt::TensorGrad input(nt::functional::randn({20, 7}, nt::DType::Float32));
	nt::Tensor wanted = nt::functional::randint(0, 1, {20, 2}).to(nt::DType::Float32);
	nt::Layer my_module = CustomModule(7, 2);
	nt::optimizers::SGD optimizer(my_module.parameters());
	nt::TensorGrad output = my_module(input);
	nt::TensorGrad loss = critereon(output, wanted);
	std::cout << loss.item() << std::endl;
	optimizer.zero_grad();
	loss.backward();
	for(const auto& l : my_module.get_all_layers()){
		//going to print parameter
		if(auto custom = l.is_layer<CustomLayer>()){
			std::cout << "parameter and grad: "<<std::endl;
			std::cout << custom->parameter << std::endl;
			std::cout << custom->parameter.grad_value() << std::endl;
		}
	}

	optimizer.step();
	/* for(const auto& l : my_module.get_all_layers()){ */
	/* 	//going to print parameter */
	/* 	if(auto custom = l.is_layer<CustomLayer>()){ */
	/* 		std::cout << "new parameter:"<<std::endl; */
	/* 		std::cout << custom->parameter << std::endl; */
	/* 	} */
	/* } */
	nt::TensorGrad output2 = my_module(input);
	loss = critereon(output2, wanted);
	std::cout << loss.item() << std::endl;
	loss.backward();
	optimizer.step();

}




void test_lnn(){
	auto critereon = nt::loss::MSE;
	nt::TensorGrad input(nt::functional::randn({1, 2, 20}, nt::DType::Float32));
	nt::Tensor wanted = nt::functional::randint(0, 1, {1, 10}).to(nt::DType::Float32);
	/* std::cout << "wanted: "<<wanted<<std::endl; */
	nt::Layer model = nt::layers::LNN(20, 10, 18, nt::layers::LNNOptions::CfC, false);
	nt::optimizers::Adam optimizer(model.parameters(), 0.1);
	optimizer.zero_grad();


	for(int64_t i = 0; i < 10; ++i){
		nt::TensorGrad output = model(input);
        if(i == 0){std::cout << output << std::endl;}
		nt::TensorGrad loss = critereon(output, wanted);
		std::cout << "loss: "<<loss.item() << std::endl;
		loss.backward();
		optimizer.step();
	}
	std::cout << model(input) << std::endl;
    std::cout << wanted << std::endl;
	/* for(const auto& parameter : model.get+ */
}

class TimeAwareHiddenLayer : public nt::Module{
    int64_t _input_size, _hidden_size;
    public:
        nt::Layer input_proj, hidden_proj, time_proj, output_proj;
        TimeAwareHiddenLayer(int64_t input_size, int64_t hidden_size)
        :_input_size(input_size),
        _hidden_size(hidden_size),
        input_proj(nt::layers::Linear(input_size, hidden_size)),
        hidden_proj(nt::layers::Linear(hidden_size, hidden_size)),
        time_proj(nt::layers::Linear(1, hidden_size)), // Scalar time which is going to be a tensor of shape (1,1)
        output_proj(nt::layers::Linear(hidden_size, hidden_size))
        {}

        // Automatically tracks if references are expected
        // For the nt::Scalar, if a scalar rvalue is passed, will automatically convert it to an nt::Scalar
        nt::TensorGrad forward(const nt::TensorGrad& x, nt::TensorGrad& h, nt::Scalar t){
            // Params: x: input (const reference)
            //         h: reference to hidden hidden
            //         t: 00:5
            
            nt::TensorGrad time(nt::Tensor(t).view(1,1), false); // The layer class expects an nt::TensorGrad argument
            
            nt::TensorGrad x_proj = this->input_proj(x);
            nt::TensorGrad h_proj = this->hidden_proj(h);
            nt::TensorGrad t_proj = this->time_proj(time);
            
            // Will modify the value the reference h points to
            if(t.to<int64_t>() == 1){
                h += h_proj;
            }
            else{
                h -= h_proj;
            }
            nt::TensorGrad combined = nt::functional::relu(x_proj + h_proj + t_proj);
            nt::TensorGrad output = this->output_proj(combined);
            return std::move(output);
        }
};


class WrapperLayer : public nt::Module{
    int64_t _hidden_size;
    public:
        nt::Layer time_aware_layer;
        nt::TensorGrad default_hidden; // Stores a learnable default hidden state
        WrapperLayer(int64_t input_size, int64_t hidden_size)
        :_hidden_size(hidden_size),
        time_aware_layer(TimeAwareHiddenLayer(input_size,  hidden_size)),
        default_hidden(nt::functional::zeros({1, hidden_size}))
        {}
        
        nt::TensorGrad forward(nt::TensorGrad x){
            int64_t batch_size = x.shape()[0];
            nt::TensorGrad h = this->default_hidden.expand({batch_size, this->_hidden_size});
            // will convert 1 to an nt::Scalar
            // and use the lvalue references of x and h
            return this->time_aware_layer(x, h, 1); //will convert 1 to an nt::Scalar and 
        }
        

};



//this adds reflection to the layer so that gradients can be tracked properly and automatically
_NT_REGISTER_LAYER_(TimeAwareHiddenLayer, input_proj, hidden_proj, time_proj, output_proj)
_NT_REGISTER_LAYER_(WrapperLayer, time_aware_layer, default_hidden)
//any variable that is a NeuroTensor object should be included


