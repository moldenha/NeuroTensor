#include "../src/layers/Layer.h"
#include "../src/layers/LNN.h"
#include "../src/layers/Optimizers.h"
#include <iostream>




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

		inline nt::TensorGrad forward(const nt::TensorGrad& x) override{
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

		inline nt::TensorGrad forward(const nt::TensorGrad& x){
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


//in this case each sequence length represents an individual neuron
//

class LNN_Use : public nt::Module{
	int64_t channels, output_size;
	public:
		nt::Layer lnn;
		LNN_Use(int64_t channels, int64_t input_size, int64_t hidden_size, int64_t output_size)
			:channels(channels), output_size(output_size),
			lnn(nt::layers::LNN(channels, input_size, hidden_size, output_size))
		{}

		inline nt::TensorGrad forward(const nt::TensorGrad& x){
			nt::TensorGrad out = lnn(x).view(-1, channels, x.shape()[-2], output_size);
			return nt::functional::sigmoid(out);
		}

};

_NT_REGISTER_LAYER_(LNN_Use, lnn)


void test_lnn(){
	auto critereon = nt::loss::MSE;
	nt::TensorGrad input(nt::functional::randn({2, 3, 30, 20}, nt::DType::Float32));
	nt::Tensor wanted = nt::functional::randint(0, 1, {2 ,3, 30, 15}).to(nt::DType::Float32);
	/* std::cout << "wanted: "<<wanted<<std::endl; */
	nt::Layer model = LNN_Use(3, 20, 80, 15);
	nt::optimizers::SGD optimizer(model.parameters(), 0.001);
	optimizer.zero_grad();
	for(int64_t i = 0; i < 10; ++i){
		nt::TensorGrad output = model(input);
		nt::TensorGrad loss = critereon(output, wanted);
		std::cout << "loss: "<<loss.item() << std::endl;
		loss.backward();
		optimizer.step();
	}
	/* std::cout << model(input) << std::endl; */
	/* for(const auto& parameter : model.get+ */
}

