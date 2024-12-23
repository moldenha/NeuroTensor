#include "../src/layers/Layer.h"
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
	nt::TensorGrad input(nt::functional::randn({20, 7}, nt::DType::Float32));
	nt::Tensor wanted = nt::functional::randint(0, 1, {20, 2}).to(nt::DType::Float32);
	nt::Layer my_module = CustomModule(7, 2);
	nt::MSELoss loss(my_module);
	nt::TensorGrad output = my_module(input);
	my_module.dump_graph();
	loss(output, wanted);
	std::cout << loss.item() << std::endl;
	loss.backward_no_update();
	/* for(const auto& l : my_module.get_all_layers()){ */
	/* 	//going to print parameter */
	/* 	if(auto custom = l.is_layer<CustomLayer>()){ */
	/* 		std::cout << "parameter and grad: "<<std::endl; */
	/* 		std::cout << custom->parameter << std::endl; */
	/* 		std::cout << custom->parameter.grad_value() << std::endl; */
	/* 	} */
	/* } */

	loss.update();
	/* for(const auto& l : my_module.get_all_layers()){ */
	/* 	//going to print parameter */
	/* 	if(auto custom = l.is_layer<CustomLayer>()){ */
	/* 		std::cout << "new parameter:"<<std::endl; */
	/* 		std::cout << custom->parameter << std::endl; */
	/* 	} */
	/* } */
	nt::TensorGrad output2 = my_module(input);
	loss(output2, wanted);
	std::cout << loss.item() << std::endl;

}

