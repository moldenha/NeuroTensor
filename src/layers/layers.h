#ifndef _NT_LAYERS_H_
#define _NT_LAYERS_H_

#include "../Tensor.h"

#include <cstdint>
#include <type_traits>
#include <variant>
#include <functional>
#include <tuple>
#include "Module.h"
#include "functional.h"
#include "TensorGrad.h"
#include "layer_reflect/reflect_macros.h"
#include "layer_reflect/layer_registry.hpp"

namespace nt{
namespace layers{
//the linear layer:

inline Tensor loss_error(const Tensor& output, const Tensor& target){
	return output - target;
}


class Linear : public Module{
	public:
		bool use_bias;
		TensorGrad Weight, Bias;
		Linear(int64_t in_dims, int64_t out_dims, bool use_bias = true)
			:Weight(functional::randn({in_dims, out_dims})), 
			Bias(use_bias ? functional::randn({out_dims}) : Tensor()), 
			use_bias(use_bias)
		{}
		inline TensorGrad forward(const TensorGrad& x) override {
			if(use_bias){return functional::matmult(x, Weight) + Bias;}
			return functional::matmult(x, Weight);
		}
};




_NT_REGISTER_LAYER_(Linear, use_bias, Weight, Bias)


class Identity : public Module{
	public:
		Identity() = default;
};

_NT_REGISTER_LAYER_(Identity)


class Conv2d : public Module{
	public:
		bool use_bias;
		int64_t groups, in_channels, out_channels;
		utils::my_tuple stride, padding, dilation;
		TensorGrad Weight, Bias;
		Conv2d(int64_t in_channels, int64_t out_channels, utils::my_tuple kernel_size, utils::my_tuple stride = 1, utils::my_tuple padding = 0, utils::my_tuple dilation = 1, int64_t groups = 1, bool use_bias = true)
			:use_bias(use_bias),
			groups(groups),
			in_channels(in_channels),
			out_channels(out_channels),
			stride(stride),
			padding(padding),
			dilation(dilation),
			Weight(functional::randn({out_channels, in_channels/groups, kernel_size[0], kernel_size[1]})),
			Bias(use_bias ? functional::randn({out_channels, 1, 1}) : Tensor())
			{
				utils::THROW_EXCEPTION(out_channels % groups == 0, "Expected in channels to be divisible by groups");
				utils::THROW_EXCEPTION(in_channels % groups == 0, "Expected in channels to be divisible by groups");
			}

		inline TensorGrad forward(const TensorGrad& x) override {
			utils::THROW_EXCEPTION(x.shape()[-3] == in_channels, "Expected input tensor to have channel size of $ but got $", in_channels, x.shape());
			TensorGrad outp = functional::conv2d(x, Weight, stride, padding, dilation, groups);
			if(!use_bias){return outp;}
			return outp + Bias;
		}
};

_NT_REGISTER_LAYER_(Conv2d, use_bias, groups, in_channels, out_channels, Weight, Bias)


class Sigmoid : public Module{
	public:
		Sigmoid() {;}
		inline TensorGrad forward(const TensorGrad& x) override {
			return functional::sigmoid(x);
		}
	
};

_NT_REGISTER_LAYER_(Sigmoid)



class ReLU : public Module{
	public:
		ReLU() {;}
		inline TensorGrad forward(const TensorGrad& x) override{
			return functional::relu(x);
		}
};

_NT_REGISTER_LAYER_(ReLU)


class Unfold2D : public Module{
	public:
		utils::my_tuple kernel_size, dilation, padding, stride;
		bool transpose_out;
		Unfold2D(utils::my_tuple kernel_size,
				utils::my_tuple dilation=1,
				utils::my_tuple padding=0,
				utils::my_tuple stride=1,
				bool transpose_out=true)
			:kernel_size(kernel_size), dilation(dilation), padding(padding), stride(stride), transpose_out(transpose_out)
		{}

		inline TensorGrad forward(const TensorGrad& x) override{
			return functional::unfold(x, kernel_size, dilation, padding, stride, transpose_out);
		}
};

_NT_REGISTER_LAYER_(Unfold2D, kernel_size, dilation, padding, stride, transpose_out)


class Fold : public Module{
	public:
		utils::my_tuple output_size, kernel_size, dilation, padding, stride;
		Fold(utils::my_tuple output_size,
				utils::my_tuple kernel_size,
				utils::my_tuple dilation = 1,
				utils::my_tuple padding = 0,
				utils::my_tuple stride = 1)
			:output_size(output_size), kernel_size(kernel_size), dilation(dilation), padding(padding), stride(stride)
		{}
		inline TensorGrad forward(const TensorGrad& x) override{
			return functional::fold(x, output_size, kernel_size, dilation, padding, stride);
		}
};

_NT_REGISTER_LAYER_(Fold, output_size, kernel_size, dilation, padding, stride)



class Unfold1D : public Module{
	public:
		Tensor::size_value_t kernel_size, dilation, padding, stride;
		bool transpose_out;
		Unfold1D(Tensor::size_value_t kernel_size,
				Tensor::size_value_t dilation=1,
				Tensor::size_value_t padding=0,
				Tensor::size_value_t stride=1,
				bool transpose_out=true)
			:kernel_size(kernel_size), dilation(dilation), padding(padding), stride(stride), transpose_out(transpose_out)
		{}

		inline TensorGrad forward(const TensorGrad& x) override{
			return functional::unfold1d(x, kernel_size, dilation, padding, stride, transpose_out);
		}
};

_NT_REGISTER_LAYER_(Unfold1D, kernel_size, dilation, padding, stride, transpose_out)

class Unfold3D : public Module{
	public:
		utils::my_n_tuple<3> kernel_size, dilation, padding, stride;
		bool transpose_out;
		Unfold3D(utils::my_n_tuple<3> kernel_size,
				utils::my_n_tuple<3> dilation=1,
				utils::my_n_tuple<3> padding=0,
				utils::my_n_tuple<3> stride=1,
				bool transpose_out=true)
			:kernel_size(kernel_size), dilation(dilation), padding(padding), stride(stride), transpose_out(transpose_out)
		{}

		inline TensorGrad forward(const TensorGrad& x) override{
			return functional::unfold3d(x, kernel_size, dilation, padding, stride, transpose_out);
		}
};

_NT_REGISTER_LAYER_(Unfold3D, kernel_size, dilation, padding, stride, transpose_out)



/* class Sequential{ */
/* 	std::vector<Layer> layers; */
/* 	std::vector<Layer> to_vec(){return {};} */
/* 	template<typename... Args> */
/* 	std::vector<Layer> to_vec(Layer first, Args... rest){ */
/* 		std::vector<Layer> result = {first}; */
/* 		auto tail = to_vec(rest...); */
/* 		result.insert(result.end(), tail.begin(), tail.end()); */
/* 		return result; */
/* 	} */
/* 	public: */
/* 		template<typename... ls> */
/* 		Sequential(Layer first, ls... for_layer) */
/* 		:layers(to_vec(first, for_layer...)) */
/* 		{} */
/* 		inline Tensor forward(const Tensor& x){ */
/* 			Tensor inp = layers[0].forward(x); */
/* 			for(uint32_t i = 1; i < layers.size(); ++i) */
/* 				inp = layers[i].forward(inp); */
/* 			return inp; */
/* 		} */
/* 		inline Tensor backward(const Tensor& dZ){ */
/* 			Tensor inp = layers.back().backward(dZ); */
/* 			for(int32_t i = layers.size()-2; i >= 0; --i) */
/* 				inp = layers[i].backward(inp); */
/* 			return inp; */
/* 		} */
/* 		inline Tensor eval(const Tensor& x) const{ */
/* 			Tensor inp = layers[0].eval(x); */
/* 			for(uint32_t i = 1; i < layers.size(); ++i) */
/* 				inp = layers[i].eval(inp); */
/* 			return inp; */	
/* 		} */
/* 		inline Layer& operator[](uint32_t i){return layers[i];} */
/* 		inline const std::size_t size() const {return layers.size();} */

/* 		Tensor parameters(){ */
/* 			uint32_t count = 0; */
/* 			for(uint32_t i = 0; i < layers.size(); ++i) */
/* 				count += layers[i].parameter_count(); */
/* 			Tensor outp({count}, DType::TensorObj); */

/* 			uint32_t current = 0; */
/* 			for(uint32_t i = 0; i < layers.size(); ++i){ */
/* 				if(layers[i].parameter_count() > 0){ */
/* 					Tensor params = layers[i].parameters(); */
/* 					for(uint32_t j = 0; j < params.shape()[0]; ++j, ++current){ */
/* 						outp[current] = params[j]; */
/* 					} */
/* 				} */

/* 			} */
/* 			return outp; */
/* 		} */

/* }; */

}
}

#endif
