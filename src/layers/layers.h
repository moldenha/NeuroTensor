#include "../Tensor.h"
#include <_types/_uint32_t.h>
#include <cstdint>
#include <type_traits>
#include <variant>
#include <functional>
#include <tuple>

namespace nt{
namespace layers{
//the linear layer:

inline Tensor loss_error(const Tensor& output, const Tensor& target){
	Tensor error({2}, DType::TensorObj);;
	error[0] = target - output;
	error[1] = target - output;
	return std::move(error);
}

class Linear{
	Tensor Weight, Bias, WeightT;
	Tensor APrev;
	Scalar lr, clip_max, clip_min;
	bool use_bias, a_prev_trans, clip;
	Tensor backprop_dx(const Tensor& dZ);
	void backprop_dw(const Tensor& dZ);
	void backprop_db(Tensor& dZ);
	friend class Layer;
	public:
		Linear(uint32_t in_rows, uint32_t out_rows, Scalar lr = Scalar(0.1), bool use_bias=true, bool clip=true, Scalar maxClip = Scalar(5), Scalar minClip = Scalar(-5), DType dt = DType::Float32);
		Tensor forward(const Tensor& x);
		Tensor eval(const Tensor& x) const;
		Tensor backward(const Tensor& dZ);
		void print();
};

class Identity{
	friend class Layer;
	public:
		Identity() {;}
		inline Tensor forward(const Tensor& x){return x;}
		inline Tensor backward(const Tensor& x){return x;}
		inline Tensor eval(const Tensor& x) const {return x;}
};


class ActivationFunction{
	friend class Layer;
	Tensor A_Prev;
	std::function<Tensor(const Tensor&)> func, dFunc;
	public:
		ActivationFunction(std::function<Tensor(const Tensor&)> f, std::function<Tensor(const Tensor&)> df);
		Tensor forward(const Tensor&);
		Tensor backward(const Tensor&);
		Tensor eval(const Tensor&) const;

};

class Sigmoid : public ActivationFunction{
	friend class Layer;
	public:
		inline Sigmoid()
			:ActivationFunction(
					[](const Tensor& x) -> Tensor
					{Tensor a = (-1) * x;
					a.exp_();
					a += 1;
					a.inverse_();
					return std::move(a);}, 
					[](const Tensor& dz) -> Tensor
					{return dz * (1-dz);})
		{}
};

class ReLU : public ActivationFunction{
	friend class Layer;
	public:
		inline ReLU()
			:ActivationFunction(
					[](const Tensor& x) -> Tensor{
						Tensor a = x.contiguous();
						a[a < 0] = 0;
						return std::move(a);
					},
					[](const Tensor& dz) -> Tensor{
						Tensor check = dz > 0;
						Tensor a = dz.contiguous();
						a[check] = 1;
						a[check == false] = 0;
						return std::move(a);
					}){}
};


class Unfold{
	friend class Layer;
	utils::my_tuple kernel_size, dilation, padding, stride;
	uint32_t LKern;
	bool out_transpose;
	public:
		Unfold(utils::my_tuple kernel_size, 
			utils::my_tuple dilation = 1,
			utils::my_tuple padding = 0,
			utils::my_tuple stride = 1,
			bool transpose_out=true);
		Tensor forward(const Tensor&);
		Tensor backward(const Tensor&);
		Tensor eval(const Tensor&) const;

};

/* class Conv2d{ */
/* 	friend class Layer; */
/* 	Tensor Bias, Kernel, APrev, Dilated, KernelF; */
/* 	uint32_t groups, s_h, s_w; */
/* 	bool use_bias, pad, use_dilation; */
/* 	std::vector<uint32_t> padding; */
/* 	void dilate_kernel(uint32_t dilation); */
/* 	SizeRef get_kernel_size(uint32_t, uint32_t, uint32_t, std::variant<uint32_t, std::tuple<uint32_t, uint32_t>>&); */
/* 	void set_stride(std::variant<uint32_t, std::tuple<uint32_t, uint32_t>>&); */
/* 	std::vector<uint32_t> get_padding(std::variant<uint32_t, std::tuple<uint32_t, uint32_t>, std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>>&); */
/* 	public: */
/* 		Conv2D(uint32_t in_channels, uint32_t out_channels, */ 
/* 				std::variant<uint32_t, std::tuple<uint32_t, uint32_t>> kernel_size, */ 
/* 				std::variant<uint32_t, std::tuple<uint32_t, uint32_t>> stride=1, */ 
/* 				std::variant<uint32_t, std::tuple<uint32_t, uint32_t>, std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> padding=0, */ 
/* 				uint32_t dilation=1, uint32_t groups=1, bool bias = true, DType dtype = DType::Float32); */
/* 		Tensor forward(const Tensor&); */
/* 		Tensor backward(const Tensor&); */
/* 		Tensor eval(const Tensor&) const; */
/* }; */


template<typename T>
constexpr bool is_layer = std::is_same_v<T, Linear> 
				|| std::is_same_v<T, Identity>
				|| std::is_same_v<T, ActivationFunction>
				|| std::is_same_v<T, Sigmoid>
				|| std::is_same_v<T, ReLU>
				|| std::is_same_v<T, Unfold>;


class Layer{
	std::variant<Identity, Linear, ActivationFunction, Sigmoid> l;
	public:
		template<typename T, std::enable_if_t<is_layer<T>, bool> = true>
		inline Layer(T layer)
		:l(layer)
		{}
		Layer();
		Tensor forward(const Tensor& x);
		Tensor backward(const Tensor& dZ);
		Tensor eval(const Tensor& x) const;
		uint32_t parameter_count() const;
		Tensor parameters();
};


class Sequential{
	std::vector<Layer> layers;
	std::vector<Layer> to_vec(){return {};}
	template<typename... Args>
	std::vector<Layer> to_vec(Layer first, Args... rest){
		std::vector<Layer> result = {first};
		auto tail = to_vec(rest...);
		result.insert(result.end(), tail.begin(), tail.end());
		return result;
	}
	public:
		template<typename... ls>
		Sequential(Layer first, ls... for_layer)
		:layers(to_vec(first, for_layer...))
		{}
		inline Tensor forward(const Tensor& x){
			Tensor inp = layers[0].forward(x);
			for(uint32_t i = 1; i < layers.size(); ++i)
				inp = layers[i].forward(inp);
			return inp;
		}
		inline Tensor backward(const Tensor& dZ){
			Tensor inp = layers.back().backward(dZ);
			for(int32_t i = layers.size()-2; i >= 0; --i)
				inp = layers[i].backward(inp);
			return inp;
		}
		inline Tensor eval(const Tensor& x) const{
			Tensor inp = layers[0].eval(x);
			for(uint32_t i = 1; i < layers.size(); ++i)
				inp = layers[i].eval(inp);
			return inp;	
		}
		inline Layer& operator[](uint32_t i){return layers[i];}
		inline const std::size_t size() const {return layers.size();}

		Tensor parameters(){
			uint32_t count = 0;
			for(uint32_t i = 0; i < layers.size(); ++i)
				count += layers[i].parameter_count();
			Tensor outp({count}, DType::TensorObj);

			uint32_t current = 0;
			for(uint32_t i = 0; i < layers.size(); ++i){
				if(layers[i].parameter_count() > 0){
					Tensor params = layers[i].parameters();
					for(uint32_t j = 0; j < params.shape()[0]; ++j, ++current){
						outp[current] = params[j];
					}
				}

			}
			return outp;
		}

};

}
}
