#ifndef NT_LAYER_OPTIMIZERS_H__
#define NT_LAYER_OPTIMIZERS_H__

#include "TensorGrad.h"
#include "Layer.h"
#include "layers.h"

namespace nt{
namespace optimizers{


class NEUROTENSOR_API SGD{
	reflect::detail::custom_typed_iterator<TensorGrad> parameters;
	double learning_rate;
	//this function basically erases all gradients to be tracked and everything
	void erase_grad_tracking();
	public:
		SGD(reflect::detail::custom_typed_iterator<TensorGrad> params, double lr=0.01)
			:parameters(params),
			learning_rate(lr)
		{}

		void step();
		void zero_grad();
};

class NEUROTENSOR_API Adam{
	reflect::detail::custom_typed_iterator<TensorGrad> parameters;
	double learning_rate, beta1, beta2, epsilon;
	Tensor m, v;
	int64_t t;
	//this function basically erases all gradients to be tracked and everything
	void erase_grad_tracking();
	public:
		Adam(reflect::detail::custom_typed_iterator<TensorGrad> params, double lr=0.001, double beta1=0.9, double beta2=0.999, double epsilon=1e-8)
			:parameters(params),
			learning_rate(lr),
			beta1(beta1),
			beta2(beta2),
			epsilon(epsilon),
			m(Tensor::makeNullTensorArray(params.size())),
			v(Tensor::makeNullTensorArray(params.size())),
			t(0)
		{
			Tensor* begin1 = reinterpret_cast<Tensor*>(m.data_ptr());
			Tensor* begin2 = reinterpret_cast<Tensor*>(v.data_ptr());
			Tensor* end1 = begin1 + params.size();
			auto begin = parameters.begin();
			for(;begin1 != end1; ++begin1, ++begin2, ++begin){
				if(begin->is_null()){continue;}
				*begin1 = functional::zeros_like(begin->detach());
				*begin2 = functional::zeros_like(begin->detach());
			}
		}

		void step();
		void zero_grad();
};

}}


#endif //_NT_LAYER_OPTIMIZERS_H_
