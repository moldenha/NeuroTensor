#ifndef _NT_MODULE_H_
#define _NT_MODULE_H_
#include "../Tensor.h"
#include "TensorGrad.h"
#include <unordered_map>
#include <string>
#include <functional>
#include <map>
#include <memory>


namespace nt{

class Module : public intrusive_ptr_target{
	public:
		Module() = default;
		inline virtual TensorGrad forward(const TensorGrad& x){return x;}
		inline virtual void backward(const Tensor& dx, intrusive_ptr<TensorGrad> parent){;}
		//example use:
		//{parent->grad->tensor += dx;return;}
		inline virtual Tensor eval(const Tensor& x){return x;}
};

}
#endif
