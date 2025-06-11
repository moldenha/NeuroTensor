//this is just a wrapper for when a tensor needs to be held by an intrusive_ptr

#include "../Tensor.h"
#ifndef _NT_INTRUSIVE_TENSOR_HOLDER_H_
#define _NT_INTRUSIVE_TENSOR_HOLDER_H_

namespace nt{

class tensor_holder : public intrusive_ptr_target{
	public:
		Tensor tensor;
		explicit tensor_holder(const Tensor& t) : tensor(t) {}
		explicit tensor_holder(Tensor&& t) : tensor(t) {}
};

}

#endif //_NT_INTRUSIVE_TENSOR_HOLDER_H_
