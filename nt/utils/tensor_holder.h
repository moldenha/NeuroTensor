//this is just a wrapper for when a tensor needs to be held by an intrusive_ptr
#ifndef NT_INTRUSIVE_TENSOR_HOLDER_H__
#define NT_INTRUSIVE_TENSOR_HOLDER_H__

#include "../Tensor.h"
#include "api_macro.h"

namespace nt{

class NEUROTENSOR_API tensor_holder : public intrusive_ptr_target{
	public:
		Tensor tensor;
		explicit tensor_holder(const Tensor& t) : tensor(t) {}
		explicit tensor_holder(Tensor&& t) : tensor(t) {}
        inline void release_resources() override {;}
};

}

#endif //NT_INTRUSIVE_TENSOR_HOLDER_H__
