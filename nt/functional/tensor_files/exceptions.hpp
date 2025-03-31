#ifndef __NT_FUNCTIONAL_TENSOR_FILES_EXCEPTIONS_HPP__
#define __NT_FUNCTIONAL_TENSOR_FILES_EXCEPTIONS_HPP__
#include "../../Tensor.h"

namespace nt{
namespace functional{

inline void exception_dtypes(const DType& a, const DType& b){
	utils::THROW_EXCEPTION(a == b, "\nRuntimeError: Expected dtype of second tensor to be $ but got $", a, b);
}

inline void exception_shapes(const SizeRef& a, const SizeRef& b, bool singletons=false){
	if(!singletons && a != b){
		utils::THROW_EXCEPTION(a == b, "\nRuntimeError: Expected shape of second tensor to be $ but got $", a, b);
	}
	if(a != b){
		if(a.size() > b.size()){
			typename SizeRef::value_type start = a.size() - b.size();
			for(typename SizeRef::value_type i = a.size() - b.size(); i < a.size(); ++i){
				if(a[i] != b[i - start] && (b[i - start] != 1 || a[i] != 1)){
					utils::THROW_EXCEPTION(b[i - start] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i], b[i - start], i);
					utils::THROW_EXCEPTION(a[i] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i], b[i - start], i);
				}
			}
		}
		else if(b.size() > a.size()){
			typename SizeRef::value_type start = b.size() - a.size();
			for(typename SizeRef::value_type i = b.size() - a.size(); i < b.size(); ++i){
				if(a[i - start] != b[i] && (b[i] != 1 || a[i - start] != 1)){
					utils::THROW_EXCEPTION(b[i] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i - start], b[i], i);
					utils::THROW_EXCEPTION(a[i - start] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i - start], b[i], i);

				}
			}
		}
		else{
			for(typename SizeRef::value_type i = 0; i < b.size(); ++i){
				if(a[i] != b[i] && (b[i] != 1 || a[i] != 1)){
					utils::THROW_EXCEPTION(b[i] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i], b[i], i);
					utils::THROW_EXCEPTION(a[i] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i], b[i], i);

				}
			}
		}

	}
}

}
}

#endif
