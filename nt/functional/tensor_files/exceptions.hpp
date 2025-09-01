#ifndef NT_FUNCTIONAL_TENSOR_FILES_EXCEPTIONS_HPP__
#define NT_FUNCTIONAL_TENSOR_FILES_EXCEPTIONS_HPP__
#include "../../Tensor.h"
#include "../../utils/name_func_macro.h"

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
            typename SizeRef::value_type max = static_cast<typename SizeRef::value_type>(a.size()); 
			for(typename SizeRef::value_type i = a.size() - b.size(); i < max; ++i){
				if(a[i] != b[i - start] && (b[i - start] != 1 || a[i] != 1)){
					utils::THROW_EXCEPTION(b[i - start] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i], b[i - start], i);
					utils::THROW_EXCEPTION(a[i] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i], b[i - start], i);
				}
			}
		}
		else if(b.size() > a.size()){
			typename SizeRef::value_type start = b.size() - a.size();
            typename SizeRef::value_type max = static_cast<typename SizeRef::value_type>(b.size()); 
			for(typename SizeRef::value_type i = b.size() - a.size(); i < max; ++i){
				if(a[i - start] != b[i] && (b[i] != 1 || a[i - start] != 1)){
					utils::THROW_EXCEPTION(b[i] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i - start], b[i], i);
					utils::THROW_EXCEPTION(a[i - start] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i - start], b[i], i);

				}
			}
		}
		else{
            typename SizeRef::value_type max = static_cast<typename SizeRef::value_type>(b.size()); 
			for(typename SizeRef::value_type i = 0; i < max; ++i){
				if(a[i] != b[i] && (b[i] != 1 || a[i] != 1)){
					utils::THROW_EXCEPTION(b[i] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i], b[i], i);
					utils::THROW_EXCEPTION(a[i] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $", a[i], b[i], i);

				}
			}
		}

	}
}

inline void nt_functional_always_check(const char* func_name, const Tensor& t){
    utils::throw_exception(!t.is_null(),
                           "Cannot perform function $ on a null tensor", func_name);
}

template<typename... T>
inline void nt_functional_always_check(const char* func_name, const Tensor& t, const T&... tensors){
    nt_functional_always_check(func_name, t);
    nt_functional_always_check(func_name, tensors...);
}



#define _NT_FUNCTIONAL_ALWAYS_CHECK_(...) nt_functional_always_check(__func__, __VA_ARGS__)

inline void check_mutability(Tensor& x, const char* func_name = __NT_FUNCTION_NAME__){
    utils::throw_exception(x.is_mutable(), "Cannot perform function $ on an immutable tensor", func_name);
}

}
}

#endif
