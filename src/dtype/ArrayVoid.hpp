#ifndef _ARRAY_VOID_HPP_
#define _ARRAY_VOID_HPP_

#include "ArrayVoid.h"
#include "DType.h"
#include "../Tensor.h"
#include "DType_enum.h"
#include "DType_list.h"
#include "compatible/DType_compatible_all.h"
#include <_types/_uint8_t.h>
#include <regex>
#include <stdexcept>
#include <type_traits>

#ifdef USE_PARALLEL
	#include <tbb/parallel_for.h>
	#include <tbb/parallel_for_each.h>
#endif

namespace nt{

//this is to make it so it is not ambiguous with calling execute_function(func, args..);
//the above is for all dtypes, this is for specific dtypes
template<DType dt, DType... dts, typename UnaryFunction, typename... Args>
inline auto ArrayVoid::execute_function(UnaryFunction&& unary_op, Args&&... args){
	using val_type = std::invoke_result_t<UnaryFunction, tdtype_list<DTypeFuncs::dtype_to_type_t<dt> >, tdtype_list<DTypeFuncs::dtype_to_type_t<dt>>, Args...>;
	bool check = dtype == dt || DTypeFuncs::is_in<dts...>(dtype);
	if(!check){
		if constexpr(std::is_same_v<val_type, void>){
			return;
		}
		else{
			val_type outp;
			return outp; 
		}
	}
	check = false;
	if(dtype == dt){
		if constexpr (std::is_same_v<val_type, void>){
			sub_handle_execute_function_void<dt>(std::forward<UnaryFunction&&>(unary_op), check, std::forward<Args&&>(args)...);
			return;
		}
		else{
			val_type outp;
			sub_handle_execute_function<dt>(std::forward<UnaryFunction&&>(unary_op), outp, check, std::forward<Args&&>(args)...);
			return std::move(outp);
		}
	}
	if constexpr(std::is_same_v<val_type, void>){
		((sub_handle_execute_function<dts>(std::forward<UnaryFunction&&>(unary_op), check, std::forward<Args&&>(args)...)), ...);
		return;
	}
	else{
		val_type outp;
		((sub_handle_execute_function<dts>(std::forward<UnaryFunction&&>(unary_op), outp, check, std::forward<Args&&>(args)...)), ...);	
		return std::move(outp);
	}
}

template<typename WrappedTypes, typename UnaryFunction, typename... Args, std::enable_if_t<
				DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value
				&& !DTypeFuncs::is_wrapped_dtype<UnaryFunction>::value, bool>>
inline auto ArrayVoid::execute_function(UnaryFunction&& unary_op, Args&&... args){
	constexpr DType m_dtype = WrappedTypes::next;
	if(m_dtype != dtype && WrappedTypes::done){
		using val_type = std::invoke_result_t<UnaryFunction&&, tdtype_list<DTypeFuncs::dtype_to_type_t<m_dtype> >, tdtype_list<DTypeFuncs::dtype_to_type_t<m_dtype>>, Args...>;
		if constexpr(std::is_same_v<val_type, void>)
			return;
		else{
			val_type outp;
			return outp;
		}
	}
	if(m_dtype != dtype){
		return execute_function<typename WrappedTypes::next_wrapper>(std::forward<UnaryFunction&&>(unary_op), std::forward<Args&&>(args)...);
	}
	bool check = false;
	using val_type = std::invoke_result_t<UnaryFunction, tdtype_list<DTypeFuncs::dtype_to_type_t<m_dtype> >, tdtype_list<DTypeFuncs::dtype_to_type_t<m_dtype> >, Args...>;
	if constexpr(std::is_same_v<val_type, void>){
		sub_handle_execute_function_void<m_dtype>(std::forward<UnaryFunction&&>(unary_op), check, std::forward<Args&&>(args)...);
		return;
	}
	else{
		val_type outp;
		sub_handle_execute_function<m_dtype>(std::forward<UnaryFunction&&>(unary_op), outp, check, std::forward<Args&&>(args)...);
		return std::move(outp);
	}
}

template<DType dt, DType... dts, typename UnaryFunction, typename... Args>
inline auto ArrayVoid::execute_function(UnaryFunction&& unary_op, ArrayVoid& inp_arr, Args&&... args){
	using val_type = std::invoke_result_t<UnaryFunction, 
	      tdtype_list<DTypeFuncs::dtype_to_type_t<dt, dts...> >, 
	      tdtype_list<DTypeFuncs::dtype_to_type_t<dt, dts...>>, 
	      tdtype_list<DTypeFuncs::dtype_to_type_t<dt, dts...>>, Args...>;
	bool check = dtype == dt || DTypeFuncs::is_in<dts...>(dtype);
	if(!check){
		if constexpr(std::is_same_v<val_type, void>){
			return;
		}
		else{
			val_type outp;
			return outp; 
		}
	}
	check = false;
	if(dtype == dt){
		if constexpr (std::is_same_v<val_type, void>){
			sub_handle_execute_function_void<dt>(std::forward<UnaryFunction&&>(unary_op), check, inp_arr, std::forward<Args&&>(args)...);
			return;
		}
		else{
			val_type outp;
			sub_handle_execute_function<dt>(std::forward<UnaryFunction&&>(unary_op), outp, check, inp_arr, std::forward<Args&&>(args)...);
			return std::move(outp);
		}
	}
	if constexpr(std::is_same_v<val_type, void>){
		((sub_handle_execute_function<dts>(std::forward<UnaryFunction&&>(unary_op), check, inp_arr, std::forward<Args&&>(args)...)), ...);
		return;
	}
	else{
		val_type outp;
		((sub_handle_execute_function<dts>(std::forward<UnaryFunction&&>(unary_op), outp, check, inp_arr, std::forward<Args&&>(args)...)), ...);	
		return std::move(outp);
	}
}


template<typename WrappedTypes, typename UnaryFunction, typename... Args>
inline auto ArrayVoid::execute_function(UnaryFunction&& unary_op, ArrayVoid& inp_arr, Args&&... args){
	constexpr DType m_dtype = WrappedTypes::next;
	if(m_dtype != dtype && WrappedTypes::done){
		using val_type = std::invoke_result_t<UnaryFunction, 
		      tdtype_list<DTypeFuncs::dtype_to_type_t<m_dtype> >, 
		      tdtype_list<DTypeFuncs::dtype_to_type_t<m_dtype>>,
		      tdtype_list<DTypeFuncs::dtype_to_type_t<m_dtype>>, Args...>;
		if constexpr(std::is_same_v<val_type, void>)
			return;
		else{
			val_type outp;
			return outp;
		}
	}
	if(m_dtype != dtype){
		return execute_function<typename WrappedTypes::next_wrapper>(std::forward<UnaryFunction&&>(unary_op), inp_arr, std::forward<Args&&>(args)...);
	}
	bool check = false;
	using val_type = std::invoke_result_t<UnaryFunction, tdtype_list<DTypeFuncs::dtype_to_type_t<m_dtype> >, tdtype_list<DTypeFuncs::dtype_to_type_t<m_dtype>>,tdtype_list<DTypeFuncs::dtype_to_type_t<m_dtype>>, Args...>;
	if constexpr(std::is_same_v<val_type, void>){
		sub_handle_execute_function_void<m_dtype>(std::forward<UnaryFunction&&>(unary_op), check, inp_arr, args...);
		return;
	}
	else{
		val_type outp;
		sub_handle_execute_function<m_dtype>(std::forward<UnaryFunction&&>(unary_op), outp, check, inp_arr, args...);
		return std::move(outp);
	}
}

template<DType dt, DType...dts>
inline auto ArrayVoid::execute_function(bool throw_error, const char* func_name){
	bool check = throw_error ? DTypeFuncs::check_dtypes<dt, dts...>(func_name, dtype) : DTypeFuncs::is_in<dt, dts...>(dtype);
	return [throw_error, func_name, this](auto&& unary_op, auto&&... args) -> std::invoke_result_t<decltype(unary_op), tdtype_list<DTypeFuncs::dtype_to_type_t<dts...>>, tdtype_list<DTypeFuncs::dtype_to_type_t<dts...>>, decltype(args)...>{
			return this->execute_function<dt, dts...>(std::forward<decltype(unary_op)&&>(unary_op), std::forward<decltype(args)&&>(args)...);	
		};
}

template<typename WrappedTypes, std::enable_if_t<DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value, bool> = true>
void throw_wrapped_types_error(const DType& dt, const char* func_name){
	if(WrappedTypes::next == dt)
		return;
	if(WrappedTypes::done)
		utils::throw_exception(WrappedTypes::next == dt, "\nRuntime Error: Got unexpected dtype $ for function $", dt, func_name);
	throw_wrapped_types_error<typename WrappedTypes::next_wrapper>(dt, func_name);
}

template<typename WrappedTypes, std::enable_if_t<DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value, bool>>
inline auto ArrayVoid::execute_function(bool throw_error, const char* func_name){
	if(throw_error){throw_wrapped_types_error<WrappedTypes>(dtype, func_name);}
	return [throw_error, func_name, this](auto&& unary_op, auto&&... args){
			return execute_function<WrappedTypes>(std::forward<decltype(unary_op)&&>(unary_op), std::forward<decltype(args)&&>(args)...);	
		};
}

template<DType dt, typename UnaryFunction, typename Output, typename... Args>
inline void ArrayVoid::sub_handle_execute_function(UnaryFunction&& unary_op, Output& v, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	v = std::forward<UnaryFunction&&>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args&&>(args)...);
}

template<DType dt, typename UnaryFunction, typename Output, typename... Args>
inline void ArrayVoid::sub_handle_execute_function(UnaryFunction&& unary_op, Output& v, bool& called, ArrayVoid& inp_arr, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	v = std::forward<UnaryFunction&&>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(), std::forward<Args&&>(args)...);
}

template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_void(UnaryFunction&& unary_op, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	std::forward<UnaryFunction&&>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args&&>(args)...);
}

template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_void(UnaryFunction&& unary_op, bool& called, ArrayVoid& inp_arr, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	std::forward<UnaryFunction&&>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(), std::forward<Args&&>(args)...);
}



template<DType dt, DType... dts, typename UnaryFunction, typename... Args>
inline auto ArrayVoid::cexecute_function(UnaryFunction&& unary_op, Args&&... args) const{
	using val_type = std::invoke_result_t<UnaryFunction, tdtype_list<const DTypeFuncs::dtype_to_type_t<dts...> >, tdtype_list<const DTypeFuncs::dtype_to_type_t<dts...>>, Args...>;
	bool check = DTypeFuncs::is_in<dt, dts...>(dtype);
	if(!check){
		if constexpr(std::is_same_v<val_type, void>)
			return;
		else
			return val_type();
	}
	check = false;
	if(dt == dtype){
		if constexpr(std::is_same_v<val_type, void>){
			sub_handle_cexecute_function_void<dt>(std::forward<UnaryFunction&&>(unary_op), check, std::forward<Args&&>(args)...);
			return;
		}
		else{
			val_type outp;
			sub_handle_cexecute_function<dt>(std::forward<UnaryFunction&&>(unary_op), outp, check, std::forward<Args&&>(args)...);
			return std::move(outp);
		}
	}
	if constexpr(std::is_same_v<val_type, void>){
		((sub_handle_cexecute_function_void<dts>(std::forward<UnaryFunction&&>(unary_op), check, std::forward<Args&&>(args)...)), ...);
		return;
	}
	else{
		val_type outp;
		((sub_handle_cexecute_function<dts>(std::forward<UnaryFunction&&>(unary_op), outp, check, std::forward<Args&&>(args)...)), ...);	
		return std::move(outp);
	}
}

template<typename WrappedTypes, typename UnaryFunction, typename... Args, std::enable_if_t<
					DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value
					&& !DTypeFuncs::is_wrapped_dtype<UnaryFunction>::value, bool>>
inline auto ArrayVoid::cexecute_function(UnaryFunction&& unary_op, Args&&... args) const{
	constexpr DType m_dtype = WrappedTypes::next;
	if(m_dtype != dtype && WrappedTypes::done){
		using val_type = std::invoke_result_t<UnaryFunction, tdtype_list<const DTypeFuncs::dtype_to_type_t<m_dtype> >, tdtype_list<const DTypeFuncs::dtype_to_type_t<m_dtype>>, Args...>;
		if constexpr(std::is_same_v<val_type, void>){
			return;
		}
		else{
			val_type outp;
			return outp;
		}
	}
	if(m_dtype != dtype){
		return (cexecute_function<typename WrappedTypes::next_wrapper>(std::forward<UnaryFunction&&>(unary_op), std::forward<Args&&>(args)...));
	}
	bool check = false;
	using val_type = std::invoke_result_t<UnaryFunction, tdtype_list<const DTypeFuncs::dtype_to_type_t<m_dtype> >, tdtype_list<const DTypeFuncs::dtype_to_type_t<m_dtype>>, Args...>;

	if constexpr(std::is_same_v<val_type, void>){
		sub_handle_cexecute_function_void<m_dtype>(std::forward<UnaryFunction&&>(unary_op), check, std::forward<Args&&>(args)...);
		return;
	}
	else{
		val_type outp;
		sub_handle_cexecute_function<m_dtype>(std::forward<UnaryFunction&&>(unary_op), outp, check, std::forward<Args&&>(args)...);
		return std::move(outp);
	}
}

template<DType dt, DType... dts, typename UnaryFunction, typename... Args>
inline auto ArrayVoid::cexecute_function(UnaryFunction&& unary_op, const ArrayVoid& inp_arr, Args&&... args) const{
	using val_type = std::invoke_result_t<UnaryFunction, tdtype_list<const DTypeFuncs::dtype_to_type_t<dts...> >, tdtype_list<const DTypeFuncs::dtype_to_type_t<dts...> >, tdtype_list<const DTypeFuncs::dtype_to_type_t<dts...>>, Args...>;
	bool check = DTypeFuncs::is_in<dt, dts...>(dtype);
	if(!check){
		if constexpr(std::is_same_v<val_type, void>)
			return;
		else
			return val_type(); 
	}
	check = false;
	if(dt == dtype){
		if constexpr(std::is_same_v<val_type, void>){
			sub_handle_cexecute_function_void<dt>(std::forward<UnaryFunction&&>(unary_op), check, inp_arr, std::forward<Args&&>(args)...);
			return;
		}
		else{
			val_type outp;
			sub_handle_cexecute_function<dt>(std::forward<UnaryFunction&&>(unary_op), outp, check, inp_arr, std::forward<Args&&>(args)...);
			return std::move(outp);
		}
	}
	if constexpr(std::is_same_v<val_type, void>){
		((sub_handle_cexecute_function_void<dts>(std::forward<UnaryFunction&&>(unary_op), check, inp_arr, std::forward<Args&&>(args)...)), ...);
		return;
	}
	else{
		val_type outp;
		((sub_handle_cexecute_function<dts>(std::forward<UnaryFunction&&>(unary_op), outp, check, inp_arr, std::forward<Args&&>(args)...)), ...);	
		return std::move(outp);
	}
}

template<typename WrappedTypes, typename UnaryFunction, typename... Args>
inline auto ArrayVoid::cexecute_function(UnaryFunction&& unary_op, const ArrayVoid& inp_arr, Args&&... args) const{
	constexpr DType m_dtype = WrappedTypes::next;
	if(m_dtype != dtype && WrappedTypes::done){
		using val_type = std::invoke_result_t<UnaryFunction, tdtype_list<const DTypeFuncs::dtype_to_type_t<m_dtype> >, tdtype_list<const DTypeFuncs::dtype_to_type_t<m_dtype>>, tdtype_list<const DTypeFuncs::dtype_to_type_t<m_dtype> >, Args...>;
		if constexpr(std::is_same_v<val_type, void>)
			return;
		else{
			val_type outp;
			return outp;
		}
	}
	if(m_dtype != dtype){
		return cexecute_function<typename WrappedTypes::next_wrapper>(std::forward<UnaryFunction&&>(unary_op), inp_arr, std::forward<Args&&>(args)...);
	}
	bool check = false;
	using val_type = std::invoke_result_t<UnaryFunction, tdtype_list<const DTypeFuncs::dtype_to_type_t<m_dtype> >, tdtype_list<const DTypeFuncs::dtype_to_type_t<m_dtype>>, tdtype_list<const DTypeFuncs::dtype_to_type_t<m_dtype> >, Args...>;
	if constexpr(std::is_same_v<val_type, void>){
		sub_handle_cexecute_function_void<m_dtype>(std::forward<UnaryFunction&&>(unary_op), check, inp_arr, std::forward<Args&&>(args)...);
	}
	else{
		val_type outp;
		sub_handle_cexecute_function<m_dtype>(std::forward<UnaryFunction&&>(unary_op), outp, check, inp_arr, std::forward<Args&&>(args)...);
		return std::move(outp);
	}
}


template<DType dt, DType...dts>
inline auto ArrayVoid::cexecute_function(bool throw_error, const char* func_name) const{
	bool check = throw_error ? DTypeFuncs::check_dtypes<dt, dts...>(func_name, dtype) : DTypeFuncs::is_in<dt, dts...>(dtype);
	return [throw_error, func_name, this](auto&& unary_op, auto&&... args) -> std::invoke_result_t<decltype(unary_op), tdtype_list<const DTypeFuncs::dtype_to_type_t<dts...>>, tdtype_list<const DTypeFuncs::dtype_to_type_t<dts...>>, decltype(args)...>{
			return this->cexecute_function<dt, dts...>(std::forward<decltype(unary_op)&&>(unary_op), std::forward<decltype(args)&&>(args)...);	
		};
}




template<typename WrappedTypes, std::enable_if_t<DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value, bool>>
inline auto ArrayVoid::cexecute_function(bool throw_error, const char* func_name) const{
	bool check = DTypeFuncs::is_in<WrappedTypes>(dtype);
	if(!check){
		if(throw_error)
			throw std::runtime_error("Unexpected dtype for current function");
	}

	return [this](auto&& unary_op, auto&&... args){
			return cexecute_function<WrappedTypes>(std::forward<decltype(unary_op)&&>(unary_op), std::forward<decltype(args)&&>(args)...);
			
		};
}


template<DType dt, typename UnaryFunction, typename Output, typename... Args>
inline void ArrayVoid::sub_handle_cexecute_function(UnaryFunction&& unary_op, Output& v, bool& called, Args&&... args) const{
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	v = std::forward<UnaryFunction&&>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args&&>(args)...);
}

template<DType dt, typename UnaryFunction, typename Output, typename... Args>
inline void ArrayVoid::sub_handle_cexecute_function(UnaryFunction&& unary_op, Output& v, bool& called, const ArrayVoid& inp_arr, Args&&... args) const{
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	v = std::forward<UnaryFunction&&>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), std::forward<Args&&>(args)...);
}

template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_cexecute_function_void(UnaryFunction&& unary_op, bool& called, Args&&... args) const{
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	std::forward<UnaryFunction&&>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args&&>(args)...);
}

template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_cexecute_function_void(UnaryFunction&& unary_op, bool& called, const ArrayVoid& inp_arr, Args&&... args) const{
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	std::forward<UnaryFunction&&>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), std::forward<Args&&>(args)...);
}

template<class UnaryOperator, class... Args>
inline auto ArrayVoid::execute_function(UnaryOperator&& unary_op, Args&&... arg){
	return execute_function<WRAP_DTYPES<AllTypesL>>(std::forward<UnaryOperator&&>(unary_op), std::forward<Args&&>(arg)...);
}
template<class UnaryOperator, class... Args>
inline auto ArrayVoid::cexecute_function(UnaryOperator&& unary_op, Args&&... arg) const{
	return cexecute_function<WRAP_DTYPES<AllTypesL>>(std::forward<UnaryOperator&&>(unary_op), std::forward<Args&&>(arg)...);
} 
template<class UnaryOperator, class... Args>
auto ArrayVoid::execute_function(UnaryOperator&& unary_op, ArrayVoid& inp_arr, Args&&... arg){
	return execute_function<WRAP_DTYPES<AllTypesL>>(std::forward<UnaryOperator&&>(unary_op), inp_arr, std::forward<Args&&>(arg)...);
} 
template<class UnaryOperator, class... Args>
inline auto ArrayVoid::cexecute_function(UnaryOperator&& unary_op, const ArrayVoid& inp_arr, Args&&... arg) const{
	return cexecute_function<WRAP_DTYPES<AllTypesL>>(std::forward<UnaryOperator&&>(unary_op), inp_arr, std::forward<Args&&>(arg)...);
}  

template<class UnaryOperator, class... Args>
inline auto ArrayVoid::execute_function_nbool(UnaryOperator&& unary_op, Args&&... arg){
	return execute_function<WRAP_DTYPES<AllTypesNBoolL>>(std::forward<UnaryOperator&&>(unary_op), std::forward<Args&&>(arg)...);
}
template<class UnaryOperator, class... Args>
inline auto ArrayVoid::cexecute_function_nbool(UnaryOperator&& unary_op, Args&&... arg) const{
	return cexecute_function<WRAP_DTYPES<AllTypesNBoolL>>(std::forward<UnaryOperator&&>(unary_op), std::forward<Args&&>(arg)...);
} 
template<class UnaryOperator, class... Args>
inline auto ArrayVoid::execute_function_nbool(UnaryOperator&& unary_op, ArrayVoid& inp_arr, Args&&... arg){
	return execute_function<WRAP_DTYPES<AllTypesNBoolL>>(std::forward<UnaryOperator&&>(unary_op), inp_arr, std::forward<Args&&>(arg)...);
} 
template<class UnaryOperator, class... Args>
inline auto ArrayVoid::cexecute_function_nbool(UnaryOperator&& unary_op, const ArrayVoid& inp_arr, Args&&... arg) const{
	return cexecute_function<WRAP_DTYPES<AllTypesNBoolL>>(std::forward<UnaryOperator&&>(unary_op), inp_arr, std::forward<Args&&>(arg)...);
}  


template<DType... dts, class OutputIt, class UnaryOperator, std::enable_if_t<!std::is_same_v<OutputIt, bool>, bool>>
inline OutputIt ArrayVoid::transform_function(UnaryOperator&& unary_op, OutputIt d_first, bool throw_error, const char* str) const{
	bool check = throw_error ? DTypeFuncs::check_dtypes<dts...>("Transform", dtype) : DTypeFuncs::is_in<dts...>(dtype);
	if(!check){
		if(throw_error)
			throw std::runtime_error("Unexpected dtype for current function");
		return d_first; 
	}
	/* return transform_function<WRAP_DTYPES<DTypeEnum<dts...>>(std::forward<UnaryOperator&&>(unary_op), d_first, , str); */


	((sub_transform_function<dts>(std::forward<UnaryOperator&&>(unary_op), d_first)), ...);
	return d_first;
}


template<typename WrappedTypes, class OutputIt, class UnaryOperation, std::enable_if_t<
		DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value
		&& !DTypeFuncs::is_wrapped_dtype<UnaryOperation>::value
		&& !DTypeFuncs::is_wrapped_dtype<OutputIt>::value, bool>>
inline OutputIt ArrayVoid::transform_function(UnaryOperation&& unary_op, OutputIt d_first, bool throw_error, const char* str) const{
	if(WrappedTypes::next != dtype && !WrappedTypes::done)
		return transform_function<typename WrappedTypes::next_wrapper>(std::forward<UnaryOperation&&>(unary_op), d_first, throw_error, str);
	bool check = (dtype == WrappedTypes::next);
	if(!check){
		if(throw_error)
			throw std::runtime_error("Unexpected dtype for current function");
		return d_first; 
	}
	sub_transform_function<WrappedTypes::next>(std::forward<UnaryOperation&&>(unary_op), d_first);
	return d_first;
}


template<class OutputIt, class UnaryOperation>
inline OutputIt ArrayVoid::transform_function(UnaryOperation&& unary_op, OutputIt d_first) const{
	return transform_function<WRAP_DTYPES<AllTypesL>>(std::forward<UnaryOperation&&>(unary_op), d_first, false);
}

template<class OutputIt, class UnaryOperation>
inline OutputIt ArrayVoid::transform_function_nbool(UnaryOperation&& unary_op, OutputIt d_first) const{
	return transform_function<WRAP_DTYPES<AllTypesNBoolL>>(std::forward<UnaryOperation&&>(unary_op), d_first, false);
}

template<DType... dts, class InputIt2, class OutputIt, class UnaryOperation, 
			std::enable_if_t<!std::is_same_v<OutputIt, bool> 
				&& !std::is_same_v<OutputIt, const char*>
				&& !std::is_same_v<InputIt2, bool> 
				&& !std::is_same_v<InputIt2, const char*>, bool>>
inline OutputIt ArrayVoid::transform_function(UnaryOperation&& unary_op, InputIt2 inp2, OutputIt d_first, bool throw_error, const char* str) const{
	bool check = throw_error ? DTypeFuncs::check_dtypes<dts...>(str, dtype) : DTypeFuncs::is_in<dts...>(dtype);
	if(!check){
		if(throw_error)
			throw std::runtime_error("Unexpected dtype for current function");
		return d_first; 
	}
	((sub_transform_function<dts...>(std::forward<UnaryOperation&&>(unary_op), inp2, d_first)));
	return d_first;

}

template<typename WrappedTypes, class InputIt2, class OutputIt, class UnaryOperation, std::enable_if_t<
			DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value
				&& !std::is_same_v<OutputIt, bool> 
				&& !std::is_same_v<OutputIt, const char*>
				&& !std::is_same_v<InputIt2, bool> 
				&& !std::is_same_v<InputIt2, const char*>
			, bool>>	
inline OutputIt ArrayVoid::transform_function(UnaryOperation&& unary_op, InputIt2 inp2, OutputIt d_first, bool throw_error, const char* str) const{
	if(WrappedTypes::next != dtype && !WrappedTypes::done)
		return transform_function<typename WrappedTypes::next_wrapper>(std::forward<UnaryOperation&&>(unary_op), inp2, d_first, throw_error, str);
	bool check = (dtype == WrappedTypes::next);
	if(!check){
		if(throw_error)
			throw std::runtime_error("Unexpected dtype for current function");
		return d_first; 
	}
	sub_transform_function<WrappedTypes::next>(std::forward<UnaryOperation&&>(unary_op), inp2, d_first);
	return d_first;
}


template<class InputIt2, class OutputIt, class UnaryOperation>
inline OutputIt ArrayVoid::transform_function(UnaryOperation&& unary_op, InputIt2 inp2, OutputIt d_first) const{
	return transform_function<WRAP_DTYPES<AllTypesL>>(std::forward<UnaryOperation&&>(unary_op), inp2, d_first, false);
}

template<class InputIt2, class OutputIt, class UnaryOperation>
inline OutputIt ArrayVoid::transform_function_nbool(UnaryOperation&& unary_op, InputIt2 inp2, OutputIt d_first) const{
	return transform_function<WRAP_DTYPES<AllTypesNBoolL>>(std::forward<UnaryOperation&&>(unary_op), inp2, d_first, false);
}


template<DType... dts, class OutputIt, class UnaryOperator, std::enable_if_t<!std::is_same_v<OutputIt, bool>, bool>>
inline OutputIt ArrayVoid::transform_function(UnaryOperator&& unary_op, const ArrayVoid& inp_arr, OutputIt d_first, bool throw_error, const char* str) const{
	bool check = throw_error ? DTypeFuncs::check_dtypes<dts...>(str, dtype) : DTypeFuncs::is_in<dts...>(dtype);
	if(!check){
		if(throw_error)
			throw std::runtime_error("Unexpected dtype for current function");
		return d_first; 
	}
	sub_transform_function<dts...>(std::forward<UnaryOperator&&>(unary_op), inp_arr, d_first);
	return d_first;
}

template<typename WrappedTypes, class OutputIt, class UnaryOperation, std::enable_if_t<
			DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value && !std::is_same_v<OutputIt, bool>, bool>>
inline OutputIt ArrayVoid::transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, OutputIt d_first, bool throw_error, const char* str) const{
	if(WrappedTypes::next != dtype && !WrappedTypes::done)
		return transform_function<typename WrappedTypes::next_wrapper>(std::forward<UnaryOperation&&>(unary_op), inp_arr, d_first, throw_error, str);
	bool check = (dtype == WrappedTypes::next);
	if(!check){
		if(throw_error)
			throw std::runtime_error("Unexpected dtype for current function");
		return d_first; 
	}
	sub_transform_function<WrappedTypes::next>(std::forward<UnaryOperation&&>(unary_op), inp_arr, d_first);
	return d_first;
}


template<class OutputIt, class UnaryOperation>
inline OutputIt ArrayVoid::transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, OutputIt d_first) const{
	return transform_function<WRAP_DTYPES<AllTypesL>>(std::forward<UnaryOperation&&>(unary_op), inp_arr, d_first, false);
}

template<class OutputIt, class UnaryOperation>
inline OutputIt ArrayVoid::transform_function_nbool(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, OutputIt d_first) const{
	return transform_function<WRAP_DTYPES<AllTypesNBoolL>>(std::forward<UnaryOperation&&>(unary_op), inp_arr, d_first, false);
}


template<DType dt, DType... dts, class UnaryOperator>
inline void ArrayVoid::transform_function(UnaryOperator&& unary_op, const ArrayVoid& inp_arr, bool throw_error, const char* str){
	bool check = throw_error ? DTypeFuncs::check_dtypes<dt, dts...>(str, dtype) : DTypeFuncs::is_in<dt, dts...>(dtype);
	if(!check){
		if(throw_error)
			throw std::runtime_error("Unexpected dtype for current function");
		return; 
	}
	/* transform_function<WRAP_DTYPES<DTypeEnum<dts...>>(std::forward<UnaryOperator&&>(unary_op), d_first, , str); */
	if(dt == dtype){
		sub_transform_function<dt>(std::forward<UnaryOperator&&>(unary_op), inp_arr);
		return;
	}
	((sub_transform_function<dts>(std::forward<UnaryOperator&&>(unary_op), inp_arr)), ...);
}


template<typename WrappedTypes, class UnaryOperation, std::enable_if_t< DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value, bool>>
inline void ArrayVoid::transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, bool throw_error, const char* str){
	if(WrappedTypes::next != dtype && !WrappedTypes::done){
		transform_function<typename WrappedTypes::next_wrapper>(std::forward<UnaryOperation&&>(unary_op), inp_arr, throw_error, str);
		return;
	}
	bool check = (dtype == WrappedTypes::next);
	if(!check){
		if(throw_error)
			throw std::runtime_error("Unexpected dtype for current function");
	}
	sub_transform_function<WrappedTypes::next>(std::forward<UnaryOperation&&>(unary_op), inp_arr);
}

template<class UnaryOperation>
inline void ArrayVoid::transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr){
	transform_function<WRAP_DTYPES<AllTypesL>>(std::forward<UnaryOperation&&>(unary_op), inp_arr, false);
}

template<class UnaryOperation>
inline void ArrayVoid::transform_function_nbool(UnaryOperation&& unary_op, const ArrayVoid& inp_arr){
	transform_function<WRAP_DTYPES<AllTypesNBoolL>>(std::forward<UnaryOperation&&>(unary_op), inp_arr, false);
}



/* template<class OutputIt, class UnaryOperation> */
/* inline void ArrayVoid::transform_function_nbool(UnaryOperation&& unary_op, const ArrayVoid& inp_arr){ */
/* 	transform_function<WRAP_DTYPES<AllTypesNBoolL>>(std::forward<UnaryOperation&&>(unary_op), inp_arr, false); */
/* } */


template<DType... dts, class UnaryOperation>
inline void ArrayVoid::for_ceach(UnaryOperation&& unary_op, bool throw_error, const char* str) const{
	bool check = throw_error ? DTypeFuncs::check_dtypes<dts...>(str, dtype) : DTypeFuncs::is_in<dts...>(dtype);
	if(!check){
		if(throw_error)
			throw std::runtime_error("Unexpected dtype for current function");
		return; 
	}
	(sub_for_ceach<dts...>(std::forward<UnaryOperation&&>(unary_op)));

}

template<DType... dts, class UnaryOperation>
inline void ArrayVoid::for_each(UnaryOperation&& unary_op, bool throw_error, const char* str){
	bool check = throw_error ? DTypeFuncs::check_dtypes<dts...>(str, dtype) : DTypeFuncs::is_in<dts...>(dtype);
	if(!check){
		if(throw_error)
			throw std::runtime_error("Unexpected dtype for current function");
		return; 
	}
	((sub_for_each<dts>(std::forward<UnaryOperation&&>(unary_op))), ...);

}


template<typename WrappedTypes, class UnaryOperation, std::enable_if_t< DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value, bool>>
void ArrayVoid::for_ceach(UnaryOperation&& unary_op, bool throw_error, const char* str) const{
	if(WrappedTypes::next != dtype && !WrappedTypes::done){
		for_ceach<typename WrappedTypes::next_wrapper>(std::forward<UnaryOperation&&>(unary_op), throw_error, str);
		return;
	}
	bool check = (dtype == WrappedTypes::next);
	if(!check){
		if(throw_error)
			throw std::runtime_error("Unexpected dtype for current function");
		return; 
	}
	sub_for_ceach<WrappedTypes::next>(std::forward<UnaryOperation&&>(unary_op)); 
}
template<typename WrappedTypes, class UnaryOperation, std::enable_if_t< DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value, bool>>
void ArrayVoid::for_each(UnaryOperation&& unary_op, bool throw_error, const char* str){
	if(WrappedTypes::next != dtype && !WrappedTypes::done){
		for_each<typename WrappedTypes::next_wrapper>(std::forward<UnaryOperation&&>(unary_op), throw_error, str);
		return;
	}
	bool check = (dtype == WrappedTypes::next);
	if(!check){
		if(throw_error)
			throw std::runtime_error("Unexpected dtype for current function");
		return; 
	}
	sub_for_each<WrappedTypes::next>(std::forward<UnaryOperation&&>(unary_op)); 
}

template<class UnaryOperation>
inline void ArrayVoid::for_ceach(UnaryOperation&& unary_op) const{
	for_ceach<WRAP_DTYPES<AllTypesL>>(std::forward<UnaryOperation&&>(unary_op));  
}

template<class UnaryOperation>
inline void ArrayVoid::for_each(UnaryOperation&& unary_op){
	for_each<WRAP_DTYPES<AllTypesL>>(std::forward<UnaryOperation&&>(unary_op));  

}

template<class UnaryOperation>
inline void ArrayVoid::for_ceach_nbool(UnaryOperation&& unary_op) const{
	for_ceach<WRAP_DTYPES<AllTypesNBoolL>>(std::forward<UnaryOperation&&>(unary_op));  
}

template<class UnaryOperation>
inline void ArrayVoid::for_each_nbool(UnaryOperation&& unary_op){
	for_each<WRAP_DTYPES<AllTypesNBoolL>>(std::forward<UnaryOperation&&>(unary_op));  

}

#ifndef USE_PARALLEL

template<DType dt, class OutputIt, class UnaryOperation>
void ArrayVoid::sub_transform_function(UnaryOperation&& unary_op, OutputIt& d_first) const{
	if(dt != dtype){return;}
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation&&>(unary_op));
}

template<DType dt, class InputIt2, class OutputIt, class UnaryOperation>
inline void ArrayVoid::sub_transform_function(UnaryOperation&& unary_op, InputIt2& inp2, OutputIt& d_first) const{
	if(dt != dtype){return;}
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, std::forward<UnaryOperation&&>(unary_op));
}


template<DType dt, class OutputIt, class UnaryOperation>
inline void ArrayVoid::sub_transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, OutputIt& d_first) const{
	if(dt != dtype){return;}
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, std::forward<UnaryOperation&&>(unary_op));
}

template<DType dt, class UnaryOperation>
inline void ArrayVoid::sub_transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr){
	if(dt != dtype){return;}
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	std::transform(tbegin<value_t>(), tend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), std::forward<UnaryOperation&&>(unary_op));
}

template<DType dt, class UnaryOperation>
inline void ArrayVoid::sub_for_ceach(UnaryOperation&& unary_op) const{
	if(dt != dtype){return;}
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation&&>(unary_op));
}

template<DType dt, class UnaryOperation>
inline void ArrayVoid::sub_for_each(UnaryOperation&& unary_op){
	if(dt != dtype){return;}
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation&&>(unary_op));
}



#else

template<DType dt, class OutputIt, class UnaryOperation>
inline void ArrayVoid::sub_transform_function(UnaryOperation&& unary_op, OutputIt& d_first) const{
	if(dt != dtype){return;}
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	tdtype_list<const value_t> start = tcbegin<value_t>();
	tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size),
			[&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation&&>(unary_op));});

}

template<DType dt, class InputIt2, class OutputIt, class UnaryOperation>
inline void ArrayVoid::sub_transform_function(UnaryOperation&& unary_op, InputIt2& inp2, OutputIt& d_first) const{
	if(dt != dtype){return;}
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	tdtype_list<const value_t> start = tcbegin<value_t>();
	tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size),
			[&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), std::forward<UnaryOperation&&>(unary_op));});
}


template<DType dt, class OutputIt, class UnaryOperation>
inline void ArrayVoid::sub_transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, OutputIt& d_first) const{
	if(dt != dtype){return;}
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	tdtype_list<const value_t> start = tcbegin<value_t>();
	tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>();
	tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size),
			[&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), std::forward<UnaryOperation&&>(unary_op));});
}

template<DType dt, class UnaryOperation>
inline void ArrayVoid::sub_transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr){
	if(dt != dtype){return;}
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	tdtype_list<const value_t> start = tcbegin<value_t>();
	tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>();
	tdtype_list<value_t> my_start = tbegin<value_t>();
	tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
}


template<DType dt, class UnaryOperation>
inline void ArrayVoid::sub_for_ceach(UnaryOperation&& unary_op) const{
	if(dt != dtype){return;}
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation&&>(unary_op));
}

template<DType dt, class UnaryOperation>
inline void ArrayVoid::sub_for_each(UnaryOperation&& unary_op){
	if(dt != dtype){return;}
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation&&>(unary_op));
}

#endif

}


#endif