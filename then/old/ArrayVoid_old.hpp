#ifndef _ARRAY_VOID_HPP_
#define _ARRAY_VOID_HPP_

#include "ArrayVoid.h"
#include "DType.h"
#include "../Tensor.h"
#include "DType_enum.h"
#include "DType_list.h"
#include <_types/_uint8_t.h>

#ifdef USE_PARALLEL
	#include <tbb/parallel_for.h>
	#include <tbb/parallel_for_each.h>
#endif

namespace nt{

template<DType... dts, typename UnaryFunction, typename... Args>
auto ArrayVoid::execute_function(UnaryFunction&& unary_op, Args&&... args){
	using val_type = std::invoke_result_t<UnaryFunction, tdtype_list<DTypeFuncs::dtype_to_type_t<dts...> >, tdtype_list<DTypeFuncs::dtype_to_type_t<dts...>>, Args...>;
	bool check = DTypeFuncs::is_in<dts...>(dtype);
	val_type outp;
	if(!check){
		return outp; 
	}
	check = false;
	((sub_handle_execute_function<dts>(std::forward<UnaryFunction>(unary_op), outp, check, args...)), ...);	
	return std::move(outp);
}

template<DType... dts, typename UnaryFunction, typename... Args>
auto ArrayVoid::execute_function(UnaryFunction&& unary_op, ArrayVoid& inp_arr, Args&&... args){
	using val_type = std::invoke_result_t<UnaryFunction, tdtype_list<DTypeFuncs::dtype_to_type_t<dts...> >, tdtype_list<DTypeFuncs::dtype_to_type_t<dts...> >, tdtype_list<DTypeFuncs::dtype_to_type_t<dts...>>, Args...>;
	bool check = DTypeFuncs::is_in<dts...>(dtype);
	val_type outp;
	if(!check){
		return outp; 
	}
	check = false;
	((sub_handle_execute_function<dts>(std::forward<UnaryFunction>(unary_op), outp, check, inp_arr, args...)), ...);	
	return std::move(outp);
}


template<DType...dts>
auto ArrayVoid::execute_function(bool throw_error, const char* func_name){
	return [throw_error, func_name, this](auto&& unary_op, auto&&... args) -> std::invoke_result_t<decltype(unary_op), tdtype_list<DTypeFuncs::dtype_to_type_t<dts...>>, tdtype_list<DTypeFuncs::dtype_to_type_t<dts...>>, decltype(args)...>{
			
			using UnaryFunction = decltype(unary_op);
			using val_type = std::invoke_result_t<decltype(unary_op), tdtype_list<DTypeFuncs::dtype_to_type_t<dts...>>, tdtype_list<DTypeFuncs::dtype_to_type_t<dts...>>, decltype(args)...>;
			bool check = throw_error ? check_dtypes<dts...>(func_name, dtype) : DTypeFuncs::is_in<dts...>(dtype);
			val_type outp;
			if(!check){
				if(throw_error)
					throw std::runtime_error("Unexpected dtype for current function");
				return outp; 
			}
			check = false;

			((sub_handle_execute_function<dts>(std::forward<UnaryFunction>(unary_op), outp, check, args...)), ...);	
			return std::move(outp);	
		};
}

template<DType dt, typename UnaryFunction, typename Output, typename... Args>
void sub_handle_execute_function(UnaryFunction&& unary_op, Output& v, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	v = std::forward<UnaryFunction>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(args)...);
}

template<DType dt, typename UnaryFunction, typename Output, typename... Args>
void sub_handle_execute_function(UnaryFunction&& unary_op, Output& v, bool& called, ArrayVoid& inp_arr, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	v = std::forward<UnaryFunction>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(), std::forward<Args>(args)...);
}

template<DType... dts, typename UnaryFunction, typename... Args>
auto ArrayVoid::cexecute_function(UnaryFunction&& unary_op, Args&&... args) const{
	using val_type = std::invoke_result_t<UnaryFunction, tdtype_list<const DTypeFuncs::dtype_to_type_t<dts...> >, tdtype_list<const DTypeFuncs::dtype_to_type_t<dts...>>, Args...>;
	bool check = DTypeFuncs::is_in<dts...>(dtype);
	val_type outp;
	if(!check){
		return outp; 
	}
	check = false;
	((sub_handle_cexecute_function<dts>(std::forward<UnaryFunction>(unary_op), outp, check, args...)), ...);	
	return std::move(outp);
}

template<DType... dts, typename UnaryFunction, typename... Args>
auto ArrayVoid::cexecute_function(UnaryFunction&& unary_op, const ArrayVoid& inp_arr, Args&&... args) const{
	using val_type = std::invoke_result_t<UnaryFunction, tdtype_list<const DTypeFuncs::dtype_to_type_t<dts...> >, tdtype_list<const DTypeFuncs::dtype_to_type_t<dts...> >, tdtype_list<const DTypeFuncs::dtype_to_type_t<dts...>>, Args...>;
	bool check = DTypeFuncs::is_in<dts...>(dtype);
	val_type outp;
	if(!check){
		return outp; 
	}
	check = false;
	((sub_handle_cexecute_function<dts>(std::forward<UnaryFunction>(unary_op), outp, check, inp_arr, args...)), ...);	
	return std::move(outp);
}


template<DType...dts>
auto ArrayVoid::cexecute_function(bool throw_error, const char* func_name) const{
	return [throw_error, func_name, this](auto&& unary_op, auto&&... args) -> std::invoke_result_t<decltype(unary_op), tdtype_list<DTypeFuncs::dtype_to_type_t<dts...>>, tdtype_list<DTypeFuncs::dtype_to_type_t<dts...>>, decltype(args)...>{
			
			using UnaryFunction = decltype(unary_op);
			using val_type = std::invoke_result_t<decltype(unary_op), tdtype_list<DTypeFuncs::dtype_to_type_t<dts...>>, tdtype_list<DTypeFuncs::dtype_to_type_t<dts...>>, decltype(args)...>;
			bool check = throw_error ? check_dtypes<dts...>(func_name, dtype) : DTypeFuncs::is_in<dts...>(dtype);
			val_type outp;
			if(!check){
				if(throw_error)
					throw std::runtime_error("Unexpected dtype for current function");
				return outp; 
			}
			check = false;

			((sub_handle_execute_function<dts>(std::forward<UnaryFunction>(unary_op), outp, check, args...)), ...);	
			return std::move(outp);	
		};
}

template<DType dt, typename UnaryFunction, typename Output, typename... Args>
void sub_handle_cexecute_function(UnaryFunction&& unary_op, Output& v, bool& called, Args&&... args) const{
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	v = std::forward<UnaryFunction>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(args)...);
}

template<DType dt, typename UnaryFunction, typename Output, typename... Args>
void sub_handle_cexecute_function(UnaryFunction&& unary_op, Output& v, bool& called, const ArrayVoid& inp_arr, Args&&... args) const{
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	v = std::forward<UnaryFunction>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), std::forward<Args>(args)...);
}


#ifndef USE_PARALLEL
template<class OutputIt, class UnaryOperation>
OutputIt ArrayVoid::transform_function(UnaryOperation&& unary_op, OutputIt d_first) const{
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;
		}
		case DType::Double:{
			using value_t = double;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;	
		}
		case DType::Float:{
			using value_t = float;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;	
		}
		case DType::int8:{
			using value_t = int8_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;	
		}
		case DType::int16:{
			using value_t = int16_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;
		}
		case DType::int64:{
			using value_t = int64_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;	
		}
		case DType::Bool:{
			using value_t = uint_bool_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;	
		}
	}
}


template<class InputIt2, class OutputIt, class UnaryOperation>
OutputIt ArrayVoid::transform_function(UnaryOperation&& unary_op, InputIt2 inp2, OutputIt d_first) const{
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;
		}
		case DType::Double:{
			using value_t = double;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;	
		}
		case DType::Float:{
			using value_t = float;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;	
		}
		case DType::int8:{
			using value_t = int8_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;	
		}
		case DType::int16:{
			using value_t = int16_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;
		}
		case DType::int64:{
			using value_t = int64_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;	
		}
		case DType::Bool:{
			using value_t = uint_bool_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;	
		}
	}

}

template<class OutputIt, class UnaryOperation>
OutputIt ArrayVoid::transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, OutputIt d_first) const{
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;
		}
		case DType::Double:{
			using value_t = double;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;	
		}
		case DType::Float:{
			using value_t = float;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;	
		}
		case DType::int8:{
			using value_t = int8_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;	
		}
		case DType::int16:{
			using value_t = int16_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;
		}
		case DType::int64:{
			using value_t = int64_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;	
		}
		case DType::Bool:{
			using value_t = uint_bool_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;	
		}
	}

}

template<class UnaryOperation>
void ArrayVoid::transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr){
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;
		}
		case DType::Double:{
			using value_t = double;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;	
		}
		case DType::Float:{
			using value_t = float;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;	
		}
		case DType::int8:{
			using value_t = int8_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;	
		}
		case DType::int16:{
			using value_t = int16_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;
		}
		case DType::int64:{
			using value_t = int64_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;	
		}
		case DType::Bool:{
			using value_t = uint_bool_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;	
		}
	}

}

#else
//std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));/tdtype_list<const value_t> start = tcbegin<value_t>();\n\t\ttbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [\&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
template<class OutputIt, class UnaryOperation>
OutputIt ArrayVoid::transform_function(UnaryOperation&& unary_op, OutputIt d_first) const{
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;
		}
		case DType::Double:{
			using value_t = double;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;	
		}
		case DType::Float:{
			using value_t = float;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;	
		}
		case DType::int8:{
			using value_t = int8_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;	
		}
		case DType::int16:{
			using value_t = int16_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;
		}
		case DType::int64:{
			using value_t = int64_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;	
		}
		case DType::Bool:{
			using value_t = uint_bool_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;	
		}
	}
}

///std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);/tdtype_list<const value_t> start = tcbegin<value_t>();\r\t\t\ttbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [\&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});

template<class InputIt2, class OutputIt, class UnaryOperation>
OutputIt ArrayVoid::transform_function(UnaryOperation&& unary_op, InputIt2 inp2, OutputIt d_first) const{
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;
		}
		case DType::Double:{
			using value_t = double;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::Float:{
			using value_t = float;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::int8:{
			using value_t = int8_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::int16:{
			using value_t = int16_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;
		}
		case DType::int64:{
			using value_t = int64_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::Bool:{
			using value_t = uint_bool_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
	}

}

///std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);/tdtype_list<const value_t> start = tcbegin<value_t>();\r\t\t\ttdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()\r\t\t\ttbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [\&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});


template<class OutputIt, class UnaryOperation>
OutputIt ArrayVoid::transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, OutputIt d_first) const{
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;
		}
		case DType::Double:{
			using value_t = double;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::Float:{
			using value_t = float;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::int8:{
			using value_t = int8_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::int16:{
			using value_t = int16_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;
		}
		case DType::int64:{
			using value_t = int64_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::Bool:{
			using value_t = uint_bool_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
	}

}


///std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);/tdtype_list<const value_t> start = tcbegin<value_t>();\r\t\t\ttdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()\r\t\t\ttdtype_list<value_t> my_start = tbegin<value_t>()\r\t\t\ttbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [\&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});

template<class UnaryOperation>
void ArrayVoid::transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr){
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;
		}
		case DType::Double:{
			using value_t = double;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;	
		}
		case DType::Float:{
			using value_t = float;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;	
		}
		case DType::int8:{
			using value_t = int8_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;	
		}
		case DType::int16:{
			using value_t = int16_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;
		}
		case DType::int64:{
			using value_t = int64_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;	
		}
		case DType::Bool:{
			using value_t = uint_bool_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;	
		}
	}

}

#endif


#ifndef USE_PARALLEL
template<class UnaryOperation>
void ArrayVoid::for_ceach(UnaryOperation&& unary_op) const{
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;
		}
		case DType::Double:{
			using value_t = double;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::Float:{
			using value_t = float;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::int8:{
			using value_t = int8_t;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::int16:{
			using value_t = int16_t;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;
		}
		case DType::int64:{
			using value_t = int64_t;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::Bool:{
			using value_t = uint_bool_t;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
	}

}

template<class UnaryOperation>
void ArrayVoid::for_each(UnaryOperation&& unary_op){
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;
		}
		case DType::Double:{
			using value_t = double;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::Float:{
			using value_t = float;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::int8:{
			using value_t = int8_t;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::int16:{
			using value_t = int16_t;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;
		}
		case DType::int64:{
			using value_t = int64_t;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::Bool:{
			using value_t = uint_bool_t;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
	}

}

#else

template<class UnaryOperation>
void ArrayVoid::for_ceach(UnaryOperation&& unary_op) const{
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;
		}
		case DType::Double:{
			using value_t = double;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::Float:{
			using value_t = float;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::int8:{
			using value_t = int8_t;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::int16:{
			using value_t = int16_t;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;
		}
		case DType::int64:{
			using value_t = int64_t;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::Bool:{
			using value_t = uint_bool_t;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
	}

}

template<class UnaryOperation>
void ArrayVoid::for_each(UnaryOperation&& unary_op){
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;
		}
		case DType::Double:{
			using value_t = double;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::Float:{
			using value_t = float;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::int8:{
			using value_t = int8_t;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::int16:{
			using value_t = int16_t;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;
		}
		case DType::int64:{
			using value_t = int64_t;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::Bool:{
			using value_t = uint_bool_t;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
	}

}

#endif


template<class UnaryOperator, class... Args>
auto ArrayVoid::execute_function(UnaryOperator&& unary_op, Args&&... arg){
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::Double:{
			using value_t = double;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::Float:{
			using value_t = float;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::Long:{
			using value_t = uint32_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::uint8:{
			using value_t = uint8_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::int8:{
			using value_t = int8_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::int16:{
			using value_t = int16_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::uint16:{
			using value_t = uint16_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::int64:{
			using value_t = int64_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::Bool:{
			using value_t = uint_bool_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
	}

}

template<class UnaryOperator, class... Args>
auto ArrayVoid::cexecute_function(UnaryOperator&& unary_op, Args&&... arg) const{
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::Double:{
			using value_t = double;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::Float:{
			using value_t = float;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::Long:{
			using value_t = uint32_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::uint8:{
			using value_t = uint8_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::int8:{
			using value_t = int8_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::int16:{
			using value_t = int16_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::uint16:{
			using value_t = uint16_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::int64:{
			using value_t = int64_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::Bool:{
			using value_t = uint_bool_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
	}

}

template<class UnaryOperator, class... Args>
auto ArrayVoid::execute_function(UnaryOperator&& unary_op, ArrayVoid& inp_arr, Args&&... arg){
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::Double:{
			using value_t = double;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::Float:{
			using value_t = float;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::Long:{
			using value_t = uint32_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::uint8:{
			using value_t = uint8_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::int8:{
			using value_t = int8_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::int16:{
			using value_t = int16_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::uint16:{
			using value_t = uint16_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::int64:{
			using value_t = int64_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::Bool:{
			using value_t = uint_bool_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(),std::forward<Args>(arg)...);
		}
	}

}

//'<,'>s/std::forward<UnaryOperator>(unary_op)(begin, end, inp2, (arg)...);/return std::forward<UnaryOperator>(unary_op)(begin, end, std::forward<Args>(arg)...); | '<,'>g/return;/d
template<class UnaryOperator, class... Args>
auto ArrayVoid::cexecute_function(UnaryOperator&& unary_op, const ArrayVoid& inp_arr, Args&&... arg) const{
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::Double:{
			using value_t = double;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::Float:{
			using value_t = float;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::Long:{
			using value_t = uint32_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::uint8:{
			using value_t = uint8_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::int8:{
			using value_t = int8_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::int16:{
			using value_t = int16_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::uint16:{
			using value_t = uint16_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::int64:{
			using value_t = int64_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(),std::forward<Args>(arg)...);
		}
		case DType::Bool:{
			using value_t = uint_bool_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(),std::forward<Args>(arg)...);
		}
	}

}

//no bool version:


#ifndef USE_PARALLEL

template<class OutputIt, class UnaryOperation>
OutputIt ArrayVoid::transform_function_nbool(UnaryOperation&& unary_op, OutputIt d_first) const{
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;
		}
		case DType::Double:{
			using value_t = double;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;	
		}
		case DType::Float:{
			using value_t = float;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;	
		}
		case DType::int8:{
			using value_t = int8_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;	
		}
		case DType::int16:{
			using value_t = int16_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;
		}
		case DType::int64:{
			using value_t = int64_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));
			return d_first;	
		}
		default:
			return d_first;	
	}
}

template<class InputIt2, class OutputIt, class UnaryOperation>
OutputIt ArrayVoid::transform_function_nbool(UnaryOperation&& unary_op, InputIt2 inp2, OutputIt d_first) const{
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;
		}
		case DType::Double:{
			using value_t = double;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;	
		}
		case DType::Float:{
			using value_t = float;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;	
		}
		case DType::int8:{
			using value_t = int8_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;	
		}
		case DType::int16:{
			using value_t = int16_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;
		}
		case DType::int64:{
			using value_t = int64_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);
			return d_first;	
		}
		default:
			return d_first;	
	}

}

template<class OutputIt, class UnaryOperation>
OutputIt ArrayVoid::transform_function_nbool(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, OutputIt d_first) const{
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;
		}
		case DType::Double:{
			using value_t = double;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;	
		}
		case DType::Float:{
			using value_t = float;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;	
		}
		case DType::int8:{
			using value_t = int8_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;	
		}
		case DType::int16:{
			using value_t = int16_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;
		}
		case DType::int64:{
			using value_t = int64_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);
			return d_first;	
		}
		default:
			return d_first;	
	}

}

template<class UnaryOperation>
void ArrayVoid::transform_function_nbool(UnaryOperation&& unary_op, const ArrayVoid& inp_arr){
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;
		}
		case DType::Double:{
			using value_t = double;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;	
		}
		case DType::Float:{
			using value_t = float;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;	
		}
		case DType::int8:{
			using value_t = int8_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;	
		}
		case DType::int16:{
			using value_t = int16_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;
		}
		case DType::int64:{
			using value_t = int64_t;
			std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);
			break;	
		}
		default:
			break;
	}

}

#else

//std::transform(tcbegin<value_t>(), tcend<value_t>(), d_first, std::forward<UnaryOperation>(unary_op));/tdtype_list<const value_t> start = tcbegin<value_t>();\n\t\ttbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [\&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});


template<class OutputIt, class UnaryOperation>
OutputIt ArrayVoid::transform_function_nbool(UnaryOperation&& unary_op, OutputIt d_first) const{
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;
		}
		case DType::Double:{
			using value_t = double;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;	
		}
		case DType::Float:{
			using value_t = float;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;	
		}
		case DType::int8:{
			using value_t = int8_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;	
		}
		case DType::int16:{
			using value_t = int16_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;
		}
		case DType::int64:{
			using value_t = int64_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation>(unary_op));});
			return d_first;	
		}
		default:
			return d_first;	
	}
}
///std::transform(tcbegin<value_t>(), tcend<value_t>(), inp2, d_first, unary_op);/tdtype_list<const value_t> start = tcbegin<value_t>();\r\t\t\ttbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [\&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});

template<class InputIt2, class OutputIt, class UnaryOperation>
OutputIt ArrayVoid::transform_function_nbool(UnaryOperation&& unary_op, InputIt2 inp2, OutputIt d_first) const{
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;
		}
		case DType::Double:{
			using value_t = double;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::Float:{
			using value_t = float;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::int8:{
			using value_t = int8_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::int16:{
			using value_t = int16_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;
		}
		case DType::int64:{
			using value_t = int64_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		default:
			return d_first;	
	}

}

///std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), d_first, unary_op);/tdtype_list<const value_t> start = tcbegin<value_t>();\r\t\t\ttdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()\r\t\t\ttbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [\&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});


template<class OutputIt, class UnaryOperation>
OutputIt ArrayVoid::transform_function_nbool(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, OutputIt d_first) const{
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;
		}
		case DType::Double:{
			using value_t = double;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::Float:{
			using value_t = float;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::int8:{
			using value_t = int8_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::int16:{
			using value_t = int16_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;
		}
		case DType::int64:{
			using value_t = int64_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), unary_op);});
			return d_first;	
		}
		default:
			return d_first;	
	}

}
///std::transform(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), tbegin<value_t>(), unary_op);/tdtype_list<const value_t> start = tcbegin<value_t>();\r\t\t\ttdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()\r\t\t\ttdtype_list<value_t> my_start = tbegin<value_t>()\r\t\t\ttbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [\&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});

template<class UnaryOperation>
void ArrayVoid::transform_function_nbool(UnaryOperation&& unary_op, const ArrayVoid& inp_arr){
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;
		}
		case DType::Double:{
			using value_t = double;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;	
		}
		case DType::Float:{
			using value_t = float;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;	
		}
		case DType::int8:{
			using value_t = int8_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;	
		}
		case DType::int16:{
			using value_t = int16_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;
		}
		case DType::int64:{
			using value_t = int64_t;
			tdtype_list<const value_t> start = tcbegin<value_t>();
			tdtype_list<const value_t> inp_start = inp_arr.tcbegin<value_t>()
			tdtype_list<value_t> my_start = tbegin<value_t>()
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, size), [&](tbb::blocked_range<uint32_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
			break;	
		}
		default:
			break;
	}

}

#endif


#ifdef USE_PARALLEL

template<class UnaryOperation>
void ArrayVoid::for_ceach_nbool(UnaryOperation&& unary_op) const{
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;
		}
		case DType::Double:{
			using value_t = double;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::Float:{
			using value_t = float;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::int8:{
			using value_t = int8_t;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::int16:{
			using value_t = int16_t;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;
		}
		case DType::int64:{
			using value_t = int64_t;
			tbb::parallel_for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		default:
			break;	
	}

}

template<class UnaryOperation>
void ArrayVoid::for_each_nbool(UnaryOperation&& unary_op){
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;
		}
		case DType::Double:{
			using value_t = double;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::Float:{
			using value_t = float;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::int8:{
			using value_t = int8_t;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::int16:{
			using value_t = int16_t;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;
		}
		case DType::int64:{
			using value_t = int64_t;
			tbb::parallel_for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		default:
			break;	
	}

}

#else

template<class UnaryOperation>
void ArrayVoid::for_ceach_nbool(UnaryOperation&& unary_op) const{
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;
		}
		case DType::Double:{
			using value_t = double;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::Float:{
			using value_t = float;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::int8:{
			using value_t = int8_t;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::int16:{
			using value_t = int16_t;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;
		}
		case DType::int64:{
			using value_t = int64_t;
			std::for_each(tcbegin<value_t>(), tcend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		default:
			break;	
	}

}

template<class UnaryOperation>
void ArrayVoid::for_each_nbool(UnaryOperation&& unary_op){
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;
		}
		case DType::Double:{
			using value_t = double;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::Float:{
			using value_t = float;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::Long:{
			using value_t = uint32_t;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::uint8:{
			using value_t = uint8_t;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::int8:{
			using value_t = int8_t;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::int16:{
			using value_t = int16_t;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		case DType::uint16:{
			using value_t = uint16_t;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;
		}
		case DType::int64:{
			using value_t = int64_t;
			std::for_each(tbegin<value_t>(), tend<value_t>(), std::forward<UnaryOperation>(unary_op));
			return;	
		}
		default:
			break;	
	}

}

#endif

template<class UnaryOperator, class... Args>
auto ArrayVoid::execute_function_nbool(UnaryOperator&& unary_op, Args&&... arg){
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::Double:{
			using value_t = double;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::Float:{
			using value_t = float;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::Long:{
			using value_t = uint32_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::uint8:{
			using value_t = uint8_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::int8:{
			using value_t = int8_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::int16:{
			using value_t = int16_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::uint16:{
			using value_t = uint16_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::int64:{
			using value_t = int64_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), std::forward<Args>(arg)...);
		}
		default:{
			return std::invoke_result_t<UnaryOperator, tdtype_list<float>, tdtype_list<float>, Args...>();}
	}

}

template<class UnaryOperator, class... Args>
auto ArrayVoid::cexecute_function_nbool(UnaryOperator&& unary_op, Args&&... arg) const{
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::Double:{
			using value_t = double;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::Float:{
			using value_t = float;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::Long:{
			using value_t = uint32_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::uint8:{
			using value_t = uint8_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::int8:{
			using value_t = int8_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::int16:{
			using value_t = int16_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::uint16:{
			using value_t = uint16_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::int64:{
			using value_t = int64_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), std::forward<Args>(arg)...);}
		case DType::Bool:{
			return std::invoke_result_t<UnaryOperator, tdtype_list<const float>, tdtype_list<const float>, Args...>();}
	}

}

template<class UnaryOperator, class... Args>
auto ArrayVoid::execute_function_nbool(UnaryOperator&& unary_op, ArrayVoid& inp_arr, Args&&... arg){
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::Double:{
			using value_t = double;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::Float:{
			using value_t = float;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::Long:{
			using value_t = uint32_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::uint8:{
			using value_t = uint8_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::int8:{
			using value_t = int8_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::int16:{
			using value_t = int16_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::uint16:{
			using value_t = uint16_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::int64:{
			using value_t = int64_t;
			return std::forward<UnaryOperator>(unary_op)(tbegin<value_t>(), tend<value_t>(), inp_arr.tbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::Bool:{
			return std::invoke_result_t<UnaryOperator, tdtype_list<float>, tdtype_list<float>, tdtype_list<float>, Args...>();}
	}

}

template<class UnaryOperator, class... Args>
auto ArrayVoid::cexecute_function_nbool(UnaryOperator&& unary_op, const ArrayVoid& inp_arr, Args&&... arg) const{
	switch(dtype){
		case DType::Integer:{
			using value_t = int32_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::Double:{
			using value_t = double;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::Float:{
			using value_t = float;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::Long:{
			using value_t = uint32_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::TensorObj:{
			using value_t = Tensor;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::cfloat:{
			using value_t = std::complex<float>;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::cdouble:{
			using value_t = std::complex<double>;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::uint8:{
			using value_t = uint8_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::int8:{
			using value_t = int8_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::int16:{
			using value_t = int16_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::uint16:{
			using value_t = uint16_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::int64:{
			using value_t = int64_t;
			return std::forward<UnaryOperator>(unary_op)(tcbegin<value_t>(), tcend<value_t>(), inp_arr.tcbegin<value_t>(), std::forward<Args>(arg)...);
		}
		case DType::Bool:{
			return std::invoke_result_t<UnaryOperator, tdtype_list<const float>, tdtype_list<const float>, tdtype_list<const float>, Args...>();}
	}

}

}


#endif
