#ifndef ARRAY_VOID_H
#define ARRAY_VOID_H
#include "ranges.h"
namespace nt {
class ArrayVoid;
}


#include <memory>
#include <cstdio>
#include <type_traits>

#include "DType_enum.h"
#include "DType_list.h"
#include <vector>
#include <cstdlib>
#include "Scalar.h"
#include "../Tensor.h"

namespace nt{


class ArrayVoid{
	std::shared_ptr<void> _vals;
	std::shared_ptr<void*> _strides;
	size_t size, _last_index, _start, type_size, available_size;
	
	void unique_strides(bool pre_order=true);
	std::shared_ptr<void*> make_unique_strides(bool pre_order=true) const;
	std::shared_ptr<void*> make_unique_strides(std::size_t start, std::size_t end) const;
	std::shared_ptr<void> make_shared(size_t, DType) const;
	const size_t dtype_size(DType) const;
	ArrayVoid(const std::shared_ptr<void>&, std::shared_ptr<void*>&&, const std::size_t, const std::size_t, const std::size_t, DType);
	public:
		DType dtype;
		ArrayVoid(uint32_t, DType);
		ArrayVoid(const std::shared_ptr<void>&, const std::shared_ptr<void*>&,  const std::size_t, const std::size_t, const std::size_t, DType);
		ArrayVoid& operator=(const ArrayVoid&);
		ArrayVoid& operator=(ArrayVoid&&);
		ArrayVoid(const ArrayVoid&);
		ArrayVoid(ArrayVoid&&);
		const std::size_t Size() const;
		const void* data_ptr() const;
		void* data_ptr();
		const void* data_ptr_end() const;
		void* data_ptr_end();
		void** strides_cbegin() const;
		void** strides_begin();
		void** strides_cend() const;
		void** strides_end();
		void operator=(Scalar);
		ArrayVoid& fill_ptr_(Scalar);
		std::shared_ptr<void> share_part(uint32_t) const;
		ArrayVoid share_array(uint32_t) const;
		ArrayVoid share_array(uint32_t, uint32_t) const;
		template<typename T>
		tdtype_list<T> tbegin();
		template<typename T>
		tdtype_list<T> tend();
		template<typename T>
		tdtype_list<const T> tcbegin() const;
		template<typename T>
		tdtype_list<const T> tcend() const;
		ArrayVoid change_stride(const std::vector<std::size_t>&);
		ArrayVoid range(std::vector<my_range>) const;
		bool is_contiguous() const;
		ArrayVoid contiguous() const;
		/* template<DType dt = DType::Integer> */
		ArrayVoid copy_strides(bool copy=true) const;
		ArrayVoid new_stride(uint32_t size) const;

		ArrayVoid& operator*=(Scalar);
		ArrayVoid& operator/=(Scalar);
		ArrayVoid& operator+=(Scalar);
		ArrayVoid& operator-=(Scalar);

		ArrayVoid operator*(Scalar) const;
		ArrayVoid operator/(Scalar) const;
		ArrayVoid operator+(Scalar) const;
		ArrayVoid operator-(Scalar) const;

		ArrayVoid& operator*=(const ArrayVoid&);
		ArrayVoid& operator/=(const ArrayVoid&);
		ArrayVoid& operator+=(const ArrayVoid&);
		ArrayVoid& operator-=(const ArrayVoid&);

		ArrayVoid operator*(const ArrayVoid&) const;
		ArrayVoid operator/(const ArrayVoid&) const;
		ArrayVoid operator+(const ArrayVoid&) const;
		ArrayVoid operator-(const ArrayVoid&) const;

		ArrayVoid& operator*=(const Tensor&);
		ArrayVoid& operator/=(const Tensor&);
		ArrayVoid& operator+=(const Tensor&);
		ArrayVoid& operator-=(const Tensor&);

		ArrayVoid operator*(const Tensor&) const;
		ArrayVoid operator/(const Tensor&) const;
		ArrayVoid operator+(const Tensor&) const;
		ArrayVoid operator-(const Tensor&) const;

		ArrayVoid operator==(Scalar) const;
		ArrayVoid operator>=(Scalar) const;
		ArrayVoid operator<=(Scalar) const;
		ArrayVoid operator>(Scalar) const;
		ArrayVoid operator<(Scalar) const;
		ArrayVoid inverse() const;
		ArrayVoid& inverse_();
		void resize(const size_t);

		
		/* const std::size_t* stride_cbegin() const; */
		/* const std::size_t* stride_cend() const; */
		/* std::size_t* stride_begin(); */
		/* std::size_t* stride_end(); */
		/* std::vector<std::size_t>::const_iterator stride_it_cbegin() const; */
		/* std::vector<std::size_t>::const_iterator stride_it_cend() const; */
		/* std::vector<std::size_t>::iterator stride_it_begin(); */
		/* std::vector<std::size_t>::iterator stride_it_end(); */

		/* dtype_list begin(); */
		/* dtype_list end(); */
		/* const_dtype_list cbegin() const; */
		/* const_dtype_list cend() const; */
		const uint32_t use_count() const;
		ArrayVoid& iota(Scalar);
		void copy(ArrayVoid&, unsigned long long i=0) const;
		ArrayVoid uint32() const;
		ArrayVoid int32() const;
		ArrayVoid Double() const;
		ArrayVoid Float() const;
		ArrayVoid cfloat() const;
		ArrayVoid cdouble() const;
		ArrayVoid tensorobj() const;
		ArrayVoid uint8() const;
		ArrayVoid int8() const;
		ArrayVoid uint16() const;
		ArrayVoid int16() const;
		ArrayVoid int64() const;
		ArrayVoid Bool() const;
		ArrayVoid to(DType) const;
#ifdef _HALF_FLOAT_SUPPORT_
		ArrayVoid Float16() const;
		ArrayVoid Complex32() const;
#endif
#ifdef _128_FLOAT_SUPPORT_
		ArrayVoid Float128() const;
#endif
#ifdef __SIZEOF_INT128__
		ArrayVoid Int128() const;
		ArrayVoid UInt128() const;
#endif
		ArrayVoid exp() const;
		ArrayVoid& exp_();
		ArrayVoid& complex_();
		ArrayVoid& floating_();
		ArrayVoid& integer_();
		ArrayVoid& unsigned_();
	
		template<DType dt, DType... dts, typename UnaryFunction, typename... Args>
		auto execute_function(UnaryFunction&& unary_op, Args&&... args);
		template<DType dt, DType... dts, typename UnaryFunction, typename... Args>
		auto execute_function(UnaryFunction&& unary_op, ArrayVoid& inp_arr, Args&&... args);
		template<DType dt, DType... dts>
		auto execute_function(bool throw_error=true, const char* func_name = __builtin_FUNCTION());
		template<DType dt, typename UnaryFunction, typename Output, typename... Args>
		void sub_handle_execute_function(UnaryFunction&& unary_op, Output& v, bool& called, ArrayVoid& inp_arr, Args&&... args);
		template<DType dt, typename UnaryFunction, typename Output, typename... Args>
		void sub_handle_execute_function(UnaryFunction&& unary_op, Output& v, bool& called, Args&&... args);
		template<DType dt, typename UnaryFunction, typename... Args>
		void sub_handle_execute_function_void(UnaryFunction&& unary_op, bool& called, ArrayVoid& inp_arr, Args&&... args);
		template<DType dt, typename UnaryFunction, typename... Args>
		void sub_handle_execute_function_void(UnaryFunction&& unary_op, bool& called, Args&&... args);
		
		template<typename WrappedTypes, typename UnaryFunction, typename... Args, std::enable_if_t<
			DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value
			&& !DTypeFuncs::is_wrapped_dtype<UnaryFunction>::value, bool> = true>
		auto execute_function(UnaryFunction&& unary_op, Args&&... args);
		template<typename WrappedTypes, typename UnaryFunction, typename... Args>
		auto execute_function(UnaryFunction&& unary_op, ArrayVoid& inp_arr, Args&&... args); 
		template<typename WrappedTypes, std::enable_if_t<DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value, bool> = true>
		auto execute_function(bool throw_error=true, const char* func_name = __builtin_FUNCTION()); 

		
		template<DType dt, DType... dts, typename UnaryFunction, typename... Args>
		auto cexecute_function(UnaryFunction&& unary_op, Args&&... args) const;
		template<DType dt, DType... dts, typename UnaryFunction, typename... Args>
		auto cexecute_function(UnaryFunction&& unary_op, const ArrayVoid& inp_arr, Args&&... args) const;
		template<DType dt, DType... dts>
		auto cexecute_function(bool throw_error=true, const char* func_name = __builtin_FUNCTION()) const;
		template<DType dt, typename UnaryFunction, typename Output, typename... Args>
		void sub_handle_cexecute_function(UnaryFunction&& unary_op, Output& v, bool& called, const ArrayVoid& inp_arr, Args&&... args) const;
		template<DType dt, typename UnaryFunction, typename Output, typename... Args>
		void sub_handle_cexecute_function(UnaryFunction&& unary_op, Output& v, bool& called, Args&&... args) const;
		template<DType dt, typename UnaryFunction, typename... Args>
		void sub_handle_cexecute_function_void(UnaryFunction&& unary_op, bool& called, const ArrayVoid& inp_arr, Args&&... args) const;
		template<DType dt, typename UnaryFunction, typename... Args>
		void sub_handle_cexecute_function_void(UnaryFunction&& unary_op, bool& called, Args&&... args) const;

		template<typename WrappedTypes, typename UnaryFunction, typename... Args, std::enable_if_t<
			DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value
			&& !DTypeFuncs::is_wrapped_dtype<UnaryFunction>::value, bool> = true>
		auto cexecute_function(UnaryFunction&& unary_op, Args&&... args) const;
		template<typename WrappedTypes, typename UnaryFunction, typename... Args>
		auto cexecute_function(UnaryFunction&& unary_op, const ArrayVoid& inp_arr, Args&&... args) const; 
		template<typename WrappedTypes, std::enable_if_t<DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value, bool> = true>
		auto cexecute_function(bool throw_error=true, const char* func_name = __builtin_FUNCTION()) const; 
		
		template<class UnaryOperator, class... Args>
		auto execute_function(UnaryOperator&& unary_op, Args&&... arg);
		template<class UnaryOperator, class... Args>
		auto execute_function_nbool(UnaryOperator&& unary_op, Args&&... arg);
		template<class UnaryOperator, class... Args>
		auto cexecute_function(UnaryOperator&& unary_op, Args&&... arg) const;
		template<class UnaryOperator, class... Args>
		auto cexecute_function_nbool(UnaryOperator&& unary_op, Args&&... arg) const;
		template<class UnaryOperator, class... Args>
		auto execute_function(UnaryOperator&& unary_op, ArrayVoid& inp_arr, Args&&... arg);
		template<class UnaryOperator, class... Args>
		auto cexecute_function(UnaryOperator&& unary_op, const ArrayVoid& inp_arr, Args&&... arg) const;
		template<class UnaryOperator, class... Args>
		auto execute_function_nbool(UnaryOperator&& unary_op, ArrayVoid& inp_arr, Args&&... arg);
		template<class UnaryOperator, class... Args>
		auto cexecute_function_nbool(UnaryOperator&& unary_op, const ArrayVoid& inp_arr, Args&&... arg) const;


		template<DType... dts, class OutputIt, class UnaryOperation, std::enable_if_t<!std::is_same_v<OutputIt, bool>, bool> = true>
		OutputIt transform_function(UnaryOperation&& unary_op, OutputIt d_first, bool throw_error = true, const char* str = __builtin_FUNCTION()) const;
		template<DType dt, class OutputIt, class UnaryOperation>
		void sub_transform_function(UnaryOperation&& unary_op, OutputIt& d_first) const;

		template<typename WrappedTypes, class OutputIt, class UnaryOperation, std::enable_if_t<
			DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value
			&& !DTypeFuncs::is_wrapped_dtype<UnaryOperation>::value
			&& !DTypeFuncs::is_wrapped_dtype<OutputIt>::value, bool> = true>
		OutputIt transform_function(UnaryOperation&& unary_op, OutputIt d_first, bool throw_error = true, const char* str = __builtin_FUNCTION()) const;


		template<class OutputIt, class UnaryOperation>
		OutputIt transform_function(UnaryOperation&& unary_op, OutputIt d_first) const;
		template<class OutputIt, class UnaryOperation>
		OutputIt transform_function_nbool(UnaryOperation&& unary_op, OutputIt d_first) const;


		template<DType... dts, class InputIt2, class OutputIt, class UnaryOperation, 
			std::enable_if_t<!std::is_same_v<OutputIt, bool> 
				&& !std::is_same_v<OutputIt, const char*>
				&& !std::is_same_v<InputIt2, bool> 
				&& !std::is_same_v<InputIt2, const char*>, bool> = true>
		OutputIt transform_function(UnaryOperation&& unary_op, InputIt2 inp2, OutputIt d_first, bool throw_error=true, const char* str=__builtin_FUNCTION()) const;
		template<DType dt, class InputIt2, class OutputIt, class UnaryOperation>
		void sub_transform_function(UnaryOperation&& unary_op, InputIt2& inp2, OutputIt& d_first) const;
		template<class InputIt2, class OutputIt, class UnaryOperation>
		OutputIt transform_function(UnaryOperation&& unary_op, InputIt2 inp2, OutputIt d_first) const;
		template<class InputIt2, class OutputIt, class UnaryOperation>
		OutputIt transform_function_nbool(UnaryOperation&& unary_op, InputIt2 inp2, OutputIt d_first) const;

		template<typename WrappedTypes, class InputIt2, class OutputIt, class UnaryOperation, std::enable_if_t<
			DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value
				&& !std::is_same_v<OutputIt, bool> 
				&& !std::is_same_v<OutputIt, const char*>
				&& !std::is_same_v<InputIt2, bool> 
				&& !std::is_same_v<InputIt2, const char*>
			, bool> = true>	
		OutputIt transform_function(UnaryOperation&& unary_op, InputIt2 inp2, OutputIt d_first, bool throw_error=true, const char* str=__builtin_FUNCTION()) const;

		template<DType... dts, class OutputIt, class UnaryOperation, std::enable_if_t<!std::is_same_v<OutputIt, bool>, bool> = true>
		OutputIt transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, OutputIt d_first, bool throw_error = true, const char* str = __builtin_FUNCTION()) const;
		template<DType dt, class OutputIt, class UnaryOperation>
		void sub_transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, OutputIt& d_first) const;
		template<class OutputIt, class UnaryOperation>
		OutputIt transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, OutputIt d_first) const;
		template<class OutputIt, class UnaryOperation>
		OutputIt transform_function_nbool(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, OutputIt d_first) const;

		template<typename WrappedTypes, class OutputIt, class UnaryOperation, std::enable_if_t<
			DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value && !std::is_same_v<OutputIt, bool>, bool> = true>
		OutputIt transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, OutputIt d_first, bool throw_error = true, const char* str = __builtin_FUNCTION()) const;


		template<DType dt, DType... dts, class UnaryOperation>
		void transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, bool throw_error=true, const char* str = __builtin_FUNCTION()); //done
		template<DType dt, class UnaryOperation>
		void sub_transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr);
		template<class UnaryOperation>
		void transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr); //done
		template<class UnaryOperation>
		void transform_function_nbool(UnaryOperation&& unary_op, const ArrayVoid& inp_arr);
	
		template<typename WrappedTypes, class UnaryOperation, std::enable_if_t< DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value, bool> = true>
		void transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, bool throw_error=true, const char* str = __builtin_FUNCTION());



		template<DType...dts, class UnaryOperation>
		void for_ceach(UnaryOperation&& unary_op, bool throw_error=true, const char* str=__builtin_FUNCTION()) const; //done
		template<DType...dts, class UnaryOperation>
		void for_each(UnaryOperation&& unary_op, bool throw_error=true, const char* str=__builtin_FUNCTION()); //done 
		template<class UnaryOperation>
		void for_ceach(UnaryOperation&& unary_op) const; //done
		template<class UnaryOperation>
		void for_each(UnaryOperation&& unary_op); //done 
		template<class UnaryOperation>
		void for_ceach_nbool(UnaryOperation&& unary_op) const;
		template<class UnaryOperation>
		void for_each_nbool(UnaryOperation&& unary_op);
		template<DType dt, class UnaryOperation>
		void sub_for_each(UnaryOperation&& unary_op);
		template<DType dt, class UnaryOperation>
		void sub_for_ceach(UnaryOperation&& unary_op) const;

		template<typename WrappedTypes, class UnaryOperation, std::enable_if_t< DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value, bool> = true>
		void for_ceach(UnaryOperation&& unary_op, bool throw_error=true, const char* str=__builtin_FUNCTION()) const;
		template<typename WrappedTypes, class UnaryOperation, std::enable_if_t< DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value, bool> = true>
		void for_each(UnaryOperation&& unary_op, bool throw_error=true, const char* str=__builtin_FUNCTION());


};
}

#endif
