#ifndef _ARRAY_VOID_HPP_
#define _ARRAY_VOID_HPP_

#include "ArrayVoid.h"
#include "DType.h"
#include "../Tensor.h"
#include "DType_enum.h"
#include "../memory/iterator.h"
#include "compatible/DType_compatible.h"
#include <regex>
#include <stdexcept>
#include <type_traits>
#include <thread>
#include <functional>

#ifdef USE_PARALLEL
	#include <tbb/parallel_for.h>
	#include <tbb/parallel_for_each.h>
#endif



//silence depreciation warnings for certain needed headers
#ifdef _MSC_VER
#ifndef _SILENCE_CXX17_C_HEADER_DEPRECATION_WARNING
#define _SILENCE_CXX17_C_HEADER_DEPRECATION_WARNING
#endif

#ifndef _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#endif

#endif

namespace nt{

//this is to make it so it is not ambiguous with calling execute_function(func, args..);
//the above is for all dtypes, this is for specific dtypes
template<DType dt, DType... dts, typename UnaryFunction, typename... Args>
inline auto ArrayVoid::execute_function(UnaryFunction&& unary_op, Args&&... args){
	using val_type = std::invoke_result_t<UnaryFunction, DTypeFuncs::dtype_to_type_t<dt>*, DTypeFuncs::dtype_to_type_t<dt>*, Args...>;
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

//tdtype_list<DTypeFuncs::dtype_to_type_t<m_dtype> >, tdtype_list<DTypeFuncs::dtype_to_type_t<m_dtype>>,/DTypeFuncs::dtype_to_type_t<m_dtype>*, DTypeFuncs::dtype_to_type_t<m_dtype>*,
//DTypeFuncs::dtype_to_type_t<m_dtype>*/DTypeFuncs::dtype_to_type_t<m_dtype>*
template<typename WrappedTypes, typename UnaryFunction, typename... Args, std::enable_if_t<
				DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value
				&& !DTypeFuncs::is_wrapped_dtype<UnaryFunction>::value, bool>>
inline auto ArrayVoid::execute_function(UnaryFunction&& unary_op, Args&&... args){
	constexpr DType m_dtype = WrappedTypes::next;
	if(m_dtype != dtype && WrappedTypes::done){
		using val_type = std::invoke_result_t<UnaryFunction&&, DTypeFuncs::dtype_to_type_t<m_dtype>*, DTypeFuncs::dtype_to_type_t<m_dtype>*, Args...>;
		if constexpr(std::is_same_v<val_type, void>){
			return;
        }else{
			val_type outp;
			return outp;
		}
	}
    else if(m_dtype != dtype){
		return execute_function<typename WrappedTypes::next_wrapper>(std::forward<UnaryFunction&&>(unary_op), std::forward<Args&&>(args)...);
	}
	bool check = false;
	using val_type = std::invoke_result_t<UnaryFunction, DTypeFuncs::dtype_to_type_t<m_dtype>*, DTypeFuncs::dtype_to_type_t<m_dtype>*, Args...>;
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


//DTypeFuncs::dtype_to_type_t<dt, dts...>*/DTypeFuncs::dtype_to_type_t<dt, dts...>*
template<DType dt, DType... dts, typename UnaryFunction, typename... Args>
inline auto ArrayVoid::execute_function(UnaryFunction&& unary_op, ArrayVoid& inp_arr, Args&&... args){
	using val_type = std::invoke_result_t<UnaryFunction, 
	      DTypeFuncs::dtype_to_type_t<dt, dts...>*, 
	      DTypeFuncs::dtype_to_type_t<dt, dts...>*, 
	      DTypeFuncs::dtype_to_type_t<dt, dts...>*, Args...>;
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


//tdtype_list<DTypeFuncs::dtype_to_type_t<m_dtype>>/DTypeFuncs::dtype_to_type_t<m_dtype>*

template<typename WrappedTypes, typename UnaryFunction, typename... Args>
inline auto ArrayVoid::execute_function(UnaryFunction&& unary_op, ArrayVoid& inp_arr, Args&&... args){
	constexpr DType m_dtype = WrappedTypes::next;
	if(m_dtype != dtype && WrappedTypes::done){
		using val_type = std::invoke_result_t<UnaryFunction, 
		      DTypeFuncs::dtype_to_type_t<m_dtype>*, 
		      DTypeFuncs::dtype_to_type_t<m_dtype>*,
		      DTypeFuncs::dtype_to_type_t<m_dtype>*, Args...>;
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
	using val_type = std::invoke_result_t<UnaryFunction, DTypeFuncs::dtype_to_type_t<m_dtype>*, DTypeFuncs::dtype_to_type_t<m_dtype>*,DTypeFuncs::dtype_to_type_t<m_dtype>*, Args...>;
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


//DTypeFuncs::dtype_to_type_t<dts...>*/DTypeFuncs::dtype_to_type_t<dts...>*
template<DType dt, DType...dts>
inline auto ArrayVoid::execute_function(bool throw_error, const char* func_name){
	bool check = throw_error ? DTypeFuncs::check_dtypes<dt, dts...>(func_name, dtype) : DTypeFuncs::is_in<dt, dts...>(dtype);
	return [throw_error, func_name, this](auto&& unary_op, auto&&... args) -> std::invoke_result_t<decltype(unary_op), DTypeFuncs::dtype_to_type_t<dts...>*, DTypeFuncs::dtype_to_type_t<dts...>*, decltype(args)...>{
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
	uint32_t type_a = bucket.iterator_type();
	if(type_a == 1){
		v = std::forward<UnaryFunction&&>(unary_op)(bucket.begin_contiguous<value_t>(),
			bucket.end_contiguous<value_t>(), std::forward<Args&&>(args)...);	
	}
	else if(type_a == 2){
		v = std::forward<UnaryFunction&&>(unary_op)(bucket.begin_blocked<value_t>(),
			bucket.end_blocked<value_t>(), std::forward<Args&&>(args)...);	
		
	}
	else if(type_a == 3){
		v = std::forward<UnaryFunction&&>(unary_op)(bucket.begin_list<value_t>(),
			bucket.end_list<value_t>(), std::forward<Args&&>(args)...);	
		
	}
}

template<DType dt, typename UnaryFunction, typename Output, typename... Args>
inline void ArrayVoid::sub_handle_execute_function(UnaryFunction&& unary_op, Output& v, bool& called, ArrayVoid& inp_arr, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	uint32_t type_a = bucket.iterator_type();
	uint32_t type_b = inp_arr.bucket.iterator_type();
	if(type_a == 1){
		auto begin = bucket.begin_contiguous<value_t>();
		auto end = bucket.end_contiguous<value_t>();
		if(type_b == 1){
			v = std::forward<UnaryFunction&&>(unary_op)(begin, end,
			inp_arr.bucket.begin_contiguous<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 2){
			v = std::forward<UnaryFunction&&>(unary_op)(begin, end,
			inp_arr.bucket.begin_blocked<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 3){
			v = std::forward<UnaryFunction&&>(unary_op)(begin, end,
			inp_arr.bucket.begin_list<value_t>(), std::forward<Args&&>(args)...);
		}

	}
	else if(type_a == 2){
		auto begin = bucket.begin_blocked<value_t>();
		auto end = bucket.end_blocked<value_t>();
		if(type_b == 1){
			v = std::forward<UnaryFunction&&>(unary_op)(begin, end,
			inp_arr.bucket.begin_contiguous<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 2){
			v = std::forward<UnaryFunction&&>(unary_op)(begin, end,
			inp_arr.bucket.begin_blocked<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 3){
			v = std::forward<UnaryFunction&&>(unary_op)(begin, end,
			inp_arr.bucket.begin_list<value_t>(), std::forward<Args&&>(args)...);
		}
	
		
	}
	else if(type_a == 3){
		auto begin = bucket.begin_list<value_t>();
		auto end = bucket.end_list<value_t>();
		if(type_b == 1){
			v = std::forward<UnaryFunction&&>(unary_op)(begin, end,
			inp_arr.bucket.begin_contiguous<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 2){
			v = std::forward<UnaryFunction&&>(unary_op)(begin, end,
			inp_arr.bucket.begin_blocked<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 3){
			v = std::forward<UnaryFunction&&>(unary_op)(begin, end,
			inp_arr.bucket.begin_list<value_t>(), std::forward<Args&&>(args)...);
		}
	
	}
}

template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_void(UnaryFunction&& unary_op, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	uint32_t type_a = bucket.iterator_type();
	if(type_a == 1){
		std::forward<UnaryFunction&&>(unary_op)(bucket.begin_contiguous<value_t>(),
			bucket.end_contiguous<value_t>(), std::forward<Args&&>(args)...);	
	}
	else if(type_a == 2){
		std::forward<UnaryFunction&&>(unary_op)(bucket.begin_blocked<value_t>(),
			bucket.end_blocked<value_t>(), std::forward<Args&&>(args)...);	
		
	}
	else if(type_a == 3){
		std::forward<UnaryFunction&&>(unary_op)(bucket.begin_list<value_t>(),
			bucket.end_list<value_t>(), std::forward<Args&&>(args)...);	
		
	}
}

template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_void(UnaryFunction&& unary_op, bool& called, ArrayVoid& inp_arr, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	uint32_t type_a = bucket.iterator_type();
	uint32_t type_b = inp_arr.bucket.iterator_type();
	if(type_a == 1){
		auto begin = bucket.begin_contiguous<value_t>();
		auto end = bucket.end_contiguous<value_t>();
		if(type_b == 1){
			std::forward<UnaryFunction&&>(unary_op)(begin, end,
			inp_arr.bucket.begin_contiguous<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 2){
			std::forward<UnaryFunction&&>(unary_op)(begin, end,
			inp_arr.bucket.begin_blocked<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 3){
			std::forward<UnaryFunction&&>(unary_op)(begin, end,
			inp_arr.bucket.begin_list<value_t>(), std::forward<Args&&>(args)...);
		}

	}
	else if(type_a == 2){
		auto begin = bucket.begin_blocked<value_t>();
		auto end = bucket.end_blocked<value_t>();
		if(type_b == 1){
			std::forward<UnaryFunction&&>(unary_op)(begin, end,
			inp_arr.bucket.begin_contiguous<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 2){
			std::forward<UnaryFunction&&>(unary_op)(begin, end,
			inp_arr.bucket.begin_blocked<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 3){
			std::forward<UnaryFunction&&>(unary_op)(begin, end,
			inp_arr.bucket.begin_list<value_t>(), std::forward<Args&&>(args)...);
		}
	
		
	}
	else if(type_a == 3){
		auto begin = bucket.begin_list<value_t>();
		auto end = bucket.end_list<value_t>();
		if(type_b == 1){
			std::forward<UnaryFunction&&>(unary_op)(begin, end,
			inp_arr.bucket.begin_contiguous<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 2){
			std::forward<UnaryFunction&&>(unary_op)(begin, end,
			inp_arr.bucket.begin_blocked<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 3){
			std::forward<UnaryFunction&&>(unary_op)(begin, end,
			inp_arr.bucket.begin_list<value_t>(), std::forward<Args&&>(args)...);
		}
	
	}
}

#ifdef USE_PARALLEL
template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_1(UnaryFunction&& unary_op, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	auto begin = bucket.begin<1, value_t>();
	tbb::parallel_for(tbb::blocked_range<uint64_t>(0, size),
			[&](const tbb::blocked_range<uint64_t>& r){
			std::invoke(unary_op, begin + r.begin(), begin + r.end(), std::forward<Args>(args)...);
			});

}

template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_3(UnaryFunction&& unary_op, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	/* uint32_t type_a = bucket.iterator_type(); */
	auto begin = bucket.begin<3, value_t>();
	tbb::parallel_for(tbb::blocked_range<uint64_t>(0, size),
			[&](const tbb::blocked_range<uint64_t>& r){
			std::invoke(unary_op, begin + r.begin(), begin + r.end(), std::forward<Args>(args)...);
			});

}

template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_2(UnaryFunction&& unary_op, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	//this is the blocked version
	auto begin = bucket.begin<2, value_t>();
	auto threadFunc = [&](value_t* b_ptr, value_t* e_ptr){
		std::invoke(unary_op, b_ptr, e_ptr, std::forward<Args>(args)...);
	};
	auto end = bucket.end<2, value_t>();

	uint64_t diff = block_diff(begin, end);
	if(diff == 0){
		DTypeFuncs::dtype_to_type_t<dt>* begin_p = (DTypeFuncs::dtype_to_type_t<dt>*)begin;
		tbb::parallel_for(tbb::blocked_range<uint64_t>(0, size),
			[&](const tbb::blocked_range<uint64_t>& r){
			std::invoke(unary_op, begin_p + r.begin(), begin_p + r.end(), std::forward<Args>(args)...);
			});

		return;
	}
	//the minimum amount of space between each list
	std::vector<std::pair<value_t*,value_t*>> bounds;
	bounds.reserve(diff);
	while(!same_block(begin, end)){
		value_t* p_begin = (value_t*)begin;
		value_t* p_end = begin.block_end();
		bounds.push_back({p_begin, p_end});
		begin = begin.get_next_block();
	}
	bounds.push_back({(DTypeFuncs::dtype_to_type_t<dt>*)begin, begin.block_end()});

	std::vector<std::thread> threads;
	for (const auto& b : bounds) {
		threads.emplace_back(threadFunc, b.first, b.second);
	}
	for (auto& t : threads) {
		t.join();
	}
}

#else
template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_1(UnaryFunction&& unary_op, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	auto begin = bucket.begin<1, value_t>();
	auto end = bucket.end<1, value_t>();
	std::invoke(unary_op, begin, end, std::forward<Args>(args)...);
}
template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_3(UnaryFunction&& unary_op, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	auto begin = bucket.begin<3, value_t>();
	auto end = bucket.end<3, value_t>();
	std::invoke(unary_op, begin, end, std::forward<Args>(args)...);
}

template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_2(UnaryFunction&& unary_op, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	//this is the blocked version
	auto begin = bucket.begin<2, value_t>();
	auto end = bucket.end<2, value_t>();
	uint64_t diff = block_diff(begin, end);
	if(diff == 0){
		DTypeFuncs::dtype_to_type_t<dt>* begin_p = (DTypeFuncs::dtype_to_type_t<dt>*)begin;
		DTypeFuncs::dtype_to_type_t<dt>* end_p = (DTypeFuncs::dtype_to_type_t<dt>*)end;
		std::invoke(unary_op, begin_p, end_p, std::forward<Args>(args)...);
		return;
	}
	//the minimum amount of space between each list

	std::vector<std::pair<value_t*,value_t*>> bounds;
	bounds.reserve(diff);
	while(!same_block(begin, end)){
		bounds.push_back({(DTypeFuncs::dtype_to_type_t<dt>*)begin, begin.block_end()});
		begin = begin.get_next_block();
	}
	bounds.push_back({(DTypeFuncs::dtype_to_type_t<dt>*)begin, begin.block_end()});

	for (const auto& b : bounds) {
		std::invoke(unary_op, b.first, b.second, std::forward<Args>(args)...);
	}
}

#endif

template<typename WrappedTypes, typename UnaryFunction, typename... Args, std::enable_if_t<
				DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value
				&& !DTypeFuncs::is_wrapped_dtype<UnaryFunction>::value, bool>>
inline void ArrayVoid::execute_function_chunk(UnaryFunction&& unary_op, Args&&... args){
	constexpr DType m_dtype = WrappedTypes::next;
	if(m_dtype != dtype && WrappedTypes::done){
		return;	
	}
	if(m_dtype != dtype){
		return execute_function_chunk<typename WrappedTypes::next_wrapper>(std::forward<UnaryFunction&&>(unary_op), std::forward<Args&&>(args)...);
	}
	bool check = false;
	uint32_t type_a = bucket.iterator_type();
	if(type_a == 1){
		sub_handle_execute_function_chunk_1<m_dtype>(std::forward<UnaryFunction&&>(unary_op), check, std::forward<Args&&>(args)...);
	}
	else if(type_a == 2){
		sub_handle_execute_function_chunk_2<m_dtype>(std::forward<UnaryFunction&&>(unary_op), check, std::forward<Args&&>(args)...);
	}
	else if(type_a == 3){
		sub_handle_execute_function_chunk_3<m_dtype>(std::forward<UnaryFunction&&>(unary_op), check, std::forward<Args&&>(args)...);
	}

}



#ifdef USE_PARALLEL
template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_1_1(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	auto begin = bucket.begin<1, value_t>();
	auto begin_2 = inp_arr.bucket.begin<1, value_t>();
	tbb::parallel_for(tbb::blocked_range<uint64_t>(0, size),
			[&](const tbb::blocked_range<uint64_t>& r){
			std::invoke(unary_op, begin + r.begin(), begin + r.end(), begin_2 + r.begin(), std::forward<Args>(args)...);
			});

}

template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_1_3(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	auto begin = bucket.begin<1, value_t>();
	auto begin_2 = inp_arr.bucket.begin<3, value_t>();
	tbb::parallel_for(tbb::blocked_range<uint64_t>(0, size),
			[&](const tbb::blocked_range<uint64_t>& r){
			std::invoke(unary_op, begin + r.begin(), begin + r.end(), begin_2 + r.begin(), std::forward<Args>(args)...);
			});

}

template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_3_3(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	auto begin = bucket.begin<3, value_t>();
	auto begin_2 = inp_arr.bucket.begin<3, value_t>();
	tbb::parallel_for(tbb::blocked_range<uint64_t>(0, size),
			[&](const tbb::blocked_range<uint64_t>& r){
			std::invoke(unary_op, begin + r.begin(), begin + r.end(), begin_2 + r.begin(), std::forward<Args>(args)...);
			});

}

template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_3_1(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	auto begin = bucket.begin<3, value_t>();
	auto begin_2 = inp_arr.bucket.begin<1, value_t>();
	tbb::parallel_for(tbb::blocked_range<uint64_t>(0, size),
			[&](const tbb::blocked_range<uint64_t>& r){
			std::invoke(unary_op, begin + r.begin(), begin + r.end(), begin_2 + r.begin(), std::forward<Args>(args)...);
			});

}


template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_2_1(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	//this is the blocked version
	auto begin_2 = inp_arr.bucket.begin<1, value_t>();
	auto begin = bucket.begin<2, value_t>();
	auto threadFunc = [&](value_t* b_ptr, value_t* e_ptr, value_t* b2_ptr){
		std::invoke(unary_op, b_ptr, e_ptr, b2_ptr, std::forward<Args>(args)...);
	};
	auto end = bucket.end<2, value_t>();

	uint64_t diff = block_diff(begin, end);
	if(diff == 0){
		DTypeFuncs::dtype_to_type_t<dt>* begin_p = (DTypeFuncs::dtype_to_type_t<dt>*)begin;
		tbb::parallel_for(tbb::blocked_range<uint64_t>(0, size),
			[&](const tbb::blocked_range<uint64_t>& r){
			std::invoke(unary_op, begin_p + r.begin(), begin_p + r.end(), begin_2 + r.begin(), std::forward<Args>(args)...);
			});

		return;
	}
	//the minimum amount of space between each list
	std::vector<std::thread> threads;
	while(!same_block(begin, end)){
		value_t* p_begin = (value_t*)begin;
		value_t* p_end = begin.block_end();
		threads.emplace_back(threadFunc, p_begin, p_end, begin_2);
		begin_2 += (p_end - p_begin);
		begin = begin.get_next_block();
	}
	threads.emplace_back(threadFunc, (DTypeFuncs::dtype_to_type_t<dt>*)begin, begin.block_end(), begin_2);


	for (auto& t : threads) {
		t.join();
	}
}


template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_2_3(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	//this is the blocked version
	auto begin_2 = inp_arr.bucket.begin<3, value_t>();
	auto begin = bucket.begin<2, value_t>();
	auto threadFunc = [&](value_t* b_ptr, value_t* e_ptr, BucketIterator_list<value_t> b2_ptr){
		std::invoke(unary_op, b_ptr, e_ptr, b2_ptr, std::forward<Args>(args)...);
	};
	auto end = bucket.end<2, value_t>();

	uint64_t diff = block_diff(begin, end);
	if(diff == 0){
		DTypeFuncs::dtype_to_type_t<dt>* begin_p = (DTypeFuncs::dtype_to_type_t<dt>*)begin;
		tbb::parallel_for(tbb::blocked_range<uint64_t>(0, size),
			[&](const tbb::blocked_range<uint64_t>& r){
			std::invoke(unary_op, begin_p + r.begin(), begin_p + r.end(), begin_2 + r.begin(), std::forward<Args>(args)...);
			});

		return;
	}
	//the minimum amount of space between each list
	std::vector<std::thread> threads;
	while(!same_block(begin, end)){
		value_t* p_begin = (value_t*)begin;
		value_t* p_end = begin.block_end();
		threads.emplace_back(threadFunc, p_begin, p_end, begin_2);
		begin_2 += (p_end - p_begin);
		begin = begin.get_next_block();
	}
	threads.emplace_back(threadFunc, (DTypeFuncs::dtype_to_type_t<dt>*)begin, begin.block_end(), begin_2);

	for (auto& t : threads) {
		t.join();
	}
}

template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_3_2(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	auto begin = bucket.begin<3, value_t>();
	auto begin_2 = inp_arr.bucket.begin<2, value_t>();
	auto end_2 = inp_arr.bucket.end<2, value_t>();
	auto threadFunc = [&](BucketIterator_list<value_t> b_ptr, BucketIterator_list<value_t> e_ptr, value_t* b2_ptr){
		std::invoke(unary_op, b_ptr, e_ptr, b2_ptr, std::forward<Args>(args)...);
	};
	auto end = bucket.end<3, value_t>();

	uint64_t diff = block_diff(begin_2, end_2);
	if(diff == 0){
		DTypeFuncs::dtype_to_type_t<dt>* begin_p = (DTypeFuncs::dtype_to_type_t<dt>*)begin_2;
		tbb::parallel_for(tbb::blocked_range<uint64_t>(0, size),
			[&](const tbb::blocked_range<uint64_t>& r){
			std::invoke(unary_op, begin + r.begin(), begin + r.end(), begin_p + r.begin(), std::forward<Args>(args)...);
			});

		return;
	}
	//the minimum amount of space between each list
	std::vector<std::thread> threads;
	while(!same_block(begin_2, end_2)){
		value_t* p_begin = (value_t*)begin_2;
		value_t* p_end = begin_2.block_end();
		threads.emplace_back(threadFunc, begin, begin + (p_end-p_begin), p_begin);
		begin += (p_end - p_begin);
		begin_2 = begin_2.get_next_block();
	}
	threads.emplace_back(threadFunc, begin, end, (value_t*)begin_2);

	for (auto& t : threads) {
		t.join();
	}

}
template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_1_2(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	auto begin = bucket.begin<1, value_t>();
	auto begin_2 = inp_arr.bucket.begin<2, value_t>();
	auto end_2 = inp_arr.bucket.end<2, value_t>();
	auto threadFunc = [&](value_t* b_ptr, value_t* e_ptr, value_t* b2_ptr){
		std::invoke(unary_op, b_ptr, e_ptr, b2_ptr, std::forward<Args>(args)...);
	};
	auto end = bucket.end<1, value_t>();

	uint64_t diff = block_diff(begin_2, end_2);
	if(diff == 0){
		DTypeFuncs::dtype_to_type_t<dt>* begin_p = (DTypeFuncs::dtype_to_type_t<dt>*)begin_2;
		tbb::parallel_for(tbb::blocked_range<uint64_t>(0, size),
			[&](const tbb::blocked_range<uint64_t>& r){
			std::invoke(unary_op, begin + r.begin(), begin + r.end(), begin_p + r.begin(), std::forward<Args>(args)...);
			});

		return;
	}
	//the minimum amount of space between each list
	std::vector<std::thread> threads;
	while(!same_block(begin_2, end_2)){
		value_t* p_begin = (value_t*)begin_2;
		value_t* p_end = begin_2.block_end();
		threads.emplace_back(threadFunc, begin, begin + (p_end-p_begin), p_begin);
		begin += (p_end - p_begin);
		begin_2 = begin_2.get_next_block();
	}
	threads.emplace_back(threadFunc, begin, end, (value_t*)begin_2);

	for (auto& t : threads) {
		t.join();
	}

}

template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_2_2(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	utils::THROW_EXCEPTION(inp_arr.size == size || inp_arr.bucket.buckets_amt() == bucket.buckets_amt(), "When chunking functions, the memory layout for 2 buckets specifies they must either have the same size, or the same number of buckets");
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	//this is the blocked version
	auto begin_2 = inp_arr.bucket.begin<2, value_t>();
	auto begin = bucket.begin<2, value_t>();
	auto end = bucket.end<2, value_t>();
	
	uint64_t diff_a = block_diff(begin, end);
	uint64_t diff_b = block_diff(begin_2, inp_arr.bucket.end<2,value_t>());
	auto threadFunc = [&](value_t* b_ptr, value_t* e_ptr, value_t* b2_ptr){
		std::invoke(unary_op, b_ptr, e_ptr, b2_ptr, std::forward<Args>(args)...);
	};
	std::vector<std::thread> threads;
	
	if(inp_arr.bucket.buckets_amt() == bucket.buckets_amt()){
		//it is not possible for them to have a bucket_size == 1, or to have a dif == 0, so skipping that part
		utils::THROW_EXCEPTION(diff_a == diff_b, "Expected to have same number of blocks for chunk operator");
		while(!same_block(begin, end)){
			value_t* p_begin = (value_t*)begin;
			value_t* p_end = begin.block_end();
			value_t* p_begin2 = (value_t*)begin_2;
			threads.emplace_back(threadFunc, p_begin, p_end, p_begin2);
			begin = begin.get_next_block();
			begin_2 = begin_2.get_next_block();
		}
		threads.emplace_back(threadFunc, (DTypeFuncs::dtype_to_type_t<dt>*)begin, begin.block_end(), (value_t*)begin_2);
	}
	else{
		auto end_2 = inp_arr.bucket.end<2, value_t>();
		value_t* p_begin = (value_t*)begin;
		value_t* p_end = begin.block_end();
		value_t* p_begin2 = (value_t*)begin_2;
		value_t* p_end2 = begin_2.block_end();
		while(!same_block(begin, end) || !same_block(begin_2, end_2)){
			std::ptrdiff_t dist_1 = p_end - p_begin;
			std::ptrdiff_t dist_2 = p_end2 - p_begin2;
			if(dist_1 == dist_2){
				threads.emplace_back(threadFunc, p_begin, p_end, p_begin2);
				begin = begin.get_next_block();
				begin_2 = begin_2.get_next_block();
				p_begin = (value_t*)begin;
				p_end = begin.block_end();
				p_begin2 = (value_t*)begin_2;
				p_end2 = begin_2.block_end();
				continue;
			}
			if(dist_1 > dist_2){
				threads.emplace_back(threadFunc, p_begin, p_begin + dist_2, p_begin2);
				p_begin += dist_2;
				begin_2 = begin_2.get_next_block();
				p_begin2 = (value_t*)begin_2;
				p_end2 = begin_2.block_end();
				continue;
			}
			//dist_2 > dist_1
			threads.emplace_back(threadFunc, p_begin, p_end, p_begin2);
			begin_2 += dist_1;
			begin = begin.get_next_block();
			p_begin = (value_t*)begin;
			p_end = begin.block_end();
		}
		utils::THROW_EXCEPTION(p_end == (value_t*)end && p_end2 == (value_t*)end_2, "Problem with chunking both same size logic");
		if(p_begin != p_end){
			threads.emplace_back(threadFunc, p_begin, p_end, p_begin2);
		}
		
	}

	for (auto& t : threads) {
		t.join();
	}
}



#else
template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_1_1(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	auto begin = bucket.begin<1, value_t>();
	auto end = bucket.end<1, value_t>();
	auto begin_2 = inp_arr.bucket.begin<1, value_t>();
	std::invoke(unary_op, begin, end, begin_2, std::forward<Args&&>(args)...);
}

template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_1_3(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	auto begin = bucket.begin<1, value_t>();
	auto end = bucket.end<1, value_t>();
	auto begin_2 = inp_arr.bucket.begin<3, value_t>();
	std::invoke(unary_op, begin, end, begin_2, std::forward<Args&&>(args)...);

}

template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_3_3(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	auto begin = bucket.begin<3, value_t>();
	auto end = bucket.end<3, value_t>();
	auto begin_2 = inp_arr.bucket.begin<3, value_t>();
	std::invoke(unary_op, begin, end, begin_2, std::forward<Args&&>(args)...);
}

template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_3_1(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	auto begin = bucket.begin<3, value_t>();
	auto end = bucket.end<3, value_t>();
	auto begin_2 = inp_arr.bucket.begin<1, value_t>();
	std::invoke(unary_op, begin, end, begin_2, std::forward<Args&&>(args)...);
}


template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_2_1(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	//this is the blocked version
	auto begin_2 = inp_arr.bucket.begin<1, value_t>();
	auto begin = bucket.begin<2, value_t>();
	auto end = bucket.end<2, value_t>();

	uint64_t diff = block_diff(begin, end);
	if(diff == 0){
		DTypeFuncs::dtype_to_type_t<dt>* begin_p = (DTypeFuncs::dtype_to_type_t<dt>*)begin;
		std::invoke(unary_op, begin_p, begin_p + size, begin_2, std::forward<Args&&>(args)...);
		return;
	}
	//the minimum amount of space between each list
	while(!same_block(begin, end)){
		value_t* p_begin = (value_t*)begin;
		value_t* p_end = begin.block_end();
		std::invoke(unary_op, p_begin, p_end, begin_2, std::forward<Args&&>(args)...);
		begin_2 += (p_end - p_begin);
		begin = begin.get_next_block();
	}
	std::invoke(unary_op, (DTypeFuncs::dtype_to_type_t<dt>*)begin, begin.block_end(), begin_2, std::forward<Args&&>(args)...);
}


template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_2_3(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	//this is the blocked version
	auto begin_2 = inp_arr.bucket.begin<3, value_t>();
	auto begin = bucket.begin<2, value_t>();
	auto end = bucket.end<2, value_t>();

	uint64_t diff = block_diff(begin, end);
	if(diff == 0){
		DTypeFuncs::dtype_to_type_t<dt>* begin_p = (DTypeFuncs::dtype_to_type_t<dt>*)begin;
		std::invoke(unary_op, begin_p, begin_p + size, begin_2, std::forward<Args&&>(args)...);
		return;
	}
	//the minimum amount of space between each list
	while(!same_block(begin, end)){
		value_t* p_begin = (value_t*)begin;
		value_t* p_end = begin.block_end();
		std::invoke(unary_op, p_begin, p_end, begin_2, std::forward<Args&&>(args)...);
		begin_2 += (p_end - p_begin);
		begin = begin.get_next_block();
	}
	std::invoke(unary_op, (DTypeFuncs::dtype_to_type_t<dt>*)begin, begin.block_end(), begin_2, std::forward<Args&&>(args)...);
}

template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_3_2(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	auto begin = bucket.begin<3, value_t>();
	auto begin_2 = inp_arr.bucket.begin<2, value_t>();
	auto end_2 = inp_arr.bucket.end<2, value_t>();
	auto end = bucket.end<3, value_t>();

	uint64_t diff = block_diff(begin_2, end_2);
	if(diff == 0){
		DTypeFuncs::dtype_to_type_t<dt>* begin_p = (DTypeFuncs::dtype_to_type_t<dt>*)begin_2;
		std::invoke(unary_op, begin, end, begin_p, std::forward<Args&&>(args)...);
		return;
	}
	//the minimum amount of space between each list
	while(!same_block(begin_2, end_2)){
		value_t* p_begin = (value_t*)begin_2;
		value_t* p_end = begin_2.block_end();
		std::invoke(unary_op, begin, begin + (p_end-p_begin), p_begin, std::forward<Args&&>(args)...);
		begin += (p_end - p_begin);
		begin_2 = begin_2.get_next_block();
	}
	std::invoke(unary_op, begin, end, (value_t*)begin_2, std::forward<Args&&>(args)...);
}
template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_1_2(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	auto begin = bucket.begin<1, value_t>();
	auto begin_2 = inp_arr.bucket.begin<2, value_t>();
	auto end_2 = inp_arr.bucket.end<2, value_t>();
	auto end = bucket.end<1, value_t>();

	uint64_t diff = block_diff(begin_2, end_2);
	if(diff == 0){
		DTypeFuncs::dtype_to_type_t<dt>* begin_p = (DTypeFuncs::dtype_to_type_t<dt>*)begin_2;
		std::invoke(unary_op, begin, end, begin_p, std::forward<Args&&>(args)...);
		return;
	}
	//the minimum amount of space between each list
	while(!same_block(begin_2, end_2)){
		value_t* p_begin = (value_t*)begin_2;
		value_t* p_end = begin_2.block_end();
		std::invoke(unary_op, begin, begin + (p_end-p_begin), p_begin, std::forward<Args&&>(args)...);
		begin += (p_end - p_begin);
		begin_2 = begin_2.get_next_block();
	}
	std::invoke(unary_op, begin, end, (value_t*)begin_2, std::forward<Args&&>(args)...);

}

template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_execute_function_chunk_2_2(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args){
	if(called || dt != dtype){return;}
	called = true;
	utils::THROW_EXCEPTION(inp_arr.size == size || inp_arr.bucket.buckets_amt() == bucket.buckets_amt(), "When chunking functions, the memory layout for 2 buckets specifies they must either have the same size, or the same number of buckets");
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	//this is the blocked version
	auto begin_2 = inp_arr.bucket.begin<2, value_t>();
	auto begin = bucket.begin<2, value_t>();
	auto end = bucket.end<2, value_t>();
	
	uint64_t diff_a = block_diff(begin, end);
	uint64_t diff_b = block_diff(begin_2, inp_arr.bucket.end<2,value_t>());
	
	if(inp_arr.bucket.buckets_amt() == bucket.buckets_amt()){
		//it is not possible for them to have a bucket_size == 1, or to have a dif == 0, so skipping that part
		utils::THROW_EXCEPTION(diff_a == diff_b, "Expected to have same number of blocks for chunk operator");
		while(!same_block(begin, end)){
			value_t* p_begin = (value_t*)begin;
			value_t* p_end = begin.block_end();
			value_t* p_begin2 = (value_t*)begin_2;
			std::invoke(unary_op, p_begin, p_end, p_begin2, std::forward<Args&&>(args)...);
			begin = begin.get_next_block();
			begin_2 = begin_2.get_next_block()
		}
		std::invoke(unary_op, (DTypeFuncs::dtype_to_type_t<dt>*)begin, begin.block_end(), (value_t*)begin_2, std::forward<Args&&>(args)...);
	}
	else{
		auto end_2 = inp_arr.bucket.end<2, value_t>();
		value_t* p_begin = (value_t*)begin;
		value_t* p_end = begin.block_end();
		value_t* p_begin2 = (value_t*)begin_2;
		value_t* p_end2 = begin_2.block_end();
		while(!same_block(begin, end) || !same_block(begin_2, end_2)){
			std::ptrdiff_t dist_1 = p_end - p_begin;
			std::ptrdiff_t dist_2 = p_end2 - p_begin2;
			if(dist_1 == dist_2){
				std::invoke(unary_op, p_begin, p_end, p_begin2, std::forward<Args&&>(args)...);
				begin = begin.get_next_block();
				begin_2 = begin_2.get_next_block();
				p_begin = (value_t*)begin;
				p_end = begin.block_end();
				p_begin2 = (value_t*)begin_2;
				p_end2 = begin_2.block_end();
				continue;
			}
			if(dist_1 > dist_2){
				std::invoke(unary_op, p_begin, p_begin + dist_2, p_begin2, std::forward<Args&&>(args)...);
				p_begin += dist_2;
				begin_2 = begin_2.get_next_block();
				p_begin2 = (value_t*)begin_2;
				p_end2 = begin_2.block_end();
				continue;
			}
			//dist_2 > dist_1
			std::invoke(unary_op, p_begin, p_end, p_begin2, std::forward<Args&&>(args)...);
			begin_2 += dist_1;
			begin = begin.get_next_block();
			p_begin = (value_t*)begin;
			p_end = begin.block_end();
		}
		utils::THROW_EXCEPTION(p_end == (value_t*)end && p_end2 == (value_t*)end2, "Problem with chunking both same size logic");
		if(p_begin != p_end){
			std::invoke(unary_op, p_begin, p_end, p_begin2, std::forward<Args&&>(args)...);
			threads.emplace_back(threadFunc, p_begin, p_end, p_begin2);
		}
		
	}
}

#endif



template<typename WrappedTypes, typename UnaryFunction, typename... Args, std::enable_if_t<
				DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value
				&& !DTypeFuncs::is_wrapped_dtype<UnaryFunction>::value, bool>>
inline void ArrayVoid::execute_function_chunk_execute(UnaryFunction&& unary_op, ArrayVoid& inp_arr, Args&&... args){
	constexpr DType m_dtype = WrappedTypes::next;
	if(m_dtype != dtype && WrappedTypes::done){
		return;	
	}
	if(m_dtype != dtype){
		return execute_function_chunk<typename WrappedTypes::next_wrapper>(std::forward<UnaryFunction&&>(unary_op), std::forward<Args&&>(args)...);
	}
	bool check = false;
	uint32_t type_a = bucket.iterator_type();
	uint32_t type_b = inp_arr.bucket.iterator_type();
	if(type_a == 1){
		if(type_b == 1){
			sub_handle_execute_function_chunk_1_1<m_dtype>(std::forward<UnaryFunction&&>(unary_op), inp_arr, check, std::forward<Args&&>(args)...);
		}
		else if(type_b == 2){
			sub_handle_execute_function_chunk_1_2<m_dtype>(std::forward<UnaryFunction&&>(unary_op), inp_arr, check, std::forward<Args&&>(args)...);
		}
		else if(type_b == 3){
			sub_handle_execute_function_chunk_1_3<m_dtype>(std::forward<UnaryFunction&&>(unary_op), inp_arr, check, std::forward<Args&&>(args)...);
		}
	}
	else if(type_a == 2){
		if(type_b == 1){
			sub_handle_execute_function_chunk_2_1<m_dtype>(std::forward<UnaryFunction&&>(unary_op), inp_arr, check, std::forward<Args&&>(args)...);
		}
		else if(type_b == 2){
			sub_handle_execute_function_chunk_2_2<m_dtype>(std::forward<UnaryFunction&&>(unary_op), inp_arr, check, std::forward<Args&&>(args)...);
		}
		else if(type_b == 3){
			sub_handle_execute_function_chunk_2_3<m_dtype>(std::forward<UnaryFunction&&>(unary_op), inp_arr, check, std::forward<Args&&>(args)...);
		}
	}
	else if(type_a == 3){
		if(type_b == 1){
			sub_handle_execute_function_chunk_3_1<m_dtype>(std::forward<UnaryFunction&&>(unary_op), inp_arr, check, std::forward<Args&&>(args)...);
		}
		else if(type_b == 2){
			sub_handle_execute_function_chunk_3_2<m_dtype>(std::forward<UnaryFunction&&>(unary_op), inp_arr, check, std::forward<Args&&>(args)...);
		}
		else if(type_b == 3){
			sub_handle_execute_function_chunk_3_3<m_dtype>(std::forward<UnaryFunction&&>(unary_op), inp_arr, check, std::forward<Args&&>(args)...);
		}
	}

}

/* #ifdef USE_PARALLEL */


/* template<DType dt, typename UnaryFunction, typename... Args> */
/* inline void ArrayVoid::sub_handle_execute_function_parallel_1(UnaryFunction&& unary_op, bool& called, Args&&... args){ */
/* 	if(called || dt != dtype){return;} */
/* 	called = true; */
/* 	using value_t = DTypeFuncs::dtype_to_type_t<dt>; */
/* 	auto begin = bucket.begin<1, value_t>(); */
/* 	tbb::parallel_for(tbb::blocked_range<uint64_t>(0, size), */
/* 			[&](const tbb::blocked_range<uint64_t>& r){ */
/* 			std::invoke(unary_op, begin + r.begin(), begin + r.end(), std::forward<Args>(args)...); */
/* 			}); */

/* } */
/* template<DType dt, typename UnaryFunction, typename... Args> */
/* inline void ArrayVoid::sub_handle_execute_function_parallel_2(UnaryFunction&& unary_op, bool& called, Args&&... args){ */
/* 	if(called || dt != dtype){return;} */
/* 	called = true; */
/* 	using value_t = DTypeFuncs::dtype_to_type_t<dt>; */
/* 	//this is the blocked version */
/* 	auto begin = bucket.begin<2, value_t>(); */
/* 	auto threadFunc = [&](value_t* b_ptr, value_t* e_ptr){ */
/* 		std::invoke(unary_op, b_ptr, e_ptr, std::forward<Args>(args)...); */
/* 	}; */
/* 	auto end = bucket.end<2, value_t>(); */
/* 	uint64_t diff = block_diff(begin, end); */
/* 	if(diff == 0){ */
/* 		DTypeFuncs::dtype_to_type_t<dt>* begin_p = (DTypeFuncs::dtype_to_type_t<dt>*)begin; */
/* 		tbb::parallel_for(tbb::blocked_range<uint64_t>(0, size), */
/* 			[&](const tbb::blocked_range<uint64_t>& r){ */
/* 			std::invoke(unary_op, begin + r.begin(), begin + r.end(), std::forward<Args>(args)...); */
/* 			}); */

/* 		return; */
/* 	} */
/* 	//the minimum amount of space between each list */
/* 	std::ptrdiff_t threshold; */
/* 	if(size <= 10){threshold = size;} */
/* 	else if(size < 500){threshold = size / 10;} */
/* 	else if(size < 5000){threshold = size / 50;} */
/* 	else if(size < 50000){threshold = size / 100;} */
/* 	else if(size < 500000){threshold = size / 100;} */
/* 	else if(size < 5000000){threshold = size / 100;} */
/* 	else {threshold = size / 1000;} */
/* 	std::vector<std::pair<value_t*,value_t*>> bounds; */
/* 	bounds.reserve(std::max(diff, size/threshold)); */
/* 	auto cur_end = begin.block_end(); */
/* 	if(begin.block_size() > threshold){ */
/* 		auto cur_begin = (DTypeFuncs::dtype_to_type_t<dt>*)begin + threshold; */
/* 		bounds.push_back({(DTypeFuncs::dtype_to_type_t<dt>*)begin, cur_begin}); */
/* 		while(cur_end - cur_begin > threshold){ */
/* 			bounds.push_back({cur_begin, cur_begin + threshold}); */
/* 			cur_begin += threshold; */
/* 		} */
/* 		if(cur_begin != cur_end){bounds.push_back({cur_begin, cur_end});} */
/* 	} */
/* 	else{ */
/* 		bounds.push_back({(DTypeFuncs::dtype_to_type_t<dt>*)begin, cur_end}); */
/* 	} */

/* 	auto cpy_begin = begin.get_next_block(); */
/* 	cur_end = cpy_begin.block_end(); */
/* 	if(same_block(cpy_begin, end)){ */
/* 		if(cpy_begin.block_size() > threshold){ */
/* 			auto cur_begin = (DTypeFuncs::dtype_to_type_t<dt>*)cpy_begin + threshold; */
/* 			bounds.push_back({(DTypeFuncs::dtype_to_type_t<dt>*)cpy_begin, cur_begin}); */
/* 			while(cur_end - cur_begin > threshold){ */
/* 				bounds.push_back({cur_begin, cur_begin + threshold}); */
/* 				cur_begin += threshold; */
/* 			} */
/* 			if(cur_begin != cur_end){bounds.push_back({cur_begin, cur_end});} */
/* 		} */
/* 		else{ */
/* 			bounds.push_back({(DTypeFuncs::dtype_to_type_t<dt>*)cpy_begin, cur_end}); */
/* 		} */
/* 	} */
/* 	while(!same_block(cpy_begin, end)){ */
/* 		cpy_begin = cpy_begin.get_next_block(); */
/* 		cur_end = cpy_begin.block_end(); */
/* 		if(cpy_begin.block_size() > threshold){ */
/* 			auto cur_begin = (DTypeFuncs::dtype_to_type_t<dt>*)cpy_begin + threshold; */
/* 			bounds.push_back({(DTypeFuncs::dtype_to_type_t<dt>*)cpy_begin, cur_begin}); */
/* 			while(cur_end - cur_begin > threshold){ */
/* 				bounds.push_back({cur_begin, cur_begin + threshold}); */
/* 				cur_begin += threshold; */
/* 			} */
/* 			if(cur_begin != cur_end){bounds.push_back({cur_begin, cur_end});} */
/* 		} */
/* 		else{ */
/* 			bounds.push_back({(DTypeFuncs::dtype_to_type_t<dt>*)cpy_begin, cur_end}); */
/* 		} */

/* 	} */
/* 	std::vector<std::thread> threads; */
/* 	for (const auto& b : bounds) { */
/* 		threads.emplace_back(threadFunc, b.first, b.second); */
/* 	} */
/* 	for (auto& t : threads) { */
/* 		t.join(); */
/* 	} */
/* } */

/* template<DType dt, typename UnaryFunction, typename... Args> */
/* inline void ArrayVoid::sub_handle_execute_function_parallel_3(UnaryFunction&& unary_op, bool& called, Args&&... args){ */
/* 	if(called || dt != dtype){return;} */
/* 	called = true; */
/* 	using value_t = DTypeFuncs::dtype_to_type_t<dt>; */
/* 	/1* uint32_t type_a = bucket.iterator_type(); *1/ */
/* 	auto begin = bucket.begin<3, value_t>(); */
/* 	tbb::parallel_for(tbb::blocked_range<uint64_t>(0, size), */
/* 			[&](const tbb::blocked_range<uint64_t>& r){ */
/* 			std::invoke(unary_op, begin + r.begin(), begin + r.end(), std::forward<Args>(args)...); */
/* 			}); */

/* } */


/* template<typename WrappedTypes, typename UnaryFunction, typename... Args, std::enable_if_t< */
/* 				DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value */
/* 				&& !DTypeFuncs::is_wrapped_dtype<UnaryFunction>::value, bool>> */
/* inline void ArrayVoid::execute_function_parallel(UnaryFunction&& unary_op, Args&&... args){ */
/* 	constexpr DType m_dtype = WrappedTypes::next; */
/* 	if(m_dtype != dtype && WrappedTypes::done){ */
/* 		return; */	
/* 	} */
/* 	if(m_dtype != dtype){ */
/* 		return execute_function_parallel<typename WrappedTypes::next_wrapper>(std::forward<UnaryFunction&&>(unary_op), std::forward<Args&&>(args)...); */
/* 	} */
/* 	bool check = false; */
/* 	uint32_t type_a = bucket.iterator_type(); */
/* 	if(type_a == 1){ */
/* 		sub_handle_execute_function_void_parallel_1<m_dtype, UnaryFunction, Args...>(std::forward<UnaryFunction&&>(unary_op), check, std::forward<Args&&>(args)...); */
/* 	} */
/* 	else if(type_a == 2){ */
/* 		sub_handle_execute_function_void_parallel_2<m_dtype, UnaryFunction, Args...>(std::forward<UnaryFunction&&>(unary_op), check, std::forward<Args&&>(args)...); */
/* 	} */
/* 	else if(type_a == 3){ */
/* 		sub_handle_execute_function_void_parallel_3<m_dtype, UnaryFunction, Args...>(std::forward<UnaryFunction&&>(unary_op), check, std::forward<Args&&>(args)...); */
/* 	} */

/* } */
/* #endif */
//const DTypeFuncs::dtype_to_type_t<dts...>*/const DTypeFuncs::dtype_to_type_t<dts...>*
template<DType dt, DType... dts, typename UnaryFunction, typename... Args>
inline auto ArrayVoid::cexecute_function(UnaryFunction&& unary_op, Args&&... args) const{
	using val_type = std::invoke_result_t<UnaryFunction, const DTypeFuncs::dtype_to_type_t<dts...>*, const DTypeFuncs::dtype_to_type_t<dts...>*, Args...>;
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


//const DTypeFuncs::dtype_to_type_t<m_dtype>*/const DTypeFuncs::dtype_to_type_t<m_dtype>*
template<typename WrappedTypes, typename UnaryFunction, typename... Args, std::enable_if_t<
					DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value
					&& !DTypeFuncs::is_wrapped_dtype<UnaryFunction>::value, bool>>
inline auto ArrayVoid::cexecute_function(UnaryFunction&& unary_op, Args&&... args) const{
	constexpr DType m_dtype = WrappedTypes::next;
	if(m_dtype != dtype && WrappedTypes::done){
		using val_type = std::invoke_result_t<UnaryFunction, const DTypeFuncs::dtype_to_type_t<m_dtype>*, const DTypeFuncs::dtype_to_type_t<m_dtype>*, Args...>;
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
	using val_type = std::invoke_result_t<UnaryFunction, const DTypeFuncs::dtype_to_type_t<m_dtype>*, const DTypeFuncs::dtype_to_type_t<m_dtype>*, Args...>;

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

//const DTypeFuncs::dtype_to_type_t<dts...>*/const DTypeFuncs::dtype_to_type_t<dts...>*
template<DType dt, DType... dts, typename UnaryFunction, typename... Args>
inline auto ArrayVoid::cexecute_function(UnaryFunction&& unary_op, const ArrayVoid& inp_arr, Args&&... args) const{
	using val_type = std::invoke_result_t<UnaryFunction, const DTypeFuncs::dtype_to_type_t<dts...>*, const DTypeFuncs::dtype_to_type_t<dts...>*, const DTypeFuncs::dtype_to_type_t<dts...>*, Args...>;
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

//const DTypeFuncs::dtype_to_type_t<m_dtype>*/const DTypeFuncs::dtype_to_type_t<m_dtype>*

template<typename WrappedTypes, typename UnaryFunction, typename... Args>
inline auto ArrayVoid::cexecute_function(UnaryFunction&& unary_op, const ArrayVoid& inp_arr, Args&&... args) const{
	constexpr DType m_dtype = WrappedTypes::next;
	if(m_dtype != dtype && WrappedTypes::done){
		using val_type = std::invoke_result_t<UnaryFunction, const DTypeFuncs::dtype_to_type_t<m_dtype>*, const DTypeFuncs::dtype_to_type_t<m_dtype>*, const DTypeFuncs::dtype_to_type_t<m_dtype>*, Args...>;
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
	using val_type = std::invoke_result_t<UnaryFunction, const DTypeFuncs::dtype_to_type_t<m_dtype>*, const DTypeFuncs::dtype_to_type_t<m_dtype>*, const DTypeFuncs::dtype_to_type_t<m_dtype>*, Args...>;
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
	return [throw_error, func_name, this](auto&& unary_op, auto&&... args) -> std::invoke_result_t<decltype(unary_op), const DTypeFuncs::dtype_to_type_t<dts...>*, const DTypeFuncs::dtype_to_type_t<dts...>*, decltype(args)...>{
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
	uint32_t type_a = bucket.iterator_type();
	if(type_a == 1){
		v = std::forward<UnaryFunction&&>(unary_op)(bucket.cbegin_contiguous<value_t>(),
			bucket.cend_contiguous<value_t>(), std::forward<Args&&>(args)...);	
	}
	else if(type_a == 2){
		v = std::forward<UnaryFunction&&>(unary_op)(bucket.cbegin_blocked<value_t>(),
			bucket.cend_blocked<value_t>(), std::forward<Args&&>(args)...);	
		
	}
	else if(type_a == 3){
		v = std::forward<UnaryFunction&&>(unary_op)(bucket.cbegin_list<value_t>(),
			bucket.cend_list<value_t>(), std::forward<Args&&>(args)...);	
		
	}
}

template<DType dt, typename UnaryFunction, typename Output, typename... Args>
inline void ArrayVoid::sub_handle_cexecute_function(UnaryFunction&& unary_op, Output& v, bool& called, const ArrayVoid& inp_arr, Args&&... args) const{
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	uint32_t type_a = bucket.iterator_type();
	uint32_t type_b = inp_arr.bucket.iterator_type();
	if(type_a == 1){
		auto begin = bucket.cbegin_contiguous<value_t>();
		auto end = bucket.cend_contiguous<value_t>();
		if(type_b == 1){
			v = std::forward<UnaryFunction&&>(unary_op)(begin, end,
				inp_arr.bucket.cend_contiguous<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 2){
			v = std::forward<UnaryFunction&&>(unary_op)(begin, end,
				inp_arr.bucket.cend_blocked<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 3){
			v = std::forward<UnaryFunction&&>(unary_op)(begin, end,
				inp_arr.bucket.cend_list<value_t>(), std::forward<Args&&>(args)...);
		}

	}
	else if(type_a == 2){
		auto begin = bucket.cbegin_blocked<value_t>();
		auto end = bucket.cend_blocked<value_t>();
		if(type_b == 1){
			v = std::forward<UnaryFunction&&>(unary_op)(begin, end,
				inp_arr.bucket.cend_contiguous<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 2){
			v = std::forward<UnaryFunction&&>(unary_op)(begin, end,
				inp_arr.bucket.cend_blocked<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 3){
			v = std::forward<UnaryFunction&&>(unary_op)(begin, end,
				inp_arr.bucket.cend_list<value_t>(), std::forward<Args&&>(args)...);
		}
	
		
	}
	else if(type_a == 3){
		auto begin = bucket.cbegin_list<value_t>();
		auto end = bucket.cend_list<value_t>();
		if(type_b == 1){
			v = std::forward<UnaryFunction&&>(unary_op)(begin, end,
				inp_arr.bucket.cend_contiguous<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 2){
			v = std::forward<UnaryFunction&&>(unary_op)(begin, end,
				inp_arr.bucket.cend_blocked<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 3){
			v = std::forward<UnaryFunction&&>(unary_op)(begin, end,
				inp_arr.bucket.cend_list<value_t>(), std::forward<Args&&>(args)...);
		}
		
	}
}


/*

std::forward<UnaryFunction&&>(unary_op)((type_a == 1) ? bucket.cbegin_contiguous<value_t>() : (type_a == 2) ? bucket.cbegin_blocked<value_t>() : bucket.cbegin_list<value_t>(),\r
\t\t\t(type_a == 1) ? bucket.cend_contiguous<value_t>() : (type_a == 2) ? bucket.cend_blocked<value_t>() : bucket.cend_list<value_t>(), std::forward<Args&&>(args)...);/

*/

template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_cexecute_function_void(UnaryFunction&& unary_op, bool& called, Args&&... args) const{
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	uint32_t type_a = bucket.iterator_type();
	if(type_a == 1){
		std::forward<UnaryFunction&&>(unary_op)(bucket.cbegin_contiguous<value_t>(),
			bucket.cend_contiguous<value_t>(), std::forward<Args&&>(args)...);	
	}
	else if(type_a == 2){
		std::forward<UnaryFunction&&>(unary_op)(bucket.cbegin_blocked<value_t>(),
			bucket.cend_blocked<value_t>(), std::forward<Args&&>(args)...);	
		
	}
	else if(type_a == 3){
		std::forward<UnaryFunction&&>(unary_op)(bucket.cbegin_list<value_t>(),
			bucket.cend_list<value_t>(), std::forward<Args&&>(args)...);	
		
	}
}

template<DType dt, typename UnaryFunction, typename... Args>
inline void ArrayVoid::sub_handle_cexecute_function_void(UnaryFunction&& unary_op, bool& called, const ArrayVoid& inp_arr, Args&&... args) const{
	if(called || dt != dtype){return;}
	called = true;
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	uint32_t type_a = bucket.iterator_type();
	uint32_t type_b = inp_arr.bucket.iterator_type();
	if(type_a == 1){
		if(type_b == 1){
			std::forward<UnaryFunction&&>(unary_op)(bucket.cbegin_contiguous<value_t>(),
				bucket.cend_contiguous<value_t>(), inp_arr.bucket.cbegin_contiguous<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 2){
			std::forward<UnaryFunction&&>(unary_op)(bucket.cbegin_contiguous<value_t>(),
				bucket.cend_contiguous<value_t>(), inp_arr.bucket.cbegin_blocked<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 3){
			std::forward<UnaryFunction&&>(unary_op)(bucket.cbegin_contiguous<value_t>(),
				bucket.cend_contiguous<value_t>(), inp_arr.bucket.cbegin_list<value_t>(), std::forward<Args&&>(args)...);
		}
	}
	else if(type_a == 2){
		if(type_b == 1){
			std::forward<UnaryFunction&&>(unary_op)(bucket.cbegin_blocked<value_t>(),
			bucket.cend_blocked<value_t>(), inp_arr.bucket.cbegin_contiguous<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 2){
			std::forward<UnaryFunction&&>(unary_op)(bucket.cbegin_blocked<value_t>(),
			bucket.cend_blocked<value_t>(), inp_arr.bucket.cbegin_blocked<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 3){
			std::forward<UnaryFunction&&>(unary_op)(bucket.cbegin_blocked<value_t>(),
			bucket.cend_blocked<value_t>(), inp_arr.bucket.cbegin_list<value_t>(), std::forward<Args&&>(args)...);
		}

		
	}
	else if(type_a == 3){
		if(type_b == 1){
			std::forward<UnaryFunction&&>(unary_op)(bucket.cbegin_list<value_t>(),
			bucket.cend_list<value_t>(), inp_arr.bucket.cbegin_contiguous<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 2){
			std::forward<UnaryFunction&&>(unary_op)(bucket.cbegin_list<value_t>(),
			bucket.cend_list<value_t>(), inp_arr.bucket.cbegin_blocked<value_t>(), std::forward<Args&&>(args)...);
		}
		else if(type_b == 3){
			std::forward<UnaryFunction&&>(unary_op)(bucket.cbegin_list<value_t>(),
			bucket.cend_list<value_t>(), inp_arr.bucket.cbegin_list<value_t>(), std::forward<Args&&>(args)...);
		}
		
	}
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
	uint32_t type_a = bucket.iterator_type();
	if(type_a == 1){
		std::transform(bucket.cbegin_contiguous<value_t>(),
			bucket.cend_contiguous<value_t>(), d_first, std::forward<UnaryOperation&&>(unary_op));	
	}
	else if(type_a == 2){
		std::transform(bucket.cbegin_blocked<value_t>(),
			bucket.cend_blocked<value_t>(), d_first, std::forward<UnaryOperation&&>(unary_op));	
		
	}
	else if(type_a == 3){
		std::transform(bucket.cbegin_list<value_t>(),
			bucket.cend_list<value_t>(), d_first, std::forward<UnaryOperation&&>(unary_op));	
		
	}
}

template<DType dt, class InputIt2, class OutputIt, class UnaryOperation>
inline void ArrayVoid::sub_transform_function(UnaryOperation&& unary_op, InputIt2& inp2, OutputIt& d_first) const{
	if(dt != dtype){return;}
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	uint32_t type_a = bucket.iterator_type();
	if(type_a == 1){
		std::transform(bucket.cbegin_contiguous<value_t>(),
			bucket.cend_contiguous<value_t>(), inp_2, d_first, std::forward<UnaryOperation&&>(unary_op));	
	}
	else if(type_a == 2){
		std::transform(bucket.cbegin_blocked<value_t>(),
			bucket.cend_blocked<value_t>(),inp_2, d_first, std::forward<UnaryOperation&&>(unary_op));	
		
	}
	else if(type_a == 3){
		std::transform(bucket.cbegin_list<value_t>(),
			bucket.cend_list<value_t>(), inp_2, d_first, std::forward<UnaryOperation&&>(unary_op));	
		
	}
}


template<DType dt, class OutputIt, class UnaryOperation>
inline void ArrayVoid::sub_transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, OutputIt& d_first) const{
	if(dt != dtype){return;}
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	uint32_t type_a = bucket.iterator_type();
	uint32_t type_b = inp_arr.bucket.iterator_type();
	if(type_a == 1){
		if(type_b == 1){
			std::transform(bucket.cbegin_contiguous<value_t>(),
				bucket.cend_contiguous<value_t>(), inp_arr.bucket.cbegin_contiguous<value_t>(), d_first, std::forward<UnaryOperation&&>(unary_op));
		}
		else if(type_b == 2){
			std::transform(unary_op)(bucket.cbegin_contiguous<value_t>(),
				bucket.cend_contiguous<value_t>(), inp_arr.bucket.cbegin_blocked<value_t>(), d_first, std::forward<UnaryOperation&&>(unary_op));
		}
		else if(type_b == 3){
			std::transform(unary_op)(bucket.cbegin_contiguous<value_t>(),
				bucket.cend_contiguous<value_t>(), inp_arr.bucket.cbegin_list<value_t>(), d_first, std::forward<UnaryOperation&&>(unary_op));
		}
	}
	else if(type_a == 2){
		if(type_b == 1){
			std::transform(unary_op)(bucket.cbegin_blocked<value_t>(),
			bucket.cend_blocked<value_t>(), inp_arr.bucket.cbegin_contiguous<value_t>(), d_first, std::forward<UnaryOperation&&>(unary_op));
		}
		else if(type_b == 2){
			std::transform(unary_op)(bucket.cbegin_blocked<value_t>(),
			bucket.cend_blocked<value_t>(), inp_arr.bucket.cbegin_blocked<value_t>(), d_first, std::forward<UnaryOperation&&>(unary_op));
		}
		else if(type_b == 3){
			std::transform(unary_op)(bucket.cbegin_blocked<value_t>(),
			bucket.cend_blocked<value_t>(), inp_arr.bucket.cbegin_list<value_t>(), d_first, std::forward<UnaryOperation&&>(unary_op));
		}

		
	}
	else if(type_a == 3){
		if(type_b == 1){
			std::transform(unary_op)(bucket.cbegin_list<value_t>(),
			bucket.cend_list<value_t>(), inp_arr.bucket.cbegin_contiguous<value_t>(), d_first, std::forward<UnaryOperation&&>(unary_op));
		}
		else if(type_b == 2){
			std::transform(unary_op)(bucket.cbegin_list<value_t>(),
			bucket.cend_list<value_t>(), inp_arr.bucket.cbegin_blocked<value_t>(), d_first, std::forward<UnaryOperation&&>(unary_op));
		}
		else if(type_b == 3){
			std::transform(unary_op)(bucket.cbegin_list<value_t>(),
			bucket.cend_list<value_t>(), inp_arr.bucket.cbegin_list<value_t>(), d_first, std::forward<UnaryOperation&&>(unary_op));
		}
		
	}
}

template<DType dt, class UnaryOperation>
inline void ArrayVoid::sub_transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr){
	if(dt != dtype){return;}
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	uint32_t type_a = bucket.iterator_type();
	uint32_t type_b = inp_arr.bucket.iterator_type();
	if(type_a == 1){
		if(type_b == 1){
			std::transform(bucket.cbegin_contiguous<value_t>(),
				bucket.cend_contiguous<value_t>(), inp_arr.bucket.cbegin_contiguous<value_t>(), bucket.begin_contiguous<value_t>(), std::forward<UnaryOperation&&>(unary_op));
		}
		else if(type_b == 2){
			std::transform(unary_op)(bucket.cbegin_contiguous<value_t>(),
				bucket.cend_contiguous<value_t>(), inp_arr.bucket.cbegin_blocked<value_t>(), bucket.begin_contiguous<value_t>(), std::forward<UnaryOperation&&>(unary_op));
		}
		else if(type_b == 3){
			std::transform(unary_op)(bucket.cbegin_contiguous<value_t>(),
				bucket.cend_contiguous<value_t>(), inp_arr.bucket.cbegin_list<value_t>(), bucket.begin_contiguous<value_t>(), std::forward<UnaryOperation&&>(unary_op));
		}
	}
	else if(type_a == 2){
		if(type_b == 1){
			std::transform(unary_op)(bucket.cbegin_blocked<value_t>(),
			bucket.cend_blocked<value_t>(), inp_arr.bucket.cbegin_contiguous<value_t>(), bucket.begin_blocked<value_t>(), std::forward<UnaryOperation&&>(unary_op));
		}
		else if(type_b == 2){
			std::transform(unary_op)(bucket.cbegin_blocked<value_t>(),
			bucket.cend_blocked<value_t>(), inp_arr.bucket.cbegin_blocked<value_t>(), bucket.begin_blocked<value_t>(), std::forward<UnaryOperation&&>(unary_op));
		}
		else if(type_b == 3){
			std::transform(unary_op)(bucket.cbegin_blocked<value_t>(),
			bucket.cend_blocked<value_t>(), inp_arr.bucket.cbegin_list<value_t>(), bucket.begin_blocked<value_t>(), std::forward<UnaryOperation&&>(unary_op));
		}

		
	}
	else if(type_a == 3){
		if(type_b == 1){
			std::transform(unary_op)(bucket.cbegin_list<value_t>(),
			bucket.cend_list<value_t>(), inp_arr.bucket.cbegin_contiguous<value_t>(), bucket.begin_list<value_t>(), std::forward<UnaryOperation&&>(unary_op));
		}
		else if(type_b == 2){
			std::transform(unary_op)(bucket.cbegin_list<value_t>(),
			bucket.cend_list<value_t>(), inp_arr.bucket.cbegin_blocked<value_t>(), bucket.begin_list<value_t>(), std::forward<UnaryOperation&&>(unary_op));
		}
		else if(type_b == 3){
			std::transform(unary_op)(bucket.cbegin_list<value_t>(),
			bucket.cend_list<value_t>(), inp_arr.bucket.cbegin_list<value_t>(), bucket.begin_list<value_t>(), std::forward<UnaryOperation&&>(unary_op));
		}
		
	}
}

template<DType dt, class UnaryOperation>
inline void ArrayVoid::sub_for_ceach(UnaryOperation&& unary_op) const{
	if(dt != dtype){return;}
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	uint32_t type_a = bucket.iterator_type();
	if(type_a == 1){
		std::for_each(bucket.cbegin_contiguous<value_t>(),
			bucket.cend_contiguous<value_t>(),std::forward<UnaryOperation&&>(unary_op));	
	}
	else if(type_a == 2){
		std::for_each(bucket.cbegin_blocked<value_t>(),
			bucket.cend_blocked<value_t>(), std::forward<UnaryOperation&&>(unary_op));	
		
	}
	else if(type_a == 3){
		std::for_each(bucket.cbegin_list<value_t>(),
			bucket.cend_list<value_t>(), std::forward<UnaryOperation&&>(unary_op));	
		
	}
}

template<DType dt, class UnaryOperation>
inline void ArrayVoid::sub_for_each(UnaryOperation&& unary_op){
	if(dt != dtype){return;}
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	uint32_t type_a = bucket.iterator_type();
	if(type_a == 1){
		std::for_each(bucket.begin_contiguous<value_t>(),
			bucket.end_contiguous<value_t>(),std::forward<UnaryOperation&&>(unary_op));	
	}
	else if(type_a == 2){
		std::for_each(bucket.begin_blocked<value_t>(),
			bucket.end_blocked<value_t>(), std::forward<UnaryOperation&&>(unary_op));	
		
	}
	else if(type_a == 3){
		std::for_each(bucket.begin_list<value_t>(),
			bucket.end_list<value_t>(), std::forward<UnaryOperation&&>(unary_op));	
		
	}
}



#else

template<DType dt, class OutputIt, class UnaryOperation>
inline void ArrayVoid::sub_transform_function(UnaryOperation&& unary_op, OutputIt& d_first) const{
	if(dt != dtype){return;}
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	uint32_t type_a = bucket.iterator_type();
	if(type_a == 1){
		auto start = bucket.cbegin_contiguous<value_t>();
		tbb::parallel_for(tbb::blocked_range<uint64_t>(0, size),
				[&](tbb::blocked_range<uint64_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation&&>(unary_op));});
	}
	else if(type_a == 2){
		auto start = bucket.cbegin_blocked<value_t>();
		tbb::parallel_for(tbb::blocked_range<uint64_t>(0, size),
				[&](tbb::blocked_range<uint64_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation&&>(unary_op));});

	}
	else{
		auto start = bucket.cbegin_list<value_t>();
		tbb::parallel_for(tbb::blocked_range<uint64_t>(0, size),
				[&](tbb::blocked_range<uint64_t> r){std::transform(start + r.begin(), start + r.end(), d_first + r.begin(), std::forward<UnaryOperation&&>(unary_op));});
	}

}

template<DType dt, class InputIt2, class OutputIt, class UnaryOperation>
inline void ArrayVoid::sub_transform_function(UnaryOperation&& unary_op, InputIt2& inp2, OutputIt& d_first) const{
	if(dt != dtype){return;}
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	uint32_t type_a = bucket.iterator_type();
	if(type_a == 1){
		auto start = bucket.cbegin_contiguous<value_t>();
		tbb::parallel_for(tbb::blocked_range<int64_t>(0, size),
				[&](tbb::blocked_range<int64_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), std::forward<UnaryOperation&&>(unary_op));});
	}
	else if(type_a == 2){
		auto start = bucket.cbegin_blocked<value_t>();
		tbb::parallel_for(tbb::blocked_range<int64_t>(0, size),
			[&](tbb::blocked_range<int64_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), std::forward<UnaryOperation&&>(unary_op));});

	}
	else{
		auto start = bucket.cbegin_list<value_t>();
		tbb::parallel_for(tbb::blocked_range<int64_t>(0, size),
			[&](tbb::blocked_range<int64_t> r){std::transform(start + r.begin(), start + r.end(), inp2 + r.begin(), d_first + r.begin(), std::forward<UnaryOperation&&>(unary_op));});
	}
}


template<DType dt, class OutputIt, class UnaryOperation>
inline void ArrayVoid::sub_transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, OutputIt& d_first) const{
	if(dt != dtype){return;}
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	uint32_t type_a = bucket.iterator_type();
	uint32_t type_b = inp_arr.bucket.iterator_type();
	if(type_a == 1){
		auto start = bucket.cbegin_contiguous<value_t>();
		if(type_b == 1){
			auto inp_start = inp_arr.bucket.cbegin_contiguous<value_t>();
			tbb::parallel_for(tbb::blocked_range<int64_t>(0, size),
				[&](tbb::blocked_range<int64_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), std::forward<UnaryOperation&&>(unary_op));});
		}
		else if(type_b == 2){
			auto inp_start = inp_arr.bucket.cbegin_blocked<value_t>();
			tbb::parallel_for(tbb::blocked_range<int64_t>(0, size),
				[&](tbb::blocked_range<int64_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), std::forward<UnaryOperation&&>(unary_op));});
		}
		else{
			auto inp_start = inp_arr.bucket.cbegin_list<value_t>();
			tbb::parallel_for(tbb::blocked_range<int64_t>(0, size),
				[&](tbb::blocked_range<int64_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), std::forward<UnaryOperation&&>(unary_op));});
		}
	}
	else if(type_a == 2){
		auto start = bucket.cbegin_blocked<value_t>();
		if(type_b == 1){
			auto inp_start = inp_arr.bucket.cbegin_contiguous<value_t>();
			tbb::parallel_for(tbb::blocked_range<int64_t>(0, size),
				[&](tbb::blocked_range<int64_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), std::forward<UnaryOperation&&>(unary_op));});
		}
		else if(type_b == 2){
			auto inp_start = inp_arr.bucket.cbegin_blocked<value_t>();
			tbb::parallel_for(tbb::blocked_range<int64_t>(0, size),
				[&](tbb::blocked_range<int64_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), std::forward<UnaryOperation&&>(unary_op));});
		}
		else{
			auto inp_start = inp_arr.bucket.cbegin_list<value_t>();
			tbb::parallel_for(tbb::blocked_range<int64_t>(0, size),
				[&](tbb::blocked_range<int64_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), std::forward<UnaryOperation&&>(unary_op));});
		}

	}
	else{
		auto start = bucket.cbegin_list<value_t>();
		if(type_b == 1){
			auto inp_start = inp_arr.bucket.cbegin_contiguous<value_t>();
			tbb::parallel_for(tbb::blocked_range<int64_t>(0, size),
				[&](tbb::blocked_range<int64_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), std::forward<UnaryOperation&&>(unary_op));});
		}
		else if(type_b == 2){
			auto inp_start = inp_arr.bucket.cbegin_blocked<value_t>();
			tbb::parallel_for(tbb::blocked_range<int64_t>(0, size),
				[&](tbb::blocked_range<int64_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), std::forward<UnaryOperation&&>(unary_op));});
		}
		else{
			auto inp_start = inp_arr.bucket.cbegin_list<value_t>();
			tbb::parallel_for(tbb::blocked_range<int64_t>(0, size),
				[&](tbb::blocked_range<int64_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), d_first + r.begin(), std::forward<UnaryOperation&&>(unary_op));});
		}
	}
}

template<DType dt, class UnaryOperation>
inline void ArrayVoid::sub_transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr){
	if(dt != dtype){return;}
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	uint32_t type_a = bucket.iterator_type();
	uint32_t type_b = inp_arr.bucket.iterator_type();
	if(type_a == 1){
		auto start = bucket.cbegin_contiguous<value_t>();
		auto my_start = bucket.begin_contiguous<value_t>();
		if(type_b == 1){
			auto inp_start = inp_arr.bucket.cbegin_contiguous<value_t>();
			tbb::parallel_for(tbb::blocked_range<int64_t>(0, size), 
					[&](tbb::blocked_range<int64_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
		}
		else if(type_b == 2){
			auto inp_start = inp_arr.bucket.cbegin_blocked<value_t>();
			tbb::parallel_for(tbb::blocked_range<int64_t>(0, size), 
				[&](tbb::blocked_range<int64_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
		}
		else{
			auto inp_start = inp_arr.bucket.cbegin_list<value_t>();
			tbb::parallel_for(tbb::blocked_range<int64_t>(0, size), 
				[&](tbb::blocked_range<int64_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
		}
	}
	else if(type_a == 2){
		auto start = bucket.cbegin_blocked<value_t>();
		auto my_start = bucket.begin_blocked<value_t>();
		if(type_b == 1){
			auto inp_start = inp_arr.bucket.cbegin_contiguous<value_t>();
			tbb::parallel_for(tbb::blocked_range<int64_t>(0, size), 
				[&](tbb::blocked_range<int64_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
		}
		else if(type_b == 2){
			auto inp_start = inp_arr.bucket.cbegin_blocked<value_t>();
			tbb::parallel_for(tbb::blocked_range<int64_t>(0, size), 
				[&](tbb::blocked_range<int64_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
		}
		else{
			auto inp_start = inp_arr.bucket.cbegin_list<value_t>();
			tbb::parallel_for(tbb::blocked_range<int64_t>(0, size), 
				[&](tbb::blocked_range<int64_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
		}

	}
	else{
		auto start = bucket.cbegin_list<value_t>();
		auto my_start = bucket.begin_list<value_t>();
		if(type_b == 1){
			auto inp_start = inp_arr.bucket.cbegin_contiguous<value_t>();
			tbb::parallel_for(tbb::blocked_range<int64_t>(0, size), 
				[&](tbb::blocked_range<int64_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
		}
		else if(type_b == 2){
			auto inp_start = inp_arr.bucket.cbegin_blocked<value_t>();
			tbb::parallel_for(tbb::blocked_range<int64_t>(0, size), 
				[&](tbb::blocked_range<int64_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
		}
		else{
			auto inp_start = inp_arr.bucket.cbegin_list<value_t>();
			tbb::parallel_for(tbb::blocked_range<int64_t>(0, size), 
				[&](tbb::blocked_range<int64_t> r){std::transform(start + r.begin(), start + r.end(), inp_start + r.begin(), my_start + r.begin(), unary_op);});
		}
	}
}


template<DType dt, class UnaryOperation>
inline void ArrayVoid::sub_for_ceach(UnaryOperation&& unary_op) const{
	if(dt != dtype){return;}
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	uint32_t type_a = bucket.iterator_type();
	if(type_a == 1){
		auto start = bucket.cbegin_contiguous<value_t>();
		auto end = bucket.cend_contiguous<value_t>();
		tbb::parallel_for_each(start, end, std::forward<UnaryOperation&&>(unary_op));
	}
	else if(type_a == 2){
		auto start = bucket.cbegin_blocked<value_t>();
		auto end = bucket.cend_blocked<value_t>();
		tbb::parallel_for_each(start, end, std::forward<UnaryOperation&&>(unary_op));

	}
	else{
		auto start = bucket.cbegin_list<value_t>();
		auto end = bucket.cend_list<value_t>();
		tbb::parallel_for_each(start, end, std::forward<UnaryOperation&&>(unary_op));
	}
}

template<DType dt, class UnaryOperation>
inline void ArrayVoid::sub_for_each(UnaryOperation&& unary_op){
	if(dt != dtype){return;}
	using value_t = DTypeFuncs::dtype_to_type_t<dt>;
	uint32_t type_a = bucket.iterator_type();
	if(type_a == 1){
		auto start = bucket.begin_contiguous<value_t>();
		auto end = bucket.end_contiguous<value_t>();
		tbb::parallel_for_each(start, end, std::forward<UnaryOperation&&>(unary_op));
	}
	else if(type_a == 2){
		auto start = bucket.begin_blocked<value_t>();
		auto end = bucket.end_blocked<value_t>();
		tbb::parallel_for_each(start, end, std::forward<UnaryOperation&&>(unary_op));

	}
	else{
		auto start = bucket.begin_list<value_t>();
		auto end = bucket.end_list<value_t>();
		tbb::parallel_for_each(start, end, std::forward<UnaryOperation&&>(unary_op));
	}
}

#endif

}


#endif
