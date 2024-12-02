#include <cstdint>
#include <ios>

#include "../Tensor.h"
#include "../refs/SizeRef.h"
#include "../dtype/ArrayVoid.h"
#include "../dtype/DType.h"
#include "../dtype/DType_enum.h"




#include <atomic>
#include <functional>
//#include <i386/types.h>
#include <memory.h>
#include <memory>
#include <algorithm>
#include <numeric>
#include <ratio>

#include <cassert>
//#include <format>
#include <sys/types.h>
#include <type_traits>
#include <vector>
#include "../utils/utils.h"
#include <chrono>
#include "../permute/permute.h"
#include "functional.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include "../dtype/ArrayVoid.hpp"
#include "functional_operator.h"

#ifdef USE_PARALLEL
	#include <tbb/parallel_for_each.h>
	#include <tbb/parallel_for.h>
#endif
#define assertm(exp, msg) assert(((void)msg, exp))



namespace nt{
namespace functional{

template<typename Iterator>
class SizeDiffGetter_b{
	const nt::SizeRef& a;
	const nt::SizeRef& b;
	nt::ArrayRef<int64_t> current_b;
	const Iterator copy;
	Iterator outp;
	int64_t total(){
		int64_t outp;
		for(int64_t i = 0; i < b.size()-1; ++i)
			outp += b.multiply(i) * current_b[i];
		return outp + current_b.back();
	}
	void adjust(int64_t inp){
		for(int64_t i = 0; i < b.size()-1; ++i){
			int64_t mult = b.multiply(i);
			int64_t minus = inp % mult;
			current_b[i] = minus;
			inp -= mult * minus;
		}
		current_b.back() = inp;
		outp = copy;
		int64_t start = b.size() - a.size();
		for(int64_t i = start; i < current_b.size()-1; ++i){
			if(b[i] == a[i - start]){
				outp += current_b[i] * a.multiply(i-start+1);
			}
		}
		if(b.back() == a.back() && current_b.back() == 0){outp += current_b.back();}
	}
	SizeDiffGetter_b(const nt::SizeRef& _a, const nt::SizeRef& _b, Iterator a, int64_t total)
			:a(_a),
			b(_b),
			current_b(nt::ArrayRef<int64_t>::zeros(_a.size())),
			copy(a),
			outp(a)
		{adjust(total);}

	public:
		SizeDiffGetter_b(const nt::SizeRef& _a, const nt::SizeRef& _b, Iterator a)
			:a(_a),
			b(_b),
			current_b(nt::ArrayRef<int64_t>::zeros(_a.size())),
			copy(a),
			outp(a)
		{
			std::fill(current_b.d_data(), current_b.d_data() + _b.size(), 0);
		}
			
		inline Iterator& a_index() {return outp;}
		
		SizeDiffGetter_b& operator+=(const int64_t i){adjust(total() + i);}
		SizeDiffGetter_b operator+(const int64_t i){
			return SizeDiffGetter_b(a, b, copy, total()+i);	
		}

		SizeDiffGetter_b& operator++(){
			int64_t size = b.size()-1;
			while(current_b[size] == b[size]-1){
				current_b[size] = 0; 
				--size;
			}
			++current_b[size];
			if(size == b.size() - 1){
				if(b.back() == a.back())
					++outp;
				return *this;
			}
			outp = copy;
			int64_t start = b.size() - a.size();
			for(int64_t i = start; i < current_b.size()-1; ++i){
				if(b[i] == a[i - start]){
					outp += current_b[i] * a.multiply(i-start+1);
				}
			}
			if(b.back() == a.back() && current_b.back() == 0){outp += current_b.back();}
			return *this;
		}
};

template<typename Iterator>
class SizeDiffGetter_a{
	const nt::SizeRef& a;
	const nt::SizeRef& b;
	nt::ArrayRef<int64_t> current_a;
	const Iterator copy;
	Iterator outp;
	int64_t total(){
		int64_t outp;
		for(int64_t i = 0; i < a.size()-1; ++i)
			outp += a.multiply(i) * current_a[i];
		return outp + current_a.back();
	}
	void adjust(int64_t inp){
		for(int64_t i = 0; i < a.size()-1; ++i){
			int64_t mult = a.multiply(i);
			int64_t minus = inp % mult;
			current_a[i] = minus;
			inp -= mult * minus;
		}
		current_a.back() = inp;
		outp = copy;
		int64_t start = a.size() - b.size();
		for(int64_t i = start; i < current_a.size()-1; ++i){
			if(a[i] == b[i - start]){
				outp += current_a[i] * b.multiply(i-start+1);
			}
		}
		if(b.back() == a.back() && current_a.back() == 0){outp += current_a.back();}
	}
	SizeDiffGetter_a(const nt::SizeRef& _a, const nt::SizeRef& _b, const Iterator a, int64_t total)
			:a(_a),
			b(_b),
			current_a(nt::ArrayRef<int64_t>::zeros(_a.size())),
			copy(a),
			outp(a)
		{adjust(total);}
	public:
		SizeDiffGetter_a(const nt::SizeRef& _a, const nt::SizeRef& _b, Iterator a)
			:a(_a),
			b(_b),
			current_a(nt::ArrayRef<int64_t>::zeros(_a.size())),
			copy(a),
			outp(a)
		{
			std::fill(current_a.d_data(), current_a.d_data() + _a.size(), 0);
		}
			
		inline Iterator& b_index() {return outp;}
		SizeDiffGetter_a& operator+=(const int64_t i){adjust(total() + i);}
		SizeDiffGetter_a operator+(const int64_t i){
			return SizeDiffGetter_a(a, b, copy, total()+i);
		}

		SizeDiffGetter_a& operator++(){
			int64_t size = b.size()-1;
			while(current_a[size] == a[size]-1){
				current_a[size] = 0; 
				--size;
			}
			++current_a[size];
			if(size == b.size() - 1){
				if(b.back() == a.back())
					++outp;
				return *this;
			}
			outp = copy;
			int64_t start = b.size() - a.size();
			for(int64_t i = start; i < current_a.size()-1; ++i){
				if(a[i] == b[i - start]){
					outp += current_a[i] * b.multiply(i-start+1);
				}
			}
			if(b.back() == a.back() && current_a.back() == 0){outp += current_a.back();}
			return *this;
		}
};


void op_exception_dtypes(const DType& a, const DType& b){
	utils::THROW_EXCEPTION(a == b, "\nRuntimeError: Expected dtype of second tensor to be $ but got $", a, b);
}

void op_exception_shapes(const SizeRef& a, const SizeRef& b){
	if(a != b){
		if(a.multiply() > b.multiply()){
			/* std::cout << "a"<<std::endl; */
			uint32_t start = a.size() - b.size();
			/* std::cout<<start<<std::endl; */
			for(uint32_t i = a.size() - b.size(); i < a.size(); ++i){
				utils::THROW_EXCEPTION(a[i] == b[i - start] || b[i - start] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $, ($),($)", a[i], b[i - start], i, a, b);
					/* utils::THROW_EXCEPTION(a[i] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $, ($),($)", a[i], b[i - start], i, a, b); */
			}
		}
		else if(b.multiply() > a.multiply()){
			/* std::cout << "b"<<std::endl; */
			uint32_t start = b.size() - a.size();
			for(uint32_t i = b.size() - a.size(); i < b.size(); ++i){
					/* utils::THROW_EXCEPTION(b[i] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $, ($),($)", a[i - start], b[i], i, a, b); */
				utils::THROW_EXCEPTION(a[i-start] == b[i] || a[i - start] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $, ($),($)", a[i - start], b[i], i, a, b);

			}
		}
		else{
			for(uint32_t i = 0; i < b.size(); ++i){
				if(a[i] != b[i] && (b[i] != 1 || a[i] != 1)){
					utils::THROW_EXCEPTION(b[i] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $, ($),($)", a[i], b[i], i, a, b);
					utils::THROW_EXCEPTION(a[i] == 1,"\nRuntimeError: The size of tensor a ($) must match the size of tensor b ($) at non-singleton dimension $, ($),($)", a[i], b[i], i, a, b);

				}
			}
		}

	}
}


#ifdef USE_PARALLEL

inline static constexpr auto  element_wise_operation = [](auto begin, auto end, auto first2, void* outp, const uint32_t total_size, const functional_operator_num& op){
	using value_t = utils::IteratorBaseType_t<decltype(begin)>;
	value_t* out = reinterpret_cast<value_t*>(outp);
	switch(op){
		case functional_operator_num::Multiply:{
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, total_size),
					[&](tbb::blocked_range<uint32_t> r){
			auto begin_a = begin + r.begin();
			auto begin_b = first2 + r.begin();
			value_t* out_c = out + r.begin();
			for(uint32_t i = r.begin(); i < r.end(); ++i, ++out_c, ++begin_a, ++begin_b){
				*out_c = *begin_a * *begin_b;
			}
			});
			return;
		}
		case functional_operator_num::Subtract:{
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, total_size),
					[&](tbb::blocked_range<uint32_t> r){
			auto begin_a = begin + r.begin();
			auto begin_b = first2 + r.begin();
			value_t* out_c = out + r.begin();
			for(uint32_t i = r.begin(); i < r.end(); ++i, ++out_c, ++begin_a, ++begin_b){
				*out_c = *begin_a - *begin_b;
			}
			});
			return;
		}
		case functional_operator_num::Divide:{
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, total_size),
					[&](tbb::blocked_range<uint32_t> r){
			auto begin_a = begin + r.begin();
			auto begin_b = first2 + r.begin();
			value_t* out_c = out + r.begin();
			for(uint32_t i = r.begin(); i < r.end(); ++i, ++out_c, ++begin_a, ++begin_b){
				*out_c = *begin_a / *begin_b;
			}
			});
			return;
		}
		case functional_operator_num::Add:{
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, total_size),
					[&](tbb::blocked_range<uint32_t> r){
			auto begin_a = begin + r.begin();
			auto begin_b = first2 + r.begin();
			value_t* out_c = out + r.begin();
			for(uint32_t i = r.begin(); i < r.end(); ++i, ++out_c, ++begin_a, ++begin_b){
				*out_c = *begin_a + *begin_b;
			}
			});
			return;
		}


	}
};

inline static constexpr auto  operator_dif_shapes = [](auto begin, auto end, auto first2, void* outp, const uint32_t total_size, const SizeRef& a_s_o, const SizeRef& b_s_o, const functional_operator_num& op){
	using value_t = utils::IteratorBaseType_t<decltype(begin)>;
	value_t* out = reinterpret_cast<value_t*>(outp);
	SizeDiffGetter_a<decltype(first2)> sg(a_s_o, b_s_o, first2);
	switch(op){
		case functional_operator_num::Multiply:{
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, total_size),
					[&](tbb::blocked_range<uint32_t> r){
			SizeDiffGetter_a<decltype(first2)> sg_2 = sg + r.begin();
			auto begin_a = begin + r.begin();
			value_t* out_c = out + r.begin();
			for(uint32_t i = r.begin(); i < r.end(); ++i, ++out_c, ++begin_a, ++sg_2){
				*out_c = *begin_a * *sg_2.b_index();
			}
			});
			return;
		}
		case functional_operator_num::Subtract:{
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, total_size),
					[&](tbb::blocked_range<uint32_t> r){
			SizeDiffGetter_a<decltype(first2)> sg_2 = sg + r.begin();
			auto begin_a = begin + r.begin();
			value_t* out_c = out + r.begin();
			for(uint32_t i = r.begin(); i < r.end(); ++i, ++out_c, ++begin_a, ++sg_2){
				*out_c = *begin_a - *sg_2.b_index();
			}
			});
			return;
		}
		case functional_operator_num::Divide:{
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, total_size),
					[&](tbb::blocked_range<uint32_t> r){
			SizeDiffGetter_a<decltype(first2)> sg_2 = sg + r.begin();
			auto begin_a = begin + r.begin();
			value_t* out_c = out + r.begin();
			for(uint32_t i = r.begin(); i < r.end(); ++i, ++out_c, ++begin_a, ++sg_2){
				*out_c = *begin_a / *sg_2.b_index();
			}
			});
			return;
		}
		case functional_operator_num::Add:{
			tbb::parallel_for(tbb::blocked_range<uint32_t>(0, total_size),
					[&](tbb::blocked_range<uint32_t> r){
			SizeDiffGetter_a<decltype(first2)> sg_2 = sg + r.begin();
			auto begin_a = begin + r.begin();
			value_t* out_c = out + r.begin();
			for(uint32_t i = r.begin(); i < r.end(); ++i, ++out_c, ++begin_a, ++sg_2){
				*out_c = *begin_a + *sg_2.b_index();
			}
			});
			return;
		}


	}
};
#else

inline static constexpr auto element_wise_operation = [](auto begin, auto end, auto first2, void* outp, const functional_operator_num& op){
	using value_t = utils::IteratorBaseType_t<decltype(begin)>;
	value_t* out = reinterpret_cast<value_t*>(outp);
	switch(op){
		case functional_operator_num::Multiply:
			std::transform(begin, end, first2, out, std::multiplies<value_t>());
			break;
		case functional_operator_num::Divide:
			std::transform(begin, end, first2, out, std::divides<value_t>());
			break;
		case functional_operator_num::Add:
			std::transform(begin, end, first2, out, std::plus<value_t>());
			break;
		case functional_operator_num::Subtract:
			std::transform(begin, end, first2, out, std::minus<value_t>());
			break;

	}
};



inline static constexpr auto operator_dif_shapes = [](auto begin, auto end, auto first2, void* outp, const SizeRef& a_s_o, const SizeRef& b_s_o, const functional_operator_num& op){
	using value_t = utils::IteratorBaseType_t<decltype(begin)>;
	/* auto first2 = first.tcbegin<value_t>(); */
	SizeDiffGetter_a<decltype(begin)> sg(a_s_o, b_s_o, first2);
	value_t* val = reinterpret_cast<value_t*>(outp);
	switch(op){
		case functional_operator_num::Multiply:
			for(;begin != end; ++begin, ++sg, ++val){
				*val = *begin * *sg.b_index();
			}
			return;
		case functional_operator_num::Divide:
			for(;begin != end; ++begin, ++sg, ++val){
				*val = *begin / *sg.b_index();
			}
			return;
		case functional_operator_num::Add:
			for(;begin != end; ++begin, ++sg, ++val){
				*val = *begin + *sg.b_index();
			}
			return;		
		case functional_operator_num::Subtract:
			for(;begin != end; ++begin, ++sg, ++val){
				*val = *begin - *sg.b_index();
			}
			return;	
	}

};
#endif


Tensor functional_operator_out(const Tensor& a, const Tensor& b, const functional_operator_num op){
	op_exception_dtypes(a.dtype, b.dtype);
	if(a.shape() == b.shape()){
		Tensor output(a.shape(), a.dtype);
#ifdef USE_PARALLEL
		a.arr_void().cexecute_function_nbool(element_wise_operation, b.arr_void(), output.data_ptr(), a.numel(), op);
#else
		a.arr_void().cexecute_function_nbool(element_wise_operation, b.arr_void(), output.data_ptr(), op);
#endif
		return std::move(output);	
	}
	op_exception_shapes(a.shape(), b.shape());
	SizeRef a_s = a.shape().size() >= b.shape().size() ? a.shape() : a.shape().unflatten(0, b.shape().size() - a.shape().size());
	SizeRef b_s = b.shape().size() >= a.shape().size() ? b.shape() : b.shape().unflatten(0, a.shape().size() - b.shape().size());

	Tensor output(a.numel() > b.numel() ? a.shape() : b.shape(), a.dtype);
	if(a.numel() > b.numel()){
#ifdef USE_PARALLEL
		a.arr_void().cexecute_function_nbool(operator_dif_shapes, b.arr_void(), output.data_ptr(), a.numel(),b_s, a_s, op);
#else

		a.arr_void().cexecute_function_nbool(operator_dif_shapes, b.arr_void(), output.data_ptr(), a_s, b_s, op);
#endif
	}
	else{
#ifdef USE_PARALLEL
		b.arr_void().cexecute_function_nbool(operator_dif_shapes, a.arr_void(), output.data_ptr(), b.numel(), b_s, a_s,op);
#else
		b.arr_void().cexecute_function_nbool(operator_dif_shapes, a.arr_void(), output.data_ptr(), b_s, a_s, op);
#endif
	}
	return std::move(output);

}



inline static constexpr auto operation_dif_shapes_this = [](auto begin, auto end, const ArrayVoid& first, const SizeRef& a_s_o, const SizeRef& b_s_o, const functional_operator_num& op){
	using value_t = utils::IteratorBaseType_t<decltype(begin)>;
	uint32_t type_a = first.get_bucket().iterator_type();
	if(type_a == 1){
		auto first2 = first.get_bucket().cbegin_contiguous<value_t>();
		SizeDiffGetter_a<decltype(first2)> sg(a_s_o, b_s_o, first2);
		switch(op){
			case functional_operator_num::Multiply:
				for(;begin != end; ++begin, ++sg)
					*begin *= *sg.b_index();
				return;
			case functional_operator_num::Divide:
				for(;begin != end; ++begin, ++sg)
					*begin /= *sg.b_index();
				return;
			case functional_operator_num::Add:
				for(;begin != end; ++begin, ++sg)
					*begin += *sg.b_index();
				return;	
			case functional_operator_num::Subtract:
				for(;begin != end; ++begin, ++sg)
					*begin -= *sg.b_index();
				return;
		}
	}
	else if(type_a == 2){
		auto first2 = first.get_bucket().cbegin_blocked<value_t>();
		SizeDiffGetter_a<decltype(first2)> sg(a_s_o, b_s_o, first2);
		switch(op){
			case functional_operator_num::Multiply:
				for(;begin != end; ++begin, ++sg)
					*begin *= *sg.b_index();
				return;
			case functional_operator_num::Divide:
				for(;begin != end; ++begin, ++sg)
					*begin /= *sg.b_index();
				return;
			case functional_operator_num::Add:
				for(;begin != end; ++begin, ++sg)
					*begin += *sg.b_index();
				return;	
			case functional_operator_num::Subtract:
				for(;begin != end; ++begin, ++sg)
					*begin -= *sg.b_index();
				return;
		}
	}
	else if(type_a == 3){
		auto first2 = first.get_bucket().cbegin_list<value_t>();
		SizeDiffGetter_a<decltype(first2)> sg(a_s_o, b_s_o, first2);
		switch(op){
			case functional_operator_num::Multiply:
				for(;begin != end; ++begin, ++sg)
					*begin *= *sg.b_index();
				return;
			case functional_operator_num::Divide:
				for(;begin != end; ++begin, ++sg)
					*begin /= *sg.b_index();
				return;
			case functional_operator_num::Add:
				for(;begin != end; ++begin, ++sg)
					*begin += *sg.b_index();
				return;	
			case functional_operator_num::Subtract:
				for(;begin != end; ++begin, ++sg)
					*begin -= *sg.b_index();
				return;
		}
	}


};



inline static constexpr auto operation_dif_shapes_bd = [](auto begin, auto end, const ArrayVoid& first, const SizeRef& a_s_o, const SizeRef& b_s_o, const functional_operator_num& op){
	using value_t = utils::IteratorBaseType_t<decltype(begin)>;
	uint32_t type_a = first.get_bucket().iterator_type();
	if(type_a == 1){
		auto first2 = first.get_bucket().cbegin_contiguous<value_t>();
		auto end2 = first.get_bucket().cend_contiguous<value_t>();
		SizeDiffGetter_b<decltype(begin)> sg(a_s_o, b_s_o, begin);
		switch(op){
			case functional_operator_num::Multiply:
				for(;first2 != end2; ++first2, ++sg)
					*(sg.a_index()) *= *first2;
				return;
			case functional_operator_num::Divide:
				for(;first2 != end2; ++first2, ++sg)
					*(sg.a_index()) /= *first2;
				return;
			case functional_operator_num::Add:
				for(;first2 != end2; ++first2, ++sg)
					*(sg.a_index()) += *first2;
				return;
			case functional_operator_num::Subtract:
				for(;first2 != end2; ++first2, ++sg)
					*(sg.a_index()) -= *first2;
				return;
		}
	}
	else if(type_a == 2){
		auto first2 = first.get_bucket().cbegin_blocked<value_t>();
		auto end2 = first.get_bucket().cend_blocked<value_t>();
		SizeDiffGetter_b<decltype(begin)> sg(a_s_o, b_s_o, begin);
		switch(op){
			case functional_operator_num::Multiply:
				for(;first2 != end2; ++first2, ++sg)
					*(sg.a_index()) *= *first2;
				return;
			case functional_operator_num::Divide:
				for(;first2 != end2; ++first2, ++sg)
					*(sg.a_index()) /= *first2;
				return;
			case functional_operator_num::Add:
				for(;first2 != end2; ++first2, ++sg)
					*(sg.a_index()) += *first2;
				return;
			case functional_operator_num::Subtract:
				for(;first2 != end2; ++first2, ++sg)
					*(sg.a_index()) -= *first2;
				return;
		}
	}
	else if(type_a == 3){
		auto first2 = first.get_bucket().cbegin_list<value_t>();
		auto end2 = first.get_bucket().cend_list<value_t>();
		SizeDiffGetter_b<decltype(begin)> sg(a_s_o, b_s_o, begin);
		switch(op){
			case functional_operator_num::Multiply:
				for(;first2 != end2; ++first2, ++sg)
					*(sg.a_index()) *= *first2;
				return;
			case functional_operator_num::Divide:
				for(;first2 != end2; ++first2, ++sg)
					*(sg.a_index()) /= *first2;
				return;
			case functional_operator_num::Add:
				for(;first2 != end2; ++first2, ++sg)
					*(sg.a_index()) += *first2;
				return;
			case functional_operator_num::Subtract:
				for(;first2 != end2; ++first2, ++sg)
					*(sg.a_index()) -= *first2;
				return;
		}
	}

};


void functional_operator_this(Tensor& a, const Tensor& b, const functional_operator_num op){
	/* std::cout << "functional oeperator this got "<<a.shape()<<" and "<<b.shape()<<std::endl; */
	op_exception_dtypes(a.dtype, b.dtype);
	const SizeRef a_shape = a.shape();
	const SizeRef b_shape = b.shape();
	if(a.shape() == b.shape()){
		switch(op){
			case functional_operator_num::Multiply:{
				a.arr_void().transform_function_nbool([](const auto& _v1, const auto& _v2){return _v1 * _v2;}, b.arr_void());
				return;
			}
			case functional_operator_num::Subtract:{
				a.arr_void().transform_function_nbool([](const auto& _v1, const auto& _v2){return _v1 - _v2;}, b.arr_void());
				return;
			}
			case functional_operator_num::Add:{
				a.arr_void().transform_function_nbool([](const auto& _v1, const auto& _v2){return _v1 + _v2;}, b.arr_void());
				return;
			}
			case functional_operator_num::Divide:{
				a.arr_void().transform_function_nbool([](const auto& _v1, const auto& _v2){return _v1 / _v2;}, b.arr_void());
				return;
			}
		}
		return;
	}
	op_exception_shapes(a_shape, b_shape);

	bool larger_a = a.shape().size() > b.shape().size();
	SizeRef a_s = a.shape().size() > b.shape().size() ? a.shape() : a.shape().unflatten(0, b.shape().size() - a.shape().size());
	SizeRef b_s = b.shape().size() > a.shape().size() ? b.shape() : b.shape().unflatten(0, a.shape().size() - b.shape().size());
	if(larger_a){
		a.arr_void().execute_function_nbool(operation_dif_shapes_this, b.arr_void(), a_s, b_s, op);
	}
	else{
		a.arr_void().execute_function_nbool(operation_dif_shapes_bd, b.arr_void(), a_s, b_s, op);
	}

}


}
}
