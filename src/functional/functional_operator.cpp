#include <cstdint>
#include <ios>

#include "../Tensor.h"
#include "../refs/SizeRef.h"
#include "../dtype/ArrayVoid.h"
#include "../dtype/DType.h"
#include "../dtype/DType_enum.h"
#include "../mp/simde_ops.h"




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


#define assertm(exp, msg) assert(((void)msg, exp))



namespace nt{
namespace functional{



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



//basically, for all functions, the shape out is the same as a in no matter if b has to be expanded or summed to fit into a.
Tensor functional_operator_out(const Tensor& a, const Tensor& b, const functional_operator_num op){
	op_exception_dtypes(a.dtype, b.dtype);
	if(a.shape() == b.shape()){
		Tensor output(a.shape(), a.dtype);
		switch(op){
			case functional_operator_num::Multiply:{
				a.arr_void().cexecute_function_nbool([](auto begin, auto end, auto begin2, void* out_p){
					using value_t = utils::IteratorBaseType_t<decltype(begin)>;
					value_t* out = reinterpret_cast<value_t*>(out_p);
					mp::multiply(begin, end, begin2, out);
				}, b.arr_void(), output.data_ptr());
				return std::move(output);
			}
			case functional_operator_num::Subtract:{
				a.arr_void().cexecute_function_nbool([](auto begin, auto end, auto begin2, void* out_p){
					using value_t = utils::IteratorBaseType_t<decltype(begin)>;
					value_t* out = reinterpret_cast<value_t*>(out_p);
					mp::subtract(begin, end, begin2, out);
				}, b.arr_void(), output.data_ptr());
				return std::move(output);
			}
			case functional_operator_num::Divide:{
				a.arr_void().cexecute_function_nbool([](auto begin, auto end, auto begin2, void* out_p){
					using value_t = utils::IteratorBaseType_t<decltype(begin)>;
					value_t* out = reinterpret_cast<value_t*>(out_p);
					mp::divide(begin, end, begin2, out);
				}, b.arr_void(), output.data_ptr());
				return std::move(output);
			}
			case functional_operator_num::Add:{
				a.arr_void().cexecute_function_nbool([](auto begin, auto end, auto begin2, void* out_p){
					using value_t = utils::IteratorBaseType_t<decltype(begin)>;
					value_t* out = reinterpret_cast<value_t*>(out_p);
					mp::add(begin, end, begin2, out);
				}, b.arr_void(), output.data_ptr());
				return std::move(output);
			}

		}
		return std::move(output);	
	}
	op_exception_shapes(a.shape(), b.shape());
	if(a.numel() > b.numel()){
		if(b.numel() == 1){
			switch(op){
				case functional_operator_num::Multiply:{
					return a * b.toScalar();
				}
				case functional_operator_num::Subtract:{
					return a - b.toScalar();
				}
				case functional_operator_num::Divide:{
					return a / b.toScalar();
				}
				case functional_operator_num::Add:{
					return a + b.toScalar();
				}
			}

		}
		Tensor nB = b.expand_as(a);
		return functional_operator_out(a, nB, op);
	}
	//b.numel > a.numel()
	if(a.numel() == 1){
		switch(op){
			case functional_operator_num::Multiply:{
				return a.toScalar() * b;
			}
			case functional_operator_num::Subtract:{
				return a.toScalar() - b;
			}
			case functional_operator_num::Divide:{
				return a.toScalar() / b;
			}
			case functional_operator_num::Add:{
				return a.toScalar() + b;
			}
		}
	
	}
	Tensor nB = b.sum_as(a);
	return functional_operator_out(a, nB, op);	
}


void functional_operator_this(Tensor& a, const Tensor& b, const functional_operator_num op){
	op_exception_dtypes(a.dtype, b.dtype);
	if(a.shape() == b.shape()){
		switch(op){
			case functional_operator_num::Multiply:{
				a.arr_void().execute_function_nbool([](auto begin, auto end, auto begin2){
					mp::multiply(begin, end, begin2, begin);
				}, const_cast<ArrayVoid&>(b.arr_void()));
				return;
			}
			case functional_operator_num::Subtract:{
				a.arr_void().execute_function_nbool([](auto begin, auto end, auto begin2){
					mp::subtract(begin, end, begin2, begin);
				}, const_cast<ArrayVoid&>(b.arr_void()));
				return;
			}
			case functional_operator_num::Divide:{
				a.arr_void().execute_function_nbool([](auto begin, auto end, auto begin2){
					mp::divide(begin, end, begin2, begin);
				}, const_cast<ArrayVoid&>(b.arr_void()));
				return;
			}
			case functional_operator_num::Add:{
				a.arr_void().execute_function_nbool([](auto begin, auto end, auto begin2){
					mp::add(begin, end, begin2, begin);
				}, const_cast<ArrayVoid&>(b.arr_void()));
				return;
			}

		}
		return;	
	}
	op_exception_shapes(a.shape(), b.shape());
	if(a.numel() > b.numel()){
		if(b.numel() == 1){
			switch(op){
				case functional_operator_num::Multiply:{
					a *= b.toScalar();
					return;
				}
				case functional_operator_num::Subtract:{
					a -= b.toScalar();
					return;
				}
				case functional_operator_num::Divide:{
					a /= b.toScalar();
					return;
				}
				case functional_operator_num::Add:{
					a += b.toScalar();
					return;
				}
			}

		}
		Tensor nB = b.expand_as(a);
		functional_operator_this(a, nB, op);
		return;
	}else{
		//b.numel() > a.numel()
		Tensor nB = b.sum_as(a);
		functional_operator_this(a, nB, op);
	}
}


}
}
