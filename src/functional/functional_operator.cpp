#include <cstdint>
#include <ios>
#include <iostream>

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



//basically, for all functions, the shape out is the same as a in no matter if b has to be expanded or summed to fit into a.
Tensor functional_operator_out(const Tensor& _a, const Tensor& _b, const functional_operator_num op){
	op_exception_dtypes(_a.dtype, _b.dtype);
	if(_a.shape() == _b.shape()){
		Tensor output(_a.shape(), _a.dtype);
		switch(op){
			case functional_operator_num::Multiply:{
				_a.arr_void().cexecute_function_nbool([](auto begin, auto end, auto begin2, void* out_p){
					using value_t = utils::IteratorBaseType_t<decltype(begin)>;
					value_t* out = reinterpret_cast<value_t*>(out_p);
					mp::multiply(begin, end, begin2, out);
				}, _b.arr_void(), output.data_ptr());
				return std::move(output);
			}
			case functional_operator_num::Subtract:{
				_a.arr_void().cexecute_function_nbool([](auto begin, auto end, auto begin2, void* out_p){
					using value_t = utils::IteratorBaseType_t<decltype(begin)>;
					value_t* out = reinterpret_cast<value_t*>(out_p);
					mp::subtract(begin, end, begin2, out);
				}, _b.arr_void(), output.data_ptr());
				return std::move(output);
			}
			case functional_operator_num::Divide:{
				_a.arr_void().cexecute_function_nbool([](auto begin, auto end, auto begin2, void* out_p){
					using value_t = utils::IteratorBaseType_t<decltype(begin)>;
					value_t* out = reinterpret_cast<value_t*>(out_p);
					mp::divide(begin, end, begin2, out);
				}, _b.arr_void(), output.data_ptr());
				return std::move(output);
			}
			case functional_operator_num::Add:{
				_a.arr_void().cexecute_function_nbool([](auto begin, auto end, auto begin2, void* out_p){
					using value_t = utils::IteratorBaseType_t<decltype(begin)>;
					value_t* out = reinterpret_cast<value_t*>(out_p);
					mp::add(begin, end, begin2, out);
				}, _b.arr_void(), output.data_ptr());
				return std::move(output);
			}

		}
		return std::move(output);	
	}



	Tensor b = (_a.dims() > _b.dims()) ? _b.unsqueeze_as(_a) : _b;
	Tensor a = (_b.dims() > _a.dims()) ? _a.unsqueeze_as(_b) : _a;
	if(b.shape() == a.shape()){return functional_operator_out(a, b, op).view(_a.shape());}
	b = b.expand_as(a).clone();
	a = a.expand_as(b).clone();
	utils::throw_exception(a.shape() == b.shape(), "Shape error for functional operator $ != $", a.shape(), b.shape());
    // const Tensor& _larger_dim = (_a.dims() > _b.dims()) ? _a : _b;
	// return functional_operator_out(a, b, op).sum_as(_larger_dim).view(_larger_dim.shape());
	return functional_operator_out(a, b, op);
}


//this is a function to handle when both need to be expanded or summed
//for example {20,1} *= {9}
//that needs to turn into b being unsqueezed as {20,1} *= {1,9}
//then b needs to be expanded such as: {20,1} *= {20,9}
//then a needs to be expanded such as: {20,9} *= {20,9}
//then they are equal, and op happens
//then a needs to be opped back into a {20,1}
void functional_operator_this(Tensor& _a, const Tensor& _b, const functional_operator_num op){
	op_exception_dtypes(_a.dtype, _b.dtype);
	if(_a.shape() == _b.shape()){
		switch(op){
			case functional_operator_num::Multiply:{
				_a.arr_void().execute_function_nbool([](auto begin, auto end, auto begin2){
					mp::multiply(begin, end, begin2, begin);
				}, const_cast<ArrayVoid&>(_b.arr_void()));
				return;
			}
			case functional_operator_num::Subtract:{
				_a.arr_void().execute_function_nbool([](auto begin, auto end, auto begin2){
					mp::subtract(begin, end, begin2, begin);
				}, const_cast<ArrayVoid&>(_b.arr_void()));
				return;
			}
			case functional_operator_num::Divide:{
				_a.arr_void().execute_function_nbool([](auto begin, auto end, auto begin2){
					mp::divide(begin, end, begin2, begin);
				}, const_cast<ArrayVoid&>(_b.arr_void()));
				return;
			}
			case functional_operator_num::Add:{
				_a.arr_void().execute_function_nbool([](auto begin, auto end, auto begin2){
					mp::add(begin, end, begin2, begin);
				}, const_cast<ArrayVoid&>(_b.arr_void()));
				return;
			}

		}
		return;	
	}
	Tensor b = (_a.dims() > _b.dims()) ? _b.unsqueeze_as(_a) : _b;
	Tensor a = (_b.dims() > _a.dims()) ? _a.unsqueeze_as(_b) : _a;
	if(b.shape() == a.shape()){functional_operator_this(a, b, op); return;}
	b = b.expand_as(a).clone();
	a = a.expand_as(b).clone();
	utils::throw_exception(a.shape() == b.shape(), "Shape error for functional operator $ != $", a.shape(), b.shape());
	Tensor c = functional_operator_out(a, b, op);

	Tensor s = (c.dims() > _a.dims()) ? _a.unsqueeze_as(c) : _a;
	s.set_(c.sum_as(s));
}


}
}
