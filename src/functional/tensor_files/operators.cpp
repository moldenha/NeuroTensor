#include <cstdint>
#include <ios>
#include <iostream>

#include "../../Tensor.h"
#include "../../refs/SizeRef.h"
#include "../../dtype/ArrayVoid.h"
#include "../../dtype/DType.h"
#include "../../dtype/DType_enum.h"
#include "exceptions.hpp"


#include <functional>
#include <algorithm>
#include <numeric>
#include <ratio>

#include <cassert>
//#include <format>
#include <vector>
#include "../../utils/utils.h"
#include "../functional.h"
#include "../../dtype/ArrayVoid.hpp"
#include "../cpu/operators.h"





namespace nt{
namespace functional{



//basically, for all functions, the shape out is the same as a in no matter if b has to be expanded or summed to fit into a.
Tensor functional_operator_out(const Tensor& _a, const Tensor& _b, const functional_operator_num op){
	exception_dtypes(_a.dtype, _b.dtype);
	if(_a.shape() == _b.shape()){
		Tensor output(_a.shape(), _a.dtype);
        if(_a.dtype == DType::TensorObj){
            switch(op){
               case functional_operator_num::Multiply:{
                    _a.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end, auto begin2, void* out_p){
                        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                        value_t* out = reinterpret_cast<value_t*>(out_p);
                        std::transform(begin, end, begin2, out, std::multiplies<Tensor>{});
                    }, _b.arr_void(), output.data_ptr());
                    return std::move(output);
                }
                case functional_operator_num::Subtract:{
                    _a.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end, auto begin2, void* out_p){
                        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                        value_t* out = reinterpret_cast<value_t*>(out_p);
                        std::transform(begin, end, begin2, out, std::minus<Tensor>{});
                    }, _b.arr_void(), output.data_ptr());
                    return std::move(output);
                }
                case functional_operator_num::Divide:{
                    _a.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end, auto begin2, void* out_p){
                        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                        value_t* out = reinterpret_cast<value_t*>(out_p);
                        std::transform(begin, end, begin2, out, std::divides<Tensor>{});
                    }, _b.arr_void(), output.data_ptr());
                    return std::move(output);
                }
                case functional_operator_num::Add:{
                    _a.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end, auto begin2, void* out_p){
                        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                        value_t* out = reinterpret_cast<value_t*>(out_p);
                        std::transform(begin, end, begin2, out, std::plus<Tensor>{});
                    }, _b.arr_void(), output.data_ptr());
                    return std::move(output);
                }
            }
        }
        int i;
		switch(op){
			case functional_operator_num::Multiply: 
                i = 0;
                break;
			case functional_operator_num::Subtract:
                i = 1;
                break;
            case functional_operator_num::Divide:
                i = 2;
                break;
            case functional_operator_num::Add:
                i = 3;
                break;
		}
        cpu::operator_mdsa_(_a.arr_void(), _b.arr_void(), output.arr_void(), i);
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


//add and subtract are the only ones that can properly handle smaller broadcast shapes
//for example [1, 5] += [3, 5]
//        and [1, 5] -= [3, 5]
// are entirely valid, and are handled correctly
// however,
//     [1, 5] *= [3, 5]
// and [1, 5] /= [3, 5]
// are not
// (everything inside the brackets are shapes)
void functional_operator_this(Tensor& _a, const Tensor& _b, const functional_operator_num op){
	exception_dtypes(_a.dtype, _b.dtype);
	if(_a.shape() == _b.shape()){
        if(_a.dtype == DType::TensorObj){
            switch(op){
               case functional_operator_num::Multiply:{
                    _a.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end, auto begin2){
                        for(;begin != end; ++begin, ++begin2) (*begin) *= (*begin2);
                    }, const_cast<ArrayVoid&>(_b.arr_void()));
                    return; 
                }
                case functional_operator_num::Subtract:{
                    _a.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end, auto begin2){
                        for(;begin != end; ++begin, ++begin2) (*begin) -= (*begin2);
                    }, const_cast<ArrayVoid&>(_b.arr_void()));
                    return;
                }
                case functional_operator_num::Divide:{
                    _a.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end, auto begin2){
                        for(;begin != end; ++begin, ++begin2) (*begin) /= (*begin2);
                    }, const_cast<ArrayVoid&>(_b.arr_void()));
                    return;
                }
                case functional_operator_num::Add:{
                    _a.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end, auto begin2){
                        for(;begin != end; ++begin, ++begin2) (*begin) += (*begin2);
                    }, const_cast<ArrayVoid&>(_b.arr_void()));
                    return;
                }
            }
        }

		int i;
		switch(op){
			case functional_operator_num::Multiply: 
                i = 0;
                break;
			case functional_operator_num::Subtract:
                i = 1;
                break;
            case functional_operator_num::Divide:
                i = 2;
                break;
            case functional_operator_num::Add:
                i = 3;
                break;
		}
        cpu::operator_mdsa_(_a.arr_void(), _b.arr_void(), i);
		return;	
	}
	Tensor b = (_a.dims() > _b.dims()) ? _b.unsqueeze_as(_a) : _b;
	Tensor a = (_b.dims() > _a.dims()) ? _a.unsqueeze_as(_b) : _a;
	if(b.shape() == a.shape()){functional_operator_this(a, b, op); return;}
    if(op == functional_operator_num::Add){
       	b = b.expand_as(a).clone();
        a = a.expand_as(b).clone();
        utils::throw_exception(a.shape() == b.shape(), "Shape error for functional operator $ != $", a.shape(), b.shape());
        Tensor c = functional_operator_out(a, b, op);

        Tensor s = (c.dims() > _a.dims()) ? _a.unsqueeze_as(c) : _a;
        s.set_(c.sum_as(s));
        return; 
    }
    else if(op == functional_operator_num::Subtract){
        b = b.expand_as(a);
        b = b.sum_as(a);
        utils::throw_exception(a.shape() == b.shape(), "Shape error for subtraction functional this operator $ != $", a.shape(), b.shape());
        Tensor c = functional_operator_out(a, b, op);
        Tensor s = (c.dims() > _a.dims()) ? _a.unsqueeze_as(c) : _a;
        s.set_(c);
        return;
    }
    b = b.expand_as(a);
    // a = a.expand_as(b).clone();
    utils::THROW_EXCEPTION(a.shape() == b.shape(), 
                           "\nRuntimeError: output with shape $ doesn't match the broadcast shape $ for operator self modification", a.shape(), b.shape());
    
    functional_operator_this(a, b, op);
    // Tensor c = functional_operator_out(a, b, op);

    // Tensor s = (c.dims() > _a.dims()) ? _a.unsqueeze_as(c) : _a;
    // s.set_(c);
}


//obviously sum as works if the operation is addition
//if subtraction:
//  A:[1, 18] -= B:[3, 18]
//      A -= B[0]
//      A -= B[1]
//      A -= B[2]
//


Tensor hadamard_multiply(const Tensor& a, const Tensor& b){
    return functional_operator_out(a, b, functional_operator_num::Multiply);
}
Tensor& hadamard_multiply_this(Tensor &a, const Tensor &b){
    functional_operator_this(a, b, functional_operator_num::Multiply); 
    return a;
}

Tensor add(const Tensor& a, const Tensor& b){
    return functional_operator_out(a, b, functional_operator_num::Add);
}
Tensor& add_(Tensor &a, const Tensor &b){
    functional_operator_this(a, b, functional_operator_num::Add); 
    return a;
}

Tensor subtract(const Tensor& a, const Tensor& b){
    return functional_operator_out(a, b, functional_operator_num::Subtract);
}
Tensor& subtract_(Tensor &a, const Tensor &b){
    functional_operator_this(a, b, functional_operator_num::Subtract); 
    return a;
}

Tensor divide(const Tensor& a, const Tensor& b){
    return functional_operator_out(a, b, functional_operator_num::Divide);
}
Tensor& divide_(Tensor &a, const Tensor &b){
    functional_operator_this(a, b, functional_operator_num::Divide); 
    return a;
}



}
}
