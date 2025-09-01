#include "../../Tensor.h"
#include "../../dtype/ArrayVoid.hpp"
#include "../cpu/compare.h"
#include <algorithm>
#include "exceptions.hpp"
#include "../../mp/Threading.h"
#include "activation_functions.h"

namespace nt{
namespace functional{


inline void compare_equal(const Tensor& a, const Tensor& b){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(a, b);
    utils::THROW_EXCEPTION(a.shape() == b.shape(),
                           "\nRuntimeError: Expected shape a ($) to be equal to shape b ($) ",
                           a.shape(), b.shape());
    utils::THROW_EXCEPTION(
        a.dtype() == b.dtype(),
        "\nRuntimeError: Expected dtype a ($) to be equal to dtype b ($)",
        a.dtype(), b.dtype());
}


using DualFunc = Tensor (*)(const Tensor&, const Tensor&);
inline Tensor tensor_of_tensors(const Tensor& a, const Tensor& b, DualFunc func){
    for(const auto& _t : a)
        _NT_FUNCTIONAL_ALWAYS_CHECK_(_t);
    for(const auto& _t : b)
        _NT_FUNCTIONAL_ALWAYS_CHECK_(_t);
    Tensor out = Tensor::makeNullTensorArray(a.numel());
    Tensor* o_begin = reinterpret_cast<Tensor*>(out.data_ptr());
    a.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >(
        [&func, &o_begin](auto begin, auto end, auto begin2){
            for(;begin != end; ++begin, ++begin2, ++o_begin){
                *o_begin = func(*begin, *begin2);
            }
    }, b.arr_void());
    return out.view(a.shape());
}

using SingleNSFunc = Tensor (*)(const Tensor&);
inline Tensor tensor_of_tensors(const Tensor& a, SingleNSFunc func){
    for(const auto& _t : a)
        _NT_FUNCTIONAL_ALWAYS_CHECK_(_t);
    Tensor out = Tensor::makeNullTensorArray(a.numel());
    Tensor* o_begin = reinterpret_cast<Tensor*>(out.data_ptr());
    a.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >(
        [&func, &o_begin](auto begin, auto end){
            for(;begin != end; ++begin, ++o_begin){
                *o_begin = func(*begin);
            }
    });
    return out.view(a.shape());
}


Tensor equal(const Tensor& a, const Tensor& b){
    compare_equal(a, b);
    if(a.dtype() == DType::TensorObj){return tensor_of_tensors(a, b, &equal);}
    Tensor out(a.shape(), DType::Bool);
    cpu::_equal(out.arr_void(), a.arr_void(), b.arr_void());
    return out;
}

Tensor not_equal(const Tensor& a, const Tensor& b){
    compare_equal(a, b);
    if(a.dtype() == DType::TensorObj){return tensor_of_tensors(a, b, &not_equal);}
    Tensor out(a.shape(), DType::Bool);
    cpu::_not_equal(out.arr_void(), a.arr_void(), b.arr_void());
    return std::move(out);
}
Tensor less_than(const Tensor& a, const Tensor& b){
    compare_equal(a, b);
    if(a.dtype() == DType::TensorObj){return tensor_of_tensors(a, b, &less_than);}
    Tensor out(a.shape(), DType::Bool);
    cpu::_less_than(out.arr_void(), a.arr_void(), b.arr_void());
    return std::move(out);
}
Tensor greater_than(const Tensor& a, const Tensor& b){
    compare_equal(a, b);
    if(a.dtype() == DType::TensorObj){return tensor_of_tensors(a, b, &greater_than);}
    Tensor out(a.shape(), DType::Bool);
    cpu::_greater_than(out.arr_void(), a.arr_void(), b.arr_void());
    return std::move(out);
}
Tensor less_than_equal(const Tensor& a, const Tensor& b){
    compare_equal(a, b);
    if(a.dtype() == DType::TensorObj){return tensor_of_tensors(a, b, &less_than_equal);}
    Tensor out(a.shape(), DType::Bool);
    cpu::_less_than_equal(out.arr_void(), a.arr_void(), b.arr_void());
    return std::move(out);
}
Tensor greater_than_equal(const Tensor& a, const Tensor& b){
    compare_equal(a, b);
    if(a.dtype() == DType::TensorObj){return tensor_of_tensors(a, b, &greater_than_equal);}
    Tensor out(a.shape(), DType::Bool);
    cpu::_greater_than_equal(out.arr_void(), a.arr_void(), b.arr_void());
    return std::move(out);
}
Tensor and_op(const Tensor& a, const Tensor& b){
    compare_equal(a, b);
    if(a.dtype() == DType::TensorObj){return tensor_of_tensors(a, b, &and_op);}
    utils::throw_exception(a.dtype() == DType::Bool,
                           "and operator only works on bool dtypes got $", a.dtype());
    Tensor out(a.shape(), DType::Bool);
    cpu::_and_op(out.arr_void(), a.arr_void(), b.arr_void());
    return std::move(out);
}
Tensor or_op(const Tensor& a, const Tensor& b){
    compare_equal(a, b);
    if(a.dtype() == DType::TensorObj){return tensor_of_tensors(a, b, &or_op);}
    utils::throw_exception(a.dtype() == DType::Bool,
                           "or operator only works on bool dtypes got $", a.dtype());
    Tensor out(a.shape(), DType::Bool);
    cpu::_or_op(out.arr_void(), a.arr_void(), b.arr_void());
    return std::move(out);
}

Tensor isnan(const Tensor& a){
    if(a.dtype() == DType::TensorObj){return tensor_of_tensors(a, &isnan);}
    Tensor out(a.shape(), DType::Bool);
    cpu::_isnan(out.arr_void(), a.arr_void());
    return std::move(out);
}

using SingleFunc = Tensor (*)(const Tensor&, Scalar);
inline Tensor tensor_of_scalars(const Tensor& a, Scalar b, SingleFunc func){
    Tensor out = Tensor::makeNullTensorArray(a.numel());
    Tensor* o_begin = reinterpret_cast<Tensor*>(out.data_ptr());
    a.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >(
        [&func, &o_begin, &b](auto begin, auto end){
            for(;begin != end; ++begin, ++o_begin){
                *o_begin = func(*begin, b);
            }
    });
    return out.view(a.shape());
}


Tensor equal(const Tensor& a, Scalar b){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(a);
    Tensor out(a.shape(), DType::Bool);
    cpu::_equal(out.arr_void(), a.arr_void(), b);
    return std::move(out);
}
Tensor not_equal(const Tensor& a, Scalar b){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(a);
    Tensor out(a.shape(), DType::Bool);
    cpu::_not_equal(out.arr_void(), a.arr_void(), b);
    return std::move(out);
}
Tensor less_than(const Tensor& a, Scalar b){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(a);
    Tensor out(a.shape(), DType::Bool);
    cpu::_less_than(out.arr_void(), a.arr_void(), b);
    return std::move(out);
}
Tensor greater_than(const Tensor& a, Scalar b){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(a);
    Tensor out(a.shape(), DType::Bool);
    cpu::_greater_than(out.arr_void(), a.arr_void(), b);
    return std::move(out);
}
Tensor less_than_equal(const Tensor& a, Scalar b){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(a);
    Tensor out(a.shape(), DType::Bool);
    cpu::_less_than_equal(out.arr_void(), a.arr_void(), b);
    return std::move(out);
}
Tensor greater_than_equal(const Tensor& a, Scalar b){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(a);
    Tensor out(a.shape(), DType::Bool);
    cpu::_greater_than_equal(out.arr_void(), a.arr_void(), b);
    return std::move(out);
}
bool all(const Tensor & t){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    if(t.dtype() == DType::TensorObj){
		return t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
			return std::all_of(begin, end, [](const Tensor& v){return all(v);});
		});
	}
    utils::throw_exception(t.dtype() == DType::Bool,
                           "Expected dtype for all to be bool got $", t.dtype());
    return cpu::_all(t.arr_void());
}
bool any(const Tensor & t){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    if(t.dtype() == DType::TensorObj){
		return t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
			return std::any_of(begin, end, [](const Tensor& v){return any(v);});
		});
	}
    utils::throw_exception(t.dtype() == DType::Bool,
                           "Expected dtype for all to be bool got $", t.dtype());
    return cpu::_any(t.arr_void());
}
bool none(const Tensor & t){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    if(t.dtype() == DType::TensorObj){
		return t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
			return std::all_of(begin, end, [](const Tensor& v){return none(v);});
		});
	}
    utils::throw_exception(t.dtype() == DType::Bool,
                           "Expected dtype for all to be bool got $", t.dtype());
    return cpu::_none(t.arr_void());
}

int64_t amount_of(Tensor t, Scalar val){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    if(t.dtype() == DType::TensorObj){
		return t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&val](auto begin, auto end){
            int64_t count = 0;
            for(;begin != end; ++begin){
                count += amount_of(*begin, val);
            }
		    return count;
        });
	}
    return cpu::_amount_of(t.arr_void(), val);
}


int64_t count(Tensor t){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    if(t.dtype() == DType::TensorObj){
		return t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
            int64_t _count = 0;
            for(;begin != end; ++begin){
                _count += count(*begin);
            }
		    return _count;
        });
	}
    return cpu::_count(t.arr_void());
}


inline void next_index(const SizeRef& s, std::vector<int64_t>& v, typename SizeRef::value_type index){
	if(v[index] == s[index] - 1){
		v[index] = 0;
		if(index == 0){
			std::fill(v.begin(), v.end(), 0);
			next_index(s, v, v.size()-1);
			return;
		}
		--index;
		next_index(s, v, index);
	}
	else{
		++v[index];
	}
}


//inefficient, but for some reason seems to be the only way it works?
//maybe look back into this function in the future
Tensor where(Tensor t){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
	utils::THROW_EXCEPTION(t.is_contiguous(), "Expected contiguous tensor for where");
    if(t.dtype() == DType::TensorObj){
        Tensor output = Tensor::makeNullTensorArray(t.numel());
        Tensor* ts_begin = reinterpret_cast<Tensor*>(output.data_ptr());
        Tensor* ts_end = ts_begin + t.numel();
        Tensor* begin = reinterpret_cast<Tensor*>(t.data_ptr());
        for(;ts_begin != ts_end; ++ts_begin, ++begin)
            *ts_begin = where(*begin);
        return std::move(output);
    }
	utils::THROW_EXCEPTION(t.dtype() == DType::Bool, "Expected dtype to be DType::Bool but got $", t.dtype());
	uint_bool_t looking(true);
	size_t amt = amount_of(t, looking);
    if(amt == 0){
        return nt::Tensor::Null();
    }
    //special case to speed this up because this happens a lot in the persistent homology class
    if(amt == 1 && t.dims() == 1){
        int64_t counter = 0;
        uint_bool_t* begin = reinterpret_cast<uint_bool_t*>(t.data_ptr());
        uint_bool_t* end = begin + t.numel();
        for(;begin != end; ++begin, ++counter){
            if(*begin == looking)
                break;
        }
        Tensor outp = Tensor::makeNullTensorArray(1);
        Tensor& cords = *reinterpret_cast<Tensor*>(outp.data_ptr());
        cords = Tensor({1}, DType::int64);
        *reinterpret_cast<int64_t*>(cords.data_ptr()) = counter;
        return std::move(outp);
    }
    if(t.dims() == 1){
        Tensor outp = Tensor::makeNullTensorArray(1);
        Tensor& cords = *reinterpret_cast<Tensor*>(outp.data_ptr());
        cords = Tensor({static_cast<int64_t>(amt)}, DType::int64);
        int64_t* data = reinterpret_cast<int64_t*>(cords.data_ptr());
        int64_t counter = 0;
        uint_bool_t* begin = reinterpret_cast<uint_bool_t*>(t.data_ptr());
        uint_bool_t* end = begin + t.numel();
        for(;begin != end; ++begin, ++counter){
            if(*begin == looking){
                *data = counter;
                ++data;
            }
        }
        return std::move(outp); 
    }
	Tensor outp({static_cast<typename SizeRef::value_type>(t.dims()), static_cast<typename SizeRef::value_type>(amt)}, DType::int64);
	
	Tensor ts = outp.split_axis_1();
	std::vector<int64_t> indexes(t.dims(), 0);
	uint_bool_t* begin = reinterpret_cast<uint_bool_t*>(t.data_ptr());
	uint_bool_t* end = begin + t.numel();
	const typename SizeRef::value_type index = indexes.size() - 1;
	int64_t keeping = 0;
	Tensor* ts_begin = reinterpret_cast<Tensor*>(ts.data_ptr());
	Tensor* ts_end = ts_begin + ts.numel();
	Tensor* ts_cpy = ts_begin;
	for(;begin != end; ++begin, next_index(t.shape(), indexes, index)){
		if(*begin == looking){
			auto cbegin = indexes.cbegin();
			ts_begin->arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::int64>>>([&cbegin](auto a_begin, auto a_end){
				for(;a_begin != a_end; ++a_begin, ++cbegin){
					*a_begin = *cbegin;
				}
			});
			++ts_begin;
		}
	}
	return outp.split_axis(0);
}


Tensor all(const Tensor t, int64_t dim){
    if(t.dtype() == DType::TensorObj){
        Tensor out = Tensor::makeNullTensorArray(t.numel());
        Tensor* begin_o = reinterpret_cast<Tensor*>(out.data_ptr());
        t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&begin_o, &dim](auto begin, auto end){
            for(;begin != end; ++begin, ++begin_o){
                *begin_o = all(*begin, dim);
            }
		});
        return std::move(out);
    }
	exception_dtypes(t.dtype(), DType::Bool);
        Tensor a = Tensor::Null();
    if(dim == (t.dims()-1) || dim == -1){
        a = t.transpose(-1, -2).contiguous();
        dim = -2;
    }else{
        a = t.contiguous();
    }
    Tensor split = a.split_axis(dim);
    
    Tensor out({split.numel()}, nt::DType::Bool);
    bool* begin_o = reinterpret_cast<bool*>(out.data_ptr());
    Tensor* begin_s = reinterpret_cast<Tensor*>(split.data_ptr());
    tbb::parallel_for(
    utils::calculateGrainSize1D(split.numel()),
    [&](const tbb::blocked_range<int64_t> &range){
        for(int64_t i = range.begin(); i != range.end(); ++i){
            begin_o[i] = std::all_of(reinterpret_cast<uint_bool_t*>(begin_s[i].data_ptr()), reinterpret_cast<uint_bool_t*>(begin_s[i].data_ptr_end()),
                                     [](const uint_bool_t& v){return v.value == 1;});
        }
    });
    return std::move(out);

}

Tensor any(const Tensor t, int64_t dim){
    if(t.dtype() == DType::TensorObj){
        Tensor out = Tensor::makeNullTensorArray(t.numel());
        Tensor* begin_o = reinterpret_cast<Tensor*>(out.data_ptr());
        t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&begin_o, &dim](auto begin, auto end){
            for(;begin != end; ++begin, ++begin_o){
                *begin_o = any(*begin, dim);
            }
		});
        return std::move(out);
    }
	exception_dtypes(t.dtype(), DType::Bool);
    // Tensor a = t.contiguous();
    Tensor a = Tensor::Null();
    if(dim == (t.dims()-1) || dim == -1){
        a = t.transpose(-1, -2).contiguous();
        dim = -2;
    }else{
        a = t.contiguous();
    }
    Tensor split = a.split_axis(dim);
    Tensor out({split.numel()}, nt::DType::Bool);
    bool* begin_o = reinterpret_cast<bool*>(out.data_ptr());
    Tensor* begin_s = reinterpret_cast<Tensor*>(split.data_ptr());
    threading::preferential_parallel_for(
    threading::block_ranges<1>(0, split.numel()),
    [&](threading::blocked_range<1> block) {
       for(int64_t i = block.begin[0]; i != block.end[0]; ++i){
            begin_o[i] = std::any_of(reinterpret_cast<uint_bool_t*>(begin_s[i].data_ptr()), 
                                     reinterpret_cast<uint_bool_t*>(begin_s[i].data_ptr_end()),
                                     [](const uint_bool_t& v){return v.value == 1;});
        } 
    });
    return std::move(out);
}

Tensor isclose(const Tensor& input, const Tensor& other, Scalar rtol, Scalar atol, bool equal_nan){
    //complex types are seperated due to things like nan values
    //for example:
    //(nan, 1.32)
    //(nan, nan)
    //if equal_nan is true, the above evaluates to true
    utils::throw_exception(input.dtype() == other.dtype(), "\nError: input dtype ($) is expected to equal other dtype ($) for isclose", input.dtype(), other.dtype());
    if(DTypeFuncs::is_complex(input.dtype())){
        return isclose(input.real(), other.real(), rtol, atol, equal_nan) && isclose(input.imag(), other.imag(), rtol, atol, equal_nan);
    }
    utils::throw_exception(input.shape() == other.shape(), "\nError: input shape ($) is expected to equal other shape ($) for isclose", input.shape(), other.shape());
    Tensor isnan_i = isnan(input);
    Tensor isnan_o = isnan(other);
    bool i_has_nan = any(isnan_i);
    bool o_has_nan = any(isnan_o);
    // if(equal_nan){
    //     if((i_has_nan && !o_has_nan) | (o_has_nan && !i_has_nan)) return false;
    // }
    // if(!equal_nan){
    //     if(i_has_nan || o_has_nan){return false;}
    // }
    Tensor subtract = (input - other);
    abs_(subtract);
    Tensor right = rtol * abs(other);
    right += atol;
    Tensor out = less_than_equal(subtract, right);
    if(equal_nan){
        if(i_has_nan){
            out = or_op(out, (isnan_i == isnan_o));
        }
    }else{
       out = and_op(out, and_op(isnan_i == false, isnan_o == false)); 
    }
    return std::move(out);
}


bool allclose(const Tensor& input, const Tensor& other, Scalar rtol, Scalar atol, bool equal_nan){
    utils::throw_exception(input.dtype() == other.dtype(), "\nError: input dtype ($) is expected to equal other dtype ($) for allclose", input.dtype(), other.dtype());
    if(DTypeFuncs::is_complex(input.dtype())){
        return allclose(input.real(), other.real(), rtol, atol, equal_nan) && allclose(input.imag(), other.imag(), rtol, atol, equal_nan);
    }
    utils::throw_exception(input.shape() == other.shape(), "\nError: input shape ($) is expected to equal other shape ($) for allclose", input.shape(), other.shape());

    Tensor isnan_i = isnan(input);
    Tensor isnan_o = isnan(other);
    bool i_has_nan = any(isnan_i);
    bool o_has_nan = any(isnan_o);
    if(equal_nan){
        if((i_has_nan && !o_has_nan) || (o_has_nan && !i_has_nan)) return false;
    }
    if(!equal_nan){
        if(i_has_nan || o_has_nan){return false;}
    }
    Tensor subtract = (input - other);
    abs_(subtract);
    Tensor right = rtol * abs(other);
    right += atol;
    Tensor out = less_than_equal(subtract, right);
    if(equal_nan && i_has_nan){
        if(!all(isnan_i == isnan_o)) return false; //is nan in the same places
        out = or_op(out, isnan_i);
    }
    return all(out);
}


}
}
