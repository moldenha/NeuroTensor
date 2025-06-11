#include "combine.h"
#include "../../Tensor.h"
#include "../../utils/utils.h"
#include <vector>
#include "../../dtype/ArrayVoid.h"
#include "../../dtype/ArrayVoid.hpp"
#include "exceptions.hpp"

namespace nt{
namespace functional{


Tensor cat_0_dim(const Tensor& _a, const Tensor& _b){
	_NT_FUNCTIONAL_ALWAYS_CHECK_(_a, _b);
    ::nt::utils::THROW_EXCEPTION(_a.dims() == _b.dims(), "\nDims must be equal $ != $", _a.dims(), _b.dims());
	::nt::utils::THROW_EXCEPTION(_a.dtype == _b.dtype, "\nDTypes must be equal $ != $", _a.dtype, _b.dtype);
	for(typename SizeRef::value_type i = 1; i < _a.dims(); ++i)
		utils::THROW_EXCEPTION(_a.shape()[i] == _b.shape()[i],
				"\nRuntimeError: Sizes of tensors must match except in dimension 0. Expected size $ but got size $ for second tensor.", _a.shape()[i], _b.shape()[i]);
	std::vector<typename SizeRef::ArrayRefInt::value_type> n_vals = _a.shape().Vec();
	n_vals[0] += _b.shape()[0];
	return Tensor(ArrayVoid::cat(_a.arr_void(), _b.arr_void()), SizeRef(std::move(n_vals))).set_mutability(_a.is_mutable() && _b.is_mutable());
}



Tensor cat(const Tensor& _a, const Tensor& _b, int64_t dim){
	_NT_FUNCTIONAL_ALWAYS_CHECK_(_a, _b);
    if(dim == 0) return cat_0_dim(_a, _b);
    return cat_0_dim(_a.transpose(0, dim), _b.transpose(0, dim)).transpose(0, dim).set_mutability(_a.is_mutable() && _b.is_mutable());
}

Tensor cat(std::vector<Tensor> t){
	for(const auto _t : t)
        _NT_FUNCTIONAL_ALWAYS_CHECK_(_t);
    bool is_mutable_ = true;
    for(const auto _t : t)
        is_mutable_ = is_mutable_ && _t.is_mutable();

	typename SizeRef::value_type last_dim = 0;
	for(typename SizeRef::value_type i = 1; i < t.size(); ++i){
		exception_dtypes(t[i-1].dtype, t[i].dtype);
		utils::THROW_EXCEPTION(t[i-1].dims() == t[i].dims(), "Runtime Error: Expected all tensors to have $ dims but got $ instead", t[i-1].dims(), t[i].dims());
		for(typename SizeRef::value_type j = 1; j < t[i].dims(); ++j){
			utils::THROW_EXCEPTION(t[i].shape()[j] == t[i-1].shape()[j],
				"Runtime Error: Expected tensors to have same shape ($) at dim $ but got ($) instead",
				t[i-1].shape()[j], j, t[i].shape()[j]);
		}
		last_dim += t[i-1].shape()[0];
	}
	last_dim += t.back().shape()[0];
	std::vector<typename SizeRef::value_type> vec = t[0].shape().Vec();
	vec[0] = last_dim;
	SizeRef a(std::move(vec));
	std::vector<std::reference_wrapper<const ArrayVoid> > arrVds;
	arrVds.reserve(t.size());
	for(Tensor& x : t)
		arrVds.push_back(std::cref(x.arr_void()));
	return Tensor(ArrayVoid::cat(arrVds), a).set_mutability(is_mutable_);

}

Tensor cat(const Tensor& t){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    bool is_mutable_ = t.is_mutable();
	for(const auto _t : t){
        _NT_FUNCTIONAL_ALWAYS_CHECK_(_t);
        is_mutable_ = is_mutable_ && _t.is_mutable();
    }
	utils::THROW_EXCEPTION(t.dtype == DType::TensorObj,
			"In order to concatenate a tensor, it must hold multiple tensors, but got type $", t.dtype);
	const typename SizeRef::value_type& num = t.numel();
	return t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&num, &is_mutable_](auto begin, auto end) -> Tensor{
		auto begin_cpy = begin;
		const SizeRef sh = begin->shape();
		++begin;
		for(;begin != end; ++begin){
			utils::THROW_EXCEPTION(begin->shape().pop_front() == sh.pop_front(), 
                          "Expected all shapes in concatenate to be the same, except for dim (0) but got $ and $", begin->shape(), sh);
		}
		begin = begin_cpy;
		std::vector<typename SizeRef::value_type> vec = sh.Vec();
        ++begin;
        for(;begin != end; ++begin)
            vec[0] += begin->shape()[0];
		std::vector<std::reference_wrapper<const ArrayVoid> > arrVds;
		arrVds.reserve(num); //okay because it is allocating a reference wrapper, putting a number there would cause an allocation error
		begin = begin_cpy;
		typename SizeRef::value_type i = 0;
		for(typename SizeRef::value_type i = 0; begin != end; ++begin, ++i){
			arrVds.push_back(std::cref(begin->arr_void()));
		}
		return Tensor(ArrayVoid::cat(arrVds), SizeRef(std::move(vec))).set_mutability(is_mutable_);
		
	});
}

Tensor cat_unordered(const Tensor& t){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    bool is_mutable_ = t.is_mutable();
	for(const auto _t : t){
        _NT_FUNCTIONAL_ALWAYS_CHECK_(_t);
        is_mutable_ = is_mutable_ && _t.is_mutable();
    }

	utils::THROW_EXCEPTION(t.dtype == DType::TensorObj,
			"In order to concatenate a single tensor, it must hold multiple tensors, but got type $", t.dtype);
	std::vector<ArrayVoid> arrVds;
	arrVds.reserve(t.numel());
	for(const auto& tensor : t){
		arrVds.push_back(tensor.arr_void());
	}

	ArrayVoid outp = ArrayVoid::cat(arrVds);
	const auto n_numel = outp.Size();
	return Tensor(std::move(outp), {static_cast<typename SizeRef::value_type>(n_numel)}).set_mutability(is_mutable_);
}

Tensor cat_unordered(const std::vector<Tensor>& t){
    bool is_mutable_ = true;
	for(const auto _t : t){
        _NT_FUNCTIONAL_ALWAYS_CHECK_(_t);
        is_mutable_ = is_mutable_ && _t.is_mutable();
    }
	std::vector<std::reference_wrapper<const ArrayVoid> > arrVds;
	arrVds.reserve(t.size());
	for(const auto& tensor : t){
		arrVds.push_back(std::cref(tensor.arr_void()));
	}
	ArrayVoid outp = ArrayVoid::cat(arrVds);
	const auto n_numel = outp.Size();
	return Tensor(std::move(outp), {static_cast<typename SizeRef::value_type>(n_numel)}).set_mutability(is_mutable_);
}

Tensor _NT_cat_vec_(std::vector<Tensor> &tgs) {
    bool is_mutable_ = true;
	for(const auto _t : tgs){
        _NT_FUNCTIONAL_ALWAYS_CHECK_(_t);
        is_mutable_ = is_mutable_ && _t.is_mutable();
    }
    const typename SizeRef::value_type &num = tgs.size();
    auto begin = tgs.begin();
    auto end = tgs.end();
    const SizeRef sh = begin->shape();
    const SizeRef sh_smaller = sh.pop_front();
    int64_t n_dim_size = sh[0];
    auto begin_cpy = begin;
    ++begin;
    for (; begin != end; ++begin) {
        n_dim_size += begin->shape()[0];
        utils::THROW_EXCEPTION(begin->shape().pop_front() == sh_smaller,
                                                     "Expected all shapes in concatenate to be the "
                                                     "same, but got $ and $",
                                                     begin->shape().pop_front(), sh_smaller);
    }
    std::vector<typename SizeRef::value_type> vec = sh.Vec();
    vec[0] = n_dim_size;
    std::vector<std::reference_wrapper<const ArrayVoid>> arrVds;
    arrVds.reserve(num); // okay because it is allocating a reference wrapper,
                                             // putting a number there would cause an allocation error
    begin = begin_cpy;
    typename SizeRef::value_type i = 0;
    for (typename SizeRef::value_type i = 0; begin != end; ++begin, ++i) {
        arrVds.push_back(std::cref(begin->arr_void()));
    }
    return Tensor(ArrayVoid::cat(arrVds), SizeRef(std::move(vec))).set_mutability(is_mutable_);
}

Tensor _NT_cat_vec_(std::vector<Tensor> &tgs, int64_t dim) {

    bool is_mutable_ = true;
	for(const auto _t : tgs){
        _NT_FUNCTIONAL_ALWAYS_CHECK_(_t);
        is_mutable_ = is_mutable_ && _t.is_mutable();
    }
    if (dim == 0) {
        return _NT_cat_vec_(tgs);
    }
    const typename SizeRef::value_type &num = tgs.size();
    auto begin = tgs.begin();
    auto end = tgs.end();
    const SizeRef sh = begin->shape().transpose(0, dim);
    int64_t n_dim_size = sh[0];
    const SizeRef sh_smaller = sh.pop_front();
    auto begin_cpy = begin;
    ++begin;
    for (; begin != end; ++begin) {
        n_dim_size += begin->shape()[dim];
        utils::THROW_EXCEPTION(begin->shape().transpose(0, dim).pop_front() ==
                                                             sh_smaller,
                                                     "Expected all shapes in concatenate to be the "
                                                     "same, but got $ and $",
                                                     begin->shape(), sh);
    }
    std::vector<typename SizeRef::value_type> vec = sh.Vec();
    vec[0] = n_dim_size;
    std::vector<ArrayVoid> arrVds;
    //arrVds.reserve(num); // okay because it is allocating a reference wrapper,
    // putting a number there would cause an allocation error
    begin = begin_cpy;
    typename SizeRef::value_type i = 0;
    for (typename SizeRef::value_type i = 0; begin != end; ++begin, ++i) {
        arrVds.push_back(begin->transpose(0, dim).arr_void());
    }
    SizeRef shape(std::move(vec));
    return Tensor(ArrayVoid::cat(arrVds), std::move(shape)).transpose(0, dim).set_mutability(is_mutable_);
}

Tensor cat(std::vector<Tensor> t, int64_t dim){
    return _NT_cat_vec_(t, dim);
}

Tensor cat(const Tensor& t, int64_t dim){

	utils::THROW_EXCEPTION(t.dtype == DType::TensorObj,
			"In order to concatenate a tensor, it must hold multiple tensors, but got type $", t.dtype);
    if(dim == 0){
        return cat(t);
    }
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    bool is_mutable_ = t.is_mutable();
	for(const auto _t : t){
        _NT_FUNCTIONAL_ALWAYS_CHECK_(_t);
        is_mutable_ = is_mutable_ && _t.is_mutable();
    }

	const typename SizeRef::value_type& num = t.numel();
	return t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&num, &dim, &is_mutable_](auto begin, auto end) -> Tensor{
		auto begin_cpy = begin;
		const SizeRef sh = begin->shape().transpose(0, dim);
		int64_t n_dim_size = sh[0];
		const SizeRef sh_smaller = sh.pop_front();
		++begin;
		for(;begin != end; ++begin){
			n_dim_size += begin->shape()[dim];
			utils::THROW_EXCEPTION(begin->shape().transpose(0, dim).pop_front() == sh_smaller, "Expected all shapes in concatenate to be the same, but got $ and $", begin->shape(), sh);
		}
		std::vector<typename SizeRef::value_type> vec = sh.Vec();
		vec[0] = n_dim_size;
		std::vector<ArrayVoid> arrVds;
		arrVds.reserve(num); //okay because it is allocating a reference wrapper, putting a number there would cause an allocation error
		begin = begin_cpy;
		typename SizeRef::value_type i = 0;
		for(typename SizeRef::value_type i = 0; begin != end; ++begin, ++i){
			arrVds.push_back(begin->transpose(0, dim).arr_void());
		}
		return Tensor(ArrayVoid::cat(arrVds), SizeRef(std::move(vec))).transpose(0, dim).set_mutability(is_mutable_);
		
	});
	
}


Tensor stack(std::vector<std::reference_wrapper<Tensor> > t, int64_t dim){
    bool is_mutable_ = true;
	for(const auto _t : t){
        _NT_FUNCTIONAL_ALWAYS_CHECK_(_t.get());
        is_mutable_ = is_mutable_ && _t.get().is_mutable();
    }

	for(typename SizeRef::value_type i = 1; i < t.size(); ++i){
		utils::THROW_EXCEPTION(!t[i-1].get().is_null(),
				"Cannot stack a null tensor!");
		utils::THROW_EXCEPTION(!t[i].get().is_null(),
				"Cannot stack a null tensor!");
		utils::THROW_EXCEPTION(t[i-1].get().shape() == t[i].get().shape(),
				"Runtime Error: Expected all Tensors to have same shape of $ but instead got $", 
				t[i-1].get().shape(), t[i].get().shape());
		exception_dtypes(t[i-1].get().dtype, t[i].get().dtype);
	}
	std::vector<typename SizeRef::value_type> vec = t[0].get().shape().Vec();
	dim = dim < 0 ? dim + t[0].get().dims() : dim;
	vec.insert(vec.begin(), t.size());
	SizeRef a(std::move(vec));
	std::vector<ArrayVoid> arrVds;
	arrVds.reserve(t.size());
	for(auto& x : t)
		arrVds.push_back(x.get().arr_void());
	return Tensor(ArrayVoid::cat(arrVds), a).transpose(0, dim).set_mutability(is_mutable_);
}


Tensor stack(const Tensor& t, int64_t dim){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    bool is_mutable_ = t.is_mutable();
	for(const auto _t : t){
        _NT_FUNCTIONAL_ALWAYS_CHECK_(_t);
        is_mutable_ = is_mutable_ && _t.is_mutable();
    }
	utils::THROW_EXCEPTION(t.dtype == DType::TensorObj,
			"In order to concatenate a tensor, it must hold multiple tensors, but got type $", t.dtype);
	return t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&dim, &is_mutable_](auto begin, auto end){
		auto begin_cpy = begin;
		const SizeRef& original_shape = begin_cpy->shape();
		DType out_dtype = begin_cpy->dtype;
        const Tensor& t = *begin;
		for(;begin_cpy != end; ++begin_cpy){
			utils::THROW_EXCEPTION(!begin_cpy->is_null(),
					"Cannot stack a null tensor!");
			utils::THROW_EXCEPTION(original_shape == begin_cpy->shape(),
				"Runtime Error: Expected all Tensors to have same shape of $ but instead got $",
				original_shape, begin_cpy->shape());
			exception_dtypes(out_dtype, begin_cpy->dtype);
		}
		std::vector<typename SizeRef::value_type> vec = original_shape.Vec();
		dim = dim < 0 ? dim + t.dims() : dim;
		vec.insert(vec.begin() + dim, (end-begin));
		SizeRef a(std::move(vec));
		std::vector<ArrayVoid> arrVds;
		arrVds.reserve((end-begin));
		for(;begin != end; ++begin){
			arrVds.push_back(begin->arr_void());
		}
		return Tensor(ArrayVoid::cat(arrVds), a).transpose(0, dim).set_mutability(is_mutable_);

		
	});
}


Tensor stack(std::vector<std::reference_wrapper<Tensor> > t){
    bool is_mutable_ = true;
	for(const auto _t : t){
        _NT_FUNCTIONAL_ALWAYS_CHECK_(_t.get());
        is_mutable_ = is_mutable_ && _t.get().is_mutable();
    }
	bool are_same = true;
	for(typename SizeRef::value_type i = 1; i < t.size(); ++i){
		if(t[i-1].get().shape() != t[i].get().shape() || t[i-1].get().dtype != t[i].get().dtype){
			are_same = false;
			break;
		}
	}
	if(are_same){
		std::vector<typename SizeRef::value_type> vec = t[0].get().shape().Vec();
		vec.insert(vec.begin(), t.size());
		SizeRef a(std::move(vec));
		Tensor output(std::move(a), t[0].get().dtype);
		for(typename SizeRef::value_type i = 0; i < t.size(); ++i)
			output[i] = t[i].get();
		return output.set_mutability(is_mutable_);
	}

	Tensor output({static_cast<typename SizeRef::value_type>(t.size())}, DType::TensorObj);
	for(typename SizeRef::value_type i = 0; i < t.size(); ++i){
		output[i] = t[i].get();
	}
	return output.set_mutability(is_mutable_);
}

Tensor stack(std::vector<Tensor> t){
    bool is_mutable_ = true;
	for(const auto _t : t){
        _NT_FUNCTIONAL_ALWAYS_CHECK_(_t);
        is_mutable_ = is_mutable_ && _t.is_mutable();
    }
	bool are_same = true;
	for(typename SizeRef::value_type i = 1; i < t.size(); ++i){
		if(t[i-1].shape() != t[i].shape() || t[i-1].dtype != t[i].dtype){
			are_same = false;
			break;
		}
	}
	if(are_same){
		std::vector<typename SizeRef::value_type> vec = t[0].shape().Vec();
		vec.insert(vec.begin(), t.size());
		SizeRef a(std::move(vec));
		Tensor output(std::move(a), t[0].dtype);
		for(typename SizeRef::value_type i = 0; i < t.size(); ++i)
			output[i] = t[i];
		return output.set_mutability(is_mutable_);
	}
	Tensor output({static_cast<typename SizeRef::value_type>(t.size())}, DType::TensorObj);
	for(typename SizeRef::value_type i = 0; i < t.size(); ++i)
		output[i] = t[i];
	return output.set_mutability(is_mutable_);
}

Tensor stack(std::vector<Tensor> t, int64_t dim){
    bool is_mutable_ = true;
	for(const auto _t : t){
        _NT_FUNCTIONAL_ALWAYS_CHECK_(_t);
        is_mutable_ = is_mutable_ && _t.is_mutable();
    }
	for(typename SizeRef::value_type i = 1; i < t.size(); ++i){
		utils::THROW_EXCEPTION(!t[i-1].is_null(),
				"Cannot stack a null tensor!");
		utils::THROW_EXCEPTION(!t[i].is_null(),
				"Cannot stack a null tensor!");
		utils::THROW_EXCEPTION(t[i-1].shape() == t[i].shape(),
				"Runtime Error: Expected all Tensors to have same shape of $ but instead got $",
				t[i-1].shape(), t[i].shape());
		exception_dtypes(t[i-1].dtype, t[i].dtype);
	}
	std::vector<typename SizeRef::value_type> vec = t[0].shape().Vec();
	dim = dim < 0 ? dim + t[0].dims() : dim;
	vec.insert(vec.begin() + dim, t.size());
	SizeRef a(std::move(vec));
	std::vector<ArrayVoid> arrVds;
	arrVds.reserve(t.size());
	for(Tensor& x : t)
		arrVds.push_back(x.arr_void());

	return Tensor(ArrayVoid::cat(arrVds), a).transpose(0, dim).set_mutability(is_mutable_);
}



Tensor vectorize(std::vector<Tensor> t){
    bool is_mutable_ = true;
	for(const auto _t : t){
        _NT_FUNCTIONAL_ALWAYS_CHECK_(_t);
        is_mutable_ = is_mutable_ && _t.is_mutable();
    }
	Tensor output = Tensor::makeNullTensorArray(t.size());
	Tensor* begin = reinterpret_cast<Tensor*>(output.data_ptr());
	Tensor* end = begin + output.numel();
	for(typename SizeRef::value_type i = 0; i < t.size(); ++i, ++begin)
		*begin = t[i];
	return output.set_mutability(is_mutable_);
}


}
}
