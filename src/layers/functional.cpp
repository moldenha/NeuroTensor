#include "../functional/functional.h"
#include "functional.h"
#include "TensorGrad.h"
#include "../Tensor.h"
#include "../intrusive_ptr/intrusive_ptr.hpp"

namespace nt{
namespace functional{
TensorGrad matmult(const TensorGrad& a, const TensorGrad& b){
	if(!a.do_track_grad){
		if(!b.do_track_grad){
			Tensor out = matmult(a.tensor, b.tensor);
			TensorGrad result(std::move(out));
			result.do_track_grad = false;
			return std::move(result);
		}
		return matmult(a.tensor, b);
	}
	if(!b.do_track_grad){return matmult(a, b.tensor);}
	// a and b are going to have to be cloned anyways so:

	intrusive_ptr<tensor_holder> a_c = make_intrusive<tensor_holder>(a.tensor.clone());
	intrusive_ptr<tensor_holder> b_c = make_intrusive<tensor_holder>(b.tensor.clone());
	TensorGrad result(::nt::functional::matmult(a_c->tensor, b_c->tensor));
	result.track_tensors(a, b);

	// Define backward function
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
					intrusive_ptr<tensor_holder> a, intrusive_ptr<tensor_holder> b) {


		parents[0]->grad->tensor += ::nt::functional::matmult(grad, b->tensor, false, true);

		parents[1]->grad->tensor += ::nt::functional::matmult(a->tensor, grad, true, false);

	}, a_c, b_c);
	return result;
}

TensorGrad matmult(const Tensor& a, const TensorGrad& b){
	if(!b.do_track_grad){
		Tensor out = matmult(a, b.tensor);
		TensorGrad result(std::move(out));
		result.do_track_grad = false;
		return std::move(result);
	}
	intrusive_ptr<tensor_holder> a_c = make_intrusive<tensor_holder>(a.clone());
	TensorGrad result(::nt::functional::matmult(a_c->tensor, b.tensor));
	result.track_tensors(b);

	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
				intrusive_ptr<tensor_holder> a){
			parents[0]->grad->tensor += ::nt::functional::matmult(a->tensor, grad, 1, 0);
	}, a_c);
	return result;
}

TensorGrad matmult(const TensorGrad& a, const Tensor& b){
	if(!a.do_track_grad){
		Tensor out = matmult(a.tensor, b);
		TensorGrad result(std::move(out));
		result.do_track_grad = false;
		return std::move(result);
	}
	intrusive_ptr<tensor_holder> b_c = make_intrusive<tensor_holder>(b.clone());
	TensorGrad result(::nt::functional::matmult(a.tensor, b_c->tensor));
	result.track_tensors(a);

	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
				intrusive_ptr<tensor_holder> b){
		parents[0]->grad->tensor += ::nt::functional::matmult(grad, b->tensor, 0, 1);
	}, b_c);
	return result;
}

TensorGrad unfold3d(const TensorGrad& x, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> dilation, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> stride, bool transpose_out){
	TensorGrad result(unfold3d(x.tensor, kernel_size, dilation, padding, stride, transpose_out));
	result.track_tensors(x);
	result.create_backward_function([kernel_size, dilation, padding, stride, transpose_out]
			(const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
		utils::my_n_tuple<3> output_size(parents[0]->grad->tensor.shape()[-3], parents[0]->grad->tensor.shape()[-2], parents[0]->grad->tensor.shape()[-1]);
		unfold3d_backward(grad, parents[0]->grad->tensor, output_size, kernel_size, dilation, padding, stride, transpose_out);
	});
	return std::move(result);

}

TensorGrad unfold1d(const TensorGrad& x, Tensor::size_value_t kernel_size, Tensor::size_value_t dilation, Tensor::size_value_t padding, Tensor::size_value_t stride, bool transpose_out){
	TensorGrad result(unfold1d(x.tensor, kernel_size, dilation, padding, stride, transpose_out));
	result.track_tensors(x);
	result.create_backward_function([kernel_size, dilation, padding, stride, transpose_out]
			(const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
		Tensor::size_value_t output_size = parents[0]->grad->tensor.shape()[-1];
		unfold1d_backward(grad, parents[0]->grad->tensor, output_size, kernel_size, dilation, padding, stride, transpose_out);
	});
	return std::move(result);

}

TensorGrad fold(const TensorGrad& x, utils::my_tuple output_size, utils::my_tuple kernel_size, utils::my_tuple dilation, utils::my_tuple padding, utils::my_tuple stride){
	TensorGrad result(fold(x.tensor, output_size, kernel_size, dilation, padding, stride));
	result.track_tensors(x);
	//it is coppied because the backward pass will go out of scope of this function
	//and so I dont want that memory to try to be referenced
	result.create_backward_function([output_size, kernel_size, dilation, padding, stride]
			(const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
			
		fold_backward(grad, parents[0]->grad->tensor, output_size, kernel_size, dilation, padding, stride);

	});
	return std::move(result);



}

TensorGrad unfold(const TensorGrad& x, utils::my_tuple kernel_size, utils::my_tuple dilation, utils::my_tuple padding, utils::my_tuple stride, bool transpose_out){
	TensorGrad result(unfold(x.tensor, kernel_size, dilation, padding, stride, transpose_out));
	result.track_tensors(x);
	result.create_backward_function([kernel_size, dilation, padding, stride, transpose_out]
			(const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
		utils::my_tuple output_size(parents[0]->grad->tensor.shape()[-2], parents[0]->grad->tensor.shape()[-1]);
		unfold_backward(grad, parents[0]->grad->tensor, output_size, kernel_size, dilation, padding, stride, transpose_out);
	});
	return std::move(result);
}

TensorGrad sigmoid(const TensorGrad& x){
	Tensor a = (-1) * x.tensor;
	a.exp_();
	a += 1;
	a.inverse_();
	intrusive_ptr<tensor_holder> sigmoid_x = make_intrusive<tensor_holder>(a.clone());
	TensorGrad result(std::move(a));
	result.track_tensors(x);
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents, intrusive_ptr<tensor_holder> x){
		parents[0]->grad->tensor += grad * (x->tensor * (1.0 - x->tensor));
	}, sigmoid_x);
	return std::move(result);
}


TensorGrad clamp(const TensorGrad& x, std::optional<int64_t> min, std::optional<int64_t> max){
	TensorGrad out = x.clone();
	if(min && max){
		out[out < min.value() && out > max.value()] = 0;
		return std::move(out);
	}
	else if(min)
		out[out < min.value()] = 0;
	else if(max)
		out[out > max.value()] = 0;
	return std::move(out);
}

TensorGrad relu(const TensorGrad& x){
	return clamp(x, 0);
}

TensorGrad var(const TensorGrad& x, utils::optional_list dim , int64_t correction, bool keepdim){
	if(!x.do_track_grad){
		Tensor out = var(x.tensor, dim, correction, keepdim);
		TensorGrad result(std::move(out));
		result.do_track_grad = false;
		return std::move(result);
	}
	intrusive_ptr<tensor_holder> x_c = make_intrusive<tensor_holder>(x.tensor.clone());
	TensorGrad result(::nt::functional::var(x_c->tensor, dim, correction, keepdim));
	result.track_tensors(x);
	result.create_backward_function([dim, correction](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents, intrusive_ptr<tensor_holder> x){
		parents[0]->grad->tensor += dvar(grad, x->tensor, dim, correction);
	}, x_c);	
	return std::move(result);
}


TensorGrad invsqrt(const TensorGrad& x){
	TensorGrad result(invsqrt(x.tensor));
	if(!x.do_track_grad){
		result.do_track_grad = false;
		return std::move(result);
	}

	result.track_tensors(x);
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
		parents[0]->grad->tensor += dinvsqrt(grad);
	});
	return std::move(result);
}

TensorGrad tanh(const TensorGrad& x){
	TensorGrad result(tanh(x.tensor));
	if(!x.do_track_grad){
		result.do_track_grad = false;
		return std::move(result);
	}

	result.track_tensors(x);
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
		parents[0]->grad->tensor += dtanh(grad);
	});
	return std::move(result);
}

TensorGrad tan(const TensorGrad& x){
	TensorGrad result(tan(x.tensor));
	if(!x.do_track_grad){
		result.do_track_grad = false;
		return std::move(result);
	}

	result.track_tensors(x);
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
		parents[0]->grad->tensor += dtan(grad);
	});
	return std::move(result);
}

Tensor cat_vec(std::vector<TensorGrad>& tgs){
	const typename SizeRef::value_type& num = tgs.size();
	auto begin = tgs.begin();
	auto end = tgs.end();
	const SizeRef sh = begin->shape();
	const SizeRef sh_smaller = sh.pop_front();
	int64_t n_dim_size = sh[0];
	auto begin_cpy = begin;
	++begin;
	for(;begin != end; ++begin){
		n_dim_size += begin->shape()[0];
		utils::THROW_EXCEPTION(begin->shape().pop_front() == sh_smaller, "Expected all shapes in concatenate to be the same, but got $ and $", begin->shape().pop_front(), sh_smaller);
	}
	std::vector<typename SizeRef::value_type> vec = sh.Vec();
	vec[0] = n_dim_size;
	std::vector<std::reference_wrapper<const ArrayVoid> > arrVds;
	arrVds.reserve(num); //okay because it is allocating a reference wrapper, putting a number there would cause an allocation error
	begin = begin_cpy;
	typename SizeRef::value_type i = 0;
	for(typename SizeRef::value_type i = 0; begin != end; ++begin, ++i){
		arrVds.push_back(std::cref(begin->tensor.arr_void()));
	}
	return Tensor(ArrayVoid::cat(arrVds), SizeRef(std::move(vec)));
		
}

Tensor cat_vec(std::vector<TensorGrad>& tgs, int64_t dim){
	if(dim == 0){return cat_vec(tgs);}
	const typename SizeRef::value_type& num = tgs.size();
	auto begin = tgs.begin();
	auto end = tgs.end();
	const SizeRef sh = begin->shape().transpose(0, dim);
	int64_t n_dim_size = sh[0];
	const SizeRef sh_smaller = sh.pop_front();
	auto begin_cpy = begin;
	++begin;
	for(;begin != end; ++begin){
		n_dim_size += begin->shape()[dim];
		utils::THROW_EXCEPTION(begin->shape().transpose(0, dim).pop_front() == sh_smaller, "Expected all shapes in concatenate to be the same, but got $ and $", begin->shape(), sh);
	}
	std::vector<typename SizeRef::value_type> vec = sh.Vec();
	vec[0] = n_dim_size;
	std::vector<std::reference_wrapper<const ArrayVoid> > arrVds;
	arrVds.reserve(num); //okay because it is allocating a reference wrapper, putting a number there would cause an allocation error
	begin = begin_cpy;
	typename SizeRef::value_type i = 0;
	for(typename SizeRef::value_type i = 0; begin != end; ++begin, ++i){
		arrVds.push_back(std::cref(begin->tensor.transpose(0, dim).arr_void()));
	}
	return Tensor(ArrayVoid::cat(arrVds), SizeRef(std::move(vec)));
		
}

Tensor cat_vec_grad(std::vector<intrusive_ptr<TensorGrad>>& tgs){
	const typename SizeRef::value_type& num = tgs.size();
	auto begin = tgs.begin();
	auto end = tgs.end();
	const SizeRef sh = (*begin)->shape();
	const SizeRef sh_smaller = sh.pop_front();
	int64_t n_dim_size = sh[0];
	auto begin_cpy = begin;
	++begin;
	for(;begin != end; ++begin){
		n_dim_size += (*begin)->shape()[0];
		utils::THROW_EXCEPTION((*begin)->shape().pop_front() == sh_smaller, "Expected all shapes in concatenate to be the same, but got $ and $", (*begin)->shape().pop_front(), sh_smaller);
	}
	std::vector<typename SizeRef::value_type> vec = sh.Vec();
	vec[0] = n_dim_size;
	std::vector<std::reference_wrapper<const ArrayVoid> > arrVds;
	arrVds.reserve(num); //okay because it is allocating a reference wrapper, putting a number there would cause an allocation error
	begin = begin_cpy;
	typename SizeRef::value_type i = 0;
	for(typename SizeRef::value_type i = 0; begin != end; ++begin, ++i){
		arrVds.push_back(std::cref((*begin)->grad->tensor.arr_void()));
	}
	return Tensor(ArrayVoid::cat(arrVds), SizeRef(std::move(vec)));
		
}

Tensor cat_vec_grad(std::vector<intrusive_ptr<TensorGrad>>& tgs, int64_t dim){
	if(dim == 0){return cat_vec_grad(tgs);}
	const typename SizeRef::value_type& num = tgs.size();
	auto begin = tgs.begin();
	auto end = tgs.end();
	const SizeRef sh = (*begin)->shape().transpose(0, dim);
	int64_t n_dim_size = sh[0];
	const SizeRef sh_smaller = sh.pop_front();
	auto begin_cpy = begin;
	++begin;
	for(;begin != end; ++begin){
		n_dim_size += (*begin)->shape()[dim];
		utils::THROW_EXCEPTION((*begin)->shape().transpose(0, dim).pop_front() == sh_smaller, "Expected all shapes in concatenate to be the same, but got $ and $", (*begin)->shape(), sh);
	}
	std::vector<typename SizeRef::value_type> vec = sh.Vec();
	vec[0] = n_dim_size;
	std::vector<std::reference_wrapper<const ArrayVoid> > arrVds;
	arrVds.reserve(num); //okay because it is allocating a reference wrapper, putting a number there would cause an allocation error
	begin = begin_cpy;
	typename SizeRef::value_type i = 0;
	for(typename SizeRef::value_type i = 0; begin != end; ++begin, ++i){
		arrVds.push_back(std::cref((*begin)->grad->tensor.transpose(0, dim).arr_void()));
	}
	return Tensor(ArrayVoid::cat(arrVds), SizeRef(std::move(vec)));
		
}




TensorGrad cat(std::vector<TensorGrad> tgs, int64_t dim){
	bool track_grad = tgs[0].do_track_grad;
	for(const auto& tg : tgs){
		utils::throw_exception(tg.do_track_grad == track_grad, "Cannot concatenate tensors that are both tracking the gradient and are not");
		utils::throw_exception(!tg.is_null(), "Cannot concatenate null tensors");
	}
	TensorGrad result(cat_vec(tgs, dim));
	if(!track_grad){
		result.do_track_grad = false;
		return std::move(result);
	}

	//tracking the gradient itself
	//rather than tracking each parent individually
	result.parents.clear();
	result.parents.reserve(tgs.size());
	for(const auto& tg : tgs){
		if(tg.grad == nullptr){
			tg.grad = make_intrusive<tensor_holder>(functional::zeros_like(tg.tensor));
		}
		result.parents.push_back(make_intrusive<TensorGrad>(tg));
	}
	result.grad = make_intrusive<tensor_holder>(cat_vec_grad(result.parents));
	intrusive_ptr<TensorGrad> res_intrusive = make_intrusive<TensorGrad>(result);
	for(const auto& tg : tgs){
		tg.children->push_back(res_intrusive);
	}
	return std::move(result);
}


}
}
