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

		/* parents[0]->grad->tensor += ::nt::functional::matmult(grad, b->tensor, false, true); */
		//TODO: transposing during matmult not working
		parents[0]->grad->tensor += ::nt::functional::matmult(grad, b->tensor, false, true);

		/* parents[1]->grad->tensor += ::nt::functional::matmult(a->tensor, grad, true, false); */
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


}
}
