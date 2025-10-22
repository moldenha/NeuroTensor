#include "normalize.h"
#include "../cpu/normalize.h"
#include "../../dtype/ArrayVoid.h"
#include "../../dtype/ArrayVoid.hpp"
#include <cmath>
#include "exceptions.hpp"
#include "activation_functions.h"
#include "fill.h"
#include "combine.h"

namespace nt::functional {


Tensor& xavier_uniform_(Tensor& tensor){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(tensor);
    utils::throw_exception(tensor.is_mutable(), "Can only perform xavier_uniform_ on a mutable tensor");
    utils::throw_exception(tensor.dims() >= 2, "For xavier uniform the dimensions of the tensor must be greater than or equal to 2");
    int64_t fan_in = tensor.shape()[-1]; //switch to [1] maybe
    int64_t fan_out = tensor.shape()[-2]; //switch to [0] maybe
    double bound = std::sqrt(6.0 / (double)(fan_in + fan_out));
    cpu::xavier_uniform_(tensor.arr_void(), bound);
    return tensor;
}


Tensor var(const Tensor& x, utils::optional_list dim, int64_t correction, bool keepdim){
	Tensor mean = x.mean(dim, true);
	Tensor squared_diff = pow((x - mean), 2);
	int64_t N = 0;
	if(!dim){
		N = x.numel();
	}else{
		N = 1;
		for(const auto& ele : dim){
			N *= x.shape()[ele];
		}
	}
	Tensor variance = squared_diff.sum(dim, keepdim) / (N - correction);
	return std::move(variance);
}

Tensor dvar(const Tensor& dx, const Tensor& x, utils::optional_list dim, int64_t correction){
	//takes both the gradient, and the input given to the variance function
	Tensor mean = x.mean(dim, true);
	int64_t N = 0;
	if(!dim){
		N = x.numel();
	}else{
		N = 1;
		for(const auto& ele : dim){
			N *= x.shape()[ele];
		}
	}
	return (2 / (N - correction)) * (x - mean);
}

namespace no_grad{

Tensor& batch_norm_(Tensor& x, Tensor running_mean, Tensor running_var, 
                                    utils::optional_tensor weight, utils::optional_tensor bias,
                                    bool training, Scalar momentum, Scalar eps,
                                    intrusive_ptr<tensor_holder> stored_means, intrusive_ptr<tensor_holder> stored_inv){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x, running_mean, running_var);
    if(weight.has_value())
        _NT_FUNCTIONAL_ALWAYS_CHECK_(weight.value());
    if(bias.has_value())
        _NT_FUNCTIONAL_ALWAYS_CHECK_(bias.value());
    check_mutability(x);
    if(training){
        utils::throw_exception(running_mean.is_mutable() && running_var.is_mutable(), 
                               "Error: when training for batch norm running mean and running var must be mutable");
    }
    utils::throw_exception(DTypeFuncs::is_floating(x.dtype()) || DTypeFuncs::is_complex(x.dtype()),
            "batch_norm_ can only happen on floating dtypes but got $", x.dtype());
    utils::throw_exception(x.dtype() == running_mean.dtype() && running_var.dtype() == x.dtype(),
                           "Expected x, running_mean, and running_var to have the same dtype but got ($, $, $)",
                           x.dtype(), running_mean.dtype(), running_var.dtype());
    if(weight.has_value())
        utils::throw_exception(x.dtype() == weight.value().dtype(),
                           "Expected x, running_mean, running_var, and weight to have the same dtype but got ($, $, $, $)",
                           x.dtype(), running_mean.dtype(), running_var.dtype(), weight.value().dtype());
    if(bias.has_value())
        utils::throw_exception(x.dtype() == bias.value().dtype(),
                           "Expected x, running_mean, running_var, and weight to have the same dtype but got ($, $, $, $)",
                           x.dtype(), running_mean.dtype(), running_var.dtype(), bias.value().dtype());
    
    utils::throw_exception(x.dims() >= 3, "Error, the number of dimensions for input ($) must be greater than or equal to 3 for batch_norm_",
                           x.dims());
    int64_t N = x.shape()[0], C = x.shape()[1];
    int64_t HW = x.numel() / (N * C);
    
    Tensor _bias = bias.has_value() ? bias.value().contiguous() : zeros({C}, x.dtype());
    Tensor _weight = weight.has_value() ? weight.value().contiguous() : ones({C}, x.dtype());
    utils::throw_exception(_bias.numel() == C,
                           "Error, bias shape must be {$} but got $ with input shape of $",
                           C, _bias.shape(), x.shape());
    utils::throw_exception(_weight.numel() == C,
                           "Error, weight shape must be {$} but got $ with input shape of $",
                           C, _weight.shape(), x.shape());
    utils::throw_exception(running_var.numel() == C,
                           "Error, running_var shape must be {$} but got $ with input shape of $",
                           C, running_var.shape(), x.shape());
    utils::throw_exception(running_mean.numel() == C,
                           "Error, running_mean shape must be {$} but got $ with input shape of $",
                           C, running_mean.shape(), x.shape());

    utils::throw_exception((!bool(stored_means) && !bool(stored_inv)) || (bool(stored_means) && bool(stored_inv)),
                           "Error: you must either have both sotred means and stored inv or none of either for no_grad::batch_norm_");
    if(stored_means){
        _NT_FUNCTIONAL_ALWAYS_CHECK_(stored_means->tensor, stored_inv->tensor);
        // stored arrays must be C long and same dtype
        utils::throw_exception(stored_means->tensor.numel() == C,
            "stored_means must have numel == C ($) but got $", C, stored_means->tensor.numel());
        utils::throw_exception(stored_inv->tensor.numel() == C,
            "stored_inv must have numel == C ($) but got $", C, stored_inv->tensor.numel());
        utils::throw_exception(stored_means->tensor.dtype() == x.dtype() && stored_inv->tensor.dtype() == x.dtype(),
            "stored_means and stored_inv must have same dtype as input");
        cpu::_batch_norm_(x.arr_void(), running_mean.arr_void(), running_var.arr_void(),
                          _weight.arr_void(), _bias.arr_void(), training, momentum, eps,
                          N, C, HW, stored_means->tensor.arr_void(), stored_inv->tensor.arr_void());
    }else{
        cpu::_batch_norm_(x.arr_void(), running_mean.arr_void(), running_var.arr_void(),
                      _weight.arr_void(), _bias.arr_void(), training, momentum, eps, N, C, HW);
    }
    return x;
}

Tensor batch_norm(const Tensor& x, Tensor running_mean, Tensor running_var, 
                                    utils::optional_tensor weight, utils::optional_tensor bias,
                                    bool training, Scalar momentum, Scalar eps,
                                    intrusive_ptr<tensor_holder> stored_means, intrusive_ptr<tensor_holder> stored_inv){
    Tensor output = x.clone();
    batch_norm_(output, std::move(running_mean), std::move(running_var), std::move(weight), std::move(bias), training, momentum, eps,
                stored_means, stored_inv);
    return std::move(output);
}



Tensor batch_norm_backward(const Tensor& grad, const Tensor& input,
                                           const Tensor& stored_means, const Tensor& stored_inv,
                                           Tensor original_weight, Tensor original_bias,
                                           Scalar momentum, Scalar eps){

    _NT_FUNCTIONAL_ALWAYS_CHECK_(grad, input, stored_means, stored_inv);
    utils::throw_exception(grad.dtype() == input.dtype(),
                           "Error, expected grad dtype $ to match input ($) dtype",
                           grad.dtype(), input.dtype());
    utils::throw_exception(grad.shape() == input.shape(),
                           "Error expected grad shape ($) to match input shape ($)",
                           grad.shape(), input.shape());
    utils::throw_exception(stored_means.dtype() == grad.dtype() && stored_inv.dtype() == grad.dtype(),
                           "Error, expected stored_means ($) and stored_inv ($) to be the same dtype as grad ($)",
                           stored_means.dtype(), stored_inv.dtype(), grad.dtype());

    if(!original_weight.is_null()){
        utils::throw_exception(grad.dtype() == original_weight.dtype(),
                               "Erorr: Expected grad dtype ($) to be the same as the original weight ($)",
                               grad.dtype(), original_weight.dtype());
    }
    if(!original_bias.is_null()){
        utils::throw_exception(grad.dtype() == original_bias.dtype(),
                               "Erorr: Expected grad dtype ($) to be the same as the original bias ($)",
                               grad.dtype(), original_bias.dtype());
    }
    int64_t N = input.shape()[0], C = input.shape()[1];
    int64_t HW = input.numel() / (N * C);
    if(original_weight.is_null() && original_bias.is_null()){
        Tensor weight = ones({C}, input.dtype());
        Tensor grad_input = zeros_like(input);
        cpu::_batch_norm_backward_input_(grad_input.arr_void(),
                                            grad.arr_void(), input.arr_void(),
                                            weight.arr_void(), momentum, eps, N, C, HW,
                                            stored_means.arr_void(), stored_inv.arr_void());
        return std::move(grad_input);
    }
    
    Tensor weight = original_weight.is_null() ? ones({C}, input.dtype()) : original_weight;
    Tensor grad_input = zeros_like(input);
    cpu::_batch_norm_backward_input_(grad_input.arr_void(),
                                        grad.arr_void(), input.arr_void(),
                                        weight.arr_void(), momentum, eps, N, C, HW,
                                        stored_means.arr_void(), stored_inv.arr_void());

    Tensor grad_weight = zeros({C}, input.dtype());
    Tensor grad_bias = zeros({C}, input.dtype());
    cpu::_batch_norm_backward_weight_bias_(grad_weight.arr_void(), grad_bias.arr_void(), grad.arr_void(),
                                            input.arr_void(), 
                                           momentum, eps, N, C, HW, stored_means.arr_void(), stored_inv.arr_void());
    return list(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
    
}


Tensor& group_norm_(Tensor& input, int64_t num_groups, 
                    utils::optional_tensor weight, utils::optional_tensor bias,
                    Scalar eps, intrusive_ptr<tensor_holder> stored_means, intrusive_ptr<tensor_holder> stored_inv){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    if(weight.has_value())
        _NT_FUNCTIONAL_ALWAYS_CHECK_(weight.value());
    if(bias.has_value())
        _NT_FUNCTIONAL_ALWAYS_CHECK_(bias.value());
    check_mutability(input);
    utils::throw_exception(DTypeFuncs::is_floating(input.dtype()) || DTypeFuncs::is_complex(input.dtype()),
            "group_norm_ can only happen on floating dtypes but got $", input.dtype());
    if(weight.has_value())
        utils::throw_exception(input.dtype() == weight.value().dtype(),
                           "Expected input ($) and weight ($) to have the same dtype",
                           input.dtype(), weight.value().dtype());
    if(bias.has_value())
        utils::throw_exception(input.dtype() == bias.value().dtype(),
                           "Expected input ($) and bias ($) to have the same dtype",
                           input.dtype(), bias.value().dtype());
    
    utils::throw_exception(input.dims() >= 3, "Error, the number of dimensions for input ($) must be greater than or equal to 3 for group_norm_",
                           input.dims());
    int64_t N = input.shape()[0], C = input.shape()[1];
    int64_t HW = input.numel() / (N * C);
    utils::throw_exception(num_groups > 0 && (C % num_groups) == 0,
        "group_norm_ num_groups must divide C (got num_groups=$, C=$)", num_groups, C);
    
    Tensor _bias = bias.has_value() ? bias.value().contiguous() : zeros({C}, input.dtype());
    Tensor _weight = weight.has_value() ? weight.value().contiguous() : ones({C}, input.dtype());
    utils::throw_exception(_bias.numel() == C,
                           "Error, bias shape must be {$} but got $ with input shape of $",
                           C, _bias.shape(), input.shape());
    utils::throw_exception(_weight.numel() == C,
                           "Error, weight shape must be {$} but got $ with input shape of $",
                           C, _weight.shape(), input.shape());
    
        // stored mean/inv must be provided together or not at all
        utils::throw_exception((!bool(stored_means) && !bool(stored_inv)) || (bool(stored_means) && bool(stored_inv)),
            "Error: you must either have both stored means and stored inv or none of either for group_norm_");    
    if(stored_means){
        _NT_FUNCTIONAL_ALWAYS_CHECK_(stored_means->tensor, stored_inv->tensor);
        // stored arrays must be num_groups long and same dtype
        utils::throw_exception(stored_means->tensor.numel() == num_groups,
            "stored_means must have numel == num_groups ($) but got $", num_groups, stored_means->tensor.numel());
        utils::throw_exception(stored_inv->tensor.numel() == num_groups,
            "stored_inv must have numel == num_groups ($) but got $", num_groups, stored_inv->tensor.numel());
        utils::throw_exception(stored_means->tensor.dtype() == input.dtype() && stored_inv->tensor.dtype() == input.dtype(),
            "stored_means and stored_inv must have same dtype as input");
        
        // call ArrayVoid implementation that writes/reads stored values
        cpu::_group_norm_(
            input.arr_void(),
            _weight.arr_void(),
            _bias.arr_void(),
            N, C, HW,
            num_groups,
            eps,
            stored_means->tensor.arr_void(),
            stored_inv->tensor.arr_void()
        );
    }else{
        cpu::_group_norm_(
            input.arr_void(),
            _weight.arr_void(),
            _bias.arr_void(),
            N, C, HW,
            num_groups,
            eps
        );

    }
    return input;
}


Tensor group_norm(const Tensor& input, int64_t num_groups, 
                    utils::optional_tensor weight, utils::optional_tensor bias,
                    Scalar eps, intrusive_ptr<tensor_holder> stored_means, intrusive_ptr<tensor_holder> stored_inv){
    Tensor output = input.clone();
    group_norm_(output, num_groups, weight, bias, eps, stored_means, stored_inv);
    return std::move(output);
}


Tensor group_norm_backward(const Tensor& grad, const Tensor& input, int64_t num_groups,
                                           const Tensor& stored_means, const Tensor& stored_inv,
                                           Tensor original_weight, Tensor original_bias, Scalar eps){

    _NT_FUNCTIONAL_ALWAYS_CHECK_(grad, input, stored_means, stored_inv); utils::throw_exception(grad.dtype() == input.dtype(),
                           "Error, expected grad dtype $ to match input ($) dtype",
                           grad.dtype(), input.dtype());
    utils::throw_exception(grad.shape() == input.shape(),
                           "Error expected grad shape ($) to match input shape ($)",
                           grad.shape(), input.shape());
    if(!original_weight.is_null()){
        utils::throw_exception(grad.dtype() == original_weight.dtype(),
                               "Erorr: Expected grad dtype ($) to be the same as the original weight ($)",
                               grad.dtype(), original_weight.dtype());
    }
    if(!original_bias.is_null()){
        utils::throw_exception(grad.dtype() == original_bias.dtype(),
                               "Erorr: Expected grad dtype ($) to be the same as the original bias ($)",
                               grad.dtype(), original_bias.dtype());
    }
    int64_t N = input.shape()[0], C = input.shape()[1];
    int64_t HW = input.numel() / (N * C);
    if(original_weight.is_null() && original_bias.is_null()){
        Tensor weight = ones({C}, input.dtype());
        Tensor grad_input = zeros_like(input);
        cpu::_group_norm_backward_input_(grad_input.arr_void(), grad.arr_void(), input.arr_void(),
                                                 weight.arr_void(), N, C, HW,
                                                 num_groups, eps, stored_means.arr_void(), stored_inv.arr_void());
        return std::move(grad_input);
    }
    
    Tensor weight = original_weight.is_null() ? ones({C}, input.dtype()) : original_weight;
    Tensor grad_input = zeros_like(input);
    cpu::_group_norm_backward_input_(grad_input.arr_void(), grad.arr_void(), input.arr_void(),
                                         weight.arr_void(), N, C, HW,
                                         num_groups, eps, stored_means.arr_void(), stored_inv.arr_void());


    Tensor grad_weight = zeros({C}, input.dtype());
    Tensor grad_bias = zeros({C}, input.dtype());
    cpu::_group_norm_backward_weight_bias_(grad_weight.arr_void(), grad_bias.arr_void(), grad.arr_void(),
                                                    input.arr_void(), N, C, HW,
                                                    num_groups, stored_means.arr_void(), stored_inv.arr_void());
    return list(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
    
}


} // nt::functional::no_grad::
} // nt::functional::
