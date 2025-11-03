#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../utils/macros.h"
#include"../../utils/optional_tensorgrad.h"

namespace nt{
namespace functional{

TensorGrad TensorGrad_Functional_Class::var(const TensorGrad &x,
                                            utils::optional_list dim,
                                            int64_t correction, bool keepdim) {
    if (!x.track_grad()) {
        Tensor out = ::nt::functional::var(x.detach(), dim, correction, keepdim);
        TensorGrad result(std::move(out), x.track_grad());
        result.track_grad_(false);
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> x_c =
            make_intrusive<tensor_holder>(x.detach().conditional_mutate_clone());
    TensorGrad result(
            ::nt::functional::var(x_c->tensor, dim, correction, keepdim), x.track_grad());
    result.track_tensors(x);
    result.create_backward_function(
            [dim, correction](const Tensor &grad,
                                                std::vector<intrusive_ptr<TensorGrad>> &parents,
                                                intrusive_ptr<tensor_holder> x) {
                parents[0]->accumulate_gradient(::nt::functional::dvar(grad, x->tensor, dim, correction));
            },
            x_c);
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::batch_norm(const TensorGrad& x, const Tensor& running_mean, const Tensor& running_var,
                                                    utils::optional_tensorgrad weight, utils::optional_tensorgrad bias,
                                                    bool training, Scalar momentum, Scalar eps){
    if(!x.track_grad()){
        if(!weight.has_value() && !bias.has_value()){
            Tensor out = ::nt::functional::no_grad::batch_norm(x.detach(), running_mean, running_var, nullptr, nullptr, training, momentum, eps);
            TensorGrad result(std::move(out), x.track_grad());
            result.track_grad_(false);
            return std::move(result);
        }
        return TensorGrad_Functional_Class::batch_norm(x.detach(), running_mean, running_var, weight, bias, training, momentum, eps);
    }
    bool get_weight_grad = false;
    bool get_bias_grad = false;
    if(weight.has_value() && weight.value().track_grad())
        get_weight_grad = true;
    if(bias.has_value() && bias.value().track_grad())
        get_bias_grad = true;
    intrusive_ptr<tensor_holder> original_weight = 
            make_intrusive<tensor_holder>(bool(weight) ? weight.value().detach().conditional_mutate_clone() : Tensor::Null());
    intrusive_ptr<tensor_holder> original_input = make_intrusive<tensor_holder>(x.detach().conditional_mutate_clone());
    intrusive_ptr<tensor_holder> stored_means = make_intrusive<tensor_holder>(Tensor(running_var.shape(), running_var.dtype()));
    intrusive_ptr<tensor_holder> stored_inv = make_intrusive<tensor_holder>(Tensor(running_var.shape(), running_var.dtype()));
    TensorGrad result(::nt::functional::no_grad::batch_norm(
        x.detach(), running_mean, running_var,
        bool(weight) ? utils::optional_tensor(weight.value().detach()) : utils::optional_tensor(nullptr),
        bool(bias) ? utils::optional_tensor(bias.value().detach()) : utils::optional_tensor(nullptr),
        training, momentum, eps, stored_means, stored_inv), x.track_grad());

    if(get_weight_grad && get_bias_grad){
        result.track_tensors(x, weight.value(), bias.value());
    }
    else if(get_weight_grad){
        result.track_tensors(x, weight.value());
    }
    else if(get_bias_grad){
        result.track_tensors(x, bias.value());
    }else{
        result.track_tensors(x);
    }

    result.create_backward_function(
        [get_weight_grad, get_bias_grad, momentum, eps]
        (const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
         intrusive_ptr<tensor_holder> original_weight, intrusive_ptr<tensor_holder> original_input,
         intrusive_ptr<tensor_holder> stored_means, intrusive_ptr<tensor_holder> stored_inv) {
            Tensor out_grad = ::nt::functional::no_grad::batch_norm_backward(
                grad, original_input->tensor, 
                stored_means->tensor, stored_inv->tensor,
                original_weight->tensor,
                (!get_weight_grad && get_bias_grad) ? 
                        Tensor({1}, DType::Float32) : Tensor::Null(), // original bias doesn't actually matter, so just something non-null
                momentum, eps
            );
            if(!get_weight_grad && !get_bias_grad){
                parents[0]->accumulate_gradient(out_grad);
            }else{
                auto [input_grad, weight_grad, bias_grad] = ::nt::get<3>(out_grad);
                parents[0]->accumulate_gradient(input_grad);
                size_t cur = 1;
                if(get_weight_grad){
                    parents[cur]->accumulate_gradient(weight_grad);
                    ++cur;
                }if(get_bias_grad){
                    parents[cur]->accumulate_gradient(bias_grad);
                }
                
            }
        }, std::move(original_weight), std::move(original_input), std::move(stored_means), std::move(stored_inv));

    return std::move(result);

}


TensorGrad TensorGrad_Functional_Class::batch_norm(const Tensor& x, const Tensor& running_mean, const Tensor& running_var,
                                                    utils::optional_tensorgrad weight, utils::optional_tensorgrad bias,
                                                    bool training, Scalar momentum, Scalar eps){
    if(!weight.has_value() && !bias.has_value()){
        Tensor out = ::nt::functional::no_grad::batch_norm(x, running_mean, running_var, nullptr, nullptr, training, momentum, eps);
        TensorGrad result(std::move(out), false);
        result.track_grad_(false);
        return std::move(result);
    }
    bool get_weight_grad = false;
    bool get_bias_grad = false;
    if(weight.has_value() && weight.value().track_grad())
        get_weight_grad = true;
    if(bias.has_value() && bias.value().track_grad())
        get_bias_grad = true;
    if(!get_weight_grad && !get_bias_grad){
        Tensor out = ::nt::functional::no_grad::batch_norm(
                            x, running_mean, running_var,
                            weight.has_value() ? utils::optional_tensor(weight.value().detach()) : utils::optional_tensor(nullptr),
                            bias.has_value() ? utils::optional_tensor(bias.value().detach()) : utils::optional_tensor(nullptr),
                            training, momentum, eps);
        TensorGrad result(std::move(out), false);
        result.track_grad_(false);
        return std::move(result);
 
    }
    intrusive_ptr<tensor_holder> original_weight = 
            make_intrusive<tensor_holder>(bool(weight) ? weight.value().detach().conditional_mutate_clone() : Tensor::Null());
    intrusive_ptr<tensor_holder> original_input = make_intrusive<tensor_holder>(x.conditional_mutate_clone());
    intrusive_ptr<tensor_holder> stored_means = make_intrusive<tensor_holder>(Tensor(running_var.shape(), running_var.dtype()));
    intrusive_ptr<tensor_holder> stored_inv = make_intrusive<tensor_holder>(Tensor(running_var.shape(), running_var.dtype()));
    TensorGrad result(::nt::functional::no_grad::batch_norm(
        x, running_mean, running_var,
        bool(weight) ? utils::optional_tensor(weight.value().detach()) : utils::optional_tensor(nullptr),
        bool(bias) ? utils::optional_tensor(bias.value().detach()) : utils::optional_tensor(nullptr),
        training, momentum, eps, stored_means, stored_inv), true);

    if(get_weight_grad && get_bias_grad){
        result.track_tensors(weight.value(), bias.value());
    }
    else if(get_weight_grad){
        result.track_tensors(weight.value());
    }
    else if(get_bias_grad){
        result.track_tensors(bias.value());
    }
    else{
        utils::THROW_EXCEPTION(false, "INTERNAL LOGIC ERROR");
    }
    result.create_backward_function(
            [get_weight_grad, get_bias_grad, 
            momentum, eps]
            (const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
             intrusive_ptr<tensor_holder> original_weight, intrusive_ptr<tensor_holder> original_input,
             intrusive_ptr<tensor_holder> stored_means, intrusive_ptr<tensor_holder> stored_inv) {
                Tensor out_grad = ::nt::functional::no_grad::batch_norm_backward(
                    grad, original_input->tensor,
                    stored_means->tensor, stored_inv->tensor,
                    original_weight->tensor,
                    (!get_weight_grad && get_bias_grad) ? 
                        Tensor({1}, DType::Float32) : Tensor::Null(), // original bias doesn't actually matter, so just something non-null
                    momentum, eps
                );
                auto [input_grad, weight_grad, bias_grad] = ::nt::get<3>(out_grad);
                size_t cur = 0;
                if(get_weight_grad){
                    parents[cur]->accumulate_gradient(weight_grad);
                    ++cur;
                }if(get_bias_grad){
                    parents[cur]->accumulate_gradient(bias_grad);
                }
            }, std::move(original_weight), std::move(original_input), std::move(stored_means), std::move(stored_inv));
    return std::move(result);


}

TensorGrad TensorGrad_Functional_Class::group_norm(const TensorGrad& input, int64_t num_groups,
                                                    utils::optional_tensorgrad weight, utils::optional_tensorgrad bias,
                                                    Scalar eps){
    if(!input.track_grad()){
        if(!weight.has_value() && !bias.has_value()){
            Tensor out = ::nt::functional::no_grad::group_norm(input.detach(), num_groups, 
                                                               nullptr, nullptr, eps);
            TensorGrad result(std::move(out), input.track_grad());
            result.track_grad_(false);
            return std::move(result);
        }
        return TensorGrad_Functional_Class::group_norm(input.detach(), num_groups, weight, bias, eps);
    }
    bool get_weight_grad = false;
    bool get_bias_grad = false;
    if(weight.has_value() && weight.value().track_grad())
        get_weight_grad = true;
    if(bias.has_value() && bias.value().track_grad())
        get_bias_grad = true;
    intrusive_ptr<tensor_holder> original_weight = 
            make_intrusive<tensor_holder>(bool(weight) ? weight.value().detach().conditional_mutate_clone() : Tensor::Null());
    intrusive_ptr<tensor_holder> original_input = make_intrusive<tensor_holder>(input.detach().conditional_mutate_clone());
    intrusive_ptr<tensor_holder> stored_means = make_intrusive<tensor_holder>(Tensor({num_groups * input.shape()[0]}, input.dtype()));
    intrusive_ptr<tensor_holder> stored_inv = make_intrusive<tensor_holder>(Tensor({num_groups * input.shape()[0]}, input.dtype()));
    TensorGrad result(::nt::functional::no_grad::group_norm(
        input.detach(), num_groups,
        bool(weight) ? utils::optional_tensor(weight.value().detach()) : utils::optional_tensor(nullptr),
        bool(bias) ? utils::optional_tensor(bias.value().detach()) : utils::optional_tensor(nullptr),
        eps, stored_means, stored_inv), input.track_grad());

    if(get_weight_grad && get_bias_grad){
        result.track_tensors(input, weight.value(), bias.value());
    }
    else if(get_weight_grad){
        result.track_tensors(input, weight.value());
    }
    else if(get_bias_grad){
        result.track_tensors(input, bias.value());
    }else{
        result.track_tensors(input);
    }

    result.create_backward_function(
        [get_weight_grad, get_bias_grad, eps, num_groups]
        (const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
         intrusive_ptr<tensor_holder> original_weight, intrusive_ptr<tensor_holder> original_input,
         intrusive_ptr<tensor_holder> stored_means, intrusive_ptr<tensor_holder> stored_inv) {
            Tensor out_grad = ::nt::functional::no_grad::group_norm_backward(
                grad, original_input->tensor, num_groups,
                stored_means->tensor, stored_inv->tensor,
                original_weight->tensor,
                (!get_weight_grad && get_bias_grad) ? 
                        Tensor({1}, DType::Float32) : Tensor::Null(), // original bias doesn't actually matter, so just something non-null
                eps
            );
            if(!get_weight_grad && !get_bias_grad){
                parents[0]->accumulate_gradient(out_grad);
            }else{
                auto [input_grad, weight_grad, bias_grad] = ::nt::get<3>(out_grad);
                parents[0]->accumulate_gradient(input_grad);
                size_t cur = 1;
                if(get_weight_grad){
                    parents[cur]->accumulate_gradient(weight_grad);
                    ++cur;
                }if(get_bias_grad){
                    parents[cur]->accumulate_gradient(bias_grad);
                }
                
            }
        }, std::move(original_weight), std::move(original_input), std::move(stored_means), std::move(stored_inv));

    return std::move(result);

}


TensorGrad TensorGrad_Functional_Class::group_norm(const Tensor& input, int64_t num_groups,
                                                    utils::optional_tensorgrad weight, utils::optional_tensorgrad bias,
                                                    Scalar eps){
    if(!weight.has_value() && !bias.has_value()){
        Tensor out = ::nt::functional::no_grad::group_norm(input, num_groups, nullptr, nullptr, eps);
        TensorGrad result(std::move(out), false);
        result.track_grad_(false);
        return std::move(result);
    }
    bool get_weight_grad = false;
    bool get_bias_grad = false;
    if(weight.has_value() && weight.value().track_grad())
        get_weight_grad = true;
    if(bias.has_value() && bias.value().track_grad())
        get_bias_grad = true;
    if(!get_weight_grad && !get_bias_grad){
        Tensor out = ::nt::functional::no_grad::group_norm(input, num_groups, 
                            weight.has_value() ? utils::optional_tensor(weight.value().detach()) : utils::optional_tensor(nullptr),
                            bias.has_value() ? utils::optional_tensor(bias.value().detach()) : utils::optional_tensor(nullptr),
                            eps);
        TensorGrad result(std::move(out), false);
        result.track_grad_(false);
        return std::move(result);
 
    }
    intrusive_ptr<tensor_holder> original_weight = 
            make_intrusive<tensor_holder>(bool(weight) ? weight.value().detach().conditional_mutate_clone() : Tensor::Null());
    intrusive_ptr<tensor_holder> original_input = make_intrusive<tensor_holder>(input.conditional_mutate_clone());
    intrusive_ptr<tensor_holder> stored_means = make_intrusive<tensor_holder>(Tensor({num_groups * input.shape()[0]}, input.dtype()));
    intrusive_ptr<tensor_holder> stored_inv = make_intrusive<tensor_holder>(Tensor({num_groups * input.shape()[0]}, input.dtype()));
    TensorGrad result(::nt::functional::no_grad::group_norm(
        input, num_groups,
        bool(weight) ? utils::optional_tensor(weight.value().detach()) : utils::optional_tensor(nullptr),
        bool(bias) ? utils::optional_tensor(bias.value().detach()) : utils::optional_tensor(nullptr),
        eps, stored_means, stored_inv), true);

    if(get_weight_grad && get_bias_grad){
        result.track_tensors(weight.value(), bias.value());
    }
    else if(get_weight_grad){
        result.track_tensors(weight.value());
    }
    else if(get_bias_grad){
        result.track_tensors(bias.value());
    }
    else{
        utils::THROW_EXCEPTION(false, "INTERNAL LOGIC ERROR");
    }
    result.create_backward_function(
            [get_weight_grad, get_bias_grad, eps, num_groups]
            (const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
             intrusive_ptr<tensor_holder> original_weight, intrusive_ptr<tensor_holder> original_input,
             intrusive_ptr<tensor_holder> stored_means, intrusive_ptr<tensor_holder> stored_inv) {
                Tensor out_grad = ::nt::functional::no_grad::group_norm_backward(
                    grad, original_input->tensor, num_groups,
                    stored_means->tensor, stored_inv->tensor,
                    original_weight->tensor,
                    (!get_weight_grad && get_bias_grad) ? 
                            Tensor({1}, DType::Float32) : Tensor::Null(), // original bias doesn't actually matter, so just something non-null
                    eps
                );
                auto [input_grad, weight_grad, bias_grad] = ::nt::get<3>(out_grad);
                size_t cur = 0;
                if(get_weight_grad){
                    parents[cur]->accumulate_gradient(weight_grad);
                    ++cur;
                }if(get_bias_grad){
                    parents[cur]->accumulate_gradient(bias_grad);
                }
            }, std::move(original_weight), std::move(original_input), std::move(stored_means), std::move(stored_inv));
    return std::move(result);


}
}
}
