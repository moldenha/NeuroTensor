#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../utils/macros.h"

namespace nt{
namespace functional{


TensorGrad  TensorGrad_Functional_Class::conv1d(const Tensor& image, const TensorGrad& kernel, int64_t stride, int64_t padding, int64_t dilation, int64_t groups){
    if( kernel.track_grad() == false ){
        return TensorGrad(::nt::functional::conv1d(image, kernel.detach(), stride, padding, dilation, groups), false);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv1d(image, kernel.detach(), stride, padding, dilation, groups, original_x));
    result.track_tensors(kernel);
    result.create_backward_function([image_shape = std::move(image_shape), kernel_shape = std::move(kernel_shape), 
                                    stride = std::move(stride), padding = std::move(padding), 
                                    dilation = std::move(dilation), groups]
                                (const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img){
        ::nt::functional::conv_dkernel(grad, img->tensor, parents[0]->grad(), {image_shape[-1]}, groups, 1);
    }, original_x);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv1d(const TensorGrad& image, const Tensor& kernel, int64_t stride, int64_t padding, int64_t dilation, int64_t groups){
	if( image.track_grad() == false ){
        TensorGrad result(::nt::functional::conv1d(image.detach(), kernel, stride, padding, dilation, groups), false);
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv1d(image.detach(), kernel, stride, padding, dilation, groups, nullptr, original_w));
    result.track_tensors(image);
    result.create_backward_function([image_shape = std::move(image_shape), kernel_shape = std::move(kernel_shape), 
                                    stride = std::move(stride), padding = std::move(padding), 
                                    dilation = std::move(dilation), groups]
                                (const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> kern){
        ::nt::functional::conv_dimage(grad, kern->tensor, parents[0]->grad(), 
                                     {kernel_shape[-1]},
                                     {stride},
                                     {padding},
                                     {dilation},
                                     groups, 1);
    }, original_w);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv1d(const TensorGrad& image, const TensorGrad& kernel, int64_t stride, int64_t padding, int64_t dilation, int64_t groups){
	if( image.track_grad() == false ){
        return conv1d(image.detach(), kernel, stride, padding, dilation, groups);
    }
    if( kernel.track_grad() == false ){
        return conv1d(image, kernel.detach(), stride, padding, dilation, groups);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv1d(image.detach(), kernel.detach(), stride, padding, dilation, groups, original_x, original_w));
    result.track_tensors(image, kernel);
    result.create_backward_function([image_shape = std::move(image_shape), kernel_shape = std::move(kernel_shape), 
                                    stride = std::move(stride), padding = std::move(padding), 
                                    dilation = std::move(dilation), groups]
                                    (const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img, intrusive_ptr<tensor_holder> kern){
        ::nt::functional::conv_dimage(grad, kern->tensor, parents[0]->grad(), 
                                     {kernel_shape[-1]},
                                     {stride},
                                     {padding},
                                     {dilation},
                                     groups, 1);
        ::nt::functional::conv_dkernel(grad, img->tensor, parents[1]->grad(), {image_shape[-1]}, groups, 1);
    }, original_x, original_w);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv2d(const Tensor& image, const TensorGrad& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple dilation, int64_t groups){
	if( kernel.track_grad() == false){
        //if the kernel isn't tracking the gradient, then the gradient for neither is tracked
        TensorGrad result(::nt::functional::conv2d(image, kernel.detach(), stride, padding, dilation, groups), false);
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv2d(image, kernel.detach(), stride, padding, dilation, groups, original_x));
    result.track_tensors( kernel);
    result.create_backward_function([image_shape = std::move(image_shape), kernel_shape = std::move(kernel_shape), 
                                    stride = std::move(stride), padding = std::move(padding), 
                                    dilation = std::move(dilation), groups]
                                    (const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img){
          ::nt::functional::conv_dkernel(grad, img->tensor, parents[0]->grad(), {image_shape[-2], image_shape[-3]}, groups, 2);
    }, original_x);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv2d(const TensorGrad& image, const Tensor& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple dilation, int64_t groups){
	if( image.track_grad() == false){
        //if one of the tensors isn't tracking the gradient, then the gradient for neither is tracked
        TensorGrad result(::nt::functional::conv2d(image.detach(), kernel, stride, padding, dilation, groups), false);
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv2d(image.detach(), kernel, stride, padding, dilation, groups, nullptr, original_w));
    result.track_tensors(image);
    result.create_backward_function([image_shape = std::move(image_shape), kernel_shape = std::move(kernel_shape), 
                                    stride = std::move(stride), padding = std::move(padding), 
                                    dilation = std::move(dilation), groups]
                                    (const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> kern){
        ::nt::functional::conv_dimage(grad, kern->tensor, parents[0]->grad(), 
                                     {kernel_shape[-2], kernel_shape[-1]},
                                     {stride[0], stride[1]},
                                     {padding[0], padding[1]},
                                     {dilation[0], dilation[1]},
                                     groups, 2);
    }, original_w);
    return std::move(result);
}


TensorGrad  TensorGrad_Functional_Class::conv2d(const TensorGrad& image, const TensorGrad& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple dilation, int64_t groups){
	if( image.track_grad() == false ){
        return conv2d(image.detach(), kernel, stride, padding, dilation, groups);
    }
    if( kernel.track_grad() == false ){
        return conv2d(image, kernel.detach(), stride, padding, dilation, groups);
    }

    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv2d(image.detach(), kernel.detach(), stride, padding, dilation, groups, original_x, original_w));
    result.track_tensors(image, kernel);
    result.create_backward_function([image_shape = std::move(image_shape), kernel_shape = std::move(kernel_shape), 
                                    stride = std::move(stride), padding = std::move(padding), 
                                    dilation = std::move(dilation), groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img, intrusive_ptr<tensor_holder> kern){
        ::nt::functional::conv_dimage(grad, kern->tensor, parents[0]->grad(), 
                                     {kernel_shape[-2], kernel_shape[-1]},
                                     {stride[0], stride[1]},
                                     {padding[0], padding[1]},
                                     {dilation[0], dilation[1]},
                                     groups, 2);
        ::nt::functional::conv_dkernel(grad, img->tensor, parents[1]->grad(), {image_shape[-2], image_shape[-3]}, groups, 2);
    }, original_x, original_w);
    return std::move(result);
}



TensorGrad  TensorGrad_Functional_Class::conv3d(const Tensor& image, const TensorGrad& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> dilation, int64_t groups){
    if( kernel.track_grad() == false ){
        return TensorGrad(::nt::functional::conv3d(image, kernel.detach(), stride, padding, dilation, groups), false);
    }

    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv3d(image, kernel.detach(), stride, padding, dilation, groups, original_x));
    result.track_tensors(kernel);
    result.create_backward_function([image_shape = std::move(image_shape), kernel_shape = std::move(kernel_shape), 
                                    stride = std::move(stride), padding = std::move(padding), 
                                    dilation = std::move(dilation), groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img){
        ::nt::functional::conv_dkernel(grad, img->tensor, parents[0]->grad(), {image_shape[-3], image_shape[-2], image_shape[-3]}, groups, 3);
    }, original_x);
    return std::move(result);

}

TensorGrad  TensorGrad_Functional_Class::conv3d(const TensorGrad& image, const Tensor& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> dilation, int64_t groups){
    if( image.track_grad() == false ){
        return TensorGrad(::nt::functional::conv3d(image.detach(), kernel, stride, padding, dilation, groups), false);
    }

    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv3d(image.detach(), kernel, stride, padding, dilation, groups, nullptr, original_w));
    result.track_tensors(image);
    result.create_backward_function([image_shape = std::move(image_shape), kernel_shape = std::move(kernel_shape), 
                                    stride = std::move(stride), padding = std::move(padding), 
                                    dilation = std::move(dilation), groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> kern){
        ::nt::functional::conv_dimage(grad, kern->tensor, parents[0]->grad(), 
                                     {kernel_shape[-3], kernel_shape[-2], kernel_shape[-1]},
                                     {stride[0], stride[1], stride[2]},
                                     {padding[0], padding[1], padding[2]},
                                     {dilation[0], dilation[1], dilation[2]},
                                     groups, 3);
    }, original_w);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv3d(const TensorGrad& image, const TensorGrad& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> dilation, int64_t groups){
	if(image.track_grad() == false ){
        return conv3d(image.detach(), kernel, stride, padding, dilation, groups);
    }
    if(kernel.track_grad() == false){
        return conv3d(image, kernel.detach(), stride, padding, dilation, groups);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv3d(image.detach(), kernel.detach(), stride, padding, dilation, groups, original_x, original_w));
    result.track_tensors(image, kernel);
    result.create_backward_function([image_shape = std::move(image_shape), kernel_shape = std::move(kernel_shape), 
                                    stride = std::move(stride), padding = std::move(padding), 
                                    dilation = std::move(dilation), groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img, intrusive_ptr<tensor_holder> kern){
        ::nt::functional::conv_dimage(grad, kern->tensor, parents[0]->grad(), 
                                     {kernel_shape[-3], kernel_shape[-2], kernel_shape[-1]},
                                     {stride[0], stride[1], stride[2]},
                                     {padding[0], padding[1], padding[2]},
                                     {dilation[0], dilation[1], dilation[2]},
                                     groups, 3);
        ::nt::functional::conv_dkernel(grad, img->tensor, parents[1]->grad(), {image_shape[-3], image_shape[-2], image_shape[-3]}, groups, 3);
    }, original_x, original_w);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::convnd(const Tensor& image, const TensorGrad& kernel, int64_t dim, utils::optional_list stride, utils::optional_list padding, utils::optional_list dilation, int64_t groups){
    if(kernel.track_grad() == false ){
        return TensorGrad(::nt::functional::convnd(image, kernel.detach(), dim, stride, padding, dilation, groups), false);
    }

    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::convnd(image, kernel.detach(), dim, stride, padding, dilation, groups, original_x));
    result.track_tensors(kernel);
    result.create_backward_function([image_shape = std::move(image_shape), kernel_shape = std::move(kernel_shape), 
                                    stride = std::move(stride), padding = std::move(padding), 
                                    dilation = std::move(dilation), groups, dim](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img){
        ::nt::functional::conv_dkernel(grad, img->tensor, parents[0]->grad(), image_shape[range_((-1 * (dim+1)), -1)].Vec(), groups, dim);
    }, original_x);
    return std::move(result);

}

TensorGrad  TensorGrad_Functional_Class::convnd(const TensorGrad& image, const Tensor& kernel, int64_t dim, utils::optional_list stride, utils::optional_list padding, utils::optional_list dilation, int64_t groups){
    if(image.track_grad() == false ){
        return TensorGrad(::nt::functional::convnd(image.detach(), kernel, dim, stride, padding, dilation, groups), false);
    }

    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::convnd(image.detach(), kernel, dim, stride, padding, dilation, groups, nullptr, original_w));
    result.track_tensors(image);
    result.create_backward_function([image_shape = std::move(image_shape), kernel_shape = std::move(kernel_shape), 
                                    stride = std::move(stride), padding = std::move(padding), 
                                    dilation = std::move(dilation), groups, dim](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> kern){

        ::nt::functional::conv_dimage(grad, kern->tensor, parents[0]->grad(), 
                                     kernel_shape[range_((-1 * (dim+1)), -1)].Vec(),
                                     stride.to_repeat_vector(dim), padding.to_repeat_vector(dim), dilation.to_repeat_vector(dim),
                                     groups, dim);
    }, original_w);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::convnd(const TensorGrad& image, const TensorGrad& kernel, int64_t dim, utils::optional_list stride, utils::optional_list padding, utils::optional_list dilation, int64_t groups){
	if( image.track_grad() == false ){
        return convnd(image.detach(), kernel, dim, stride, padding, dilation, groups);
    }
    if( kernel.track_grad() == false ){
        return convnd(image, kernel.detach(), dim, stride, padding, dilation, groups);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::convnd(image.detach(), kernel.detach(), dim, stride, padding, dilation, groups, original_x, original_w));
    result.track_tensors(image, kernel);
    result.create_backward_function([image_shape = std::move(image_shape), kernel_shape = std::move(kernel_shape), 
                                    stride = std::move(stride), padding = std::move(padding), 
                                    dilation = std::move(dilation), groups, dim]
                        (const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img, intrusive_ptr<tensor_holder> kern){
        ::nt::functional::conv_dimage(grad, kern->tensor, parents[0]->grad(), 
                                     kernel_shape[range_((-1 * (dim+1)), -1)].Vec(),
                                     stride.to_repeat_vector(dim), padding.to_repeat_vector(dim), dilation.to_repeat_vector(dim),
                                     groups, dim);
        ::nt::functional::conv_dkernel(grad, img->tensor, parents[1]->grad(), image_shape[range_((-1 * (dim+1)), -1)].Vec(), groups, dim);
    }, original_x, original_w);
    return std::move(result);
}


inline std::vector<int64_t> fix_conv_input_shape(std::vector<int64_t> shape, int64_t dim){
    if(shape.size() == dim+2) return std::move(shape);
    if(shape.size() == dim+1) shape.insert(shape.begin(), 1);
    return std::move(shape);
}

TensorGrad  TensorGrad_Functional_Class::conv_transpose1d(const Tensor& image, const TensorGrad& kernel, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups){
    if( kernel.track_grad() == false ){
        return TensorGrad(::nt::functional::conv_transpose1d(image, kernel.detach(), stride, padding, output_padding, dilation, groups), false);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    utils::throw_exception(kernel_shape.size() == 3 || kernel_shape.size() == 2, "Error for conv1d kernel shape must be 2 or 3 dims but got $", kernel_shape);
    utils::throw_exception(image_shape.size() == 3 || image_shape.size() == 2, "Error for conv1d kernel shape must be 2 or 3 dims but got $", image_shape);
    TensorGrad result(::nt::functional::conv_transpose1d(image, kernel.detach(), stride, padding, output_padding, dilation, groups, original_x));
    result.track_tensors(kernel);
    result.create_backward_function(
        [image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img){
        ::nt::functional::convt_dkernel(grad, img->tensor, parents[0]->grad(), {padding}, {image_shape[-1]}, groups, 1);
    }, original_x);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv_transpose1d(const TensorGrad& image, const Tensor& kernel, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups){
	if( image.track_grad() == false ){
        TensorGrad result(::nt::functional::conv_transpose1d(image.detach(), kernel, stride, padding, output_padding, dilation, groups), false);
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    utils::throw_exception(kernel_shape.size() == 3 || kernel_shape.size() == 2, "Error for conv1d kernel shape must be 2 or 3 dims but got $", kernel_shape);
    utils::throw_exception(image_shape.size() == 3 || image_shape.size() == 2, "Error for conv1d kernel shape must be 2 or 3 dims but got $", image_shape);
    TensorGrad result(::nt::functional::conv_transpose1d(image.detach(), kernel, stride, padding, output_padding, dilation, groups, nullptr, original_w));
    result.track_tensors(image);
    result.create_backward_function(
        [image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> kern){
        ::nt::functional::convt_dimage(grad, kern->tensor, parents[0]->grad(),
                                     fix_conv_input_shape(kernel_shape.Vec(), 1),
                                     {stride},
                                     {padding},
                                     {output_padding},
                                     {dilation},
                                     groups, 1);
    }, original_w);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv_transpose1d(const TensorGrad& image, const TensorGrad& kernel, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups){
	if( image.track_grad() == false ){
        return conv_transpose1d(image.detach(), kernel, stride, padding, output_padding, dilation, groups);
    }
    if( kernel.track_grad() == false ){
        return conv_transpose1d(image, kernel.detach(), stride, padding, output_padding, dilation, groups);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    utils::throw_exception(kernel_shape.size() == 3 || kernel_shape.size() == 2, "Error for conv1d kernel shape must be 2 or 3 dims but got $", kernel_shape);
    utils::throw_exception(image_shape.size() == 3 || image_shape.size() == 2, "Error for conv1d kernel shape must be 2 or 3 dims but got $", image_shape);
    TensorGrad result(::nt::functional::conv_transpose1d(image.detach(), kernel.detach(), stride, padding, output_padding, dilation, groups, original_x, original_w));
    result.track_tensors(image, kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img, intrusive_ptr<tensor_holder> kern){
        ::nt::functional::convt_dimage(grad, kern->tensor, parents[0]->grad(), 
                                     fix_conv_input_shape(kernel_shape.Vec(), 1),
                                     {stride},
                                     {padding},
                                     {output_padding},
                                     {dilation},
                                     groups, 1);
        ::nt::functional::convt_dkernel(grad, img->tensor, parents[1]->grad(), {padding}, {image_shape[-1]}, groups, 1);
    }, original_x, original_w);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv_transpose2d(const Tensor& image, const TensorGrad& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple output_padding, utils::my_tuple dilation, int64_t groups){
	if( kernel.track_grad() == false){
        //if the kernel isn't tracking the gradient, then the gradient for neither is tracked
        TensorGrad result(::nt::functional::conv_transpose2d(image, kernel.detach(), stride, padding, output_padding, dilation, groups), false);
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    utils::throw_exception(kernel_shape.size() == 4 || kernel_shape.size() == 3, "Error for conv2d kernel shape must be 4 or 3 dims but got $", kernel_shape);
    utils::throw_exception(image_shape.size() == 4 || image_shape.size() == 3, "Error for conv2d kernel shape must be 4 or 3 dims but got $", image_shape);
    TensorGrad result(::nt::functional::conv_transpose2d(image, kernel.detach(), stride, padding, output_padding, dilation, groups, original_x));
    result.track_tensors( kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img){
          ::nt::functional::convt_dkernel(grad, img->tensor, parents[0]->grad(), {padding[0], padding[1]}, {image_shape[-2], image_shape[-3]}, groups, 2);
    }, original_x);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv_transpose2d(const TensorGrad& image, const Tensor& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple output_padding, utils::my_tuple dilation, int64_t groups){
	if( image.track_grad() == false){
        //if one of the tensors isn't tracking the gradient, then the gradient for neither is tracked
        TensorGrad result(::nt::functional::conv_transpose2d(image.detach(), kernel, stride, padding, output_padding, dilation, groups), false);
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    utils::throw_exception(kernel_shape.size() == 4 || kernel_shape.size() == 3, "Error for conv2d kernel shape must be 4 or 3 dims but got $", kernel_shape);
    utils::throw_exception(image_shape.size() == 4 || image_shape.size() == 3, "Error for conv2d kernel shape must be 4 or 3 dims but got $", image_shape);
    TensorGrad result(::nt::functional::conv_transpose2d(image.detach(), kernel, stride, padding, output_padding, dilation, groups, nullptr, original_w));
    result.track_tensors(image);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> kern){
        ::nt::functional::convt_dimage(grad, kern->tensor, parents[0]->grad(), 
                                     fix_conv_input_shape(kernel_shape.Vec(), 2),
                                     {stride[0], stride[1]},
                                     {padding[0], padding[1]},
                                     {output_padding[0], output_padding[1]},
                                     {dilation[0], dilation[1]},
                                     groups, 2);
    }, original_w);
    return std::move(result);
}


TensorGrad  TensorGrad_Functional_Class::conv_transpose2d(const TensorGrad& image, const TensorGrad& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple output_padding, utils::my_tuple dilation, int64_t groups){
	if( image.track_grad() == false ){
        return conv_transpose2d(image.detach(), kernel, stride, padding, output_padding, dilation, groups);
    }
    if( kernel.track_grad() == false ){
        return conv_transpose2d(image, kernel.detach(), stride, padding, output_padding, dilation, groups);
    }

    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    utils::throw_exception(kernel_shape.size() == 4 || kernel_shape.size() == 3, "Error for conv2d kernel shape must be 4 or 3 dims but got $", kernel_shape);
    utils::throw_exception(image_shape.size() == 4 || image_shape.size() == 3, "Error for conv2d kernel shape must be 4 or 3 dims but got $", image_shape);
    TensorGrad result(::nt::functional::conv_transpose2d(image.detach(), kernel.detach(), stride, padding, output_padding, dilation, groups, original_x, original_w));
    result.track_tensors(image, kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img, intrusive_ptr<tensor_holder> kern){
        ::nt::functional::convt_dimage(grad, kern->tensor, parents[0]->grad(), 
                                     fix_conv_input_shape(kernel_shape.Vec(), 2),
                                     {stride[0], stride[1]},
                                     {padding[0], padding[1]},
                                     {output_padding[0], output_padding[1]},
                                     {dilation[0], dilation[1]},
                                     groups, 2);
        ::nt::functional::convt_dkernel(grad, img->tensor, parents[1]->grad(), {padding[0], padding[1]}, {image_shape[-2], image_shape[-3]}, groups, 2);
    }, original_x, original_w);
    return std::move(result);
}



TensorGrad  TensorGrad_Functional_Class::conv_transpose3d(const Tensor& image, const TensorGrad& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> output_padding, utils::my_n_tuple<3> dilation, int64_t groups){
    if( kernel.track_grad() == false ){
        return TensorGrad(::nt::functional::conv_transpose3d(image, kernel.detach(), stride, padding, output_padding, dilation, groups), false);
    }

    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    utils::throw_exception(kernel_shape.size() == 4 || kernel_shape.size() == 5, "Error for conv3d kernel shape must be 4 or 5 dims but got $", kernel_shape);
    utils::throw_exception(image_shape.size() == 4 || image_shape.size() == 5, "Error for conv3d kernel shape must be 4 or 5 dims but got $", image_shape);
    TensorGrad result(::nt::functional::conv_transpose3d(image, kernel.detach(), stride, padding, output_padding, dilation, groups, original_x));
    result.track_tensors(kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img){
        ::nt::functional::convt_dkernel(grad, img->tensor, parents[0]->grad(), {padding[0], padding[1], padding[2]}, {image_shape[-3], image_shape[-2], image_shape[-3]}, groups, 3);
    }, original_x);
    return std::move(result);

}

TensorGrad  TensorGrad_Functional_Class::conv_transpose3d(const TensorGrad& image, const Tensor& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> output_padding, utils::my_n_tuple<3> dilation, int64_t groups){
    if( image.track_grad() == false ){
        return TensorGrad(::nt::functional::conv_transpose3d(image.detach(), kernel, stride, padding, output_padding, dilation, groups), false);
    }

    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    utils::throw_exception(kernel_shape.size() == 4 || kernel_shape.size() == 5, "Error for conv3d kernel shape must be 4 or 5 dims but got $", kernel_shape);
    utils::throw_exception(image_shape.size() == 4 || image_shape.size() == 5, "Error for conv3d kernel shape must be 4 or 5 dims but got $", image_shape);
    TensorGrad result(::nt::functional::conv_transpose3d(image.detach(), kernel, stride, padding, output_padding, dilation, groups, nullptr, original_w));
    result.track_tensors(image);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> kern){
        ::nt::functional::convt_dimage(grad, kern->tensor, parents[0]->grad(), 
                                     fix_conv_input_shape(kernel_shape.Vec(), 3),
                                     {stride[0], stride[1], stride[2]},
                                     {padding[0], padding[1], padding[2]},
                                     {output_padding[0], output_padding[1], output_padding[2]},
                                     {dilation[0], dilation[1], dilation[2]},
                                     groups, 3);
    }, original_w);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv_transpose3d(const TensorGrad& image, const TensorGrad& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> output_padding, utils::my_n_tuple<3> dilation, int64_t groups){
	if(image.track_grad() == false ){
        return conv_transpose3d(image.detach(), kernel, stride, padding, output_padding, dilation, groups);
    }
    if(kernel.track_grad() == false ){
        return conv_transpose3d(image, kernel.detach(), stride, padding, output_padding, dilation, groups);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    utils::throw_exception(kernel_shape.size() == 4 || kernel_shape.size() == 5, "Error for conv3d transpose kernel shape must be 4 or 5 dims but got $", kernel_shape);
    utils::throw_exception(image_shape.size() == 4 || image_shape.size() == 5, "Error for conv3d transpose kernel shape must be 4 or 5 dims but got $", image_shape);
    TensorGrad result(::nt::functional::conv_transpose3d(image.detach(), kernel.detach(), stride, padding, output_padding, dilation, groups, original_x, original_w));
    result.track_tensors(image, kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img, intrusive_ptr<tensor_holder> kern){
        ::nt::functional::convt_dimage(grad, kern->tensor, parents[0]->grad(), 
                                     fix_conv_input_shape(kernel_shape.Vec(), 3),
                                     {stride[0], stride[1], stride[2]},
                                     {padding[0], padding[1], padding[2]},
                                     {output_padding[0], output_padding[1], output_padding[2]},
                                     {dilation[0], dilation[1], dilation[2]},
                                     groups, 3);
        ::nt::functional::convt_dkernel(grad, img->tensor, parents[1]->grad(), {padding[0], padding[1], padding[2]}, {image_shape[-3], image_shape[-2], image_shape[-3]}, groups, 3);
    }, original_x, original_w);
    return std::move(result);
}


TensorGrad  TensorGrad_Functional_Class::conv_transposend(const Tensor& image, const TensorGrad& kernel, int64_t dim,
                                                          utils::optional_list stride, utils::optional_list padding, 
                                                          utils::optional_list output_padding, utils::optional_list dilation, 
                                                          int64_t groups){
    if(kernel.track_grad() == false ){
        return TensorGrad(::nt::functional::conv_transposend(image, kernel.detach(), dim, stride, padding, output_padding, dilation, groups), false);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    utils::throw_exception(kernel_shape.size() == dim+1 || kernel_shape.size() == dim+2, 
                           "Error for conv$d transpose kernel shape must be $ or $ dims but got $", 
                           dim, dim+1, dim+2, kernel_shape);
    utils::throw_exception(image_shape.size() == dim+1 || image_shape.size() == dim+2, 
                           "Error for conv$d transpose kernel shape must be $ or $ dims but got $", 
                           dim, dim+1, dim+2, image_shape);
    TensorGrad result(::nt::functional::conv_transposend(image, kernel.detach(), dim, stride, padding, output_padding, dilation, groups, original_x, nullptr));
    result.track_tensors(kernel);
    result.create_backward_function([image_shape = std::move(image_shape), kernel_shape = std::move(kernel_shape), 
                                    stride = std::move(stride), padding = std::move(padding), output_padding = std::move(output_padding), 
                                    dilation = std::move(dilation), groups, dim](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img){
        ::nt::functional::convt_dkernel(grad, img->tensor, parents[1]->grad(), 
                                        padding.to_repeat_vector(dim), image_shape[(-1 * (dim+1)) <range> -1].Vec(),
                                        groups, dim);
    }, original_x);
    return std::move(result);
}


TensorGrad  TensorGrad_Functional_Class::conv_transposend(const TensorGrad& image, const Tensor& kernel, int64_t dim,
                                                          utils::optional_list stride, utils::optional_list padding, 
                                                          utils::optional_list output_padding, utils::optional_list dilation, 
                                                          int64_t groups){
	if(image.track_grad() == false ){
        return TensorGrad(::nt::functional::conv_transposend(image.detach(), kernel, dim, stride, padding, output_padding, dilation, groups), false);
    }
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    utils::throw_exception(kernel_shape.size() == dim+1 || kernel_shape.size() == dim+2, 
                           "Error for conv$d transpose kernel shape must be $ or $ dims but got $", 
                           dim, dim+1, dim+2, kernel_shape);
    utils::throw_exception(image_shape.size() == dim+1 || image_shape.size() == dim+2, 
                           "Error for conv$d transpose kernel shape must be $ or $ dims but got $", 
                           dim, dim+1, dim+2, image_shape);
    TensorGrad result(::nt::functional::conv_transposend(image.detach(), kernel, dim, stride, padding, output_padding, dilation, groups, nullptr, original_w));
    result.track_tensors(image);
    result.create_backward_function([image_shape = std::move(image_shape), kernel_shape = std::move(kernel_shape), 
                                    stride = std::move(stride), padding = std::move(padding), output_padding = std::move(output_padding), 
                                    dilation = std::move(dilation), groups, dim](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> kern){
        ::nt::functional::convt_dimage(grad, kern->tensor, parents[0]->grad(), 
                                     fix_conv_input_shape(kernel_shape.Vec(), dim),
                                     stride.to_repeat_vector(dim), padding.to_repeat_vector(dim),
                                     output_padding.to_repeat_vector(dim), dilation.to_repeat_vector(dim),
                                     groups, dim);
    }, original_w);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv_transposend(const TensorGrad& image, const TensorGrad& kernel, int64_t dim,
                                                          utils::optional_list stride, utils::optional_list padding, 
                                                          utils::optional_list output_padding, utils::optional_list dilation, 
                                                          int64_t groups){
	if(image.track_grad() == false ){
        return conv_transposend(image.detach(), kernel, dim, stride, padding, output_padding, dilation, groups);
    }
    if(kernel.track_grad() == false ){
        return conv_transposend(image, kernel.detach(), dim, stride, padding, output_padding, dilation, groups);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    utils::throw_exception(kernel_shape.size() == dim+1 || kernel_shape.size() == dim+2, 
                           "Error for conv$d transpose kernel shape must be $ or $ dims but got $", 
                           dim, dim+1, dim+2, kernel_shape);
    utils::throw_exception(image_shape.size() == dim+1 || image_shape.size() == dim+2, 
                           "Error for conv$d transpose kernel shape must be $ or $ dims but got $", 
                           dim, dim+1, dim+2, image_shape);
    TensorGrad result(::nt::functional::conv_transposend(image.detach(), kernel.detach(), dim, stride, padding, output_padding, dilation, groups, original_x, original_w));
    result.track_tensors(image, kernel);
    result.create_backward_function([image_shape = std::move(image_shape), kernel_shape = std::move(kernel_shape), 
                                    stride = std::move(stride), padding = std::move(padding), output_padding = std::move(output_padding), 
                                    dilation = std::move(dilation), groups, dim](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img, intrusive_ptr<tensor_holder> kern){
        ::nt::functional::convt_dimage(grad, kern->tensor, parents[0]->grad(), 
                                     fix_conv_input_shape(kernel_shape.Vec(), dim),
                                     stride.to_repeat_vector(dim), padding.to_repeat_vector(dim),
                                     output_padding.to_repeat_vector(dim), dilation.to_repeat_vector(dim),
                                     groups, dim);
        ::nt::functional::convt_dkernel(grad, img->tensor, parents[1]->grad(), 
                                        padding.to_repeat_vector(dim), image_shape[(-1 * (dim+1)) <range> -1].Vec(),
                                        groups, dim);
    }, original_x, original_w);
    return std::move(result);
}

}
}
