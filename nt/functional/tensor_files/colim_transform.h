#ifndef NT_FUNCTIONAL_COLIM_TRANSFORM_H__
#define NT_FUNCTIONAL_COLIM_TRANSFORM_H__

#include "../../Tensor.h"
#include "../../utils/utils.h"


namespace nt{
namespace functional{



NEUROTENSOR_API Tensor fold1d(const Tensor&, Tensor::size_value_t output_size, 
                            Tensor::size_value_t kernel_size, Tensor::size_value_t dilation=1, 
                            Tensor::size_value_t padding = 0, Tensor::size_value_t stride = 1);
NEUROTENSOR_API Tensor fold1d_backward(const Tensor& grad_output, const Tensor::size_value_t& output_size, 
                                     const Tensor::size_value_t& kernel_size, const Tensor::size_value_t& dilation, 
                                     const Tensor::size_value_t& padding, const Tensor::size_value_t& stride);
NEUROTENSOR_API Tensor& fold1d_backward(const Tensor& grad_output, Tensor& output, 
                                      const Tensor::size_value_t& output_size, const Tensor::size_value_t& kernel_size, 
                                      const Tensor::size_value_t& dilation, const Tensor::size_value_t& padding, 
                                      const Tensor::size_value_t& stride);


NEUROTENSOR_API Tensor fold2d(const Tensor&, utils::my_tuple output_size, 
                            utils::my_tuple kernel_size, utils::my_tuple dilation=1, 
                            utils::my_tuple padding = 0, utils::my_tuple stride = 1);
NEUROTENSOR_API Tensor fold2d_backward(const Tensor& grad_output, const utils::my_tuple& output_size, 
                                     const utils::my_tuple& kernel_size, const utils::my_tuple& dilation, 
                                     const utils::my_tuple& padding, const utils::my_tuple& stride);
NEUROTENSOR_API Tensor& fold2d_backward(const Tensor& grad_output, Tensor& output, 
                                      const utils::my_tuple& output_size, const utils::my_tuple& kernel_size, 
                                      const utils::my_tuple& dilation, const utils::my_tuple& padding, 
                                      const utils::my_tuple& stride);


NEUROTENSOR_API Tensor fold3d(const Tensor&, utils::my_n_tuple<3> output_size, 
                            utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> dilation=1, 
                            utils::my_n_tuple<3> padding = 0, utils::my_n_tuple<3> stride = 1);
NEUROTENSOR_API Tensor fold3d_backward(const Tensor& grad_output, const utils::my_n_tuple<3>& output_size, 
                                     const utils::my_n_tuple<3>& kernel_size, const utils::my_n_tuple<3>& dilation, 
                                     const utils::my_n_tuple<3>& padding, const utils::my_n_tuple<3>& stride);
NEUROTENSOR_API Tensor& fold3d_backward(const Tensor& grad_output, Tensor& output, 
                                      const utils::my_n_tuple<3>& output_size, const utils::my_n_tuple<3>& kernel_size, 
                                      const utils::my_n_tuple<3>& dilation, const utils::my_n_tuple<3>& padding, 
                                      const utils::my_n_tuple<3>& stride);

NEUROTENSOR_API Tensor foldnd(const Tensor&, int64_t dim, utils::optional_list output_size, 
                            utils::optional_list kernel_size, utils::optional_list dilation=1, 
                            utils::optional_list padding = 0, utils::optional_list stride = 1,
                            bool test_mode = false);
NEUROTENSOR_API Tensor foldnd_backward(const Tensor& grad_output, int64_t dim, const utils::optional_list& output_size, 
                                     const utils::optional_list& kernel_size, const utils::optional_list& dilation, 
                                     const utils::optional_list& padding, const utils::optional_list& stride,
                                     const bool& test_mode);
NEUROTENSOR_API Tensor& foldnd_backward(const Tensor& grad_output, Tensor& output, int64_t dim, 
                                      const utils::optional_list& output_size, const utils::optional_list& kernel_size, 
                                      const utils::optional_list& dilation, const utils::optional_list& padding, 
                                      const utils::optional_list& stride, const bool& test_mode);



//1d unfold transforms:
NEUROTENSOR_API Tensor unfold1d(const Tensor& x, Tensor::size_value_t kernel_size,
                                Tensor::size_value_t dilation=1, Tensor::size_value_t padding=0, 
                                Tensor::size_value_t stride=1, bool transpose_out=true);
NEUROTENSOR_API Tensor unfold1d_backward(const Tensor& x, Tensor::size_value_t output_size, 
                                         Tensor::size_value_t kernel_size, Tensor::size_value_t dilation, 
                                         Tensor::size_value_t padding, Tensor::size_value_t stride, 
                                         bool transpose_out);
NEUROTENSOR_API Tensor& unfold1d_backward(const Tensor& x, Tensor& output, 
                                          Tensor::size_value_t output_size, Tensor::size_value_t kernel_size, 
                                          Tensor::size_value_t dilation, Tensor::size_value_t padding, 
                                          Tensor::size_value_t stride, bool transpose_out);

//2d undold transforms:
NEUROTENSOR_API Tensor unfold2d(const Tensor&, utils::my_tuple kernel_size, 
                              utils::my_tuple dilation=1, 
                              utils::my_tuple padding = 0,
                              utils::my_tuple stride = 1, bool transpose_out = true);
NEUROTENSOR_API Tensor unfold2d_backward(const Tensor& x, const utils::my_tuple& output_size, 
                                       const utils::my_tuple& kernel_size, const utils::my_tuple& dilation, 
                                       const utils::my_tuple& padding, const utils::my_tuple& stride, 
                                       const bool& transpose_out);
NEUROTENSOR_API Tensor& unfold2d_backward(const Tensor& x, Tensor& output, const utils::my_tuple& output_size, 
                                        const utils::my_tuple& kernel_size, const utils::my_tuple& dilation, 
                                        const utils::my_tuple& padding, const utils::my_tuple& stride, 
                                        const bool& transpose_out);

//3d unfold transforms:
NEUROTENSOR_API Tensor unfold3d(const Tensor& x, utils::my_n_tuple<3> kernel_size, 
                                utils::my_n_tuple<3> dilation=1, utils::my_n_tuple<3> padding=0, 
                                utils::my_n_tuple<3> stride=1, bool transpose_out=true);
NEUROTENSOR_API Tensor unfold3d_backward(const Tensor& x, const utils::my_n_tuple<3>& output_size, 
                                         const utils::my_n_tuple<3>& kernel_size, const utils::my_n_tuple<3>& dilation, 
                                         const utils::my_n_tuple<3>& padding, const utils::my_n_tuple<3>& stride, 
                                         bool transpose_out);
NEUROTENSOR_API Tensor& unfold3d_backward(const Tensor& x, Tensor& output, 
                                          const utils::my_n_tuple<3>& output_size, const utils::my_n_tuple<3>& kernel_size, 
                                          const utils::my_n_tuple<3>& dilation, const utils::my_n_tuple<3>& padding, 
                                          const utils::my_n_tuple<3>& stride, bool transpose_out);

// nd unfold transforms:
// test_mode when activated does not default to checking if the dim is 1, 2, 3
// this is so that unfoldnd can be properly checked for accuracy during testing
NEUROTENSOR_API Tensor unfoldnd(const Tensor& x, int64_t dim,
                utils::optional_list kernel_size, 
                utils::optional_list dilation = 1, 
                utils::optional_list padding = 0, 
                utils::optional_list stride = 1, bool transpose_out = true, bool test_mode = false);

Tensor unfoldnd_backward(const Tensor& x, int64_t dim,
                         const std::vector<int64_t>& output_size, 
                         utils::optional_list kernel_size, 
                         utils::optional_list dilation, 
                         utils::optional_list padding, 
                         utils::optional_list stride, const bool transpose_out, bool test_mode);


Tensor& unfoldnd_backward(const Tensor& x, Tensor& output, int64_t dim,
                         const std::vector<int64_t>& output_size, 
                         utils::optional_list kernel_size, 
                         utils::optional_list dilation, 
                         utils::optional_list padding, 
                         utils::optional_list stride, const bool transpose_out, bool test_mode);

}
}

#endif
