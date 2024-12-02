#include <oneapi/dnnl/dnnl.hpp>
#include "../Tensor.h"
#include "../types/bfloat16.h"


namespace nt{
namespace functional{
namespace functional_dnn{


void conv_2d_cpu_f32(float* Img, float* Kernel, 
                      const int64_t& kernel_batches, const int64_t& kernel_channels, const int64_t& kernel_rows, const int64_t& kernel_cols,
                      const int64_t& img_batches, const int64_t& img_channels, const int64_t& img_rows, const int64_t& img_cols,
                      const int64_t& dilation_rows, const int64_t& dilation_cols,
                      const int64_t& stride_rows, const int64_t& stride_cols,
                      const int64_t& pad_rows, const int64_t& pad_cols,
                      const int64_t& groups,
                      float* dst) {

    using namespace dnnl;

    // Create a oneDNN engine and stream
    /* std::cout << "made engine"<<std::endl; */
    engine eng(engine::kind::cpu, 0);
    stream s(eng);

    // Create memory descriptors for the source, weights, and destination
    memory::dims src_dims = {img_batches, img_channels, img_rows, img_cols};
    memory::dims weights_dims = {kernel_batches, kernel_channels / groups, kernel_rows, kernel_cols};
    memory::dims dst_dims = {img_batches, kernel_batches, 
                             (img_rows + 2 * pad_rows - dilation_rows * (kernel_rows - 1) - 1) / stride_rows + 1, 
                             (img_cols + 2 * pad_cols - dilation_cols * (kernel_cols - 1) - 1) / stride_cols + 1};
    memory::dims strides = {stride_rows, stride_cols};
    memory::dims padding = {pad_rows, pad_cols};
    memory::dims dilation = {dilation_rows - 1, dilation_cols - 1};

    // Create memory objects
    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::oihw);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32,memory::format_tag::nchw);



    auto dst_mem = memory(dst_md, eng, dst);
    auto weights_mem = memory(weights_md, eng, (void*)Kernel);
    auto src_mem = memory(src_md, eng, (void*)Img);


    // Create convolution primitive descriptor
    auto conv_pd = convolution_forward::primitive_desc(eng, prop_kind::forward_inference, algorithm::convolution_direct,
			src_md, weights_md, dst_md,
			 strides, dilation, padding, padding);

    // Reorder data if necessary
    src_mem = memory(conv_pd.src_desc(), eng, (void*)Img);
    weights_mem = memory(conv_pd.weights_desc(), eng, (void*)Kernel);
    dst_mem = memory(conv_pd.dst_desc(), eng, dst);

    // Create convolution primitive
    auto conv_prim = convolution_forward(conv_pd);

    // Execute convolution
    conv_prim.execute(s, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_DST, dst_mem}
    });

    // Wait for the computation to finish
    s.wait();
}



#ifdef _HALF_FLOAT_SUPPORT_
void conv_2d_cpu_f16(bfloat16_t* Img, bfloat16_t* Kernel, 
                      const int64_t& kernel_batches, const int64_t& kernel_channels, const int64_t& kernel_rows, const int64_t& kernel_cols,
                      const int64_t& img_batches, const int64_t& img_channels, const int64_t& img_rows, const int64_t& img_cols,
                      const int64_t& dilation_rows, const int64_t& dilation_cols,
                      const int64_t& stride_rows, const int64_t& stride_cols,
                      const int64_t& pad_rows, const int64_t& pad_cols,
                      const int64_t& groups,
                      bfloat16_t* dst) {

    using namespace dnnl;

    // Create a oneDNN engine and stream
    engine eng(engine::kind::cpu, 0);
    stream s(eng);

    // Create memory descriptors for the source, weights, and destination
    memory::dims src_dims = {img_batches, img_channels, img_rows, img_cols};
    memory::dims weights_dims = {kernel_batches, kernel_channels / groups, kernel_rows, kernel_cols};
    memory::dims dst_dims = {img_batches, kernel_batches, 
                             (img_rows + 2 * pad_rows - dilation_rows * (kernel_rows - 1) - 1) / stride_rows + 1, 
                             (img_cols + 2 * pad_cols - dilation_cols * (kernel_cols - 1) - 1) / stride_cols + 1};
    memory::dims strides = {stride_rows, stride_cols};
    memory::dims padding = {pad_rows, pad_cols};
    memory::dims dilation = {dilation_rows - 1, dilation_cols - 1};

    // Create memory objects
    auto src_md = memory::desc(src_dims, memory::data_type::f16, memory::format_tag::nchw);
    auto weights_md = memory::desc(weights_dims, memory::data_type::f16, memory::format_tag::oihw);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f16, memory::format_tag::nchw);

    auto src_mem = memory(src_md, eng, (void*)Img);
    auto weights_mem = memory(weights_md, eng, (void*)Kernel);
    auto dst_mem = memory(dst_md, eng, dst);

    // Create convolution descriptor
    auto conv_pd = convolution_forward::primitive_desc(eng, prop_kind::forward_inference, algorithm::convolution_direct,
			src_md, weights_md, dst_md,
			 strides, dilation, padding, padding);

    // Reorder data if necessary
    src_mem = memory(conv_pd.src_desc(), eng, (void*)Img);
    weights_mem = memory(conv_pd.weights_desc(), eng, (void*)Kernel);
    dst_mem = memory(conv_pd.dst_desc(), eng, dst);

    // Create convolution primitive
    auto conv_prim = convolution_forward(conv_pd);

    // Execute convolution
    conv_prim.execute(s, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_DST, dst_mem}
    });

    // Wait for the computation to finish
    s.wait();
}


#endif //_HALF_FLOAT_SUPPORT_


Tensor conv2d(const Tensor& image, const Tensor& kernel, utils::my_tuple& stride, utils::my_tuple& padding, utils::my_tuple& dilation, int64_t& groups){
	utils::throw_exception(kernel.dims() == 3 || kernel.dims() == 4, "Expected the input kernel to convolution to be 3D or 4D but is $D", kernel.dims());
	utils::throw_exception(image.dims() == 3 || image.dims() == 4, "Expected input image to convolution to be 3D or 4D but got $D", image.dims());
	utils::throw_exception(image.dtype == kernel.dtype, "Expected to convolute input and kernel of same dtype but got $ and $", image.dtype, kernel.dtype);
#ifdef _HALF_FLOAT_SUPPORT_
	utils::throw_exception(image.dtype == DType::Float32 || image.dtype == DType::Float16, "Expected for dnn for dtype to be float16 or float32 but got unsupported type $", image.dtype);
#else
	utils::throw_exception(image.dtype == DType::Float32, "Expected for dnn for dtype to be float32 but got unsupported type $", image.dtype);
#endif //_HALF_FLOAT_SUPPORT

	Tensor x = (image.dims() == 3) ? image.unsqueeze(0).contiguous() : image.contiguous();
	/* x = x.pad({padding[0], padding[0], padding[1], padding[1]}); */
	Tensor w = kernel.dims() == 3 ? kernel.unsqueeze(0).contiguous() : kernel.contiguous();

	int64_t Rout = ((image.shape()[-2] + 2 * padding[0] - dilation[0] * (w.shape()[-2] - 1) - 1) / stride[0]) + 1;
	int64_t Cout = ((image.shape()[-1] + 2 * padding[1] - dilation[1] * (w.shape()[-1] - 1) - 1) / stride[1]) + 1;
	int64_t CHout = w.shape()[0];
	int64_t Bout = x.shape()[0];
	Tensor dst = ::nt::functional::zeros({Bout, CHout, Rout, Cout}, image.dtype);
	if(image.dtype == DType::Float32){
		conv_2d_cpu_f32(reinterpret_cast<float*>(x.data_ptr()), reinterpret_cast<float*>(w.data_ptr()),
			w.shape()[0], w.shape()[1], w.shape()[2], w.shape()[3],
			x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3],
			dilation[0], dilation[1],
			stride[0], stride[1],
			padding[0], padding[1],
			groups, reinterpret_cast<float*>(dst.data_ptr()));
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if(image.dtype == DType::Float16){
		conv_2d_cpu_f16(
			convert::inplace_to_bfloat16_from_float16(reinterpret_cast<float16_t*>(x.data_ptr()), x.numel()),
			convert::inplace_to_bfloat16_from_float16(reinterpret_cast<float16_t*>(w.data_ptr()), w.numel()),
			w.shape()[0], w.shape()[1], w.shape()[2], w.shape()[3],
			x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3],
			dilation[0], dilation[1],
			stride[0], stride[1],
			padding[0], padding[1],
			groups, reinterpret_cast<bfloat16_t*>(dst.data_ptr()));
		//the inplace methods save memory
		if(image.is_contiguous())
			convert::inplace_to_float16_from_bfloat16(reinterpret_cast<bfloat16_t*>(x.data_ptr()), x.numel());
		if(kernel.is_contiguous())
			convert::inplace_to_float16_from_bfloat16(reinterpret_cast<bfloat16_t*>(w.data_ptr()), w.numel());
		convert::inplace_to_float16_from_bfloat16(reinterpret_cast<bfloat16_t*>(dst.data_ptr()), dst.numel());
	}
#endif //_HALF_FLOAT_SUPPORT
	return std::move(dst);
}


}
}
}
