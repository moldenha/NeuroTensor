#include <nt/Tensor.h>
#include <nt/nn/layers.h>
#include <nt/nn/Layer.h>
#include <nt/functional/functional.h>

void max_pool(int64_t n){
    int64_t kernel_size = 3;
    int64_t input_size = 20;
    nt::Layer pool(
        n == 1 ? nt::Layer(nt::layers::MaxPool1D(kernel_size, ntarg_(padding) = 1, ntarg_(return_indices) = true))
        : n == 2 ? nt::Layer(nt::layers::MaxPool2D(kernel_size, ntarg_(padding) = 1, ntarg_(return_indices) = true))
        : nt::Layer(nt::layers::MaxPool3D(kernel_size, ntarg_(padding) = 1, ntarg_(return_indices) = true)));
    
    nt::Layer unpool(
        n == 1 ? nt::Layer(nt::layers::MaxUnPool1D(kernel_size, ntarg_(padding) = 1))
        : n == 2 ? nt::Layer(nt::layers::MaxUnPool2D(kernel_size, ntarg_(padding) = 1))
        : nt::Layer(nt::layers::MaxUnPool3D(kernel_size, ntarg_(padding) = 1)));


    nt::TensorGrad input(
        n == 1 ? nt::functional::randn({2, input_size}, nt::DType::Float32)
        : n == 2 ? nt::functional::randn({2, input_size, input_size}, nt::DType::Float32)
        : nt::functional::randn({2, input_size, input_size, input_size}, nt::DType::Float32));

    
    auto [output, indices] = nt::get<2>(pool(input));
    std::cout << "output: "<<output << std::endl;
    std::cout << "indices: "<<indices << std::endl;
    nt::TensorGrad unpooled = unpool(output, indices);
    std::cout << "unpooled: "<<unpooled<<std::endl;
    output.backward(output.detach()-1);
    unpooled.backward(unpooled.detach()-1);
}

void lp_pool(int64_t n){
    int64_t kernel_size = 3;
    nt::Scalar power = 2;
    int64_t input_size = 20;
    nt::Layer pool(
        n == 1 ? nt::Layer(nt::layers::LPPool1D(power, kernel_size))
        : n == 2 ? nt::Layer(nt::layers::LPPool2D(power, kernel_size))
        : nt::Layer(nt::layers::LPPool3D(power, kernel_size)));
    
    
    nt::TensorGrad input(
        n == 1 ? nt::functional::randn({2, input_size}, nt::DType::Float32)
        : n == 2 ? nt::functional::randn({2, input_size, input_size}, nt::DType::Float32)
        : nt::functional::randn({2, input_size, input_size, input_size}, nt::DType::Float32));

    
    nt::TensorGrad output = pool(input);
    std::cout << "output: "<<output << std::endl;
    output.backward(output.detach()-1);
}

void avg_pool(int64_t n){
    int64_t kernel_size = 3;
    nt::Scalar power = 2;
    int64_t input_size = 20;


    nt::Layer pool(
        n == 1 ? nt::Layer(nt::layers::AvgPool1D(kernel_size, ntarg_(ceil_mode) = true))
        : n == 2 ? nt::Layer(nt::layers::AvgPool2D(kernel_size, ntarg_(ceil_mode) = true))
        : nt::Layer(nt::layers::AvgPool3D(kernel_size, ntarg_(ceil_mode) = true)));
    
    nt::TensorGrad input(
        n == 1 ? nt::functional::randn({2, input_size}, nt::DType::Float32)
        : n == 2 ? nt::functional::randn({2, input_size, input_size}, nt::DType::Float32)
        : nt::functional::randn({2, input_size, input_size, input_size}, nt::DType::Float32));

    
    nt::TensorGrad output = pool(input);
    std::cout << "output: "<<output << std::endl;
    output.backward(output.detach()-1);

}


void fractional_max_pool(int64_t n){
    int64_t kernel_size = 3;
    nt::Scalar power = 2;
    int64_t input_size = 20;


    nt::Layer pool(
        n == 2 ? nt::Layer(nt::layers::FractionalMaxPool2D(kernel_size, ntarg_(output_ratio) = 0.3))
        : nt::Layer(nt::layers::FractionalMaxPool3D(kernel_size, ntarg_(output_size) = 8)));

    nt::TensorGrad input(
        n == 1 ? nt::functional::randn({2, input_size}, nt::DType::Float32)
        : n == 2 ? nt::functional::randn({2, input_size, input_size}, nt::DType::Float32)
        : nt::functional::randn({2, input_size, input_size, input_size}, nt::DType::Float32));

    nt::TensorGrad output = pool(input);
    std::cout << "output: "<<output << std::endl;
    output.backward(output.detach()-1);

}

