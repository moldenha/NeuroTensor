#include <nt/Tensor.h>
#include <nt/functional/functional.h>
#include "pooling/avg_pool1d.h"
#include "pooling/lp_pool1d.h"
#include "pooling/max_pool1d.h"
#include "pooling/avg_pool2d.h"
#include "pooling/lp_pool2d.h"
#include "pooling/max_pool2d.h"
#include "pooling/avg_pool3d.h"
#include "pooling/lp_pool3d.h"
#include "pooling/max_pool3d.h"
#include "pooling/fractional.h"

void avg_pool_1d(){
    using namespace nt;
    Tensor example = functional::arange({3, 3, 5}, DType::Float32);
    // Tensor t = example.pad({0,1});
    int64_t kernel_size = 3;
    int64_t stride = -1;
    int64_t padding = 0;
    bool ciel_mode = true;
    bool count_include_padding = false;

    Tensor avg_padded = avg_pool1d(example, kernel_size, stride,  padding, ciel_mode, count_include_padding);
    std::cout << avg_padded << std::endl;
    Tensor error = avg_padded-1;
    Tensor grad = backward_avg_pool1d(example.shape(), error, kernel_size, stride,  padding, ciel_mode, count_include_padding);
    std::cout << grad << std::endl;
}


void lp_pool_1d(){
    using namespace nt;
    Tensor example = functional::arange({3, 3, 5}, DType::Float32);
    // Tensor t = example.pad({0,1});
    int64_t kernel_size = 3;
    int64_t stride = -1;
    Scalar power = 2;
    bool ciel_mode = false;



    Tensor lp_pooled = lp_pool1d(example, power, kernel_size, stride, ciel_mode);
    std::cout << lp_pooled << std::endl;
    Tensor error = lp_pooled-1;
    Tensor grad = backward_lp_pool1d(example, error, 
                                     power, kernel_size, stride, ciel_mode);
    std::cout << grad << std::endl; 
}


void max_pool_1d(){
    using namespace nt;
    Tensor example = functional::arange({3, 3, 5}, DType::Float32);
    int64_t kernel_size = 3;
    int64_t stride = -1;
    int64_t padding = 0;
    int64_t dilation = 1;
    bool ceil_mode = true;
    bool return_indices = true;

    auto [output, indices] = get<2>(max_pool1d(example, kernel_size, stride, padding, dilation, ceil_mode, return_indices));
    std::cout << output << std::endl;
    std::cout << indices << std::endl;
    std::cout << functional::at_tensor_split(example, indices, -2) << std::endl;
    Tensor error = output-1;
    Tensor grad = backward_max_pool1d(example.shape(), error, indices);
    std::cout << "grad: "<<grad<<std::endl;
    Tensor unpooled = max_unpool1d(output, indices, kernel_size, stride, padding);
    std::cout << "unpooled: "<<unpooled<<std::endl;
    Tensor unpooled_error = unpooled-1;
    Tensor unpooled_grad = backward_max_unpool1d(output.shape(), unpooled_error, indices, padding);
    std::cout << "unpooled grad: "<<unpooled_grad<<std::endl;
}


void avg_pool_2d(){
    using namespace nt;
    Tensor example = functional::arange({3, 3, 5}, DType::Float32);
    // Tensor t = example.pad({0,1});
    utils::my_tuple kernel_size = {3, 2};
    int64_t stride = -1;
    utils::my_tuple padding(0, 0);
    bool ceil_mode = true;
    bool count_include_padding = false;
    std::cout << "made padding"<<std::endl;


    Tensor avg_padded = avg_pool2d(example, kernel_size, stride,  padding, ceil_mode, count_include_padding);
    std::cout << avg_padded << std::endl;
    Tensor error = avg_padded-1;
    Tensor grad = backward_avg_pool2d(example.shape(), error, kernel_size, stride,  padding, ceil_mode, count_include_padding);
    std::cout << grad << std::endl;
}


void lp_pool_2d(){
    using namespace nt;
    Tensor example = functional::arange({3, 3, 5}, DType::Float32);
    // Tensor t = example.pad({0,1});
    utils::my_tuple kernel_size = {3, 3};
    int64_t stride = -1;
    Scalar power = 2;
    bool ciel_mode = true;



    Tensor lp_pooled = lp_pool2d(example, power, kernel_size, stride, ciel_mode);
    std::cout << lp_pooled << std::endl;
    Tensor error = lp_pooled-1;
    Tensor grad = backward_lp_pool2d(example, error, 
                                     power, kernel_size, stride, ciel_mode);
    std::cout << grad << std::endl; 
}


void max_pool_2d(){
    using namespace nt;
    Tensor example = functional::arange({3, 4, 4}, DType::Float32);
    int64_t kernel_size = 2;
    int64_t stride = -1;
    int64_t padding = 1;
    int64_t dilation = 1;
    bool ceil_mode = true;
    bool return_indices = true;

    auto [output, indices] = get<2>(max_pool2d(example, kernel_size, stride, padding, dilation, ceil_mode, return_indices));
    std::cout << output << std::endl;
    std::cout << indices << std::endl;
    std::cout << example << std::endl;
    // std::cout << functional::at_tensor_split(example, indices, -2) << std::endl;
    Tensor error = output-1;
    std::cout << "getting grad..."<<std::endl;
    Tensor grad = backward_max_pool2d(example.shape(), error, indices);
    std::cout << "grad: "<<grad<<std::endl;
    Tensor unpooled = max_unpool2d(output, indices, kernel_size, stride, padding);
    std::cout << "unpooled: "<<unpooled<<std::endl;
    Tensor unpooled_error = unpooled-1;
    Tensor unpooled_grad = backward_max_unpool2d(output.shape(), unpooled_error, indices, padding);
    std::cout << "unpooled grad: "<<unpooled_grad<<std::endl;
}

void avg_pool_3d(){
    using namespace nt;
    int64_t cube_size_rows = 2, cube_size_cols = 2, cube_size_depth = 2;
    int64_t stride_rows = 2, stride_cols = 2, stride_depth = 2;
    const int64_t batch_size = 1, channels = 3, depth = 4, height = 4, width = 4;
    //ceil mode:
    //padding_depth = (cube_size - depth % stride) % stride
    //padding_height = (cube_size - height % stride) % stride
    //padding_width = (cube_size - width % stride) % stride

    utils::my_n_tuple<3> kernel_size = 3;
    utils::my_n_tuple<3> stride = -1;
    utils::my_n_tuple<3> padding = 1;
    bool ciel_mode = true;
    bool count_include_padding = false;
    Tensor example = functional::arange({batch_size, channels, depth, height, width}, DType::Float32);



    Tensor avg_padded = avg_pool3d(example, kernel_size, stride,  padding, ciel_mode, count_include_padding);
    std::cout << avg_padded << std::endl;
    Tensor error = avg_padded-1;
    Tensor grad = backward_avg_pool3d(example.shape(), error, kernel_size, stride,  padding, ciel_mode, count_include_padding);
    std::cout << grad << std::endl;
}

void lp_pool_3d(){
    using namespace nt;
    Tensor example = functional::arange({4, 3, 3, 5}, DType::Float32);
    // Tensor t = example.pad({0,1});
    utils::my_n_tuple<3> kernel_size = {3, 3, 2};
    int64_t stride = -1;
    Scalar power = 2;
    bool ciel_mode = true;



    Tensor lp_pooled = lp_pool3d(example, power, kernel_size, stride, ciel_mode);
    std::cout << lp_pooled << std::endl;
    Tensor error = lp_pooled-1;
    Tensor grad = backward_lp_pool3d(example, error, 
                                     power, kernel_size, stride, ciel_mode);
    std::cout << grad << std::endl; 
}


void max_pool_3d(){
    using namespace nt;
    Tensor example = functional::arange({3, 4, 4, 4}, DType::Float32);
    int64_t kernel_size = 2;
    int64_t stride = -1;
    int64_t padding = 1;
    int64_t dilation = 1;
    bool ceil_mode = true;
    bool return_indices = true;

    auto [output, indices] = get<2>(max_pool3d(example, kernel_size, stride, padding, dilation, ceil_mode, return_indices));
    std::cout << output << std::endl;
    std::cout << indices << std::endl;
    std::cout << example << std::endl;
    // std::cout << functional::at_tensor_split(example, indices, -2) << std::endl;
    Tensor error = output-1;
    std::cout << "getting grad..."<<std::endl;
    Tensor grad = backward_max_pool3d(example.shape(), error, indices);
    std::cout << "grad: "<<grad<<std::endl;
    Tensor unpooled = max_unpool3d(output, indices, kernel_size, stride, padding);
    std::cout << "unpooled: "<<unpooled<<std::endl;
    Tensor unpooled_error = unpooled-1;
    Tensor unpooled_grad = backward_max_unpool3d(output.shape(), unpooled_error, indices, padding);
    std::cout << "unpooled grad: "<<unpooled_grad<<std::endl;
}


void fractional_test(){
    int64_t kernel_size = 3;
    int64_t input_size = 28;
    int64_t output_size = 10;
    std::vector<int64_t> sliding = getSlidingWindowSizesOutput(kernel_size, input_size, output_size);
    for(const auto s : sliding)
        std::cout << s << std::endl;
    std::cout << sliding.size() << std::endl;

    nt::Tensor example = nt::functional::arange({2,input_size, input_size}, nt::DType::Float32);
    std::cout << example << std::endl;
    nt::Tensor slided = extract_sliding_windows_max_2d(sliding, sliding, example);
    std::cout << slided << std::endl;
    nt::Tensor out_max = example[slided].view(example.shape().redo_index(-2, sliding.size()).redo_index(-1, sliding.size()));
    std::cout << out_max << std::endl;
    nt::Tensor indices = nt::functional::where(slided.flatten(-2, -1))[-1].item<nt::Tensor>().view(out_max.shape());
    std::cout << indices << std::endl;
}


void fractional_max_pool_2d(){
    nt::Tensor input = nt::functional::arange({2, 28, 27}, nt::DType::Float32);
    auto [output, indices] = nt::get<2>(fractional_max_pool2d(input, {4,3}, {11,10}, -1.0, true));
    std::cout << "output: "<<output<<std::endl;
    std::cout << "indices: "<<indices<<std::endl;
}

void fractional_max_pool_3d(){
    nt::Tensor input = nt::functional::arange({2, 19, 28, 27}, nt::DType::Float32);
    auto [output, indices] = nt::get<2>(fractional_max_pool3d(input, {3,4,3}, {9,11,10}, -1.0, true));
    std::cout << "output: "<<output<<std::endl;
    std::cout << "indices: "<<indices<<std::endl;
}

