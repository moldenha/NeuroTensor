// #include "tests/tensor_test.h"
// #include "tests/tensorgrad_test.h"
// #include "tests/layer_test.h"
// #include "tests/tda_test.h"
#include "tests/nn_tda_test.h"
// #include "tests/linalg_test.h"
// #include "tests/fmri_test.h"

#include "tests/pooling.h"
#include <chrono>
#include <nt/functional/cpu/fused.h>
#include <nt/mp/simde_traits.h>
#include <nt/mp/simde_traits/simde_traits_iterators.h>
#include <nt/dtype/ArrayVoid.hpp>
#include <nt/functional/tensor_files/rand.h>


void ex_pts_test(){
    nt::TensorGrad x(nt::Tensor({4, 3}, nt::DType::Float32), true); 
    x.detach() << 3.0, 4.0, 5.0, 3.2,
                  4.2, 6.0, 6.7, 8.9,
                  7.0, 9.0, 1.0, 9.0; 
    nt::Tensor example_ng = nt::functional::ones({1, 2}, nt::DType::Float32); 
    example_ng << 1.0, 3.0; 
    nt::Tensor example_ng2 = nt::functional::ones({6, 1}, nt::DType::Float32); 
    nt::TensorGrad points(nt::functional::zeros({4, 2}, x.dtype()), true); 
    nt::TensorGrad logits = x.flatten(0, -1); 
    nt::TensorGrad remaining = logits; 
    for(int k = 0; k < 4; ++k){ 
        nt::TensorGrad weights = nt::functional::softmax(remaining);
        nt::TensorGrad w_ = weights.view(12, 1); 
        nt::Tensor ex_ = example_ng.view(1, 2); 
        nt::TensorGrad p = (w_ * ex_).sum(0); 
        points[k] += p; 
        nt::TensorGrad dist = (example_ng2 - p); 
        remaining -= dist.flatten(0, -1);
    } 
    points *= 5; 
    nt::TensorGrad sq = nt::functional::sqrt(points);
    nt::Tensor grad = nt::functional::ones_like(sq.detach()); 
    

    auto autograd = sq.get_auto_grad();
    auto path = autograd.to_list();
    for(const auto& node : path)
        std::cout << node->name() << "->";
    std::cout << "done" << std::endl;

    sq.backward(grad, /*retain_graph = */true);


    std::cout << points.grad() << std::endl; 
    std::cout << x.grad() << std::endl;
    std::cout << "doing second derivative for test..." << std::endl;
    sq.backward(grad, false);
    std::cout << points.grad() << std::endl; 
    std::cout << x.grad() << std::endl;

 
}

// This is the beggining of the TensorCompile, is testing the speed differences
// When using the base nt::functional::cpu ArrayVoid implementation, using the ArrayVoid lambda implementation
//  and using the base, in this example float* implementation
/*
❯ ./main.out
Average Native CPU time taken: 0.00184562 microseconds
ArrayVoid implementation time taken: 0.00173424 microseconds
Native data ptr implementation time taken: 0.00175222 microseconds
❯ ./main.out
Average Native CPU time taken: 0.00172124 microseconds
ArrayVoid implementation time taken: 0.0017344 microseconds
Native data ptr implementation time taken: 0.00173138 microseconds
*/

// Based on the above resuls, it seems that there is a zero cost abstraction due to the design of ArrayVoid
//  (which is nice because memory views are automatically handled)
// From there, the fusion of ops is easy, and a TensorCompiled is really no problem at all from there
//  just holder for operations at that point.


std::chrono::microseconds triple_time_fuse_cpu_(){
    nt::Tensor c = nt::functional::randn({512, 512, 64}, nt::DType::Float32);
    nt::Tensor a = nt::functional::randn({512, 512, 64}, nt::DType::Float32);
    nt::Scalar b = 0.5;
    auto start = std::chrono::high_resolution_clock::now();
    nt::functional::cpu::_fused_multiply_add_(c.arr_void(), a.arr_void(), b);
    auto stop = std::chrono::high_resolution_clock::now();
    // make sure result is used
    float sum = 0.0f;
    float* ptr = reinterpret_cast<float*>(c.data_ptr());
    for (int i = 0; i < 100; ++i) sum += ptr[i];
    volatile float sink = sum;
    return std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout << "Native CPU time taken: " << duration.count() << " microseconds" << std::endl;
}

void triple_time_fuse_cpu(){
    triple_time_fuse_cpu_();
    const int N = 50;
    double total = 0.0;
    for(int i = 0; i < N; ++i){
        double time = std::chrono::duration<double>(triple_time_fuse_cpu_()).count() / 50.0;
        total += time;
    }
    std::cout << "Average Native CPU time taken: " << total << " microseconds" << std::endl;
}


namespace nt::mp_test{

template<typename T, typename U>
inline void fused_multiply_add_scalar(T begin_a, T end_a, U begin_c, ::nt::utils::IteratorBaseType_t<T> num){
	static_assert(std::is_same_v<::nt::utils::IteratorBaseType_t<T>, ::nt::utils::IteratorBaseType_t<U> > 
                , "Expected to get base types the same for simde optimized routes");
	using base_type = ::nt::utils::IteratorBaseType_t<T>;
	if constexpr (::nt::mp::simde_supported_v<base_type>){
		static constexpr size_t pack_size = ::nt::mp::pack_size_v<base_type>;
		::nt::mp::simde_type<base_type> nums = ::nt::mp::SimdTraits<base_type>::set1(num);
		for(;begin_a + pack_size <= end_a; begin_a += pack_size, begin_c += pack_size){
			::nt::mp::simde_type<base_type> a = ::nt::mp::it_loadu(begin_a);
			::nt::mp::simde_type<base_type> c = ::nt::mp::it_loadu(begin_c);
            ::nt::mp::SimdTraits<base_type>::fmadd(a, nums, c);
			::nt::mp::it_storeu(begin_c, c);
		}
		for(; begin_a != end_a; ++begin_a, ++begin_c)
			*begin_c += *begin_a * num;
	}else{
		for(; begin_a != end_a; ++begin_a, ++begin_c)
			*begin_c += *begin_a * num;
	}
    
}

}


std::chrono::microseconds triple_time_arr_void_cpu_(){
    nt::Tensor c = nt::functional::randn({512, 512, 64}, nt::DType::Float32);
    nt::Tensor a = nt::functional::randn({512, 512, 64}, nt::DType::Float32);
    nt::Scalar b = 0.5;
    auto start = std::chrono::high_resolution_clock::now();
    a.arr_void().execute_function<::nt::WRAP_DTYPES<::nt::NumberTypesL> >([&b](auto begin_a, auto end_a, auto begin_c){
		using value_t = nt::utils::IteratorBaseType_t<decltype(begin_a)>;
        value_t num = b.to<value_t>();
        nt::mp_test::fused_multiply_add_scalar(begin_a, end_a, begin_c, num);
	}, c.arr_void());
    auto stop = std::chrono::high_resolution_clock::now();
    float sum = 0.0f;
    float* ptr = reinterpret_cast<float*>(c.data_ptr());
    for (int i = 0; i < 100; ++i) sum += ptr[i];
    volatile float sink = sum;
    return std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout << "Native CPU time taken: " << duration.count() << " microseconds" << std::endl;
}

void triple_time_arr_void_cpu(){
    triple_time_arr_void_cpu_();
    const int N = 50;
    double total = 0.0;
    for(int i = 0; i < N; ++i){
        double time = std::chrono::duration<double>(triple_time_arr_void_cpu_()).count() / 50.0;
        total += time;
    }
    std::cout << "ArrayVoid implementation time taken: " << total << " microseconds" << std::endl;
}

std::chrono::microseconds triple_time_data_ptr_cpu_(){
    nt::Tensor c_ = nt::functional::randn({512, 512, 64}, nt::DType::Float32);
    nt::Tensor a_ = nt::functional::randn({512, 512, 64}, nt::DType::Float32);
    nt::Scalar b_ = 0.5;
    auto start = std::chrono::high_resolution_clock::now();
    float* c = reinterpret_cast<float*>(c_.data_ptr());
    float* a = reinterpret_cast<float*>(a_.data_ptr());
    float* a_end = reinterpret_cast<float*>(a_.data_ptr_end());
    float b = b_.to<float>();
    nt::mp_test::fused_multiply_add_scalar(a, a_end, c, b);
    auto stop = std::chrono::high_resolution_clock::now();
    float sum = 0.0f;
    float* ptr = reinterpret_cast<float*>(c_.data_ptr());
    for (int i = 0; i < 100; ++i) sum += ptr[i];
    volatile float sink = sum;
    return std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

}

void triple_time_data_ptr_cpu(){
    triple_time_arr_void_cpu_();
    const int N = 50;
    double total = 0.0;
    for(int i = 0; i < N; ++i){
        double time = std::chrono::duration<double>(triple_time_data_ptr_cpu_()).count() / 50.0;
        total += time;
    }
    std::cout << "Native data ptr implementation time taken: " << total << " microseconds" << std::endl;
}

void triple_time(){
    // going to use the fused operation for c += (a * b); -> b is a scalar type
    triple_time_fuse_cpu();
    triple_time_arr_void_cpu();
    triple_time_data_ptr_cpu();
}

int main(){
    triple_time();
    // svd_test();
    // qr_test();
    // inv_test();
    // pinv_test();
    // persistent_diagram_test();
	// convT_gradient_tests();
	/* test_layers(); */
	// test_lnn();
    // operator_test();
    // fmri_load();
    // eye_test();
    // nn_laplacian_2_test();
    // nn_laplacian_2_test_sub();
    // nn_boundary_test();
    // row_swap_test();
    // softmax_test();
    // auto func1 = [](const nt::TensorGrad& x){return nt::functional::relu(x);}; 
    // auto func2 = [](const nt::TensorGrad& x){return nt::functional::softmax(x);}; 
    // bool worked = activation_function_test(func1, func2);
    // linear_test();
    // bool worked = test_gumbel_softmax_activation();
    // std::cout << std::boolalpha << "worked: "<<worked<<std::noboolalpha << std::endl;
    // symmetric_mult_test();
    // fractional_max_pool(2);
    
    return 0;
}
