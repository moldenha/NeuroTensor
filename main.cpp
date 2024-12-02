#include "src/Tensor.h"
#include "src/dtype/DType.h"
#include "src/dtype/DType_enum.h"
#include <_types/_uint32_t.h>
#include <ios>
#include <numeric>
#include <ostream>
/* #include "autograd_test.h" */
#include "src/dtype/ArrayVoid.hpp"
#include "src/mp/Threading.h" //testing this at the moment
#include "tda_test.h"
#include <chrono>
#include <random>

#include <immintrin.h>
#include <mutex>



/* #if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) */
/* void print_working_simd(){ */
/* 	std::cout << "simd is supported"<<std::endl; */
/* } */
/* #else */
/* void print_working_simd(){ */
/* 	std::cout << "simd is not supported"<<std::endl; */
/* } */
/* #endif */





void print_getting_vec(const std::vector<int64_t> j){
	for(int i = 0; i < j.size(); ++i)
		std::cout << j[i] << ',';
	std::cout << std::endl;
}


void print_ref_vec(const std::vector<int64_t>& j){
	for(int i = 0; i < j.size(); ++i)
		std::cout << j[i] << ',';
	std::cout << ')'<<std::endl;
}
int64_t generateRandomInt() {
    // Create a random device to seed the generator
    std::random_device rd;
    
    // Use the Mersenne Twister engine
    std::mt19937 gen(rd());
    
    // Define the range [1, 10]
    std::uniform_int_distribution<> distr(1, 10);
    
    // Generate and return the random number
    return static_cast<int64_t>(distr(gen));
}






void conv_test(){
	nt::Tensor a_1d = nt::functional::randn({3,400,500});
	nt::Tensor k_1d = nt::functional::randn({500,400,10});
	std::cout << "convoluting..."<<std::endl;
	nt::Tensor o_1d = nt::functional::conv1d(a_1d, k_1d); //shaoe should be (3,500,491)
	std::cout << "conv1d output shape: "<<o_1d.shape()<<std::endl;

	nt::Tensor a_2d = nt::functional::randn({2,500,90,90});
	nt::Tensor k_2d = nt::functional::randn({600,500,3,3});
	std::cout << "convoluting..."<<std::endl;
	nt::Tensor o_2d = nt::functional::conv2d(a_2d, k_2d); //shaoe should be (2,600,88,88)
	std::cout << "conv2d output shape: "<<o_2d.shape()<<std::endl;

	nt::Tensor a_3d = nt::functional::randn({2,500,90,90,90});
	nt::Tensor k_3d = nt::functional::randn({600,500,3,3,3});
	std::cout << "convoluting..."<<std::endl;
	nt::Tensor o_3d = nt::functional::conv3d(a_3d, k_3d); //shaoe should be (2,600,88,88,88) (also pytorch's is really slow)
	std::cout << "conv3d output shape: "<<o_3d.shape() << std::endl;

	
}


void mkl_matmult(const nt::Tensor& a, const nt::Tensor& b){
	std::cout << "made tensor a and tensor b, now going to measure speed of matmult on mkl..."<<std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	nt::Tensor out = nt::functional::matmult(a, b);
	auto stop = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = stop - start;
	std::cout << "finished execution of function successfully in "<<duration.count() << " seconds"<<std::endl;
	std::cout << "of mkl (3,2000,2000) x (3,2000,3000)"<<std::endl;
}

void std_matmult(const nt::Tensor& a, const nt::Tensor& b){
	std::cout << "made tensor a and tensor b, now going to measure speed of matmult on std..."<<std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	nt::Tensor out = nt::functional::matmult_std(a, b);
	auto stop = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = stop - start;
	std::cout << "finished execution of function successfully in "<<duration.count() << " seconds"<<std::endl;
	std::cout << "of std (3,2000,2000) x (3,2000,3000)"<<std::endl;	
}


void mkl_vs_std_matmult(){
	nt::Tensor a = nt::functional::randn({3, 2000, 2000});
	nt::Tensor b = nt::functional::randn({3, 2000, 3000});
	mkl_matmult(a, b);
	std_matmult(a, b);


}



//1-800-318-2596

int main(){
	
	/* new_batched_tda_refinement_numpy<2>("/Users/sammoldenhauer/Downloads/college/southwell/presentations/NeuroRetreat_3/outputs.npy", */
	/* 		"/Users/sammoldenhauer/Downloads/new_tensor/tda_outputs_2log", */
	/* 		"/Users/sammoldenhauer/Downloads/college/southwell/presentations/NeuroRetreat_3/names.txt"); */
		
	/* new_batched_tda_refinement_numpy<4>("/Users/sammoldenhauer/Downloads/college/southwell/presentations/NeuroRetreat_3/outputs.npy", */
	/* 		"/Users/sammoldenhauer/Downloads/new_tensor/tda_outputs_4log", */
	/* 		"/Users/sammoldenhauer/Downloads/college/southwell/presentations/NeuroRetreat_3/names.txt"); */
	
	/* new_batched_tda_refinement_numpy<0>("/Users/sammoldenhauer/Downloads/college/southwell/presentations/NeuroRetreat_3/outputs.npy", */
	/* 		"/Users/sammoldenhauer/Downloads/new_tensor/tda_outputs", */
	/* 		"/Users/sammoldenhauer/Downloads/college/southwell/presentations/NeuroRetreat_3/names.txt"); */
	

	/* new_batched_tda_refinement_numpy<8>("/Users/sammoldenhauer/Downloads/college/southwell/presentations/NeuroRetreat_3/outputs.npy", */
	/* 		"/Users/sammoldenhauer/Downloads/new_tensor/tda_outputs_8log", */
	/* 		"/Users/sammoldenhauer/Downloads/college/southwell/presentations/NeuroRetreat_3/names.txt"); */
	tda_test();

	//there is an issue with =(Scalar) when the tensor is uint8_t (it was originally set to only work with floats)
	/* nt::Tensor a = nt::functional::randn({20,20}); */


	

	/* force_contiguity_and_bucket_test(); */

	/* nt::Tensor rand_b = nt::functional::cat(nt::functional::arange({20,30,30,4}, nt::DType::Float32), */ 
	/* 				nt::functional::arange({30,30,30,4}, nt::DType::Float32)); */
	/* /1* nt::Tensor rand_b({2,20,50,600,600}, nt::DType::Float32); *1/ */ 
	/* std::cout << "rand_b numel: "<<rand_b.numel()<<std::endl; */
	/* std::cout << "testing if they are the same:"<<std::endl; */
	/* current_test([&rand_b](){ */
	/* 			nt::Tensor split_a = rand_b.split_axis_experimental(2); */
	/* 			nt::Tensor split_b = rand_b.split_axis(2); */
	/* 			std::cout <<"they are all the same: "<<std::boolalpha<< nt::functional::all(split_a == split_b) << std::noboolalpha<< std::endl; */
	/* 		}); */
	
	/* std::cout << "made rand_b, testing original split_axis():"<<std::endl; */
	/* current_test([&rand_b](){ */
	/* 			nt::Tensor split = rand_b.split_axis(2); */
	/* 		}); */

	/* tensor_calloc(); */


	
	/* std::cout << "has "<<nt::functional::count(rand_a == 1) << " ones"<<std::endl; */
	/* nt::Tensor rand_b = current_test(test_set, rand_a, 1, 0); */
	/* std::cout << "rand_b has "<<nt::functional::count(rand_b == 1) << " ones"<<std::endl; */
	/* nt::Tensor rand_c = current_test(test_transpose, rand_b, -1, -2); */
	/* std::cout << "rand_c was transposed"<<std::endl; */

	/* current_test(pool_multi_processing_test); */
	/* print_working_simd(); */
	/* test_working(); */
	/* const nt::Tensor a = nt::functional::randint(0, 5, {1,7744,1800}, nt::DType::Float32); */
	/* const nt::Tensor b = nt::functional::randint(0, 5, {400,1800}, nt::DType::Float32); */
	/* const nt::Tensor a = nt::functional::randint(0, 5, {1,70,18}, nt::DType::Float32); */
	/* const nt::Tensor b = nt::functional::randint(0, 5, {40,18}, nt::DType::Float32); */
	/* nt::Tensor a({1,70,18}, nt::DType::Float32); */
	/* nt::Tensor b({40,18}, nt::DType::Float32); */

	/* std::cout << "done making a and b"<<std::endl; */
	/* std::cout << a[0] <<std::endl; */
	/* std::cout << b[0] << std::endl; */
	/* current_testT<const nt::Tensor&, const nt::Tensor&>(std::function<void (const nt::Tensor &, const nt::Tensor &)>(better_matMult), a, b); */
	return 0;
}

	/* current_test(shared_memory_test); */
