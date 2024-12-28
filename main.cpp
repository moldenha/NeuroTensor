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
/* #include "tda_test.h" */
#include <chrono>
#include <random>

#include <immintrin.h>
#include <mutex>
#include "tests/tensor_test.h"
#include "tests/tensorgrad_test.h"
#include "tests/layer_test.h"






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










int main(){
	
	/* test_layers(); */
	test_lnn();

	return 0;
}

