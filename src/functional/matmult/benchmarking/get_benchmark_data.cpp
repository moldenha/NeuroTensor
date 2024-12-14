//compiling with the mkl library
//g++ -O3 -std=c++17 -o benchmark_data get_benchmark_data.cpp -march=native -I/opt/intel/oneapi/mkl/latest/include -L/opt/intel/oneapi/mkl/latest/lib -Wl,-rpath,/opt/intel/oneapi/mkl/latest/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -I/opt/homebrew/include -ltbb -I../../../../third_party/simde
//
//when testing for accuracy and segmentation faults:
//
//g++ -O3 -std=c++17 -o matmult_simde_fma matmult_simde_fma.cpp -march=native -I/opt/intel/oneapi/mkl/latest/include -L/opt/intel/oneapi/mkl/latest/lib -Wl,-rpath,/opt/intel/oneapi/mkl/latest/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -I/opt/homebrew/include -ltbb -I../../../../third_party/simde -fsanitize=address -g


#include <iostream>
#include <simde/x86/svml.h>

#include <string.h>
#include <random>
#include "float16_type.h" //so that NeuroTensor types are overriden and everything does not have to be compiled
#include "../nt_matmult.hpp"

#include <chrono>
#include <cstdlib> // For std::aligned_alloc
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/blocked_range3d.h>
#include <tbb/global_control.h>
#include <tbb/task_group.h>
#include <functional>
#include <unistd.h>
#include <simde/x86/sse3.h>
#include <complex>
#include <random>
#include <algorithm>
#include <fstream>

#include <mkl.h>
#include <mkl_types.h>

//Link against Intel MKL library
#pragma comment(lib, "mkl_intel_lp64.lib")
#pragma comment(lib, "mkl_sequential.lib")

namespace nt{
namespace functional{
namespace std_functional{


_NT_MATMULT_DECLARE_STATIC_BLOCK_(float)

//a_rows, a_cols, b_cols
void matmul_mkl(const float* A, const float* B, float* C, MKL_INT64 M, MKL_INT64 K, MKL_INT64 N) {
    // A: MxK matrix
    // B: KxN matrix
    // C: MxN result matrix
    float alpha = 1.0f;  // Scalar multiplier for A * B
    float beta = 0.0f;   // Scalar multiplier for C (if C is preloaded with values)

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    cblas_sgemm_64(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,      // Dimensions of matrices
                alpha,        // Alpha value
                A, K,         // Matrix A and its leading dimension
                B, N,         // Matrix B and its leading dimension
                beta,         // Beta value
                C, N);        // Matrix C and its leading dimension
}

template<typename T>
void generate_random_matrix(T* var, int64_t rows, int64_t cols){
	std::random_device rd;
	std::minstd_rand gen(rd());
	std::uniform_int_distribution<> dis(0, 10);
	std::generate(var, var + (rows * cols), [&]() { return static_cast<T>(dis(gen)); });
}

template<typename T>
void generate_matrix(T* var, T start, int64_t rows, int64_t cols, T outter = 16){
	bool subtract = (start != 1);
	for(int64_t i = 0; i < (rows * cols); ++i, ++var){
		*var = start;
		start += (subtract) ? -1 : 1;
		if(subtract){
			if(start < 1){
				start = outter;
			}
		}else if(start > outter){start = 1;}
	}
}

template<typename T>
int64_t unit_time_test(int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols){
	T* A = static_cast<T*>(std::aligned_alloc(64, _NT_MATMULT_ENSURE_ALIGNMENT_(T, 64, a_rows * a_cols)));
	T* B = static_cast<T*>(std::aligned_alloc(64, _NT_MATMULT_ENSURE_ALIGNMENT_(T, 64, b_rows * b_cols)));
	generate_random_matrix<T>(A, a_rows, a_cols);
	generate_random_matrix<T>(B, b_rows, b_cols);

	/* T* C1 = static_cast<T*>(std::aligned_alloc(64, ENSURE_ALIGNMENT(T, 64, a_rows * b_cols))); */
	T* C2 = static_cast<T*>(std::aligned_alloc(64, _NT_MATMULT_ENSURE_ALIGNMENT_(T, 64, a_rows * b_cols)));
	T* C3 = static_cast<T*>(std::aligned_alloc(64, _NT_MATMULT_ENSURE_ALIGNMENT_(T, 64, a_rows * b_cols)));

	/* zero_memory(C1, a_rows * b_cols); */
	zero_memory(C2, a_rows * b_cols);
	zero_memory(C3, a_rows * b_cols);

	/* std::memset(C1, 0, sizeof(T) * a_rows * b_cols); */
	/* std::memset(C2, 0, sizeof(T) * a_rows * b_cols); */

	auto start = std::chrono::high_resolution_clock::now();
	/* matmult_naive<T>(A, B, C1, a_rows, a_cols, b_rows, b_cols, false, false); */
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	/* std::cout << "Time taken by native: " << duration.count() << " microseconds" << std::endl; */
	start = std::chrono::high_resolution_clock::now();
	nt_matmult<T>(A, B, C2, a_rows, a_cols, b_rows, b_cols, false, false);
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Time taken by new simde: " << duration.count() << " microseconds" << std::endl;
	int64_t simde_time = duration.count();

	/* start = std::chrono::high_resolution_clock::now(); */
	/* kmatmult_simde<T>(A, B, C3, a_rows, a_cols, b_rows, b_cols); */
	/* stop = std::chrono::high_resolution_clock::now(); */
	/* duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); */
	/* std::cout << "Time taken by  ksimde: " << duration.count() << " microseconds" << std::endl; */
	/* int64_t ksimde_time = duration.count(); */


	
	if constexpr (std::is_same_v<T, float>){
		T* C4 = static_cast<T*>(std::aligned_alloc(64, _NT_MATMULT_ENSURE_ALIGNMENT_(T, 64, a_rows * b_cols)));
		std::memset(C4, 0, sizeof(T) * a_rows * b_cols);
		start = std::chrono::high_resolution_clock::now();
		matmul_mkl(A, B, C4, a_rows, a_cols, b_cols);
		stop = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		std::cout << "Time taken by mkl: " << duration.count() << " microseconds" << std::endl;
		std::free(C4);
		int64_t mkl_time = duration.count();
		double dif = (double)((double)simde_time / (double)mkl_time);
		std::cout << "the mkl time is "<<dif<<" times faster than new simde"<<std::endl;
		/* double difk = (double)((double)ksimde_time / (double)mkl_time); */
		/* std::cout << "the mkl time is "<<difk<<" times faster than ksimde"<<std::endl; */
	}


	std::free(A);
	std::free(B);
	/* std::free(C1); */
	std::free(C2);
	std::free(C3);
	return 0;
}

template<typename T>
void test_matmult_time(size_t max = 10){
	std::random_device rd; // Seed source
	std::mt19937 gen(rd()); // Mersenne Twister generator seeded with rd()
	std::uniform_int_distribution<> distrib(800, 1024);
	std::cout << "Tile size is: "<<tile_size_v<T><<std::endl;
	for(size_t i = 0; i < max; ++i){
		int64_t a_rows = distrib(gen);
		int64_t a_cols = distrib(gen);
		int64_t b_rows = a_cols;
		int64_t b_cols = distrib(gen);
		std::cout << "testing ("<<a_rows<<','<<a_cols<<") x ("<<b_rows<<','<<b_cols<<"): "<<std::endl;
		unit_time_test<T>(a_rows, a_cols, b_rows, b_cols); //in this case, tile size of 8 is faster
		/* std::cout << "Tile size 16:"<<std::endl; */
		/* unit_time_test<16>(a_rows, a_cols, b_rows, b_cols); */
		/* std::cout << "Tile size 32:"<<std::endl; */
		/* unit_time_test<32>(a_rows, a_cols, b_rows, b_cols); */

	}
}


int64_t unit_benchmark_test(int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, int64_t& nt_time, int64_t& mkl_time, float* A, float* B, float* C1, float* C2){

	zero_memory(C1, a_rows * b_cols);
	zero_memory(C2, a_rows * b_cols);

	auto start = std::chrono::high_resolution_clock::now();
	nt_matmult<float>(A, B, C1, a_rows, a_cols, b_rows, b_cols, false, false);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	/* std::cout << "Time taken by new simde: " << duration.count() << " microseconds" << std::endl; */
	nt_time += duration.count();

	start = std::chrono::high_resolution_clock::now();
	matmul_mkl(A, B, C2, a_rows, a_cols, b_cols);
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	/* std::cout << "Time taken by mkl: " << duration.count() << " microseconds" << std::endl; */
	mkl_time += duration.count();

	return 0;
}

void benchmark_matmult(){
	constexpr int64_t max_size = 2000;
	size_t warmup = 10;
	for(size_t i = 0; i < warmup; ++i){
		std::cout << "Warming up "<<i<<std::endl;
		unit_time_test<float>(max_size, max_size, max_size, max_size);
	}
	std::array<int64_t, max_size> mkl_microseconds;
	std::array<int64_t, max_size> nt_microseconds;
	std::cout << "srarting"<<std::endl;
	for(int64_t i = 0; i < max_size; ++i){
		int64_t mkl_time = 0;
		int64_t nt_time  = 0;
		float* A = static_cast<float*>(std::aligned_alloc(64, _NT_MATMULT_ENSURE_ALIGNMENT_(float, 64, i * i)));
		float* B = static_cast<float*>(std::aligned_alloc(64, _NT_MATMULT_ENSURE_ALIGNMENT_(float, 64, i * i)));
		generate_random_matrix<float>(A, i, i);
		generate_random_matrix<float>(B, i, i);

		float* C1 = static_cast<float*>(std::aligned_alloc(64, _NT_MATMULT_ENSURE_ALIGNMENT_(float, 64, i * i)));
		float* C2 = static_cast<float*>(std::aligned_alloc(64, _NT_MATMULT_ENSURE_ALIGNMENT_(float, 64, i * i)));
		for(size_t j = 0; j < 10; ++j){
			unit_benchmark_test(i, i, i, i, nt_time, mkl_time, A, B, C1, C2);
		}
		std::free(A);
		std::free(B);
		std::free(C1);
		std::free(C2);
		mkl_time /= 10;
		nt_time /= 10;
		std::cout << '(' << i << " , " << i << "): Average MKL: "<<mkl_time<< ", Average NeuroTensor: "<<nt_time<<", NeuroTensor is "<<((double)mkl_time / (double)nt_time)<<" times faster"<<std::endl;
		mkl_microseconds[i] = mkl_time;
		nt_microseconds[i] = nt_time;
	}
	std::ofstream file("benchmark.txt");
	if (!file) {
		std::cerr << "Error: Could not open file benchmark.txt for writing." << std::endl;
		return;
	}
	for(int64_t i = 0; i < max_size; ++i){
		file << mkl_microseconds[i] << ' ' << nt_microseconds[i] << '\n';
	}
	
}

}}}














int main(){
	nt::functional::std_functional::benchmark_matmult();

	return 0;
}
