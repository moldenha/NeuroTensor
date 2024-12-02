#ifndef _NT_THREADING_H_
#define _NT_THREADING_H_

#include <array>
#include <type_traits>
//going to replace this all with simde instructions
/* #include <immintrin.h> */
#include <algorithm>
#include <iostream>
#include "../utils/utils.h"
#include <numeric>
#include "../memory/iterator.h"

#ifdef USE_PARALLEL
	#include <thread>
	#include <tbb/parallel_for.h>
	#include <tbb/blocked_range.h>
	#include <tbb/blocked_range2d.h>
	#include <tbb/blocked_range3d.h>
#endif

/* #if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) */
/*     // Compiler supports AVX, AVX2, and AVX-512F instruction sets */
/*     #define DOT_PRODUCT_SIMD_SUPPORTED 1 */
/* #else */
#define DOT_PRODUCT_SIMD_SUPPORTED 0
/* #endif */

namespace nt{
namespace threading{



/* // Dot product using AVX-512F instructions */
/* /1* #ifdef __AVX512F__ *1/ */

/* // Function pointer type for dot product function */
/* template<typename T> */
/* using DotProductFunc = T(*)(const T*, const T*, const T*, T); */

/* // Default implementation for other types */
/* template<typename T> */
/* T inner_product_default(const T* a, const T* b, const T* a_end, T final_sum) { */
/*     return std::inner_product(a, a_end, b, final_sum); */
/* } */
/* inline float float_dot_product(const float* a, const float* b, const float* a_end, float final_sum) { */
/*     __m512 sum = _mm512_setzero_ps(); */
/*     while ((a+16) < a_end) { */
/*         __m512 va = _mm512_loadu_ps(a); */
/*         __m512 vb = _mm512_loadu_ps(b); */
/*         sum = _mm512_fmadd_ps(va, vb, sum); */
/*         a += 16; */
/*         b += 16; */
/*     } */
/*     alignas(64) float result[16]; */
/*     _mm512_store_ps(result, sum); */
/*     for (int i = 0; i < 16; ++i) { */
/*         final_sum += result[i]; */
/*     } */
/*     return std::inner_product(a, a_end, b, final_sum); */
/*     /1* final_sum += std::inner_product(a, a_end, b, 0); *1/ */
/* } */


/* inline double double_dot_product(const double* a, const double* b, const double* a_end, double final_sum) { */
/*     __m512d sum = _mm512_setzero_pd(); */
/*     while ((a + 8) < a_end) { */
/*         __m512d va = _mm512_loadu_pd(a); */
/*         __m512d vb = _mm512_loadu_pd(b); */
/*         sum = _mm512_add_pd(sum, _mm512_mul_pd(va, vb)); */
/*         a += 8; */
/*         b += 8; */
/*     } */
/*     alignas(64) double result[8]; */
/*     _mm512_store_pd(result, sum); */
/*     for (int i = 0; i < 8; ++i) { */
/*         final_sum += result[i]; */
/*     } */
/*     return std::inner_product(a, a_end, b, final_sum); */
/* } */


/* // Define struct template for current inner product */
/* template<typename T> */
/* struct CURRENT_INNER_PRODUCT { */
/*     static constexpr DotProductFunc<T> function = &inner_product_default<T>; */
/* }; */

/* // Specializations for float and double */
/* template<> */
/* constexpr DotProductFunc<float> CURRENT_INNER_PRODUCT<float>::function = &float_dot_product; */

/* template<> */
/* constexpr DotProductFunc<double> CURRENT_INNER_PRODUCT<double>::function = &double_dot_product; */

/* /1* template<> *1/ */
/* /1* constexpr DotProductFunc<int8_t> CURRENT_INNER_PRODUCT<int8_t>::function = &int8_dot_product; *1/ */

/* /1* template<> *1/ */
/* /1* constexpr DotProductFunc<uint8_t> CURRENT_INNER_PRODUCT<uint8_t>::function = &uint8_dot_product; *1/ */

/* /1* template<> *1/ */
/* /1* constexpr DotProductFunc<int16_t> CURRENT_INNER_PRODUCT<int16_t>::function = &int16_dot_product; *1/ */

/* /1* template<> *1/ */
/* /1* constexpr DotProductFunc<uint16_t> CURRENT_INNER_PRODUCT<uint16_t>::function = &uint16_dot_product; *1/ */

/* /1* template<> *1/ */
/* /1* constexpr DotProductFunc<int32_t> CURRENT_INNER_PRODUCT<int32_t>::function = &int32_dot_product; *1/ */

/* /1* template<> *1/ */
/* /1* constexpr DotProductFunc<uint32_t> CURRENT_INNER_PRODUCT<uint32_t>::function = &uint32_dot_product; *1/ */

/* /1* template<> *1/ */
/* /1* constexpr DotProductFunc<int64_t> CURRENT_INNER_PRODUCT<int64_t>::function = &int64_dot_product; *1/ */

/* // Function pointer type for dot product function */
/* template<typename T> */
/* using DotProductStridedFunc = T(*)(const T*, const T*, const T*, T, const int64_t, const int64_t); */


/* template<typename T> */
/* inline T dot_product_strided(const T* a_begin, const T* b_begin, const T* a_end, T final_product const int64_t a_stride, const int64_t b_stride){ */
/*     while (a_begin < a_end) { */
/*         final_product += (*a_begin) * (*b_begin); */
/*         a_begin += a_stride; */
/*         b_begin += b_stride; */
/*     } */
/*     return final_product; */
/* } */


/* // Define struct template for current inner product */
/* template<typename T> */
/* struct CURRENT_INNER_STRIDED_PRODUCT { */
/*     static constexpr DotProductStridedFunc<T> function = &inner_product_default<T>; */
/* }; */




/* #elif defined(__AVX2__) */

/* #include "AVX2_DOT_PRODUCT.hpp" */

/* #elif defined(__AVX__) */



/* // Function pointer type for dot product function */
/* template<typename T> */
/* using DotProductFunc = T(*)(const T*, const T*, const T*, T); */

/* // Default implementation for other types */
/* template<typename T> */
/* T inner_product_default(const T* a, const T* b, const T* a_end, T final_sum) { */
/*     return std::inner_product(a, a_end, b, final_sum); */
/* } */


/* inline float float_dot_product(const float* a, const float* b, const float* a_end, float final_sum) { */
/*     __m256 sum = _mm256_setzero_ps(); */
/*     while ((a+8) < a_end) { */
/*         __m256 va = _mm256_loadu_ps(a); */
/*         __m256 vb = _mm256_loadu_ps(b); */
/*         sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb)); */
/* 	a += 8; */
/* 	b += 8; */
/*     } */
/*     alignas(32) float result[8]; */
/*     _mm256_store_ps(result, sum); */
/*     for (int i = 0; i < 8; ++i) { */
/*         final_sum += result[i]; */
/*     } */
/*     return std::inner_product(a, a_end, b, final_sum); */
/*     /1* final_sum += std::inner_product(a, a_end, b, 0); *1/ */
/* } */

/* inline double double_dot_product(const double* a, const double* b, const double* a_end, double final_sum) { */
/*     __m256d sum = _mm256_setzero_pd(); */
/*     while ((a + 4) < a_end) { */
/*         __m256d va = _mm256_loadu_pd(a); */
/*         __m256d vb = _mm256_loadu_pd(b); */
/*         sum = _mm256_add_pd(sum, _mm256_mul_pd(va, vb)); */
/*         a += 4; */
/*         b += 4; */
/*     } */
/*     alignas(32) double result[4]; */
/*     _mm256_store_pd(result, sum); */
/*     for (int i = 0; i < 4; ++i) { */
/*         final_sum += result[i]; */
/*     } */
/*     return std::inner_product(a, a_end, b, final_sum); */
/* } */

/* // Define struct template for current inner product */
/* template<typename T> */
/* struct CURRENT_INNER_PRODUCT { */
/*     static constexpr DotProductFunc<T> function = &inner_product_default<T>; */
/* }; */

/* // Specializations for float and double */
/* template<> */
/* constexpr DotProductFunc<float> CURRENT_INNER_PRODUCT<float>::function = &float_dot_product; */

/* template<> */
/* constexpr DotProductFunc<double> CURRENT_INNER_PRODUCT<double>::function = &double_dot_product; */


/* // Function pointer type for dot product function */
/* template<typename T> */
/* using DotProductStridedFunc = T(*)(const T*, const T*, const T*, T, const int64_t, const int64_t); */


/* template<typename T> */
/* inline T dot_product_strided(const T* a_begin, const T* b_begin, const T* a_end, T final_product const int64_t a_stride, const int64_t b_stride){ */
/*     while (a_begin < a_end) { */
/*         final_product += (*a_begin) * (*b_begin); */
/*         a_begin += a_stride; */
/*         b_begin += b_stride; */
/*     } */
/*     return final_product; */
/* } */


/* // Define struct template for current inner product */
/* template<typename T> */
/* struct CURRENT_INNER_STRIDED_PRODUCT { */
/*     static constexpr DotProductStridedFunc<T> function = &inner_product_default<T>; */
/* }; */



/* #else */

template<typename T>
struct CURRENT_INNER_PRODUCT{
	static constexpr auto function = std::inner_product<const T*, const T*, T>;
};

// Function pointer type for dot product function
template<typename T>
using DotProductStridedFunc = T(*)(const T*, const T*, T*, T, const int64_t, const int64_t);


template<typename T>
inline T dot_product_strided(const T* a_begin, const T* b_begin, const T* a_end, T final_product, const int64_t a_stride, const int64_t b_stride){
    while (a_begin < a_end) {
        final_product += (*a_begin) * (*b_begin);
        a_begin += a_stride;
        b_begin += b_stride;
    }
    return final_product;
}


// Define struct template for current inner product
template<typename T>
struct CURRENT_INNER_STRIDED_PRODUCT {
    static constexpr DotProductStridedFunc<T> function = &dot_product_strided<T>;
};



/* #endif */


template<typename T>
inline T dot_product(const T* begin, const T* begin_2, const T* end, T init){
	return CURRENT_INNER_PRODUCT<T>::function(begin, begin_2, end, init);
}


template<typename T>
inline T strided_dot_product(const T* begin, const T* begin_2, const T* end, T init, const int64_t a_stride, const int64_t b_stride){
	return CURRENT_INNER_STRIDED_PRODUCT<T>::function(begin, begin_2, end, init, a_stride, b_stride);
}




template<size_t N>
class block_ranges;




template<size_t N>
class blocked_range{
	public:
		std::array<int64_t, N> begin;
		std::array<int64_t, N> end;
	private:
	inline const int64_t block_size() const {
			int64_t size = 1;
			for(int64_t i = 0; i < N; ++i){size *= (pairs[i].second - pairs[i].first);}
			return size;
		}
	std::array<std::pair<int64_t, int64_t>, N> pairs;
	
	inline static std::array<int64_t, N> extract_begin(std::array<std::pair<int64_t, int64_t>, N>& pairs) noexcept{
		std::array<int64_t, N> outp;
		for(uint32_t i = 0; i < N; ++i){outp[i] = pairs[i].first;}
		return outp;
	}
	inline static std::array<int64_t, N> extract_end(std::array<std::pair<int64_t, int64_t>, N>& pairs) noexcept{
		std::array<int64_t, N> outp;
		for(uint32_t i = 0; i < N; ++i){outp[i] = pairs[i].second;}
		return outp;
	}


	blocked_range(std::array<std::pair<int64_t, int64_t>, N> arr)
		:begin(blocked_range<N>::extract_begin(arr)), end(blocked_range<N>::extract_end(arr)), pairs(arr), index(0), blockSize(block_size())
	{}
	blocked_range(std::array<std::pair<int64_t, int64_t>, N> arr, const int64_t block_num)
		:begin(blocked_range<N>::extract_begin(arr)), end(blocked_range<N>::extract_end(arr)), pairs(arr), index(block_num), blockSize(block_size())
	{}

	blocked_range(std::array<int64_t, N> b, std::array<int64_t, N> e, std::array<std::pair<int64_t, int64_t>, N> arr)
		:begin(b), end(e), pairs(arr), index(0), blockSize(block_size())
	{}
	blocked_range(std::array<int64_t, N> b, std::array<int64_t, N> e, std::array<std::pair<int64_t, int64_t>, N> arr, const int64_t block_num)
		:begin(b), end(e), pairs(arr), index(block_num), blockSize(block_size())
	{}


	template<typename T1, typename T2>
	inline static void make_pair(std::array<std::pair<int64_t, int64_t>, N>& arr, int64_t index, const T1& begin, const T2& end){
		static_assert(std::is_integral_v<T1> && std::is_integral_v<T2>, "When making ranges, expected to get only integral types for begin and end");
		arr[index] = std::pair<int64_t, int64_t>(begin, end);
	}

	template<typename T1, typename T2, typename... Args>
	inline static void make_pair(std::array<std::pair<int64_t, int64_t>, N>& arr, int64_t index, const T1& begin, const T2& end, const Args&... args){
		static_assert(std::is_integral_v<T1> && std::is_integral_v<T2>, "When making ranges, expected to get only integral types for begin and end");
		arr[index] = std::pair<int64_t, int64_t>(begin, end);
		make_pair(arr, index+1, args...);
	}

	template<typename T1, typename T2>
	inline static void make_bend(std::array<int64_t, N>& b, std::array<int64_t, N>& e, std::array<std::pair<int64_t, int64_t>, N>& arr,
			int64_t index, const T1& begin, const T2& end){
		static_assert(std::is_integral_v<T1> && std::is_integral_v<T2>, "When making ranges, expected to get only integral types for begin and end");
		b[index] = begin;
		e[index] = end;
		arr[index] = std::pair<int64_t, int64_t>(begin, end);
	}

	template<typename T1, typename T2, typename... Args>
	inline static void make_bend(std::array<int64_t, N>& b, std::array<int64_t, N>& e, std::array<std::pair<int64_t, int64_t>, N>& arr, 
			int64_t index, const T1& begin, const T2& end, const Args&... args){
		static_assert(std::is_integral_v<T1> && std::is_integral_v<T2>, "When making ranges, expected to get only integral types for begin and end");
		b[index] = begin;
		e[index] = end;
		arr[index] = std::pair<int64_t, int64_t>(begin, end);
		make_pair(b, e, arr, index+1, args...);
	}

	

	friend class block_ranges<N>;
	public:
		
		int64_t index;
		int64_t blockSize;
		blocked_range() = default;


		/* template<size_t M> */
		/* inline const int64_t& begin() const { */
		/*         static_assert(M < N, "Expected to have M < N for begin() for blocked_range"); */
		/*         return pairs[M].first; */
		/* } */

		/* template<size_t M> */
		/* inline const int64_t& end() const { */
		/*         static_assert(M < N, "Expected to have M < N for begin() for blocked_range"); */
		/*         return pairs[M].second; */
		/* } */

		/* inline constexpr int64_t begin_i(const size_t M) const { */
		/*         /1* static_assert(M < N, "Expected to have M < N for begin() for blocked_range"); *1/ */
		/* 	return pairs[M].first; */
		/* } */
		/* template<size_t M> */
		/* inline constexpr int64_t end_i() const { */
		/*         static_assert(M < N, "Expected to have M < N for begin() for blocked_range"); */
		/*         return pairs[M].second; */
		/* } */
		inline static blocked_range<1> make_range(const tbb::blocked_range<int64_t>& r){
			return blocked_range<1>(std::array<std::pair<int64_t, int64_t>, 1>({
						std::pair<int64_t, int64_t>(r.begin(), r.end())
						}));
		}
		inline static blocked_range<2> make_range(const tbb::blocked_range2d<int64_t>& r){
			return blocked_range<2>(std::array<std::pair<int64_t, int64_t>, 2>({
						std::pair<int64_t, int64_t>(r.rows().begin(), r.rows().end())
						, std::pair<int64_t, int64_t>(r.cols().begin(), r.cols().end())}));
		}
		inline static blocked_range<3> make_range(const tbb::blocked_range3d<int64_t>& r){
			return blocked_range<3>(std::array<std::pair<int64_t, int64_t>, 3>({
						std::pair<int64_t, int64_t>(r.pages().begin(), r.pages().end()),
						std::pair<int64_t, int64_t>(r.rows().begin(), r.rows().end()), 
						std::pair<int64_t, int64_t>(r.cols().begin(), r.cols().end())}));
		}
		template<typename... Args>
		inline static blocked_range<sizeof...(Args) / 2> make_range(Args... args){
			static_assert(sizeof...(Args) % 2 == 0, "Expected to get an even number of ranges for make_range");
			constexpr size_t my_n = sizeof...(Args) / 2;
			std::array<std::pair<int64_t, int64_t>, my_n> arr;
			std::array<int64_t, my_n> b, e;
			blocked_range<my_n>::make_bend(b, e, arr, 0, args...);
			return blocked_range<my_n>(std::move(b), std::move(e), std::move(arr));
		}

};


template<size_t N>
inline std::ostream& operator << (std::ostream& os, const blocked_range<N>& b){
	os << "(";
	for(int64_t i = 0; i < N-1; ++i){
		os << '{'<<b.begin_i(i)<<','<<b.end_i(i)<<"} ";
	}
	os << '{'<<b.begin_i(N-1)<<','<<b.end_i(N-1)<<"}, Index: "<<b.index<<", Block Size: "<<b.blockSize<<")";
	return os;
}

//by default, this breaks the ranges into the max number of threads per processes blocks
//this can then be used for my own version of the tbb::parallel_for
//this will probably replace the tbb::parallel_for in most of its implementations
//unless tbb::parallel_for is faster, then this will only be used for child processes made with fork()
template<size_t N>
class block_ranges{
	std::vector<blocked_range<N>> blocks;
	std::array<std::pair<int64_t, int64_t>, N> pairs;
	static_assert(N > 0, "Must have an N greater than 0 for block_ranges");

	template<typename Arg1, typename Arg2, typename... Args>
	inline void initializeHelper(Arg1 arg1, Arg2 arg2, Args... args){
		pairs[N - sizeof...(Args) / 2 - 1] = std::make_pair(static_cast<int64_t>(arg1), static_cast<int64_t>(arg2));
		if constexpr (sizeof...(Args) > 0) {
			initializeHelper(args...);
		}
	}

	
	const int64_t total_nums() const;
	const bool at_end(const std::array<std::pair<int64_t, int64_t>, N>& b) const;
	void split_all_by_one(const int64_t& nums);

	void add_n(std::array<std::pair<int64_t, int64_t>, N> &block, const int64_t &n, int64_t current_index, const int64_t &restart_st);

	void split_all_by_n(const int64_t& nums, const int64_t& n);

	void first_split(const int64_t& num_blocks);

	public:
		block_ranges() = delete;

		template<typename... Args>
		block_ranges(Args... args)
		{
			static_assert(sizeof...(args) == N * 2, "Wrong number of arguments for blocked_range<N>, expected N * 2");
			initializeHelper(std::forward<Args>(args)...);
			/* for(int64_t i = 0; i < N; ++i){ */
			/* 	std::cout << */ 
			/* } */
			
			for(int64_t i = 0; i < N; ++i){
				if(pairs[i].second < pairs[i].first)
					std::swap(pairs[i].second, pairs[i].first);
				utils::throw_exception(pairs[i].second != pairs[i].first, "cannot make blocks when a pair is {$,$}", pairs[i].first, pairs[i].second);
			}
/* #ifdef USE_PARALLEL */
/* 			perform_block(); */
/* #else */
/* 			perform_one_block(); */
/* #endif */

		}
		
		inline void perform_one_block() {
			blocks.clear();
			blocks.push_back(blocked_range<N>(pairs));
		}
		void perform_block();
		inline const std::vector<blocked_range<N> >& getBlocks() const {return blocks;}
		inline const std::array<std::pair<int64_t, int64_t>, N>& getPairs() const {return pairs;}


};

template<size_t N>
inline std::ostream& operator << (std::ostream& os, const block_ranges<N>& b){
	const std::vector<blocked_range<N>>& blocks = b.getBlocks();
	for(auto begin  = blocks.cbegin(); begin < blocks.cend(); ++begin){
		os << *begin << std::endl;
	}
	return os;
}



#ifndef USE_PARALLEL
template<size_t N, typename UnaryFunction>
inline void preferential_parallel_for(block_ranges<N>&& b, UnaryFunction&& inFunc){
	static_assert(std::is_invocable_v<UnaryFunction, const blocked_range<N>&>,
                  "Expected input function to take blocked_range argument");
	
	b.peform_one_block();
	const auto& blocks = b.getBlocks();
	inFunc(blocks[0]);
}
#else
template<size_t N, typename UnaryFunction>
inline void preferential_parallel_for(block_ranges<N>&& b, UnaryFunction&& inFunc){
	static_assert(std::is_invocable_v<UnaryFunction, const blocked_range<N>&>,
                  "Expected input function to take blocked_range argument");


	if constexpr (N == 1){
		const std::array<std::pair<int64_t, int64_t>, 1> range = b.getPairs();
		tbb::parallel_for(tbb::blocked_range<int64_t>(range[0].first, range[0].second),
				[&inFunc](const tbb::blocked_range<int64_t>& r){
					inFunc(blocked_range<1>::make_range(r));
				});
	}
	else if constexpr (N == 2){
		const std::array<std::pair<int64_t, int64_t>, 2> range = b.getPairs();
		tbb::parallel_for(tbb::blocked_range2d<int64_t>(range[0].first, range[0].second, range[1].first, range[1].second),
				[&inFunc](const tbb::blocked_range2d<int64_t>& r){
					inFunc(blocked_range<2>::make_range(r));
				});
	
	}
	else if constexpr (N == 3){
		const std::array<std::pair<int64_t, int64_t>, 3> range = b.getPairs();
		tbb::parallel_for(tbb::blocked_range3d<int64_t>(range[0].first, range[0].second, range[1].first, range[1].second, range[2].first, range[2].second),
				[&inFunc](const tbb::blocked_range3d<int64_t>& r){
					inFunc(blocked_range<3>::make_range(r));
				});
	
	}
	else{
		b.perform_block();
		const uint32_t n_threads = b.getBlocks().size();
		auto& blocks = b.getBlocks();

		tbb::parallel_for(tbb::blocked_range<size_t>(0, n_threads),
			      [&blocks, &inFunc](const tbb::blocked_range<size_t>& range) {
				  for (size_t i = range.begin(); i != range.end(); ++i) {
				      inFunc(blocks[i]);
				  }
			      });
	}

}


#endif


#ifdef USE_PARALLEL
template<size_t N, typename UnaryFunction>
inline void my_tbb_parallel_for(block_ranges<N> b, UnaryFunction&& inFunc) {
    static_assert(std::is_invocable_v<UnaryFunction, const blocked_range<N>&>,
                  "Expected input function to take blocked_range argument");

	if constexpr (N == 1){
		const std::array<std::pair<int64_t, int64_t>, 1> range = b.getPairs();
		tbb::parallel_for(tbb::blocked_range<int64_t>(range[0].first, range[0].second),
				[&inFunc](const tbb::blocked_range<int64_t>& r){
					inFunc(blocked_range<1>::make_range(r));
				});
	}
	else if (N == 2){
		const std::array<std::pair<int64_t, int64_t>, 2> range = b.getPairs();
		tbb::parallel_for(tbb::blocked_range2d<int64_t>(range[0].first, range[0].second, range[1].first, range[1].second),
				[&inFunc](const tbb::blocked_range2d<int64_t>& r){
					inFunc(blocked_range<2>::make_range(r));
				});
	
	}
	else if (N == 3){
		const std::array<std::pair<int64_t, int64_t>, 3> range = b.getPairs();
		tbb::parallel_for(tbb::blocked_range3d<int64_t>(range[0].first, range[0].second, range[1].first, range[1].second, range[2].first, range[2].second),
				[&inFunc](const tbb::blocked_range3d<int64_t>& r){
					inFunc(blocked_range<3>::make_range(r));
				});
	
	}
	else{
		b.perform_block();
		const uint32_t n_threads = b.getBlocks().size();
		auto& blocks = b.getBlocks();

		tbb::parallel_for(tbb::blocked_range<size_t>(0, n_threads),
			      [&blocks, &inFunc](const tbb::blocked_range<size_t>& range) {
				  for (size_t i = range.begin(); i != range.end(); ++i) {
				      inFunc(blocks[i]);
				  }
			      });
	}

}

template<size_t N, typename UnaryFunction>
inline void parallel_for(block_ranges<N> b, UnaryFunction&& inFunc){

	static_assert(std::is_invocable_v<UnaryFunction, const blocked_range<N>&>,
			"Expected input function to take blocked_range argument");
	
	b.perform_block();
	const uint32_t n_threads = b.getBlocks().size();

	std::thread threads[n_threads];
	for(uint32_t i = 0; i < n_threads; ++i){
		threads[i] = std::thread(std::forward<UnaryFunction&&>(inFunc), b.getBlocks()[i]);
	}

	//join threads
	for(uint32_t i = 0; i < n_threads; ++i)
		threads[i].join();

}


#endif



/*
in_func example:
[](const auto& range, t* begin){
	for(int64_t i = range.begin(); i < range.end(); ++i, begin += stride)
		*begin += 10;
}
 */
}

}

#include "iterator_parallel_for.hpp"


#endif // _NT_THREADING_H_
