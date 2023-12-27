#include <iostream>

#ifdef __has_keyword

	#if __has_keyword(_Float16)
	#define _HALF_FLOAT_SUPPORT_
		using float16_t = _Float16;

	#elif __has_keyword(__fp16)
	#define _HALF_FLOAT_SUPPORT_
		using float16_t = __fp16;
	#else
	#define _NO_HALF_FLOAT_SUPPORT_
	#endif
#else
	#if defined(__STDC_IEC_559__) && defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
	// C11 standard with IEC 559 (floating-point arithmetic) support
	// // You can use _Float16 here
	#define _HALF_FLOAT_SUPPORT_
		using float16_t = _Float16;
	#elif defined(__GNUC__) && defined(__FP16__)
	// GCC with support for __fp16
	// You can use __fp16 here
	#define _HALF_FLOAT_SUPPORT_
		using float16_t = __fp16;
	#elif defined(_MSC_VER) && defined(_M_AMD64) && defined(_MSC_VER) && (_MSC_VER >= 1920)
	// Visual Studio 2019 and later with AMD64 architecture
	// You can use _Float16 here
	#define _HALF_FLOAT_SUPPORT_
		using float16_t = _Float16;
	
	#else
	// Handle the case when _Float16 is not supported
	#define _NO_HALF_FLOAT_SUPPORT_
	#endif
#endif


#if defined(__SIZEOF_LONG_DOUBLE__) && __SIZEOF_LONG_DOUBLE__ == 16
#define _128_FLOAT_SUPPORT_
using float128_t = long double;
#elif defined(__has_keyword)
	#if __has_keyword(__float128)
	#define _128_FLOAT_SUPPORT_
		using float128_t = __float128;
	#elif __has_keyword(__fp128)
	#define _128_FLOAT_SUPPORT_
		using float128_t = __fp128;
	#else
	#define _NO_128_SUPPORT_
	#endif
#else

	#if defined(__STDC_IEC_559__) && defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
		// C11 standard with IEC 559 (floating-point arithmetic) support
		// You can use __float128 here
		#define __128_FLOAT_SUPPORT_
		using float128_t = __float128;
	#elif defined(__GNUC__) && defined(__SIZEOF_FLOAT128__)
		// GCC with support for __float128
		// You can use __float128 here
		#define __128_FLOAT_SUPPORT_
		using float128_t = __float128;
	#elif defined(_MSC_VER) && defined(_M_AMD64) && defined(_MSC_VER) && (_MSC_VER >= 1920)
		// Visual Studio 2019 and later with AMD64 architecture
		// You can use __float128 here
		#define __128_FLOAT_SUPPORT_
		using float128_t = __float128;
	#elif defined(__clang__) && defined(__SIZEOF_FLOAT128__)
		// Clang with support for __float128
		// You can use __float128 here
		#define __128_FLOAT_SUPPORT_
		using float128_t = __float128;
	#else
		#define _NO_128_SUPPORT_
		// Handle the case when __float128 is not supported
	#endif
#endif

#ifdef __SIZEOF_INT128__
using uint128_t = __uint128_t;
using int128_t = __int128_t;
#endif


int main(){
	std::cout << "finding compatible scalars...."<<std::endl;
#ifdef __SIZEOF_INT128__
	std::cout << "int128 is compatible"<<std::endl;
#elif !defined(__SIZEOF_INT128__)
	std::cout << "int128 is not compatible"<<std::endl;
#else
	std::cout << "error finding if int128 is compatible"<<std::endl;
#endif
#ifdef __128_FLOAT_SUPPORT_
	std::cout << "float128 is compatible" <<std::endl;
#elif defined(_NO_128_SUPPORT_)
	std::cout << "float128 is not compatible"<<std::endl;
#else
	std::cout << "error finding if float128 is compatible"<<std::endl;
#endif
#ifdef _HALF_FLOAT_SUPPORT_
	std::cout << "float16 is compatible"<<std::endl;
#elif defined(_NO_HALF_FLOAT_SUPPORT_)
	std::cout << "half float is not compatible"<<std::endl;
#else
	std::cout << "trouble seeing if half float is compatible"<<std::endl;
#endif
}
