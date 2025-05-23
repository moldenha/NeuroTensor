#include "rand.h"
#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"
#include "../../dtype/DType.h"
#include "../../refs/SizeRef.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include <random>

namespace nt{
namespace functional{
namespace cpu{

void randint_(ArrayVoid& output, Scalar upper, Scalar lower){
    std::random_device rd;
	std::minstd_rand gen(rd()); //minimal version
	DType dt = output.dtype;
    if(DTypeFuncs::is_unsigned(dt) || DTypeFuncs::is_integer(dt)){
		output.execute_function<WRAP_DTYPES<IntegerTypesL>>(
			[&upper, &lower, &gen](auto begin, auto end){
				using value_t = utils::IteratorBaseType_t<decltype(begin)>;
#ifdef __SIZEOF_INT128__
				if constexpr(std::is_same_v<value_t, uint128_t>){
					uint64_t low = lower.to<int64_t>();
					uint64_t up = upper.to<int64_t>();
					std::uniform_int_distribution<uint64_t> dis(low, up);
					std::generate(begin, end, [&]() { return static_cast<value_t>(dis(gen)); });
				}
				else if(std::is_same_v<value_t, int128_t>){
					int64_t low = lower.to<int64_t>();
					int64_t up = upper.to<int64_t>();
					std::uniform_int_distribution<int64_t> dis(low, up);
					std::generate(begin, end, [&]() { return static_cast<value_t>(dis(gen)); });
				}
				else{
					value_t low = lower.to<value_t>();
					value_t up = upper.to<value_t>();
					std::uniform_int_distribution<value_t> dis(low, up);
					std::generate(begin, end, [&]() { return dis(gen); });
				}
#else
				value_t low = lower.to<value_t>();
				value_t up = upper.to<value_t>();
				std::uniform_int_distribution<value_t> dis(low, up);
				std::generate(begin, end, [&]() { return dis(gen); });
#endif
			});
	}
	else if(DTypeFuncs::is_complex(dt)){
		output.execute_function<WRAP_DTYPES<ComplexTypesL> >(
		[&upper, &lower, &gen](auto begin, auto end){
			using complex_t = utils::IteratorBaseType_t<decltype(begin)>;
			using value_t = typename complex_t::value_type;
#ifdef _HALF_FLOAT_SUPPORT_
			if constexpr (std::is_same_v<value_t, float16_t>){
				float low = lower.to<float>();
				float up = upper.to<float>();
				std::uniform_real_distribution<float> dis(low, up);
				std::generate(begin, end, [&]() {return complex_t(static_cast<value_t>(std::round(dis(gen))), static_cast<value_t>(std::round(dis(gen))));});
			}
			else{
				value_t low = lower.to<value_t>();
				value_t up = upper.to<value_t>();
				std::uniform_real_distribution<value_t> dis(low, up);
				std::generate(begin, end, [&]() {return complex_t(std::round(dis(gen)), std::round(dis(gen)));}); 
			}
#else
			value_t low = lower.to<value_t>();
			value_t up = upper.to<value_t>();
			std::uniform_real_distribution<value_t> dis(low, up);
			std::generate(begin, end, [&]() {return complex_t(std::round(dis(gen)), std::round(dis(gen)));});
#endif

		});
	}
	else if(DTypeFuncs::is_floating(dt)){
		output.execute_function<WRAP_DTYPES<FloatingTypesL> >(
		[&upper, &lower, &gen](auto begin, auto end){
			using value_t = utils::IteratorBaseType_t<decltype(begin)>;
#ifdef _HALF_FLOAT_SUPPORT_
			if constexpr (std::is_same_v<value_t, float16_t>){
				float low = lower.to<float>();
				float up = upper.to<float>();
				std::uniform_real_distribution<float> dis(low, up);
				std::generate(begin, end, [&]() {return static_cast<value_t>(std::round(dis(gen)));});
			}
			else{
				value_t low = lower.to<value_t>();
				value_t up = upper.to<value_t>();
				std::uniform_real_distribution<value_t> dis(low, up);
				std::generate(begin, end, [&]() {return std::round(dis(gen));}); 
			}
#else
			value_t low = lower.to<value_t>();
			value_t up = upper.to<value_t>();
			std::uniform_real_distribution<value_t> dis(low, up);
			std::generate(begin, end, [&]() {return std::round(dis(gen));});
#endif
		});
	}

}

void rand_(ArrayVoid& output, Scalar upper, Scalar lower){
    std::random_device rd;
	std::minstd_rand gen(rd()); //minimal version
	DType dt = output.dtype;
if(DTypeFuncs::is_unsigned(dt) || DTypeFuncs::is_integer(dt)){
		output.execute_function<WRAP_DTYPES<IntegerTypesL>>(
			[&upper, &lower, &gen](auto begin, auto end){
				using value_t = utils::IteratorBaseType_t<decltype(begin)>;
#ifdef __SIZEOF_INT128__
				if constexpr(std::is_same_v<value_t, uint128_t>){
					uint64_t low = lower.to<int64_t>();
					uint64_t up = upper.to<int64_t>();
					std::uniform_int_distribution<uint64_t> dis(low, up);
					std::generate(begin, end, [&]() { return static_cast<value_t>(dis(gen)); });
				}
				else if(std::is_same_v<value_t, int128_t>){
					int64_t low = lower.to<int64_t>();
					int64_t up = upper.to<int64_t>();
					std::uniform_int_distribution<int64_t> dis(low, up);
					std::generate(begin, end, [&]() { return static_cast<value_t>(dis(gen)); });
				}
				else{
					value_t low = lower.to<value_t>();
					value_t up = upper.to<value_t>();
					std::uniform_int_distribution<value_t> dis(low, up);
					std::generate(begin, end, [&]() { return dis(gen); });
				}
#else
				value_t low = lower.to<value_t>();
				value_t up = upper.to<value_t>();
				std::uniform_int_distribution<value_t> dis(low, up);
				std::generate(begin, end, [&]() { return dis(gen); });
#endif
			});
	}
	else if(DTypeFuncs::is_complex(dt)){
		output.execute_function<WRAP_DTYPES<ComplexTypesL> >(
		[&upper, &lower, &gen](auto begin, auto end){
			using complex_t = utils::IteratorBaseType_t<decltype(begin)>;
			using value_t = typename complex_t::value_type;
#ifdef _HALF_FLOAT_SUPPORT_
			if constexpr (std::is_same_v<value_t, float16_t>){
				float low = lower.to<float>();
				float up = upper.to<float>();
				std::uniform_real_distribution<float> dis(low, up);
				std::generate(begin, end, [&]() {return complex_t(static_cast<value_t>(dis(gen)), static_cast<value_t>(dis(gen)));});
			}
			else{
				value_t low = lower.to<value_t>();
				value_t up = upper.to<value_t>();
				std::uniform_real_distribution<value_t> dis(low, up);
				std::generate(begin, end, [&]() {return complex_t(dis(gen), dis(gen));}); 
			}
#else
			value_t low = lower.to<value_t>();
			value_t up = upper.to<value_t>();
			std::uniform_real_distribution<value_t> dis(low, up);
			std::generate(begin, end, [&]() {return complex_t(dis(gen), dis(gen));});
#endif

		});
	}
	else if(DTypeFuncs::is_floating(dt)){
		output.execute_function<WRAP_DTYPES<FloatingTypesL> >(
		[&upper, &lower, &gen](auto begin, auto end){
			using value_t = utils::IteratorBaseType_t<decltype(begin)>;
#ifdef _HALF_FLOAT_SUPPORT_
			if constexpr (std::is_same_v<value_t, float16_t>){
				float low = lower.to<float>();
				float up = upper.to<float>();
				std::uniform_real_distribution<float> dis(low, up);
				std::generate(begin, end, [&]() {return static_cast<value_t>(dis(gen));});
			}
			else{
				value_t low = lower.to<value_t>();
				value_t up = upper.to<value_t>();
				std::uniform_real_distribution<value_t> dis(low, up);
				std::generate(begin, end, [&]() {return dis(gen);}); 
			}
#else
			value_t low = lower.to<value_t>();
			value_t up = upper.to<value_t>();
			std::uniform_real_distribution<value_t> dis(low, up);
			std::generate(begin, end, [&]() {return dis(gen);});
#endif
		});
	}
}



}
}
}
