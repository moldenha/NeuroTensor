#include "rand.h"
#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"
#include "../../dtype/DType.h"
#include "../../refs/SizeRef.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include "../../convert/Convert.h"
#include <random>

#include "../../types/float128.h"

//if this is defined
//this means that for float128_t boost's 128 bit floating point is used
#ifdef BOOST_MP_STANDALONE
namespace std{
inline ::nt::float128_t round(const ::nt::float128_t& x){
    ::nt::float128_t int_part = trunc(x);  // integer part (toward zero)
    ::nt::float128_t frac = x - int_part;  // fractional part

    if (x >= 0) {
        return frac < 0.5 ? int_part : int_part + 1;
    } else {
        return frac > -0.5 ? int_part : int_part - 1;
    }
}

}

#endif //BOOST_MP_STANDALONE

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
                if constexpr (std::is_same_v<value_t, ::nt::uint128_t>){
                    int64_t low = lower.to<int64_t>();
                    int64_t up = upper.to<int64_t>();
					std::uniform_int_distribution<int64_t> dis(low, up);
					std::generate(begin, end, [&]() { return convert::convert<::nt::uint128_t>(std::abs(dis(gen))); });
                }
                else if constexpr (std::is_same_v<int64_t, ::nt::int128_t>){
                    int64_t low = lower.to<int64_t>();
                    int64_t up = upper.to<int64_t>();
					std::uniform_int_distribution<int64_t> dis(low, up);
					std::generate(begin, end, [&]() { return convert::convert<::nt::int128_t>(static_cast<int64_t>(dis(gen))); });
                }
                else{
                    value_t low = lower.to<value_t>();
                    value_t up = upper.to<value_t>();
                    std::uniform_int_distribution<value_t> dis(low, up);
                    std::generate(begin, end, [&]() { return dis(gen); });
                }
			});
	}
	else if(DTypeFuncs::is_complex(dt)){
		output.execute_function<WRAP_DTYPES<ComplexTypesL> >(
		[&upper, &lower, &gen](auto begin, auto end){
			using complex_t = utils::IteratorBaseType_t<decltype(begin)>;
			using value_t = typename complex_t::value_type;
			if constexpr (std::is_same_v<value_t, float16_t>){
				float low = lower.to<float>();
				float up = upper.to<float>();
				std::uniform_real_distribution<float> dis(low, up);
				std::generate(begin, end, [&]() {return complex_t(_NT_FLOAT32_TO_FLOAT16_(std::round(dis(gen))), _NT_FLOAT32_TO_FLOAT16_(std::round(dis(gen))));});
			}
			else{
				value_t low = lower.to<value_t>();
				value_t up = upper.to<value_t>();
				std::uniform_real_distribution<value_t> dis(low, up);
				std::generate(begin, end, [&]() {return complex_t(std::round(dis(gen)), std::round(dis(gen)));}); 
			}

		});
	}
	else if(DTypeFuncs::is_floating(dt)){
		output.execute_function<WRAP_DTYPES<FloatingTypesL> >(
		[&upper, &lower, &gen](auto begin, auto end){
			using value_t = utils::IteratorBaseType_t<decltype(begin)>;
			if constexpr (std::is_same_v<value_t, float16_t>){
				float low = lower.to<float>();
				float up = upper.to<float>();
				std::uniform_real_distribution<float> dis(low, up);
				std::generate(begin, end, [&]() {return _NT_FLOAT32_TO_FLOAT16_(std::round(dis(gen)));});
			}
            else if constexpr(std::is_same_v<value_t, float128_t>){
				double low = lower.to<double>();
				double up = upper.to<double>();
				std::uniform_real_distribution<double> dis(low, up);
				std::generate(begin, end, [&]() {return convert::convert<float128_t>(std::round(dis(gen)));});
            }
			else{
				value_t low = lower.to<value_t>();
				value_t up = upper.to<value_t>();
				std::uniform_real_distribution<value_t> dis(low, up);
				std::generate(begin, end, [&]() {return std::round(dis(gen));}); 
			}
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
                if constexpr (std::is_same_v<value_t, ::nt::uint128_t>){
                    int64_t low = lower.to<int64_t>();
                    int64_t up = upper.to<int64_t>();
					std::uniform_int_distribution<int64_t> dis(low, up);
					std::generate(begin, end, [&]() { return convert::convert<::nt::uint128_t>(std::abs(dis(gen))); });
                }
                else if constexpr (std::is_same_v<int64_t, ::nt::int128_t>){
                    int64_t low = lower.to<int64_t>();
                    int64_t up = upper.to<int64_t>();
					std::uniform_int_distribution<int64_t> dis(low, up);
					std::generate(begin, end, [&]() { return convert::convert<::nt::int128_t>(static_cast<int64_t>(dis(gen))); });
                }
                else{
                    value_t low = lower.to<value_t>();
                    value_t up = upper.to<value_t>();
                    std::uniform_int_distribution<value_t> dis(low, up);
                    std::generate(begin, end, [&]() { return dis(gen); });
                }
			});
	}
	else if(DTypeFuncs::is_complex(dt)){
		output.execute_function<WRAP_DTYPES<ComplexTypesL> >(
		[&upper, &lower, &gen](auto begin, auto end){
			using complex_t = utils::IteratorBaseType_t<decltype(begin)>;
			using value_t = typename complex_t::value_type;
			if constexpr (std::is_same_v<value_t, float16_t>){
				float low = lower.to<float>();
				float up = upper.to<float>();
				std::uniform_real_distribution<float> dis(low, up);
				std::generate(begin, end, [&]() {return complex_t(_NT_FLOAT32_TO_FLOAT16_(dis(gen)), _NT_FLOAT32_TO_FLOAT16_(dis(gen)));});
			}
			else{
				value_t low = lower.to<value_t>();
				value_t up = upper.to<value_t>();
				std::uniform_real_distribution<value_t> dis(low, up);
				std::generate(begin, end, [&]() {return complex_t(dis(gen), dis(gen));}); 
			}

		});
	}
	else if(DTypeFuncs::is_floating(dt)){
		output.execute_function<WRAP_DTYPES<FloatingTypesL> >(
		[&upper, &lower, &gen](auto begin, auto end){
			using value_t = utils::IteratorBaseType_t<decltype(begin)>;
			if constexpr (std::is_same_v<value_t, float16_t>){
				float low = lower.to<float>();
				float up = upper.to<float>();
				std::uniform_real_distribution<float> dis(low, up);
				std::generate(begin, end, [&]() {return _NT_FLOAT32_TO_FLOAT16_(dis(gen));});
			}
            else if constexpr(std::is_same_v<value_t, float128_t>){
				double low = lower.to<double>();
				double up = upper.to<double>();
				std::uniform_real_distribution<double> dis(low, up);
				std::generate(begin, end, [&]() {return convert::convert<float128_t>(dis(gen));});
            }
			else{
				value_t low = lower.to<value_t>();
				value_t up = upper.to<value_t>();
				std::uniform_real_distribution<value_t> dis(low, up);
				std::generate(begin, end, [&]() {return dis(gen);}); 
			}
		});
	}
}



}
}
}
