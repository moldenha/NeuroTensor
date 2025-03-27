#include "normalize.h"
#include "../../dtype/ArrayVoid.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include "../../dtype/DType_enum.h"
#include "../../dtype/Scalar.h"
#include "../../refs/SizeRef.h"
#include <random>

namespace nt {
namespace functional {
namespace cpu {

void xavier_uniform_(ArrayVoid &output, double bound) {
    DType dt = output.dtype;
    std::random_device rd;
    std::minstd_rand gen(rd()); // minimal version
    if (DTypeFuncs::is_complex(dt)) {
        output.execute_function<WRAP_DTYPES<ComplexTypesL>>(
            [&bound, &gen](auto begin, auto end) {
                using complex_t = utils::IteratorBaseType_t<decltype(begin)>;
                using value_t = typename complex_t::value_type;
#ifdef _HALF_FLOAT_SUPPORT_
                if constexpr (std::is_same_v<value_t, float16_t>) {
                    std::uniform_real_distribution<float> dis((float)(-bound),
                                                              (float)(bound));
                    std::generate(begin, end, [&]() {
                        return complex_t(static_cast<value_t>(dis(gen)),
                                         static_cast<value_t>(dis(gen)));
                    });
                } else {
                    std::uniform_real_distribution<value_t> dis(
                        (value_t)(-bound), (value_t)(bound));
                    std::generate(begin, end, [&]() {
                        return complex_t(dis(gen), dis(gen));
                    });
                }
#else
                std::uniform_real_distribution<value_t> dis((value_t)(-bound),
                                                            (value_t)(bound));
                std::generate(begin, end,
                              [&]() { return complex_t(dis(gen), dis(gen)); });
#endif
            });
    } else if (DTypeFuncs::is_floating(dt)) {
        output.execute_function<WRAP_DTYPES<FloatingTypesL>>(
            [&bound, &gen](auto begin, auto end) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
#ifdef _HALF_FLOAT_SUPPORT_
                if constexpr (std::is_same_v<value_t, float16_t>) {
                    std::uniform_real_distribution<float> dis((float)(-bound),
                                                              (float)(bound));
                    std::generate(begin, end, [&]() {
                        return static_cast<value_t>(dis(gen));
                    });
                } else {
                    std::uniform_real_distribution<value_t> dis(
                        (value_t)(-bound), (value_t)(bound));
                    std::generate(begin, end, [&]() { return dis(gen); });
                }
#else
                std::uniform_real_distribution<value_t> dis((value_t)(-bound),
                                                            (value_t)(bound));
                std::generate(begin, end, [&]() { return dis(gen); });
#endif
            });
    }
}

} // namespace cpu
} // namespace functional
} // namespace nt
