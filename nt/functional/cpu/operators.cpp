#include "operators.h"
// tensors need to be included in this one
#include "../../mp/simde_traits.h"
#include "../../mp/simde_traits/simde_traits_iterators.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include <algorithm>

namespace nt {
namespace mp {

#define _NT_SIMDE_OP_TRANSFORM_EQUIVALENT_TWO_(func_name, simde_op,            \
                                               transform_op)                   \
    template <typename T, typename U, typename O>                              \
    inline void func_name(T begin, T end, U begin2, O out) {                   \
        static_assert(                                                         \
            std::is_same_v<utils::IteratorBaseType_t<T>,                       \
                           utils::IteratorBaseType_t<U>> &&                    \
                std::is_same_v<utils::IteratorBaseType_t<T>,                   \
                               utils::IteratorBaseType_t<O>>,                  \
            "Expected to get base types the same for simde optimized routes"); \
        using base_type = utils::IteratorBaseType_t<T>;                        \
        if constexpr (simde_supported_v<base_type>) {                          \
            static constexpr size_t pack_size = pack_size_v<base_type>;        \
            for (; begin + pack_size <= end;                                   \
                 begin += pack_size, begin2 += pack_size, out += pack_size) {  \
                simde_type<base_type> a = it_loadu(begin);                     \
                simde_type<base_type> b = it_loadu(begin2);                    \
                simde_type<base_type> c =                                      \
                    SimdTraits<base_type>::simde_op(a, b);                     \
                it_storeu(out, c);                                             \
            }                                                                  \
            std::transform(begin, end, begin2, out, transform_op<>{});         \
        } else {                                                               \
            std::transform(begin, end, begin2, out, transform_op<>{});         \
        }                                                                      \
    }

_NT_SIMDE_OP_TRANSFORM_EQUIVALENT_TWO_(add, add, std::plus);
_NT_SIMDE_OP_TRANSFORM_EQUIVALENT_TWO_(subtract, subtract, std::minus);
_NT_SIMDE_OP_TRANSFORM_EQUIVALENT_TWO_(multiply, multiply, std::multiplies);
_NT_SIMDE_OP_TRANSFORM_EQUIVALENT_TWO_(divide, divide, std::divides);

#undef _NT_SIMDE_OP_TRANSFORM_EQUIVALENT_TWO_

} // namespace mp
} // namespace nt

namespace nt {
namespace functional {
namespace cpu {

// multiply divide subtract add
void operator_mdsa_(const ArrayVoid &a, const ArrayVoid &b, ArrayVoid &o,
                    int op) {
    if (op == 0) {
        // multiply
        a.cexecute_function<WRAP_DTYPES<NumberTypesL>>(
            [](auto begin, auto end, auto begin2, void *out_p) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                value_t *out = reinterpret_cast<value_t *>(out_p);
                mp::multiply(begin, end, begin2, out);
            },
            b, o.data_ptr());
    } else if (op == 1) {
        // subtract
        a.cexecute_function<WRAP_DTYPES<NumberTypesL>>(
            [](auto begin, auto end, auto begin2, void *out_p) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                value_t *out = reinterpret_cast<value_t *>(out_p);
                mp::subtract(begin, end, begin2, out);
            },
            b, o.data_ptr());
    } else if (op == 2) {
        // divide
        a.cexecute_function<WRAP_DTYPES<NumberTypesL>>(
            [](auto begin, auto end, auto begin2, void *out_p) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                value_t *out = reinterpret_cast<value_t *>(out_p);
                mp::divide(begin, end, begin2, out);
            },
            b, o.data_ptr());
    } else if (op == 3) {
        // add
        a.cexecute_function<WRAP_DTYPES<NumberTypesL>>(
            [](auto begin, auto end, auto begin2, void *out_p) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                value_t *out = reinterpret_cast<value_t *>(out_p);
                mp::add(begin, end, begin2, out);
            },
            b, o.data_ptr());
    }
}

// multiply divide subtract add
void operator_mdsa_(ArrayVoid &a, const ArrayVoid &b, int op) {
    if (op == 0) {
        // multiply
        a.execute_function<WRAP_DTYPES<NumberTypesL>>(
            [](auto begin, auto end, auto begin2) {
                mp::multiply(begin, end, begin2, begin);
            },
            const_cast<ArrayVoid &>(b));
    } else if (op == 1) {
        // subtract
        a.execute_function<WRAP_DTYPES<NumberTypesL>>(
            [](auto begin, auto end, auto begin2) {
                mp::subtract(begin, end, begin2, begin);
            },
            const_cast<ArrayVoid &>(b));
    } else if (op == 2) {
        // divide
        a.execute_function<WRAP_DTYPES<NumberTypesL>>(
            [](auto begin, auto end, auto begin2) {
                mp::divide(begin, end, begin2, begin);
            },
            const_cast<ArrayVoid &>(b));
    } else if (op == 3) {
        // add
        a.execute_function<WRAP_DTYPES<NumberTypesL>>(
            [](auto begin, auto end, auto begin2) {
                mp::add(begin, end, begin2, begin);
            },
            const_cast<ArrayVoid &>(b));
    }
}

} // namespace cpu
} // namespace functional
} // namespace nt
