#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include "../../mp/Threading.h"
#include <stdexcept>
#include <algorithm>


namespace nt{
namespace functional{
namespace cpu{

template<typename DualFunc>
inline void run_dual_compare(bool* o_begin, const ArrayVoid& a, const ArrayVoid& b, DualFunc&& func){
    a.cexecute_function<WRAP_DTYPES<NumberTypesL> >(
        [&o_begin, func](auto begin, auto end, auto begin2){
            for(;begin != end; ++begin, ++begin2, ++o_begin){
                *o_begin = func(*begin, *begin2);
            }
            // threading::preferential_parallel_for(
            //     threading::block_ranges<1>(0, end-begin),
            //     [&](threading::blocked_range<1> block) {
            //         auto cur_begin = begin + block.begin[0];
            //         auto cur_begin2 = begin2 + block.begin[0];
            //         for (int64_t i = block.begin[0]; i < block.end[0]; ++i, ++cur_begin, ++cur_begin2){
            //             o_begin[i] = func(*cur_begin, *cur_begin2);
            //         }
            //     }
            // );
        },
    b);
}

inline void check_array_voids(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b){
    if(out.dtype != DType::Bool){
        throw std::logic_error("Expected out array void to be bools");
    }
    if(a.dtype != b.dtype){
        throw std::logic_error("Expected a and b array voids to be the same dtype");
    }
    if(out.Size() != a.Size() || out.Size() != b.Size()){
        throw std::length_error("Expected all array voids to be the same size");
    }
    if(!out.is_contiguous()){
        throw std::logic_error("Expected out array void to be contiguous");
    }
}

void _equal(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b){
    check_array_voids(out, a, b);
    run_dual_compare(
        reinterpret_cast<bool*>(out.data_ptr()),
        a, b,
        [](auto& s_a, auto& s_b){return s_a == s_b;}
    );
}
void _not_equal(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b){
    check_array_voids(out, a, b);
    run_dual_compare(
        reinterpret_cast<bool*>(out.data_ptr()),
        a, b,
        [](auto& s_a, auto& s_b){return s_a != s_b;}
    );
}
void _less_than(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b){
    check_array_voids(out, a, b);
    run_dual_compare(
        reinterpret_cast<bool*>(out.data_ptr()),
        a, b,
        [](auto& s_a, auto& s_b){return s_a < s_b;}
    );
}
void _greater_than(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b){
    check_array_voids(out, a, b);
    run_dual_compare(
        reinterpret_cast<bool*>(out.data_ptr()),
        a, b,
        [](auto& s_a, auto& s_b){return s_a > s_b;}
    );
}
void _less_than_equal(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b){
    check_array_voids(out, a, b);
    run_dual_compare(
        reinterpret_cast<bool*>(out.data_ptr()),
        a, b,
        [](auto& s_a, auto& s_b){return s_a <= s_b;}
    );
}
void _greater_than_equal(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b){
    check_array_voids(out, a, b);
    run_dual_compare(
        reinterpret_cast<bool*>(out.data_ptr()),
        a, b,
        [](auto& s_a, auto& s_b){return s_a >= s_b;}
    );
}

template<typename DualFunc>
inline void run_dual_compare(bool* o_begin, const ArrayVoid& a, Scalar& b, DualFunc&& func){
    a.cexecute_function<WRAP_DTYPES<NumberTypesL> >(
        [o_begin, func, &b](auto begin, auto end){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            value_t val = b.to<value_t>();
            threading::preferential_parallel_for(
                threading::block_ranges<1>(0, end-begin),
                [&](threading::blocked_range<1> block) {
                    auto cur_begin = begin + block.begin[0];
                    for (int64_t i = block.begin[0]; i < block.end[0]; ++i, ++cur_begin ){
                        o_begin[i] = func(*cur_begin, val);
                    }
                }
            );
        }
    );
}

inline void check_array_voids(ArrayVoid& out, const ArrayVoid& a, Scalar& b){
    if(out.dtype != DType::Bool){
        throw std::logic_error("Expected out array void to be bools");
    }
    if(out.Size() != a.Size()){
        throw std::length_error("Expected all array voids to be the same size");
    }
    if(!out.is_contiguous()){
        throw std::logic_error("Expected out array void to be contiguous");
    }
}

void _equal(ArrayVoid& out, const ArrayVoid& a, Scalar b){
    check_array_voids(out, a, b);
    run_dual_compare(
        reinterpret_cast<bool*>(out.data_ptr()),
        a, b,
        [](auto& s_a, auto& s_b){return s_a == s_b;}
    );
}
void _not_equal(ArrayVoid& out, const ArrayVoid& a, Scalar b){
    check_array_voids(out, a, b);
    run_dual_compare(
        reinterpret_cast<bool*>(out.data_ptr()),
        a, b,
        [](auto& s_a, auto& s_b){return s_a != s_b;}
    );
}
void _less_than(ArrayVoid& out, const ArrayVoid& a, Scalar b){
    check_array_voids(out, a, b);
    run_dual_compare(
        reinterpret_cast<bool*>(out.data_ptr()),
        a, b,
        [](auto& s_a, auto& s_b){return s_a < s_b;}
    );
}
void _greater_than(ArrayVoid& out, const ArrayVoid& a, Scalar b){
    check_array_voids(out, a, b);
    run_dual_compare(
        reinterpret_cast<bool*>(out.data_ptr()),
        a, b,
        [](auto& s_a, auto& s_b){return s_a > s_b;}
    );
}
void _less_than_equal(ArrayVoid& out, const ArrayVoid& a, Scalar b){
    check_array_voids(out, a, b);
    run_dual_compare(
        reinterpret_cast<bool*>(out.data_ptr()),
        a, b,
        [](auto& s_a, auto& s_b){return s_a <= s_b;}
    );
}
void _greater_than_equal(ArrayVoid& out, const ArrayVoid& a, Scalar b){
    check_array_voids(out, a, b);
    run_dual_compare(
        reinterpret_cast<bool*>(out.data_ptr()),
        a, b,
        [](auto& s_a, auto& s_b){return s_a >= s_b;}
    );
}

void _and_op(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b){
    check_array_voids(out, a, b);
    if(a.dtype != DType::Bool){
        throw std::logic_error("And operator runs on bools only");
    }
    bool* o_begin = reinterpret_cast<bool*>(out.data_ptr());
    a.cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Bool> > >(
        [o_begin](auto begin, auto end, auto begin2){
            threading::preferential_parallel_for(
                threading::block_ranges<1>(0, end-begin),
                [&](threading::blocked_range<1> block) {
                    auto cur_begin = begin + block.begin[0];
                    auto cur_begin2 = begin2 + block.begin[0];
                    for (int64_t i = block.begin[0]; i < block.end[0]; ++i, ++cur_begin, ++cur_begin2){
                        o_begin[i] = *cur_begin && *cur_begin2;
                    }
                }
            );
        },
    b);
}

void _or_op(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b){
    check_array_voids(out, a, b);
    if(a.dtype != DType::Bool){
        throw std::logic_error("or operator runs on bools only");
    }
    bool* o_begin = reinterpret_cast<bool*>(out.data_ptr());
    a.cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Bool> > >(
        [o_begin](auto begin, auto end, auto begin2){
            threading::preferential_parallel_for(
                threading::block_ranges<1>(0, end-begin),
                [&](threading::blocked_range<1> block) {
                    auto cur_begin = begin + block.begin[0];
                    auto cur_begin2 = begin2 + block.begin[0];
                    for (int64_t i = block.begin[0]; i < block.end[0]; ++i, ++cur_begin, ++cur_begin2){
                        o_begin[i] = *cur_begin || *cur_begin2;
                    }
                }
            );
        },
    b);
}


bool _all(const ArrayVoid& a){
    if(a.dtype != DType::Bool){
        throw std::logic_error("expected bools to detect all");
    }
    return a.cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Bool> > >([](auto begin, auto end){
			return std::all_of(begin, end, [](const uint_bool_t& v){return v.value == 1;});});
}


bool _none(const ArrayVoid& a){
    if(a.dtype != DType::Bool){
        throw std::logic_error("expected bools to detect all");
    }
    return a.cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Bool> > >([](auto begin, auto end){
			return std::none_of(begin, end, [](const uint_bool_t& v){return v.value == 1;});});
}

bool _any(const ArrayVoid& a){
    if(a.dtype != DType::Bool){
        throw std::logic_error("expected bools to detect all");
    }
    return a.cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Bool> > >([](auto begin, auto end){
			return std::any_of(begin, end, [](const uint_bool_t& v){return v.value == 1;});});
}

}
}
}
