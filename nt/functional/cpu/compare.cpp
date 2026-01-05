#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include "../../mp/Threading.h"
#include <stdexcept>
#include <algorithm>
#include <atomic>
#include "../../mp/simde_traits.h"
#include "../../mp/simde_traits/simde_traits_iterators.h"
#include "../../utils/type_traits.h"

namespace nt{
namespace mp{

template<typename T, typename T2>
inline void equal(T begin, T end, T2 begin2, bool* out){
    static_assert(::nt::type_traits::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<T2>>, "Expected to get base types the same for simde optimized routes [equal]");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		for(;begin + pack_size <= end; begin += pack_size, begin2 += pack_size, out += pack_size){
			simde_type<base_type> current_a = it_loadu(begin);
			simde_type<base_type> current_b = it_loadu(begin2);
            SimdTraits<base_type>::store_compare_equal(current_a, current_b, out);
		}
		for(;begin < end; ++begin, ++begin2, ++out)
			*out = (*begin == *begin2);
	}else{
        for(;begin != end; ++begin, ++begin2, ++out){
			*out = (*begin == *begin2);
        }
	}
}

template<typename T, typename T2>
inline void not_equal(T begin, T end, T2 begin2, bool* out){
    static_assert(::nt::type_traits::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<T2>>, "Expected to get base types the same for simde optimized routes [not equal]");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		for(;begin + pack_size <= end; begin += pack_size, begin2 += pack_size, out += pack_size){
			simde_type<base_type> current_a = it_loadu(begin);
			simde_type<base_type> current_b = it_loadu(begin2);
            SimdTraits<base_type>::store_compare_not_equal(current_a, current_b, out);
		}
		for(;begin < end; ++begin, ++begin2, ++out){
            *out = (*begin != *begin2);
        }
	}else{
        for(;begin != end; ++begin, ++begin2, ++out){
			*out = (*begin != *begin2);
        }
	}
}

template<typename T, typename T2>
inline void less_than_equal(T begin, T end, T2 begin2, bool* out){
    static_assert(::nt::type_traits::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<T2>>, "Expected to get base types the same for simde optimized routes [equal]");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		for(;begin + pack_size <= end; begin += pack_size, begin2 += pack_size, out += pack_size){
			simde_type<base_type> current_a = it_loadu(begin);
			simde_type<base_type> current_b = it_loadu(begin2);
            SimdTraits<base_type>::store_compare_less_than_equal(current_a, current_b, out);
		}
		for(;begin < end; ++begin, ++begin2, ++out)
			*out = (*begin <= *begin2);
	}else{
        for(;begin != end; ++begin, ++begin2, ++out){
			*out = (*begin <= *begin2);
        }
	}
}


}}

namespace nt{
namespace functional{
namespace cpu{





inline void check_array_voids(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b){
    if(out.dtype() != DType::Bool){
        throw std::logic_error("Expected out array void to be bools");
    }
    if(a.dtype() != b.dtype()){
        throw std::logic_error("Expected a and b array voids to be the same dtype");
    }
    if(out.Size() != a.Size() || out.Size() != b.Size()){
        throw std::length_error("Expected all array voids to be the same size");
    }
    if(!out.is_contiguous()){
        throw std::logic_error("Expected out array void to be contiguous");
    }
}

#define NT_DUAL_MP_COMPARE(func_name)\
    bool* o_begin = reinterpret_cast<bool*>(out.data_ptr());\
    a.cexecute_function<WRAP_DTYPES<NumberTypesL>>(\
        [&o_begin](auto begin, auto end, auto begin2){\
            threading::preferential_parallel_for(\
                threading::block_ranges<1>(0, end-begin),\
                [&](threading::blocked_range<1> block) {\
                    auto cur_begin = begin + block.begin[0];\
                    auto cur_end = begin + block.end[0];\
                    auto cur_begin2 = begin2 + block.begin[0];\
                    func_name(cur_begin, cur_end, cur_begin2, &o_begin[block.begin[0]]);\
                });\
            }, b);


#define NT_DUAL_MP_COMPARE_BOOL(func_name)\
    bool* o_begin = reinterpret_cast<bool*>(out.data_ptr());\
    a.cexecute_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool>>>(\
        [&o_begin](auto begin, auto end, auto begin2){\
            int64_t size = end-begin;\
            if(size < 200){func_name(begin, end, begin2, o_begin); return;}\
            threading::preferential_parallel_for(\
                threading::block_ranges<1>(0, end-begin),\
                [&](threading::blocked_range<1> block) {\
                    auto cur_begin = begin + block.begin[0];\
                    auto cur_end = begin + block.end[0];\
                    auto cur_begin2 = begin2 + block.begin[0];\
                    func_name(cur_begin, cur_end, cur_begin2, &o_begin[block.begin[0]]);\
                });\
            }, b);
// #define NT_DUAL_MP_COMPARE(func_name)\
//     a.cexecute_function<WRAP_DTYPES<NumberTypesL>>(\
//         [&out](auto begin, auto end, auto begin2){\
//             bool* o_begin = reinterpret_cast<bool*>(out.data_ptr());\
//             func_name(begin, end, begin2, o_begin);\
//         }, b);

void _equal(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b){
    check_array_voids(out, a, b);
    NT_DUAL_MP_COMPARE_BOOL(::nt::mp::equal)
}
void _not_equal(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b){
    check_array_voids(out, a, b);
    NT_DUAL_MP_COMPARE_BOOL(::nt::mp::not_equal)
}

template<typename DualFunc>
inline void run_dual_compare(bool* o_begin, const ArrayVoid& a, const ArrayVoid& b, DualFunc&& func){
    a.cexecute_function<WRAP_DTYPES<NumberTypesL> >(
        [&o_begin, func](auto begin, auto end, auto begin2){
            // for(;begin != end; ++begin, ++begin2, ++o_begin){
            //     *o_begin = func(*begin, *begin2);
            // }
            threading::preferential_parallel_for(
                threading::block_ranges<1>(0, end-begin),
                [&](threading::blocked_range<1> block) {
                    auto cur_begin = begin + block.begin[0];
                    auto cur_begin2 = begin2 + block.begin[0];
                    for (int64_t i = block.begin[0]; i < block.end[0]; ++i, ++cur_begin, ++cur_begin2){
                        o_begin[i] = func(*cur_begin, *cur_begin2);
                    }
                }
            );
        },
    b);
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
    NT_DUAL_MP_COMPARE(::nt::mp::less_than_equal)
}


void _greater_than_equal(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b){
    check_array_voids(out, a, b);
    run_dual_compare(
        reinterpret_cast<bool*>(out.data_ptr()),
        a, b,
        [](auto& s_a, auto& s_b){return s_a >= s_b;}
    );
}

#undef NT_DUAL_MP_COMPARE 
#undef NT_DUAL_MP_COMPARE_BOOL 

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

template<typename DualFunc>
inline void run_dual_compare_bool(bool* o_begin, const ArrayVoid& a, Scalar& b, DualFunc&& func){
    a.cexecute_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool>> >(
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
    if(out.dtype() != DType::Bool){
        throw std::logic_error("Expected out array void to be bools");
    }
    if(out.Size() != a.Size()){
        throw std::length_error("Expected all array voids to be the same size");
    }
    if(!out.is_contiguous()){
        throw std::logic_error("Expected out array void to be contiguous");
    }
}

void _isnan(ArrayVoid& out, const ArrayVoid& a){
    _not_equal(out, a, a); // NaN values are not equal to themselves
}

void _equal(ArrayVoid& out, const ArrayVoid& a, Scalar b){
    check_array_voids(out, a, b);
    run_dual_compare_bool(
        reinterpret_cast<bool*>(out.data_ptr()),
        a, b,
        [](auto& s_a, auto& s_b){return s_a == s_b;}
    );
}

void _not_equal(ArrayVoid& out, const ArrayVoid& a, Scalar b){
    check_array_voids(out, a, b);
    run_dual_compare_bool(
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
    if(a.dtype() != DType::Bool){
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
    if(a.dtype() != DType::Bool){
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
    if(a.dtype() != DType::Bool){
        throw std::logic_error("expected bools to detect all");
    }
    return a.cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Bool> > >([](auto begin, auto end){
			return std::all_of(begin, end, [](const uint_bool_t& v){return v.value == 1;});});
}


bool _none(const ArrayVoid& a){
    if(a.dtype() != DType::Bool){
        throw std::logic_error("expected bools to detect all");
    }
    return a.cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Bool> > >([](auto begin, auto end){
			return std::none_of(begin, end, [](const uint_bool_t& v){return v.value == 1;});});
}

bool _any(const ArrayVoid& a){
    if(a.dtype() != DType::Bool){
        throw std::logic_error("expected bools to detect all");
    }
    return a.cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Bool> > >([](auto begin, auto end){
			return std::any_of(begin, end, [](const uint_bool_t& v){return v.value == 1;});});
}


int64_t _amount_of(const ArrayVoid& a, Scalar s){
    if(a.dtype() == DType::Bool){
		uint_bool_t b = s.to<uint_bool_t>();
		return a.cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Bool>>>(
				[&b](auto begin, auto end) -> int64_t{
					int64_t amt = 0;
					for(;begin != end; ++begin)
						if(*begin == b){++amt;}
					return amt;
				});
	}
	return a.cexecute_function<WRAP_DTYPES<NumberTypesL>>(
			[&s](auto a_begin, auto a_end) -> int64_t{
				using value_t = utils::IteratorBaseType_t<decltype(a_begin)>;
				value_t ns = s.to<value_t>();
				int64_t amt = 0;
				for(;a_begin != a_end; ++a_begin)
					if(*a_begin == ns){++amt;}
				return amt;
			});
}

int64_t _count(const ArrayVoid& a){
    if(a.dtype() != DType::Bool){
        throw std::logic_error("expected bools to detect count");
    }
	uint_bool_t b = uint_bool_t(true);
	return a.cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Bool>>>(
			[&b](auto begin, auto end) -> int64_t{
                std::atomic<int64_t> amt{0};
                threading::preferential_parallel_for(
                    threading::block_ranges<1>(0, end-begin),
                    [&](threading::blocked_range<1> block) {
                        auto cur_begin = begin + block.begin[0];
                        auto cur_end = begin + block.end[0];
                        for(;cur_begin != cur_end; ++cur_begin)
                            if(*cur_begin == b) ++amt;
                });
				return amt.load();
			});
}



}
}
}
