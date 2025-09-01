#include "../../mp/simde_traits.h"
#include "../../mp/simde_traits/simde_traits_iterators.h"
#include "min_max.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include "../../utils/macros.h"


#define NT_CPU_CUR_MIN_(a, b) ((a < b) ? a : b)

#define NT_CPU_CUR_MAX_(a, b) ((a > b) ? a : b)

namespace nt::mp {


template<typename T, typename O>
inline void min(T begin, T end, O out, utils::IteratorBaseType_t<T> val){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<O>>, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
        simde_type<base_type> m_val = SimdTraits<base_type>::set1(val);
		for(; begin + pack_size <= end; begin += pack_size, out += pack_size){\
			simde_type<base_type> a = it_loadu(begin);
			simde_type<base_type> c = SimdTraits<base_type>::min(a, m_val);
			it_storeu(out, c);
		}
		std::transform(begin, end, out,
				[&val](base_type x) { return NT_CPU_CUR_MIN_(x, val); });
	}
	else{
		std::transform(begin, end, out,
				[&val](base_type x) { return NT_CPU_CUR_MIN_(x, val); });
	}
}

template<typename T, typename O>
inline void max(T begin, T end, O out, utils::IteratorBaseType_t<T> val){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<O>>, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
        simde_type<base_type> m_val = SimdTraits<base_type>::set1(val);
		for(; begin + pack_size <= end; begin += pack_size, out += pack_size){\
			simde_type<base_type> a = it_loadu(begin);
			simde_type<base_type> c = SimdTraits<base_type>::max(a, m_val);
			it_storeu(out, c);
		}
		std::transform(begin, end, out,
				[&val](base_type x) { return NT_CPU_CUR_MAX_(x, val); });
	}
	else{
		std::transform(begin, end, out,
				[&val](base_type x) { return NT_CPU_CUR_MAX_(x, val); });
	}
}

template<typename T, typename O>
inline void clamp(T begin, T end, O out, utils::IteratorBaseType_t<T> min_val, utils::IteratorBaseType_t<T> max_val){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<O>>, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
        simde_type<base_type> mi_val = SimdTraits<base_type>::set1(min_val);
        simde_type<base_type> ma_val = SimdTraits<base_type>::set1(max_val);
		for(; begin + pack_size <= end; begin += pack_size, out += pack_size){\
			simde_type<base_type> a = it_loadu(begin);
			simde_type<base_type> c = SimdTraits<base_type>::max(a, mi_val);
			                      c = SimdTraits<base_type>::min(c, ma_val);
			it_storeu(out, c);
		}
		std::transform(begin, end, out,
				[&min_val, &max_val](base_type x) { return std::clamp(x, min_val, max_val); });
	}
	else{
		std::transform(begin, end, out,
				[&min_val, &max_val](base_type x) { return std::clamp(x, min_val, max_val); });
	}
}


}

namespace nt{
namespace functional{
namespace cpu{

// min < x < max
void _clamp(ArrayVoid& a, Scalar min, Scalar max){
    a.execute_function<WRAP_DTYPES<NumberTypesL>>([&min, &max](auto begin, auto end){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        value_t n_min = min.to<value_t>();
        value_t n_max = max.to<value_t>();
        ::nt::mp::clamp(begin, end, begin, n_min, n_max);
    });
}

// x < max
void _clamp_above(ArrayVoid& a, Scalar max){
    a.execute_function<WRAP_DTYPES<NumberTypesL>>([&max](auto begin, auto end){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        value_t n_max = max.to<value_t>();
        ::nt::mp::min(begin, end, begin, n_max);
    });
}

// min < x
void _clamp_below(ArrayVoid& a, Scalar min){
    a.execute_function<WRAP_DTYPES<NumberTypesL>>([&min](auto begin, auto end){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        value_t n_min = min.to<value_t>();
        ::nt::mp::max(begin, end, begin, n_min);
    });
}



bool all_contiguous(std::vector<ArrayVoid>& arrvds){
    for(const auto& arr : arrvds){
        utils::THROW_EXCEPTION(arr.is_contiguous(), "INTERNAL LOGIC ERROR: Expected all array voids to be contiguous in cpu processing");
    }
    return true;
}



void _min(ArrayVoid& out, std::vector<ArrayVoid>& arrvds){
    all_contiguous(arrvds);

    out.execute_function<WRAP_DTYPES<NumberTypesL> >([&arrvds](auto begin, auto end){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        const size_t numel = arrvds.size();
        NT_VLA(value_t*, iterators, numel);
        // value_t* iterators[numel];
        size_t i;
        for(i = 0; i < numel; ++i){
            iterators[i] = reinterpret_cast<value_t*>(arrvds[i].data_ptr());
        }
        for(;begin != end; ++begin){
            for(i = 0; i < numel; ++i){
                *begin = NT_CPU_CUR_MIN_(*begin, *iterators[i]);
                ++iterators[i];
            }
        }
        NT_VLA_DEALC(iterators);
    });
}




void _max(ArrayVoid& out, std::vector<ArrayVoid>& arrvds){
    all_contiguous(arrvds);

    out.execute_function<WRAP_DTYPES<NumberTypesL> >([&arrvds](auto begin, auto end){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        const size_t numel = arrvds.size();
        NT_VLA(value_t*, iterators, numel);
        // value_t* iterators[numel];
        size_t i;
        for(i = 0; i < numel; ++i){
            iterators[i] = reinterpret_cast<value_t*>(arrvds[i].data_ptr());
        }
        for(;begin != end; ++begin){
            for(i = 0; i < numel; ++i){
                *begin = NT_CPU_CUR_MAX_(*begin, *iterators[i]);
                ++iterators[i];
            }
        }
        NT_VLA_DEALC(iterators);
    });
}



Scalar _min_scalar(const ArrayVoid& in, ArrayVoid& indices){
    if(indices.is_contiguous()){
        return in.cexecute_function<WRAP_DTYPES<NumberTypesL>>(
        [&indices](auto begin, auto end) -> Scalar{
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                uint_bool_t *i_begin =
                    reinterpret_cast<uint_bool_t *>(indices.data_ptr());
                value_t min_element = *begin;
                uint_bool_t *min_indice = i_begin;
                ++begin;
                ++i_begin;
                for (; begin != end; ++begin, ++i_begin) {
                    if (*begin < min_element) {
                        min_element = *begin;
                        min_indice = i_begin;
                    }
                }
                *min_indice = uint_bool_t(true);
                return min_element;
        });
    }else{
        return in.cexecute_function<WRAP_DTYPES<NumberTypesL>>(
        [&indices](auto begin, auto end) -> Scalar{
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                return indices.execute_function<WRAP_DTYPES<DTypeEnum<DType::Bool>>>(
                 [&](auto i_begin, auto i_end) -> Scalar{
                    value_t min_element = *begin;
                    auto min_indice = i_begin;
                    ++begin;
                    ++i_begin;
                    for (; begin != end; ++begin, ++i_begin) {
                        if (*begin < min_element) {
                            min_element = *begin;
                            min_indice = i_begin;
                        }
                    }
                    *min_indice = uint_bool_t(true);
                    return min_element;
                });
        });

    
    }
}



Scalar _max_scalar(const ArrayVoid& in, ArrayVoid& indices){
    if(indices.is_contiguous()){
        return in.cexecute_function<WRAP_DTYPES<NumberTypesL>>(
        [&indices](auto begin, auto end) -> Scalar{
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                uint_bool_t *i_begin =
                    reinterpret_cast<uint_bool_t *>(indices.data_ptr());
                value_t max_element = *begin;
                uint_bool_t *max_indice = i_begin;
                ++begin;
                ++i_begin;
                for (; begin != end; ++begin, ++i_begin) {
                    if (*begin > max_element) {
                        max_element = *begin;
                        max_indice = i_begin;
                    }
                }
                *max_indice = uint_bool_t(true);
                return max_element;
        });
    }else{
        return in.cexecute_function<WRAP_DTYPES<NumberTypesL>>(
        [&indices](auto begin, auto end) -> Scalar{
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                return indices.execute_function<WRAP_DTYPES<DTypeEnum<DType::Bool>>>(
                 [&](auto i_begin, auto i_end) -> Scalar{
                    value_t max_element = *begin;
                    auto max_indice = i_begin;
                    ++begin;
                    ++i_begin;
                    for (; begin != end; ++begin, ++i_begin) {
                        if (*begin > max_element) {
                            max_element = *begin;
                            max_indice = i_begin;
                        }
                    }
                    *max_indice = uint_bool_t(true);
                    return max_element;
                });
        });
    }
}


void _min_strided(const ArrayVoid& in, ArrayVoid& indices, int64_t cols){
    in.cexecute_function<WRAP_DTYPES<NumberTypesL>>(
    [&indices, &cols](auto begin, auto end){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            indices.execute_function<WRAP_DTYPES<DTypeEnum<DType::Bool>>>(
            [&](auto b_begin, auto b_end){
                while (begin != end) {
                    auto b_min_ele = b_begin;
                    value_t min_ele = *begin;
                    auto current_end = begin + cols;
                    ++begin;
                    ++b_begin;
                    for (; begin != current_end; ++begin, ++b_begin) {
                        if (*begin < min_ele) {
                            min_ele = *begin;
                            b_min_ele = b_begin;
                        }
                    }
                    *b_min_ele = uint_bool_t(true);
                }
            });
    });
    
}

void _max_strided(const ArrayVoid& in, ArrayVoid& indices, int64_t cols){
    in.cexecute_function<WRAP_DTYPES<NumberTypesL>>(
    [&indices, &cols](auto begin, auto end){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            indices.execute_function<WRAP_DTYPES<DTypeEnum<DType::Bool>>>(
            [&](auto b_begin, auto b_end){
                while (begin != end) {
                    auto b_max_ele = b_begin;
                    value_t max_ele = *begin;
                    auto current_end = begin + cols;
                    ++begin;
                    ++b_begin;
                    for (; begin != current_end; ++begin, ++b_begin) {
                        if (*begin > max_ele) {
                            max_ele = *begin;
                            b_max_ele = b_begin;
                        }
                    }
                    *b_max_ele = uint_bool_t(true);
                }
            });
    });
    
}


#undef NT_CPU_CUR_MAX_
#undef NT_CPU_CUR_MIN_

}
}
}
