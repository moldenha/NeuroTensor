#include "operators.h"
// tensors need to be included in this one
#include "../../mp/simde_traits.h"
#include "../../mp/simde_traits/simde_traits_iterators.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include "../../convert/Convert.h"
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

template<typename T, typename U>
inline void add_num(T begin, T end, U out, utils::IteratorBaseType_t<T> num){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> nums = SimdTraits<base_type>::set1(num);
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> a = it_loadu(begin);
			simde_type<base_type> c = SimdTraits<base_type>::add(a, nums);
			it_storeu(out, c);
		}
		for(;begin < end; ++begin, ++out)
			*out = *begin + num;
	}else{
		for(; begin != end; ++begin, ++out)
			*out = *begin + num;
	}	
}

template<typename T, typename U>
inline void subtract_num(T begin, T end, U out, utils::IteratorBaseType_t<T> num){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> nums = SimdTraits<base_type>::set1(num);
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> a = it_loadu(begin);
			simde_type<base_type> c = SimdTraits<base_type>::subtract(a, nums);
			it_storeu(out, c);
		}
		for(;begin < end; ++begin, ++out)
			*out = *begin - num;
	}else{
		for(; begin != end; ++begin, ++out)
			*out = *begin - num;
	}	
}

template<typename T, typename U>
inline void multiply_num(T begin, T end, U out, utils::IteratorBaseType_t<T> num){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> nums = SimdTraits<base_type>::set1(num);
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> a = it_loadu(begin);
			simde_type<base_type> c = SimdTraits<base_type>::multiply(a, nums);
			it_storeu(out, c);
		}
		for(;begin < end; ++begin, ++out)
			*out = *begin * num;
	}else{
		for(; begin != end; ++begin, ++out)
			*out = *begin * num;
	}	
}

template<typename T, typename U>
inline void divide_num(T begin, T end, U out, utils::IteratorBaseType_t<T> num){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> nums = SimdTraits<base_type>::set1(num);
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> a = it_loadu(begin);
			simde_type<base_type> c = SimdTraits<base_type>::divide(a, nums);
			it_storeu(out, c);
		}
		for(;begin < end; ++begin, ++out)
			*out = *begin / num;
	}else{
		for(; begin != end; ++begin, ++out)
			*out = *begin / num;
	}	
}


//use this to make a cpu::_dot in the future
template<typename T, typename U>
inline utils::IteratorBaseType_t<T> inner_product(T begin, T end, U begin2, utils::IteratorBaseType_t<T> init){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> initial = SimdTraits<base_type>::zero();
		for(;begin + pack_size <= end; begin += pack_size, begin2 += pack_size){
			SimdTraits<base_type>::fmadd(it_loadu(begin), it_loadu(begin2), initial);
		}
		init += SimdTraits<base_type>::sum(initial);
		for(;begin < end; ++begin, ++begin2)
			init += (*begin * *begin2);
		return init;
	}else{
		return std::inner_product(begin, end, begin2, init);
	}	
}

#define _NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(func_name, simde_op, transform_op) \
template<typename T, typename O>\
inline void func_name(T begin, T end, O out){\
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<O>>, "Expected to get base types the same for simde optimized routes");\
	using base_type = utils::IteratorBaseType_t<T>;\
	if constexpr (simde_svml_supported_v<base_type>){\
		static constexpr size_t pack_size = pack_size_v<base_type>;\
		for(; begin + pack_size <= end; begin += pack_size, out += pack_size){\
			simde_type<base_type> a = it_loadu(begin);\
			simde_type<base_type> c = SimdTraits<base_type>::simde_op(a);\
			it_storeu(out, c);\
		}\
		std::transform(begin, end, out,\
				[](base_type x) { return transform_op(x); });\
	}\
	else{\
		std::transform(begin, end, out,\
				[](base_type x) { return transform_op(x); });\
	}\
}\
\

_NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_(reciprical, reciprical, 1.0/ );
#undef _NT_SIMDE_SVML_OP_TRANSFORM_EQUIVALENT_ONE_

} // namespace mp
} // namespace nt

namespace nt {
namespace functional {
namespace cpu {

// multiply divide subtract add
void _operator_mdsa(const ArrayVoid &a, const ArrayVoid &b, ArrayVoid &o,
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
    } else if (op == 2) {
        // subtract
        a.cexecute_function<WRAP_DTYPES<NumberTypesL>>(
            [](auto begin, auto end, auto begin2, void *out_p) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                value_t *out = reinterpret_cast<value_t *>(out_p);
                mp::subtract(begin, end, begin2, out);
            },
            b, o.data_ptr());
    } else if (op == 1) {
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
void _operator_mdsa_(ArrayVoid &a, const ArrayVoid &b, int op) {
    if (op == 0) {
        // multiply
        a.execute_function<WRAP_DTYPES<NumberTypesL>>(
            [](auto begin, auto end, auto begin2) {
                mp::multiply(begin, end, begin2, begin);
            },
            const_cast<ArrayVoid &>(b));
    } else if (op == 2) {
        // subtract
        a.execute_function<WRAP_DTYPES<NumberTypesL>>(
            [](auto begin, auto end, auto begin2) {
                mp::subtract(begin, end, begin2, begin);
            },
            const_cast<ArrayVoid &>(b));
    } else if (op == 1) {
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


void _operator_mdsa_scalar(const ArrayVoid& in, ArrayVoid& out, Scalar s, int op){
   if (op == 0) {
        // multiply
        const_cast<ArrayVoid &>(in).execute_function<WRAP_DTYPES<NumberTypesL>>(
            [&s](auto begin, auto end, auto begin2) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                auto v = s.to<value_t>();
                mp::multiply_num(begin, end, begin2, v);
            },
            out);
    } else if (op == 2) {
        // subtract
        const_cast<ArrayVoid &>(in).execute_function<WRAP_DTYPES<NumberTypesL>>(
            [&s](auto begin, auto end, auto begin2) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                auto v = s.to<value_t>();
                mp::subtract_num(begin, end, begin2, v);
            },
            out);
    } else if (op == 1) {
        // divide
        const_cast<ArrayVoid &>(in).execute_function<WRAP_DTYPES<NumberTypesL>>(
            [&s](auto begin, auto end, auto begin2) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                auto v = s.to<value_t>();
                mp::divide_num(begin, end, begin2, v);
            },
            out);
    } else if (op == 3) {
        // add
        const_cast<ArrayVoid &>(in).execute_function<WRAP_DTYPES<NumberTypesL>>(
            [&s](auto begin, auto end, auto begin2) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                auto v = s.to<value_t>();
                mp::add_num(begin, end, begin2, v);
            },
            out);
    }
}


void _operator_mdsa_scalar_(ArrayVoid& out, Scalar s, int op){
   if (op == 0) {
        // multiply
        out.execute_function_chunk<WRAP_DTYPES<NumberTypesL>>(
            [&s](auto begin, auto end) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                auto v = s.to<value_t>();
                mp::multiply_num(begin, end, begin, v);
            });
    } else if (op == 2) {
        // subtract
        out.execute_function_chunk<WRAP_DTYPES<NumberTypesL>>(
            [&s](auto begin, auto end) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                auto v = s.to<value_t>();
                mp::subtract_num(begin, end, begin, v);
            });
    } else if (op == 1) {
        // divide
        out.execute_function_chunk<WRAP_DTYPES<NumberTypesL>>(
            [&s](auto begin, auto end) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                auto v = s.to<value_t>();
                mp::divide_num(begin, end, begin, v);
            });
    } else if (op == 3) {
        // add
        out.execute_function_chunk<WRAP_DTYPES<NumberTypesL>>(
            [&s](auto begin, auto end) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                auto v = s.to<value_t>();
                mp::add_num(begin, end, begin, v);
            });
    }
}

ArrayVoid _inverse(const ArrayVoid& in){
    DType dtype = in.dtype;
    uint64_t size = in.Size();
    if(dtype == DType::LongLong){
		ArrayVoid output(size, DType::Double);
		in.transform_function<DType::LongLong>([](const auto& inp) -> double {return 1.0/((double)inp);}, reinterpret_cast<double*>(output.data_ptr()));
		return std::move(output);
	}
#ifdef __SIZEOF_INT128__
	if(dtype == DType::int128 || dtype == DType::uint128){
#ifdef _128_FLOAT_SUPPORT_
		ArrayVoid output(size, DType::Float128);
		in.transform_function<DType::uint128, DType::int128>([](const auto& inp) -> float128_t 
				{return 1.0/(::nt::convert::convert<DType::Float128>(inp));}, reinterpret_cast<float128_t*>(output.data_ptr()));
#else
		ArrayVoid output(size, DType::Double);
		in,transform_function<DType::uint128, DType::int128>([](const auto& inp) -> double
				{return 1.0/(::nt::convert::convert<DType::Double>(inp));}, reinterpret_cast<double*>(output.data_ptr()));
#endif
        return std::move(output);
	}
#endif
	if(dtype == DType::Integer || dtype == DType::Long || dtype == DType::uint8 || dtype == DType::int8 || dtype == DType::uint16 || dtype == DType::int16){
		if(dtype == DType::int64){
			return _inverse(in.to(DType::Float64)); //double is slower with the registers but works
		}else{
			return _inverse(in.to(DType::Float32));
		}
	}
    ArrayVoid output(size, dtype);
	//all svml compatible types
	in.cexecute_function<WRAP_DTYPES<FloatingTypesL,ComplexTypesL> >(
			[&output](auto begin, auto end){
		using value_t = utils::IteratorBaseType_t<decltype(begin)>;
		mp::reciprical(begin, end, output.get_bucket().begin_contiguous<value_t>());
	});
    return std::move(output);
}

void _inverse_(ArrayVoid& arr){
    DType dtype = arr.dtype;
    if(dtype != DType::TensorObj && !DTypeFuncs::is_floating(dtype) && !DTypeFuncs::is_complex(dtype)){
		_inverse_(arr.floating_());
        return;
    }
    arr.execute_function_chunk<WRAP_DTYPES<FloatingTypesL,ComplexTypesL> >([](auto begin, auto end){
		mp::reciprical(begin, end, begin);
	});

}

} // namespace cpu
} // namespace functional
} // namespace nt
