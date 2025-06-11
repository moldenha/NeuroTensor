#include "fused.h"
// tensors need to be included in this one
#include "../../mp/simde_traits.h"
#include "../../mp/simde_traits/simde_traits_iterators.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include <algorithm>

namespace nt {
namespace mp {

//this uses fused multiply add functionality to make this operation much faster
//for operations such as LU and rank reduction
//note to self: make an operation such that it doesn't automatically go into c
template<typename T, typename U, typename V>
inline void fused_multiply_add(T begin_a, T end_a, U begin_b, V begin_c){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> > && 
                    std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<V>>
                , "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		for(;begin_a + pack_size <= end_a; begin_a += pack_size, begin_b += pack_size, begin_c += pack_size){
			simde_type<base_type> a = it_loadu(begin_a);
			simde_type<base_type> b = it_loadu(begin_b);
			simde_type<base_type> c = it_loadu(begin_c);
            SimdTraits<base_type>::fmadd(a, b, c);
			it_storeu(begin_c, c);
		}
		for(; begin_a != end_a; ++begin_a, ++begin_b, ++begin_c)
			*begin_c += *begin_a * *begin_b;
	}else{
		for(; begin_a != end_a; ++begin_a, ++begin_b, ++begin_c)
			*begin_c += *begin_a * *begin_b;
	}
    
}

template<typename T, typename U>
inline void fused_multiply_add_scalar(T begin_a, T end_a, U begin_c, utils::IteratorBaseType_t<T> num){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> > 
                , "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> nums = SimdTraits<base_type>::set1(num);
		for(;begin_a + pack_size <= end_a; begin_a += pack_size, begin_c += pack_size){
			simde_type<base_type> a = it_loadu(begin_a);
			simde_type<base_type> c = it_loadu(begin_c);
            SimdTraits<base_type>::fmadd(a, nums, c);
			it_storeu(begin_c, c);
		}
		for(; begin_a != end_a; ++begin_a, ++begin_c)
			*begin_c += *begin_a * num;
	}else{
		for(; begin_a != end_a; ++begin_a, ++begin_c)
			*begin_c += *begin_a * num;
	}
    
}

//this uses fused multiply subtract functionality to make this operation much faster
//for operations such as LU and rank reduction
template<typename T, typename U, typename V, typename O>
inline void fused_multiply_subtract(T begin_a, T end_a, U begin_b, V begin_c, O out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> > && 
                    std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<V>>
                , "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		for(;begin_a + pack_size <= end_a; begin_a += pack_size, begin_b += pack_size, begin_c += pack_size, out += pack_size){
			simde_type<base_type> a = it_loadu(begin_a);
			simde_type<base_type> b = it_loadu(begin_b);
			simde_type<base_type> c = it_loadu(begin_c);
            simde_type<base_type> o = SimdTraits<base_type>::fmsub(a, b, c);
			it_storeu(out, o);
		}
		for(; begin_a != end_a; ++begin_a, ++begin_b, ++begin_c, ++out)
			*out = *begin_c - (*begin_a * *begin_b);
	}else{
		for(; begin_a != end_a; ++begin_a, ++begin_b, ++begin_c, ++out)
			*out = *begin_c - (*begin_a * *begin_b);
	}
    
}



template<typename T, typename U, typename O>
inline void fused_multiply_subtract_scalar(T begin_a, T end_a, U begin_c, O out, utils::IteratorBaseType_t<T> num){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> > && 
                    std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<O>>
                , "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		simde_type<base_type> nums = SimdTraits<base_type>::set1(num);
		for(;begin_a + pack_size <= end_a; begin_a += pack_size, begin_c += pack_size, out += pack_size){
			simde_type<base_type> a = it_loadu(begin_a);
			simde_type<base_type> c = it_loadu(begin_c);
            simde_type<base_type> o = SimdTraits<base_type>::fmsub(a, nums, c);
			it_storeu(out, o);
		}
		for(; begin_a != end_a; ++begin_a, ++begin_c, ++out)
			*out = *begin_c - (*begin_a * num);
	}else{
		for(; begin_a != end_a; ++begin_a, ++begin_c, ++out)
			*out = *begin_c - (*begin_a * num);
	}
    
}



} // namespace mp
} // namespace nt

namespace nt {
namespace functional {
namespace cpu {


void _fused_multiply_add(ArrayVoid& a, ArrayVoid& b, ArrayVoid& o){
    a.execute_function<WRAP_DTYPES<NumberTypesL > >([&o](auto begin_a, auto end_a, auto begin_b){
		using value_t = utils::IteratorBaseType_t<decltype(begin_a)>;
        value_t* begin_o = reinterpret_cast<value_t*>(o.data_ptr());
        mp::fused_multiply_add(begin_a, end_a, begin_b, begin_o);
	}, b);
}

void _fused_multiply_add(ArrayVoid& a, Scalar b, ArrayVoid& o){
    a.execute_function<WRAP_DTYPES<NumberTypesL> >([&o, &b](auto begin_a, auto end_a){
		using value_t = utils::IteratorBaseType_t<decltype(begin_a)>;
        value_t num = b.to<value_t>();
        value_t* begin_o = reinterpret_cast<value_t*>(o.data_ptr());
        mp::fused_multiply_add_scalar(begin_a, end_a, begin_o, num);
	});
}

void _fused_multiply_add_(ArrayVoid& c, ArrayVoid& a, ArrayVoid& b){
    a.execute_function<WRAP_DTYPES<NumberTypesL> >([&c](auto begin_a, auto end_a, auto begin_b){
		using value_t = utils::IteratorBaseType_t<decltype(begin_a)>;
        c.execute_function<WRAP_DTYPES<DTypeEnum<DTypeFuncs::type_to_dtype<value_t> > > >([&begin_a, &end_a, &begin_b](auto begin_c, auto end_c){
            mp::fused_multiply_add(begin_a, end_a, begin_b, begin_c);
            
        });
	}, b);
}
void _fused_multiply_add_(ArrayVoid& c, ArrayVoid& a, Scalar b){
    a.execute_function<WRAP_DTYPES<NumberTypesL> >([&b](auto begin_a, auto end_a, auto begin_c){
		using value_t = utils::IteratorBaseType_t<decltype(begin_a)>;
        value_t num = b.to<value_t>();
        mp::fused_multiply_add_scalar(begin_a, end_a, begin_c, num);
	}, c);
}

void _fused_multiply_subtract(ArrayVoid& c, ArrayVoid& a, ArrayVoid& b, ArrayVoid& o){
    a.execute_function<WRAP_DTYPES<NumberTypesL> >([&c, &o](auto begin_a, auto end_a, auto begin_b){
		using value_t = utils::IteratorBaseType_t<decltype(begin_a)>;
        c.execute_function<WRAP_DTYPES<DTypeEnum<DTypeFuncs::type_to_dtype<value_t> > > >([&begin_a, &end_a, &begin_b, &o](auto begin_c, auto end_c){
            value_t* begin_o = reinterpret_cast<value_t*>(o.data_ptr());
            mp::fused_multiply_subtract(begin_a, end_a, begin_b, begin_c, begin_o);
            
        });
	}, b);
}
void _fused_multiply_subtract(ArrayVoid& c, ArrayVoid& a, Scalar b, ArrayVoid& o){
    a.execute_function<WRAP_DTYPES<NumberTypesL > >([&b, &o](auto begin_a, auto end_a, auto begin_c){
		using value_t = utils::IteratorBaseType_t<decltype(begin_a)>;
        value_t* begin_o = reinterpret_cast<value_t*>(o.data_ptr());
        value_t num = b.to<value_t>();
        mp::fused_multiply_subtract_scalar(begin_a, end_a, begin_c, begin_o, num);
	}, c);
}


void _fused_multiply_subtract_(ArrayVoid& c, ArrayVoid& a, ArrayVoid& b){
    a.execute_function<WRAP_DTYPES<NumberTypesL > >([&c](auto begin_a, auto end_a, auto begin_b){
		using value_t = utils::IteratorBaseType_t<decltype(begin_a)>;
        c.execute_function<WRAP_DTYPES<DTypeEnum<DTypeFuncs::type_to_dtype<value_t> > > >([&begin_a, &end_a, &begin_b](auto begin_c, auto end_c){
            mp::fused_multiply_subtract(begin_a, end_a, begin_b, begin_c, begin_c);
            
        });
	}, b);
}
void _fused_multiply_subtract_(ArrayVoid& c, ArrayVoid& a, Scalar b){
    a.execute_function<WRAP_DTYPES<NumberTypesL > >([&b](auto begin_a, auto end_a, auto begin_c){
		using value_t = utils::IteratorBaseType_t<decltype(begin_a)>;
        value_t num = b.to<value_t>();
        mp::fused_multiply_subtract_scalar(begin_a, end_a, begin_c, begin_c, num);
	}, c);
}


} // namespace cpu
} // namespace functional
} // namespace nt
