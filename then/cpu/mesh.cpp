#include "../../mp/simde_traits.h"
#include "../../mp/simde_traits/simde_traits_iterators.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include "../../refs/SizeRef.h"
#include <algorithm>


namespace nt {
namespace functional {
namespace cpu {

void _meshgrid(const ArrayVoid& x, const ArrayVoid& y, ArrayVoid& outX, ArrayVoid& outY){
    const typename SizeRef::value_type x_n = x.Size();
	const typename SizeRef::value_type y_n = y.Size();

    x.cexecute_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool>>>([&outX, &outY, &x_n, &y_n](auto a_begin, auto a_end, auto b_begin){
        const typename SizeRef::value_type total_size = x_n * y_n;
        using value_t = utils::IteratorBaseType_t<decltype(a_begin)>;
        value_t* x_begin = reinterpret_cast<value_t*>(outX.data_ptr());
        value_t* y_begin = reinterpret_cast<value_t*>(outY.data_ptr());
        auto b_end = b_begin + y_n;
        auto b_cpy = b_begin;
        
        for(;a_begin != a_end; ++a_begin){
            for(;b_begin != b_end; ++b_begin, ++x_begin, ++y_begin){
                *x_begin = *a_begin;
                *y_begin = *b_begin;
            }
            b_begin = b_cpy;
        }
    }, y);
}

} // namespace cpu
} // namespace functional
} // namespace nt
