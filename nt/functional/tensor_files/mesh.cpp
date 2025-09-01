#include "mesh.h"
#include "../../Tensor.h"
#include "../../utils/utils.h"
#include "fill.h"
#include "exceptions.hpp"
#include "../cpu/mesh.h"
#include "../../dtype/ArrayVoid.hpp"

namespace nt{
namespace functional{

Tensor one_hot(const Tensor& indices, int64_t num_classes){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(indices);
    utils::throw_exception(indices.dtype() == DType::int64, "Expected input tensor to one hot to have dtype int64 got $", indices.dtype()); // Must be integer
    utils::throw_exception(indices.is_contiguous(), "Expected input tensor to one hot to be contiguous", indices.dtype()); // Must be integer

    std::vector<int64_t> out_shape = indices.shape().Vec();
    const int64_t* idx_data = reinterpret_cast<const int64_t*>(indices.data_ptr());
    if(num_classes == -1){
        num_classes = (*std::max_element(idx_data, idx_data + indices.numel()))+1;
    }
    out_shape.push_back(num_classes);

    Tensor out = functional::zeros(SizeRef(std::move(out_shape)), DType::int64);
    int64_t* out_data = reinterpret_cast<int64_t*>(out.data_ptr());

    for (int64_t i = 0; i < indices.numel(); ++i) {
        int64_t class_idx = idx_data[i];
        utils::throw_exception(class_idx >= 0 && class_idx < num_classes, "Got invalid range for num classes $ and indice $", num_classes, class_idx);
        out_data[i * num_classes + class_idx] = 1;
    }

    return out;
}

Tensor meshgrid(const Tensor& x, const Tensor& y){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x, y);
	utils::THROW_EXCEPTION(x.dtype() == y.dtype(), "Runtime Error for meshgrid: Expected tensors to have same dtype but got $ and $", x.dtype(), y.dtype());
	/* utils::THROW_EXCEPTION(a.numel() == b.numel(), "RuntimeError: Expected tensors to have same number of elements but got $ and $", a.numel(), b.numel()) */
	Tensor xy({2}, DType::TensorObj);
	Tensor* xy_p = reinterpret_cast<Tensor*>(xy.data_ptr());
	*xy_p = Tensor({static_cast<typename SizeRef::value_type>(x.numel()), static_cast<typename SizeRef::value_type>(y.numel())}, x.dtype());
	*(xy_p + 1) = Tensor({static_cast<typename SizeRef::value_type>(x.numel()), static_cast<typename SizeRef::value_type>(y.numel())}, x.dtype());
	if(x.dtype() == DType::TensorObj){
        const typename SizeRef::value_type x_n = x.numel();
        const typename SizeRef::value_type y_n = y.numel();
        x.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj>>>([xy_p, &x_n, &y_n](auto a_begin, auto a_end, auto b_begin){
                    Tensor& X = *xy_p;
                    Tensor& Y = *(xy_p+1);
                    const typename SizeRef::value_type total_size = x_n * y_n;
                    using value_t = utils::IteratorBaseType_t<decltype(a_begin)>;
                    value_t* x_begin = reinterpret_cast<value_t*>(X.data_ptr());
                    value_t* y_begin = reinterpret_cast<value_t*>(Y.data_ptr());
                    auto b_end = b_begin + y_n;
                    auto b_cpy = b_begin;
                    
                    for(;a_begin != a_end; ++a_begin){
                        for(;b_begin != b_end; ++b_begin, ++x_begin, ++y_begin){
                            *x_begin = *a_begin;
                            *y_begin = *b_begin;
                        }
                        b_begin = b_cpy;
                    }

                }, y.arr_void());
        return std::move(xy);
    }
    cpu::_meshgrid(x.arr_void(), y.arr_void(), xy_p->arr_void(), xy_p[1].arr_void());
    return std::move(xy);
}

}
}
