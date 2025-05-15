#include "../../Tensor.h"
#include "../../utils/optional_list.h"
#include "sum_exp_log.h"
#include "../cpu/sum_exp_log.h"
#include "../../dtype/ArrayVoid.hpp"
#include "softmax.h"
#include "fill.h"
#include "exceptions.hpp"

namespace nt {
namespace functional {

//ln(x)
Tensor log(Tensor x){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    if(x.dtype == DType::TensorObj){
        Tensor out = Tensor::makeNullTensorArray(x.numel());
        Tensor* begin = reinterpret_cast<Tensor*>(out.data_ptr());
        Tensor* end = reinterpret_cast<Tensor*>(out.data_ptr_end());
        x.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&begin, &end](auto begin2, auto end2){
            for(;begin2 != end2; ++begin2, ++begin){
                *begin = log(*begin2);
            }
        });
        return out.view(x.shape());
    }
    utils::throw_exception(x.dtype != DType::Bool, "Cannot take the log of boolean values");
    Tensor out(x.shape(), x.dtype);
    cpu::_log(x.arr_void(), out.arr_void());
    return std::move(out);
}
//derivative of ln(x) is just 1/x
Tensor dlog(Tensor x){return x.inverse();}
Tensor exp(Tensor x){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
   if(x.dtype == DType::TensorObj){
        Tensor out = Tensor::makeNullTensorArray(x.numel());
        Tensor* begin = reinterpret_cast<Tensor*>(out.data_ptr());
        Tensor* end = reinterpret_cast<Tensor*>(out.data_ptr_end());
        x.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&begin, &end](auto begin2, auto end2){
            for(;begin2 != end2; ++begin2, ++begin){
                *begin = exp(*begin2);
            }
        });
        return out.view(x.shape());
    }
    utils::throw_exception(x.dtype != DType::Bool, "Cannot take the log of boolean values");
    Tensor out(x.shape(), x.dtype);
    cpu::_exp(x.arr_void(), out.arr_void());
    return std::move(out);    
}




Tensor sum_one(const Tensor &t, Tensor::size_value_t dim) {
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    if (dim == t.dims() || dim == (-1) * (t.dims() + 1))
        return t;
    dim = dim < 0 ? dim + t.dims() : dim;
    if (t.dtype == DType::TensorObj) {
        Tensor outp(t.shape(), DType::TensorObj);
        t.arr_void().transform_function<DType::TensorObj>(
            [&dim](const Tensor &output) -> Tensor { return sum(output, dim); },
            reinterpret_cast<Tensor *>(outp.data_ptr()));
        return std::move(outp);
    }
    if(dim != 0){
        if(dim == -1 || dim == (t.dims()-1)){
            SizeRef out_shape = t.shape().clone();
            out_shape = out_shape.redo_index(-1, 1);
            Tensor out = zeros(std::move(out_shape), t.dtype);
            cpu::_sum_every(t.arr_void(), out.arr_void(), t.shape()[-1]);
            return std::move(out);
        }
        return sum_one(t.transpose(0, dim), 0).transpose(0, dim);
    }
    Tensor::size_value_t total_size = t.shape()[0];
    Tensor split = t.split_axis(0);
    Tensor a = split[0].item<Tensor>().clone();
    const Tensor *begin = reinterpret_cast<const Tensor *>(split.data_ptr());
    // in the future, use a mutex to lock this and save individual indices
    // otherwise it returns the incorrect result
    // threading::preferential_parallel_for(
    //     threading::block_ranges<1>(1, split.numel()),
    //     [&a, &begin](const auto &range) {
    //         for (int64_t i = range.begin[0]; i < range.end[0]; ++i) {
    //             a += begin[i];
    //         }
    //     });
    const Tensor* end = reinterpret_cast<const Tensor*>(split.data_ptr_end());
    ++begin;
    for(;begin != end; ++begin){
        a += *begin;
    }
    auto Vec = t.shape().Vec();
    Vec[0] = 1;
    return a.view(SizeRef(std::move(Vec)));
}

Tensor sum(Tensor x, utils::optional_list list, bool keepdim){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    if (x.dtype == DType::TensorObj) {
        Tensor outp(x.shape(), DType::TensorObj);
        x.arr_void().transform_function<DType::TensorObj>(
            [&](const Tensor &output) -> Tensor {
                return sum(output, list, keepdim);
            },
            reinterpret_cast<Tensor *>(outp.data_ptr()));
        return std::move(outp);
    }
    if (!list) {
        Tensor outp(1, x.dtype);
        outp = cpu::_accumulate(x.arr_void(), 0);
        if (keepdim) {
            std::vector<SizeRef::ArrayRefInt::value_type> Vec(x.dims());
            std::fill(Vec.begin(), Vec.end(), 1);
            outp = outp.view(SizeRef(std::move(Vec)));
        }
        return std::move(outp);
    }
    int64_t dim = list[0] < 0 ? list[0] + x.dims() : list[0];
    Tensor output = sum_one(x, dim);
    for (auto begin = list->cbegin() + 1; begin != list->cend(); ++begin) {
        dim = *begin < 0 ? *begin + x.dims() : *begin;
        output = sum_one(output, dim);
    }
    if (!keepdim) {
        return output.squeeze();
    }
    return std::move(output);
}

void dsum(Tensor grad, Tensor& out, SizeRef summed_shape){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(grad, out);
    out += grad.view(summed_shape).expand(out.shape());
}

//an optimized version of this will be added
Tensor logsumexp(Tensor x, utils::optional_list list,
                 bool keepdim){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    return log(sum(exp(x), list, keepdim));
}
Tensor dlogsumexp(Tensor grad, Tensor x, utils::optional_list list){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(grad, x);
    if(!list){
        return grad * softmax_stable(x);
    }
    utils::throw_exception(list->size() == 1, "logsumexp gradient is not implemented for more than 1 dimension currently");
    if(grad.dims() != x.dims()){
        utils::THROW_EXCEPTION(x.dims() - grad.dims() == 1, "EXPECTED SIZE DIFF 1 INTERNAL LOGIC ERROR");
        std::vector<int64_t> n_grad_shape = grad.shape().Vec();
        int64_t dim = list[0] < 0 ? list[0] + x.dims() : list[0];
        n_grad_shape.insert(n_grad_shape.begin() + dim, 1);
        grad = grad.view(SizeRef(n_grad_shape));
    }
    Tensor out = grad * softmax_stable(x, list[0]);
    return out;
}

} // namespace functional
} // namespace nt
