#include "loss.h"
#include "../../functional/functional.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace nt::tda::loss {

ScalarGrad filtration_loss(const TensorGrad &output, const Tensor &target,
                           Scalar epsilon) {
    utils::throw_exception(output.shape() == target.shape(),
                           "Expected output and target to have the same shape");
    Tensor max_val = functional::maximum(std::vector<Tensor>({output.detach(), target}), epsilon);
    Tensor dx = output.detach() - target / max_val;
    Scalar item = dx.sum().toScalar();
    return ScalarGrad(item, dx, output);
}

ScalarGrad path_loss(const TensorGrad& output, const Tensor& target){
    float item = 1.0;
    utils::throw_exception(target.is_contiguous() && output.is_contiguous(),
                           "Expected target and output to be contiguous tensors when back propogating path");
    utils::throw_exception(target.dtype() == DType::TensorObj && output.dtype() == DType::TensorObj,
                           "Expected paths to be tensor objects but got $ and $", output.dtype(), target.dtype());
    int64_t total_wanted = target.numel();
    int64_t extracted_and_wanted = 0;
    const Tensor* t_begin = reinterpret_cast<const Tensor*>(target.data_ptr());
    const Tensor* t_end = reinterpret_cast<const Tensor*>(target.data_ptr_end());
    const Tensor* o_begin = reinterpret_cast<const Tensor*>(output.detach().data_ptr());
    const Tensor* o_end = reinterpret_cast<const Tensor*>(output.detach().data_ptr_end());
    for(;t_begin != t_end; ++t_begin){
        const Tensor* cpy = o_begin;
        for(;cpy != o_end; ++cpy){
            if(t_begin->numel() == cpy->numel() && functional::all(*t_begin == *cpy)){
                ++extracted_and_wanted;
                break;
            }
        }
    }
    // if(extracted_and_wanted != 0){
    //     item -= (float(extracted_and_wanted) / float(total_wanted));
    // }

    return ScalarGrad(total_wanted-extracted_and_wanted, target, output);
}

} // nt::tda::loss
