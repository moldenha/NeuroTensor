#include "mesh.h"
#include "../../Tensor.h"
#include "../../utils/utils.h"
#include "fill.h"
#include "exceptions.hpp"

namespace nt{
namespace functional{

Tensor one_hot(Tensor indices, int64_t num_classes){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(indices);
    utils::throw_exception(indices.dtype == DType::int64, "Expected input tensor to one hot to have dtype int64 got $", indices.dtype); // Must be integer
    utils::throw_exception(indices.is_contiguous(), "Expected input tensor to one hot to be contiguous", indices.dtype); // Must be integer

    std::vector<int64_t> out_shape = indices.shape().Vec();
    const int64_t* idx_data = reinterpret_cast<const int64_t*>(indices.data_ptr());
    if(num_classes == -1){
        num_classes = *std::max_element(idx_data, idx_data + indices.numel());
    }
    out_shape.push_back(num_classes);

    Tensor out = functional::zeros(SizeRef(std::move(out_shape)), DType::Float32);
    float* out_data = reinterpret_cast<float*>(out.data_ptr());

    for (int64_t i = 0; i < indices.numel(); ++i) {
        int64_t class_idx = idx_data[i];
        utils::throw_exception(class_idx >= 0 && class_idx < num_classes, "Got invalid range for num classes $ and indice $", num_classes, class_idx);
        out_data[i * num_classes + class_idx] = 1.0f;
    }

    return out;
}

}
}
