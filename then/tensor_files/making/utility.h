//5th to be implemented
#ifndef __NT_FMRI_TENSOR_FILES_UTILITY_H__
#define __NT_FMRI_TENSOR_FILES_UTILITY_H__

#include "../../Tensor.h"
#include <string>
#include <vector>

namespace nt {
namespace fmri {

Tensor batch_process(const std::vector<std::string> &files);
Tensor resample(const Tensor &fmri, const std::vector<int> &new_shape);
Tensor interpolate(const Tensor &fmri, const std::vector<double> &scale);
Tensor affine_transform(const Tensor &fmri, const Tensor &matrix);

} // namespace fmri
} // namespace nt

#endif
