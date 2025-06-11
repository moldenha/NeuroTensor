//3rd to be made
#ifndef __NT_FMRI_TENSOR_FILES_ANALYSIS_H__
#define __NT_FMRI_TENSOR_FILES_ANALYSIS_H__

#include "../../Tensor.h"
#include <utility>

namespace nt{
namespace fmri{

Tensor voxelwise_ttest(const Tensor& fmri1, const Tensor& fmri2);  
Tensor glm_fit(const Tensor& fmri, const Tensor& design_matrix);  
Tensor functional_connectivity(const Tensor& fmri, const Tensor& mask);  
std::pair<Tensor, Tensor> ica(const Tensor& fmri, int n_components);  
std::pair<Tensor, Tensor> pca(const Tensor& fmri, int n_components);


}
}

#endif
