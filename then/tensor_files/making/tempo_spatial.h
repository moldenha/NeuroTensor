//4th to be implemented
#ifndef __NT_FMRI_TENSOR_FILES_TEMPO_SPATIAL_H__
#define __NT_FMRI_TENSOR_FILES_TEMPO_SPATIAL_H__
#include "../../Tensor.h"

namespace nt{
namespace fmri{

Tensor temporal_convolution(const Tensor& fmri, const Tensor& kernel);  
Tensor spatiotemporal_convolution(const Tensor& fmri, const Tensor& kernel);  
Tensor temporal_mean(const Tensor& fmri);  
Tensor temporal_variance(const Tensor& fmri);  
Tensor zscore(const Tensor& fmri);

}
}

#endif
