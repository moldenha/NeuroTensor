#ifndef NT_FMRI_TENSOR_FILES_SAVE_LOAD_H__
#define NT_FMRI_TENSOR_FILES_SAVE_LOAD_H__

#include "../../Tensor.h"
#include <vector>
#include <string>
#include "../../utils/api_macro.h"

namespace nt{
namespace fmri{

NEUROTENSOR_API Tensor load_nifti(const std::string& filename, bool return_grid_spacings=false); 
NEUROTENSOR_API void save_nifti(const std::string& filename, const Tensor& tensor);


}
}

#endif

