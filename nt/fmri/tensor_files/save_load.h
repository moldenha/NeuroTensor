#ifndef __NT_FMRI_TENSOR_FILES_SAVE_LOAD_H__
#define __NT_FMRI_TENSOR_FILES_SAVE_LOAD_H__

#include "../../Tensor.h"
#include <vector>
#include <string>

namespace nt{
namespace fmri{

Tensor load_nifti(const std::string& filename, bool return_grid_spacings=false); 
void save_nifti(const std::string& filename, const Tensor& tensor);


}
}

#endif

