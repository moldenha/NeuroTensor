// this file has not been defined or implemented
// there are plans to implement the functions in this file
// it just has not yet been done
#ifndef __NT_FMRI_TENSOR_FILES_VISUALIZATION_H__
#define __NT_FMRI_TENSOR_FILES_VISUALIZATION_H__

#include "../../Tensor.h"
namespace nt {
namespace fmri {

void show_slice(const nt::Tensor &fmri, int slice_idx);
void show_time_series(const nt::Tensor &fmri, int x, int y, int z);
void plot_connectivity_matrix(const nt::Tensor &conn_matrix);

} // namespace fmri
} // namespace nt

#endif
