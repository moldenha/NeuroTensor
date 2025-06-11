// 2nd to be implemented

#ifndef __NT_FMRI_TENSOR_FILES_PROCESSING_H__
#define __NT_FMRI_TENSOR_FILES_PROCESSING_H__

#include "../../Tensor.h"

namespace nt {
namespace fmri {

Tensor motion_correction(const Tensor &fmri);
Tensor slice_timing_correction(const Tensor &fmri, double TR);
Tensor spatial_smoothing(const Tensor &fmri, double fwhm);
Tensor temporal_filter(const Tensor &fmri, double low, double high);
Tensor mask_brain(const Tensor &fmri, const Tensor &mask);

} // namespace fmri
} // namespace nt

#endif
