#include <nt/Tensor.h>
#include <nt/fmri/fmri.h>

void fmri_load(){
    auto [out, spacings] = nt::get<2>(nt::fmri::load_nifti("../files/sub-A00000368_ses-20110101_task-rest_bold.nii", true));
    std::cout << out.shape() << std::endl;
    std::cout << out[0][0] << std::endl;
    std::cout << nt::functional::count(out == 0) << ':' << out.numel() << std::endl;
    std::cout << spacings << std::endl;
}
