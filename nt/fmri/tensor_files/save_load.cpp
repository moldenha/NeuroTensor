#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../dtype/DType.h"
#include <iostream>
#include <stdexcept>
#include <nifti1_io.h>


namespace nt{
namespace fmri{

Tensor load_nifti(const std::string& filename, bool return_grid_spacings) {
    nifti_image* nim = nifti_image_read(filename.c_str(), 1);
    if (!nim) {
        throw std::runtime_error("Failed to load NIfTI file: " + filename);
    }

    // Determine Tensor DType from NIfTI datatype
    DType dtype;
    switch (nim->datatype) {
        case NIFTI_TYPE_UINT8:   dtype = DType::uint8; break;
        case NIFTI_TYPE_INT16:   dtype = DType::int16;  break;
        case NIFTI_TYPE_INT32:   dtype = DType::int32;  break;
        case NIFTI_TYPE_INT64:   dtype = DType::int64;  break;
        case NIFTI_TYPE_FLOAT32: dtype = DType::Float32; break;
        case NIFTI_TYPE_FLOAT64: dtype = DType::Float64; break;
        case NIFTI_TYPE_COMPLEX64:  dtype = DType::Complex64; break;
        case NIFTI_TYPE_COMPLEX128: dtype = DType::Complex128; break;
        default:
            nifti_image_free(nim);
            throw std::runtime_error("Unsupported NIfTI data type.");
    }
    
    // std::cout << "dx: "<<nim->pixdim[1]<<std::endl;
    // std::cout << "dy: "<<nim->pixdim[2]<<std::endl;
    // std::cout << "dz: "<<nim->pixdim[3]<<std::endl;
    // std::cout << "dt: "<<nim->pixdim[4]<<std::endl;
    // Get shape and create Neuro Tensor
    uint32_t num_dims = nim->dim[0];
    std::vector<int64_t> shape;
    shape.reserve(num_dims);
    for(uint32_t i = 0; i < num_dims; ++i){
        shape.push_back(nim->dim[i+1]);
    }
    std::reverse(shape.begin(), shape.end());
    Tensor tensor(SizeRef(std::move(shape)), dtype);


    // Copy NIfTI data to Tensor
    std::memcpy(tensor.data_ptr(), nim->data, tensor.numel() * DTypeFuncs::size_of_dtype(dtype));

    if(!return_grid_spacings){
        nifti_image_free(nim);
        return tensor;
    }
    Tensor spacings({static_cast<int64_t>(tensor.dims())}, DType::Float32);
    std::memcpy(spacings.data_ptr(), &nim->pixdim[1], tensor.dims() * sizeof(float));
    std::reverse(reinterpret_cast<float*>(spacings.data_ptr()), reinterpret_cast<float*>(spacings.data_ptr_end()));
    nifti_image_free(nim);
    return functional::list(tensor, spacings);
}


//expects i to be negative
inline int64_t shape_to_nim_dim(const Tensor& t, int64_t i){
    i = i + t.dims();
    if(i >= 0)
        return t.shape()[i];
    return 1;
}

void save_nifti(const std::string& filename, const Tensor& tensor) {
    // Create NIfTI header
    utils::throw_exception(tensor.numel() > 1, "Cannot save tensor of 1 element as fmri data");
    nifti_image* nim = nifti_simple_init_nim();


    int64_t i = 0;
    while(tensor.shape()[i] == 1){++i;}
    nim->ndim = tensor.dims() - i;
    nim->dim[0] = nim->ndim;
    nim->nx = shape_to_nim_dim(tensor, -1);
    nim->ny = shape_to_nim_dim(tensor, -2);
    nim->nz = shape_to_nim_dim(tensor, -3);
    nim->nt = shape_to_nim_dim(tensor, -4);
    nim->nu = shape_to_nim_dim(tensor, -5);
    nim->nv = shape_to_nim_dim(tensor, -6);
    nim->nw = shape_to_nim_dim(tensor, -7);
    nim->dim[1] = nim->nx;
    nim->dim[2] = nim->ny;
    nim->dim[3] = nim->nz;
    nim->dim[4] = nim->nt;
    nim->dim[5] = nim->nu;
    nim->dim[6] = nim->nv;
    nim->dim[7] = nim->nw;

    // Set datatype based on Tensor's DType
    switch (tensor.dtype) {
        case DType::uint8: nim->datatype = NIFTI_TYPE_UINT8; break;
        case DType::int16:  nim->datatype = NIFTI_TYPE_INT16; break;
        case DType::int32:  nim->datatype = NIFTI_TYPE_INT32; break;
        case DType::int64:  nim->datatype = NIFTI_TYPE_INT64; break;
        case DType::Float32: nim->datatype = NIFTI_TYPE_FLOAT32; break;
        case DType::Float64: nim->datatype = NIFTI_TYPE_FLOAT64; break;
        case DType::Complex64:  nim->datatype = NIFTI_TYPE_COMPLEX64; break;
        case DType::Complex128: nim->datatype = NIFTI_TYPE_COMPLEX128; break;
        default:
            nifti_image_free(nim);
            utils::throw_exception(false, "Unsupported Tensor DType $", tensor.dtype);
    }

    // Allocate memory for data
    nim->nbyper = DTypeFuncs::size_of_dtype(tensor.dtype);
    nim->nvox = tensor.numel();
    nim->data = std::malloc(nim->nbyper * nim->nvox);
    if (!nim->data) {
        nifti_image_free(nim);
        throw std::runtime_error("Failed to allocate NIfTI data.");
    }

    // Copy Tensor data into NIfTI image
    std::memcpy(nim->data, tensor.data_ptr(), nim->nbyper * nim->nvox);

    // Save NIfTI image
    nifti_set_filenames(nim, filename.c_str(), 0, 1);
    nifti_image_write(nim);

    // Clean up
    nifti_image_free(nim);
}


}
}
