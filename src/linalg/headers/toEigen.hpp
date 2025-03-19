#ifndef _NT_LINALG_TO_EIGEN_HPP_
#define _NT_LINALG_TO_EIGEN_HPP_

#include "../../Tensor.h"
#include "../../utils/utils.h"
#include "../../dtype/ArrayVoid.hpp"
#include "../../convert/Convert.h"
#include "EigenDetails.hpp"
#include <Eigen/Dense>
#include <complex>
#include <type_traits>

namespace nt{
namespace linalg{

template <typename It>
Eigen::Matrix<detail::NT_Type_To_Eigen_Type_t<utils::IteratorBaseType_t<It>>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> tensorToEigen(It begin, It end, const int64_t& rows, const int64_t& cols){
    using type = utils::IteratorBaseType_t<It>;
    using eigen_type = detail::NT_Type_To_Eigen_Type_t<type>;
    constexpr bool needs_conversion = detail::NT_Transform_Type_To_Eigen_v<type>;
    if constexpr (utils::iterator_is_contiguous_v<It>){
        if constexpr (needs_conversion){
           Eigen::Matrix<eigen_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out_mat(rows, cols);
            std::transform(begin, end, out_mat.data(), [](type in_t) {
                if constexpr (std::is_same_v<type, complex_64>){
                    return std::complex<float>(in_t);
                }else if constexpr (std::is_same_v<type, complex_128>){
                    return std::complex<double>(in_t);
                }else if constexpr (std::is_same_v<type, complex_32>){
                    return std::complex<float>(convert::convert<complex_64>(in_t));
                }else{
                    return convert::convert<eigen_type, type>(in_t);
                }
            });
            return out_mat; 
        }else{
            return Eigen::Map<Eigen::Matrix<eigen_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                    reinterpret_cast<eigen_type*>(begin), rows, cols
                );
        }
    }else{
        Eigen::Matrix<eigen_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out_mat(rows, cols);
        if constexpr (needs_conversion){
            std::transform(begin, end, out_mat.data(), [](type in_t) {
                if constexpr (std::is_same_v<type, complex_64>){
                    return std::complex<float>(in_t);
                }else if constexpr (std::is_same_v<type, complex_128>){
                    return std::complex<double>(in_t);
                }else if constexpr (std::is_same_v<type, complex_32>){
                    return std::complex<float>(convert::convert<complex_64>(in_t));
                }else{
                    return convert::convert<eigen_type, type>(in_t);
                }
            });
        }else{
           std::transform(begin, end, out_mat.data(), [](type in_t) {
                return in_t;
            });
        }
        return out_mat;
    } 
}

template<typename Func, typename... Args>
Tensor runEigenFunction(const Tensor& tensor, Func func, Args... args){
    utils::throw_exception(tensor.dims() == 2, "Only 2D tensors can be converted to Eigen matricies, got $ dims", tensor.dims());
    utils::throw_exception(tensor.dtype != DType::Bool && tensor.dtype != DType::TensorObj, "Can only use number types for linear algebra functions got $", tensor.dtype);
    Tensor t = tensor.contiguous();
    Tensor out = t.arr_void().execute_function<WRAP_DTYPES<NumberTypesL>>([&](auto begin, auto end){
        const int64_t& rows = tensor.shape()[0];
        const int64_t& cols = tensor.shape()[1];
        auto matrix = tensorToEigen(begin, end, rows, cols);
        Tensor out = func(matrix, std::forward<Args>(args)...);
        return std::move(out);
    });
    if(out.dtype != DType::TensorObj && out.dtype != DType::Bool){
        return out.to(tensor.dtype);
    }
    return std::move(out);
}

template<typename Func, typename... Args>
Tensor runDualEigenFunction(const Tensor& tensor, const Tensor& tensor2, Func func, Args... args){
    utils::throw_exception(tensor.dims() == 2, "Only 2D tensors can be converted to Eigen matricies, got $ dims", tensor.dims());
    utils::throw_exception(tensor2.dims() == 2, "Only 2D tensors can be converted to Eigen matricies, got $ dims", tensor2.dims());
    utils::throw_exception(tensor.dtype != DType::Bool && tensor.dtype != DType::TensorObj, "Can only use number types for linear algebra functions got $", tensor.dtype);
    utils::throw_exception(tensor2.dtype != DType::Bool && tensor2.dtype != DType::TensorObj, "Can only use number types for linear algebra functions got $", tensor2.dtype);
    Tensor t1 = tensor.contiguous();
    Tensor t2 = tensor2.contiguous();
    Tensor out = t1.arr_void().execute_function<WRAP_DTYPES<NumberTypesL>>([&](auto begin, auto end){
        using type = utils::IteratorBaseType_t<decltype(begin)>;
        const int64_t& rows1 = tensor.shape()[0];
        const int64_t& cols1 = tensor.shape()[1];
        const int64_t& rows2 = tensor2.shape()[0];
        const int64_t& cols2 = tensor2.shape()[1];
        type* b2 = reinterpret_cast<type*>(t2.data_ptr());
        type* e2 = reinterpret_cast<type*>(t2.data_ptr_end());
        auto matrix1 = tensorToEigen(begin, end, rows1, cols1);
        auto matrix2 = tensorToEigen(b2, e2, rows2, cols2);
        Tensor out = func(matrix1, matrix2, std::forward<Args>(args)...);
        return std::move(out);
    });
    if(out.dtype != DType::TensorObj && out.dtype != DType::Bool){
        return out.to(tensor.dtype);
    }
    return std::move(out);
}


template<typename Func, typename... Args>
Tensor runTrippleEigenFunction(const Tensor& tensor, const Tensor& tensor2, const Tensor& tensor3, Func func, Args... args){
    utils::throw_exception(tensor.dims() == 2, "Only 2D tensors can be converted to Eigen matricies for Linalg functions, got $ dims", tensor.dims());
    utils::throw_exception(tensor2.dims() == 2, "Only 2D tensors can be converted to Eigen matricies for Linalg functions, got $ dims", tensor2.dims());
    utils::throw_exception(tensor3.dims() == 2, "Only 2D tensors can be converted to Eigen matricies for Linalg functions, got $ dims", tensor3.dims());
    utils::throw_exception(tensor.dtype != DType::Bool && tensor.dtype != DType::TensorObj, "Can only use number types for linear algebra functions got $", tensor.dtype);
    utils::throw_exception(tensor2.dtype != DType::Bool && tensor2.dtype != DType::TensorObj, "Can only use number types for linear algebra functions got $", tensor2.dtype);
    utils::throw_exception(tensor3.dtype != DType::Bool && tensor3.dtype != DType::TensorObj, "Can only use number types for linear algebra functions got $", tensor3.dtype);
    Tensor t1 = tensor.contiguous();
    Tensor t2 = tensor2.contiguous();
    Tensor t3 = tensor3.contiguous();
    Tensor out = t1.arr_void().execute_function<WRAP_DTYPES<NumberTypesL>>([&](auto begin, auto end){
        using type = utils::IteratorBaseType_t<decltype(begin)>;
        const int64_t& rows1 = tensor.shape()[0];
        const int64_t& cols1 = tensor.shape()[1];
        const int64_t& rows2 = tensor2.shape()[0];
        const int64_t& cols2 = tensor2.shape()[1];
        const int64_t& rows3 = tensor3.shape()[0];
        const int64_t& cols3 = tensor3.shape()[1];
        type* b2 = reinterpret_cast<type*>(t2.data_ptr());
        type* e2 = reinterpret_cast<type*>(t2.data_ptr_end());
        type* b3 = reinterpret_cast<type*>(t3.data_ptr());
        type* e3 = reinterpret_cast<type*>(t3.data_ptr_end());
        auto matrix1 = tensorToEigen(begin, end, rows1, cols1);
        auto matrix2 = tensorToEigen(b2, e2, rows2, cols2);
        auto matrix3 = tensorToEigen(b3, e3, rows2, cols2);
        Tensor out = func(matrix1, matrix2, matrix3, std::forward<Args>(args)...);
        return std::move(out);
    });
    if(out.dtype != DType::TensorObj && out.dtype != DType::Bool){
        return out.to(tensor.dtype);
    }
    return std::move(out);
}

template<typename T>
Tensor fromEigen(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& mat){
    constexpr DType conv_dtype = detail::EigenType_to_DType<T>::dt;
    static_assert(conv_dtype != DType::Bool, "Invalid type from Eigen matrix");

    // Get the number of rows and columns
    int rows = mat.rows();
    int cols = mat.cols();
    
    // Flatten the Eigen matrix data to a pointer
    T* data = mat.data();  // Eigen's `data()` function returns a pointer to the first element of the matrix
    Tensor out({static_cast<int64_t>(rows), static_cast<int64_t>(cols)}, conv_dtype);
    std::memcpy(out.data_ptr(), data, sizeof(T) * rows * cols);
    return std::move(out);
}


template<typename T>
Tensor fromEigen(Eigen::Matrix<T, Eigen::Dynamic, 1>& vec) {
    constexpr DType conv_dtype = detail::EigenType_to_DType<T>::dt;
    static_assert(conv_dtype != DType::Bool, "Invalid type from Eigen matrix");
    // Get the number of elements (size of the vector)
    int size = vec.size();
    
    // Flatten the Eigen vector data to a pointer
    T* data = vec.data();  // Eigen's `data()` function returns a pointer to the first element of the vector
    
    
    // Create the Tensor with the appropriate shape (size is the dimension for a vector)
    Tensor out({static_cast<int64_t>(size)}, conv_dtype);  // One-dimensional tensor
    std::memcpy(out.data_ptr(), data, sizeof(T) * size);  // Copy data from the Eigen vector
    
    return std::move(out);  // Return the Tensor
}






}}

#endif
