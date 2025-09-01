#include "../../nn/TensorGrad.h"
#include "../../nn/functional.h"
#include "../../dtype/ArrayVoid.hpp"
#include "../../functional/functional.h"

#include "../../types/float128.h"

//if this is defined
//this means that for float128_t boost's 128 bit floating point is used
#ifdef BOOST_MP_STANDALONE
namespace std{
inline ::nt::float128_t round(const ::nt::float128_t& x){
    ::nt::float128_t int_part = trunc(x);  // integer part (toward zero)
    ::nt::float128_t frac = x - int_part;  // fractional part

    if (x >= 0) {
        return frac < 0.5 ? int_part : int_part + 1;
    } else {
        return frac > -0.5 ? int_part : int_part - 1;
    }
}

}

#endif //BOOST_MP_STANDALONE

namespace nt{
namespace tda{

TensorGrad coordsToDist(const TensorGrad& points){
    if(points.dtype() != DType::TensorObj){
        utils::throw_exception(points.dims() == 2, "Expected to get point dim of 2 but with shape {N,D} where there are N points for D dimensions but got shape $", points.shape());
        int64_t D = points.shape()[1];
        TensorGrad dif = points.view(-1, 1, D) - points.view(1, -1, D);
        TensorGrad a = dif.pow(2).sum(-1);
        TensorGrad out = functional::sqrt(a);
        return std::move(out);
    }
    TensorGrad array = TensorGrad::makeNullTensorArray(points.numel());
    int64_t num = points.numel();
    for(int64_t i = 0; i < num; ++i){
        array[i] = coordsToDist(points[i]);
    }
    return array;
}

Tensor coordsToDist(const Tensor& points){
    if(points.dtype() != DType::TensorObj){
        utils::throw_exception(points.dims() == 2, "Expected to get point dim of 2 but with shape {N,D} where there are N points for D dimensions but got shape $", points.shape());
        int64_t D = points.shape()[1];
        Tensor dif = points.view(-1, 1, D) - points.view(1, -1, D);
        Tensor a = dif.pow(2).sum(-1);
        Tensor out = functional::sqrt(a);
        return std::move(out);
    }
    Tensor array = Tensor::makeNullTensorArray(points.numel());
    int64_t num = points.numel();
    for(int64_t i = 0; i < num; ++i){
        array[i].item<Tensor>() = coordsToDist(points[i]);
    }
    return array;
}




inline void round_tensor(Tensor& t){
    if(t.dtype() == DType::Float16){
        t.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::Float16> > >
        ([](auto begin, auto end){
            std::transform(begin, end, begin, [](float16_t x){
                return _NT_FLOAT32_TO_FLOAT16_(std::round(_NT_FLOAT16_TO_FLOAT32_(x)));
            });
        });
        return;
    }
#ifdef _128_FLOAT_SUPPORT_
    if(t.dtype() == DType::Float128){
        t.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::Float128> > >
        ([](auto begin, auto end){
            std::transform(begin, end, begin, [](float128_t x){return std::round(x);});
        });
        return;
    }
#endif
    if(t.dtype() == DType::Float32 || t.dtype() == DType::Float64){
        t.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::Float32, DType::Float64> > >
        ([](auto begin, auto end){
            std::transform(begin, end, begin, [](auto x){return std::round(x);});
        });
        return;
    }
    if(t.dtype() == DType::Complex64 || t.dtype() == DType::Complex128){
        t.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::Complex64, DType::Complex128> > >
        ([](auto begin, auto end){
            // using type = utils::ItteratorBaseType_t<decltype(begin)>;
            std::transform(begin, end, begin, [](auto x){
                    return decltype(x)(std::round(x.real()), std::round(x.imag()));
            });
        });
    }
    if(t.dtype() == DType::Complex32){
        t.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::Complex32> > >
        ([](auto begin, auto end){
            std::transform(begin, end, begin, [](auto x){
                    return complex_32(
                        _NT_FLOAT32_TO_FLOAT16_(std::round(_NT_FLOAT16_TO_FLOAT32_(x.real()))), 
                        _NT_FLOAT32_TO_FLOAT16_(std::round(_NT_FLOAT16_TO_FLOAT32_(x.imag()))));
            });
        });
    }
}


TensorGrad cloudToDist(const TensorGrad& _cloud, Scalar threshold, Scalar grad_lr, int64_t dims){
    if(dims == -1) dims = _cloud.dims();
    Tensor cloud = _cloud.detach();
    utils::throw_exception(
        cloud.dims() >= dims,
        "Expected to process cloud with dims of at least $ but got $", dims,
        cloud.dims());
    if (cloud.dims() > dims) {

        cloud = cloud.flatten(-1, dims);
        TensorGrad c_cloud = _cloud.flatten(-1, dims);
        TensorGrad out_mat = TensorGrad::makeNullTensorArray(c_cloud.shape()[0]);
        for(int64_t i = 0; i < c_cloud.shape()[0]; ++i){
            out_mat[i] = cloudToDist(c_cloud[i], threshold, grad_lr, dims);
        }
        return std::move(out_mat);
    }else{
        // int64_t D = cloud.dims();
        Tensor where = cloud >= threshold;
        Tensor points_w = functional::where(where);
        Tensor points_s = functional::stack(points_w).transpose(-1, -2).to(nt::DType::Float32);
        int64_t D = points_s.shape()[-1];
        utils::THROW_EXCEPTION(D == dims, "Internal logic error");
        Tensor dif = points_s.view(-1, 1, D) - points_s.view(1, -1, D);
        Tensor dist_matrix = functional::sqrt(dif.pow(2).sum(-1));
        int64_t N = dist_matrix.shape()[0];
        intrusive_ptr<tensor_holder> dist_holder = make_intrusive<tensor_holder>(dist_matrix.clone());
        TensorGrad out = TensorGrad::make_tensor_grad(dist_matrix,
                    [dif, dist_holder, N, D, grad_lr, points_s](const Tensor& delta_dist_matrix, std::vector<intrusive_ptr<TensorGrad>>& parents){
                        Tensor G_a = dif * delta_dist_matrix.view(N, N, 1);
                        dist_holder->tensor.fill_diagonal_(1.0);
                        Tensor G_b = G_a / dist_holder->tensor.view(N, N, 1);
                        Tensor Gd = G_b.sum(-2);
                        Tensor updated = points_s - (Gd * grad_lr);
                        Tensor old = points_s.transpose(-1, -2).to(nt::DType::int64).split_axis(-2);
                        parents[0]->grad()[old] = 1;
                        Tensor updated_index = updated.transpose(-1, -2).to(nt::DType::int64).split_axis(-2);
                        Tensor* begin = reinterpret_cast<Tensor*>(updated_index.data_ptr());
                        Tensor* end = reinterpret_cast<Tensor*>(updated_index.data_ptr_end());
                        int64_t counter = 0;
                        for(;begin != end; ++begin, ++counter) {
                            *begin = functional::clamp(*begin, 0, parents[0]->grad().shape()[counter]-1);
                        }
                        parents[0]->grad()[updated_index] = -1;
                    }, _cloud);
        return out;
    }    
}

Tensor cloudToDist(const Tensor& _cloud, Scalar threshold, int64_t dims){
    if(dims == -1) dims = _cloud.dims();
    const Tensor& cloud = _cloud;
    utils::throw_exception(
        cloud.dims() >= dims,
        "Expected to process cloud with dims of at least $ but got $", dims,
        cloud.dims());
    if (cloud.dims() > dims) {

        Tensor c_cloud = _cloud.flatten(-1, dims);
        Tensor out_mat = Tensor::makeNullTensorArray(c_cloud.shape()[0]);
        for(int64_t i = 0; i < c_cloud.shape()[0]; ++i){
            out_mat[i] = cloudToDist(c_cloud[i].item<Tensor>(), threshold, dims);
        }
        return std::move(out_mat);
    }else{
        // int64_t D = cloud.dims();
        Tensor where = cloud >= threshold;
        Tensor points_w = functional::where(where);
        Tensor points_s = functional::stack(points_w).transpose(-1, -2).to(nt::DType::Float32);
        int64_t D = points_s.shape()[-1];
        utils::THROW_EXCEPTION(D == dims, "Internal logic error");
        Tensor dif = points_s.view(-1, 1, D) - points_s.view(1, -1, D);
        Tensor dist_matrix = functional::sqrt(dif.pow(2).sum(-1));
        return std::move(dist_matrix);
    }
}

}
}
