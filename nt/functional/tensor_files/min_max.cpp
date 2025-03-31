#include "min_max.h"
#include "../cpu/min_max.h"
#include "../functional.h"

namespace nt{
namespace functional{

Tensor clamp(const Tensor& x, std::optional<Scalar> min, std::optional<Scalar> max){
	Tensor out = x.clone();
	if(min && max){
        cpu::_clamp(out.arr_void(), min.value(), max.value());
		return std::move(out);
	}
	else if(min)
		out[out < min.value()] = 0;
	else if(max)
		out[out > max.value()] = max.value();
	return std::move(out);
}
Tensor relu(const Tensor& x){return clamp(x, 0);}

Tensor silu(const Tensor& x){
	return x * sigmoid(x);
}

Tensor dsilu(const Tensor& x){
	Tensor sigmoid_x = sigmoid(x);
	Tensor grad = sigmoid_x * (1 + x * (1 - sigmoid_x));
	return std::move(grad);
}



Tensor gelu(const Tensor& x){
	Scalar sqrt_2_pi = std::sqrt(2.0 / M_PI);
	return 0.5 * x * (1.0 + tanh(sqrt_2_pi * (x + 0.044715 * std::pow(x, 3))));
}

Tensor dgelu(const Tensor& x) {
    const Scalar sqrt_2_pi(std::sqrt(2.0 / M_PI));
    const Scalar c(0.044715);

    Tensor z = sqrt_2_pi * (x + c * std::pow(x, 3));
    // Compute tanh(z) and its derivative
    z = tanh(z);
    Tensor tanh_derivative = 1 - (z * z);

    // Gradient of z with respect to x
    Tensor dz_dx = sqrt_2_pi * (1 + 3 * c.to<double>() * x * x);

    // Final gradient
    return 0.5 * (1 + z) + 0.5 * x * tanh_derivative * dz_dx;
}

inline bool is_complex(const std::vector<Scalar>& scalars)noexcept {
    return scalars[0].isComplex();
}

inline bool is_floating(const std::vector<Scalar>& scalars)noexcept {
    return scalars[0].isFloatingPoint();
}

inline bool is_integral(const std::vector<Scalar>& scalars)noexcept {
    return scalars[0].isIntegral();
}

inline bool is_bool(const std::vector<Scalar>& scalars)noexcept {
    return scalars[0].isBoolean();
}


inline bool same_type(std::vector<Scalar>& scalars){
    if(is_complex(scalars)){
        for(const auto& s : scalars){
            if(!s.isComplex())
                return false;
        }
        return true;
    }
    if(is_floating(scalars)){
        for(const auto& s : scalars){
            if(!s.isFloatingPoint())
                return false;
        }
        return true;
    }
    if(is_integral(scalars)){
        for(const auto& s : scalars){
            if(!s.isIntegral())
                return false;
        }
        return true;
    }
    if(is_bool(scalars)){
        for(const auto& s : scalars){
            if(!s.isBoolean())
                return false;
        }
        return true;
    }
    return false;
}

template<typename Comp, typename T>
inline Scalar compare_scalars_type(const std::vector<Scalar>& scs, Comp&& func){
    T start = scs[0].to<T>();
    for(size_t i = 1; i < scs.size(); ++i){
        start = func(start, scs[i].to<T>());
    }
    return Scalar(start);
}

template<typename Comp>
inline Scalar compare_scalars_complex(const std::vector<Scalar>& scs, Comp&& func){
    return compare_scalars_type<Comp, complex_128>(scs, std::forward<Comp&&>(func)); 
    // Scalar start = scs[0];
    // complex_128 start = = scs[0].to<complex_128>();
    // for(size_t i = 1; i < scs.size(); ++i){
    //     start = func(scs[i].to<complex_128>(), start);
    // }
    // return Scalar(start);
}

template<typename Comp>
inline Scalar compare_scalars_integral(const std::vector<Scalar>& scs, Comp&& func){
    return compare_scalars_type<Comp, int64_t>(scs, std::forward<Comp&&>(func)); 
    // Scalar start = scs[0];
    // int64_t start = scs[0].to<int64_t>();
    // for(size_t i = 1; i < scs.size(); ++i){
    //     start = func(scs[i].v.to<int64_t>(), start.v.i);
    // }
}


template<typename Comp>
inline Scalar compare_scalars_floating(const std::vector<Scalar>& scs, Comp&& func){
    return compare_scalars_type<Comp, double>(scs, std::forward<Comp&&>(func)); 
    // Scalar start = scs[0];
    // for(size_t i = 1; i < scs.size(); ++i){
    //     start = func(scs[i].v.d, start.v.d);
    // }
}

Scalar min(std::vector<Scalar> scalars){
    if(scalars.size() == 1){return scalars[0];}
    utils::throw_exception(scalars.size() > 0, "Cannot find min of no scalars");
    utils::throw_exception(same_type(scalars), "Expected to compare all scalars of the same type");
    utils::throw_exception(!is_bool(scalars), "Cannot find min of bools");
    auto func = [](auto& a, auto b){return std::min(a, b);};
    if(is_complex(scalars)){
        return compare_scalars_complex(scalars, func); 
    }
    if(is_integral(scalars)){
        return compare_scalars_integral(scalars, func); 
    }
    return compare_scalars_floating(scalars, func); 
}
Scalar max(std::vector<Scalar> scalars){
    if(scalars.size() == 1){return scalars[0];}
    utils::throw_exception(scalars.size() > 0, "Cannot find max of no scalars");
    utils::throw_exception(same_type(scalars), "Expected to compare all scalars of the same type");
    utils::throw_exception(!is_bool(scalars), "Cannot find max of bools");
    auto func = [](auto& a, auto b){return std::max(a, b);};
    if(is_complex(scalars)){
        return compare_scalars_complex(scalars, func); 
    }
    if(is_integral(scalars)){
        return compare_scalars_integral(scalars, func); 
    }
    return compare_scalars_floating(scalars, func); 

}

inline bool same_type(const std::vector<Tensor>& tensors){
    DType dt = tensors[0].dtype;
    for(size_t i = 1; i < tensors.size(); ++i){
        if(tensors[i].dtype != dt){return false;}
    }
    return true;
}

inline bool same_shape(const std::vector<Tensor>& tensors){
    const SizeRef& shape = tensors[0].shape();
    for(size_t i = 1; i < tensors.size(); ++i){
        if(tensors[i].shape() != shape){return false;}
    }
    return true;
}


Tensor min(std::vector<Tensor> tensors){
    if(tensors.size() == 1){return tensors[0];}
    utils::throw_exception(tensors.size() > 0, "Cannot find min of no tensors");
    const SizeRef& shape = tensors[0].shape();
    utils::throw_exception(same_type(tensors), "Expected to compare all tensors of the same dtype for min");
    utils::throw_exception(same_shape(tensors), "Expected to compare all tensors of the same shape for min");
    const DType& dt = tensors[0].dtype;
    utils::throw_exception(dt != DType::Bool && dt != DType::TensorObj, 
                           "Expected to get dtype of number type for min got $", dt);
    Tensor out = tensors[0].clone();
    std::vector<ArrayVoid> arrvds;
    arrvds.reserve(tensors.size()-1);
    for(size_t i = 1; i < tensors.size(); ++i){
        arrvds.emplace_back(tensors[i].arr_void().contiguous());
    }
    cpu::_min(out.arr_void(), arrvds);
    return std::move(out);
}

Tensor max(std::vector<Tensor> tensors){
    if(tensors.size() == 1){return tensors[0];}
    utils::throw_exception(tensors.size() > 0, "Cannot find max of no tensors");
    const SizeRef& shape = tensors[0].shape();
    utils::throw_exception(same_type(tensors), "Expected to compare all tensors of the same dtype for max");
    utils::throw_exception(same_shape(tensors), "Expected to compare all tensors of the same shape for max");
    const DType& dt = tensors[0].dtype;
    utils::throw_exception(dt != DType::Bool && dt != DType::TensorObj, 
                           "Expected to get dtype of number type for max got $", dt);
    Tensor out = tensors[0].clone();
    std::vector<ArrayVoid> arrvds;
    arrvds.reserve(tensors.size()-1);
    for(size_t i = 1; i < tensors.size(); ++i){
        arrvds.emplace_back(tensors[i].arr_void().contiguous());
    }
    cpu::_max(out.arr_void(), arrvds);
    return std::move(out);
}


Tensor min(std::vector<Tensor> tensors, Scalar sc){
    if(tensors.size() == 1){return tensors[0];}
    utils::throw_exception(tensors.size() > 0, "Cannot find min of no tensors");
    const SizeRef& shape = tensors[0].shape();
    utils::throw_exception(same_type(tensors), "Expected to compare all tensors of the same dtype for min");
    utils::throw_exception(same_shape(tensors), "Expected to compare all tensors of the same shape for min");
    const DType& dt = tensors[0].dtype;
    utils::throw_exception(dt != DType::Bool && dt != DType::TensorObj, 
                           "Expected to get dtype of number type for min got $", dt);
    Tensor out(shape, dt);
    out = sc;
    std::vector<ArrayVoid> arrvds;
    arrvds.reserve(tensors.size());
    for(size_t i = 0; i < tensors.size(); ++i){
        arrvds.emplace_back(tensors[i].arr_void().contiguous());
    }
    cpu::_min(out.arr_void(), arrvds);
    return std::move(out);

}
Tensor max(std::vector<Tensor> tensors, Scalar sc){
    if(tensors.size() == 1){return tensors[0];}
    utils::throw_exception(tensors.size() > 0, "Cannot find max of no tensors");
    const SizeRef& shape = tensors[0].shape();
    utils::throw_exception(same_type(tensors), "Expected to compare all tensors of the same dtype for max");
    utils::throw_exception(same_shape(tensors), "Expected to compare all tensors of the same shape for max");
    const DType& dt = tensors[0].dtype;
    utils::throw_exception(dt != DType::Bool && dt != DType::TensorObj, 
                           "Expected to get dtype of number type for max got $", dt);
    Tensor out(shape, dt);
    out = sc;
    std::vector<ArrayVoid> arrvds;
    arrvds.reserve(tensors.size());
    for(size_t i = 0; i < tensors.size(); ++i){
        arrvds.emplace_back(tensors[i].arr_void().contiguous());
    }
    cpu::_max(out.arr_void(), arrvds);
    return std::move(out);
 
}
}
}
