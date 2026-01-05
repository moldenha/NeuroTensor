//this will be integrated into the overal neurotensor framework at some point, but is very useful for optimizations
//currently, it only really works for contiguous tensors and iterating that
//however, it will be generalized for other iterators in the future
#ifndef NT_TENSOR_ACCESSOR_H__
#define NT_TENSOR_ACCESSOR_H__
#include "../Tensor.h"
#include "../utils/utils.h"
#include <memory>
#include <vector>
#include <type_traits>

namespace nt{

template<typename T, size_t N>
class TensorAccessor{
    using size_value_t = Tensor::size_value_t;
    const size_value_t* _strides;
    const size_value_t* _shape;
    T* data;
    TensorAccessor(T* data, size_value_t* _sh, size_value_t* n_strides)
    :data(data), _strides(n_strides), _shape(_sh)
    {}

public:
    static_assert(N >= 1, "Cannot make a TensorAccessor with a dimensionality less than 1");
    TensorAccessor(Tensor& inp)
    :TensorAccessor(reinterpret_cast<T*>(inp.data_ptr()), inp.forceStrideStore(), inp.shape().begin())
    {
        utils::throw_exception(inp.dims() == N, "Expected to get tensor of same dims as tensor accessor made of $ but got $", N, inp.dims());
    }

    inline std::conditional_t<N > 1, TensorAccessor<T, N-1>, T&> operator[](int64_t idx){
        if constexpr (N > 1){
            return TensorAccessor<T, N - 1>(data + idx * _strides[0], _stored_strides, _shape+1, _strides+1);
        }else{
            return data[idx];
        }
    }
};

}

#endif //NT_TENSOR_ACCESSOR_H__
