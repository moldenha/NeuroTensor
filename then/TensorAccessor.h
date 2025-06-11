//this will be integrated into the overal neurotensor framework at some point, but is very useful for optimizations
//currently, it only really works for contiguous tensors and iterating that
//however, it will be generalized for other iterators in the future
#ifndef _NT_TENSOR_ACCESSOR_H_
#define _NT_TENSOR_ACCESSOR_H_
#include "../Tensor.h"
#include "../utils/utils.h"
#include <memory>
#include <vector>
#include <type_traits>

namespace nt{

template<typename T, size_t N>
class TensorAccessor{
    using size_value_t = Tensor::size_value_t;
    std::shared_ptr<std::vector<size_value_t> > _stored_strides;
    size_value_t* _strides, _shape ;
    T* data;
    TensorAccessor(T* data, std::shared_ptr<std::vector<size_value_t>> _str, size_value_t* _sh, size_value_t* n_strides)
    :data(data), _stored_strides(std::move(_str)), _strides(n_strides), _shape(_sh)
    {}

    TensorAccessor(T* data, std::shared_ptr<std::vector<size_value_t>> _str, size_value_t* _sh)
    :data(data), _stored_strides(std::move(_str)), _strides(&(*_stored_strides)[0]), _shape(_sh)
    {}
public:
    static_assert(N >= 1, "Cannot make a TensorAccessor with a dimensionality less than 1");
    TensorAccessor(Tensor& inp)
    :TensorAccessor(reinterpret_cast<T*>(inp.data_ptr()), std::make_shared<std::vector<size_value_t>>(inp.shape().strides().pop_front()), inp.shape().begin())
    {
        utils::throw_exception(inp.dims() == N, "Expected to get tensor of same dims as tensor accessor made of $ but got $", N, inp.dims());
    }

    std::conditional_t<N > 1, TensorAccessor<T, N-1>, T&> operator[](int64_t idx){
        if constexpr (N > 1){
            return TensorAccessor<T, N - 1>(data + idx * _strides[0], _stored_strides, _shape+1, _strides+1);
        }else{
            return data[idx];
        }
    }
};

}

#endif //_NT_TENSOR_ACCESSOR_H_
