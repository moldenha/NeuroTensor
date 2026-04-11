//this will be integrated into the overal neurotensor framework at some point, but is very useful for optimizations
//currently, it only really works for contiguous tensors and iterating that
//however, it will be generalized for other iterators in the future
#ifndef NT_TENSOR_ACCESSOR_H__
#define NT_TENSOR_ACCESSOR_H__
#include "../Tensor.h"
#include "../utils/utils.h"
#include "../memory/iterator.h"
#include "../utils/type_traits.h"
#include <memory>
#include <vector>
#include <type_traits>

namespace nt{

namespace details{
template<typename Iterator>
inline Iterator get_iterator_nt_tensor__(Tensor& t){
    static_assert(utils::iterator_is_contiguous_v<Iterator> || utils::iterator_is_blocked_v<Iterator> || utils::iterator_is_list_v<Iterator>,
            "Error: Iterator must be contiguous, blocked or strided");
    using base_type = type_traits::remove_cvref_t<utils::IteratorBaseType<Iterator>>;
    if constexpr (utils::iterator_is_contiguous_v<base_type>){ // T*
        if constexpr (type_traits::is_const_v<Iterator){
            return t.arr_void().get_bucket().cbegin_contiguous<base_type>();
        }else{
            return t.arr_void().get_bucket().begin_contiguous<base_type>();
        }
    }
    else if constexpr (utils::iterator_is_blocked_v<Iterator>){
        if constexpr (type_traits::is_const_v<base_type){
            return t.arr_void().get_bucket().cbegin_blocked<base_type>();
        }else{
            return t.arr_void().get_bucket().begin_blocked<base_type>();
        }
    }else{
        if constexpr (type_traits::is_const_v<base_type){
            return t.arr_void().get_bucket().cbegin_list<base_type>();
        }else{
            return t.arr_void().get_bucket().begin_list<base_type>();
        }
    }
}

template<typename Iterator>
inline Iterator get_iterator_nt_tensor__(const Tensor& t){
    static_assert(utils::iterator_is_contiguous_v<Iterator> || utils::iterator_is_blocked_v<Iterator> || utils::iterator_is_list_v<Iterator>,
            "Error: Iterator must be contiguous, blocked or strided");
    using base_type = type_traits::remove_cvref_t<utils::IteratorBaseType<Iterator>>;
    static_assert(type_traits::is_const_v<base_type>, "Error: Got const tensor reference, expected constant iterator type output");
    if constexpr (utils::iterator_is_contiguous_v<base_type>){ // T*
        return t.arr_void().get_bucket().cbegin_contiguous<base_type>();
    }
    else if constexpr (utils::iterator_is_blocked_v<Iterator>){
        return t.arr_void().get_bucket().cbegin_blocked<base_type>();
    }else{
        return t.arr_void().get_bucket().cbegin_list<base_type>();
    }
}

}



// this is for general Tensors
template<typename T, size_t N, typename Iterator = T*>
class TensorAccessor_iter{
    using size_value_t = Tensor::size_value_t;
    const size_value_t* _strides;
    const size_value_t* _shape;
    Iterator data;
    TensorAccessor_iter(Iterator iter, const size_value_t* shape, const size_value_t* strides)
        :_strides(strides), _shape(shape), data(iter)
    {}

    public:
        static_assert(N >= 1, "Cannot make a TensorAccessor with a dimensionality less than 1");
        static_assert(utils::iterator_is_contiguous_v<Iterator> || utils::iterator_is_blocked_v<Iterator> || utils::iterator_is_list_v<Iterator>,
                "Error: Iterator must be contiguous, blocked or strided");
        static_assert(type_traits::is_same_v<type_traits::remove_cvref_t<utils::IteratorBaseType<Iterator>>, type_traits::remove_cvref_t<T>>,
                "Error: Iterator must point to the value given by the value type");
        TensorAccessor_iter(Tensor& inp)
            :TensorAccessor_iter(details::get_iterator_nt_tensor__(inp), inp.shape().begin())
        {
            utils::throw_exception(inp.shape().size() == N, "Error: Expected input tensor shape to have the same size as the number"
                                                            " of initial dimensions ($) but got $", N, inp.shape());
        }
        TensorAccessor_iter(const Tensor& inp)
            :TensorAccessor_iter(details::get_iterator_nt_tensor__(inp), inp.shape().begin())
        {
            utils::throw_exception(inp.shape().size() == N, "Error: Expected input tensor shape to have the same size as the number"
                                                            " of initial dimensions ($) but got $", N, inp.shape());
        }
        inline auto operator[](int64_t idx){
            if constexpr (N > 2){
                return TensorAccessor_iter<T, N - 1, Iterator>(data + idx * _strides[0], _stored_strides, _shape+1, _strides+1);
            }else if constexpr (N == 2 && utils::iterator_is_contiguous_v<Iterator>){
                return data + idx * _strides[0];
            }else{
                return static_cast<T&>(data[idx]);
            }
        }

};

// template<typename T, size_t N>
// class TensorAccessor_contiguous{
//     using size_value_t = Tensor::size_value_t;
//     const size_value_t* _strides;
//     const size_value_t* _shape;
//     T* data;
//     TensorAccessor(T* data, size_value_t* _sh, size_value_t* n_strides)
//     :data(data), _strides(n_strides), _shape(_sh)
//     {}

// public:
//     static_assert(N >= 1, "Cannot make a TensorAccessor with a dimensionality less than 1");
//     TensorAccessor(Tensor& inp)
//     :TensorAccessor(reinterpret_cast<T*>(inp.data_ptr()), inp.forceStrideStore(), inp.shape().begin())
//     {
//         utils::throw_exception(inp.dims() == N, "Expected to get tensor of same dims as tensor accessor made of $ but got $", N, inp.dims());
//     }

//     inline std::conditional_t<N > 1, TensorAccessor<T, N-1>, T&> operator[](int64_t idx){
//         if constexpr (N > 1){
//             return TensorAccessor<T, N - 1>(data + idx * _strides[0], _stored_strides, _shape+1, _strides+1);
//         }else{
//             return data[idx];
//         }
//     }
// };

}

#endif //NT_TENSOR_ACCESSOR_H__
