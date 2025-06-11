#include "sort.h"
#include "../cpu/sort.h"
#include "combine.h"
#include "fill.h"

namespace nt{
namespace functional{

void sort_vals_only(Tensor& values, const bool& descending, const int64_t& dim_size){
    if(values.dtype != DType::TensorObj){
        cpu::_sort_vals_only_(values.arr_void(), descending, dim_size);
    }else{
        cpu::_sort_vals_dtype_tensor_only_(values.arr_void(), descending, dim_size);
    }
}

Tensor sort(const Tensor& input, const Tensor::size_value_t dim, bool descending, bool return_sorted, bool return_indices){
    utils::throw_exception(return_sorted || return_indices, "Sort function must return indices or the sorted tensor");
    utils::throw_exception(input.dtype != DType::Bool, "Cannot sort boolean values");
    auto shape = input.shape();
    int64_t _dim = dim < 0 ? dim + shape.size() : dim;
    int64_t dim_size = shape[_dim];
    utils::throw_exception(_dim >= 0 && _dim < shape.size(), "Invalid dimension $ for sorting", dim);
    Tensor values = return_indices ? input.transpose(_dim, -1).contiguous() : input.transpose(_dim, -1).clone();
    if(!return_indices){
        sort_vals_only(values, descending, dim_size); 
        return values.transpose(_dim, -1).contiguous();
    }
    Tensor indices = arange(values.shape(), DType::int64, 0);  // Create index tensor
    int64_t* indices_begin = reinterpret_cast<int64_t*>(indices.data_ptr());
    int64_t* indices_end = reinterpret_cast<int64_t*>(indices.data_ptr_end());
    if(input.dtype != DType::TensorObj){
        cpu::_sort_(values.arr_void(), indices_begin, indices_end, descending, dim_size);
    }else{
        cpu::_sort_tensor_(values.arr_void(), indices_begin, indices_end, descending, dim_size);
    }
    indices = indices.transpose(dim, -1).contiguous();
    if(!return_sorted){return std::move(indices);}
    Tensor n_values = input.flatten(0, -1)[indices.flatten(0, -1)];
    n_values = n_values.contiguous().view(shape);
    return list(std::move(n_values), std::move(indices));
}

//this is a function designed to sort the first elements of a row or channel of a tensor
Tensor coordsort(const Tensor& input, Tensor::size_value_t dim, bool descending, bool return_sorted, bool return_indices){
    utils::throw_exception(return_sorted || return_indices, "Sort function must return indices or the sorted tensor, or both, got none");
    auto shape = input.shape();
    int64_t per_dim = input.shape()[dim];
    Tensor split = input.split_axis(dim).view(-1, per_dim);
    if(!return_sorted){
        return sort(split, -1, descending, false, true);
    }
    auto [split_sorted, indices] = get<2>(sort(split, -1, descending));
    Tensor un_split = cat(std::move(split_sorted));
    if(!return_indices){return un_split.view(shape);}
    return list(un_split.view(shape), std::move(indices));
}


}
}
