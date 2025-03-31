#include "ranges.h"
#include "combine.h"

namespace nt{
namespace functional{

inline bool idx_is_total(const my_range &range, const SizeRef &shape,
                         const size_t idx) noexcept {
    return range.begin == 0 && range.end == shape[idx];
}



Tensor get_range(const Tensor &t, const my_range &r, size_t idx) {
    if (idx == 0) {
        if (idx_is_total(r, t.shape(), 0)) {
            return t.split_axis(0);
        }
        Tensor split = t.split_axis(0);
        Tensor* splits = reinterpret_cast<Tensor*>(split.data_ptr());
        Tensor::size_value_t a = r.begin < 0 ? r.begin + t.shape()[0] : r.begin;
        Tensor::size_value_t b = r.end < 0 ? r.end + t.shape()[0] : r.end;
        utils::THROW_EXCEPTION(
            a < t.shape()[0] && b <= t.shape()[0],
            "Expected a,b to be less than $ for dim $ but got (a = $), (b = $)",
            t.shape()[0], t.dims(), a, b);

        Tensor out = Tensor::makeNullTensorArray(b-a);
        Tensor* begin = reinterpret_cast<Tensor*>(out.data_ptr());
        for(int64_t i = a; i < b; ++i, ++begin){
            *begin = splits[i];
        }
        return std::move(out);
    }
    utils::THROW_EXCEPTION(t.dtype == DType::TensorObj,
                           "Error with dtype format");
    const Tensor *begin_i = reinterpret_cast<const Tensor *>(t.data_ptr());
    const Tensor *end_i = begin_i + t.numel();
    if (begin_i->dims() == 1) {
        /* std::cout << "doing dims1"<<std::endl; */
        if (idx_is_total(r, begin_i->shape(), 0)) {
            return t;
        }
        Tensor output = Tensor::makeNullTensorArray(t.numel());
        Tensor *begin_o = reinterpret_cast<Tensor *>(output.data_ptr());
        for (; begin_i != end_i; ++begin_i, ++begin_o) {
            *begin_o = op_range(*begin_i, r);
        }
        return std::move(output);
    }
    Tensor output = Tensor::makeNullTensorArray(r.length() * t.numel());
    Tensor *begin_o = reinterpret_cast<Tensor *>(output.data_ptr());

    for (; begin_i != end_i; ++begin_i) {
        /* std::cout << "before split shape: "<<begin_i->shape()<<std::endl; */
        Tensor o = begin_i->split_axis(0);
        Tensor *o_b = reinterpret_cast<Tensor *>(o.data_ptr()) + r.begin;
        Tensor *o_e = reinterpret_cast<Tensor *>(o.data_ptr()) + r.end;
        for (; o_b != o_e; ++o_b, ++begin_o) {
            /* std::cout << o_b->shape() << std::endl; */
            *begin_o = std::move(*o_b);
        }
    }
    return std::move(output);
}

Tensor op_range(Tensor t, my_range r){
   if (idx_is_total(r, t.shape(), 0)) {
        return t;
    }
    // std::cout << "op range "<<r.begin<<','<<r.end<<std::endl;
    Tensor::size_value_t a = r.begin < 0 ? r.begin + t.shape()[0] : r.begin;
    Tensor::size_value_t b = r.end < 0 ? r.end + t.shape()[0] : r.end;
    // std::cout << "a and b: "<<a<<','<<b<<std::endl;
    utils::THROW_EXCEPTION(
        a < t.shape()[0] && b <= t.shape()[0],
        "Expected a,b to be less than $ for dim $ but got (a = $), (b = $)",
        t.shape()[0], t.dims(), a, b);
    
    std::vector<typename SizeRef::ArrayRefInt::value_type> vec = t.shape().Vec();
    vec[0] = b - a;
    SizeRef n_size(std::move(vec));
    int64_t multiply = n_size.multiply(1);
    Tensor out(t.arr_void().share_array(a * multiply, (b - a) * multiply),
            std::move(n_size));
    return std::move(out);
}

//I am considering making this just transpose(0, idx) -> op_range(t, r[idx]) -> transpose(0, idx)
Tensor op_range(const Tensor& t, std::vector<my_range> r){
    Tensor cpy = t;
    const SizeRef& shape = cpy.shape();
    std::vector<Tensor::size_value_t> vec = shape.Vec();
    utils::THROW_EXCEPTION(
        r.size() <= t.dims(),
        "Expected to get less than or equal to $ ranges but got $ ranges",
        t.dims(), r.size());
    if(r.size() == 0){return t;}
    // Tensor ranged = op_range(cpy, r[0]);
    // const size_t original_size = r.size();
    // for(size_t dim = 1; dim < original_size; ++dim) {
    //     if (dim >= r.size()){
    //         break;  // Prevent out-of-bounds access
    //     }
    //     if(dim == r.size()) break;
    //     int64_t val = vec.at(dim);  
    //     int64_t begin = r.at(dim).begin;  
    //     int64_t end = r.at(dim).end;  
    //     if (r[dim].begin == 0 && r[dim].end == val) continue;
    //     std::cout << "transposing"<<std::endl;
    //     Tensor transpose = ranged.transpose(0, dim);
    //     std::cout << "transpose shape: "<<transpose.shape()<<std::endl;
    //     // std::cout << "inter"<<std::endl;
    //     Tensor inter = op_range(transpose, r.at(dim));
    //     std::cout << "inter shape: "<<inter.shape()<<std::endl;
    //     // std::cout << "transposing"<<std::endl;
    //     ranged = inter.transpose(0, dim);
    //     std::cout << "ranged shape: "<<ranged.shape()<<std::endl;
    //     // ranged = op_range(ranged.transpose(0, dim), r.at(dim)).transpose(0, dim);  
    // }
    // return std::move(ranged);

    while (r.size() > 0 && idx_is_total(r.back(), t.shape(), r.size() - 1)) {
        r.pop_back();
    }
    if (r.size() == 1) {
        return op_range(t, r[0]);
    } else if (r.size() == 0) {
        return t;
    }
    for (Tensor::size_value_t i = 0; i < r.size(); ++i) {
        r[i].fix(t.shape()[i]);
    }

    Tensor outs = get_range(t, r[0], 0);
    for (size_t i = 1; i < r.size(); ++i) {
        outs = get_range(outs, r[i], i);
    }
    std::vector<Tensor::size_value_t> n_shape = t.shape().Vec();
    for (Tensor::size_value_t i = 0; i < r.size(); ++i)
        n_shape[i] = r[i].length();

    return cat_unordered(outs).view(n_shape);
}

}
}
