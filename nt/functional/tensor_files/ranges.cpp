#include "ranges.h"
#include "combine.h"
#include "exceptions.hpp"

namespace nt{
namespace functional{

inline bool idx_is_total(const range_ &r, const SizeRef &shape,
                         const size_t idx) noexcept {
    return r.begin == 0 && r.end == shape[idx];
}

namespace details{


// this takes a pair from for example 0 to the number of elements
// it then splits it into sub pairs that represent the next dimension down (like split_axis(n))
// and then extracts all the pairs that are within the given range
std::vector<std::pair<int64_t, int64_t>> split_pair(std::pair<int64_t, int64_t> p, const SizeRef& shape, const range_& r){
    int64_t split_every = shape.multiply(1);
    int64_t start = p.first;
    int64_t end = p.second;
    utils::THROW_EXCEPTION((end - start) % split_every == 0, "Error, internal debug error, splitting not multiplied correctly");
    // int64_t amt = (end - start) / split_every;
    std::vector<std::pair<int64_t, int64_t>> out(r.length());
    start += (split_every * r.begin);
    for(int64_t i = 0; i < r.length(); ++i){
        out[i] = {start, split_every + start};
        start += split_every;
    }
    return std::move(out);
}

// this takes a vector full of pairs representing the upper dimensionality
// it then runs split pair, and stores all the pairs in a new vector
// it then recursively goes to the next dimension until all dimensions are completed
std::vector<std::pair<int64_t, int64_t>> 
        make_range_vector(int64_t index, std::vector<range_> r, 
                          std::vector<std::pair<int64_t, int64_t>> vec, SizeRef shape){
    if(index >= r.size()) return std::move(vec);
    r[index].fix(shape[0]);
    if(shape.size() == 1 && shape[0] == r[index].end && r[index].begin == 0){return std::move(vec);}
    if(shape.size() == 1){
        for(size_t i = 0; i < vec.size(); ++i){
            vec[i].first += r[index].begin;
            vec[i].second -= (shape[0] - r[index].end);
        }
        return std::move(vec);
    }
    std::vector<std::pair<int64_t, int64_t>> n_vec;
    n_vec.reserve(vec.size() * r[index].length());
    for(size_t i = 0; i < vec.size(); ++i){
        std::vector<std::pair<int64_t, int64_t>> cur = split_pair(vec[i], shape, r[index]);
        std::copy(cur.begin(), cur.end(), std::back_inserter(n_vec));    
    }
    if(shape.size() == 1) return std::move(n_vec);
    return make_range_vector(index + 1, std::move(r), std::move(n_vec), shape.pop_front());
}


std::vector<std::pair<int64_t, int64_t>> make_range_vector(std::vector<range_> r, const SizeRef& shape, int64_t numel){
    utils::throw_exception(r.size() <= shape.size() && r.size() != 0,
                           "Error, cannot get ranges $ for shape of $", r, shape);
    std::vector<std::pair<int64_t, int64_t>> 
                            out_ranges = make_range_vector(0, std::move(r), 
                            std::vector<std::pair<int64_t, int64_t>> { {0, numel} }, shape.clone());
    
    // now just combine all the pairs where abs(out_ranges[i].second - out_ranges[i+1].first) < 2 

    if(out_ranges.empty()) return out_ranges;
    
    std::vector<std::pair<int64_t, int64_t>> merged;
    merged.reserve(out_ranges.size());
    
    auto cur = out_ranges[0];
    for(size_t i = 1; i < out_ranges.size(); ++i) {
        auto& nxt = out_ranges[i];
        if(cur.second == nxt.first){ 
            // overlap or directly adjacent: merge
            cur.second = std::max(cur.second, nxt.second);
        } else {
            // gap: push current and move to next
            merged.push_back(cur);
            cur = nxt;
        }
    }
    merged.push_back(cur);
    return std::move(merged);
}


}


// Tensor get_range(const Tensor &t, const range_ &r, size_t idx) {
//     _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
//     if (idx == 0) {
//         if (idx_is_total(r, t.shape(), 0)) {
//             return t.split_axis(0);
//         }
//         Tensor split = t.split_axis(0);
//         Tensor* splits = reinterpret_cast<Tensor*>(split.data_ptr());
//         Tensor::size_value_t a = r.begin < 0 ? r.begin + t.shape()[0] : r.begin;
//         Tensor::size_value_t b = r.end < 0 ? r.end + t.shape()[0] : r.end;
//         utils::THROW_EXCEPTION(
//             a < t.shape()[0] && b <= t.shape()[0],
//             "Expected a,b to be less than $ for dim $ but got (a = $), (b = $)",
//             t.shape()[0], t.dims(), a, b);

//         Tensor out = Tensor::makeNullTensorArray(b-a);
//         Tensor* begin = reinterpret_cast<Tensor*>(out.data_ptr());
//         for(int64_t i = a; i < b; ++i, ++begin){
//             *begin = splits[i];
//         }
//         return out.set_mutability(t.is_mutable());
//     }
//     utils::THROW_EXCEPTION(t.dtype() == DType::TensorObj,
//                            "Error with dtype format");
//     const Tensor *begin_i = reinterpret_cast<const Tensor *>(t.data_ptr());
//     const Tensor *end_i = begin_i + t.numel();
//     if (begin_i->dims() == 1) {
//         if (idx_is_total(r, begin_i->shape(), 0)) {
//             return t;
//         }
//         Tensor output = Tensor::makeNullTensorArray(t.numel());
//         Tensor *begin_o = reinterpret_cast<Tensor *>(output.data_ptr());
//         for (; begin_i != end_i; ++begin_i, ++begin_o) {
//             *begin_o = op_range(*begin_i, r);
//         }
//         return output.set_mutability(t.is_mutable());
//     }

//     Tensor output = Tensor::makeNullTensorArray(r.length() * t.numel());
//     Tensor *begin_o = reinterpret_cast<Tensor *>(output.data_ptr());

//     for (; begin_i != end_i; ++begin_i) {
//         Tensor o = begin_i->split_axis(0);
//         Tensor *o_b = reinterpret_cast<Tensor *>(o.data_ptr()) + r.begin;
//         Tensor *o_e = reinterpret_cast<Tensor *>(o.data_ptr()) + r.end;
//         for (; o_b != o_e; ++o_b, ++begin_o) {
//             *begin_o = *o_b;
//         }
//     }
//     output.set_mutability(t.is_mutable());
//     return std::move(output);
// }


// Fine as is
Tensor op_range(Tensor t, range_ r){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
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
    out.set_mutability(t.is_mutable());
    return std::move(out);
}

//I am considering making this just transpose(0, idx) -> op_range(t, r[idx]) -> transpose(0, idx)
Tensor op_range(const Tensor& t, std::vector<range_> r){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
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

    std::vector<std::pair<int64_t, int64_t>> range_numels = details::make_range_vector(r, t.shape(), t.numel());

    // while (r.size() > 0 && idx_is_total(r.back(), t.shape(), r.size() - 1)) {
    //     r.pop_back();
    // }
    // if (r.size() == 1) {
    //     return op_range(t, r[0]);
    // } else if (r.size() == 0) {
    //     return t;
    // }
    // for (Tensor::size_value_t i = 0; i < r.size(); ++i) {
    //     r[i].fix(t.shape()[i]);
    // }
    // size_t i = 0;
    // while(idx_is_total(r[i], t.shape(), i)){
    //     ++i;
    // }

    // Tensor outs = i == 0 ? get_range(t, r[0], 0) : t.split_axis(i-1);
    // if(i == 0) ++i;
    // for (; i < r.size(); ++i) {
    //     outs = get_range(outs, r[i], i);
    // }
    std::vector<Tensor::size_value_t> n_shape = t.shape().Vec();
    for (Tensor::size_value_t i = 0; i < r.size(); ++i){
        r[i].fix(t.shape()[i]);
        n_shape[i] = r[i].length();
    }
    Tensor out = t.arr_void().get_bucket().range<Tensor>(std::move(range_numels)).view(SizeRef(n_shape));
    out.set_mutability(t.is_mutable());
    return std::move(out);
}

}
}
