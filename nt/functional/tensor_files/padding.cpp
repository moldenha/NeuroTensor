#include "../../Tensor.h"
#include "exceptions.hpp"

namespace nt{
namespace functional{

Tensor pad(const Tensor& t, std::vector<Tensor::size_value_t> p, const char* mode, Scalar value){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    using size_value_t = Tensor::size_value_t;
    utils::THROW_EXCEPTION(
        p.size() % 2 == 0,
        "RuntimeError: The size of the pad must have 2 per dimension");
    utils::THROW_EXCEPTION(
        (p.size() / 2) <= t.dims(),
        "RuntimeError: expected padding for at most $ dims but instead got $",
        t.dims(), int(p.size() / 2));

    std::vector<size_value_t> n_shape = t.shape().Vec();
    {
        auto p_begin = p.crbegin();
        auto p_end = p.crend();
        auto sh_begin = n_shape.rbegin();
        for(;p_begin != p_end; ++p_begin, ++sh_begin){
            *sh_begin += *p_begin;
            ++p_begin;
            *sh_begin += *p_begin;
        }
    }
    
    Tensor output(SizeRef(std::move(n_shape)), t.dtype());
    output = value;
    std::vector<nt::range_> ranges(t.dims(), range);
    // for(int64_t i = 0; i < ranges.size(); ++i){
    //     ranges[i].end = output.shape()[i];
    // }
    {
        auto p_begin = p.crbegin();
        auto p_end = p.crend();
        auto r_begin = ranges.rbegin();
        int64_t index = -1;
        for(;p_begin != p_end; ++p_begin, ++r_begin, --index){
            r_begin->end = output.shape()[index]-*p_begin;
            ++p_begin;
            r_begin->begin = *p_begin;
        }
    }
    output[ranges].fill_(t);
    return std::move(output);
    
}
Tensor unpad(const Tensor& t, std::vector<Tensor::size_value_t> vec, bool no_contiguous){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    using size_value_t = Tensor::size_value_t;
    std::vector<range_> ranges(t.dims(), range);
    utils::throw_exception((vec.size()/2) <= ranges.size(),
                           "Cannot unpad greater than the dimensions of the tensor");
    auto in_shape = t.shape();
    for(int64_t i = 0; i < ranges.size(); ++i){
        ranges[i].end = in_shape[i];
    }
    auto begin = vec.crbegin();
    auto end = vec.crend();
    auto range_begin = ranges.rbegin();
    int64_t index = -1;
    for(;begin != end; ++begin, ++range_begin, --index){
        range_begin->end = in_shape[index]-*begin;
        ++begin;
        range_begin->begin = *begin;
    }
    if(no_contiguous) return t[std::move(ranges)];
    return t[std::move(ranges)].contiguous();
}


}
}



