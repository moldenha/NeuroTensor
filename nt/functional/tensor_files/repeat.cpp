#include "../../Tensor.h"
#include "../../dtype/ArrayVoid.hpp"
#include "../../refs/SizeRef.h"
#include "combine.h"
#include <vector>
#include "exceptions.hpp"

namespace nt {
namespace functional {

Tensor repeat_(const Tensor &t, Tensor::size_value_t amt) {
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    using size_value_t = Tensor::size_value_t;
    if (t.dtype() == DType::TensorObj) {
        Tensor output = Tensor::makeNullTensorArray(amt * t.numel());
        t.arr_void()
            .cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj>>>(
                [&output, &amt](auto start, auto end) {
                    Tensor *begin =
                        reinterpret_cast<Tensor *>(output.data_ptr());
                    auto start_cpy = start;
                    for (size_value_t i = 0; i < amt; ++i) {
                        for (; start != end; ++start, ++begin) {
                            *begin = *start;
                        }
                        start = start_cpy;
                    }
                });
        return std::move(output);
    }
    Tensor output = Tensor::makeNullTensorArray(amt);
    Tensor *begin = reinterpret_cast<Tensor *>(output.data_ptr());
    Tensor *end = begin + output.numel();
    for (; begin != end; ++begin)
        *begin = t;
    output.set_mutability(t.is_mutable());
    return cat(output);
}

Tensor repeat_(const Tensor &t, Tensor::size_value_t dim,
               Tensor::size_value_t amt) {
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    using size_value_t = Tensor::size_value_t;
    dim = dim < 0 ? dim + t.dims() : dim;
    if (dim == 0) {
        return repeat_(t, amt).set_mutability(t.is_mutable());
    }
    Tensor transposed = t.transpose(0, dim);
    return repeat_(transposed, amt).transpose(0, dim);
}

Tensor expand(const Tensor &t, SizeRef s) {
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    using size_value_t = Tensor::size_value_t;
    const auto &shape = t.shape();
    if (shape == s)
        return t;
    utils::THROW_EXCEPTION(
        s.size() >= shape.size(),
        "Expected to expand with same dimensions but got $ compared to $",
        s.size(), t.dims());

    if (s.size() > shape.size())
        return expand(t.unsqueeze_as(s), std::move(s)).set_mutability(t.is_mutable());;

    using size_value_t = Tensor::size_value_t;
    std::vector<size_value_t> out_shape = t.shape().Vec();
    std::vector<std::pair<size_value_t, size_value_t>>
        expandings; // which dimensions to repeat by how many
    size_value_t repeat_amt = 1;
    for (int64_t i = 0; i < s.size(); ++i) {
        if (s[i] != shape[i] && s[i] != 1) {
            utils::THROW_EXCEPTION(
                shape[i] == 1,
                "The expanded size of the tensor ($) must match "
                "the existing size ($) at non-singleton dimension "
                "$.  Target sizes: $.  Tensor sizes: $",
                s[i], shape[i], i, s, shape);
            expandings.push_back({i, s[i]});
            repeat_amt *= s[i];
            out_shape[i] = s[i];
        }
    }

    if (expandings.size() == 0)
        return t;

    // is faster if it is all 1's for the first n dimensions that are expanded:
    bool only_repeat = true;
    if (expandings[0].first != 0) {
        only_repeat = false;
    } else {
        for (size_t i = 1; i < expandings.size(); ++i) {
            if (expandings[i].first != (expandings[i - 1].first + 1)) {
                only_repeat = false;
                break;
            }
        }
    }
    if (only_repeat) {
        Tensor expanded = repeat_(t.flatten(0, -1), repeat_amt).view(SizeRef(std::move(out_shape)));
        return std::move(expanded);
    }
    Tensor expanded = repeat_(t, expandings[0].first, expandings[0].second);
    for (size_t i = 1; i < expandings.size(); ++i)
        expanded = repeat_(expanded, expandings[i].first, expandings[i].second);
    return std::move(expanded);
}

Tensor expand_as(const Tensor &i, const Tensor &t) {
    _NT_FUNCTIONAL_ALWAYS_CHECK_(i, t);
    if (i.shape() == t.shape())
        return i;
    return expand(i, t.shape());
}

} // namespace functional
} // namespace nt
