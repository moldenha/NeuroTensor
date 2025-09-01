#include "../../Tensor.h"
#include "combine.h"
#include "compare.h"
#include "../../mp/Threading.h"
#include "exceptions.hpp"
#include "../../utils/macros.h"
#include "../../dtype/ArrayVoid.hpp"

namespace nt{
namespace functional{



Tensor at(const Tensor& t, Tensor::size_value_t x){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    using size_value_t = Tensor::size_value_t;

    if(t.dims() == 1){
        if (t.numel() == 1) {
            utils::THROW_EXCEPTION(
                x == 0 || x == -1,
                "\nRuntimeError: Expected singleton to have indexed "
                "of at most $ but instead got $",
                0, x);
            return t;
        }
        x = x < 0 ? x + t.shape()[0] : x;
        utils::THROW_EXCEPTION(
            x < t.shape()[0], "RuntimeError: Expected x to be less than $ but got $",
            t.shape()[0], x);

        return Tensor(t.arr_void().share_array(x, 1), SizeRef({1}));
    }
    x = x < 0 ? x + t.dims() : x;
    uint64_t nx = static_cast<uint64_t>(x);
    if (t.numel() == 1) {
        utils::THROW_EXCEPTION(
            x == 0,
            "\nRuntimeError: Expected singleton to have indexed "
            "of at most $ but instead got $",
            0, x);
        return t;
    }
    utils::THROW_EXCEPTION(
        x < t.shape()[0], "RuntimeError: Expected x to be less than $ but got $",
        t.shape()[0], x);
    SizeRef n_size = t.shape().pop_front();
    uint64_t mult = static_cast<uint64_t>(n_size.multiply());
    return Tensor(t.arr_void().share_array(nx * mult, mult), std::move(n_size)).set_mutability(t.is_mutable());

}

Tensor at(const Tensor& self, const Tensor &t) {
    _NT_FUNCTIONAL_ALWAYS_CHECK_(self, t);
    using size_value_t = Tensor::size_value_t;

    utils::THROW_EXCEPTION(
        t.dtype() == DType::Bool || t.dtype() == DType::TensorObj || t.dtype() == DType::int64,
        "RuntimeError: expected DType Bool, TensorObj, or int64 but got $", t.dtype());
    if (t.dtype() == DType::TensorObj) {
        //if it is operations of tensors of tensors, then jus repeat the operation
        if(self.dtype() == DType::TensorObj){
            Tensor output = Tensor::makeNullTensorArray(self.numel());
            Tensor* ts_begin = reinterpret_cast<Tensor*>(output.data_ptr());
            Tensor* ts_end = ts_begin + self.numel();
            const Tensor* begin = reinterpret_cast<const Tensor*>(self.data_ptr());
            const Tensor* t_begin = reinterpret_cast<const Tensor*>(t.data_ptr());
            for(;ts_begin != ts_end; ++ts_begin, ++begin, ++t_begin)
                *ts_begin = (*begin)[*t_begin];
            return output.set_mutability(self.is_mutable());
        }
        utils::THROW_EXCEPTION(
            t.is_contiguous(),
            "RuntimeError: Expected indexing tensor to be contiguous");
        utils::THROW_EXCEPTION(
            t.numel() == self.dims(),
            "Expected indexing tensor to have $ tensors but had $", self.dims(),
            t.numel());
        ArrayVoid my_vals = self.arr_void().bucket_all_indices();
        const Tensor *begin = reinterpret_cast<const Tensor *>(t.data_ptr());
        const Tensor *end = begin + t.numel();
        const Tensor *begin_cpy = begin;
        for (; begin != end; ++begin)
            utils::THROW_EXCEPTION(
                begin->dtype() == DType::int64 && begin->is_contiguous(),
                "Expected indexing tensor to have dtype int64 but got $ and expected to be contiguous",
                begin->dtype());
        begin = begin_cpy;

        // making the strides for indexing:
        const std::vector<size_value_t> s = self.strides();
        std::vector<size_value_t> ns(s.size());
        std::copy(s.begin(), s.end(), ns.begin());

        // keeping track of each int64_t pointer for the indexing
        NT_VLA(const size_value_t*, ptrs, self.dims());
        // const size_value_t *ptrs[self.dims()];
        size_value_t i = 0;
        for (; begin != end; ++begin, ++i) {
            ptrs[i] = reinterpret_cast<const size_value_t *>(begin->data_ptr());
        }
        // making a new ArrayVoid to keep track of all the indices
        const size_value_t &n_size = begin_cpy->numel();
        ArrayVoid new_vals = self.arr_void().new_strides(n_size);
        void **out_begin = new_vals.stride_begin();
        void **my_begin = my_vals.stride_begin();
        // finding each index
        for (size_value_t i = 0; i < n_size; ++i, ++out_begin) {
            // getting the ith index to copy
            size_value_t index = 0;
            for (size_value_t j = 0; j < t.numel() - 1; ++j) {
                index += ptrs[j][i] * ns[j + 1];
            }
            index += ptrs[t.numel() - 1][i];
            *out_begin = my_begin[index];
        }
        NT_VLA_DEALC(ptrs);
        Tensor output(new_vals, {static_cast<size_value_t>(n_size)});
        return output.set_mutability(self.is_mutable()); 
    }
    else if (t.dtype() == DType::int64){
       utils::THROW_EXCEPTION(
            t.is_contiguous(),
            "RuntimeError: Expected indexing tensor to be contiguous");
        const int64_t* t_begin = reinterpret_cast<const int64_t*>(t.data_ptr());
        const int64_t* t_end = reinterpret_cast<const int64_t*>(t.data_ptr_end());
        if(self.dims() == 1){
            const size_value_t &n_size = t.numel();
            if(self.dtype() == DType::TensorObj){
                //it is important that it is a null tensor array and that each tensor is coppied
                //otherwise ownership is not properly handled
                //and when going out of scope, there will be a double-free error
                Tensor output = Tensor::makeNullTensorArray(n_size);
                self.arr_void().cexecute_function([&](auto my_begin, auto my_end){
                    const int64_t& max =self.shape()[0];
                    Tensor* out_begin = reinterpret_cast<Tensor*>(output.data_ptr());
                    for(;t_begin != t_end; ++t_begin, ++out_begin){
                        utils::THROW_EXCEPTION(*t_begin < max,
                                        "Trying to get index of tensor at dim 0 of $ with that dimension only holding $", *t_begin, max);
                        *out_begin = my_begin[*t_begin];
                    }
 
                });
                return std::move(output);
            }
            ArrayVoid my_vals = self.arr_void().bucket_all_indices();
            ArrayVoid new_vals = self.arr_void().new_strides(n_size);
            void **out_begin = new_vals.stride_begin();
            void **my_begin = my_vals.stride_begin();
            const int64_t& max =self.shape()[0];
            for(;t_begin != t_end; ++t_begin, ++out_begin){
                utils::THROW_EXCEPTION(*t_begin < max,
                                "Trying to get index of tensor at dim 0 of $ with that dimension only holding $", *t_begin, max);
                *out_begin = my_begin[*t_begin];
            }
            Tensor output(new_vals, {static_cast<size_value_t>(n_size)});
            return output.set_mutability(self.is_mutable()); 

        }
        std::vector<SizeRef::value_type> Vec = self.shape().Vec();
        Vec[0] = t.numel();
        Tensor split = self.split_axis(0);
        std::vector<Tensor> catting(t.numel(), Tensor::Null());
        auto out_begin = catting.begin();
        Tensor* s_begin = reinterpret_cast<Tensor*>(split.data_ptr());
        for(;t_begin != t_end; ++t_begin, ++out_begin){
            *out_begin = s_begin[*t_begin];
        }
        return functional::cat(std::move(catting)).view(SizeRef(std::move(Vec)));
    
    }
    utils::THROW_EXCEPTION(
        t.dtype() == DType::Bool,
        "RuntimeError (at end, logic error): expected DType Bool, TensorObj, or int64 but got $", t.dtype());
    utils::THROW_EXCEPTION(
        t.is_contiguous(),
        "RuntimeError: Expected indexing tensor to be contiguous");
    if(t.numel() != self.numel() && t.numel() ==self.shape()[0]){
        const uint_bool_t *begin =
            reinterpret_cast<const uint_bool_t *>(t.data_ptr());
        const uint_bool_t *end = begin + t.numel();

        std::vector<SizeRef::value_type> Vec = self.shape().Vec();
        Vec[0] = functional::count(t);
        Tensor split = self.split_axis(0);
        std::vector<Tensor> catting(functional::count(t), Tensor::Null());
        auto out_begin = catting.begin();
        Tensor* s_begin = reinterpret_cast<Tensor*>(split.data_ptr());
        for(;begin != end; ++begin, ++s_begin){
            if(*begin){
                *out_begin++ = *s_begin;
            }
        }
        return functional::cat(std::move(catting)).view(SizeRef(std::move(Vec)));
 
    }
    utils::THROW_EXCEPTION(
        t.numel() == self.numel(),
        "Numels must be equal for [] operator on Tensor DType::Bool, or equal to shape()[0] ($)",self.shape()[0]);
    
    if(self.dtype() == DType::TensorObj){
        size_value_t new_size = ::nt::functional::count(t);
        Tensor new_vals = Tensor::makeNullTensorArray(new_size);
        self.arr_void().cexecute_function([&](auto my_strides, auto my_strides_end){
            Tensor* new_strides = reinterpret_cast<Tensor*>(new_vals.data_ptr());
            const uint_bool_t *begin =
                reinterpret_cast<const uint_bool_t *>(t.data_ptr());
            const uint_bool_t *end = begin + self.numel();
            for(; begin != end; ++begin, ++my_strides){
                if(*begin == true){
                    *new_strides = *my_strides;
                    ++new_strides;
                }
            }
        });
        new_vals.set_mutability(self.is_mutable());
        return std::move(new_vals);
    }

    ArrayVoid my_vals = self.arr_void().bucket_all_indices();
    size_value_t new_size = ::nt::functional::count(t);
    ArrayVoid new_vals = self.arr_void().new_strides(new_size);
    const uint_bool_t *begin =
        reinterpret_cast<const uint_bool_t *>(t.data_ptr());
    const uint_bool_t *end = begin + self.numel();
    void **my_stride = my_vals.stride_begin();
    void **new_stride = new_vals.stride_begin();
    for (; begin != end; ++begin, ++my_stride) {
        if (*begin == true) {
            *new_stride = *my_stride;
            ++new_stride;
        }
    }
    Tensor output(std::move(new_vals), {static_cast<size_value_t>(new_size)});
    return output.set_mutability(self.is_mutable()); 
}


Tensor at(const Tensor& self, std::vector<Tensor::size_value_t> xs){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(self);
    using size_value_t = Tensor::size_value_t;

    utils::THROW_EXCEPTION(
            xs.size() <= self.dims(),
            "Expected to get less than or equal to $ indices but got $ indices",
            self.dims(), xs.size());
    if(xs.size() == 1){return at(self, xs[0]);}
    if(xs.size() == 0){return self;}
    for (size_value_t i = 0; i < xs.size(); ++i) {
        xs[i] = xs[i] < 0 ? xs[i] + self.dims() : xs[i];
    }
    
    uint64_t cur_mult = 0;
    auto begin = xs.begin();
    auto end = xs.end();
    SizeRef n_size = self.shape().pop_front();
    cur_mult += (*begin * n_size.multiply());
    ++begin;
    for(;begin != end; ++begin){
        n_size = n_size.pop_front();
        cur_mult += (*begin * n_size.multiply());
    }
    uint64_t mult = n_size.size() == 0 ? 1 : static_cast<uint64_t>(n_size.multiply());
    Tensor output(self.arr_void().share_array(cur_mult, mult), std::move(n_size));
    return output.set_mutability(self.is_mutable()); 
}
Tensor index_except(const Tensor& self, int64_t dim, Tensor::size_value_t excluding_index) {
    _NT_FUNCTIONAL_ALWAYS_CHECK_(self);
    using size_value_t = Tensor::size_value_t;

    dim = dim < 0 ? dim + self.dims() : dim;
    utils::THROW_EXCEPTION(dim < self.dims() && dim >= 0, "Got invalid dim $", dim);
    bool end_dim = (dim == dim-1);
    auto sh = self.shape();
    excluding_index = excluding_index < 0 ? excluding_index + sh[dim] : excluding_index;
    utils::THROW_EXCEPTION(excluding_index < sh[dim] && excluding_index >= 0, "Got invalid index $", excluding_index);
    std::vector<size_value_t> Vec = sh.Vec();
    Vec[dim] -= 1;

    Tensor split = self.split_axis(dim);
    int64_t total = (split.numel() / sh[dim]) * Vec[dim];
    Tensor out = Tensor::makeNullTensorArray(total);
    Tensor* o_begin = reinterpret_cast<Tensor*>(out.data_ptr());
    Tensor* s_begin = reinterpret_cast<Tensor*>(split.data_ptr());
    Tensor* s_end = reinterpret_cast<Tensor*>(split.data_ptr_end());
    int64_t counter = 0;
    const int64_t max_dim = (sh[dim]-1);
    while(s_begin != s_end){
        if(counter != excluding_index){
            *o_begin = *s_begin;
            ++o_begin;
        }
        ++s_begin;
        counter = counter == max_dim ? 0 : counter + 1;
    }
    if(end_dim){
        std::swap(Vec[dim], Vec[dim-1]);
        return functional::cat_unordered(out).view(SizeRef(std::move(Vec))).transpose(-1, -2);
    }
    return functional::cat_unordered(out).view(SizeRef(std::move(Vec)));
}

Tensor at_tensor_split(const Tensor& _self, const Tensor& _index, Tensor::size_value_t splitting){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(_self, _index);
    using size_value_t = Tensor::size_value_t;

    // Tensor _output(_index.shape(), _self.dtype());
    Tensor self = _self.split_axis(splitting);
    Tensor index = _index.split_axis(splitting);
    // Tensor output = _output.split_axis(splitting);
    utils::throw_exception(self.numel() == index.numel(),
                           "Splitting at dimension $ leads to different numbers of tensors for input ($) and indexing tensor ($)",
                           splitting, self.numel(), index.numel());

    Tensor* self_begin = reinterpret_cast<Tensor*>(self.data_ptr());
    Tensor* index_begin = reinterpret_cast<Tensor*>(index.data_ptr());

    Tensor output = Tensor::makeNullTensorArray(self.numel());
    Tensor* output_begin = reinterpret_cast<Tensor*>(output.data_ptr());
    threading::preferential_parallel_for(threading::block_ranges<1>(0, self.numel()),
    [&](threading::blocked_range<1> block){
	for (Tensor::size_value_t i = block.begin[0]; i < block.end[0]; ++i) {
        output_begin[i] = at(self_begin[i], index_begin[i]);
    }
    });
    return cat_unordered(output).view(_index.shape());

}

Tensor& at_tensor_split_out_equal_shape(const Tensor& _self, const Tensor& _index, Tensor::size_value_t splitting, Tensor& _output){
    utils::throw_exception(_output.is_mutable(), "output from at tensor split must be mutable");
    _NT_FUNCTIONAL_ALWAYS_CHECK_(_self, _index);
    using size_value_t = Tensor::size_value_t;

    Tensor self = _self.split_axis(splitting);
    Tensor index = _index.split_axis(splitting);
    Tensor output = _output.split_axis(splitting);
    utils::throw_exception(self.numel() == index.numel() && self.numel() == output.numel(),
                           "Splitting at dimension $ leads to different numbers of tensors for input ($) and indexing tensor ($)",
                           splitting, self.numel(), index.numel());

    Tensor* self_begin = reinterpret_cast<Tensor*>(self.data_ptr());
    Tensor* index_begin = reinterpret_cast<Tensor*>(index.data_ptr());
    Tensor* output_begin = reinterpret_cast<Tensor*>(output.data_ptr());
    threading::preferential_parallel_for(threading::block_ranges<1>(0, self.numel()),
    [&](threading::blocked_range<1> block){
	for (Tensor::size_value_t i = block.begin[0]; i < block.end[0]; ++i) {
        output_begin[i].set_(at(self_begin[i], index_begin[i]));
    }
    });
    return _output;

}

Tensor& at_tensor_split(const Tensor& _self, const Tensor& _index, Tensor::size_value_t splitting, Tensor& _output){
    utils::throw_exception(_output.is_mutable(), "output from at tensor split must be mutable");
    _NT_FUNCTIONAL_ALWAYS_CHECK_(_self, _index);
    using size_value_t = Tensor::size_value_t;

    if(_output.shape() == _index.shape()){
        return at_tensor_split_out_equal_shape(_self, _index, splitting, _output);
    }
    Tensor self = _self.split_axis(splitting);
    Tensor index = _index.split_axis(splitting);
    Tensor output = _output.split_axis(splitting);
    utils::throw_exception(self.numel() == index.numel() && self.numel() == output.numel(),
                           "Splitting at dimension $ leads to different numbers of tensors for input ($) and indexing tensor ($)",
                           splitting, self.numel(), index.numel());

    Tensor* self_begin = reinterpret_cast<Tensor*>(self.data_ptr());
    Tensor* index_begin = reinterpret_cast<Tensor*>(index.data_ptr());
    Tensor* output_begin = reinterpret_cast<Tensor*>(output.data_ptr());
    threading::preferential_parallel_for(threading::block_ranges<1>(0, self.numel()),
    [&](threading::blocked_range<1> block){
	for (Tensor::size_value_t i = block.begin[0]; i < block.end[0]; ++i) {
        output_begin[i][index_begin[i]].set_(at(self_begin[i], index_begin[i]));
    }
    });
    return _output;

}



inline std::vector<Tensor> get_all(Tensor& t){
	std::vector<Tensor> output(t.shape()[0]);
	for(typename SizeRef::value_type i = 0; i < t.shape()[0]; ++i)
		output[i] = t[i];
	return std::move(output);
}

inline std::vector<Tensor> get_all(std::vector<Tensor>& ts){
	std::vector<Tensor> output(ts[0].shape()[0]*ts.size());
	typename SizeRef::value_type a_counter = 0;
	typename SizeRef::value_type b = ts[0].shape()[0];
	typename SizeRef::value_type a = 0;
	typename SizeRef::value_type ts_counter = 0;
	for(typename SizeRef::value_type i = 0; i < output.size(); ++i){
		output[i] = ts[ts_counter][a_counter];
		if(++a_counter == b){
			++ts_counter;
			a_counter = a;
		}
	}
	return std::move(output);
}

Tensor get_indices(std::vector<Tensor>& ts, int64_t* begin, int64_t* end){
	std::ptrdiff_t diff = std::distance(begin, end);
    Tensor output = Tensor::makeNullTensorArray(diff * ts.size());
    Tensor* output_ = reinterpret_cast<Tensor*>(output.data_ptr());
	int64_t* begin_cpy = begin;
	uint64_t index = 0;
	for(uint64_t i = 0; i < output.numel(); ++index){
		for(;begin != end; ++begin, ++i){
			Tensor cur = ts[index][*begin];
            output_[i] = cur;
        }
		begin = begin_cpy;
	}
	return std::move(output);

}

Tensor index_select(Tensor input, int64_t dim, Tensor index){
	dim = (dim < 0) ? dim + input.dims() : dim;
	utils::THROW_EXCEPTION(dim < input.dims(), "Expected (dim = $) to be less than dims of input which is $", dim, input.dims());
	utils::THROW_EXCEPTION(dim >= 0, "Expected (dim = $) to be greater than or equal to zero", dim);
	utils::THROW_EXCEPTION(index.dims() == 1, "Expected indexing tensor to have a dimensional size of 1 but got $", index.dims());
	utils::THROW_EXCEPTION(index.dtype() == DType::int64, "Expected indexing tensor to be dtype int64 but got $", index.dtype());
    index = index.contiguous();
	if(dim == 0){
		std::vector<Tensor> output(index.numel());
		int64_t* begin = reinterpret_cast<int64_t*>(index.data_ptr());
		int64_t* end = reinterpret_cast<int64_t*>(index.data_ptr_end());
		auto setting = output.begin();
		for(;begin != end; ++begin, ++setting)
			*setting = input[*begin];
		return cat(output);
	}
	auto n_shape = input.shape().Vec();
	n_shape[dim] = index.numel();
	std::vector<Tensor> output = get_all(input);
	--dim;
	while(dim > 0){
		output = get_all(output);
		--dim;
	}
	return cat_unordered(
        get_indices(output, reinterpret_cast<int64_t*>(index.data_ptr()), 
                    reinterpret_cast<int64_t*>(index.data_ptr_end()))).view(SizeRef(std::move(n_shape)));
}


Tensor select(Tensor input, int64_t dim, int64_t index){
	dim = (dim < 0) ? dim + input.dims() : dim;
	if(dim == 0)
		return input[index];
	std::vector<range_> ranges(dim+1, range);
	// for(size_t i = 0; i < ranges.size(); ++i){
        // ranges[i] = range_(0, input.shape()[i]);
    // }
    ranges.back() = range_(index, index+1);
    
	return input[std::move(ranges)];
}

}
}
