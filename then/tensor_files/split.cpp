#include "split.h"
#include "combine.h"

namespace nt{
namespace functional{

Tensor split(Tensor input, typename SizeRef::value_type split_size, int64_t dim){
	dim = (dim < 0) ? dim + input.dims() : dim;
	utils::THROW_EXCEPTION(dim < input.dims(), "Expected (dim = $) to be less than dims of input which is $", dim, input.dims());
	utils::THROW_EXCEPTION(dim >= 0, "Expected (dim = $) to be greater than or equal to zero", dim);
    if(dim != 0) return split(input.transpose(0, dim), split_size, 0).transpose(0, dim);
	typename SizeRef::value_type total_tensors = input.shape()[dim] / split_size;
	bool remainder = false;
	if(input.shape()[dim] % split_size != 0){++total_tensors; remainder = true;}

	// Tensor output({total_tensors}, DType::TensorObj);
    Tensor output = Tensor::makeNullTensorArray(total_tensors);
    typename SizeRef::value_type begin = 0;
    typename SizeRef::value_type end = split_size;
    if(!remainder){
        for(typename SizeRef::value_type i = 0; i < total_tensors; ++i){
            output[i] = input[my_range(begin, end)];
            begin += split_size;
            end += split_size;
        }
        return std::move(output);
    }
    for(typename SizeRef::value_type i = 0; i < total_tensors-1; ++i){
        output[i] = input[my_range(begin, end)];
        begin += split_size;
        end += split_size;
    }
    output[total_tensors-1] = input[my_range(begin, -1)];
    return std::move(output);

}


Tensor split(Tensor input, std::vector<typename SizeRef::value_type> split_sections, int64_t dim){
	dim = (dim < 0) ? dim + input.dims() : dim;
	typename SizeRef::value_type sum = std::accumulate(split_sections.cbegin(), split_sections.cend(), 0);
	utils::THROW_EXCEPTION(dim < input.dims(), "Expected (dim = $) to be less than dims of input which is $", dim, input.dims());
	utils::THROW_EXCEPTION(dim >= 0, "Expected (dim = $) to be greater than or equal to zero", dim);
	utils::THROW_EXCEPTION(sum == input.shape()[dim], "Expected the sum of split_sections to be equal to the shape along dim $ which is $, instead got $",   dim, input.shape()[dim], sum);

	if(dim == 0){
        Tensor output = Tensor::makeNullTensorArray(static_cast<typename SizeRef::value_type>(split_sections.size()));
        Tensor* o_begin = reinterpret_cast<Tensor*>(output.data_ptr());
		typename SizeRef::value_type begin = 0;
		for(typename SizeRef::value_type i = 0; i < split_sections.size(); ++i, ++o_begin){
            // std::cout << "doing range from "<<begin<<" to "<<split_sections[i]<<std::endl;
			*o_begin = input[my_range(begin, split_sections[i]+begin)];
			begin += split_sections[i];
		}
		return std::move(output);
	}
    Tensor output = split(input.transpose(0, dim), std::move(split_sections), 0);
    Tensor* begin = reinterpret_cast<Tensor*>(output.data_ptr());
    Tensor* end = begin + output.numel();
    for(;begin != end; ++begin)
        *begin = begin->transpose(0, dim);
    return std::move(output);

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


Tensor chunk(Tensor input, typename Tensor::size_value_t chunks, int64_t dim){
	dim = (dim < 0) ? dim + input.dims() : dim;
	utils::THROW_EXCEPTION(dim < input.dims(), "Expected (dim = $) to be less than dims of input which is $", dim, input.dims());
	utils::THROW_EXCEPTION(dim >= 0, "Expected (dim = $) to be greater than or equal to zero", dim);
	Tensor output = Tensor::makeNullTensorArray(chunks);
	typename SizeRef::value_type adding = input.shape()[dim] / chunks;
	if(dim == 0){
		typename SizeRef::value_type begin = 0;
		typename SizeRef::value_type end = adding;
        Tensor* o_begin = reinterpret_cast<Tensor*>(output.data_ptr());
		for(typename SizeRef::value_type i = 0; i < chunks-1; ++i, ++o_begin){
			*o_begin = input[my_range(begin, end)];
			begin += adding;
			end += adding;
		}
		*o_begin = input[my_range(begin, -1)];
		return std::move(output);
	}
	std::vector<Tensor> vec = get_all(input);
	int64_t dim_cpy = dim;
	--dim;
	while(dim > 0){
		vec = get_all(vec);
		--dim;
	}
	typename SizeRef::value_type begin = 0;
	typename SizeRef::value_type end = adding;
	auto n_shape = input.shape().Vec();
	n_shape[dim_cpy] = adding;
	SizeRef curr_shape(n_shape);
    Tensor* o_begin = reinterpret_cast<Tensor*>(output.data_ptr());
	for(typename SizeRef::value_type i = 0; i < chunks-1; ++i, ++o_begin){
		std::vector<Tensor> vec_cpy(vec.size());
		for(typename SizeRef::value_type j = 0; j < vec.size(); ++j){
			vec_cpy[j] = vec[i][my_range(begin, end)];
		}
		*o_begin = cat_unordered(vec_cpy).view(curr_shape);
		begin += adding;
		end += adding;
	}
	n_shape[dim_cpy] = input.shape()[dim_cpy] - (adding * (chunks-1));


	std::vector<Tensor> vec_cpy(vec.size());
	for(typename SizeRef::value_type j = 0; j < vec.size(); ++j){
		vec_cpy[j] = vec[j][my_range(begin, -1)];
	}
	*o_begin = cat_unordered(vec_cpy).view(SizeRef(std::move(n_shape)));
	return std::move(output);
}

}
}
