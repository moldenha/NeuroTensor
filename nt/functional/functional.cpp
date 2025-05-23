#include <cstdint>

#include "functional.h"
#include "../Tensor.h"
#include "../memory/iterator.h"
#include "../refs/SizeRef.h"
#include "../dtype/ArrayVoid.h"
#include "../dtype/DType.h"
#include "../dtype/DType_enum.h"
#include "../mp/simde_ops.h"
#include "../convert/std_convert.h"
#include "tensor_files/exceptions.hpp"

//#include "TensorAccessor.h"


#include <atomic>
#include <functional>
//#include <i386/types.h>
#include <memory.h>
#include <memory>
#include <algorithm>
#include <numeric>
#include <random>
#include <ratio>
#include <iterator>

#include <cassert>
//#include <format>
#include <sys/_types/_int32_t.h>
#include <sys/_types/_int64_t.h>
#include <sys/types.h>
#include <type_traits>
#include <vector>
#include "../utils/utils.h"
#include <chrono>
#include "../permute/permute_old.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <random>
#include <cmath>
#include "../dtype/ArrayVoid.hpp"
#include "../mp/Threading.h"
#include <unordered_map>

#ifdef USE_PARALLEL
	#include <tbb/parallel_for_each.h>
	#include <tbb/parallel_for.h>
	#include <tbb/blocked_range.h>
	#include <tbb/blocked_range2d.h>
	#include <tbb/blocked_range3d.h>
	#include <tbb/parallel_reduce.h>
	#include <thread>
	/* #include "../mp/MP.hpp" */
	/* #include "../mp/Pool.hpp" */
#endif
#define assertm(exp, msg) assert(((void)msg, exp))


namespace nt{
namespace functional{





Tensor all(const Tensor t, int64_t dim){
    if(t.dtype == DType::TensorObj){
        Tensor out = Tensor::makeNullTensorArray(t.numel());
        Tensor* begin_o = reinterpret_cast<Tensor*>(out.data_ptr());
        t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&begin_o, &dim](auto begin, auto end){
            for(;begin != end; ++begin, ++begin_o){
                *begin_o = all(*begin, dim);
            }
		});
        return std::move(out);
    }
	exception_dtypes(t.dtype, DType::Bool);
    Tensor a = Tensor::Null();
    if(dim == (t.dims()-1) || dim == -1){
        a = t.transpose(-1, -2).contiguous();
        dim = -2;
    }else{
        a = t.contiguous();
    }
    Tensor split = a.split_axis(dim);
    
    Tensor out({split.numel()}, nt::DType::Bool);
    bool* begin_o = reinterpret_cast<bool*>(out.data_ptr());
    Tensor* begin_s = reinterpret_cast<Tensor*>(split.data_ptr());
    tbb::parallel_for(
    utils::calculateGrainSize1D(split.numel()),
    [&](const tbb::blocked_range<int64_t> &range){
        for(int64_t i = range.begin(); i != range.end(); ++i){
            begin_o[i] = std::all_of(reinterpret_cast<uint_bool_t*>(begin_s[i].data_ptr()), reinterpret_cast<uint_bool_t*>(begin_s[i].data_ptr_end()),
                                     [](const uint_bool_t& v){return v.value == 1;});
        }
    });
    return std::move(out);
}


Tensor any(const Tensor t, int64_t dim){
    if(t.dtype == DType::TensorObj){
        Tensor out = Tensor::makeNullTensorArray(t.numel());
        Tensor* begin_o = reinterpret_cast<Tensor*>(out.data_ptr());
        t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&begin_o, &dim](auto begin, auto end){
            for(;begin != end; ++begin, ++begin_o){
                *begin_o = any(*begin, dim);
            }
		});
        return std::move(out);
    }
	exception_dtypes(t.dtype, DType::Bool);
    // Tensor a = t.contiguous();
    Tensor a = Tensor::Null();
    if(dim == (t.dims()-1) || dim == -1){
        a = t.transpose(-1, -2).contiguous();
        dim = -2;
    }else{
        a = t.contiguous();
    }
    Tensor split = a.split_axis(dim);
    Tensor out({split.numel()}, nt::DType::Bool);
    bool* begin_o = reinterpret_cast<bool*>(out.data_ptr());
    Tensor* begin_s = reinterpret_cast<Tensor*>(split.data_ptr());
    tbb::parallel_for(
    utils::calculateGrainSize1D(split.numel()),
    [&](const tbb::blocked_range<int64_t> &range){
        for(int64_t i = range.begin(); i != range.end(); ++i){
            begin_o[i] = std::any_of(reinterpret_cast<uint_bool_t*>(begin_s[i].data_ptr()), reinterpret_cast<uint_bool_t*>(begin_s[i].data_ptr_end()),
                                     [](const uint_bool_t& v){return v.value == 1;});
        }
    });
    return std::move(out);
}




std::vector<Tensor> get_all(Tensor& t){
	std::vector<Tensor> output(t.shape()[0]);
	for(typename SizeRef::value_type i = 0; i < t.shape()[0]; ++i)
		output[i] = t[i];
	return std::move(output);
}

std::vector<Tensor> get_all(std::vector<Tensor>& ts){
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

std::vector<Tensor> get_indices(std::vector<Tensor>& ts, int64_t* begin, int64_t* end){
	std::ptrdiff_t diff = std::distance(begin, end);
	std::vector<Tensor> output(diff*ts.size());
	int64_t* begin_cpy = begin;
	uint64_t index = 0;
	for(uint64_t i = 0; i < output.size(); ++i, ++index){
		for(;begin != end; ++begin, ++i)
			output[i] = ts[index][*begin];
		begin = begin_cpy;
	}
	return std::move(output);

}



Tensor index_select(Tensor input, int8_t dim, Tensor index){
	dim = (dim < 0) ? dim + input.dims() : dim;
	utils::THROW_EXCEPTION(dim < input.dims(), "Expected (dim = $) to be less than dims of input which is $", dim, input.dims());
	utils::THROW_EXCEPTION(dim >= 0, "Expected (dim = $) to be greater than or equal to zero", dim);
	utils::THROW_EXCEPTION(index.dims() == 1, "Expected indexing tensor to have a dimensional size of 1 but got $", index.dims());
	utils::THROW_EXCEPTION(index.dtype == DType::int64, "Expected indexing tensor to be dtype int64 but got $", index.dtype);
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

	return cat_unordered(get_indices(output, reinterpret_cast<int64_t*>(index.data_ptr()), reinterpret_cast<int64_t*>(index.data_ptr_end()))).view(SizeRef(std::move(n_shape)));
}

Tensor select(Tensor input, int8_t dim, int64_t index){
	dim = (dim < 0) ? dim + input.dims() : dim;
	if(dim == 0)
		return input[index];
	std::vector<my_range> ranges(dim+1, my_range());
	ranges.back() = my_range(index);
	return input[std::move(ranges)];
}

Tensor split(Tensor input, typename SizeRef::value_type split_size, int64_t dim){
	dim = (dim < 0) ? dim + input.dims() : dim;
	utils::THROW_EXCEPTION(dim < input.dims(), "Expected (dim = $) to be less than dims of input which is $", dim, input.dims());
	utils::THROW_EXCEPTION(dim >= 0, "Expected (dim = $) to be greater than or equal to zero", dim);
	typename SizeRef::value_type total_tensors = input.shape()[dim] / split_size;
	bool remainder = false;
	if(input.shape()[dim] % split_size != 0){++total_tensors; remainder = true;}
	Tensor output({total_tensors}, DType::TensorObj);
	if(dim == 0){
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
    return split(input.transpose(0, dim), split_size, 0).transpose(0, dim);
	// std::vector<Tensor> vec = get_all(input);
	// int8_t dim_cpy = dim;
	// --dim;
	// while(dim > 0){
	// 	vec = get_all(vec);
	// 	--dim;
	// }
	// typename SizeRef::value_type begin = 0;
	// typename SizeRef::value_type end = split_size;
	// auto n_shape = input.shape().Vec();
	// n_shape[dim_cpy] = split_size;
	// SizeRef curr_shape(n_shape);
	// if(!remainder){
	// 	for(typename SizeRef::value_type i = 0; i < total_tensors; ++i){
	// 		std::vector<Tensor> vec_cpy(vec.size());
	// 		for(typename SizeRef::value_type j = 0; j < vec.size(); ++j){
	// 			vec_cpy[j] = vec[i][my_range(begin, end)];
	// 		}
	// 		output[i] = cat_unordered(vec_cpy).view(curr_shape);
	// 		begin += split_size;
	// 		end += split_size;
	// 	}
	// 	return std::move(output);
	// }
	// for(typename SizeRef::value_type i = 0; i < total_tensors-1; ++i){
	// 	std::vector<Tensor> vec_cpy(vec.size());
	// 	for(typename SizeRef::value_type j = 0; j < vec.size(); ++j){
	// 		vec_cpy[j] = vec[i][my_range(begin, end)];
	// 	}
	// 	output[i] = cat_unordered(vec_cpy).view(curr_shape);
	// 	begin += split_size;
	// 	end += split_size;
	// }
	// std::vector<Tensor> vec_cpy(vec.size());
	// for(typename SizeRef::value_type j = 0; j < vec.size(); ++j){
	// 	vec_cpy[j] = vec[j][my_range(begin, -1)];
	// }
	// n_shape[dim_cpy] = input.shape()[dim_cpy] % split_size;
	// output[total_tensors-1] = cat_unordered(vec_cpy).view(SizeRef(std::move(n_shape)));
	// return std::move(output);
}

Tensor split(Tensor input, std::vector<typename SizeRef::value_type> split_sections, int64_t dim){
	dim = (dim < 0) ? dim + input.dims() : dim;
	typename SizeRef::value_type sum = std::accumulate(split_sections.cbegin(), split_sections.cend(), 0);
	utils::THROW_EXCEPTION(dim < input.dims(), "Expected (dim = $) to be less than dims of input which is $", dim, input.dims());
	utils::THROW_EXCEPTION(dim >= 0, "Expected (dim = $) to be greater than or equal to zero", dim);
	utils::THROW_EXCEPTION(sum == input.shape()[dim], "Expected the sum of split_sections to be equal to the shape along dim $ which is $, instead got $", (int)dim, input.shape()[dim], sum);

	if(dim == 0){
        Tensor output({static_cast<typename SizeRef::value_type>(split_sections.size())}, DType::TensorObj);
		typename SizeRef::value_type begin = 0;
		for(typename SizeRef::value_type i = 0; i < split_sections.size(); ++i){
            // std::cout << "doing range from "<<begin<<" to "<<split_sections[i]<<std::endl;
			output[i] = input[my_range(begin, split_sections[i]+begin)];
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
	// std::vector<Tensor> vec = get_all(input);
	// int8_t dim_cpy = dim;
	// --dim;
	// while(dim > 0){
	// 	vec = get_all(vec);
	// 	--dim;
	// }
	// typename SizeRef::value_type begin = 0;
	// auto n_shape = input.shape().Vec();
	// for(typename SizeRef::value_type i = 0; i < split_sections.size(); ++i){
	// 	std::vector<Tensor> vec_cpy(vec.size());
	// 	for(typename SizeRef::value_type j = 0; j < vec.size(); ++j){
	// 		vec_cpy[j] = vec[i][my_range(begin, split_sections[i])];
	// 	}
	// 	n_shape[dim_cpy] = split_sections[i];
	// 	output[i] = cat_unordered(vec_cpy).view(SizeRef(n_shape));
	// 	begin += split_sections[i];
	// }
	// return std::move(output);
}

template<typename Iterator>
inline bool _nt_sort_descending_(const int64_t& a, const int64_t& b, const Iterator& data){
    return data[a] > data[b];
}

template<typename Iterator>
inline bool _nt_sort_ascending_(const int64_t& a, const int64_t& b, const Iterator& data){
    return data[a] < data[b];
}


inline bool _nt_sort_descending_tensor_(const int64_t& a, const int64_t& b, const Tensor* data){
    if(data[b].numel() != data[a].numel()){return data[a].numel() > data[b].numel();}
    const ArrayVoid& arv = data[b].arr_void();
    return data[a].arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL> >([&arv](auto begin, auto end) -> bool{
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        return arv.cexecute_function<DTypeFuncs::type_to_dtype<value_t>>([&begin, &end](auto second, auto s_end) -> bool{
            for(;begin != end; ++begin, ++second){
                if(*second != *begin){return *begin > *second;}
            }
            return false;
        });
    });
}

inline bool _nt_sort_descending_tensor_valsonly_(const Tensor& a, const Tensor& b){
    if(b.numel() != a.numel()){return a.numel() > b.numel();}
    const ArrayVoid& arv = b.arr_void();
    return a.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL> >([&arv](auto begin, auto end) -> bool{
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        return arv.cexecute_function<DTypeFuncs::type_to_dtype<value_t>>([&begin, &end](auto second, auto s_end) -> bool{
            for(;begin != end; ++begin, ++second){
                if(*second != *begin){return *begin > *second;}
            }
            return false;
        });
    });
}


inline bool _nt_sort_ascending_tensor_(const int64_t& a, const int64_t& b, const Tensor* data){
    if(data[b].numel() != data[a].numel()){return data[a].numel() < data[b].numel();}
    const ArrayVoid& arv = data[b].arr_void();
    return data[a].arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL> >([&arv](auto begin, auto end) -> bool{
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        return arv.cexecute_function<DTypeFuncs::type_to_dtype<value_t>>([&begin, &end](auto second, auto s_end) -> bool{
            for(;begin != end; ++begin, ++second){
                if(*second != *begin){return *begin < *second;}
            }
            return false;
        });
    });
}

inline bool _nt_sort_ascending_tensor_valsonly_(const Tensor& a, const Tensor& b){
    if(a.numel() != b.numel()){return a.numel() < b.numel();}
    const ArrayVoid& arv = b.arr_void();
    return a.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL> >([&arv](auto begin, auto end) -> bool{
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        return arv.cexecute_function<DTypeFuncs::type_to_dtype<value_t>>([&begin, &end](auto second, auto s_end) -> bool{
            for(;begin != end; ++begin, ++second){
                if(*second != *begin){return *begin < *second;}
            }
            return false;
        });
    });
}


void sort_vals_only(Tensor& values, const bool& descending, const int64_t& dim_size){
    if(values.dtype != DType::TensorObj){
        values.arr_void().execute_function<WRAP_DTYPES<NumberTypesL> >(
            [&descending, &dim_size](auto begin, auto end){
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                using iterator_t = decltype(begin);
                if constexpr (std::is_pointer_v<iterator_t>){ // because values were cloned should all be contiguous, but this ensures it
                int64_t total = (end - begin) / dim_size;
#ifdef USE_PARALLEL
                if(descending){
                    tbb::parallel_for(
                        utils::calculateGrainSize1D(total),
                        [&](const tbb::blocked_range<int64_t> &range){
                            // auto s_begin = begin + (range.begin() * dim_size);
                            // auto s_end = begin + (range.end() * dim_size);
                            auto i_begin = begin + (range.begin() * dim_size);
                            auto i_end = begin + (range.end() * dim_size);
                            for(;i_begin < i_end; i_begin += dim_size){
                                std::sort(i_begin, i_begin + dim_size, std::greater<value_t>());
                            }
                    });
                }else{
                   tbb::parallel_for(
                        utils::calculateGrainSize1D(total),
                        [&](const tbb::blocked_range<int64_t> &range){
                            // auto s_begin = begin + (range.begin() * dim_size);
                            // auto s_end = begin + (range.end() * dim_size);
                            auto i_begin = begin + (range.begin() * dim_size);
                            auto i_end = begin + (range.end() * dim_size);
                            for(;i_begin < i_end; i_begin += dim_size){
                                std::sort(i_begin, i_begin + dim_size, std::less<value_t>());
                            }
                    });
     
                }
#else
                if(descending){
                for(;begin != _end; begin += dim_size){
                    std::sort(begin, begin + dim_size, std::greater<value_t>());
                }
                }else{
                for(;indices_begin != indices_end; indices_begin += dim_size){

                    std::sort(indices_begin, indices_begin + dim_size, std::less<value_t>());


                }

                }
#endif
            }
        }
        );
    }else{
        Tensor* begin = reinterpret_cast<Tensor*>(values.data_ptr());
        Tensor* end = reinterpret_cast<Tensor*>(values.data_ptr_end());
        int64_t total = (end - begin) / dim_size;
#ifdef USE_PARALLEL
        if(descending){
            tbb::parallel_for(
                utils::calculateGrainSize1D(total),
                [&](const tbb::blocked_range<int64_t> &range){
                    // auto s_begin = begin + (range.begin() * dim_size);
                    // auto s_end = begin + (range.end() * dim_size);
                    auto i_begin = begin + (range.begin() * dim_size);
                    auto i_end = begin + (range.end() * dim_size);
                    for(;i_begin < i_end; i_begin += dim_size){
                        std::sort(i_begin, i_begin + dim_size, _nt_sort_descending_tensor_valsonly_);
                    }
            });
        }else{
           tbb::parallel_for(
                utils::calculateGrainSize1D(total),
                [&](const tbb::blocked_range<int64_t> &range){
                    // auto s_begin = begin + (range.begin() * dim_size);
                    // auto s_end = begin + (range.end() * dim_size);
                    auto i_begin = begin + (range.begin() * dim_size);
                    auto i_end = begin + (range.end() * dim_size);
                    for(;i_begin < i_end; i_begin += dim_size){
                        std::sort(i_begin, i_begin + dim_size, _nt_sort_ascending_tensor_valsonly_);
                    }
            });

        }
#else
        if(descending){
        for(;begin != end; begin += dim_size){
            std::sort(begin, begin + dim_size, _nt_sort_descending_tensor_valsonly_);
        }
        }else{
        for(;begin != end; begin += dim_size){
            std::sort(begin, begin + dim_size, _nt_sort_ascending_tensor_valsonly_);
        }

        }
#endif
 
    }

}

Tensor sort(const Tensor& input, const Tensor::size_value_t dim, bool descending, bool return_sorted, bool return_indices){
    utils::throw_exception(return_sorted || return_indices, "Sort function must return indices or the sorted tensor");
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
        values.arr_void().execute_function<WRAP_DTYPES<NumberTypesL> >(
            [indices_begin, indices_end, &descending, &dim_size](auto begin, auto end){
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                using iterator_t = decltype(begin);
                int64_t total = (end - begin) / dim_size;
#ifdef USE_PARALLEL
                if(descending){
                    tbb::parallel_for(
                        utils::calculateGrainSize1D(total),
                        [&](const tbb::blocked_range<int64_t> &range){
                            // auto s_begin = begin + (range.begin() * dim_size);
                            // auto s_end = begin + (range.end() * dim_size);
                            auto i_begin = indices_begin + (range.begin() * dim_size);
                            auto i_end = indices_begin + (range.end() * dim_size);
                            for(;i_begin < i_end; i_begin += dim_size){
                                std::sort(i_begin, i_begin + dim_size, [&begin](const int64_t& a, const int64_t& b){
                                    return _nt_sort_descending_<iterator_t>(a, b, begin);    
                                });
                            }
                    });
                }else{
                   tbb::parallel_for(
                        utils::calculateGrainSize1D(total),
                        [&](const tbb::blocked_range<int64_t> &range){
                            // auto s_begin = begin + (range.begin() * dim_size);
                            // auto s_end = begin + (range.end() * dim_size);
                            auto i_begin = indices_begin + (range.begin() * dim_size);
                            auto i_end = indices_begin + (range.end() * dim_size);
                            for(;i_begin < i_end; i_begin += dim_size){
                                std::sort(i_begin, i_begin + dim_size, [&begin](const int64_t& a, const int64_t& b){
                                    return _nt_sort_ascending_<iterator_t>(a, b, begin);    
                                });
                            }
                    });
     
                }
#else
                if(descending){
                for(;indices_begin != indices_end; indices_begin += dim_size){
                    std::sort(indices_begin, indices_begin + dim_size, [&begin](const int64_t& a, const int64_t& b){
                        return _nt_sort_descending_<iterator_t>(a, b, begin);    
                    });
                }
                }else{
                for(;indices_begin != indices_end; indices_begin += dim_size){

                    std::sort(indices_begin, indices_begin + dim_size, [&begin](const int64_t& a, const int64_t& b){
                        bool ascended = _nt_sort_ascending_<iterator_t>(a, b, begin);
                        std::cout << std::boolalpha << "ascended is "<<ascended<< std::noboolalpha << std::endl;
                        return ascended;    
                    });


                }

                }
#endif
            }
        );
    }else{
        Tensor* begin = reinterpret_cast<Tensor*>(values.data_ptr());
        Tensor* end = reinterpret_cast<Tensor*>(values.data_ptr_end());
        int64_t total = (end - begin) / dim_size;
#ifdef USE_PARALLEL
        if(descending){
            tbb::parallel_for(
                utils::calculateGrainSize1D(total),
                [&](const tbb::blocked_range<int64_t> &range){
                    // auto s_begin = begin + (range.begin() * dim_size);
                    // auto s_end = begin + (range.end() * dim_size);
                    auto i_begin = indices_begin + (range.begin() * dim_size);
                    auto i_end = indices_begin + (range.end() * dim_size);
                    for(;i_begin < i_end; i_begin += dim_size){
                        std::sort(i_begin, i_begin + dim_size, [&begin](const int64_t& a, const int64_t& b){
                            return _nt_sort_descending_tensor_(a, b, begin);    
                        });
                    }
            });
        }else{
           tbb::parallel_for(
                utils::calculateGrainSize1D(total),
                [&](const tbb::blocked_range<int64_t> &range){
                    // auto s_begin = begin + (range.begin() * dim_size);
                    // auto s_end = begin + (range.end() * dim_size);
                    auto i_begin = indices_begin + (range.begin() * dim_size);
                    auto i_end = indices_begin + (range.end() * dim_size);
                    for(;i_begin < i_end; i_begin += dim_size){
                        std::sort(i_begin, i_begin + dim_size, [&begin](const int64_t& a, const int64_t& b){
                            return _nt_sort_ascending_tensor_(a, b, begin);    
                        });
                    }
            });

        }
#else
        if(descending){
        for(;indices_begin != indices_end; indices_begin += dim_size){
            std::sort(indices_begin, indices_begin + dim_size, [&begin](const int64_t& a, const int64_t& b){
                return _nt_sort_descending_tensor_(a, b, begin);    
            });
        }
        }else{
        for(;indices_begin != indices_end; indices_begin += dim_size){
            std::sort(indices_begin, indices_begin + dim_size, [&begin](const int64_t& a, const int64_t& b){
                return _nt_sort_ascending_tensor_(a, b, begin);
            });
        }

        }
#endif
 
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


template <typename T>
struct NumericVectorHash {

    std::size_t operator()(const nt::Tensor& vec){
        return vec.arr_void().cexecute_function<nt::WRAP_DTYPES<nt::DTypeEnum<nt::DTypeFuncs::type_to_dtype<T> > > >([](auto begin, auto end){
            std::size_t hash = 0;
            for(;begin != end; ++begin){
                if constexpr (std::is_same_v<nt::my_complex<nt::float16_t>, T>){
                    hash ^= std::hash<float>{}(_NT_FLOAT16_TO_FLOAT32_(begin->real())) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                    hash ^= std::hash<float>{}(_NT_FLOAT16_TO_FLOAT32_(begin->imag())) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr (std::is_same_v<nt::my_complex<float>, T>){
                    hash ^= std::hash<float>{}(begin->real()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                    hash ^= std::hash<float>{}(begin->imag()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr (std::is_same_v<nt::my_complex<double>, T>){
                    hash ^= std::hash<double>{}(begin->real()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                    hash ^= std::hash<double>{}(begin->imag()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr (std::is_same_v<nt::float16_t, T>){
                    hash ^= std::hash<float>{}(_NT_FLOAT16_TO_FLOAT32_(*begin)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr(std::is_same_v<nt::uint_bool_t, T>){
                    hash ^= std::hash<float>{}(*begin ? float(1) : float(0)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
#ifdef __SIZEOF_INT128__
                else if constexpr(std::is_same_v<nt::uint128_t, T>){
                    hash ^= std::hash<int64_t>{}(nt::convert::convert<int64_t, nt::uint128_t>(*begin)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr(std::is_same_v<nt::int128_t, T>){
                    hash ^= std::hash<int64_t>{}(nt::convert::convert<int64_t, nt::int128_t>(*begin)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
#endif
                else{
                    hash ^= std::hash<T>{}(*begin) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
            }
            return hash;
        });
    }
};

template<typename T>
struct NumericVectorEqual {
    bool operator()(const nt::Tensor& a, const nt::Tensor& b) const {
        if(a.numel() != b.numel() || a.dtype != b.dtype){return false;}
        if(a.is_null() || b.is_null()){return false;}
        const nt::ArrayVoid& arr_v = b.arr_void();
        return a.arr_void().cexecute_function<nt::DTypeFuncs::type_to_dtype<T> >([&arr_v](auto begin, auto end) -> bool{
            using value_t = nt::utils::IteratorBaseType_t<decltype(begin)>;
            return arr_v.cexecute_function<nt::DTypeFuncs::type_to_dtype<value_t>>([&begin, &end](auto second, auto s_end) -> bool{
                return std::equal(begin, end, second);
            });
        });
    }
};


template<typename T>
struct tensor_hashed{
    const Tensor* a;
    std::size_t hash;
    tensor_hashed(const Tensor* a_) : a(a_) {
        hash = a_->arr_void().cexecute_function<nt::WRAP_DTYPES<nt::DTypeEnum<nt::DTypeFuncs::type_to_dtype<T> > > >([](auto begin, auto end){
            std::size_t hash = 0;
            for(;begin != end; ++begin){
                if constexpr (std::is_same_v<nt::my_complex<nt::float16_t>, T>){
                    hash ^= std::hash<float>{}(_NT_FLOAT16_TO_FLOAT32_(begin->real())) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                    hash ^= std::hash<float>{}(_NT_FLOAT16_TO_FLOAT32_(begin->imag())) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr (std::is_same_v<nt::my_complex<float>, T>){
                    hash ^= std::hash<float>{}(begin->real()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                    hash ^= std::hash<float>{}(begin->imag()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr (std::is_same_v<nt::my_complex<double>, T>){
                    hash ^= std::hash<double>{}(begin->real()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                    hash ^= std::hash<double>{}(begin->imag()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr (std::is_same_v<nt::float16_t, T>){
                    hash ^= std::hash<float>{}(_NT_FLOAT16_TO_FLOAT32_(*begin)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr(std::is_same_v<nt::uint_bool_t, T>){
                    hash ^= std::hash<float>{}(*begin ? float(1) : float(0)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
#ifdef __SIZEOF_INT128__
                else if constexpr(std::is_same_v<nt::uint128_t, T>){
                    hash ^= std::hash<int64_t>{}(nt::convert::convert<int64_t, nt::uint128_t>(*begin)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr(std::is_same_v<nt::int128_t, T>){
                    hash ^= std::hash<int64_t>{}(nt::convert::convert<int64_t, nt::int128_t>(*begin)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
#endif
                else{
                    hash ^= std::hash<T>{}(*begin) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
            }
            return hash;
        });

    }
};

template <typename T>
struct HashedTensorHash {
    std::size_t operator()(const tensor_hashed<T>& vec) const { return vec.hash;}
};

template<typename T>
struct  HashedTensorEqual {
    bool operator()(const tensor_hashed<T>& h_a, const tensor_hashed<T>& h_b) const {
        const Tensor& a = *h_a.a;
        const Tensor& b = *h_b.a;
        if(a.numel() != b.numel() || a.dtype != b.dtype){return false;}
        if(a.is_null() || b.is_null()){return false;}
        const nt::ArrayVoid& arr_v = b.arr_void();
        return a.arr_void().cexecute_function<nt::DTypeFuncs::type_to_dtype<T> >([&arr_v](auto begin, auto end) -> bool{
            using value_t = nt::utils::IteratorBaseType_t<decltype(begin)>;
            return arr_v.cexecute_function<nt::DTypeFuncs::type_to_dtype<value_t>>([&begin, &end](auto second, auto s_end) -> bool{
                return std::equal(begin, end, second);
            });
        });
    }
};

nt::Tensor unique(nt::Tensor input, int64_t dim, bool return_sorted, bool return_indices){
    utils::throw_exception(return_sorted || return_indices, "unique function must return indices or the sorted tensor, or both, got none");
    input = input.transpose(-1, dim).contiguous();
    int64_t last_dim = input.shape().back();
    nt::Tensor splits = input.split_axis(-2);
    nt::Tensor* s_begin = reinterpret_cast<nt::Tensor*>(splits.data_ptr());
    nt::Tensor* s_end = reinterpret_cast<nt::Tensor*>(splits.data_ptr_end());
    return input.arr_void().execute_function<nt::WRAP_DTYPES<nt::NumberTypesL> >(
    [&s_begin, &s_end, &last_dim, &return_sorted, &return_indices](auto begin, auto end) -> nt::Tensor {
        using value_t = nt::utils::IteratorBaseType_t<decltype(begin)>;
        std::unordered_map<tensor_hashed<value_t>, int64_t, 
            HashedTensorHash<value_t>, HashedTensorEqual<value_t>> unique_map; //tensor and its indice
        int64_t counter = 0;
        unique_map[tensor_hashed<value_t>(s_begin)] = counter;
        ++counter;
        ++s_begin;
        for(;s_begin != s_end; ++s_begin, ++counter){
            tensor_hashed<value_t> check(s_begin);
            if (unique_map.find(check) == unique_map.end()) {
                unique_map[check] = counter;
            } 
        }
        if(!return_indices){
            nt::Tensor output = nt::Tensor::makeNullTensorArray(static_cast<long long>(unique_map.size()));
            nt::Tensor* o_begin = reinterpret_cast<nt::Tensor*>(output.data_ptr());
            nt::Tensor* o_end = o_begin + output.numel();
            for(const auto& [tensor, indice] : unique_map){
                *o_begin = *tensor.a;
                ++o_begin;
            }
            nt::Tensor out = nt::functional::cat_unordered(output);
            return out.view(-1, last_dim);
        }
        if(!return_sorted){
            nt::Tensor output_indices({static_cast<long long>(unique_map.size())}, nt::DType::int64);
            int64_t* i_begin = reinterpret_cast<int64_t*>(output_indices.data_ptr());
            for(const auto& [tensor, indice] : unique_map){
                *i_begin = indice;
                ++i_begin;
            }
            return std::move(output_indices);
 
        }
        nt::Tensor output = nt::Tensor::makeNullTensorArray(static_cast<long long>(unique_map.size()));
        nt::Tensor output_indices({static_cast<long long>(unique_map.size())}, nt::DType::int64);
        nt::Tensor* o_begin = reinterpret_cast<nt::Tensor*>(output.data_ptr());
        nt::Tensor* o_end = o_begin + output.numel();
        int64_t* i_begin = reinterpret_cast<int64_t*>(output_indices.data_ptr());
        for(const auto& [tensor, indice] : unique_map){
            *i_begin = indice;
            *o_begin = *tensor.a;
            ++i_begin;
            ++o_begin;
        }
        nt::Tensor out = nt::functional::cat_unordered(output);
        return nt::functional::list(out.view(-1, last_dim), output_indices);
    });
}


int64_t num_combinations(int64_t n, int64_t r) {
    if (r > n) return 0;
    int64_t result = 1;
    for (int64_t i = 0; i < r; ++i) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

nt::Tensor combinations(nt::Tensor vec, int64_t r, int64_t start){
    //takes a vector
    //returns combinations
    //similar to pythons itertools.combinations
    nt::utils::throw_exception(vec.dims() == 1, "Expected to get a vector of dimensions 1 but got dimensionality of $", vec.dims());
    const int64_t n = vec.shape()[0];
    nt::Tensor myints = nt::functional::arange(r, nt::DType::int64, start);
    nt::Tensor out = nt::Tensor::makeNullTensorArray(num_combinations(n, r));
    nt::Tensor* begin = reinterpret_cast<nt::Tensor*>(out.data_ptr());
    *begin = vec[myints];
    ++begin;
    int64_t* first = reinterpret_cast<int64_t*>(myints.data_ptr());
    int64_t* last = reinterpret_cast<int64_t*>(myints.data_ptr_end());
    while((*first) != n-r+start){
        int64_t* mt = last;
        --mt; // Ensure mt is decremented before use
        while (*mt == n - int64_t(last - mt) + start) {
            --mt;
        }
        (*mt)++;
        while (++mt != last) *mt = *(mt-1)+1;
        *begin = vec[myints];
        ++begin;
    }
    return nt::functional::stack(out).clone();

}


Tensor chunk(Tensor input, typename Tensor::size_value_t chunks, int64_t dim){
	dim = (dim < 0) ? dim + input.dims() : dim;
	utils::THROW_EXCEPTION(dim < input.dims(), "Expected (dim = $) to be less than dims of input which is $", dim, input.dims());
	utils::THROW_EXCEPTION(dim >= 0, "Expected (dim = $) to be greater than or equal to zero", dim);
	Tensor output({chunks}, DType::TensorObj);
	typename SizeRef::value_type adding = input.shape()[dim] / chunks;
	if(dim == 0){
		typename SizeRef::value_type begin = 0;
		typename SizeRef::value_type end = adding;
		for(typename SizeRef::value_type i = 0; i < chunks-1; ++i){
			output[i] = input[my_range(begin, end)];
			begin += adding;
			end += adding;
		}
		output[chunks-1] = input[my_range(begin, -1)];
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
	for(typename SizeRef::value_type i = 0; i < chunks-1; ++i){
		std::vector<Tensor> vec_cpy(vec.size());
		for(typename SizeRef::value_type j = 0; j < vec.size(); ++j){
			vec_cpy[j] = vec[i][my_range(begin, end)];
		}
		output[i] = cat_unordered(vec_cpy).view(curr_shape);
		begin += adding;
		end += adding;
	}
	n_shape[dim_cpy] = input.shape()[dim_cpy] - (adding * (chunks-1));


	std::vector<Tensor> vec_cpy(vec.size());
	for(typename SizeRef::value_type j = 0; j < vec.size(); ++j){
		vec_cpy[j] = vec[j][my_range(begin, -1)];
	}
	output[chunks-1] = cat_unordered(vec_cpy).view(SizeRef(std::move(n_shape)));
	return std::move(output);
}


Tensor sigmoid(const Tensor& x){
	if(x.dtype == DType::TensorObj){
		Tensor a = (-1) * x;
		a.exp_();
		a += 1;
		a.inverse_();
		return std::move(a);
	}
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL> >([](auto begin, auto end){
		mp::sigmoid(begin, end, begin);
	});
	return std::move(a);
}

Tensor dsigmoid(const Tensor & x, bool apply_sigmoid){
	if(x.dtype == DType::TensorObj){
		if(!apply_sigmoid)
			return x * (1-x);
		Tensor sigmoid_x = sigmoid(x);
		return sigmoid_x * (1 - sigmoid_x);	
	}
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL> >([&apply_sigmoid](auto begin, auto end){
		mp::dsigmoid(begin, end, begin, apply_sigmoid);
	});
	return std::move(a);
	
}

//TODO: silu function
/* double gelu_approx_grad(double x) { */
/*     const double sqrt_2_pi = std::sqrt(2.0 / M_PI); */
/*     const double c = 0.044715; */

/*     // Compute z = sqrt(2/pi) * (x + c * x^3) */
/*     double z = sqrt_2_pi * (x + c * std::pow(x, 3)); */

/*     // Compute tanh(z) and its derivative */
/*     double tanh_z = std::tanh(z); */
/*     double tanh_derivative = 1 - tanh_z * tanh_z; */

/*     // Gradient of z with respect to x */
/*     double dz_dx = sqrt_2_pi * (1 + 3 * c * x * x); */

/*     // Final gradient */
/*     return 0.5 * (1 + tanh_z) + 0.5 * x * tanh_derivative * dz_dx; */
/* } */

Tensor tanh(const Tensor& x){
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
		mp::tanh(begin, end, begin);
	});
	return std::move(a);
}

Tensor tan(const Tensor& x){
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
		mp::tan(begin, end, begin);
	});
	return std::move(a);
}

Tensor dtanh(const Tensor& x){
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
		mp::dtanh(begin, end, begin);
	});
	return std::move(a);
}

Tensor dtan(const Tensor& x){
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
		mp::dtan(begin, end, begin);
	});
	return std::move(a);
}

Tensor sinh(const Tensor& x){
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
		mp::sinh(begin, end, begin);
	});
	return std::move(a);
}

Tensor sin(const Tensor& x){
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
		mp::sin(begin, end, begin);
	});
	return std::move(a);
}

Tensor cosh(const Tensor& x){
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
		mp::cosh(begin, end, begin);
	});
	return std::move(a);
}

Tensor cos(const Tensor& x){
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
		mp::cos(begin, end, begin);
	});
	return std::move(a);
}

Tensor sqrt(const Tensor& x){
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
		mp::sqrt(begin, end, begin);
	});
	return std::move(a);
	
}

Tensor dsqrt(const Tensor& x){
    return (0.5 * invsqrt(x));
}

Tensor invsqrt(const Tensor& x){
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
		mp::invsqrt(begin, end, begin);
	});
	return std::move(a);
}

Tensor dinvsqrt(const Tensor& x){
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
		mp::dinvsqrt(begin, end, begin);
	});
	return std::move(a);
}



Tensor abs(const Tensor& x){
	Tensor a = x.clone();
	a.arr_void().execute_function_chunk<WRAP_DTYPES< FloatingTypesL, SignedTypesL> >([](auto begin, auto end){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
#ifdef __SIZEOF_INT128__
        if constexpr (std::is_same_v<value_t, int128_t>){
            for(;begin != end; ++begin)
                *begin = static_cast<int128_t>(std::abs(static_cast<int64_t>(*begin)));
                
        }
        else{
            for(;begin != end; ++begin)
                *begin = std::abs(*begin);
        }
#else
        for(;begin != end; ++begin)
            *begin = std::abs(*begin);
#endif
	});
    a.arr_void().execute_function_chunk<WRAP_DTYPES<ComplexTypesL> >([](auto begin, auto end){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
		for(;begin != end; ++begin)
			*begin = value_t(std::abs(std::get<0>(*begin)), std::abs(std::get<1>(*begin)));
	});
    a.arr_void().execute_function_chunk<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
		for(;begin != end; ++begin)
			*begin = abs(*begin);
	});
    return std::move(a);
}





Tensor softplus(const Tensor& x, Scalar beta, Scalar threshold){
	Tensor softplus_x = x * beta;
	Tensor where = x > threshold;
	softplus_x[where].set_(log(1 + std::exp(softplus_x[where])).divide_(beta));
	return std::move(softplus_x);
}

Tensor var(const Tensor& x, utils::optional_list dim, int64_t correction, bool keepdim){
	Tensor mean = x.mean(dim, true);
	Tensor squared_diff = std::pow((x - mean), 2);
	int64_t N = 0;
	if(!dim){
		N = x.numel();
	}else{
		N = 1;
		for(const auto& ele : dim){
			N *= x.shape()[ele];
		}
	}
	Tensor variance = squared_diff.sum(dim, keepdim) / (N - correction);
	return std::move(variance);
}
Tensor dvar(const Tensor& dx, const Tensor& x, utils::optional_list dim, int64_t correction){
	//takes both the gradient, and the input given to the variance function
	Tensor mean = x.mean(dim, true);
	int64_t N = 0;
	if(!dim){
		N = x.numel();
	}else{
		N = 1;
		for(const auto& ele : dim){
			N *= x.shape()[ele];
		}
	}
	return (2 / (N - correction)) * (x - mean);
}

Tensor dot(const Tensor& a, const Tensor& b, utils::optional_list dim, bool keepdim){
	Tensor c = a * b;
	return c.sum(dim, keepdim);
}



void next_index(const SizeRef& s, std::vector<int64_t>& v, typename SizeRef::value_type index){
	if(v[index] == s[index] - 1){
		v[index] = 0;
		if(index == 0){
			std::fill(v.begin(), v.end(), 0);
			next_index(s, v, v.size()-1);
			return;
		}
		--index;
		next_index(s, v, index);
	}
	else{
		++v[index];
	}
}




Tensor meshgrid(Tensor&& x, Tensor&& y){
	utils::THROW_EXCEPTION(x.dtype == y.dtype, "Runtime Error: Expected tensors to have same dtype but got $ and $", x.dtype, y.dtype);
	/* utils::THROW_EXCEPTION(a.numel() == b.numel(), "RuntimeError: Expected tensors to have same number of elements but got $ and $", a.numel(), b.numel()) */
	Tensor xy({2}, DType::TensorObj);
	Tensor* xy_p = reinterpret_cast<Tensor*>(xy.data_ptr());
	*xy_p = Tensor({static_cast<typename SizeRef::value_type>(x.numel()), static_cast<typename SizeRef::value_type>(y.numel())}, x.dtype);
	*(xy_p + 1) = Tensor({static_cast<typename SizeRef::value_type>(x.numel()), static_cast<typename SizeRef::value_type>(y.numel())}, x.dtype);
	
	const typename SizeRef::value_type x_n = x.numel();
	const typename SizeRef::value_type y_n = y.numel();
	x.arr_void().execute_function([xy_p, &x_n, &y_n](auto a_begin, auto a_end, auto b_begin){
				Tensor& X = *xy_p;
				Tensor& Y = *(xy_p+1);
				const typename SizeRef::value_type total_size = x_n * y_n;
				using value_t = utils::IteratorBaseType_t<decltype(a_begin)>;
				value_t* x_begin = reinterpret_cast<value_t*>(X.data_ptr());
				value_t* y_begin = reinterpret_cast<value_t*>(Y.data_ptr());
				auto b_end = b_begin + y_n;
				auto b_cpy = b_begin;
				
				for(;a_begin != a_end; ++a_begin){
					for(;b_begin != b_end; ++b_begin, ++x_begin, ++y_begin){
						*x_begin = *a_begin;
						*y_begin = *b_begin;
					}
					b_begin = b_cpy;
				}

			}, y.arr_void());
	return std::move(xy);

}




Tensor droupout(const Tensor& input, double p){
	Tensor bools = randbools(input.shape(), p);
	Tensor out = input.clone();
	out[bools] = 0;
	return std::move(out);
}

}
}
