// #include "../linalg.h"
#include "../utils/utils.h"
#include "../Tensor.h"
#include "../functional/functional.h"
#include <limits>
#include <tuple>
#include "../dtype/ArrayVoid.hpp"

#ifdef USE_PARALLEL
	#include <tbb/parallel_for_each.h>
	#include <tbb/parallel_for.h>
	#include <tbb/blocked_range.h>
	#include <tbb/blocked_range2d.h>
	#include <tbb/blocked_range3d.h>
	#include <tbb/parallel_reduce.h>
	#include <thread>
    #include <tbb/concurrent_vector.h>
	/* #include "../mp/MP.hpp" */
	/* #include "../mp/Pool.hpp" */
#endif

namespace nt{
namespace linalg{


Tensor extract_free_rows(const Tensor& t){
    utils::throw_exception(t.dims() == 2, "Expected to get a matrix to extract the free columns and have an rref matrix");
    const int64_t& rows = t.shape()[0];
    const int64_t& cols = t.shape()[1];
    auto [one_rows, one_cols] = get<2>(functional::where(t == 1));
    std::vector<bool> is_pivot_row(rows, false);
    int64_t* r_begin = reinterpret_cast<int64_t*>(one_rows.data_ptr());
    int64_t* r_end = reinterpret_cast<int64_t*>(one_rows.data_ptr_end());
    int64_t* c_begin = reinterpret_cast<int64_t*>(one_cols.data_ptr());
    int64_t last = -1;
    for(;r_begin != r_end; ++r_begin, ++c_begin){
        if(*c_begin != last){
            is_pivot_row[*r_begin] = true;
            last = *c_begin;
        }
    }
    
    Tensor split = t.split_axis(-2);
    Tensor* access = reinterpret_cast<Tensor*>(split.data_ptr());
    std::vector<Tensor> out;
    out.reserve(std::count(is_pivot_row.begin(), is_pivot_row.end(), false));
    for(int64_t i = 0; i < rows; ++i){
        if(!is_pivot_row[i]){
            out.push_back(access[i]);
        }
    }
    if(out.size() == 0){return Tensor::Null();}
    return functional::stack(std::move(out));
}

Tensor extract_pivot_rows(const Tensor& t){
    utils::throw_exception(t.dims() == 2, "Expected to get a matrix to extract the free columns and have an rref matrix");
    const int64_t& rows = t.shape()[0];
    const int64_t& cols = t.shape()[1];
    auto [one_rows, one_cols] = get<2>(functional::where(t == 1));
    std::vector<bool> is_pivot_row(rows, false);
    int64_t* r_begin = reinterpret_cast<int64_t*>(one_rows.data_ptr());
    int64_t* r_end = reinterpret_cast<int64_t*>(one_rows.data_ptr_end());
    int64_t* c_begin = reinterpret_cast<int64_t*>(one_cols.data_ptr());
    int64_t last = -1;
    for(;r_begin != r_end; ++r_begin, ++c_begin){
        if(*c_begin != last){
            is_pivot_row[*r_begin] = true;
            last = *c_begin;
        }
    }
    
    Tensor split = t.split_axis(-2);
    Tensor* access = reinterpret_cast<Tensor*>(split.data_ptr());
    std::vector<Tensor> out;
    out.reserve(std::count(is_pivot_row.begin(), is_pivot_row.end(), true));
    for(int64_t i = 0; i < rows; ++i){
        if(is_pivot_row[i]){
            out.push_back(access[i]);
        }
    }
    if(out.size() == 0){return Tensor::Null();}
    return functional::stack(std::move(out));
}


Tensor pivot_rows_matrix(const Tensor& t){
    const int64_t& rows = t.shape()[0];
    const int64_t& cols = t.shape()[1];
    auto [one_rows, one_cols] = get<2>(functional::where(t == 1));
    std::vector<bool> is_pivot_row(rows, false);
    int64_t* r_begin = reinterpret_cast<int64_t*>(one_rows.data_ptr());
    int64_t* r_end = reinterpret_cast<int64_t*>(one_rows.data_ptr_end());
    int64_t* c_begin = reinterpret_cast<int64_t*>(one_cols.data_ptr());
    int64_t last = -1;
    for(;r_begin != r_end; ++r_begin, ++c_begin){
        if(*c_begin != last){
            is_pivot_row[*r_begin] = true;
            last = *c_begin;
        }
    }

    // std::cout << t << std::endl;
    // print_vec(is_pivot_col);
    std::vector<int64_t> pivots;
    pivots.reserve(rows);
    for(int64_t k = 0; k < rows; ++k){
        if(is_pivot_row[k]){pivots.push_back(k);}
    }
    return functional::vector_to_tensor(pivots);
}

Tensor pivot_rows(const Tensor& t){
    utils::throw_exception(t.dims() >= 2, "Cannot get pivot rows from tensor with dimensions less than 2 got $", t.dims());
    utils::throw_exception(t.dtype() != DType::Bool && t.dtype() != DType::TensorObj, "Can only find the pivot rows of numerical types, got $", t.dtype());
    if(t.dims() == 2){
        return pivot_rows_matrix(t); 
    }
    std::vector<int64_t> out_shape = t.shape().Vec();
    out_shape.pop_back();
    out_shape.pop_back();
    int64_t batches = t.numel() / (t.shape()[-1] * t.shape()[-2]);
    Tensor out = Tensor::makeNullTensorArray(batches).view(SizeRef(std::move(out_shape)));
    Tensor* o_begin = reinterpret_cast<Tensor*>(out.data_ptr());
    Tensor* o_end = reinterpret_cast<Tensor*>(out.data_ptr_end());
    Tensor split = t.split_axis(-3);
    Tensor* s_begin = reinterpret_cast<Tensor*>(split.data_ptr());
    utils::THROW_EXCEPTION(split.numel() == batches, "INTERNAL LOGIC ERROR $ $", split.numel(), batches);
    #ifdef USE_PARALLEL
    tbb::parallel_for(utils::calculateGrainSize1D(batches),
    [&](const tbb::blocked_range<int64_t> &range){
        for(int64_t i = range.begin(); i != range.end(); ++i){
            o_begin[i] = pivot_rows_matrix(s_begin[i]);
        }
    });
    #else
    for(;o_begin != o_end; ++s_begin, ++o_begin) *o_begin = pivot_rows_matrix(*s_begin);
    #endif
    return std::move(out);
}


Tensor extract_free_cols(const Tensor& t){
    utils::throw_exception(t.dims() == 2, "Expected to get a matrix to extract the free columns and have an rref matrix");
    const int64_t& rows = t.shape()[0];
    const int64_t& cols = t.shape()[1];
    auto [one_rows, one_cols] = get<2>(functional::where(t == 1));
    std::vector<bool> is_pivot_col(cols, false);
    int64_t* r_begin = reinterpret_cast<int64_t*>(one_rows.data_ptr());
    int64_t* r_end = reinterpret_cast<int64_t*>(one_rows.data_ptr_end());
    int64_t* c_begin = reinterpret_cast<int64_t*>(one_cols.data_ptr());
    int64_t last = -1;
    // std::cout << "one rows: "<<one_rows<<std::endl;
    // std::cout << "one cols: "<<one_cols<<std::endl;
    for(;r_begin != r_end; ++r_begin, ++c_begin){
        if(*r_begin != last){
            is_pivot_col[*c_begin] = true;
            last = *r_begin;
        }
    }
    
    Tensor split = t.split_axis(-1);
    Tensor* access = reinterpret_cast<Tensor*>(split.data_ptr());
    std::vector<Tensor> out;
    out.reserve(std::count(is_pivot_col.begin(), is_pivot_col.end(), false));
    for(int64_t i = 0; i < cols; ++i){
        if(!is_pivot_col[i]){
            out.push_back(access[i]);
        }
    }
    if(out.size() == 0){return Tensor::Null();}
    return functional::stack(std::move(out)).transpose(-1, -2);
}

Tensor extract_pivot_cols(const Tensor& t){
    utils::throw_exception(t.dims() == 2, "Expected to get a matrix to extract the free columns and have an rref matrix");
    const int64_t& rows = t.shape()[0];
    const int64_t& cols = t.shape()[1];
    auto [one_rows, one_cols] = get<2>(functional::where(t == 1));
    std::vector<bool> is_pivot_col(cols, false);
    int64_t* r_begin = reinterpret_cast<int64_t*>(one_rows.data_ptr());
    int64_t* r_end = reinterpret_cast<int64_t*>(one_rows.data_ptr_end());
    int64_t* c_begin = reinterpret_cast<int64_t*>(one_cols.data_ptr());
    int64_t last = -1;
    // std::cout << "one rows: "<<one_rows<<std::endl;
    // std::cout << "one cols: "<<one_cols<<std::endl;
    for(;r_begin != r_end; ++r_begin, ++c_begin){
        if(*r_begin != last){
            is_pivot_col[*c_begin] = true;
            last = *r_begin;
        }
    }
    
    Tensor split = t.split_axis(-1);
    Tensor* access = reinterpret_cast<Tensor*>(split.data_ptr());
    std::vector<Tensor> out;
    out.reserve(std::count(is_pivot_col.begin(), is_pivot_col.end(), true));
    for(int64_t i = 0; i < cols; ++i){
        if(is_pivot_col[i]){
            out.push_back(access[i]);
        }
    }
    if(out.size() == 0){return Tensor::Null();}
    return functional::stack(std::move(out)).transpose(-1, -2);
}

Tensor pivot_cols_matrix(const Tensor& t, bool return_where){
    const int64_t& rows = t.shape()[0];
    const int64_t& cols = t.shape()[1];
    auto [one_rows, one_cols] = get<2>(functional::where(t == 1));
    std::vector<bool> is_pivot_col(cols, false);
    int64_t* r_begin = reinterpret_cast<int64_t*>(one_rows.data_ptr());
    int64_t* r_end = reinterpret_cast<int64_t*>(one_rows.data_ptr_end());
    int64_t* c_begin = reinterpret_cast<int64_t*>(one_cols.data_ptr());
    int64_t last = -1;
    // std::cout << "one rows: "<<one_rows<<std::endl;
    // std::cout << "one cols: "<<one_cols<<std::endl;
    for(;r_begin != r_end; ++r_begin, ++c_begin){
        if(*r_begin != last){
            is_pivot_col[*c_begin] = true;
            last = *r_begin;
        }
    }

    // std::cout << t << std::endl;
    // print_vec(is_pivot_col);
    std::vector<int64_t> pivots;
    pivots.reserve(cols);
    for(int64_t k = 0; k < cols; ++k){
        if(is_pivot_col[k]){pivots.push_back(k);}
    }
    // std::cout << "has "<<pivots.size()<<" pivot cols"<<std::endl;
    if(pivots.size() == 0){return Tensor({1}, DType::int64);}
    if(!return_where)
        return functional::vector_to_tensor(pivots);
    if(pivots.size() == 1){
        Tensor o_rows = functional::arange(SizeRef({static_cast<int64_t>(rows)}), DType::int64);
        Tensor o_cols({rows}, DType::int64);
        o_cols = pivots[0];
        return functional::list(o_rows, o_cols);
    }
    int64_t num_pivots = pivots.size();
    Tensor o_rows({static_cast<int64_t>(rows) * num_pivots}, DType::int64);
    int64_t* or_begin = reinterpret_cast<int64_t*>(o_rows.data_ptr());
    Tensor o_cols({static_cast<int64_t>(rows) * num_pivots}, DType::int64);
    int64_t* oc_begin = reinterpret_cast<int64_t*>(o_cols.data_ptr());
    for(int64_t r = 0; r < rows; ++r){
        for(int64_t i = 0; i < num_pivots; ++i, ++or_begin, ++oc_begin){
            *or_begin = r;
            *oc_begin = pivots[i];
        }
    }
    return functional::list(o_rows, o_cols);
}



Tensor pivot_cols(const Tensor& t, bool return_where){
    utils::throw_exception(t.dims() >= 2, "Cannot get pivot cols from tensor with dimensions less than 2 got $", t.dims());
    utils::throw_exception(t.dtype() != DType::Bool && t.dtype() != DType::TensorObj, "Can only find the pivot cols of numerical types, got $", t.dtype());
    if(t.dims() == 2){
        return pivot_cols_matrix(t, return_where); 
    }
    std::vector<int64_t> out_shape = t.shape().Vec();
    out_shape.pop_back();
    out_shape.pop_back();
    int64_t batches = t.numel() / (t.shape()[-1] * t.shape()[-2]);
    Tensor out = Tensor::makeNullTensorArray(batches).view(SizeRef(std::move(out_shape)));
    Tensor* o_begin = reinterpret_cast<Tensor*>(out.data_ptr());
    Tensor* o_end = reinterpret_cast<Tensor*>(out.data_ptr_end());
    Tensor split = t.split_axis(-3);
    Tensor* s_begin = reinterpret_cast<Tensor*>(split.data_ptr());
    utils::THROW_EXCEPTION(split.numel() == batches, "INTERNAL LOGIC ERROR $ $", split.numel(), batches);
    #ifdef USE_PARALLEL
    tbb::parallel_for(utils::calculateGrainSize1D(batches),
    [&](const tbb::blocked_range<int64_t> &range){
        for(int64_t i = range.begin(); i != range.end(); ++i){
            o_begin[i] = pivot_cols_matrix(s_begin[i], return_where);
        }
    });
    #else
    for(;o_begin != o_end; ++s_begin, ++o_begin) *o_begin = pivot_cols_matrix(*s_begin);
    #endif
    return std::move(out);
}


//this is able to just return a number by only taking a matrix
int64_t num_pivot_rows_matrix(const Tensor& t){
    utils::throw_exception(t.dims() == 2, "Expected to process a matrix in num_pivot_rows_matrix got $ dims", t.dims());
    utils::throw_exception(t.dtype() != DType::TensorObj, "Cannot process tensor of tensors for pivots");
    const int64_t& rows = t.shape()[0];
    const int64_t& cols = t.shape()[1];
    auto [one_rows, one_cols] = get<2>(functional::where(t == 1));
    std::vector<bool> is_pivot_row(rows, false);
    int64_t* r_begin = reinterpret_cast<int64_t*>(one_rows.data_ptr());
    int64_t* r_end = reinterpret_cast<int64_t*>(one_rows.data_ptr_end());
    int64_t* c_begin = reinterpret_cast<int64_t*>(one_cols.data_ptr());
    int64_t last = -1;
    for(;r_begin != r_end; ++r_begin, ++c_begin){
        if(*c_begin != last){
            is_pivot_row[*r_begin] = true;
            last = *c_begin;
        }
    }
    int64_t cntr = 0;
    for(const bool& a : is_pivot_row){if(a){++cntr;}}
    return cntr;

}
int64_t num_pivot_cols_matrix(const Tensor& t){
    utils::throw_exception(t.dims() == 2, "Expected to process a matrix in num_pivot_rows_matrix got $ dims", t.dims());
    utils::throw_exception(t.dtype() != DType::TensorObj, "Cannot process tensor of tensors for pivots");
    const int64_t& rows = t.shape()[0];
    const int64_t& cols = t.shape()[1];
    auto [one_rows, one_cols] = get<2>(functional::where(t == 1));
    std::vector<bool> is_pivot_col(cols, false);
    int64_t* r_begin = reinterpret_cast<int64_t*>(one_rows.data_ptr());
    int64_t* r_end = reinterpret_cast<int64_t*>(one_rows.data_ptr_end());
    int64_t* c_begin = reinterpret_cast<int64_t*>(one_cols.data_ptr());
    int64_t last = -1;
    for(;r_begin != r_end; ++r_begin, ++c_begin){
        if(*r_begin != last){
            is_pivot_col[*c_begin] = true;
            last = *r_begin;
        }
    }

    int64_t cntr = 0;
    for(const bool& a : is_pivot_col){if(a){++cntr;}}
    return cntr;
}


//returns the number of pivot cols or rows in every matrix
Tensor num_pivot_rows(const Tensor& t){
    if(t.dims() == 2){
        Tensor out({1}, DType::int64);
        out = num_pivot_rows_matrix(t);
        return std::move(out);
    }
    utils::throw_exception(t.dims() > 2, "Expected to process a matrix or tensor in num_pivot_rows got $ dims", t.dims());
    utils::throw_exception(t.dtype() != DType::TensorObj, "Cannot process tensor of tensors for pivots");
    std::vector<int64_t> out_shape = t.shape().Vec();
    out_shape.pop_back();
    out_shape.pop_back();
    Tensor a = t.flatten(0, -3);
    const int64_t batches = a.shape()[0];
    const int64_t rows = a.shape()[1];
    const int64_t cols = a.shape()[2];
    Tensor out = nt::functional::zeros(SizeRef(std::move(out_shape)), DType::int64);
    utils::THROW_EXCEPTION(out.numel() == batches, "INTERNAL LOGIC ERROR!");
    std::vector<std::vector<bool> > track_pivots(batches, std::vector<bool>(rows, false));
    auto [_one_batches, _one_rows, _one_cols] = nt::get<3>(nt::functional::where(a == 1));
    Tensor one_batches(_one_batches);
    Tensor one_rows(_one_rows);
    Tensor one_cols(_one_cols);

    if(one_batches.is_null()){
        return std::move(out);
    }
    #ifdef USE_PARALLEL
    tbb::parallel_for(utils::calculateGrainSize1D(batches),
    [&](const tbb::blocked_range<int64_t> &range){
        int64_t start = range.begin();
        int64_t end = range.end();
        const int64_t* b_begin = reinterpret_cast<const int64_t*>(one_batches.data_ptr());
        const int64_t* b_end = reinterpret_cast<const int64_t*>(one_batches.data_ptr_end());
        const int64_t* r_begin = reinterpret_cast<const int64_t*>(one_rows.data_ptr());
        const int64_t* c_begin = reinterpret_cast<const int64_t*>(one_cols.data_ptr());
        int64_t* o_begin = reinterpret_cast<int64_t*>(out.data_ptr());
        while(b_begin != b_end && *b_begin != start){++b_begin; ++r_begin; ++c_begin;}
        if(b_begin == b_end){return;}
        int64_t last_batch = -1;
        int64_t last = -1;
        while(b_begin != b_end && *b_begin != end){
            last_batch = *b_begin;
            std::vector<bool>& is_pivot_row = track_pivots[last_batch];
            for(;b_begin != b_end && *b_begin == last_batch; ++b_begin, ++c_begin, ++r_begin){
                if(*c_begin != last){
                    is_pivot_row[*r_begin] = true;
                    last = *c_begin;
                }
            }
            last = -1;
            o_begin[last_batch] = std::count(is_pivot_row.begin(), is_pivot_row.end(), true);
        }
    });
    #else
    const int64_t* b_begin = reinterpret_cast<const int64_t*>(one_batches.data_ptr());
    const int64_t* b_end = reinterpret_cast<const int64_t*>(one_batches.data_ptr_end());
    const int64_t* r_begin = reinterpret_cast<const int64_t*>(one_rows.data_ptr());
    const int64_t* c_begin = reinterpret_cast<const int64_t*>(one_cols.data_ptr());
    int64_t* o_begin = reinterpret_cast<int64_t*>(out.data_ptr());
    if(b_begin == b_end){return std::move(out);}
    int64_t last_batch = -1;
    int64_t last = -1;
    while(b_begin != b_end){
        last_batch = *b_begin;
        std::vector<bool>& is_pivot_row = track_pivots[last_batch];
        for(;b_begin != b_end && *b_begin == last_batch; ++b_begin, ++c_begin, ++r_begin){
            if(*c_begin != last){
                is_pivot_row[*r_begin] = true;
                last = *c_begin;
            }
        }
        last = -1;
        o_begin[last_batch] = std::count(is_pivot_row.begin(), is_pivot_row.end(), true);
    }

    #endif
    return std::move(out);
}

Tensor num_pivot_cols(const Tensor& t){
    if(t.dims() == 2){
        Tensor out({1}, DType::int64);
        out = num_pivot_rows_matrix(t);
        return std::move(out);
    }
    utils::throw_exception(t.dims() > 2, "Expected to process a matrix or tensor in num_pivot_rows got $ dims", t.dims());
    utils::throw_exception(t.dtype() != DType::TensorObj, "Cannot process tensor of tensors for pivots");
    std::vector<int64_t> out_shape = t.shape().Vec();
    out_shape.pop_back();
    out_shape.pop_back();
    Tensor a = t.flatten(0, -3);
    const int64_t batches = a.shape()[0];
    const int64_t rows = a.shape()[1];
    const int64_t cols = a.shape()[2];
    Tensor out = nt::functional::zeros(SizeRef(std::move(out_shape)), DType::int64);
    utils::THROW_EXCEPTION(out.numel() == batches, "INTERNAL LOGIC ERROR!");
    std::vector<std::vector<bool> > track_pivots(batches, std::vector<bool>(rows, false));
    auto [_one_batches, _one_rows, _one_cols] = nt::get<3>(nt::functional::where(a == 1));
    Tensor one_batches(_one_batches);
    Tensor one_rows(_one_rows);
    Tensor one_cols(_one_cols);
    if(one_batches.is_null()){
        return std::move(out);
    }
    #ifdef USE_PARALLEL
    tbb::parallel_for(utils::calculateGrainSize1D(batches),
    [&](const tbb::blocked_range<int64_t> &range){
        int64_t start = range.begin();
        int64_t end = range.end();
        const int64_t* b_begin = reinterpret_cast<const int64_t*>(one_batches.data_ptr());
        const int64_t* b_end = reinterpret_cast<const int64_t*>(one_batches.data_ptr_end());
        const int64_t* r_begin = reinterpret_cast<const int64_t*>(one_rows.data_ptr());
        const int64_t* c_begin = reinterpret_cast<const int64_t*>(one_cols.data_ptr());
        int64_t* o_begin = reinterpret_cast<int64_t*>(out.data_ptr());
        while(b_begin != b_end && *b_begin != start){++b_begin; ++r_begin; ++c_begin;}
        if(b_begin == b_end){return;}
        int64_t last_batch = -1;
        int64_t last = -1;
        while(b_begin != b_end && *b_begin != end){
            last_batch = *b_begin;
            std::vector<bool>& is_pivot_col = track_pivots[last_batch];
            for(;b_begin != b_end && *b_begin == last_batch; ++b_begin, ++c_begin, ++r_begin){
                if(*r_begin != last){
                    is_pivot_col[*r_begin] = true;
                    last = *r_begin;
                }
            }
            last = -1;
            o_begin[last_batch] = std::count(is_pivot_col.begin(), is_pivot_col.end(), true);
        }
    });
    #else
    const int64_t* b_begin = reinterpret_cast<const int64_t*>(one_batches.data_ptr());
    const int64_t* b_end = reinterpret_cast<const int64_t*>(one_batches.data_ptr_end());
    const int64_t* r_begin = reinterpret_cast<const int64_t*>(one_rows.data_ptr());
    const int64_t* c_begin = reinterpret_cast<const int64_t*>(one_cols.data_ptr());
    int64_t* o_begin = reinterpret_cast<int64_t*>(out.data_ptr());
    if(b_begin == b_end){return std::move(out);}
    int64_t last_batch = -1;
    int64_t last = -1;
    while(b_begin != b_end){
        last_batch = *b_begin;
        std::vector<bool>& is_pivot_col = track_pivots[last_batch];
        for(;b_begin != b_end && *b_begin == last_batch; ++b_begin, ++c_begin, ++r_begin){
            if(*r_begin != last){
                is_pivot_col[*c_begin] = true;
                last = *r_begin;
            }
        }
        last = -1;
        o_begin[last_batch] = std::count(is_pivot_row.begin(), is_pivot_row.end(), true);
    }

    #endif
    return std::move(out);
} 


}} //nt::linalg::
