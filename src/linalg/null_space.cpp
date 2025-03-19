// #include "../linalg.h"
#include "../utils/utils.h"
#include "../Tensor.h"
#include "../functional/functional.h"
#include <limits>
#include <tuple>
#include "../dtype/ArrayVoid.hpp"
#include "headers/toEigen.hpp"
#include "headers/null_space.h"

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

// template<typename T>
// inline void print_vec(const std::vector<T>& vals){
//     std::cout << '{';
//     for(size_t i = 0; i < vals.size()-1; ++i){
//         std::cout << vals[i] << ',';
//     }
//     std::cout << vals.back() << '}' << std::endl;
// }


Tensor get_reduced_rows_null_space(const Tensor& t, bool pivots_first){
    Tensor rT = t;
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
    std::vector<int64_t> pivots, free_rows;
    pivots.reserve(rows);
    free_rows.reserve(rows);
    for(int64_t k = 0; k < rows; ++k){
        if(is_pivot_row[k]){pivots.push_back(k);}
        else {free_rows.push_back(k);}
    }
    int64_t pivot_rows = pivots.size();
    int64_t free_vars = free_rows.size();
    if(free_vars == 0){
        return Tensor::Null(); //empty tensor, trivial null space
    }
    // std::cout << "pivot cols:";
    // print_vec(pivots);
    // std::cout << "free cols:";
    // print_vec(free_cols);
    // std::cout << "null space is {"<<cols<<','<<free_vars<<"} "<<rows<<',' << cols<<std::endl;

    if(pivots_first){
        Tensor null_space = functional::zeros({free_vars, rows}, t.dtype);
        #ifdef USE_PARALLEL
        null_space.arr_void().execute_function<WRAP_DTYPES<NumberTypesL> > (
            [&](auto begin, auto end, auto begin2){
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                tbb::parallel_for(nt::utils::calculateGrainSize1D(free_rows.size()),
                [&](const tbb::blocked_range<int64_t> &range){
                for(int64_t k = range.begin(); k != range.end(); ++k){
                    const int64_t& free_row = free_rows[k];
                    begin[k * rows + free_row] = value_t(1);  // Set free variable to 1              
                    for(size_t i = 0; i < pivots.size(); ++i){
                        const int64_t& pivot_row = pivots[i];
                        begin[k * rows + pivot_row] = -(begin2[free_row * cols + i]);
                    }
                }});
        }, rT.arr_void());
        #else

        null_space.arr_void().execute_function<WRAP_DTYPES<NumberTypesL> > (
            [&](auto begin, auto end, auto begin2){
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                for(size_t k = 0; k < free_rows.size(); ++k){
                    const int64_t& free_row = free_rows[k];
                    begin[k * rows + free_row] = value_t(1);  // Set free variable to 1              
                    for(size_t i = 0; i < pivots.size(); ++i){
                        const int64_t& pivot_row = pivots[i];
                        begin[k * rows + pivot_row] = -(begin2[free_row * cols + i]);
                    }
                }
        }, rT.arr_void());

        #endif
        return std::move(null_space);
    }
    Tensor null_space = functional::zeros({rows, free_vars}, t.dtype);
    #ifdef USE_PARALLEL
    null_space.arr_void().execute_function<WRAP_DTYPES<NumberTypesL> > (
        [&](auto begin, auto end, auto begin2){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            tbb::parallel_for(nt::utils::calculateGrainSize1D(free_rows.size()),
            [&](const tbb::blocked_range<int64_t> &range){
            for(int64_t k = range.begin(); k != range.end(); ++k){
                const int64_t& free_row = free_rows[k];
                begin[free_row * free_vars + k] = value_t(1);  // Set free variable to 1              
                for(size_t i = 0; i < pivots.size(); ++i){
                    const int64_t& pivot_row = pivots[i];
                    begin[pivot_row * free_vars + k] = -(begin2[free_row * cols + i]);
                }
            }});
    }, rT.arr_void());
    #else

    null_space.arr_void().execute_function<WRAP_DTYPES<NumberTypesL> > (
        [&](auto begin, auto end, auto begin2){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            for(size_t k = 0; k < free_rows.size(); ++k){
                const int64_t& free_row = free_rows[k];
                begin[free_row * free_vars + k] = value_t(1);  // Set free variable to 1              
                for(size_t i = 0; i < pivots.size(); ++i){
                    const int64_t& pivot_row = pivots[i];
                    begin[pivot_row * free_vars + k] = -(begin2[free_row * cols + i]);
                }

            }
    }, rT.arr_void());

    #endif
    return std::move(null_space);

 
}

Tensor reduced_null_space(const Tensor& t, bool pivot_rows, bool pivot_first) { 
    utils::throw_exception(t.dims() == 2, "expected to get the null space of a reduced matrix");
    utils::throw_exception(t.dtype != DType::Bool && t.dtype != DType::TensorObj, "Can only find the null space of numerical types, got $", t.dtype);
    if(pivot_rows){return get_reduced_rows_null_space(t, pivot_first);}
    Tensor rT = t;
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
    // std::cout << t << std::endl;
    // print_vec(is_pivot_col);
    std::vector<int64_t> pivots, free_cols;
    pivots.reserve(cols);
    free_cols.reserve(cols);
    for(int64_t k = 0; k < cols; ++k){
        if(is_pivot_col[k]){pivots.push_back(k);}
        else {free_cols.push_back(k);}
    }
    int64_t pivot_cols = pivots.size();
    int64_t free_vars = free_cols.size();
    if(free_vars == 0){
        return Tensor::Null(); //empty tensor, trivial null space
    }
    // std::cout << "pivot cols:";
    // print_vec(pivots);
    // std::cout << "free cols:";
    // print_vec(free_cols);
    // std::cout << "null space is {"<<cols<<','<<free_vars<<"} "<<rows<<',' << cols<<std::endl;


    if(!pivot_first){
        Tensor null_space = functional::zeros({cols, free_vars}, t.dtype);
        #ifdef USE_PARALLEL
        null_space.arr_void().execute_function<WRAP_DTYPES<NumberTypesL> > (
            [&](auto begin, auto end, auto begin2){
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                tbb::parallel_for(nt::utils::calculateGrainSize1D(free_cols.size()),
                [&](const tbb::blocked_range<int64_t> &range){
                for(int64_t k = range.begin(); k != range.end(); ++k){
                    const int64_t& free_col = free_cols[k];
                    begin[free_col * free_vars + k] = value_t(1);  // Set free variable to 1              
                    for(size_t i = 0; i < pivots.size(); ++i){
                        const int64_t& pivot_col = pivots[i];
                        begin[pivot_col * free_vars + k] = -(begin2[i * cols+ free_col]);
                    }
                }});
        }, rT.arr_void());
        #else

        null_space.arr_void().execute_function<WRAP_DTYPES<NumberTypesL> > (
            [&](auto begin, auto end, auto begin2){
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                for(size_t k = 0; k < free_cols.size(); ++k){
                    const int64_t& free_col = free_cols[k];
                    begin[free_col * free_vars + k] = value_t(1);  // Set free variable to 1              
                    for(size_t i = 0; i < pivots.size(); ++i){
                        const int64_t& pivot_col = pivots[i];
                        begin[pivot_col * free_vars + k] = -(begin2[i * cols+ free_col]);
                    }
                }
        }, rT.arr_void());

        #endif
        return std::move(null_space);
    }
   Tensor null_space = functional::zeros({free_vars, cols}, t.dtype);
    #ifdef USE_PARALLEL
    null_space.arr_void().execute_function<WRAP_DTYPES<NumberTypesL> > (
        [&](auto begin, auto end, auto begin2){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            tbb::parallel_for(nt::utils::calculateGrainSize1D(free_cols.size()),
            [&](const tbb::blocked_range<int64_t> &range){
            for(int64_t k = range.begin(); k != range.end(); ++k){
                const int64_t& free_col = free_cols[k];
                begin[k * cols + free_col] = value_t(1);  // Set free variable to 1              
                for(size_t i = 0; i < pivots.size(); ++i){
                    const int64_t& pivot_col = pivots[i];
                    begin[k * cols + pivot_col] = -(begin2[i * cols+ free_col]);
                }
            }});
    }, rT.arr_void());
    #else

    null_space.arr_void().execute_function<WRAP_DTYPES<NumberTypesL> > (
        [&](auto begin, auto end, auto begin2){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            for(size_t k = 0; k < free_cols.size(); ++k){
                const int64_t& free_col = free_cols[k];
                begin[k * cols + free_col] = value_t(1);  // Set free variable to 1              
                for(size_t i = 0; i < pivots.size(); ++i){
                    const int64_t& pivot_col = pivots[i];
                    begin[k * cols + pivot_col] = -(begin2[i * cols+ free_col]);
                }
            }
    }, rT.arr_void());

    #endif
    return std::move(null_space);
  

}
Tensor null_space(const Tensor& _t, std::string mode){
    utils::throw_exception(mode == "svd" || mode == "lu", "Only accepted modes for null space computation are 'svd' and 'lu' got '$'", mode);
    return runEigenFunction(_t, [&mode](auto& mat) -> Tensor{
        using MatrixType = std::remove_cvref_t<decltype(mat)>;
        using ScalarType = typename MatrixType::Scalar;
        if(mode == "lu"){
            auto _n = lu_null_space_eigen(mat);
            return fromEigen(_n);
        }
        auto _n = svd_null_space_eigen(mat);
        return fromEigen(_n);
    });
}



}} //nt::linalg::
