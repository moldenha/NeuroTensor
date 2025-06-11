// #include "../linalg.h"
#include "../utils/utils.h"
#include "headers/toEigen.hpp"
#include "headers/column_space.h"
#include "../functional/functional.h"

namespace nt{
namespace linalg{


Tensor col_space(const Tensor& _t, std::string mode){
    utils::throw_exception(mode == "svd" || mode == "lu", "Only accepted modes for column space computation are 'svd' and 'lu' got '$'", mode);
    return runEigenFunction(_t, [&mode](auto& mat) -> Tensor{
        using MatrixType = std::remove_cvref_t<decltype(mat)>;
        using ScalarType = typename MatrixType::Scalar;
        if(mode == "lu"){
            auto _n = lu_col_space_eigen(mat);
            return fromEigen(_n);
        }
        auto _n = svd_col_space_eigen(mat);
        return fromEigen(_n);
    });
}


Tensor col_space(const Tensor& original, const Tensor& reduced_space){
    utils::throw_exception(original.dims() == reduced_space.dims() && original.dims() == 2, "Expected to get matrices in order to get the column space of a matrix got $ and $ dims", original.dims(), reduced_space.dims());
    const int64_t& cols = reduced_space.shape()[-1];
    auto [one_rows, one_cols] = get<2>(functional::where(reduced_space == 1));
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
    if(std::all_of(is_pivot_col.begin(), is_pivot_col.end(), [](bool b){ return b; })){return original;}
    if(std::all_of(is_pivot_col.begin(), is_pivot_col.end(), [](bool b){ return !b; })){return Tensor::Null();}
    std::vector<Tensor> out;
    out.reserve(cols);
    Tensor split = original.split_axis(-1);
    Tensor* access = reinterpret_cast<Tensor*>(split.data_ptr());
    utils::THROW_EXCEPTION(split.numel() == cols, "INTERNAL LOGIC ERROR WRONG SIZE $ -? $", split.numel(), cols);
    for(int64_t i = 0; i < cols; ++i){
        if(is_pivot_col[i]){out.emplace_back(access[i]);}
    }
    return functional::stack(out).transpose(-1, -2);
}



}} //nt::linalg::
