#include "SparseMatrix.h"
#include <cstdlib>
#include <cstring>
#include "../functional/tensor_files/fill.h" //zeros
#include "SparseMacros.h"

namespace nt{



template <typename T>
SparseMatrix
SparseMatrix::from_sortedX(std::vector<int64_t> x, std::vector<int64_t> y,
                           std::vector<T> values, int64_t r, int64_t c, bool sort_y) {
    static_assert(DTypeFuncs::type_is_dtype<T>,
                  "Type in vector is unsupported");
    constexpr DType dt = DTypeFuncs::type_to_dtype<T>;
    std::vector<int64_t> row_ptrs(r + 1, 0);
    utils::throw_exception(
        x.size() == y.size() && x.size() == values.size(),
        "Got different size for values from sorted x to sparse matrix"
        "$, $, $",
        x.size(), y.size(), values.size());

    // std::cout << "dimensions of matrix: {"<<r<<','<<c<<"}"<<std::endl;
    auto row_begin = row_ptrs.begin();
    ++row_begin;
    int64_t last = 0;
    for (const auto &row : x) {
        if (row == last) {
            ++(*row_begin);
            continue;
        }
        int64_t next = *row_begin;
        while (last != row) {
            ++row_begin;
            *row_begin = next;
            ++last;
        }
        ++(*row_begin);
    }

    // std::memcpy(mem, values.data_ptr(), bytes);

    int64_t bytes = sizeof(T) * values.size();
    T *memory = (T*)MetaMalloc(bytes);
    std::vector<int64_t> col_indices;
    if(sort_y){
        T *begin = memory;
        col_indices.resize(y.size());
        auto col_begin = col_indices.begin();

        for (size_t i = 0; i < row_ptrs.size() - 1; ++i) {
            if (row_ptrs[i] == row_ptrs[i + 1])
                continue;
            // sort all of the column indices properly
            std::vector<size_t> indices(row_ptrs[i + 1] - row_ptrs[i]);
            std::iota(indices.begin(), indices.end(), row_ptrs[i]);
            std::sort(indices.begin(), indices.end(),
                      [&](size_t a, size_t b) { return y[a] < y[b]; });
            for (size_t j = 0; j < indices.size(); ++j, ++col_begin, ++begin) {
                *col_begin = y[indices[j]];
                *begin = values[indices[j]];
            }
        }
    }else{
        col_indices = std::move(y);
        if constexpr (std::is_same_v<T, bool>){
            for(size_t i = 0; i < values.size(); ++i)
                memory[i] = values[i];
        }else{
            std::memcpy(memory, &values[0], bytes);
        }
    }

    intrusive_ptr<sparse_details::SparseMemoryMatrixData> data =
        make_intrusive<sparse_details::SparseMemoryMatrixData>(
            sizeof(T), bytes, bytes, memory, std::move(row_ptrs),
            std::move(col_indices));

    return SparseMatrix(r, c, data, dt);
}

#define _NT_DEFINE_SORTED_X_SPARSE_(type)                                      \
    template SparseMatrix SparseMatrix::from_sortedX<type>(                    \
        std::vector<int64_t>, std::vector<int64_t>, std::vector<type>,         \
        int64_t, int64_t, bool);
_NT_DEFINE_SORTED_X_SPARSE_(float)
_NT_DEFINE_SORTED_X_SPARSE_(double)
_NT_DEFINE_SORTED_X_SPARSE_(int64_t)
_NT_DEFINE_SORTED_X_SPARSE_(int32_t)
_NT_DEFINE_SORTED_X_SPARSE_(uint32_t)
_NT_DEFINE_SORTED_X_SPARSE_(int16_t)
_NT_DEFINE_SORTED_X_SPARSE_(uint16_t)
_NT_DEFINE_SORTED_X_SPARSE_(int8_t)
_NT_DEFINE_SORTED_X_SPARSE_(uint8_t)
_NT_DEFINE_SORTED_X_SPARSE_(uint_bool_t)
_NT_DEFINE_SORTED_X_SPARSE_(bool)
_NT_DEFINE_SORTED_X_SPARSE_(complex_64)
_NT_DEFINE_SORTED_X_SPARSE_(complex_128)
_NT_DEFINE_SORTED_X_SPARSE_(uint128_t)
_NT_DEFINE_SORTED_X_SPARSE_(int128_t)
_NT_DEFINE_SORTED_X_SPARSE_(float16_t)
_NT_DEFINE_SORTED_X_SPARSE_(complex_32)
_NT_DEFINE_SORTED_X_SPARSE_(float128_t)

#undef _NT_DEFINE_SORTED_X_SPARSE_

template <typename Iterator> void run_print(Iterator begin, Iterator end, const int64_t& rows, const int64_t& cols) {
    // if constexpr (std::is_same_v<std::remove_const_t<typename Iterator::value_type>, int8_t>){
    //     auto cpy_begin = begin;
    //     for(;cpy_begin != end; ++cpy_begin){
    //         std::cout << "begin at ("<<cpy_begin.Row()<<','<<cpy_begin.Col()<<"): "<<int(*cpy_begin)<<std::endl;
    //     }
    // }
    const int64_t& cur_row = begin.Row();
    int64_t cur_col = begin.Col();
    std::cout << "SparseMatrix([";
    for(int64_t r = 0; r < rows; ++r){
       if(r != 0){
           std::cout << "              ";
       }
       std::cout << '[';
        for(int64_t c = 0; c < cols-1; ++c){
            if(begin == end){std::cout << "0,"; continue;}
            // std::cout << r << " == "<<cur_row<<" "<<c<<" == " << cur_col<<std::endl;
            if(r == cur_row && c == cur_col){
               if constexpr (std::is_same_v<std::remove_const_t<typename Iterator::value_type>
            , int8_t> || std::is_same_v<std::remove_const_t<typename Iterator::value_type>, uint8_t>){
                    std::cout << int(*begin) << ",";
                }else{
                    std::cout << *begin << ",";
                }
                ++begin;
                cur_col = begin.Col();
            }else{
                std::cout << "0,";
            }
        }
        if(r == cur_row && cur_col == cols-1){
            if constexpr (std::is_same_v<std::remove_const_t<typename Iterator::value_type>
            , int8_t> || std::is_same_v<std::remove_const_t<typename Iterator::value_type>, uint8_t>){
                if(r == rows-1){
                    std::cout << int(*begin)<<"]]) ("<<rows<<','<<cols<<')'<<std::endl;    
                }else{
                    std::cout << int(*begin) << "],"<<std::endl;
                }
            }else{
                if(r == rows-1){
                    std::cout << *begin<<"]]) ("<<rows<<','<<cols<<')'<<std::endl;    
                }
                else{
                    std::cout << *begin << "],"<<std::endl;
                }
            }
            ++begin;
            cur_col = begin.Col();
        }else{
            if(r == rows-1){
                std::cout << "0]]) ("<<rows<<','<<cols<<')'<<std::endl;
            }else{
                std::cout << "0],"<<std::endl;
            }
        }
    }
    // std::cout << " ("<<rows<<','<<cols<<")"<<std::endl;
}


// template<typename T>
// inline void _nt_sparse_print_vec(const std::vector<T>& vec){
//     std::cout << '{';
//     for(size_t i = 0; i < vec.size()-1; ++i)
//         std::cout << vec[i]<<',';
//     std::cout << vec.back()<<'}'<<std::endl;
// }

void SparseMatrix::print() const {
    // std::cout << "cols:"<<std::endl;
    // _nt_sparse_print_vec(this->data->get_cols());
    // std::cout << "rows:"<<std::endl;
    // _nt_sparse_print_vec(this->data->get_rows());
    _NT_SPARSE_RUN_CONST_FUNCTION_(_dtype, run_print, this->data, rows, cols);
}

template <typename Iterator> 
void make_tensor(Iterator begin, Iterator end, Tensor& out){
    //Tensor should have been made in zeros already
    const int64_t& rows = out.shape()[0];
    const int64_t& cols = out.shape()[1];
    using value_t = std::remove_const_t<typename Iterator::value_type>;
    value_t* ptr = reinterpret_cast<value_t*>(out.data_ptr());
    for(;begin != end; ++begin){
        const int64_t& row = begin.Row();
        const int64_t& col = begin.Col();
        ptr[(row * cols) + col] = *begin;
    }
}

SparseMatrix::operator Tensor() const {
    Tensor out = functional::zeros({rows, cols}, _dtype);
    _NT_SPARSE_RUN_CONST_FUNCTION_(_dtype, make_tensor, this->data, out);
    return std::move(out);
}

template<typename Iterator>
void perform_transpose(Iterator begin, Iterator end, const std::vector<int64_t>& rows, const std::vector<int64_t>& cols,
                std::vector<int64_t>& trans_rows, std::vector<int64_t>& trans_cols, void* _trans_values,
                const int64_t& num_cols){
    
    using value_t = std::remove_const_t<typename Iterator::value_type>;
    value_t* trans_values = reinterpret_cast<value_t*>(_trans_values);
    const value_t* values = begin.ptr();
    const value_t* values_end = end.ptr();

    int64_t num_rows = rows.size() - 1;
    trans_rows.resize(num_cols + 1, 0);
    trans_cols.resize(cols.size());

    // Step 1: Count occurrences of each column index (future row index)
    std::vector<int64_t> col_counts(num_cols, 0);
    for (int64_t col : cols) {
        col_counts[col]++;
    }


    // Step 2: Compute row pointers for the transposed matrix
    trans_rows[0] = 0;
    for (int64_t i = 0; i < num_cols; ++i) {
        trans_rows[i + 1] = trans_rows[i] + col_counts[i];
    }

    // Step 3: Fill transposed structure
    std::vector<int64_t> position = trans_rows; // Copy of row pointers for tracking insertions
    for (int64_t i = 0; i < num_rows; ++i) {
        for (int64_t j = rows[i]; j < rows[i + 1]; ++j) {
            int64_t col = cols[j];
            int64_t dest = position[col]++;

            trans_cols[dest] = i;
            trans_values[dest] = values[j];
        }
    }

    
    // Step 4: Restore row-major order
    for (int64_t i = 0; i < num_cols; ++i) {
        int64_t start = trans_rows[i], end = trans_rows[i + 1];
        std::vector<std::pair<int64_t, value_t>> row_entries;
        row_entries.reserve(end - start);

        // Collect row elements
        for (int64_t j = start; j < end; ++j) {
            row_entries.emplace_back(trans_cols[j], trans_values[j]);
        }

        // Sort by row index to restore CSR ordering
        std::sort(row_entries.begin(), row_entries.end());

        // Copy back sorted results
        for (int64_t j = start, k = 0; j < end; ++j, ++k) {
            trans_cols[j] = row_entries[k].first;
            trans_values[j] = row_entries[k].second;
        }
    }

    
}


SparseMatrix SparseMatrix::transpose() const{
    const std::vector<int64_t>& rows = this->data->get_rows();
    const std::vector<int64_t>& cols = this->data->get_cols();

    std::vector<int64_t> trans_rows;
    std::vector<int64_t> trans_cols;
    size_t dtype_size = DTypeFuncs::size_of_dtype(_dtype);
    int64_t bytes = dtype_size * this->data->Size(); 
    void* memory = MetaMalloc(bytes);
    
    _NT_SPARSE_RUN_CONST_FUNCTION_(_dtype, perform_transpose, this->data, rows, cols, trans_rows, trans_cols, memory, this->cols);
    
    intrusive_ptr<sparse_details::SparseMemoryMatrixData> out_data =
        make_intrusive<sparse_details::SparseMemoryMatrixData>(
            dtype_size, bytes, bytes, memory, std::move(trans_rows),
            std::move(trans_cols));

    return SparseMatrix(this->cols, this->rows, out_data, _dtype);

}


template<typename Iterator>
void perform_extract_cols(Iterator begin, Iterator end, const intrusive_ptr<sparse_details::SparseMemoryMatrixData>& my_data,
                          const std::vector<bool>& pivot_cols, const int64_t& max_row,
                          intrusive_ptr<sparse_details::SparseMemoryMatrixData>& out_data){
    std::vector<int64_t> col_cntrs(pivot_cols.size());
    int64_t current = 0;
    for(size_t i = 0; i < pivot_cols.size(); ++i){
        col_cntrs[i] = current;
        if(pivot_cols[i]) ++current;
    }
    std::vector<int64_t> n_row_ptrs(max_row+1, 0);
    std::vector<int64_t> n_col_indices;
    const std::vector<int64_t>& row_ptrs = my_data->get_rows();
    const std::vector<int64_t>& col_indices = my_data->get_cols();
    int64_t max_size = row_ptrs[max_row+1];
    n_col_indices.reserve(max_size);
    using value_t = std::remove_const_t<typename Iterator::value_type>;
    value_t* out_mem = (value_t*)MetaMalloc(sizeof(value_t) * (max_size));
    value_t* mem_cpy = out_mem;
    int64_t cntr = 0;
    // int64_t last_row = 0;
    for(;begin != end; ++begin){
        if(begin.Row() >= max_row) {break;}
        while(cntr < begin.Row()){++cntr; n_row_ptrs[cntr+1] = n_row_ptrs[cntr];}
        if(begin.Col() >= pivot_cols.size() || !pivot_cols[begin.Col()]){ continue;}
        *out_mem = *begin;
        ++out_mem;
        n_col_indices.push_back(col_cntrs[begin.Col()]);
        ++n_row_ptrs[cntr+1];
    }
    for(size_t i = 1; i < n_row_ptrs.size(); ++i){
        if(n_row_ptrs[i] == 0){n_row_ptrs[i] = n_row_ptrs[i-1];}
    }
    out_data = make_intrusive<sparse_details::SparseMemoryMatrixData>(
        sizeof(value_t), max_size, n_row_ptrs.back(), mem_cpy, 
                std::move(n_row_ptrs), std::move(n_col_indices));
}

SparseMatrix SparseMatrix::extract_cols(std::vector<bool> pivot_cols, int64_t max_row){
    max_row = (max_row == -1) ? this->rows : max_row;
    utils::throw_exception(max_row <= this->rows, "Got max row of $ but this matrix has $ rows", max_row, this->rows);
    utils::throw_exception(pivot_cols.size() <= this->cols, "This matrix has $ cols but is potentially getting $ cols", this->cols, pivot_cols.size());

    utils::throw_exception(!std::all_of(pivot_cols.begin(), pivot_cols.end(), [](const bool& b){return !b;}), "Cannot extract no columns");
    if(std::all_of(pivot_cols.begin(), pivot_cols.end(), [](const bool& b){return b;})){
        return this->block(0, max_row, 0, pivot_cols.size());
    }
    intrusive_ptr<sparse_details::SparseMemoryMatrixData> out_data(nullptr);
    _NT_SPARSE_RUN_CONST_FUNCTION_(_dtype, perform_extract_cols, this->data, this->data, pivot_cols, max_row, out_data);
    size_t n_cols = std::count(pivot_cols.begin(), pivot_cols.end(), true);
    return SparseMatrix(max_row, n_cols, out_data, _dtype);
    
}

#undef _NT_SPARSE_RUN_FUNCTION_
#undef _NT_SPARSE_RUN_CONST_FUNCTION_ 
#undef _NT_SPARSE_RUN_SINGLE_FUNCTION_
#undef _NT_SPARSE_RUN_SINGLE_CONST_FUNCTION_

} // namespace nt
