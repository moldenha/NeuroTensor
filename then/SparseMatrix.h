#ifndef _NT_SPARSE_MATRIX_H_
#define _NT_SPARSE_MATRIX_H_
#include "../Tensor.h"
#include "SparseDataMatrix.h"
#include "SparseMatrixIterator.hpp"
#include <vector>
#include <utility>

namespace nt{

class SparseMatrix{
    int64_t rows, cols;
    DType _dtype;
    intrusive_ptr<sparse_details::SparseMemoryMatrixData> data;
    SparseMatrix(int64_t r, int64_t c, intrusive_ptr<sparse_details::SparseMemoryMatrixData> _data, DType dt)
    :rows(r), cols(c), data(_data), _dtype(dt) {}

public:
    SparseMatrix() = delete;
    SparseMatrix(int64_t r, int64_t c, DType dt = DType::Float32)
    :rows(r), cols(c), data(make_intrusive<sparse_details::SparseMemoryMatrixData>(DTypeFuncs::size_of_dtype(dt), 100, r)),
    _dtype(dt)
    {}
    SparseMatrix(const SparseMatrix& other)
    :rows(other.rows), cols(other.cols), _dtype(other._dtype), data(other.data) {}
    SparseMatrix(SparseMatrix&& other)
    :rows(std::exchange(other.rows, 0)), cols(std::exchange(other.cols, 0)), _dtype(other._dtype), data(std::move(other.data)) {}
    inline const int64_t& Rows() const noexcept {return rows;}
    inline const int64_t& Cols() const noexcept {return cols;}
    inline const DType& dtype() const noexcept {return _dtype;}
    inline const intrusive_ptr<sparse_details::SparseMemoryMatrixData>& get_data() const noexcept {return data;}
    
    void print() const;
    operator Tensor() const;
    SparseMatrix transpose() const;
    template<typename T>
    static SparseMatrix from_sortedX(std::vector<int64_t> x, std::vector<int64_t> y, 
                                     std::vector<T> values, int64_t r, int64_t c, bool sort_y = true);
    inline SparseMatrix block(int64_t start_rows, int64_t end_rows, int64_t start_cols, int64_t end_cols){
        if(start_rows == 0 && end_rows == rows && start_cols == 0 && end_cols == cols) return *this;
        utils::throw_exception(end_cols <= cols, "Expected to get at most $ cols for end cols but got $", cols, end_cols);
        if(end_cols == cols){++end_cols;}
        intrusive_ptr<sparse_details::SparseMemoryMatrixData> n_data = 
            make_intrusive<sparse_details::SparseMemoryMatrixData>(data->block(start_rows, end_rows, start_cols, end_cols));
        if(end_cols == (cols+1)){--end_cols;}
        return SparseMatrix(end_rows-start_rows, end_cols-start_cols, n_data, _dtype);
    }
    SparseMatrix extract_cols(std::vector<bool> pivot_cols, int64_t max_row=-1);
    template<typename T>
    inline details::SMDenseIterator<T> mem_begin() noexcept {return details::SMDenseIterator<T>(this->data, false);}
    template<typename T>
    inline details::SMDenseIterator<const T> cmem_begin() const noexcept {return details::SMDenseIterator<const T>(this->data, false);}
    template<typename T>
    inline details::SMDenseIterator<T> mem_end() noexcept {return details::SMDenseIterator<T>(this->data, true);}
    template<typename T>
    inline details::SMDenseIterator<const T> cmem_end() const noexcept {return details::SMDenseIterator<const T>(this->data, true);}
};

}


#endif

