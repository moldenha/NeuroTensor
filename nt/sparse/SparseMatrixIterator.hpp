#ifndef __NT_SPARSE_MATRIX_ITERATOR_HPP__
#define __NT_SPARSE_MATRIX_ITERATOR_HPP__

#include "SparseDataMatrix.h"
#include <iterator>
#include <type_traits>
#include <unistd.h>
#include <vector>

namespace nt{
namespace details{


// this is an iterator that returns the indices that correspond to memory saved
template <typename T> class SMDenseIterator {
  using input_type = std::conditional_t<
      std::is_const_v<T>,
      const intrusive_ptr<sparse_details::SparseMemoryMatrixData> &,
      intrusive_ptr<sparse_details::SparseMemoryMatrixData> &>;
    using vec_type = std::vector<int64_t>::const_iterator;

public : 
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using pointer = T*;
    using value_type = T;

    explicit SMDenseIterator(input_type data, bool end)
        : row_b(data->get_rows().cbegin()), row_e(data->get_rows().cend()),
          col_b(data->get_cols().cbegin()), col_e(data->get_cols().cend()),
          row(end ? data->get_rows().size() - 1 : 0),
          i(end ? data->get_cols().size() - 1 : 0),
          memory(end ? data->template data_ptr_end<std::remove_const_t<T>>()
                     : data->template data_ptr<std::remove_const_t<T>>()) {this->prep_rows();}

    inline const bool operator!=(const SMDenseIterator &b) {
        return memory != b.memory;
    }
    inline const bool operator==(const SMDenseIterator &b) {
        return memory == b.memory;
    }
    inline reference operator*() noexcept {
        return *memory;
    }
    inline pointer operator->() noexcept {return memory;}
    inline const int64_t& Row() const noexcept {return row;}
    inline const int64_t& Col() const noexcept {return *col_b;}
    inline SMDenseIterator &operator++() noexcept {
        ++memory;
        if(col_b+1 == col_e){return *this;}
        ++col_b;
        // std::cout << "incremented col_b and is now "<<*col_b<<std::endl;
        ++i;
        if(row_b+1 == row_e) return *this;
        if(i == row_b[1]){
            do{
                ++row; ++row_b;
            }while(i == row_b[1]);
        }
        return *this;
    }
    inline SMDenseIterator operator++(int) noexcept {
        SMDenseIterator tmp(*this);
        ++(*this);
        return tmp;
    }
    inline T* ptr() noexcept {return memory;}

  private:
    vec_type row_b, row_e, col_b, col_e;
    int64_t row, i;
    T *memory;
    inline void prep_rows(){
        while(row_b != row_e && *row_b == *(row_b+1)){
            ++row; ++row_b;
        }
    }
};


}
}


#endif
