#include "SparseDataMatrix.h"
#include "../utils/utils.h"

namespace nt{
namespace sparse_details{

SparseMemoryMatrixData SparseMemoryMatrixData::block(int64_t row_start, int64_t row_end, int64_t col_start, int64_t col_end) const {
    utils::throw_exception(row_start < row_end, "Expected row start ($) to be less than row end ($)", row_start, row_end);
    utils::throw_exception(col_start < col_end, "Expected col start ($) to be less than col end ($)", col_start, col_end);
    utils::throw_exception(row_end < row_ptrs.size(), "Expected to get at most $ rows to end at but got $", row_ptrs.size()-1, row_end);
    std::vector<int64_t> n_row_ptrs(row_end-row_start+1, 0);
    std::vector<int64_t> n_col_indices;
    int64_t max_size = (size - (row_ptrs[row_start] + (row_ptrs.back()-row_ptrs[row_end])));
    n_col_indices.reserve(max_size);
    int8_t* out_mem = (int8_t*)std::malloc(type_size * max_size);
    void* mem_cpy = out_mem;
    const int8_t* begin = reinterpret_cast<const int8_t*>(this->memory) + (row_ptrs[row_start] * this->type_size);
    int64_t n_row_ptrs_cur = 0;
    for(int64_t r = row_start; r < row_end; ++r, ++n_row_ptrs_cur){
        n_row_ptrs[n_row_ptrs_cur+1] = n_row_ptrs[n_row_ptrs_cur];
        int64_t col_cntr = 0;
        for(int64_t c = row_ptrs[r]; c < row_ptrs[r+1]; ++c, ++col_cntr, begin += this->type_size){
            if(col_indices[c] < col_start || col_indices[c] >= col_end) continue;
            std::memcpy(out_mem, begin, this->type_size);
            out_mem += this->type_size;
            // std::cout << "got col indice "<<col_indices[c] << std::endl;
            // std::cout << "pushing back "<<(col_indices[c] - col_start)<<std::endl;
            // std::cout << "pushing back column "<<col_indices[c] - col_start<<" and row "<<r<<std::endl;
            n_col_indices.push_back(col_indices[c] - col_start);
            ++n_row_ptrs[n_row_ptrs_cur+1];
        }
    }
    // std::cout << "row ptrs: ";
    // for(const auto& val : n_row_ptrs)
    //     std::cout << val << ' ';
    // std::cout << std::endl;
    // std::cout << "col indices: ";
    // for(const auto& val : n_col_indices)
    //     std::cout << val << ' ';
    // std::cout << std::endl;
    // std::cout << "max size: "<<max_size<<std::endl;
    return SparseMemoryMatrixData(this->type_size, max_size, n_row_ptrs.back(), mem_cpy, 
                                  std::move(n_row_ptrs), std::move(n_col_indices));
}

}
}
