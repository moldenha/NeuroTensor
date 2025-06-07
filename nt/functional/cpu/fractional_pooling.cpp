#include "../../refs/SizeRef.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include "../../mp/Threading.h"
#include <algorithm>
#include <vector>
#include <stdexcept>

namespace nt {
namespace functional {
namespace cpu {

void _extract_sliding_windows_max_2d(const ArrayVoid& _input, ArrayVoid& output, 
                const std::vector<int64_t>& rows, const std::vector<int64_t>& cols, 
                int64_t batches, const SizeRef& in_shape){
    ArrayVoid input = _input.contiguous();
    if(!output.is_contiguous() || output.dtype != DType::Bool || in_shape.size() < 2){
        throw std::invalid_argument("Expected output to be contiguous and have a dtype of bool," 
                                    "and the dimensions of the input tensor to be greater than or equal to 2");
    }
    
    int64_t rows_size = static_cast<int64_t>(rows.size());
    int64_t cols_size = static_cast<int64_t>(cols.size());

    bool* o_begin = reinterpret_cast<bool*>(output.data_ptr());
    input.cexecute_function<WRAP_DTYPES<NumberTypesL>>(
    [&rows, &cols, o_begin, &batches, &in_shape](auto begin, auto end){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        int64_t _row = rows.size();
        int64_t _col = cols.size();
        const int64_t& in_cols = in_shape[-1];
        int64_t batch_add = in_shape[-1] * in_shape[-2];
        threading::preferential_parallel_for(
        threading::block_ranges<1>(0, batches),
        [&](threading::blocked_range<1> block) {
        int64_t b_add = (batch_add * block.begin[0]);
        for(int64_t b = block.begin[0]; b < block.end[0]; ++b, b_add += batch_add){
            int64_t cur_row = 0;
            for(int64_t r = 0; r < rows_size; ++r){
                int64_t cur_col = 0;
                for(int64_t c = 0; c < cols_size; ++c){
                    value_t val = begin[(b_add) + cur_row * in_cols + cur_col];
                    bool* b_val = &o_begin[(b_add) + cur_row * in_cols + cur_col];
                    for(int64_t _r = rows[r]-1; _r >= 0; --_r){
                        for(int64_t _c = cols[c]-1; _c >= 0; --_c){
                            if(begin[(b_add) + (cur_row + _r) * in_cols + (cur_col + _c)] > val){
                                val = begin[(b_add) + (cur_row + _r) * in_cols + (cur_col + _c)];
                                b_val = &o_begin[(b_add) + (cur_row + _r) * in_cols + (cur_col + _c)];
                            }
                        }
                    }
                    *b_val = true;
                    cur_col += cols[c];
                }
                cur_row += rows[r];
            }
        }
        });
    });
}

void _extract_sliding_windows_max_3d(const ArrayVoid& _input, ArrayVoid& output, 
                const std::vector<int64_t>& channels, const std::vector<int64_t>& rows, const std::vector<int64_t>& cols, 
                int64_t batches, const SizeRef& in_shape){
    ArrayVoid input = _input.contiguous();
    if(!output.is_contiguous() || output.dtype != DType::Bool || in_shape.size() < 3){
        throw std::invalid_argument("Expected output to be contiguous and have a dtype of bool,"
                                    "and the dimensions of the input tensor to be greater than or equal to 3");
    }


    bool* o_begin = reinterpret_cast<bool*>(output.data_ptr());
    input.cexecute_function<WRAP_DTYPES<NumberTypesL>>(
    [&channels, &rows, &cols, o_begin, &batches, &in_shape](auto begin, auto end){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        int64_t _chan = channels.size();
        int64_t _row = rows.size();
        int64_t _col = cols.size();
        const int64_t& in_cols = in_shape[-1];
        const int64_t in_matrix = in_shape[-2] * in_shape[-1];
        int64_t batch_add = in_shape[-1] * in_shape[-2] * in_shape[-3];
        threading::preferential_parallel_for(
        threading::block_ranges<1>(0, batches),
        [&](threading::blocked_range<1> block) {
        int64_t b_add = (batch_add * block.begin[0]);
        for(int64_t b = block.begin[0]; b < block.end[1]; ++b, b_add += batch_add){
            int64_t cur_chan = 0;
            for(int64_t d = 0; d < channels.size(); ++d){
                int64_t cur_row = 0;
                for(int64_t r = 0; r < rows.size(); ++r){
                    int64_t cur_col = 0;
                    for(int64_t c = 0; c < cols.size(); ++c){
                        value_t val = begin[(b_add) + cur_chan * in_matrix + cur_row * in_cols + cur_col];
                        bool* b_val = &o_begin[(b_add) + cur_chan * in_matrix + cur_row * in_cols + cur_col];
                        for(int64_t _d = channels[d]-1; _d >= 0; --_d){
                            for(int64_t _r = rows[r]-1; _r >= 0; --_r){
                                for(int64_t _c = cols[c]-1; _c >= 0; --_c){
                                    if(begin[(b_add) + (cur_chan + _d) * in_matrix + (cur_row + _r) * in_cols + (cur_col + _c)] > val){
                                        val = begin[(b_add) + (cur_chan + _d) * in_matrix + (cur_row + _r) * in_cols + (cur_col + _c)];
                                        b_val = &o_begin[(b_add) + (cur_chan + _d) * in_matrix + (cur_row + _r) * in_cols + (cur_col + _c)];
                                    }
                                }
                            }
                        }
                        *b_val = true;
                        cur_col += cols[c];
                    }
                    cur_row += rows[r];
                }
                cur_chan += channels[d];
            }
        }
        });
    });

}



} // namespace cpu
} // namespace functional
} // namespace nt
