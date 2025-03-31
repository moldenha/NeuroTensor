#ifndef _NT_SPARSE_MATRIX_H_
#define _NT_SPARSE_MATRIX_H_
#include "../Tensor.h"
#include "SparseDataMatrix.h"

namespace nt{

class SparseMatrix{
    int64_t rows, cols;
    intrusive_ptr<SparseMemoryMatrixData> data;

public:
    SparseMatrix() = delete;
    SparseMatrix(int64_t r, int64_t c, DType dt = DType::Float32)
    :rows(r), cols(c), data(make_intrusive<SparseMemoryMatrixData>(DTypeFuncs::size_of_dtype(dt), 100, r)
    {}

};

}


#endif

