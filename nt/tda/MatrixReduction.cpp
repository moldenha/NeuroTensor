#include "MatrixReduction.h"
#include "../functional/functional.h"
#include "../linalg/linalg.h"
#include "../utils/macros.h"

namespace nt {
namespace tda {

Tensor simultaneousReduce(SparseTensor &d_k, SparseTensor &d_kplus1) {
    utils::throw_exception(
        d_k.dtype() == DType::int8 && d_kplus1.dtype() == DType::int8,
        "Expected boundary matrices to have dtype int8 but got $ and $",
        d_k.dtype(), d_kplus1.dtype());
    utils::throw_exception(
        d_k.dims() == 2 && d_kplus1.dims() == 2,
        "Expected both matrices to have dimensionality of 2 but got $ and $",
        d_k.dims(), d_kplus1.dims());
    utils::throw_exception(
        d_k.shape()[1] == d_kplus1.shape()[0],
        "Matrices have incompatible shapes: d_k is $, d_kplus1 is $",
        d_k.shape(), d_kplus1.shape());

    auto shape_k = d_k.shape();
    int64_t num_rows = shape_k[0];
    int64_t num_cols = shape_k[1];

    int64_t i = 0, j = 0;
    Tensor A = d_k.transpose(-1, -2).underlying_tensor().to(
        DType::Float32); // faster/easier to look at cols with
    int64_t T_cols = A.shape()[-1];
    Tensor B = d_kplus1.underlying_tensor().to(DType::Float32);
    Tensor A_split = A.split_axis(-2);
    Tensor B_split = B.split_axis(-2);
    Tensor *A_begin = reinterpret_cast<Tensor *>(A_split.data_ptr());
    Tensor *B_begin = reinterpret_cast<Tensor *>(B_split.data_ptr());
    utils::throw_exception(A_split.numel() == B_split.numel(),
                           "Expected nummels to be same but got $ and $",
                           A_split.numel(), B_split.numel());

    NT_VLA(const float *, a_access, A_split.numel());
    NT_VLA(const float *, b_access, A_split.numel());
    // const float *a_access[A_split.numel()];
    // const float *b_access[A_split.numel()];
    for (int64_t i = 0; i < A_split.numel(); ++i) {
        a_access[i] = reinterpret_cast<const float *>(A_begin[i].data_ptr());
        b_access[i] = reinterpret_cast<const float *>(B_begin[i].data_ptr());
    }
    i = 0; j = 0;
    while (i < num_rows && j < num_cols) {
        // numpy version is i, j [without transpose]
        //  bool do_continue = false;
        //  bool was_zero = false;
        //  int64_t nonzeroCol = 0;
        if (a_access[j][i] == 0) {
            //if A at row i and column j is 0
            int64_t nonzeroCol = j;
            //go down that row until it is not zero
            while (nonzeroCol < num_cols && a_access[nonzeroCol][i] == 0) {
                ++nonzeroCol;
            }
            //if the entire row is 0, just skip this row
            if (nonzeroCol == num_cols) {
                ++i;
                continue;
            }
            // swap rows
            //
            std::swap(a_access[j], a_access[nonzeroCol]);
            std::swap(b_access[j], b_access[nonzeroCol]);
            std::swap(A_begin[j], A_begin[nonzeroCol]);
            std::swap(B_begin[j], B_begin[nonzeroCol]);
        }

        // without transpose: pivot = A[i, j]
        const float pivot = 1.0 / a_access[j][i];
        // float pivot =
        // A.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Float32>
        // > >(
        //     [&i, &j, &T_cols](auto begin, auto end) -> float {return begin[(j
        //     * T_cols) + i];});

        A_begin[j] *= (int64_t)pivot;
        B_begin[j] *= pivot;

        for (int64_t otherCol = 0; otherCol < num_cols; ++otherCol) {
            if (otherCol == j)
                continue;
            float scaleAmt = a_access[otherCol][i];
            // float scaleAmt =
            // A.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Float32>
            // > >(
            //     [&i, &otherCol, &T_cols](auto begin, auto end) -> float
            //     {return begin[(otherCol * T_cols) + i];});
            if (scaleAmt == 0)
                continue;
            scaleAmt *= -1;
            functional::fused_multiply_add_(A_begin[otherCol], A_begin[j],
                                            scaleAmt);
            functional::fused_multiply_add_(B_begin[j], B_begin[otherCol],
                                            -scaleAmt);
            // colCombine(A, otherCol, j, scaleAmt);
            // rowCombine(B, j, otherCol, -scaleAmt);
        }

        ++i;
        ++j;
    }
    
    NT_VLA_DEALC(a_access);
    NT_VLA_DEALC(b_access);

    return functional::list(functional::cat_unordered(A_split).view(A.shape()),
                            functional::cat_unordered(B_split).view(B.shape()));
}

Tensor &finishRowReducing(Tensor &B) {
    int64_t numRows = B.shape()[0];
    int64_t numCols = B.shape()[1];
    Tensor B_split = B.split_axis(-2);
    Tensor *B_begin = reinterpret_cast<Tensor *>(B_split.data_ptr());
    NT_VLA(const float *, b_access, B_split.numel());
    // const float *b_access[B_split.numel()];
    for (int64_t i = 0; i < B_split.numel(); ++i) {
        b_access[i] = reinterpret_cast<const float *>(B_begin[i].data_ptr());
    }

    int64_t i = 0, j = 0;
    while (i < numRows && j < numCols) {
        if (b_access[i][j] == 0) {
            int64_t nonzeroRow = i + 1;
            while (nonzeroRow < numRows && b_access[nonzeroRow][j] == 0) {
                ++nonzeroRow;
            }
            if (nonzeroRow == numRows) {
                ++j;
                continue;
            }
            // swap rows
            std::swap(b_access[i], b_access[nonzeroRow]);
            std::swap(B_begin[i], B_begin[nonzeroRow]);
        }

        float pivot = 1.0 / b_access[i][j];

        B_begin[i] *= pivot;

        // could probably use a parallel for loop to speed this up
        for (int64_t otherRow = 0; otherRow < numRows; ++otherRow) {
            if (otherRow == i)
                continue;
            float scaleAmt = b_access[otherRow][j];

            if (scaleAmt == 0)
                continue;
            // scaleAmt *= -1;
            functional::fused_multiply_add_(B_begin[otherRow], B_begin[i],
                                            scaleAmt);
        }

        ++i;
        ++j;
    }
    NT_VLA_DEALC(b_access);
    Tensor catted = nt::functional::cat_unordered(B_split).view(B.shape());
    std::swap(B, catted);
    return B;
}

int64_t numPivotRows(Tensor &A) {
    Tensor zeros = functional::all(A == 0, -2);
    return A.shape()[-2] - functional::count(zeros);
}

Tensor rowReduce(Tensor A) {
    A = A.contiguous();
    int64_t m = A.shape()[-2];
    int64_t n = A.shape()[-1];
    int64_t row = 0;
    Tensor A_split = A.split_axis(-2);
    Tensor *A_begin = reinterpret_cast<Tensor *>(A_split.data_ptr());
    NT_VLA(const float*, a_access, A_split.numel());
    // const float *a_access[A_split.numel()];
    for (int64_t i = 0; i < A_split.numel(); ++i) {
        a_access[i] = reinterpret_cast<const float *>(A_begin[i].data_ptr());
    }

    for (int64_t col = 0; col < n; ++col) {
        if (row >= m)
            break;

        // Find pivot row
        int64_t pivot_row = row;
        while (pivot_row < m && a_access[pivot_row][col] == 0) {
            pivot_row++;
        }

        if (pivot_row == m)
            continue; // No pivot in this column, skip

        // Swap rows if necessary
        if (pivot_row != row) {
            std::swap(A_begin[pivot_row], A_begin[row]);
            std::swap(a_access[pivot_row], a_access[row]);
        }

        // Normalize pivot row
        float pivot_val = 1.0 / a_access[row][col];
        A_begin[row] *= pivot_val;

        // Eliminate below pivot
        for (int64_t i = 0; i < m; ++i) {
            if (i == row)
                continue;
            float factor = a_access[i][col];
            functional::fused_multiply_add_(A_begin[i], A_begin[row],
                                            (-1 * factor));
            // matrix.row(i) -= factor * matrix.row(row);
        }

        row++; // Move to the next row
    }

    NT_VLA_DEALC(a_access);
    return std::move(A);
}

//this partially reduces a matrix given a start and end
//d_k is transposed compared to what simultaneousReduce gets
Tensor simultaneousCatReduce(Tensor &d_k, Tensor &d_kplus1, 
                             int64_t start_rows, int64_t start_cols, int64_t end_rows, int64_t end_cols) {
    utils::throw_exception(
        d_k.dtype() == DType::Float32 && d_kplus1.dtype() == DType::Float32,
        "Expected boundary matrices to have dtype int8 but got $ and $",
        d_k.dtype(), d_kplus1.dtype());
    utils::throw_exception(
        d_k.dims() == 2 && d_kplus1.dims() == 2,
        "Expected both matrices to have dimensionality of 2 but got $ and $",
        d_k.dims(), d_kplus1.dims());
    //d_k.shape()[1] == d_kplus1.shape()[0] <- Before transpose
    utils::throw_exception(
        d_k.shape()[0] == d_kplus1.shape()[0],
        "Matrices have incompatible shapes: d_k is $, d_kplus1 is $",
        d_k.shape(), d_kplus1.shape());
    //the swaps are done to correspond now to the original d_k before it was transposed
    std::swap(start_rows, start_cols);
    std::swap(end_rows, end_cols);
    if(start_rows == d_k.shape()[1]) return functional::list(d_k, d_kplus1);
    utils::throw_exception(start_rows < end_rows,
                           "Expected start rows ($) to be less than end rows ($)", start_rows, end_rows);
    utils::throw_exception(start_cols < end_cols,
                           "Expected start cols ($) to be less than end cols ($)", start_cols, end_cols);
    utils::throw_exception(end_cols <= d_k.shape()[0],
                           "Expected end rows ($) to be less than the number of rows in d_k ($) and d_k+1 ($)",
                           end_cols, d_k.shape()[0], d_kplus1.shape()[0]);
    utils::throw_exception(end_rows <= d_k.shape()[1] && end_rows <= d_kplus1.shape()[1],
                           "Expected end cols ($) to be less than the number of cols in d_k ($) and d_k+1 ($)",
                           end_rows, d_k.shape()[1], d_kplus1.shape()[0]);


    // auto shape_k = d_k.shape();
    // int64_t num_rows = shape_k[0];
    // int64_t num_cols = shape_k[1];

    int64_t i = 0, j = 0;
    // Tensor A = d_k.transpose(-1, -2).underlying_tensor().to(
        // DType::Float32); // faster/easier to look at cols with
    // int64_t T_cols = A.shape()[-1];
    // Tensor B = d_kplus1.underlying_tensor().to(DType::Float32);
    Tensor A_split = d_k.split_axis(-2);
    Tensor B_split = d_kplus1.split_axis(-2);
    Tensor *A_begin = reinterpret_cast<Tensor *>(A_split.data_ptr());
    Tensor *B_begin = reinterpret_cast<Tensor *>(B_split.data_ptr());
    utils::throw_exception(A_split.numel() == B_split.numel(),
                           "Expected nummels to be same but got $ and $",
                           A_split.numel(), B_split.numel());

    int64_t num_cols = A_split.numel();
    NT_VLA(const float*, a_access, num_cols);
    NT_VLA(const float*, b_access, num_cols);
    // const float *a_access[num_cols];
    // const float *b_access[num_cols];
    for (int64_t i = 0; i < num_cols; ++i) {
        a_access[i] = reinterpret_cast<const float *>(A_begin[i].data_ptr());
        b_access[i] = reinterpret_cast<const float *>(B_begin[i].data_ptr());
    }
    i = 0; j = 0;
    while(i < start_rows && j < start_cols){
        if(a_access[j][i] == 0){
           //if A at row i and column j is 0
            int64_t nonzeroCol = std::max(j+1, start_cols);
            //go down that row until it is not zero
            while (nonzeroCol < end_cols && a_access[nonzeroCol][i] == 0) {
                ++nonzeroCol;
            }
            //if the entire row is 0, just skip this row
            if (nonzeroCol == end_cols) {
                ++i;
                continue;
            }
 
        }
        ++j;
        ++i;
    }
    while (i < end_rows && j < end_cols) {
        // numpy version is i, j [without transpose]
        //  bool do_continue = false;
        //  bool was_zero = false;
        //  int64_t nonzeroCol = 0;
        if (a_access[j][i] == 0) {
            //if A at row i and column j is 0
            int64_t nonzeroCol = j+1;
            // int64_t nonzeroCol = std::max(j+1, start_cols);
            //go down that row until it is not zero
            while (nonzeroCol < end_cols && a_access[nonzeroCol][i] == 0) {
                ++nonzeroCol;
            }
            //if the entire row is 0, just skip this row
            if (nonzeroCol == end_cols) {
                ++i;
                continue;
            }
            // swap cols
            //
            std::swap(a_access[j], a_access[nonzeroCol]);
            std::swap(b_access[j], b_access[nonzeroCol]);
            std::swap(A_begin[j], A_begin[nonzeroCol]);
            std::swap(B_begin[j], B_begin[nonzeroCol]);
        }

        // without transpose: pivot = A[i, j]
        const float pivot = 1.0 / a_access[j][i];
        // float pivot =
        // A.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Float32>
        // > >(
        //     [&i, &j, &T_cols](auto begin, auto end) -> float {return begin[(j
        //     * T_cols) + i];});

        A_begin[j] *= (int64_t)pivot;
        B_begin[j] *= pivot;

        for (int64_t otherCol = 0; otherCol < end_cols; ++otherCol) {
            if (otherCol == j)
                continue;
            float scaleAmt = a_access[otherCol][i];
            // float scaleAmt =
            // A.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Float32>
            // > >(
            //     [&i, &otherCol, &T_cols](auto begin, auto end) -> float
            //     {return begin[(otherCol * T_cols) + i];});
            if (scaleAmt == 0)
                continue;
            scaleAmt *= -1;
            functional::fused_multiply_add_(A_begin[otherCol], A_begin[j],
                                            scaleAmt);
            functional::fused_multiply_add_(B_begin[j], B_begin[otherCol],
                                            -scaleAmt);
            // colCombine(A, otherCol, j, scaleAmt);
            // rowCombine(B, j, otherCol, -scaleAmt);
        }

        ++i;
        ++j;
    }

    NT_VLA_DEALC(a_access);
    NT_VLA_DEALC(b_access);
    return functional::list(functional::cat_unordered(A_split).view(d_k.shape()),
                            functional::cat_unordered(B_split).view(d_kplus1.shape()));
}




Tensor &finishCatRowReducing(Tensor &B,
                             int64_t start_rows, int64_t start_cols, int64_t end_rows, int64_t end_cols) {
    int64_t numRows = B.shape()[0];
    // int64_t numCols = B.shape()[1];
    Tensor B_split = B.split_axis(-2);
    Tensor *B_begin = reinterpret_cast<Tensor *>(B_split.data_ptr());
    NT_VLA(const float*, b_access, B_split.numel());
    // const float *b_access[B_split.numel()];
    for (int64_t i = 0; i < B_split.numel(); ++i) {
        b_access[i] = reinterpret_cast<const float *>(B_begin[i].data_ptr());
    }

    int64_t i = 0, j = 0;
    while (i < end_rows && j < end_cols) {
        if (b_access[i][j] == 0) {
            int64_t nonzeroRow = std::max(i + 1, start_rows);
            while (nonzeroRow < end_rows && b_access[nonzeroRow][j] == 0) {
                ++nonzeroRow;
            }
            if (nonzeroRow == end_rows) {
                ++j;
                continue;
            }
            // swap rows
            std::swap(b_access[i], b_access[nonzeroRow]);
            std::swap(B_begin[i], B_begin[nonzeroRow]);
        }

        float pivot = 1.0 / b_access[i][j];

        B_begin[i] *= pivot;

        // could probably use a parallel for loop to speed this up
        for (int64_t otherRow = 0; otherRow < end_rows; ++otherRow) {
            if (otherRow == i)
                continue;
            float scaleAmt = b_access[otherRow][j];

            if (scaleAmt == 0)
                continue;
            // scaleAmt *= -1;
            functional::fused_multiply_add_(B_begin[otherRow], B_begin[i],
                                            scaleAmt);
        }

        ++i;
        ++j;
    }
    NT_VLA_DEALC(b_access);
    return B;
}



//this partially reduces a matrix given a start and end
//d_k is transposed compared to what simultaneousReduce gets
Tensor& partialColReduce(Tensor &d_k, 
                             int64_t start_rows, int64_t start_cols, int64_t end_rows, int64_t end_cols) {
    utils::throw_exception(
        d_k.dtype() == DType::Float32,
        "Expected boundary matrices to have dtype float32 but got $",
        d_k.dtype());
    utils::throw_exception(
        d_k.dims() == 2,
        "Expected both matrices to have dimensionality of 2 but got $",
        d_k.dims());
    //d_k.shape()[1] == d_kplus1.shape()[0] <- Before transpose
    //the swaps are done to correspond now to the original d_k before it was transposed
    std::swap(start_rows, start_cols);
    std::swap(end_rows, end_cols);
    if(start_rows == d_k.shape()[1]) return d_k;
    utils::throw_exception(start_rows < end_rows,
                           "Expected start rows ($) to be less than end rows ($)", start_rows, end_rows);
    utils::throw_exception(start_cols < end_cols,
                           "Expected start cols ($) to be less than end cols ($)", start_cols, end_cols);
    utils::throw_exception(end_cols <= d_k.shape()[0],
                           "Expected end rows ($) to be less than the number of rows in d_k ($)",
                           end_cols, d_k.shape()[0]);
    utils::throw_exception(end_rows <= d_k.shape()[1] ,
                           "Expected end cols ($) to be less than the number of cols in d_k ($)",
                           end_rows, d_k.shape()[1]);


    // auto shape_k = d_k.shape();
    // int64_t num_rows = shape_k[0];
    // int64_t num_cols = shape_k[1];

    int64_t i = 0, j = 0;
    // Tensor A = d_k.transpose(-1, -2).underlying_tensor().to(
        // DType::Float32); // faster/easier to look at cols with
    // int64_t T_cols = A.shape()[-1];
    // Tensor B = d_kplus1.underlying_tensor().to(DType::Float32);
    Tensor A_split = d_k.split_axis(-2);
    Tensor *A_begin = reinterpret_cast<Tensor *>(A_split.data_ptr());

    int64_t num_cols = A_split.numel();
    NT_VLA(const float*, a_access, num_cols);
    // const float *a_access[num_cols];
    for (int64_t i = 0; i < num_cols; ++i) {
        a_access[i] = reinterpret_cast<const float *>(A_begin[i].data_ptr());
    }
    i = 0; j = 0;
    while(i < start_rows && j < start_cols){
        if(a_access[j][i] == 0){
           //if A at row i and column j is 0
            int64_t nonzeroCol = std::max(j+1, start_cols);
            //go down that row until it is not zero
            while (nonzeroCol < end_cols && a_access[nonzeroCol][i] == 0) {
                ++nonzeroCol;
            }
            //if the entire row is 0, just skip this row
            if (nonzeroCol == end_cols) {
                ++i;
                continue;
            }
 
        }
        ++j;
        ++i;
    }
    while (i < end_rows && j < end_cols) {
        // numpy version is i, j [without transpose]
        //  bool do_continue = false;
        //  bool was_zero = false;
        //  int64_t nonzeroCol = 0;
        if (a_access[j][i] == 0) {
            //if A at row i and column j is 0
            int64_t nonzeroCol = std::max(j+1, start_cols);
            //go down that row until it is not zero
            while (nonzeroCol < end_cols && a_access[nonzeroCol][i] == 0) {
                ++nonzeroCol;
            }
            //if the entire row is 0, just skip this row
            if (nonzeroCol == end_cols) {
                ++i;
                continue;
            }
            // swap cols
            //
            std::swap(a_access[j], a_access[nonzeroCol]);
            std::swap(A_begin[j], A_begin[nonzeroCol]);
        }

        // without transpose: pivot = A[i, j]
        const float pivot = 1.0 / a_access[j][i];
        // float pivot =
        // A.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Float32>
        // > >(
        //     [&i, &j, &T_cols](auto begin, auto end) -> float {return begin[(j
        //     * T_cols) + i];});

        A_begin[j] *= (int64_t)pivot;

        for (int64_t otherCol = 0; otherCol < end_cols; ++otherCol) {
            if (otherCol == j)
                continue;
            float scaleAmt = a_access[otherCol][i];
            // float scaleAmt =
            // A.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Float32>
            // > >(
            //     [&i, &otherCol, &T_cols](auto begin, auto end) -> float
            //     {return begin[(otherCol * T_cols) + i];});
            if (scaleAmt == 0)
                continue;
            scaleAmt *= -1;
            functional::fused_multiply_add_(A_begin[otherCol], A_begin[j],
                                            scaleAmt);
            // colCombine(A, otherCol, j, scaleAmt);
            // rowCombine(B, j, otherCol, -scaleAmt);
        }

        ++i;
        ++j;
    }
    Tensor catted = nt::functional::cat_unordered(A_split).view(d_k.shape());
    std::swap(d_k, catted);
    NT_VLA_DEALC(a_access);
    // std::cout << A_split[0] << ',' << d_k[0] << std::endl;
    // std::cout << A_begin[0] << ',' << d_k[0] << std::endl;
    // std::cout << std::boolalpha << nt::functional::all(A_split[0] == d_k[0]) << std::noboolalpha << std::endl;
    return d_k;
}




Tensor &partialRowReduce(Tensor &B,
                             int64_t start_rows, int64_t start_cols, int64_t end_rows, int64_t end_cols) {
    utils::throw_exception(
        B.dtype() == DType::Float32,
        "Expected boundary matrices to have dtype float32 but got $",
        B.dtype());
    utils::throw_exception(
        B.dims() == 2,
        "Expected both matrices to have dimensionality of 2 but got $",
        B.dims());
    utils::throw_exception(start_rows < end_rows,
                           "Expected start rows ($) to be less than end rows ($)", start_rows, end_rows);
    utils::throw_exception(start_cols < end_cols,
                           "Expected start cols ($) to be less than end cols ($)", start_cols, end_cols);
    utils::throw_exception(end_rows <= B.shape()[0],
                           "Expected end rows ($) to be less than the number of rows in d_k ($)",
                           end_rows, B.shape()[0]);
    utils::throw_exception(end_cols <= B.shape()[1] ,
                           "Expected end cols ($) to be less than the number of cols in d_k ($)",
                           end_cols, B.shape()[1]);
    int64_t numRows = B.shape()[0];
    // int64_t numCols = B.shape()[1];
    Tensor B_split = B.split_axis(-2);
    Tensor *B_begin = reinterpret_cast<Tensor *>(B_split.data_ptr());
    NT_VLA(const float*, b_access, B_split.numel());
    // const float *b_access[B_split.numel()];
    for (int64_t i = 0; i < B_split.numel(); ++i) {
        b_access[i] = reinterpret_cast<const float *>(B_begin[i].data_ptr());
    }

    int64_t i = 0, j = 0;
    //while(i < start_rows && j < start_cols){
    //    if(b_access[i][j] == 0){
    //       //if A at row i and column j is 0
    //        int64_t nonzeroRow = std::max(i+1, start_rows);
    //        //go down that row until it is not zero
    //        while (nonzeroRow < end_rows && b_access[nonzeroRow][j] == 0) {
    //            ++nonzeroRow;
    //        }
    //        //if the entire row is 0, just skip this row
    //        if (nonzeroRow == end_rows) {
    //            ++j;
    //            continue;
    //        }
 
    //    }
    //    ++j;
    //    ++i;
    //}
    while (i < end_rows && j < end_cols) {
        if (b_access[i][j] == 0) {
            int64_t nonzeroRow = std::max(i + 1, start_rows);
            while (nonzeroRow < end_rows && b_access[nonzeroRow][j] == 0) {
                ++nonzeroRow;
            }
            if (nonzeroRow == end_rows) {
                ++j;
                continue;
            }
            // swap rows
            std::swap(b_access[i], b_access[nonzeroRow]);
            std::swap(B_begin[i], B_begin[nonzeroRow]);
        }

        float pivot = 1.0 / b_access[i][j];

        B_begin[i] *= pivot;

        // could probably use a parallel for loop to speed this up
        //if you want the exact same matrix output at the simultaneous reduce
        //set otherRow = 0
        for (int64_t otherRow = i+1; otherRow < end_rows; ++otherRow) {
            // if (otherRow == i)
            //     continue;
            float scaleAmt = b_access[otherRow][j];

            if (scaleAmt == 0)
                continue;
            // scaleAmt *= -1;
            functional::fused_multiply_add_(B_begin[otherRow], B_begin[i],
                                            scaleAmt);
        }

        ++i;
        ++j;
    }
    Tensor catted = nt::functional::cat_unordered(B_split).view(B.shape());
    std::swap(B, catted);
    NT_VLA_DEALC(b_access);
    return B;
}


void print_matrix(float** mat, int64_t rows, int64_t cols){
    std::cout << '[';
    for(int64_t r = 0; r < rows; ++r){
        std::cout << '[';
        for(int64_t c = 0; c < cols-1; ++c){
            std::cout << mat[r][c]<<',';
        }
        if(r != rows-1){
            std::cout << mat[r][cols-1] << "],"<<std::endl;
        }
        else{
            std::cout << mat[r][cols-1] << "]]"<<std::endl;
        }
    }
}


void partialColReduce(Tensor*& A_begin, float** &a_access,
                             int64_t start_rows, int64_t start_cols, int64_t end_rows, int64_t end_cols) {

    

    std::swap(start_rows, start_cols);
    std::swap(end_rows, end_cols);
    // if(end_rows == 30 && end_cols == 13){
    //     std::cout << "input matrix:"<<std::endl;
    //     print_matrix(a_access, end_rows, end_cols);
    // }
    int64_t i = 0, j = 0;

    while (i < end_rows && j < end_cols) {
        // numpy version is i, j [without transpose]
        //  bool do_continue = false;
        //  bool was_zero = false;
        //  int64_t nonzeroCol = 0;
        if (a_access[j][i] == 0) {
            //if A at row i and column j is 0
            int64_t nonzeroCol = std::max(j+1, start_cols);
            //go down that row until it is not zero
            while (nonzeroCol < end_cols && a_access[nonzeroCol][i] == 0) {
                ++nonzeroCol;
            }
            //if the entire row is 0, just skip this row
            if (nonzeroCol == end_cols) {
                // if(end_rows == 30 && end_cols == 25){
                //     std::cout << "skipping "<<i<<std::endl;
                // }

                ++i;
                continue;
            }
            // swap cols
            // if(end_rows == 30 && end_cols == 25){
            //     std::cout << "swapping "<<j<<" and "<<nonzeroCol<<std::endl;
            // }
            std::swap(a_access[j], a_access[nonzeroCol]);
            std::swap(A_begin[j], A_begin[nonzeroCol]);
        }

        // without transpose: pivot = A[i, j]
        const float pivot = 1.0 / a_access[j][i];
        // if(end_rows == 30 && end_cols == 25){
        //     std::cout << "pivot is ("<<j<<") "<<pivot<<std::endl;
        // }
        // if(end_rows == 30 && end_cols == 25 && (j == 2 || j == 11)){
        //     print_matrix(a_access, end_rows, end_cols);
        //     std::cout << "printed"<<std::endl;
        // }

        A_begin[j] *= (int64_t)pivot;
        // if(end_rows == 30 && end_cols == 25 && (j == 2 || j == 11)){
        //     print_matrix(a_access, end_rows, end_cols);
        //     std::cout << "printed after"<<std::endl;
        // }
        for (int64_t otherCol = j+1; otherCol < end_cols; ++otherCol) {
            if (otherCol == j)
                continue;
            float scaleAmt = a_access[otherCol][i];
            if (scaleAmt == 0)
                continue;
            scaleAmt *= -1;
            functional::fused_multiply_add_(A_begin[otherCol], A_begin[j],
                                            scaleAmt);
        }
        
        ++i;
        ++j;
    }

}


void partialRowReduce(Tensor*& B_begin, float** &b_access,
                             int64_t start_rows, int64_t start_cols, int64_t end_rows, int64_t end_cols) {


    int64_t i = 0, j = 0;
    while (i < end_rows && j < end_cols) {
        if (b_access[i][j] == 0) {
            int64_t nonzeroRow = std::max(i + 1, start_rows);
            while (nonzeroRow < end_rows && b_access[nonzeroRow][j] == 0) {
                ++nonzeroRow;
            }
            if (nonzeroRow == end_rows) {
                ++j;
                continue;
            }
            // swap rows
            std::swap(b_access[i], b_access[nonzeroRow]);
            std::swap(B_begin[i], B_begin[nonzeroRow]);
        }

        float pivot = 1.0 / b_access[i][j];

        B_begin[i] *= pivot;

        // could probably use a parallel for loop to speed this up
        for (int64_t otherRow = i+1; otherRow < end_rows; ++otherRow) {
            if (otherRow == i)
                continue;
            float scaleAmt = b_access[otherRow][j];

            if (scaleAmt == 0)
                continue;
            scaleAmt *= -1;
            functional::fused_multiply_add_(B_begin[otherRow], B_begin[i],
                                            scaleAmt);
        }

        ++i;
        ++j;
    }
}

int64_t numPivotRows(float** &access, int64_t rows, int64_t cols) {
    int64_t count = 0;
    for(int64_t i = 0; i < rows; ++i){
        if(std::all_of(access[i], access[i] + cols, [](const float& val){return val == 0;})) ++count;
    }
    return rows - count;
}


std::map<double, int64_t>
    getBettiNumbers(SparseTensor& d_k, SparseTensor& d_kplus1, std::map<double, std::tuple<int64_t, int64_t, int64_t> > radi_bounds, double max, bool add_zeros){
    utils::throw_exception(
        d_k.dtype() == DType::int8 && d_kplus1.dtype() == DType::int8,
        "Expected boundary matrices to have dtype int8 but got $ and $",
        d_k.dtype(), d_kplus1.dtype());
    utils::throw_exception(
        d_k.dims() == 2 && d_kplus1.dims() == 2,
        "Expected both matrices to have dimensionality of 2 but got $ and $",
        d_k.dims(), d_kplus1.dims());
    utils::throw_exception(
        d_k.shape()[1] == d_kplus1.shape()[0],
        "Matrices have incompatible shapes: d_k is $, d_kplus1 is $",
        d_k.shape(), d_kplus1.shape());
 
    Tensor A = d_k.underlying_tensor().transpose(-1, -2).to(DType::Float32);
    Tensor B = d_kplus1.underlying_tensor().to(DType::Float32);
    
    int64_t astart_rows = 0; //retain
    int64_t astart_cols = 0;

    int64_t bstart_rows = 0; //retain
    int64_t bstart_cols = 0;
    Tensor A_split = A.split_axis(-2);
    Tensor B_split = B.split_axis(-2);
    Tensor *A_begin = reinterpret_cast<Tensor *>(A_split.data_ptr());
    Tensor *B_begin = reinterpret_cast<Tensor *>(B_split.data_ptr());
    utils::throw_exception(A_split.numel() == B_split.numel(),
                           "Expected nummels to be same but got $ and $",
                           A_split.numel(), B_split.numel());
    
    float **a_access = MetaNewArr(float*, A_split.numel());
    float **b_access = MetaNewArr(float*, A_split.numel());
    for (int64_t i = 0; i < A_split.numel(); ++i) {
        a_access[i] = reinterpret_cast<float *>(A_begin[i].data_ptr());
        b_access[i] = reinterpret_cast<float *>(B_begin[i].data_ptr());
    }
    //partialColReduce(A_begin, a_access, astart_rows, astart_cols, aend_rows [d_k.shape()[0]], aend_cols)
    int i = 0;
    std::map<double, int64_t> out;
    int cntr = 0;
    for(const auto& correspond : radi_bounds){
        auto [km1_size, k_size, kp1_size] = correspond.second;
        if(km1_size < astart_rows || km1_size > A.shape()[1]) continue;
        if(k_size < astart_cols || k_size > A.shape()[0]) continue;
        if(k_size < bstart_rows || k_size > B.shape()[0]) continue;
        if(kp1_size < bstart_cols || kp1_size > B.shape()[1]) continue;
        if(max > 0 && correspond.first > max) break;
        // SparseTensor _sub_bk = d_k[{range_(0, km1_size), range_(0, k_size)}];
        // SparseTensor _sub_bk1 = d_kplus1[{nt::range_(0, k_size), nt::range_(0, kp1_size)}];
        // auto [_sub_A, _sub_B] = get<2>(simultaneousReduce(_sub_bk, _sub_bk1));
        // finishRowReducing(_sub_B);
        // int64_t _rank_k = numPivotRows(_sub_A);
        // int64_t _rank_kp1 = numPivotRows(_sub_B);
        // int64_t dimKChains = _sub_bk.shape()[1];
        // int64_t kernelDim = dimKChains - _rank_k;
        // int64_t _betti = kernelDim - _rank_kp1;
        partialColReduce(A_begin, a_access,
                         astart_cols, astart_rows,
                         k_size, km1_size);

        astart_cols = k_size;
        //the following works which is nice
        partialRowReduce(B_begin, b_access,
                        bstart_rows, bstart_cols,
                        k_size, kp1_size);
        bstart_cols = kp1_size;
        // if(i == 0

        int64_t rank_k = numPivotRows(a_access, k_size, km1_size);
        int64_t rank_kp1 = numPivotRows(b_access, k_size, kp1_size);
        // std::cout << k_size<<','<<km1_size<<": "<<rank_k<<std::endl;
        // std::cout << k_size<<','<<kp1_size<<": "<<rank_kp1<<std::endl;

        //dimKChains = k_size
        //kernelDim = dimKChains - rank_k = k_size - rank_k;
        //betti = kernelDim - rank_kp1
        int64_t betti = (k_size - rank_k) - rank_kp1;
        // std::cout << "ranks: {"<<rank_k<<","<<rank_kp1<<"}, {"<<_rank_k<<","<<_rank_kp1<<"}"<<std::endl;
        // std::cout << correspond.first << ": old betti: "<<_betti<<"new betti: "<<betti<<std::endl;
        if(betti > 0){ 
            out[correspond.first] = betti;
        }else if (add_zeros){
            out[correspond.first] = 0;
        }
        // if(k_size == 25 && km1_size == 30){
        //     Tensor new_A = nt::functional::cat_unordered(A_split).view(d_k.shape().transpose(-1, -2))[range_(0, 32)];
        //     // Tensor new_A = nt::functional::cat_unordered(A_split).view(d_k.shape().transpose(-1, -2));
        //     std::cout << new_A<<std::endl;
        //     // d_k.underlying_tensor().transpose(-1, -2)[{range_(0, k_size), range_(0, km1_size)}].print();
        //     std::cout << "rank for a: "<<rank_k<<std::endl;
        // }
        ++cntr;
    }
    MetaFreeArr<float*>(a_access);
    MetaFreeArr<float*>(b_access);
    return std::move(out);              
}


std::pair<std::map<double, int64_t>, std::map<double, Tensor>> getBettiNumbersColSpace(
    SparseTensor &d_k, SparseTensor &d_kplus1,
    std::map<double, std::tuple<int64_t, int64_t, int64_t>> radi_bounds,
    double max, bool add_zeros){

    utils::throw_exception(
        d_k.dtype() == DType::int8 && d_kplus1.dtype() == DType::int8,
        "Expected boundary matrices to have dtype int8 but got $ and $",
        d_k.dtype(), d_kplus1.dtype());
    utils::throw_exception(
        d_k.dims() == 2 && d_kplus1.dims() == 2,
        "Expected both matrices to have dimensionality of 2 but got $ and $",
        d_k.dims(), d_kplus1.dims());
    utils::throw_exception(
        d_k.shape()[1] == d_kplus1.shape()[0],
        "Matrices have incompatible shapes: d_k is $, d_kplus1 is $",
        d_k.shape(), d_kplus1.shape());
 
    Tensor A = d_k.underlying_tensor().transpose(-1, -2).to(DType::Float32);
    Tensor B = d_kplus1.underlying_tensor().to(DType::Float32);
    
    int64_t astart_rows = 0; //retain
    int64_t astart_cols = 0;

    int64_t bstart_rows = 0; //retain
    int64_t bstart_cols = 0;
    Tensor A_split = A.split_axis(-2);
    Tensor B_split = B.split_axis(-2);
    Tensor *A_begin = reinterpret_cast<Tensor *>(A_split.data_ptr());
    Tensor *B_begin = reinterpret_cast<Tensor *>(B_split.data_ptr());
    utils::throw_exception(A_split.numel() == B_split.numel(),
                           "Expected nummels to be same but got $ and $",
                           A_split.numel(), B_split.numel());

    float **a_access = MetaNewArr(float*, A_split.numel());
    float **b_access = MetaNewArr(float*, A_split.numel());
    for (int64_t i = 0; i < A_split.numel(); ++i) {
        a_access[i] = reinterpret_cast<float *>(A_begin[i].data_ptr());
        b_access[i] = reinterpret_cast<float *>(B_begin[i].data_ptr());
    }
    //partialColReduce(A_begin, a_access, astart_rows, astart_cols, aend_rows [d_k.shape()[0]], aend_cols)
    // int cntr = 0;
    std::map<double, Tensor > col_spaces;
    std::map<double, int64_t> betti_numbers;
    for(const auto& correspond : radi_bounds){
        //maps are sorted in ascending order
        //which makes all of the following a lot easier
        if(max > 0 && correspond.first > max) break;
        auto [km1_size, k_size, kp1_size] = correspond.second;
        if(km1_size < astart_rows || km1_size > A.shape()[1]) continue;
        if(k_size < astart_cols || k_size > A.shape()[0]) continue;
        if(k_size < bstart_rows || k_size > B.shape()[0]) continue;
        if(kp1_size < bstart_cols || kp1_size > B.shape()[1]) continue;
        // SparseTensor _sub_bk = d_k[{range_(0, km1_size), range_(0, k_size)}];
        // SparseTensor _sub_bk1 = d_kplus1[{nt::range_(0, k_size), nt::range_(0, kp1_size)}];
        // auto [_sub_A, _sub_B] = get<2>(simultaneousReduce(_sub_bk, _sub_bk1));
        // finishRowReducing(_sub_B);
        // int64_t _rank_k = numPivotRows(_sub_A);
        // int64_t _rank_kp1 = numPivotRows(_sub_B);
        // int64_t dimKChains = _sub_bk.shape()[1];
        // int64_t kernelDim = dimKChains - _rank_k;
        // int64_t _betti = kernelDim - _rank_kp1;
        partialColReduce(A_begin, a_access,
                         astart_cols, astart_rows,
                         k_size, km1_size);

        astart_cols = k_size;
        //the following works which is nice
        partialRowReduce(B_begin, b_access,
                        bstart_rows, bstart_cols,
                        k_size, kp1_size);
        bstart_cols = kp1_size;
        // if(i == 0

        int64_t rank_k = numPivotRows(a_access, k_size, km1_size);
        int64_t rank_kp1 = numPivotRows(b_access, k_size, kp1_size);

        //dimKChains = k_size
        //kernelDim = dimKChains - rank_k = k_size - rank_k;
        //betti = kernelDim - rank_kp1
        int64_t betti = (k_size - rank_k) - rank_kp1;
        // std::cout << "ranks: {"<<rank_k<<","<<rank_kp1<<"}, {"<<_rank_k<<","<<_rank_kp1<<"}"<<std::endl;
        // std::cout << correspond.first << ": old betti: "<<_betti<<"new betti: "<<betti<<std::endl;
        if(betti > 0){
            Tensor new_B = B(range> k_size, range> kp1_size);
            Tensor boundary_kp1 = d_kplus1(range> k_size, range> kp1_size).underlying_tensor();
            // if(cntr == 0){
            //     std::cout << boundary_kp1<<std::endl;
            //     ++cntr;
            // }
            Tensor col_space = linalg::col_space(boundary_kp1, new_B).to(DType::Float32);
            col_spaces[correspond.first] = col_space;
            betti_numbers[correspond.first] = betti;
        }else if(add_zeros){
            betti_numbers[correspond.first] = 0;
        }
    }
    MetaFreeArr<float*>(a_access);
    MetaFreeArr<float*>(b_access);
    return std::pair<std::map<double, int64_t>, std::map<double, Tensor>>(betti_numbers, col_spaces);      
}


} // namespace tda
} // namespace nt


/*

11,30: 9
11,2: 2
13,30: 11
13,2: 2
15,30: 12
15,2: 2
19,30: 14
19,5: 5
23,30: 16
23,9: 7
25,30: 18
25,9: 7
32,30: 21
32,13: 11
36,30: 22
36,17: 14
39,30: 24

*/
