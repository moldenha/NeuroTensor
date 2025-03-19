#include "MatrixReduction.h"
#include "../functional/functional.h"

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

    const float *a_access[A_split.numel()];
    const float *b_access[A_split.numel()];
    for (int64_t i = 0; i < A_split.numel(); ++i) {
        a_access[i] = reinterpret_cast<const float *>(A_begin[i].data_ptr());
        b_access[i] = reinterpret_cast<const float *>(B_begin[i].data_ptr());
    }
    while (i < num_rows && j < num_cols) {
        // numpy version is i, j [without transpose]
        //  bool do_continue = false;
        //  bool was_zero = false;
        //  int64_t nonzeroCol = 0;
        if (a_access[j][i] == 0) {
            int64_t nonzeroCol = j;
            while (nonzeroCol < num_cols && a_access[nonzeroCol][i] == 0) {
                ++nonzeroCol;
            }
            if (nonzeroCol == num_cols) {
                ++i;
                continue;
            }
            // swap rows
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

    return functional::list(functional::cat_unordered(A_split).view(A.shape()),
                            functional::cat_unordered(B_split).view(B.shape()));
}

Tensor &finishRowReducing(Tensor &B) {
    int64_t numRows = B.shape()[0];
    int64_t numCols = B.shape()[1];
    Tensor B_split = B.split_axis(-2);
    Tensor *B_begin = reinterpret_cast<Tensor *>(B_split.data_ptr());
    const float *b_access[B_split.numel()];
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
            scaleAmt *= -1;
            functional::fused_multiply_add_(B_begin[otherRow], B_begin[i],
                                            scaleAmt);
        }

        ++i;
        ++j;
    }
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
    const float *a_access[A_split.numel()];
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

    return std::move(A);
}

} // namespace tda
} // namespace nt
