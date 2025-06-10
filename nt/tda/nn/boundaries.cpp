#include "boundaries.h"
#include "../Boundaries.h"
#include "../../nn/functional.h"
#include "../../functional/functional.h"
#include "../../functional/tensor_files/mesh.h"
#include "../../utils/macros.h"

namespace nt{
namespace tda{


void linear_sum_assignment(const nt::Tensor& _cost_matrix,
                           std::vector<int64_t>& row_assignment,
                           std::vector<int64_t>& col_assignment,
                           float& total_cost) {
    
    const int64_t& n_rows = _cost_matrix.shape()[0];
    const int64_t& n_cols = _cost_matrix.shape()[1];
    const float* cost_matrix_ptr = reinterpret_cast<const float*>(_cost_matrix.data_ptr());
    NT_VLA(const float*, cost_matrix, n_rows);
    // const float* cost_matrix[n_rows];
    for(int64_t i = 0; i < n_rows; ++i){
        cost_matrix[i] = &cost_matrix_ptr[i * n_cols];
    }

    int64_t dim = std::max(n_rows, n_cols);
    
    // Pad cost matrix to square
    nt::Tensor _cost = (n_rows != n_cols) ? nt::functional::zeros({dim, dim}, nt::DType::Float32) : _cost_matrix;
    float* cost_ptr = reinterpret_cast<float*>(_cost.data_ptr());
    NT_VLA(float*, cost, dim);
    // float* cost[dim];
    for(int64_t i = 0; i < dim; ++i){
        cost[i] = &cost_ptr[i * dim];
    }
    
    if(n_rows != n_cols){
        for (int i = 0; i < n_rows; ++i){
            for (int j = 0; j < n_cols; ++j)
                cost[i][j] = cost_matrix[i][j];
        }
    }
   
    // row_assignment.resize(n_rows);
    row_assignment.assign(n_rows, -1);
    // col_assignment.resize(n_cols);
    col_assignment.assign(n_cols, -1);
    total_cost = 0.0;

    std::vector<float> row_min(dim, std::numeric_limits<float>::infinity());
    std::vector<float> col_min(dim, std::numeric_limits<float>::infinity());
    std::vector<int64_t> row_match(dim, -1);
    std::vector<int64_t> col_match(dim, -1);
    std::vector<float> slack(dim);
    std::vector<int64_t> slack_col(dim, -1);
    std::vector<bool> row_covered(dim, false);
    std::vector<bool> col_covered(dim, false);
    std::vector<int64_t> parent_row(dim, -1);
    std::vector<int64_t> path(dim, -1);
    // 1. Reduce the cost matrix
    for (int64_t i = 0; i < dim; ++i) {
        for (int64_t j = 0; j < dim; ++j) {
            row_min[i] = std::min(row_min[i], cost[i][j]);
        }
    }
    for (int64_t i = 0; i < dim; ++i) {
        for (int64_t j = 0; j < dim; ++j) {
            cost[i][j] -= row_min[i];
        }
    }
    for (int64_t j = 0; j < dim; ++j) {
            for (int64_t i = 0; i < dim; ++i) {
        col_min[j] = std::min(col_min[j], cost[i][j]);
        }
    }
    for (int64_t i = 0; i < dim; ++i) {
        for (int64_t j = 0; j < dim; ++j) {
            cost[i][j] -= col_min[j];
        }
    }
    // 2. Initial matching (greedy)
    int64_t matches = 0;
    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < n_cols; ++j) {
            if (cost[i][j] == 0 && col_match[j] == -1) {
                row_match[i] = j;
                col_match[j] = i;
                matches++;
                break;
            }
        }
    }
    
      // 3. Main loop
    while (matches < std::min(n_rows, n_cols)) {
    int64_t free_row = -1;
    for (int64_t i = 0; i < n_rows; ++i) {
      if (row_match[i] == -1) {
        free_row = i;
        break;
      }
    }

    if (free_row == -1) {
      break; // Should not happen if matches < min(n_rows, n_cols)
    }

    std::fill(row_covered.begin(), row_covered.end(), false);
    std::fill(col_covered.begin(), col_covered.end(), false);
    std::fill(parent_row.begin(), parent_row.end(), -1);
    std::fill(slack.begin(), slack.end(), std::numeric_limits<float>::infinity());
    std::fill(slack_col.begin(), slack_col.end(), -1);

    int64_t current_row = free_row;
    int64_t current_col = -1;
    bool path_found = false;

        while (!path_found) {
      row_covered[current_row] = true;

      for (int64_t j = 0; j < dim; ++j) {
        if (!col_covered[j]) {
          float new_slack = cost[current_row][j];
          if (new_slack < slack[j]) {
            slack[j] = new_slack;
            slack_col[j] = current_row;
          }
        }
      }

      float min_slack = std::numeric_limits<float>::infinity();
      int64_t next_col = -1;
      for (int64_t j = 0; j < dim; ++j) {
        if (!col_covered[j] && slack[j] < min_slack) {
          min_slack = slack[j];
          next_col = j;
        }
      }

      if (std::isinf(min_slack)) {
        break; // No augmenting path found in this iteration
      }

      col_covered[next_col] = true;
      current_col = next_col;
      int64_t matched_row = col_match[current_col];

      if (matched_row == -1) {
        path_found = true;
      } else {
        current_row = matched_row;
      }
    }

    if (path_found) {
      int64_t j = current_col;
      while (j != -1) {
        int64_t prev_row = slack_col[j];
        int64_t next_j = row_match[prev_row];
        row_match[prev_row] = j;
        col_match[j] = prev_row;
        j = next_j;
      }
      matches++;
    } else {
      // Increase minimum uncovered cost and update cost matrix
      float min_val = std::numeric_limits<float>::infinity();
      for (int64_t i = 0; i < dim; ++i) {
        if (row_covered[i]) {
          for (int64_t j = 0; j < dim; ++j) {
            if (!col_covered[j]) {
              min_val = std::min(min_val, cost[i][j]);
            }
          }
        }
      }

      if (std::isinf(min_val)) {
        break; // Should not happen in a valid cost matrix
      }

      for (int64_t i = 0; i < dim; ++i) {
        if (row_covered[i]) {
          for (int64_t j = 0; j < dim; ++j) {
            cost[i][j] -= min_val;
          }
        }
        if (!col_covered[i]) {
          for (int64_t j = 0; j < dim; ++j) {
            cost[j][i] += min_val;
          }
        }
      }
    }
  }

  // Extract assignments and total cost
  for (int64_t i = 0; i < n_rows; ++i) {
    if (row_match[i] < n_cols) {
      row_assignment[i] = row_match[i];
      col_assignment[row_match[i]] = i;
      total_cost += cost_matrix[i][row_match[i]];
    } else {
      row_assignment[i] = -1; // Not assigned to a real column
    }
  }
  for (int64_t j = 0; j < n_cols; ++j) {
    if (col_assignment[j] == -1) {
      // Not assigned to a real row (if n_cols > n_rows)
    }
  }

    
    NT_VLA_DEALC(cost);
    NT_VLA_DEALC(cost_matrix);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t> >
    findBestRowColPermutations(nt::Tensor& A, const nt::Tensor& B, float& error){
    const int64_t& m = A.shape()[0];
    const int64_t& n = A.shape()[1];
    nt::utils::throw_exception(A.shape() == B.shape(), "Expected boundary error to match shape of boundary");
    nt::Tensor A_split = A.split_axis(-2);
    nt::Tensor B_split = B.split_axis(-2);

    nt::Tensor *A_begin = reinterpret_cast<nt::Tensor *>(A_split.data_ptr());
    nt::Tensor *B_begin = reinterpret_cast<nt::Tensor *>(B_split.data_ptr());
    nt::utils::throw_exception(A_split.numel() == B_split.numel(),
                           "Expected nummels to be same but got $ and $",
                           A_split.numel(), B_split.numel());
    
    NT_VLA(const float*, a_access, A_split.numel());
    NT_VLA(const float*, b_access, A_split.numel());

    // const float *a_access[A_split.numel()];
    // const float *b_access[A_split.numel()];
    for (int64_t i = 0; i < A_split.numel(); ++i) {
        a_access[i] = reinterpret_cast<const float *>(A_begin[i].data_ptr());
        b_access[i] = reinterpret_cast<const float *>(B_begin[i].data_ptr());
    }

    std::vector<int64_t> col_perm, row_perm;

    {
        //finding row cost matrix
        nt::Tensor cost_matrix = nt::functional::zeros({m, m}, nt::DType::Float32);
        float* cost_matrix_out = reinterpret_cast<float*>(cost_matrix.data_ptr());
        for(int64_t i = 0; i < m; ++i){
            for(int64_t j = 0; j < m; ++j){
                float sum = 0;
                for(int64_t k = 0; k < n; ++k){
                    sum += std::pow((a_access[i][k] - b_access[j][k]), 2);
                }
                cost_matrix_out[i * m + j] = sum;
            }
        }
        std::vector<int64_t> row_assignment, col_assignment;
        linear_sum_assignment(cost_matrix, row_assignment, col_assignment, error);
        std::vector<size_t> indices(m);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&row_assignment](const size_t& a, const size_t& b){return row_assignment[a] < row_assignment[b];});
        row_perm.resize(m);
        for(int64_t i = 0; i < m; ++i){
            row_perm[i] = col_assignment[indices[i]];
        }
    }
    //swapping rows:
    for(int64_t i = 0; i < m; ++i){
        a_access[i] = reinterpret_cast<const float *>(A_begin[row_perm[i]].data_ptr());
    }


    {
        //finding col cost matrix
        nt::Tensor cost_matrix = nt::functional::zeros({n, n}, nt::DType::Float32);
        float* cost_matrix_out = reinterpret_cast<float*>(cost_matrix.data_ptr());
        for(int64_t i = 0; i < n; ++i){
            for(int64_t j = 0; j < n; ++j){
                float sum = 0;
                for(int64_t k = 0; k < m; ++k){
                    sum += std::pow((a_access[k][i] - b_access[k][j]), 2);
                }
                cost_matrix_out[i * n + j] = sum;
            }
        }
        std::vector<int64_t> row_assignment, col_assignment;
        linear_sum_assignment(cost_matrix, row_assignment, col_assignment, error);
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&row_assignment](const size_t& a, const size_t& b){return row_assignment[a] < row_assignment[b];});
        col_perm.resize(n);
        for(int64_t i = 0; i < n; ++i){
            col_perm[i] = col_assignment[indices[i]];
        }
    }
    return std::make_tuple(row_perm, col_perm);

    NT_VLA_DEALC(a_access);
    NT_VLA_DEALC(b_access);
}


std::tuple<std::vector<int64_t>, std::vector<int64_t> >
    findBestRowColPermutations(const nt::Tensor& cost, float& error){
    nt::Tensor row_cost = functional::matmult(cost, cost, false, true);

    const int64_t& m = cost.shape()[0];
    const int64_t& n = cost.shape()[1];
    std::vector<int64_t> col_perm, row_perm;

    {
        std::vector<int64_t> row_assignment, col_assignment;
        linear_sum_assignment(row_cost, row_assignment, col_assignment, error);
        std::vector<size_t> indices(m);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&row_assignment](const size_t& a, const size_t& b){return row_assignment[a] < row_assignment[b];});
        row_perm.resize(m);
        for(int64_t i = 0; i < m; ++i){
            row_perm[i] = col_assignment[indices[i]];
        }
    }

    nt::Tensor col_cost = functional::matmult(cost, cost, true, false);


    {
        std::vector<int64_t> row_assignment, col_assignment;
        linear_sum_assignment(col_cost, row_assignment, col_assignment, error);
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&row_assignment](const size_t& a, const size_t& b){return row_assignment[a] < row_assignment[b];});
        col_perm.resize(n);
        for(int64_t i = 0; i < n; ++i){
            col_perm[i] = col_assignment[indices[i]];
        }
    }
    return std::make_tuple(row_perm, col_perm);

    
}


Tensor permute_cols_float(const Tensor& mult, const std::vector<int64_t>& col_perm){
    utils::throw_exception(mult.dtype == DType::Float32, "Expected to permute float matrix columns");
    Tensor out(mult.shape(), mult.dtype);
    float* out_ptr = reinterpret_cast<float*>(out.data_ptr());
    const float* mult_ptr = reinterpret_cast<const float*>(mult.data_ptr());
    const int64_t& rows = mult.shape()[0];
    const int64_t& cols = mult.shape()[1];
    for(int64_t r = 0; r < rows; ++r, mult_ptr += cols){
        for(int64_t c = 0; c < cols; ++c, ++out_ptr){
            *out_ptr = mult_ptr[col_perm[c]];
        }
    }
    return std::move(out);
}


//row and col permutations
//the absolute value of grad should be entered
std::tuple<std::vector<int64_t>, std::vector<int64_t>>
    findPermutations(Tensor grad, const Tensor& mult){
    std::vector<int64_t> row_perms, col_perms;
    const int64_t& n = mult.shape()[1];

    {
        //finding col cost matrix
        nt::Tensor cost_matrix = functional::matmult(mult, grad, true);
        float error;
        std::vector<int64_t> row_assignment, col_assignment;
        linear_sum_assignment(cost_matrix, row_assignment, col_assignment, error);
        col_perms = col_assignment;
    }
    {
        //finding row cost matrix, but first. transposing
        Tensor A_col_swapped = permute_cols_float(mult,  col_perms);
        Tensor cost_matrix = functional::matmult(A_col_swapped, grad, false, true);
        float error;
        std::vector<int64_t> row_assignment, col_assignment;
        linear_sum_assignment(cost_matrix, row_assignment, col_assignment, error);
        row_perms = col_assignment;
    }
    return std::make_tuple(row_perms, col_perms);
}



Tensor Sinkhorn(const Tensor& cost, float tau = 0.1){
    Tensor log_p = -cost / tau;
    //could itter the following
    
    //row normalization (log-softmax accross rows)
    log_p = log_p - functional::logsumexp(log_p, -1, true);
    //column normalization
    log_p = log_p - functional::logsumexp(log_p, -2, true);

    //end itter
    return functional::exp(log_p); // back to normal space
}

TensorGrad Sinkhorn(const TensorGrad& cost, float tau = 0.01, int n_iters = 20) {
    TensorGrad log_p = -cost / tau;

    for (int i = 0; i < n_iters; ++i) {
        log_p -= functional::logsumexp(log_p, -1, true); // row norm
        log_p -= functional::logsumexp(log_p, -2, true); // col norm
    }

    
    log_p.exp_();
    return log_p;
    // TensorGrad out(log_p.tensor);
    // TensorGrad::redefine_tracking(out, log_p,
    //             [](const Tensor& grad, intrusive_ptr<TensorGrad> parent){
    //                 std::cout << "gradient for sink horn is "<<grad<<std::endl;
    //                 parent->grad->tensor += grad;
    //             });
    // return out;
}

Tensor pairwise_l2(const Tensor& x, const Tensor& y){
    //x: (m), y: (n)
    return (x.view(-1, 1) - y.view(1, -1)).pow(2);
}

inline TensorGrad pairwise_l2(const TensorGrad& x, const TensorGrad& y){
    return (x.view(-1, 1) - y.view(1, -1)).pow(2);
}

Tensor sample_gumbel(const TensorGrad& logits){
    Tensor u = functional::rand(0, 1, logits.shape(), logits.dtype); //uniform (0,1)
    return -functional::log(-functional::log(u + 1e-10));       // Gumbel(0,1)
}


TensorGrad gumbel_softmax(const TensorGrad& logits, float tau, bool hard = false) {
    Tensor gumbel_noise = sample_gumbel(logits);
    // std::cout << "gumble noise is "<<nt::noprintdtype<<gumbel_noise<<nt::printdtype<<std::endl;
    // std::cout << "logits are: "<<logits<<std::endl;
    TensorGrad y = ((logits * 100) + gumbel_noise) / tau;
    // std::cout << "y is "<<nt::noprintdtype<<y<<nt::printdtype<<std::endl;
    y = functional::softmax(y, -1); // apply softmax along last dim
    // std::cout << "y is "<<nt::noprintdtype<<y<<nt::printdtype<<std::endl;

    if (hard) {
        // Straight-through: make y_hard one-hot
        Tensor y_hard = functional::one_hot(functional::argmax(y.tensor, -1), y.shape()[-1]).to(y.dtype);
        // Use straight-through estimator
        return (y_hard - y).tensor + y;
    }

    return y;
}


TensorGrad BoundaryMatrix(Tensor simplex_complex_kp1, Tensor simplex_complex_k,
                          TensorGrad radi_kp1, TensorGrad radi_k){
    
    auto [x_indexes, y_indexes, boundaries] = 
                compute_differentiable_boundary_sparse_matrix_index(simplex_complex_kp1, simplex_complex_k,
                                                                    radi_kp1.tensor, radi_k.tensor);

    
    //this constructs a regular boundary matrix
    //this doesn't necessarily change per simplex
    //[autograd not tracked] 
    Tensor boundary_no_grad = functional::zeros({radi_k.shape()[0], radi_kp1.shape()[0]}, DType::Float32);
    const int64_t& cols = boundary_no_grad.shape()[-1];
    float* access = reinterpret_cast<float*>(boundary_no_grad.data_ptr());
    for(size_t i = 0; i < x_indexes.size(); ++i){
        access[x_indexes[i] * cols + y_indexes[i]] = (boundaries[i] < 1 ? -1 : 1.0);
    }

    //[boundary radi does have autograd tracked]
    TensorGrad boundary_radi = radi_k.view(-1, 1) * radi_kp1;
    boundary_radi.tensor *= boundary_no_grad;
    
    //computing soft cost for rows and columns [autograd tracked] 
    TensorGrad logits_r = pairwise_l2(radi_k, radi_k);
    TensorGrad logits_c = pairwise_l2(radi_kp1, radi_kp1);


    //computing soft permutations accross the matrices  [autograd tracked]
    TensorGrad P_row = gumbel_softmax(logits_r, 1.0, true);
    TensorGrad P_col = gumbel_softmax(logits_c, 1.0, true);

    //applying soft differentiable permutations
    //the gradients on the permutations are tracked
    //the construction of the original boundary matrix is a discrete function
    //the original boundary matrix itself does not have its gradient tracked
    
    TensorGrad boundary = functional::matmult(functional::matmult(P_row, boundary_radi), P_col);
    return std::move(boundary);
}

Tensor sample_gumbel_sigmoid(const TensorGrad& logits) {
    Tensor u = functional::rand(0, 1, logits.shape(), logits.dtype); // Uniform(0,1)
    return -functional::log(-functional::log(u + 1e-10));
}

TensorGrad gumbel_sigmoid(const TensorGrad& logits, float tau = 1.0, bool hard = false) {
    Tensor gumbel_noise = sample_gumbel_sigmoid(logits);
    TensorGrad y = (logits * 100 + gumbel_noise) / tau;
    y = functional::sigmoid(y);

    if (hard) {
        Tensor y_hard = (y.tensor > 0.5).to(y.dtype);  // binary threshold
        return (y_hard - y).tensor + y;  // Straight-through estimator
    }

    return y;
}


TensorGrad soft_row_col_eliminate(TensorGrad& boundary_permuted, TensorGrad& radi_kp1, TensorGrad& radi_k){
    //learn soft row and column masks using gumbel sigmoid
    TensorGrad row_mask = functional::sigmoid(radi_k);
    TensorGrad col_mask = functional::sigmoid(radi_kp1);
    TensorGrad boundary_final = boundary_permuted * row_mask * col_mask;
    return std::move(boundary_final);
    
}

TensorGrad hard_row_col_eliminate(TensorGrad& boundary_permuted, TensorGrad& radi_kp1, TensorGrad& radi_k){
    //learn hard row and column masks using gumbel sigmoid
    TensorGrad row_mask = gumbel_sigmoid(radi_k, 1.0, true).view(-1, 1);   // shape (rows, 1)
    TensorGrad col_mask = gumbel_sigmoid(radi_kp1, 1.0, true).view(1, -1); // shape (1, cols)
    TensorGrad mask = row_mask * col_mask;
    // std::cout << nt::noprintdtype;
    // std::cout << "col mask: "<< col_mask.view(-1) << std::endl;
    // std::cout << "mask: "<<mask<<std::endl;
    // std::cout << nt::printdtype;
    TensorGrad boundary_final = boundary_permuted * mask;
    return std::move(boundary_final);
}




//this is when the rows and columns of the boundary matrix are learnable
TensorGrad _nt_boundary_learnable_rows_matrix_(TensorGrad& boundary_radi, TensorGrad& radi_k, TensorGrad& radi_kp1, TensorGrad& radi_k_2, TensorGrad& radi_kp1_2){
    // std::cout << "
    //computing soft cost for rows and columns [autograd tracked] 
    TensorGrad logits_r = pairwise_l2(radi_k, radi_k);
    TensorGrad logits_c = pairwise_l2(radi_kp1, radi_kp1);

    //computing soft permutations accross the matrices  [autograd tracked]
    TensorGrad P_row = gumbel_softmax(logits_r, 1.0, true);
    TensorGrad P_col = gumbel_softmax(logits_c, 1.0, true);
    
    //applying soft differentiable permutations
    //the gradients on the permutations are tracked
    //the construction of the original boundary matrix is a discrete function
    //the original boundary matrix itself does not have its gradient tracked
    
    TensorGrad boundary_permuted = functional::matmult(functional::matmult(P_row, boundary_radi), P_col);
    // std::cout << "boundary permuted: "<<boundary_permuted<<std::endl;

    //now eliminate rows and columns from boundary permuted:
    //theres I can either use the soft or hard elimination
    //the hard permutation worked better than the soft one
    //for that reason, I am thinking I use that one for the elimination as well
    return hard_row_col_eliminate(boundary_permuted, radi_kp1_2, radi_k_2);
}

TensorGrad _nt_boundary_learnable_cols_matrix_(TensorGrad& boundary_radi, TensorGrad& radi_kp1, TensorGrad& radi_kp1_2){
    std::cout << "boundary: "<<boundary_radi<<std::endl;
    TensorGrad logits_c = pairwise_l2(radi_kp1, radi_kp1);
    TensorGrad P_col = gumbel_softmax(logits_c, 1.0, true);
    TensorGrad boundary_permuted = functional::matmult(boundary_radi, P_col);
    
    TensorGrad col_mask = gumbel_sigmoid(radi_kp1_2, 1.0, true).view(1, -1); // shape (1, cols)
    
    TensorGrad mask = nt::functional::ones({boundary_permuted.shape()[0], 1}, boundary_permuted.dtype) * col_mask;
    TensorGrad boundary_final = boundary_permuted * mask;
    return std::move(boundary_final);

}



TensorGrad BoundaryMatrix(Tensor simplex_complex_kp1, Tensor simplex_complex_k,
                          TensorGrad radi_kp1, TensorGrad radi_k, 
                          TensorGrad radi_kp1_2, TensorGrad radi_k_2){

    auto [x_indexes, y_indexes, boundaries] = 
                compute_differentiable_boundary_sparse_matrix_index(simplex_complex_kp1, simplex_complex_k,
                                                                    radi_kp1.tensor, radi_k.tensor);
    bool rows_learnable = (simplex_complex_k.shape()[-1] > 1);
    // std::cout << simplex_complex_k.shape() << std::endl;
    // this constructs a regular boundary matrix
    //this doesn't necessarily change per simplex
    //[autograd not tracked] 
    Tensor boundary_no_grad = functional::zeros({radi_k.shape()[0], radi_kp1.shape()[0]}, radi_k.dtype);
    const int64_t& cols = boundary_no_grad.shape()[-1];
    float* access = reinterpret_cast<float*>(boundary_no_grad.data_ptr());
    for(size_t i = 0; i < x_indexes.size(); ++i){
        access[x_indexes[i] * cols + y_indexes[i]] = (boundaries[i] < 1 ? -1 : 1.0);
    }

    //[boundary radi does have autograd tracked]
    // TensorGrad boundary_radi = radi_k.view(-1, 1) * radi_kp1;
    TensorGrad boundary_radi(nt::functional::ones({radi_k.shape()[0], radi_kp1.shape()[0]}, radi_k.dtype));
    boundary_radi.tensor *= boundary_no_grad;
    // std::cout << "boundary: "<<boundary_radi<<std::endl;
    if(rows_learnable){
        return _nt_boundary_learnable_rows_matrix_(boundary_radi, radi_k, radi_kp1, radi_k_2, radi_kp1_2);
    }else{
        return _nt_boundary_learnable_cols_matrix_(boundary_radi, radi_kp1, radi_kp1_2);
    }

}

}
}

/*

    // intrusive_ptr<tensor_holder> rk = make_intrusive<tensor_holder>(radi_k.tensor.clone());
    // intrusive_ptr<tensor_holder> rkp1 = make_intrusive<tensor_holder>(radi_kp1.tensor.clone());
    
    // TensorGrad sig_radi_k = (apply_sigmoid ? functional::sigmoid(radi_k * alpha) : radi_k);
    // TensorGrad sig_radi_kp1 = (apply_sigmoid ? functional::sigmoid(radi_kp1 * alpha) : radi_kp1);
    // TensorGrad boundary_radi = sig_radi_k.view(-1, 1) * sig_radi_kp1;
    // std::cout << "made initial boundary"<<std::endl;


this is the old gradient calculation:
         // Tensor B = xc->tensor - grad;
            // Tensor B = grad;
            // std::cout << "boundary gradient: "<<grad<<std::endl;
            // std::cout << "tensor B: "<<B<<std::endl;
            // std::cout << "tensor A: "<<xc->tensor<<std::endl;
            // std::cout << "gradient: "<<std::endl;
            // std::cout << grad << std::endl;
            // Tensor B = grad.clone();
            // B[B != 0] = 1;
            // Tensor n_mult = xc->tensor.clone();
            // n_mult[n_mult != 0] = 1;
            // std::cout <<"subtract: "<< B - n_mult << std::endl;
            // std::cout << functional::all(grad == 0, -2) << std::endl;
            // std::cout << functional::all(grad == 0, -1) << std::endl;
            // float error;
            std::tuple<std::vector<int64_t>, std::vector<int64_t> > assignments = findPermutations(xc->tensor, grad);
            std::vector<int64_t> row_assignments = std::get<0>(assignments), col_assignments = std::get<1>(assignments);
            utils::THROW_EXCEPTION(row_assignments.size() == rk->tensor.numel(),
                                   "ERROR WITH ROW ASSIGNMENT SIZE");
            utils::THROW_EXCEPTION(col_assignments.size() == rkp1->tensor.numel(),
                                   "ERROR WITH COL ASSIGNMENT SIZE");
            Tensor rk_grad = functional::zeros({rk->tensor.numel()}, DType::Float32);
            float* rkg_begin = reinterpret_cast<float*>(rk_grad.data_ptr());
            float* rk_begin = reinterpret_cast<float*>(rk->tensor.data_ptr());
            for(size_t i = 0; i < row_assignments.size(); ++i){
                if(row_assignments[i] == i) continue;
                // std::cout << "swapping "<<row_assignments[i] << " and "<<i<<" rows"<<std::endl;
                rkg_begin[i] += (rk_begin[row_assignments[i]] - rk_begin[i]) / 2;
            }
            Tensor rkp1_grad = functional::zeros({rkp1->tensor.numel()}, DType::Float32);
            float* rkp1g_begin = reinterpret_cast<float*>(rkp1_grad.data_ptr());
            float* rkp1_begin = reinterpret_cast<float*>(rkp1->tensor.data_ptr());
            for(size_t i = 0; i < col_assignments.size(); ++i){
                if(col_assignments[i] == i) continue;
                // std::cout << "swapping "<<col_assignments[i] << " and "<<i<<" columns"<<std::endl;
                rkp1g_begin[i] += (rkp1_begin[col_assignments[i]] - rkp1_begin[i]) / 2;
            }
            //maybe something for if an entire row or column of B is zeros
            
            parents[1]->grad->tensor += rk_grad;
            parents[2]->grad->tensor += rkp1_grad;
            parents[0]->grad->tensor += grad;



    TensorGrad out_tensor = TensorGrad::make_tensor_grad(
            mult,
        [xc, rk, rkp1](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad> >& parents){
            // Tensor B = xc->tensor - grad;
            // Tensor B = grad;
            // std::cout << "boundary gradient: "<<grad<<std::endl;
            // std::cout << "tensor B: "<<B<<std::endl;
            // std::cout << "tensor A: "<<xc->tensor<<std::endl;
            // std::cout << "gradient: "<<std::endl;
            // std::cout << grad << std::endl;
            // Tensor B = grad.clone();
            // B[B != 0] = 1;
            // Tensor n_mult = xc->tensor.clone();
            // n_mult[n_mult != 0] = 1;
            // std::cout <<"subtract: "<< B - n_mult << std::endl;
            // std::cout << functional::all(grad == 0, -2) << std::endl;
            // std::cout << functional::all(grad == 0, -1) << std::endl;
            // float error;
            std::tuple<std::vector<int64_t>, std::vector<int64_t> > assignments = findPermutations(xc->tensor, grad);
            std::vector<int64_t> row_assignments = std::get<0>(assignments), col_assignments = std::get<1>(assignments);
            utils::THROW_EXCEPTION(row_assignments.size() == rk->tensor.numel(),
                                   "ERROR WITH ROW ASSIGNMENT SIZE");
            utils::THROW_EXCEPTION(col_assignments.size() == rkp1->tensor.numel(),
                                   "ERROR WITH COL ASSIGNMENT SIZE");
            Tensor rk_grad = functional::zeros({rk->tensor.numel()}, DType::Float32);
            float* rkg_begin = reinterpret_cast<float*>(rk_grad.data_ptr());
            float* rk_begin = reinterpret_cast<float*>(rk->tensor.data_ptr());
            for(size_t i = 0; i < row_assignments.size(); ++i){
                if(row_assignments[i] == i) continue;
                // std::cout << "swapping "<<row_assignments[i] << " and "<<i<<" rows"<<std::endl;
                rkg_begin[i] += (rk_begin[row_assignments[i]] - rk_begin[i]) / 2;
            }
            Tensor rkp1_grad = functional::zeros({rkp1->tensor.numel()}, DType::Float32);
            float* rkp1g_begin = reinterpret_cast<float*>(rkp1_grad.data_ptr());
            float* rkp1_begin = reinterpret_cast<float*>(rkp1->tensor.data_ptr());
            for(size_t i = 0; i < col_assignments.size(); ++i){
                if(col_assignments[i] == i) continue;
                // std::cout << "swapping "<<col_assignments[i] << " and "<<i<<" columns"<<std::endl;
                rkp1g_begin[i] += (rkp1_begin[col_assignments[i]] - rkp1_begin[i]) / 2;
            }
            //maybe something for if an entire row or column of B is zeros
            
            parents[1]->grad->tensor += rk_grad;
            parents[2]->grad->tensor += rkp1_grad;
            parents[0]->grad->tensor += grad;

        }, boundary_radi, radi_k, radi_kp1);
    return out_tensor;
*/
