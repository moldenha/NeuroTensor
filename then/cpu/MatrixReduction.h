#ifndef __NT_TDA_CPU_MATRIX_REDUCTION_H__
#define __NT_TDA_CPU_MATRIX_REDUCTION_H__

//this is a file for specialized faster route for getting the betti numbers from Sparse Matrices
//These calculations are the most intensive and take the longest when calculating persistent homology

#include "../../sparse/SparseMatrix.h"
#include <map>
#include <tuple>

namespace nt{
namespace tda{
namespace cpu{

std::map<double, int64_t> getBettiNumbers(SparseMatrix& d_k, SparseMatrix &d_kplus1,
                                          std::map<double, std::tuple<int64_t, int64_t, int64_t> > radi_bounds,
                                          double max, bool add_zeros);
std::pair<std::map<double, int64_t>, std::vector<std::pair<double, SparseMatrix>> > 
                                    getBettiNumbersColSpace(SparseMatrix& d_k, SparseMatrix &d_kplus1,
                                          std::map<double, std::tuple<int64_t, int64_t, int64_t> > radi_bounds,
                                          double max, bool add_zeros);
}
}
}


#endif
