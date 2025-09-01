#include "combinations.h"
#include "combine.h"
#include "fill.h"

namespace nt{
namespace functional{

int64_t num_combinations(int64_t n, int64_t r) {
    if (r > n) return 0;
    int64_t result = 1;
    for (int64_t i = 0; i < r; ++i) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

nt::Tensor combinations(nt::Tensor vec, int64_t r, int64_t start){
    //takes a vector
    //returns combinations
    //similar to pythons itertools.combinations
    nt::utils::throw_exception(vec.dims() == 1, "Expected to get a vector of dimensions 1 but got dimensionality of $", vec.dims());
    const int64_t n = vec.shape()[0];
    nt::Tensor myints = nt::functional::arange(r, nt::DType::int64, start);
    // nt::Tensor out = nt::Tensor::makeNullTensorArray(num_combinations(n, r));
    nt::Tensor out({num_combinations(n, r)}, DType::TensorObj);
    nt::Tensor* begin = reinterpret_cast<nt::Tensor*>(out.data_ptr());
    Tensor* end = begin + out.numel();
    for(;begin != end; ++begin){
        *begin = Tensor(nullptr);
    }
    begin = reinterpret_cast<nt::Tensor*>(out.data_ptr());
    *begin = vec[myints];
    ++begin;
    int64_t* first = reinterpret_cast<int64_t*>(myints.data_ptr());
    int64_t* last = reinterpret_cast<int64_t*>(myints.data_ptr_end());
    while((*first) != n-r+start){
        int64_t* mt = last;
        --mt; // Ensure mt is decremented before use
        while (*mt == n - int64_t(last - mt) + start) {
            --mt;
        }
        (*mt)++;
        while (++mt != last) *mt = *(mt-1)+1;
        *begin = vec[myints];
        ++begin;
    }
    return nt::functional::stack(out).clone();

}



}
}
