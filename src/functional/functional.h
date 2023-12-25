#ifndef _NT_FUNCTIONAL_H_
#define _NT_FUNCTIONAL_H_
#include "../Tensor.h"
#include <_types/_uint32_t.h>
#include <sys/_types/_int32_t.h>
#include <vector>

namespace nt{
namespace functional{

Tensor zeros(SizeRef, DType dt=DType::Float);
Tensor nums(SizeRef, const float&, DType dt = DType::Float);
Tensor randint(int32_t, int32_t, SizeRef, DType dt = DType::int32);
Tensor randn(SizeRef, DType dt = DType::Float);
Tensor cat(const Tensor& _a, const Tensor &_b, int8_t dim=0);
Tensor matmult(const Tensor&, const Tensor&, bool untranspose=true);
Tensor hadamard_multiply(const Tensor&, const Tensor&);
Tensor& hadamard_multiply_this(Tensor&, const Tensor&);
Tensor add(const Tensor&, const Tensor&);
Tensor& add_(Tensor&, const Tensor&);
Tensor subtract(const Tensor&, const Tensor&);
Tensor& subtract_(Tensor&, const Tensor&);
Tensor divide(const Tensor&, const Tensor&);
Tensor& divide_(Tensor&, const Tensor&);

Tensor arange(uint32_t total_size, DType dt = DType::Float);
Tensor arange(SizeRef, DType dt = DType::Float);
bool all(const Tensor&);
void save(const Tensor&, const char*);
Tensor load(const char*);
Tensor conv2d(const Tensor& image, const Tensor& kernel, const uint32_t s_h=1, const uint32_t s_w=1);
Tensor conv2dT(const Tensor& image, const Tensor& kernel, const uint32_t s_h=1, const uint32_t s_w=1);
void softmax_(Tensor&);
void softmax_(Tensor&, uint32_t);
void softmax_stable_(Tensor&);
void softmax_stable_(Tensor&, uint32_t);
Tensor softmax(Tensor&);
Tensor softmax(Tensor&, uint32_t);
Tensor softmax_stable(Tensor&);
Tensor softmax_stable(Tensor&, uint32_t);
Tensor cat(std::vector<Tensor>);
Tensor cat(std::vector<Tensor>, int32_t dim);
Tensor stack(std::vector<Tensor>);
Tensor stack(std::vector<Tensor>, int8_t dim);

}
}

#endif
