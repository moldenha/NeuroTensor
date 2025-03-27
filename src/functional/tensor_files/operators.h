#include "../../Tensor.h"

namespace nt{
namespace functional{
enum class functional_operator_num{
	Multiply = 0,
	Divide = 1,
	Subtract = 2,
	Add = 3
};

Tensor functional_operator_out(const Tensor& a, const Tensor& b, const functional_operator_num op);
void functional_operator_this(Tensor& a, const Tensor& b, const functional_operator_num op);
Tensor hadamard_multiply(const Tensor&, const Tensor&);
Tensor& hadamard_multiply_this(Tensor&, const Tensor&);
Tensor add(const Tensor&, const Tensor&);
Tensor& add_(Tensor&, const Tensor&);
Tensor subtract(const Tensor&, const Tensor&);
Tensor& subtract_(Tensor&, const Tensor&);
Tensor divide(const Tensor&, const Tensor&);
Tensor& divide_(Tensor&, const Tensor&);


}
}
