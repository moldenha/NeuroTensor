#include "../Tensor.h"

namespace nt{
namespace functional{
enum class functional_operator_num{
	Multiply,
	Divide,
	Subtract,
	Add
};

Tensor functional_operator_out(const Tensor& a, const Tensor& b, const functional_operator_num op);

void functional_operator_this(Tensor& a, const Tensor& b, const functional_operator_num op);

}
}
