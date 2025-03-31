#include "loss.h"
#include "../../functional/functional.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include <algorithm>
#include <cmath>

namespace nt {
namespace tda {
namespace loss {
TensorGrad filtration_loss(const TensorGrad &output, const Tensor &target,
                           Scalar epsilon) {
    utils::throw_exception(output.shape() == target.shape(),
                           "Expected output and target to have the same shape");
    Tensor max_val = functional::max(output.tensor, target, epsilon);
    Tensor dx = output.tensor - target / max_val;
    Scalar item = dx.sum().toScalar();
    TensorGrad loss(item);
    loss.grad = make_intrusive<tensor_holder>(dx);
	TensorGrad::redefine_tracking(loss, output, [](const Tensor& grad, intrusive_ptr<TensorGrad>& parent){
		parent->grad->tensor = grad;
	});
	return std::move(loss);
}

}
}
}
