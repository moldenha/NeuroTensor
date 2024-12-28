#ifndef _NT_LAYERS_LNN_H_
#define _NT_LAYERS_LNN_H_
#include <map>
#include "../Tensor.h"

#include <cstdint>
#include <sys/_types/_int64_t.h>
#include <type_traits>
#include <variant>
#include <functional>
#include <tuple>
#include "Module.h"
#include "functional.h"
#include "TensorGrad.h"
#include "layer_reflect/reflect_macros.h"
#include "layer_reflect/layer_registry.hpp"
#include "layers.h"

namespace nt{
namespace layers{

enum LNNOptions{
	Default,
	HodgkinsHuxley
};


//liquid neural network
class LNN : public Module{
	int64_t _channels, _input_size, _hidden_size, _output_size;
	public:
		Layer ODE;
		Layer W_out;
		LNN(int64_t channels, int64_t input_size, int64_t hidden_size, int64_t output_size, LNNOptions option = LNNOptions::Default);
		TensorGrad forward(const TensorGrad&) override;
};


_NT_REGISTER_LAYER_(LNN, ODE, W_out);

}}



#endif // _NT_LAYERS_LNN_H_ 
