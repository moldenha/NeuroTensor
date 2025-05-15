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
#include "../reflection/layer_reflect/reflect_macros.h"
#include "../reflection/layer_reflect/layer_registry.hpp"
#include "layers.h"
#include "ncps/wiring/wiring.h"


namespace nt{
namespace layers{

enum LNNOptions{
	LTC,
	CfC
};


//liquid neural network
class LNN : public Module{
    intrusive_ptr<ncps::Wiring> _wiring;
    Tensor construct_hidden(int64_t batch_size);
    //input_size, output_size, neurons
    static intrusive_ptr<ncps::Wiring> build_wiring(int64_t, int64_t, int64_t);
	public:
        TensorGrad hidden_state;
        Layer lnn;
		LNN(int64_t input_size, int64_t output_size, int64_t neurons, LNNOptions option = LNNOptions::LTC, bool return_sequences = true);
		TensorGrad forward(TensorGrad);
};




}}




#endif // _NT_LAYERS_LNN_H_ 
