#include "lstm_cell.h"

namespace nt {
namespace ncps {

LSTMCell::LSTMCell(int64_t input_size, int64_t hidden_size)
    : input_size(input_size), hidden_size(hidden_size),
      input_map(layers::Linear(input_size, 4 * hidden_size, true)),
      recurrent_map(layers::Linear(hidden_size, 4 * hidden_size, false)) {
    this->init_weights();
}

void LSTMCell::init_weights() {
    for (auto &w : this->input_map.parameters()) {
        if (w.dims() == 1) {
            w.tensor = functional::rand(-0.1, 0.1, w.shape(), w.tensor.dtype);
        } else {
            functional::xavier_uniform_(w.tensor);
        }
    }
    for (auto &w : this->recurrent_map.parameters()) {
        if (w.dims() == 1) {
            w.tensor = functional::rand(-0.1, 0.1, w.shape(), w.tensor.dtype);
        } else {
            w.tensor = functional::rand(0.0, 1.0, w.shape(), w.tensor.dtype);
        }
    }
}

TensorGrad LSTMCell::forward(const TensorGrad &inputs,
                             const TensorGrad &output_state,
                             const TensorGrad &cell_state) {
    TensorGrad z = this->input_map(inputs) + this->recurrent_map(output_state);
    auto [i, ig, fg, og] = get<4>(functional::chunk(z, 4, 1));

    TensorGrad input_activation = functional::tanh(i);
    TensorGrad input_gate = functional::sigmoid(ig);
    TensorGrad forget_gate = functional::sigmoid(fg + 1.0);
    TensorGrad output_gate = functional::sigmoid(og);
    TensorGrad new_cell =
        cell_state * forget_gate + input_activation * input_gate;
    TensorGrad output_cell = functional::tanh(new_cell) * output_gate;
    return functional::list(std::move(output_cell), std::move(new_cell));
}

} // namespace ncps
} // namespace nt
