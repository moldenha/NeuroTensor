#ifndef _NT_MODULE_H_
#define _NT_MODULE_H_
namespace nt {
class Module; // forward declaration
}

#include "../Tensor.h"
#include "../utils/any_ref.h"
#include "TensorGrad.h"
#include "forward_Layer.h"
#include "layer_reflect/custom_iterator.hpp"
#include "layer_reflect/custom_iterator_map.hpp"
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

namespace nt {

class Module : public intrusive_ptr_target {
    std::map<std::string, std::reference_wrapper<TensorGrad>>
        _mapped_grad_names_wrapped;
    std::map<std::string, std::reference_wrapper<Layer>>
        _mapped_layer_names_wrapped;
    void _run_forward(std::vector<utils::any_ref>);

  protected:
    void register_parameter(std::string name, TensorGrad &tg);
    void register_module(std::string name, Layer &l);

  public:
    Module() = default;
    virtual ~Module() = default;
    inline virtual void backward(const Tensor &dx,
                                 intrusive_ptr<TensorGrad> parent) {
        ;
    }
    std::map<std::string, std::reference_wrapper<TensorGrad>> &
    _get_mapped_grad_names_wrapped() noexcept;
    std::map<std::string, std::reference_wrapper<Layer>> &
    _get_mapped_layer_names_wrapped() noexcept;

    reflect::detail::custom_typed_iterator<Layer> get_all_layers();
    reflect::detail::custom_typed_map<Layer> get_all_named_layers();
    reflect::detail::custom_typed_iterator<TensorGrad> parameters();
    reflect::detail::custom_typed_map<TensorGrad> named_parameters();
    std::string name() const noexcept;
    template <typename... Args> inline TensorGrad forward(Args &&...args) {
        return this->_run_forward(
            utils::make_any_ref_vector(std::forward<Args>(args)...));
    }
    template <typename... Args> inline TensorGrad operator()(Args &&...args) {
        return this->_run_forward(
            utils::make_any_ref_vector(std::forward<Args>(args)...));
    }
};

} // namespace nt
#endif
