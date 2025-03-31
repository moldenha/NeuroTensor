//as the Module class becomes more advanced, and is starting to be able to replace
//this class, this will become depreceated
//more built as a way to hold onto the pointers

#ifndef _NT_LAYER_H_
#define _NT_LAYER_H_

#include "forward_Layer.h"
#include "../Tensor.h"
#include "../dtype/compatible/DType_compatible.h"
#include "../functional/functional.h"
#include "../utils/any_ref.h"
#include "Module.h"
#include "TensorGrad.h"
#include "layers.h"
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <variant>

namespace nt {

class Layer {
    intrusive_ptr<Module> ptr_;
    /* intrusive_ptr<LayerGraph> MyGraph, ParentGraph; */
    std::type_index type_;
    bool is_eval;
    void get_all_layers(reflect::detail::custom_typed_map<Layer> &map,
                        std::string add);
    /* void register_sub_layers() noexcept; */
    TensorGrad _run_forward(std::vector<utils::any_ref>);

  public:
    Layer() = delete;

    template <typename T,
              std::enable_if_t<std::is_base_of_v<Module, T>, bool> = true>
    Layer(intrusive_ptr<T> mod)
        : ptr_(mod), type_(typeid(T)), is_eval(false)
    /* Layer(intrusive_ptr<T> mod) */
    /* :ptr_(mod), MyGraph(make_intrusive<LayerGraph>()), ParentGraph(nullptr),
       type_(typeid(T)), is_eval(false) */
    /* {this->register_sub_layers();} */
    {}

    template <typename T,
              std::enable_if_t<std::is_base_of_v<Module, T>, bool> = true>
    Layer(const T &mod)
        : ptr_(make_intrusive<T>(mod)), type_(typeid(T)), is_eval(false)
    /* Layer(const T& mod) */
    /* :ptr_(make_intrusive<T>(mod)), MyGraph(make_intrusive<LayerGraph>()),
       ParentGraph(nullptr), type_(typeid(T)), is_eval(false) */
    /* {this->register_sub_layers();} */
    {}

    Layer(Layer&& l)
    :ptr_(std::move(l.ptr_)), type_(l.type_), is_eval(l.is_eval)
    {}

    Layer(const Layer& l)
    :ptr_(l.ptr_), type_(l.type_), is_eval(l.is_eval)
    {}

    template <typename T,
              std::enable_if_t<std::is_base_of_v<Module, T>, bool> = true>
    inline Layer &operator=(intrusive_ptr<T> mod) {
        ptr_ = mod;
        /* MyGraph = make_intrusive<LayerGraph>(); */
        /* ParentGraph = nullptr; */
        type_ = typeid(T);
        is_eval = false;
        /* this->register_sub_layers(); */
        return *this;
    }

    inline Layer &operator=(const Layer& l) {
        ptr_ = l.ptr_;
        type_ = l.type_;
        is_eval = l.is_eval;
        return *this;
    }
    
    inline Layer &operator=(Layer&& l) {
        ptr_ = std::move(l.ptr_);
        type_ = l.type_;
        is_eval = l.is_eval;
        return *this;
    }

    template <typename T>
    inline reflect::detail::custom_typed_iterator<T> get_vars() {
        return reflect::detail::get_module_vars<T>(ptr_);
    }
    template <typename T>
    inline reflect::detail::custom_typed_map<T> get_named_vars() {
        return reflect::detail::get_named_module_vars<T>(ptr_);
    }
    inline std::string name() const noexcept {
        return reflect::detail::get_module_name(ptr_);
    }

    inline reflect::detail::custom_typed_map<Layer> get_all_named_layers() {
        reflect::detail::custom_typed_map<Layer> outp;
        this->get_all_layers(outp, name());
        return std::move(outp);
    }
    reflect::detail::custom_typed_iterator<Layer> get_all_layers();
    reflect::detail::custom_typed_iterator<TensorGrad> parameters();
    reflect::detail::custom_typed_map<TensorGrad> named_parameters();

    template <typename T> inline T *is_layer() noexcept {
        return dynamic_cast<T *>(ptr_.get());
    }
    template <typename T> inline const T *is_layer() const noexcept {
        return dynamic_cast<const T *>(ptr_.get());
    }
    Layer &eval();
    Layer &train();
    void update();

    // the idea is that each connecting gradient will have its gradient tracker
    // in place so if you notice, back in the LayerGraph class, everywhere where
    // a new node is made, and an input is set, that tensor grad does not have a
    // gradient graph associated with it but the output does
    /* TensorGrad forward(TensorGrad _x); */
    template <typename... Args> inline TensorGrad forward(Args &&...args) {
        return this->_run_forward(
            utils::make_any_ref_vector(std::forward<Args &&>(args)...));
    }
    /* Tensor backward(Tensor grad); //returns dx */
    template <typename... Args> inline TensorGrad operator()(Args &&...args) {
        return this->forward(std::forward<Args>(args)...);
    }
    /* inline void dump_graph() noexcept {MyGraph->dump_graph();} */
};

} // namespace nt

#include "Loss.h"
#endif //_NT_LAYER_H_
