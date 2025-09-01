#include "../functional/functional.h"
#include "functional.h"
#include "TensorGrad.h"
#include <chrono>
#include <functional>
#include <iostream>

namespace nt {

// If a constructor needs to be made
// Only the following 4 should actually initialize Node
// This is done to make it easier to remake aspects in the future
// Also, it ensures that the Tensor (called by detach) value is always initialized
TensorGrad::TensorGrad(const Tensor &t, bool grad_required)
    : Node(grad_required ? make_intrusive<grad::utility::GraphNode>(make_intrusive<tensor_holder>(t))  
           : make_intrusive<grad::utility::GraphNode>(make_intrusive<tensor_holder>(t), grad::utility::DontTrackGrad{}) ) , 
            internal_allow_grad_tracking_(true) 
    {}

TensorGrad::TensorGrad(Tensor &&t, bool grad_required)
    : Node(grad_required ? make_intrusive<grad::utility::GraphNode>(make_intrusive<tensor_holder>(std::move(t)))  
           : make_intrusive<grad::utility::GraphNode>(make_intrusive<tensor_holder>(std::move(t)), grad::utility::DontTrackGrad{}) )
            , internal_allow_grad_tracking_(true)
    {}

TensorGrad::TensorGrad(const TensorGrad& tg)
    : Node(tg.Node), internal_allow_grad_tracking_(tg.internal_allow_grad_tracking_)  {}

TensorGrad::TensorGrad(TensorGrad &&tg)
    : Node(std::move(tg.Node)), internal_allow_grad_tracking_(tg.internal_allow_grad_tracking_) {
    // if(!tg.track_grad()){
    //     Node = std::move(tg.Node);
    //     return;
    // }
    // std::cout << "TensorGrad(TensorGrad&&) constructor called" << std::endl;
    // auto graph = tg.get_auto_grad();
    // auto path = graph.get_path();
    // for(const auto& parent : path){
    //     std::cout << parent->backwardFunc->get_name() << "->";
    // }
    // std::cout << "done" << std::endl;
    // Node = tg.Node;
}

TensorGrad::TensorGrad(Scalar value, bool grad_required)
    : TensorGrad(Tensor(value), grad_required) {}


TensorGrad::TensorGrad(std::nullptr_t, bool grad_required)
    : TensorGrad(Tensor::Null(), grad_required) {}



void TensorGrad::track_tensors(std::vector<TensorGrad> &tgs, bool ignore_dont_track) {
    if (!this->track_grad()) return;

    this->Node->ensure_backward_initialization();
    
    for(const auto& tg : tgs){
        if(ignore_dont_track && !tg.track_grad()) continue;
        else{
            utils::THROW_EXCEPTION(tg.track_grad(), "\nError: told to track tensor that is not tracking a gradient"
                                                    ", this will cause a segmentation fault, please look at function implementation"
                                                    "\nTry calling track_grad(std::vector<TensorGrad>, ignore_dont_track = true)");
        }
        tg.Node->ensure_backward_initialization();
        this->Node->parents.emplace_back(tg.Node);
        tg.Node->children.emplace_back(this->Node);
    }
}

void TensorGrad::swap(TensorGrad &tg) {
    this->Node.swap(tg.Node);
    std::swap(this->internal_allow_grad_tracking_, tg.internal_allow_grad_tracking_);
}

TensorGrad &TensorGrad::operator=(const TensorGrad &tg) {
    this->internal_allow_grad_tracking_ = tg.internal_allow_grad_tracking_;
    if (this == &tg) return *this;
    if(tg.is_null()) return *this;
    if(!tg.track_grad()){
        return this->operator=(tg.detach());
    }
    if(this->track_grad() && tg.track_grad()){
        // Make a brand new tensor grad that propogates to both
        const Tensor& result = tg.detach();
        this->detach() = result;
        intrusive_ptr<grad::utility::GraphNode> new_node = make_intrusive<grad::utility::GraphNode>(make_intrusive<tensor_holder>(result));
        new_node->ensure_backward_initialization(true);
        intrusive_ptr<grad::utility::GraphNode> this_node = make_intrusive<grad::utility::GraphNode>();
        *this_node = *this->Node;
        this->Node.reset();
        new_node->parents.emplace_back(this_node);
        new_node->parents.emplace_back(tg.Node);
        this_node->children.emplace_back(new_node);
        tg.Node->children.emplace_back(new_node);
        new_node->backwardFunc->set(
            grad::utility::backward_func::func_type(
            [](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
                // if(parents[0]->Node->grad && !parents[0]->Node->grad->tensor.is_null()){std::cout << parents[0]->grad().shape();}
                // if(parents[1]->Node->grad && !parents[1]->Node->grad->tensor.is_null()){std::cout << parents[1]->grad().shape();}
                parents[0]->accumulate_gradient(grad);
                parents[1]->accumulate_gradient(grad);
            }
           )
        );
        new_node->backwardFunc->set_name("EqualOperator&");
        this->Node = std::move(new_node);
        return *this;
    }
    //!this->track_grad() && tg.track_grad()
    this->Node = tg.Node;
    return *this;
}

TensorGrad &TensorGrad::operator=(TensorGrad &&tg) {
    this->internal_allow_grad_tracking_ = tg.internal_allow_grad_tracking_; 
    if (this == &tg) return *this;
    if(tg.is_null()) return *this;
    if(!tg.track_grad()){
        return this->operator=(std::move(tg.Node->tensor->tensor));
    }
    if(this->track_grad() && tg.track_grad()){
        // Make a brand new tensor grad that propogates to both
        const Tensor& result = tg.detach();
        this->detach() = result;
        intrusive_ptr<grad::utility::GraphNode> new_node = make_intrusive<grad::utility::GraphNode>(make_intrusive<tensor_holder>(result));
        new_node->ensure_backward_initialization(true);
        intrusive_ptr<grad::utility::GraphNode> this_node = make_intrusive<grad::utility::GraphNode>();
        *this_node = *this->Node;
        this->Node.reset();
        new_node->parents.emplace_back(this_node);
        new_node->parents.emplace_back(tg.Node);
        this_node->children.emplace_back(new_node);
        tg.Node->children.emplace_back(new_node);
        new_node->backwardFunc->set(
            grad::utility::backward_func::func_type(
            [](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
                // if(parents[0]->Node->grad && !parents[0]->Node->grad->tensor.is_null()){std::cout << parents[0]->grad().shape();}
                // if(parents[1]->Node->grad && !parents[1]->Node->grad->tensor.is_null()){std::cout << parents[1]->grad().shape();}
                parents[0]->accumulate_gradient(grad);
                parents[1]->accumulate_gradient(grad);
            }
           )
        );
        new_node->backwardFunc->set_name("EqualOperator&&");
        this->Node = std::move(new_node);
        return *this;
    }
    //!this->track_grad() && tg.track_grad()
    this->Node = std::move(tg.Node);
    return *this;
}

TensorGrad &TensorGrad::operator=(Scalar s) {
    if (is_null()) {
        utils::throw_exception(this->Node != nullptr, "Error, trying to use TensorGrad with null node");
        if(!this->Node->tensor){this->Node->tensor = make_intrusive<tensor_holder>(Tensor(s));}
        else{
            this->Node->tensor->tensor = s;
        }
        return *this;
    }
    detach() = s;
    this->track_self_mod_tensors(
        [](const Tensor &grad,
           std::vector<intrusive_ptr<TensorGrad>> &parents) {
             parents[0]->accumulate_gradient(0);
        },
        "OperatorEqualScalar"
    );
    return *this;
}

TensorGrad &TensorGrad::_unsafe_nullify() {
    this->Node.reset();
    return *this;
}

void TensorGrad::ensure_usable() const {
    utils::THROW_EXCEPTION(this->Node != nullptr, "Error: tried to use TensorGrad with null Node, making it unusable");
    utils::THROW_EXCEPTION(this->Node->tensor != nullptr, "Error: tried to use TensorGrad with null Node, making it unusable");
}

TensorGrad &TensorGrad::set_(const Tensor &t) {
    if (is_null()) {
        utils::throw_exception(this->Node != nullptr, "Error, trying to use TensorGrad with null node");
        if(!this->Node->tensor){this->Node->tensor = make_intrusive<tensor_holder>(t);}
        else{
            this->Node->tensor->tensor = t;
        }
        return *this;
    }

    this->detach().set_(t);
    this->track_self_mod_tensors(
        [](const Tensor &grad,
           std::vector<intrusive_ptr<TensorGrad>> &parents) {
             parents[0]->accumulate_gradient(0);
        },
        "Set_"
    );

    return *this;
}

inline bool is_tracking_grad(const Tensor &t) { return false; }

inline bool is_tracking_grad(const TensorGrad &t) noexcept {
    return t.track_grad();
}
template <typename... Args>
inline bool is_tracking_grad(const TensorGrad &t,
                             const Args &...args) noexcept {
    if (t.track_grad()) {
        return true;
    }
    return is_tracking_grad(args...);
}

inline void handle_null_tensors(const TensorGrad &t) {
    utils::throw_exception(!t.is_null(),
                           "Unable to perform operations on null tensors");
}

inline void handle_null_tensors(const Tensor &t) {
    utils::throw_exception(!t.is_null(),
                           "Unable to perform operations on null tensors");
}

template <typename... Args>
inline void handle_null_tensors(const Tensor &t, const Args &...args) {
    handle_null_tensors(t);
    handle_null_tensors(args...);
}

template <typename... Args>
inline void handle_null_tensors(const TensorGrad &t, const Args &...args) {
    handle_null_tensors(t);
    handle_null_tensors(args...);
}

// Addition operation
TensorGrad TensorGrad::operator+(const TensorGrad &other) const {
    handle_null_tensors(*this, other);
    return functional::add(*this, other); 
}

TensorGrad TensorGrad::operator+(const Scalar other) const {
    handle_null_tensors(*this);
    return functional::add(*this, other); 
}

void TensorGrad::redefine_tracking(
    TensorGrad &tg, const TensorGrad &parent,
    std::function<void(const Tensor &, intrusive_ptr<TensorGrad> &)> func, const char* func_name) {
    if (tg.is_null()) {
        return;
    }
    tg.Node->parents.clear();
    tg.Node->children.clear();
    tg.Node->backwardFunc->set(nullptr);
    tg.track_tensors(parent);
    tg.create_backward_function(
        [func](const Tensor &grad,
               std::vector<intrusive_ptr<TensorGrad>> &parents) {
            func(grad, parents[0]);
        });
    tg.Node->backwardFunc->set_name(std::string(func_name));
    // tg.ensure_grads_initialized();
    // this will ensure there is no segmentation faults or anything if a
    // gradient has not been initialized
}

TensorGrad TensorGrad::operator+(const Tensor &other) const {
    handle_null_tensors(*this, other);
    return functional::add(*this, other); 
}

TensorGrad operator+(const Tensor &other, const TensorGrad &tg) {
    handle_null_tensors(tg, other);
    return functional::add(other, tg); 
}

TensorGrad operator+(const Scalar other, const TensorGrad &tg) {
    handle_null_tensors(tg);
    return functional::add(other, tg); 
}

// This Addition operation
TensorGrad &TensorGrad::operator+=(const TensorGrad &other) {
    handle_null_tensors(*this, other);
    return functional::add_(*this, other);
}

TensorGrad &TensorGrad::operator+=(const Tensor &other) {
    handle_null_tensors(*this, other);
    return functional::add_(*this, other);
}

TensorGrad &TensorGrad::operator+=(const Scalar other) {
    handle_null_tensors(*this);
    return functional::add_(*this, other);
}


Tensor &operator+=(Tensor &t, const TensorGrad &tg) {
    handle_null_tensors(t, tg);
    return functional::add_(t, tg);
}

// Subtraction operation
TensorGrad TensorGrad::operator-(const TensorGrad &other) const {
    handle_null_tensors(*this, other);
    return functional::subtract(*this, other);
}

TensorGrad TensorGrad::operator-(const Scalar other) const {
    handle_null_tensors(*this);
    return functional::subtract(*this, other);
}

TensorGrad TensorGrad::operator-(const Tensor &other) const {
    handle_null_tensors(*this, other);
    return functional::subtract(*this, other);
}

TensorGrad operator-(const Tensor &other, const TensorGrad &tg) {
    handle_null_tensors(tg, other);
    return functional::subtract(other, tg);
}

TensorGrad operator-(const Scalar other, const TensorGrad &tg) {
    handle_null_tensors(tg);
    return functional::subtract(other, tg);
}

// This Subtraction operation
TensorGrad &TensorGrad::operator-=(const TensorGrad &other) {
    handle_null_tensors(*this, other);
    return functional::subtract_(*this, other);
}

TensorGrad &TensorGrad::operator-=(const Tensor &other) {
    handle_null_tensors(*this, other);
    return functional::subtract_(*this, other);
}

TensorGrad &TensorGrad::operator-=(const Scalar other) {
    handle_null_tensors(*this);
    return functional::subtract_(*this, other);
}

Tensor &operator-=(Tensor &t, const TensorGrad &tg) {
    handle_null_tensors(t, tg);
    return functional::subtract_(t, tg);
}

// Division operation
TensorGrad TensorGrad::operator/(const TensorGrad &other) const {
    handle_null_tensors(*this, other);
    return functional::divide(*this, other);
}
TensorGrad TensorGrad::operator/(const Scalar other) const {
    handle_null_tensors(*this);
    return functional::divide(*this, other);
}

TensorGrad TensorGrad::operator/(const Tensor &other) const {
    handle_null_tensors(*this, other);
    return functional::divide(*this, other);
}

TensorGrad operator/(const Tensor &other, const TensorGrad &tg) {
    handle_null_tensors(tg, other);
    return functional::divide(other, tg);
}

TensorGrad operator/(const Scalar other, const TensorGrad &tg) {
    handle_null_tensors(tg);
    return functional::divide(other, tg);
}

// This division operation
TensorGrad &TensorGrad::operator/=(const TensorGrad &other) {
    handle_null_tensors(*this, other);
    return functional::divide_(*this, other);
}

TensorGrad &TensorGrad::operator/=(const Tensor &other) {
    handle_null_tensors(*this, other);
    return functional::divide_(*this, other);
}

TensorGrad &TensorGrad::operator/=(const Scalar other) {
    handle_null_tensors(*this);
    return functional::divide_(*this, other);
}

Tensor &operator/=(Tensor &t, const TensorGrad &tg) {
    handle_null_tensors(t, tg);
    return functional::divide_(t, tg);
}

// Multiplication operation
TensorGrad TensorGrad::operator*(const TensorGrad &other) const {
    handle_null_tensors(*this, other);
    return functional::multiply(*this, other);
}

TensorGrad TensorGrad::operator*(const Scalar other) const {
    handle_null_tensors(*this);
    return functional::multiply(*this, other);
}

TensorGrad TensorGrad::operator*(const Tensor &other) const {
    handle_null_tensors(*this, other);
    return functional::multiply(*this, other);
}

TensorGrad operator*(const Tensor &other, const TensorGrad &tg) {
    handle_null_tensors(tg, other);
    return functional::multiply(other, tg);
}

TensorGrad operator*(const Scalar other, const TensorGrad &tg) {
    handle_null_tensors(tg);
    return functional::multiply(other, tg);
}

// This multiplication operation
TensorGrad &TensorGrad::operator*=(const TensorGrad &other) {
    handle_null_tensors(*this, other);
    return functional::multiply_(*this, other);
}

// This multiplication operation
TensorGrad &TensorGrad::operator*=(const Tensor &other) {
    handle_null_tensors(*this, other);
    return functional::multiply_(*this, other);
}

TensorGrad &TensorGrad::operator*=(const Scalar other) {
    handle_null_tensors(*this);
    return functional::multiply_(*this, other);
}

Tensor &operator*=(Tensor &t, const TensorGrad &tg) {
    handle_null_tensors(t, tg);
    return functional::multiply_(t, tg);
}

std::ostream &operator<<(std::ostream &out, const TensorGrad &tg) {
    if (tg.is_null()) {
        return out << "Null";
    } else {
        out << tg.detach();
        if(tg.Node->backwardFunc != nullptr){
            return out << ", grad_fn = " << tg.Node->backwardFunc->get_name();
        }
        return out << ", grad_fn = None";
    }
}



TensorGrad TensorGrad::to_complex_from_real() const {
    handle_null_tensors(*this);
    return functional::to_complex_from_real(*this);
}

TensorGrad TensorGrad::to_complex_from_imag() const {
    handle_null_tensors(*this);
    return functional::to_complex_from_imag(*this);
}

// need to make expand and expand_as before doing this:
TensorGrad TensorGrad::sum(utils::optional_list list, bool keepdim) const {
    // perform the forward sum operation
    handle_null_tensors(*this);
    return functional::sum(*this, list, keepdim);
}

TensorGrad TensorGrad::mean(utils::optional_list list, bool keepdim) const {
    // perform the forward mean operation
    handle_null_tensors(*this);
    Tensor meaned = this->detach().mean(list, true);
    std::vector<int64_t> dims = meaned.shape().Vec();
    TensorGrad result(keepdim ? meaned : meaned.squeeze());
    if (!is_tracking_grad(*this)) {
        result.track_grad_(false);
        return std::move(result);
    }

    // track the current tensor in the result for backward computation
    result.track_tensors(*this);
    size_value_t dim_size;
    if (!list) {
        dim_size = numel();
    } else {
        dim_size = 1;
        for (const auto &dim : list) {
            dim_size *= shape()[dim];
        }
    }

    // define the backward function
    result.create_backward_function(
        [dims, dim_size](const Tensor &grad,
                         std::vector<intrusive_ptr<TensorGrad>> &parents) {
            // calculate the size of the dimension along which the mean was
            // computed this is dim_size

            // expand the gradient to the shape of the original tensor

            // divide the gradient by the size of the dimension to distribute it
            // equally
            parents[0]->accumulate_gradient(
                grad.view(SizeRef(std::move(dims)))
                    .expand_as(parents[0]->grad()) /
                dim_size);
        });

    return std::move(result);
}

result_types::max<TensorGrad, Tensor> TensorGrad::max(utils::optional_list dim, bool keepdim) const{
    handle_null_tensors(*this);
    return functional::max(*this, dim, keepdim);
}

result_types::max<TensorGrad, Tensor> TensorGrad::min(utils::optional_list dim, bool keepdim) const{
    handle_null_tensors(*this);
    return functional::min(*this, dim, keepdim);
}

TensorGrad TensorGrad::exp() const {
    // perform the forward exp operation
    handle_null_tensors(*this);
    return functional::exp(*this);
}

TensorGrad &TensorGrad::exp_() {

    // apply the in-place exponential operation
    handle_null_tensors(*this);
    return functional::exp_(*this);
}

TensorGrad TensorGrad::to_dtype(DType dt) const {
    handle_null_tensors(*this);
    TensorGrad result(this->detach().to(dt), this->track_grad());
    if (!is_tracking_grad(*this)) {
        result.track_grad_(false);
        return std::move(result);
    }
    DType cur_dtype = this->dtype();

    result.track_tensors(*this);
    result.create_backward_function(
        [cur_dtype](const Tensor &grad,
           std::vector<intrusive_ptr<TensorGrad>> &parents) {
            parents[0]->accumulate_gradient(grad.to(cur_dtype));
        });
    return std::move(result);
}

TensorGrad TensorGrad::to_device(DeviceType dt) const {
    handle_null_tensors(*this);
    TensorGrad result(this->detach().to(dt), this->track_grad());
    if (!is_tracking_grad(*this)) {
        result.track_grad_(false);
        return std::move(result);
    }
    
    DeviceType cur_device_type = this->device();

    result.track_tensors(*this);
    result.create_backward_function(
        [cur_device_type](const Tensor &grad,
           std::vector<intrusive_ptr<TensorGrad>> &parents) {
            parents[0]->accumulate_gradient(grad.to(cur_device_type));
        });
    return std::move(result);
}

TensorGrad TensorGrad::contiguous() const {
    handle_null_tensors(*this);
    TensorGrad result(this->detach().contiguous(), this->track_grad());
    if (!is_tracking_grad(*this)) {
        result.track_grad_(false);
        return std::move(result);
    }

    result.track_tensors(*this);
    result.create_backward_function(
        [](const Tensor &grad,
           std::vector<intrusive_ptr<TensorGrad>> &parents) {
            parents[0]->accumulate_gradient(grad);
        });
    return std::move(result);
}

TensorGrad TensorGrad::clone() const {
    handle_null_tensors(*this);
    TensorGrad result(this->detach().clone(), this->track_grad());
    if (!is_tracking_grad(*this)) {
        result.track_grad_(false);
        return std::move(result);
    }

    result.track_tensors(*this);
    result.create_backward_function(
        [](const Tensor &grad,
           std::vector<intrusive_ptr<TensorGrad>> &parents) {
            parents[0]->accumulate_gradient(grad);
        });
    return std::move(result);
}

TensorGrad TensorGrad::pow(Scalar exponent) const {
    handle_null_tensors(*this);
    return functional::pow(*this, exponent); 
}

TensorGrad &TensorGrad::inverse_() {
    handle_null_tensors(*this);
    return functional::inverse_(*this);
}

TensorGrad TensorGrad::inverse() const {
    handle_null_tensors(*this);
    return functional::inverse(*this);
}

TensorGrad TensorGrad::clip(Scalar lower, Scalar higher) const {
    handle_null_tensors(*this);
    return functional::clamp(*this, lower, higher);
}

TensorGrad &TensorGrad::clip_(Scalar lower, Scalar higher) {
    handle_null_tensors(*this);
    return functional::clamp_(*this, lower, higher);
}

TensorGrad TensorGrad::pad(std::vector<size_value_t> p, const char *mode,
                           Scalar value) const {
    handle_null_tensors(*this);
    return functional::pad(*this, p, mode, value);
}

TensorGrad TensorGrad::unpad(std::vector<size_value_t> p) const {
    return functional::unpad(*this, p, false);
}

TensorGrad TensorGrad::flip(utils::optional_list list) const {
    handle_null_tensors(*this);
    return functional::flip(*this, list);
}



TensorGrad TensorGrad::dilate(size_value_t dil) const {
    handle_null_tensors(*this);
    return functional::dilate(*this, dil);
}

TensorGrad TensorGrad::undilate(size_value_t dil) const {
    handle_null_tensors(*this);
    return functional::undilate(*this, dil);
}

// these are all the operations where it is just the stride or view changed
// when that happens, the same thing cam just happen to the gradient that is
// being tracked and when that gradient is corrected, it will automatically
// update the gradient of the original tensor appropriately

#define COMBINE_PAIR(type, name) type name

#define COMBINE_RECURSIVE_0()
#define COMBINE_RECURSIVE_1(type1, name1) COMBINE_PAIR(type1, name1)
#define COMBINE_RECURSIVE_2(type1, name1, type2, name2)                        \
    COMBINE_PAIR(type1, name1), COMBINE_PAIR(type2, name2)
#define COMBINE_RECURSIVE_3(type1, name1, type2, name2, type3, name3)          \
    COMBINE_PAIR(type1, name1), COMBINE_PAIR(type2, name2),                    \
        COMBINE_PAIR(type3, name3)
#define COMBINE_RECURSIVE_4(type1, name1, type2, name2, type3, name3, type4,   \
                            name4)                                             \
    COMBINE_PAIR(type1, name1), COMBINE_PAIR(type2, name2),                    \
        COMBINE_PAIR(type3, name3), COMBINE_PAIR(type4, name4)
// Add more as needed

#define COMBINE_SELECT_MACRO(_1, _1b, _2, _2b, _3, _3b, _4, _4b, NAME, ...) NAME
#define COMBINE_ARGUMENTS(...)                                                 \
    COMBINE_SELECT_MACRO(                                                      \
        __VA_ARGS__, COMBINE_RECURSIVE_4, COMBINE_RECURSIVE_4,                 \
        COMBINE_RECURSIVE_3, COMBINE_RECURSIVE_3, COMBINE_RECURSIVE_2,         \
        COMBINE_RECURSIVE_2, COMBINE_RECURSIVE_1, COMBINE_RECURSIVE_0,         \
        COMBINE_RECURSIVE_0)(__VA_ARGS__)

#define EXTRACT_ODD_PAIR(type, name) name
#define EXTRACT_ODD_RECURSIVE_0()
#define EXTRACT_ODD_RECURSIVE_1(type1, name1) name1
#define EXTRACT_ODD_RECURSIVE_2(type1, name1, type2, name2) name1, name2
#define EXTRACT_ODD_RECURSIVE_3(type1, name1, type2, name2, type3, name3)      \
    name1, name2, name3
#define EXTRACT_ODD_RECURSIVE_4(type1, name1, type2, name2, type3, name3,      \
                                type4, name4)                                  \
    name1, name2, name3, name4
#define EXTRACT_SELECT_MACRO(_1, _1b, _2, _2b, _3, _3b, _4, _4b, NAME, ...) NAME

#define EXTRACT_ODD_ARGUMENTS(...)                                             \
    EXTRACT_SELECT_MACRO(__VA_ARGS__, EXTRACT_ODD_RECURSIVE_4,                 \
                         EXTRACT_ODD_RECURSIVE_4, EXTRACT_ODD_RECURSIVE_3,     \
                         EXTRACT_ODD_RECURSIVE_3, EXTRACT_ODD_RECURSIVE_2,     \
                         EXTRACT_ODD_RECURSIVE_2, EXTRACT_ODD_RECURSIVE_1,     \
                         EXTRACT_ODD_RECURSIVE_0,                              \
                         EXTRACT_ODD_RECURSIVE_0)(__VA_ARGS__)

#define TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(op, ...)                       \
    TensorGrad TensorGrad::op(COMBINE_ARGUMENTS(__VA_ARGS__)) const {          \
        handle_null_tensors(*this);                                            \
        TensorGrad result(detach().op(EXTRACT_ODD_ARGUMENTS(__VA_ARGS__)),     \
                          this->track_grad());                                 \
        result.track_grad(                                                     \
            *this, [EXTRACT_ODD_ARGUMENTS(__VA_ARGS__)](Tensor &grad) {        \
                return grad.op(EXTRACT_ODD_ARGUMENTS(__VA_ARGS__));            \
            });                                                                \
        return std::move(result);                                              \
    }


// Important rules for DType::TensorObj:
// - The way that gradients for if the dtype is DType::TensorObj:
//      - If every single Tensor is not tracking the gradient:
//          - Then this->track_grad() == false;
//      - Otherwise:
//          - the do track grad is: !this->Node->grad->tensor[i].item<Tensor>().is_null()
//              - if the tensor at the gradient at that point is Null, then the gradient is not being tracked
//                Otherwise, it is being tracked
//
TensorGrad TensorGrad::operator[](size_value_t i) const {
    handle_null_tensors(*this);
    if (this->dtype() == DType::TensorObj && dims() == 1) {
        bool grad_being_tracked = this->track_grad() && !(this->Node->grad->tensor[i].item<Tensor>().is_null());
        // look above for rule

        TensorGrad result(detach()[i].item<Tensor>(), grad_being_tracked);
        if (!grad_being_tracked) {
            result.track_grad_(false);
            return std::move(result);
        }
        result.track_grad(*this, [i](Tensor& grad) { return grad[i].item<Tensor>(); });
    }
    TensorGrad result(detach()[i], this->track_grad());
    result.track_grad(*this, [i](Tensor &grad) { return grad[i]; });
    return std::move(result);
}


TensorGrad TensorGrad::operator[](std::vector<size_value_t> vec) const {
    handle_null_tensors(*this);
    if(this->dtype() == DType::TensorObj){
        Tensor out = detach()[vec];
        if(out.numel() > 1){
            TensorGrad result(out, this->track_grad());
            result.track_grad(*this, [&vec](Tensor& grad) {return grad[vec];});
            return std::move(result);
        }
        if(!this->track_grad()){
            return TensorGrad(out.item<Tensor>(), false);
        }
        
        Tensor grad = this->Node->grad->tensor[vec].item<Tensor>();
        if(grad.is_null())
            return TensorGrad(out.item<Tensor>(), false);
        TensorGrad result(out.item<Tensor>(), true);
        result.track_grad(*this, [&grad](Tensor& gr){return grad;});
        return std::move(result);
    }
    TensorGrad result(detach()[vec], this->track_grad());
    result.track_grad(*this, [&vec](Tensor &grad) { return grad[vec]; });
    return std::move(result);
}

TensorGrad TensorGrad::transpose(size_value_t a, size_value_t b) const {
    if (a == b) {
        return *this;
    }
    handle_null_tensors(*this);
    TensorGrad result(detach().transpose(a, b), this->track_grad());
    result.track_grad(*this,
                      [a, b](Tensor &grad) { return grad.transpose(a, b); });
    return std::move(result);
}

// these are all operations where the stride or view of the memory is changed
// (the actual values in the memory are not)
// for that reason, the same operation can just be done to track the gradient
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(view, SizeRef, s)
/* TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(operator[], size_value_t, i) */
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(operator[], range_, i)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(operator[], std::vector<range_>, i)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(operator[], Tensor, i)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(permute, std::vector<size_value_t>, v)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(unsqueeze, size_value_t, dim)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(unsqueeze_as, const Tensor &, dim)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(unsqueeze_as, const SizeRef &, dim)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(squeeze, utils::optional_list, list)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(flatten, size_value_t, a, size_value_t,
                                        b)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(unflatten, size_value_t, a,
                                        size_value_t, b)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(unfold, size_value_t, dim, size_value_t,
                                        size, size_value_t, step);
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(split_axis, std::vector<range_>, i)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(split_axis, size_value_t, i)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(split_axis_1)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(div, size_value_t, i)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(real)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(imag)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(flip_view, utils::optional_list, list)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(undilate_, size_value_t, dil)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(repeat_, size_value_t, amt)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(repeat_, size_value_t, dim,
                                        size_value_t, amt)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(expand, SizeRef, s)

/* TensorGrad TensorGrad::operator[](TensorGrad::size_value_t&&) const; */
/* TensorGrad TensorGrad::operator[](Tensor&&) const; */
/* TensorGrad TensorGrad::operator[](range_&&) const; */
/* TensorGrad TensorGrad::operator[](std::vector<range_>&&) const; */

#undef TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION
#undef COMBINE_PAIR
#undef COMBINE_RECURSIVE_0
#undef COMBINE_RECURSIVE_1
#undef COMBINE_RECURSIVE_2
#undef COMBINE_RECURSIVE_3
#undef COMBINE_RECURSIVE_4
#undef COMBINE_SELECT_MACRO
#undef COMBINE_ARGUMENTS
#undef EXTRACT_ODD_PAIR
#undef EXTRACT_ODD_RECURSIVE_0
#undef EXTRACT_ODD_RECURSIVE_1
#undef EXTRACT_ODD_RECURSIVE_2
#undef EXTRACT_ODD_RECURSIVE_3
#undef EXTRACT_ODD_RECURSIVE_4
#undef EXTRACT_SELECT_MACRO
#undef EXTRACT_ODD_ARGUMENTS

grad::AutoGrad<std::unordered_set<intrusive_ptr<grad::utility::GraphNode>>> TensorGrad::get_auto_grad(bool validate){
    return validate ? grad::AutoGrad(this->Node) : grad::AutoGrad(this->Node, grad::utility::DontValidateGraph{});
}


void TensorGrad::backward() {
    auto graph = this->get_auto_grad(/* validate = */ true);
    graph.backward();
}

void TensorGrad::backward(const Tensor& initial_grad) {
    auto graph = this->get_auto_grad(/* validate = */ true);
    graph.backward(initial_grad);
}

void TensorGrad::zero_grad() {
    // No need to validate because validation in this case would make sure the gradient is define
    // and that the backward function is defined, and if not throw an exception
    // Zero grad will make sure the gradient is defined and zero anyways
    auto graph = this->get_auto_grad(/* validate = */ false);
    graph.zero_grad();
}

void TensorGrad::zero_self_grad_only() {
    Node->zero_grad();
}

void TensorGrad::accumulate_gradient(const Tensor& in_grad){
    this->Node->accumulate_gradient(in_grad);
}

void TensorGrad::accumulate_gradient(Scalar num){
    this->Node->accumulate_gradient(num);
}

void TensorGrad::update() {
    if(this->is_null()) return;
    utils::throw_exception(this->Node->grad && !this->grad().is_null(), 
                           "Error: Backward has not been run, gradient is not defined when running update()");
    handle_null_tensors(*this);
    this->detach() -= this->grad();
}

void TensorGrad::update_mutable(){
    if(this->is_null()) return;
    handle_null_tensors(*this);
    if(detach().is_mutable()){update(); return;}
    // utils::throw_exception(!this->backwardFunc || this->backwardFunc->used(),
    //                        "Trying to update immutable tensor inside of tensor grad"
    //                        "But the backward function has not been used"
    //                        "Could still potentially be being used by auto grad");
    // utils::throw_exception(!this->parents || this->parents->size() == 0,
    //                        "Trying to update immutable tensor inside of tensor grad"
    //                        "But parents are not empty"
    //                        "Could still potentially be being used by auto grad");
    // utils::throw_exception(!this->children || this->children->size() == 0,
    //                        "Trying to update immutable tensor inside of tensor grad"
    //                        "But children are not empty"
    //                        "Could still potentially be being used by auto grad");
    Tensor& grad = this->grad();
    this->detach().force_mutable_function(
        [&grad](Tensor& self){self -= grad;});
}

// this function doesn't change any grads
// just if any of the parent grads are not initialized
// this will initialize them
void TensorGrad::ensure_grads_initialized() {
    auto graph = this->get_auto_grad(false);
    for(const auto& node : graph.get_path()){
        node->ensure_backward_initialization(true);
    }
}

// should only be called internally
void TensorGrad::_erase_graph(){
    this->Node->children.clear();
    this->Node->parents.clear();
    this->Node->backwardFunc->set(nullptr);
}


} // namespace nt
