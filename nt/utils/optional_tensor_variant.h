#ifndef NT_UTILS_OPTIONAL_TENSORVARIANT_H__
#define NT_UTILS_OPTIONAL_TENSORVARIANT_H__

#include "optional_forward.h"
#include "../Tensor.h"
#include "../intrusive_ptr/intrusive_ptr.hpp"
#include "../nn/TensorGrad.h"
#include "api_macro.h"
#include "optional_except.h"
#include "optional_tensor.h"
#include "optional_tensorgrad.h"
#include "type_traits.h"
#include <initializer_list>

namespace nt::utils {

namespace details{

template<typename T>
struct valid_optional_tensor_variant_type : std::false_type {};

template<>
struct valid_optional_tensor_variant_type<Tensor> : std::true_type {};

template<>
struct valid_optional_tensor_variant_type<std::nullptr_t> : std::true_type {};

template<>
struct valid_optional_tensor_variant_type<TensorGrad> : std::true_type {};

}


// this combines optional_tensor and optional_tensorgrad
// it allows for the functional and nt namespaces to be combined for things
// like batch_norm where there is an optional tensor argument
class NEUROTENSOR_API optional_tensor_variant {
    bool grad;

    union data {
        intrusive_ptr<TensorGrad> tensor_grad;
        intrusive_ptr<tensor_holder> tensor;

        data() {}  // no default construction
        ~data() {} // no automatic destruction

        void construct_grad(const intrusive_ptr<TensorGrad>& tg) {
            new(&tensor_grad) intrusive_ptr<TensorGrad>(tg);
        }

        void construct_tensor(const intrusive_ptr<tensor_holder>& t) {
            new(&tensor) intrusive_ptr<tensor_holder>(t);
        }

        void construct_grad(intrusive_ptr<TensorGrad>&& tg) {
            new(&tensor_grad) intrusive_ptr<TensorGrad>(std::move(tg));
        }

        void construct_tensor(intrusive_ptr<tensor_holder>&& t) {
            new(&tensor) intrusive_ptr<tensor_holder>(std::move(t));
        }

        void destroy(bool grad) {
            if (grad) tensor_grad.~intrusive_ptr<TensorGrad>();
            else tensor.~intrusive_ptr<tensor_holder>();
        }
    } t;

  public:
    optional_tensor_variant(const optional_tensor_variant &other)
        : grad(other.grad) {
            grad ? t.construct_grad(other.t.tensor_grad) : t.construct_tensor(other.t.tensor);
       }
    optional_tensor_variant(optional_tensor_variant &&other)
        : grad(other.grad){
            grad ? t.construct_grad(std::move(other.t.tensor_grad)) : t.construct_tensor(std::move(other.t.tensor));
            other.reset();
            other.grad = false;
            other.t.construct_tensor(intrusive_ptr<tensor_holder>(nullptr));
        }
    optional_tensor_variant(const intrusive_ptr<TensorGrad> &other)
        : grad(true){
            t.construct_grad(other);
        }
    optional_tensor_variant(intrusive_ptr<TensorGrad> &&other)
        : grad(true){
            t.construct_grad(std::move(other));
        }
    optional_tensor_variant(const intrusive_ptr<tensor_holder> &other)
        : grad(false){
            t.construct_tensor(other);
        }
    optional_tensor_variant(intrusive_ptr<tensor_holder> &&other)
        : grad(false){
            t.construct_tensor(std::move(other));
        }
    optional_tensor_variant(const TensorGrad &other)
        : grad(true){
            t.construct_grad(make_intrusive<TensorGrad>(other));
        }
    optional_tensor_variant(TensorGrad &&other)
        : grad(true){
            t.construct_grad(make_intrusive<TensorGrad>(std::move(other)));
        }
    optional_tensor_variant(const Tensor &other)
        : grad(false){
            t.construct_tensor(make_intrusive<tensor_holder>(other));
        }
    optional_tensor_variant(Tensor &&other)
        : grad(false){
            t.construct_tensor(make_intrusive<tensor_holder>(std::move(other)));
        }
    optional_tensor_variant(std::nullptr_t) : grad(false){
        t.construct_tensor(intrusive_ptr<tensor_holder>(nullptr));
    }
    optional_tensor_variant() : grad(false){
        t.construct_tensor(intrusive_ptr<tensor_holder>(nullptr));
    }
    ~optional_tensor_variant() {
        this->reset();
    }

    inline optional_tensor_variant &
    operator=(const optional_tensor_variant &other) {
        this->reset();
        grad = other.grad;
        grad ? this->t.construct_grad(other.t.tensor_grad) : this->t.construct_tensor(other.t.tensor);
        return *this;
    }
    inline optional_tensor_variant &operator=(optional_tensor_variant &&other) {
        this->reset();
        grad = other.grad;
        grad ? t.construct_grad(std::move(other.t.tensor_grad)) : t.construct_tensor(std::move(other.t.tensor));
        other.reset();
        other.t.construct_tensor(intrusive_ptr<tensor_holder>(nullptr));
        other.grad = false;
        return *this;
    }
    inline optional_tensor_variant &operator=(TensorGrad &&other) {
        this->reset();
        grad = true;
        this->t.construct_grad(make_intrusive<TensorGrad>(std::move(other)));
        return *this;
    }
    inline optional_tensor_variant &operator=(Tensor &&other) {
        this->reset();
        grad = false;
        this->t.construct_tensor(make_intrusive<tensor_holder>(std::move(other)));
        return *this;
    }
    inline optional_tensor_variant &operator=(const TensorGrad &other) {
        this->reset();
        grad = true;
        this->t.construct_grad(make_intrusive<TensorGrad>(other));
        return *this;
    }
    inline optional_tensor_variant &operator=(const Tensor &other) {
        this->reset();
        grad = false;
        this->t.construct_tensor(make_intrusive<tensor_holder>(other));
        return *this;
    }
    inline optional_tensor_variant &
    operator=(intrusive_ptr<TensorGrad> &&other) {
        this->reset();
        grad = true;
        this->t.construct_grad(std::move(other));
        return *this;
    }
    inline optional_tensor_variant &
    operator=(intrusive_ptr<tensor_holder> &&other) {
        this->reset();
        grad = false;
        this->t.construct_tensor(std::move(other));
        return *this;
    }

    inline optional_tensor_variant &
    operator=(const intrusive_ptr<TensorGrad> &other) {
        this->reset();
        grad = true;
        this->t.construct_grad(other);
        return *this;
    }
    inline optional_tensor_variant &
    operator=(const intrusive_ptr<tensor_holder> &other) {
        this->reset();
        grad = false;
        this->t.construct_tensor(other);
        return *this;
    }

    inline optional_tensor_variant &operator=(std::nullptr_t) {
        this->reset();
        this->grad = false;
        this->t.construct_tensor(intrusive_ptr<tensor_holder>(nullptr));
        return *this;
    }

    inline bool has_value() const noexcept {
        return grad ? bool(t.tensor_grad) : bool(t.tensor);
    }
    inline explicit operator bool() const noexcept { return has_value(); }
    inline explicit operator optional_tensor() const noexcept {return !grad ? optional_tensor(t.tensor) : optional_tensor(nullptr);}
    inline explicit operator optional_tensorgrad() const noexcept {return grad ? optional_tensorgrad(t.tensor_grad) : optional_tensorgrad(nullptr);}
    inline bool tracking_grad() const noexcept {return grad && bool(t.tensor_grad);}

    template<typename T>
    inline std::conditional_t<std::is_same_v<T, TensorGrad>, optional_tensorgrad, optional_tensor> to_single_optional() const noexcept {
        static_assert(std::is_same_v<T, Tensor> || std::is_same_v<T, TensorGrad>,
                      "Error, cannot process value that is not tensor or tensor grad");
        if constexpr (std::is_same_v<T, Tensor>){
            return optional_tensor(*this);
        }
        else{
            return optional_tensorgrad(*this);
        }
    }
    template<typename T>
    inline const T& value() const& {
        static_assert(std::is_same_v<T, Tensor> || std::is_same_v<T, TensorGrad>,
                      "Error, cannot return value that is not tensor or tensor grad");
        if constexpr (std::is_same_v<T, Tensor>){
            if(!has_value() || grad){throw bad_optional_access();}
            return t.tensor->tensor;
        }else{
            if(!has_value() || !grad){throw bad_optional_access();}
            return *t.tensor_grad;
        }
    }

    template<typename T>
    inline T& value() & {
        static_assert(std::is_same_v<T, Tensor> || std::is_same_v<T, TensorGrad>,
                      "Error, cannot return value that is not tensor or tensor grad");
        if constexpr (std::is_same_v<T, Tensor>){
            if(!has_value() || grad){throw bad_optional_access();}
            return t.tensor->tensor;
        }else{
            if(!has_value() || !grad){throw bad_optional_access();}
            return *t.tensor_grad;
        }
    }

    template<typename T, class U>
    inline T value_or(U&& default_value) const& {
        static_assert(std::is_same_v<T, Tensor> || std::is_same_v<T, TensorGrad>,
                      "Error, cannot return value that is not tensor or tensor grad");
        return bool(*this) ? this->value<T>() : T(std::forward<U>(default_value));
    }

    template<typename T, class U>
    inline T value_or(U&& default_value) && {
        static_assert(std::is_same_v<T, Tensor> || std::is_same_v<T, TensorGrad>,
                      "Error, cannot return value that is not tensor or tensor grad");
        return bool(*this) ? std::move(this->value<T>()) : T(std::forward<U>(default_value));
    }

    template<typename T, class F>
    inline auto and_then(F&& f) const& { return this->to_single_optional<T>().and_then(std::forward<F>(f)); }
    template<typename T, class F>
    inline auto and_then(F&& f) && { return this->to_single_optional<T>().and_then(std::forward<F>(f)); }

    inline void swap(optional_tensor_variant& op) noexcept {
        if(grad == op.grad){
            if(grad)
                op.t.tensor_grad.swap(t.tensor_grad);
            else
                op.t.tensor.swap(t.tensor);
        }else{
            if(grad){
                intrusive_ptr<TensorGrad> hold = std::move(t.tensor_grad);
                t.tensor = std::move(op.t.tensor);
                op.t.tensor_grad = std::move(hold);
                std::swap(grad, op.grad);
            }else{
                intrusive_ptr<TensorGrad> hold = std::move(op.t.tensor_grad);
                op.t.tensor = std::move(this->t.tensor);
                t.tensor_grad = std::move(hold);
                std::swap(grad, op.grad);
            }
        }
    }

    inline void reset() {this->t.destroy(grad);}
    template<typename T, typename... Args>
    inline auto& emplace(Args&&... args){
        static_assert(std::is_same_v<T, Tensor> || std::is_same_v<T, TensorGrad>,
                      "Error, cannot return value that is not tensor or tensor grad");
        if(*this){
            reset();
            return emplace(std::forward<Args>(args)...);
        }
        if constexpr(std::is_same_v<T, TensorGrad>){
            grad = true;
            this->t.tensor_grad = make_intrusive<T>(T(std::forward<Args>(args)...));
            return *this->t.tensor_grad;
        }else{
            grad = false;
            this->t.tensor = make_intrusive<tensor_holder>(T(std::forward<Args>(args)...));
            return this->t.tensor->tensor;
        }
    }
};


namespace details{

// this is a function that forces a tensograd optional
// by default the utils::optional_tensorgrad(var) will be none if var is holding only a tensor
// This will make it so that if it does not have a value only then it will be none
inline utils::optional_tensorgrad force_optional_tg(utils::optional_tensor_variant var){
    if(!var.has_value()) return utils::optional_tensorgrad(nullptr);
    if(var.tracking_grad()) return utils::optional_tensorgrad(var);
    return utils::optional_tensorgrad(TensorGrad(var.value<Tensor>(), false));
}

}

} // namespace nt::utils

namespace std{
inline void swap(nt::utils::optional_tensor_variant& a, nt::utils::optional_tensor_variant& b) noexcept {a.swap(b);}
} //std::


#endif
