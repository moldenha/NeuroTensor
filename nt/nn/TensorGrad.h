#ifndef NT_TENSOR_GRAD_H__
#define NT_TENSOR_GRAD_H__
#include "forward_Grad.h"

#include "../Tensor.h"
#include "../intrusive_ptr/intrusive_ptr.hpp"
#include "../utils/name_func_macro.h"
#include "../utils/optional_list.h"
#include "../utils/tensor_holder.h"
#include "ScalarGrad.h"
#include "functional_class.h"
#include "AutoGrad.h"
#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace nt {

namespace functional {
template <
    typename T, typename... Args,
    std::enable_if_t<std::is_same_v<std::decay_t<T>, TensorGrad>, int> = 0>
TensorGrad list(T &&first, Args &&...rest); // forward declaration
}

class NEUROTENSOR_API TensorGrad : public intrusive_ptr_target {
    // friend classes:

    friend class ScalarGrad;
    template<class HashSetAny> friend class grad::AutoGrad;
    friend struct grad::utility::GraphNode;
    friend struct grad::utility::backward_func;
    friend class functional::TensorGrad_Functional_Class;
    friend class intrusive_ptr<TensorGrad>;

    // Private variables:
    mutable intrusive_ptr<grad::utility::GraphNode> Node;
    // Node holds the tensor, grad, and the backwardFunc
    // to get the tensor:
    //  - Node->tensor->tensor
    // to get the gradient:
    //  - Node->grad->tensor
    // to get the backward function:
    //  - Node->backwardFunc
    bool internal_allow_grad_tracking_;
    // This is a boolean generally set to true
    // However, in certain cases, when a layer doesn't want the tensor to be able to track the gradient
    // this can be set to false
    
    // Node-Specific constructors:
    TensorGrad(const intrusive_ptr<grad::utility::GraphNode>& node__)
    :Node(node__), internal_allow_grad_tracking_(true) {
        if(!Node->tensor)
            Node->tensor = make_intrusive<tensor_holder>(Tensor::Null());
    }
    TensorGrad(intrusive_ptr<grad::utility::GraphNode>&& node__)
    :Node(std::move(node__)), internal_allow_grad_tracking_(true) {
        if(!Node->tensor)
            Node->tensor = make_intrusive<tensor_holder>(Tensor::Null());
    }

  public:
    // Public variables:
    using size_value_t = Tensor::size_value_t;
    // This used to be a variable, but is now a function
    inline bool track_grad() const noexcept {return bool(Node->grad);}
    inline void track_grad_(bool do_track_grad){
        if(track_grad() == do_track_grad) return;
        if(!internal_allow_grad_tracking_){
            if(track_grad()){
                Node->grad.reset();
                Node->backwardFunc.reset();
            }
            return;
        }
        if(do_track_grad){
            if(Node->tensor->tensor.is_null())
                Node->grad = make_intrusive<tensor_holder>(Tensor::Null());
            else 
                Node->grad = make_intrusive<tensor_holder>(nt::functional::zeros_like(Node->tensor->tensor));
            Node->backwardFunc = make_intrusive<grad::utility::backward_func>();
        }else{
            Node->grad.reset();
            Node->backwardFunc.reset();
        }
    }

    inline void perm_disable_grad_tracking() {  internal_allow_grad_tracking_ = false; }
    // Public friend functions
    template <
        typename T, typename... Args,
        std::enable_if_t<std::is_same_v<std::decay_t<T>, TensorGrad>, int>>
    friend TensorGrad functional::list(T &&first, Args &&...rest);

  private:


    inline static intrusive_ptr<tensor_holder>
    make_tensor_holder(const TensorGrad &t) {
        return nt::intrusive_ptr<tensor_holder>::make(
            t.detach().conditional_mutate_clone());
    }
    inline static intrusive_ptr<tensor_holder>
    make_tensor_holder(intrusive_ptr<tensor_holder> t) {
        return t;
    }
    inline static intrusive_ptr<tensor_holder>
    make_tensor_holder(const Tensor &t) {
        return intrusive_ptr<tensor_holder>::make(t.conditional_mutate_clone());
    }


    // Use of track tensors:
    //  TensorGrad out(op(a, b), true);
    //  out.track_tensors(a, b); <- now the gradient of a and b can be updated with the backward function
    template <typename... Args>
    void track_tensors(const TensorGrad &, const Args &...args);                   // TensorGrad.hpp
    void track_tensors(const TensorGrad &);                                        // TensorGrad.hpp
    void track_tensors(std::vector<TensorGrad> &, bool ignore_dont_track = false); // TensorGrad.cpp
    template <typename BackFunc, typename... Args>
    void track_self_mod_tensors(BackFunc&&, const char*, const TensorGrad &, const Args &...args);          // TensorGrad.hpp
    template <typename BackFunc>
    void track_self_mod_tensors(BackFunc&&, const char*);                                                 // TensorGrad.hpp
    // function designed to track view and stride changes of gradient
    //[when memory remains unmodified]
    template <typename OutOperator>
    void track_grad(const TensorGrad &t, OutOperator &&op,
                    const char *func_name = __NT_FUNCTION_NAME__);                 // TensorGrad.hpp
    // create backward function for output
    template <typename backward_func>
    void create_backward_function(backward_func &&func,
                                  const char *func_name = __NT_FUNCTION_NAME__);   // TensorGrad.hpp
    template <typename backward_func, typename Arg1>
    void create_backward_function(backward_func &&func, Arg1 &&arg1,
                                  const char *func_name = __NT_FUNCTION_NAME__);   // TensorGrad.hpp
    template <typename backward_func, typename Arg1, typename Arg2>
    void create_backward_function(backward_func &&func, Arg1 &&arg1,
                                  Arg2 &&arg2,
                                  const char *func_name = __NT_FUNCTION_NAME__);   // TensorGrad.hpp
    template <typename backward_func, typename Arg1, typename Arg2,
              typename Arg3>
    void create_backward_function(backward_func &&func, Arg1 &&arg1,
                                  Arg2 &&arg2, Arg3 &&arg3,
                                  const char *func_name = __NT_FUNCTION_NAME__);   // TensorGrad.hpp
    template <typename backward_func, typename Arg1, typename Arg2,
              typename Arg3, typename Arg4>
    void create_backward_function(backward_func &&func, Arg1 &&arg1,
                                  Arg2 &&arg2, Arg3 &&arg3, Arg4 &&arg4,
                                  const char *func_name = __NT_FUNCTION_NAME__);   // TensorGrad.hpp

    // the next 3 are related to handling children and branching
    // and making sure that the gradient is calculated in the correct order of
    // branches
    void run_backward(weak_intrusive_ptr<TensorGrad>);
    
    // nullify is now a private function because it sets Node to a nullptr
    // This is a dangerous function, and basically renders TensorGrad useless
    TensorGrad &_unsafe_nullify();
    void ensure_usable() const;



  public:
    
    inline const DType &dtype() const { return detach().dtype(); }
    inline Tensor &detach() { return this->Node->tensor->tensor; }
    inline const Tensor &detach() const { return this->Node->tensor->tensor; }
    inline Tensor &grad() { utils::THROW_EXCEPTION(this->track_grad(), "Error: Grad not being tracked, cannot access grad"); return this->Node->grad->tensor;}
    inline const Tensor &grad() const { utils::THROW_EXCEPTION(this->track_grad(), "Error: Grad not being tracked, cannot access grad"); return this->Node->grad->tensor;}


    explicit TensorGrad(Scalar value, bool grad_required = true);
    explicit TensorGrad(const Tensor &, bool grad_required = true);
    explicit TensorGrad(Tensor &&t, bool grad_required = true);
    explicit TensorGrad(std::nullptr_t, bool grad_required = true);
    TensorGrad(TensorGrad &&tg);
    TensorGrad(const TensorGrad &tg);
    TensorGrad &operator=(const TensorGrad &tg);
    TensorGrad &operator=(TensorGrad &&tg);
    TensorGrad &operator=(Scalar s);
    TensorGrad &operator=(const Tensor &t) { return set_(t); }
    TensorGrad &set_(const Tensor &t);
    // Cannot track gradient with the following opperation
    // Therefore user must do t.detach() << instead of t << 
    // inline CommaOperator operator<<(Scalar s) { return detach() << s; }
    inline const DeviceType &device() const noexcept { return detach().device(); }
    void swap(TensorGrad &);
    // Addition operation
    TensorGrad operator+(const TensorGrad &other) const;
    TensorGrad operator+(const Scalar other) const;
    TensorGrad operator+(const Tensor &other) const;
    friend TensorGrad operator+(const Tensor &, const TensorGrad &);
    friend TensorGrad operator+(const Scalar, const TensorGrad &);
    // This Addition operation
    TensorGrad &operator+=(const TensorGrad &other);
    TensorGrad &operator+=(const Scalar other);
    TensorGrad &operator+=(const Tensor &);
    // Subtraction operation
    TensorGrad operator-(const TensorGrad &other) const;
    TensorGrad operator-(const Scalar other) const;
    TensorGrad operator-(const Tensor &other) const;
    friend TensorGrad operator-(const Tensor &, const TensorGrad &);
    friend TensorGrad operator-(const Scalar, const TensorGrad &);
    // This Subtraction operation
    TensorGrad &operator-=(const TensorGrad &other);
    TensorGrad &operator-=(const Tensor &other);
    TensorGrad &operator-=(const Scalar other);
    // Division operation
    TensorGrad operator/(const TensorGrad &other) const;
    TensorGrad operator/(const Scalar other) const;
    TensorGrad operator/(const Tensor &other) const;
    friend TensorGrad operator/(const Tensor &, const TensorGrad &);
    friend TensorGrad operator/(const Scalar, const TensorGrad &);
    // This division operation
    TensorGrad &operator/=(const TensorGrad &other);
    TensorGrad &operator/=(const Tensor &other);
    TensorGrad &operator/=(const Scalar other);
    // Multiplication operation
    TensorGrad operator*(const TensorGrad &other) const;
    TensorGrad operator*(const Scalar other) const;
    TensorGrad operator*(const Tensor &other) const;
    friend TensorGrad operator*(const Tensor &, const TensorGrad &);
    friend TensorGrad operator*(const Scalar, const TensorGrad &);
    // This multiplication operation
    TensorGrad &operator*=(const Tensor &other);
    TensorGrad &operator*=(const TensorGrad &other);
    TensorGrad &operator*=(const Scalar other);
    inline const nt::SizeRef &shape() const { return detach().shape(); }
    inline const size_t dims() const { return detach().dims(); }
    inline const size_value_t &numel() const { return detach().numel(); }
    template <typename... Args>
    inline TensorGrad view(int64_t i, Args &&...args) const {
        TensorGrad result(detach().view(i, args...));
        result.track_grad(*this, [i, args...](nt::Tensor &grad) {
            return grad.view(i, args...);
        });
        return result;
    }
    TensorGrad view(SizeRef s) const;
    TensorGrad operator[](size_value_t) const;
    TensorGrad operator[](std::vector<size_value_t>) const;
    TensorGrad operator[](range_) const;
    TensorGrad operator[](Tensor) const;
    TensorGrad operator[](std::vector<range_>) const;
    template <typename Arg, typename... Args>
    inline TensorGrad operator()(Arg &&arg, Args &&...args) {
        auto vec = utils::collect_integers_or_ranges<size_value_t>(
            std::forward<Arg>(arg), std::forward<Args>(args)...);
        return (*this)[vec];
    }


    inline TensorGrad &operator++() { return *this += 1; }

    inline Tensor operator>=(const TensorGrad &t) const {
        return this->detach() >= t.detach();
    }
    inline Tensor operator<=(const TensorGrad &t) const {
        return this->detach() <= t.detach();
    }
    inline Tensor operator==(const TensorGrad &t) const {
        return this->detach() == t.detach();
    }
    inline Tensor operator>=(const Tensor &t) const {
        return this->detach() >= t;
    }
    inline Tensor operator<=(const Tensor &t) const {
        return this->detach() <= t;
    }
    inline Tensor operator==(const Tensor &t) const {
        return this->detach() == t;
    }
    inline Tensor operator>=(Scalar s) const { return this->detach() >= s; }
    inline Tensor operator<=(Scalar s) const { return this->detach() <= s; }
    inline Tensor operator==(Scalar s) const { return this->detach() == s; }
    inline Tensor operator!=(Scalar s) const { return this->detach() != s; }
    /* inline Tensor operator&&(Tensor t) const            {return this->detach()
     * && t;} */
    /* inline Tensor operator||(Tensor t) const            {return this->detach()
     * || t;} */

    inline Tensor operator>(const TensorGrad &t) const {
        return this->detach() > t.detach();
    }
    inline Tensor operator<(const TensorGrad &t) const {
        return this->detach() < t.detach();
    }
    inline Tensor operator>(const Tensor &t) const { return this->detach() > t; }
    inline Tensor operator<(const Tensor &t) const { return this->detach() < t; }
    inline Tensor operator>(Scalar s) const { return this->detach() > s; }
    inline Tensor operator<(Scalar s) const { return this->detach() < s; }
    inline TensorGrad operator-() const { return *this * -1; }

    inline TensorGrad &fill_(Scalar s) { return *this = s; }
    inline TensorGrad &fill_(const TensorGrad &val) { return *this = val; }
    inline TensorGrad &fill_(const Tensor &val) { return *this = val; }
    inline TensorGrad &add_(Scalar val) { return *this += val; }
    inline TensorGrad &add_(const TensorGrad &val) { return *this += val; }
    inline TensorGrad &add_(const Tensor &val) { return *this += val; }
    inline TensorGrad &subtract_(Scalar val) { return *this -= val; }
    inline TensorGrad &subtract_(const TensorGrad &val) { return *this -= val; }
    inline TensorGrad &subtract_(const Tensor &val) { return *this -= val; }
    inline TensorGrad &multiply_(Scalar val) { return *this *= val; }
    inline TensorGrad &multiply_(const TensorGrad &val) { return *this *= val; }
    inline TensorGrad &multiply_(const Tensor &val) { return *this *= val; }
    inline TensorGrad &divide_(Scalar val) { return *this /= val; }
    inline TensorGrad &divide_(const TensorGrad &val) { return *this /= val; }
    inline TensorGrad &divide_(const Tensor &val) { return *this /= val; }
    inline Scalar toScalar() const { return this->detach().toScalar(); }
    template <typename T = Scalar>
    inline std::conditional_t<std::is_same_v<T, Scalar>, Scalar, T &> item() {
        if constexpr (std::is_same_v<T, Scalar>) {
            return toScalar();
        } else {
            return this->detach().item<T>();
        }
    }
    template <typename T = Scalar>
    inline std::conditional_t<std::is_same_v<T, Scalar>, Scalar, const T &>
    item() const {
        if constexpr (std::is_same_v<T, Scalar>) {
            return toScalar();
        } else {
            return this->detach().item<T>();
        }
    }
    inline bool is_contiguous() const { return (this->detach().is_contiguous()); }
    inline bool is_empty() const { return (this->Node == nullptr || this->Node->tensor == nullptr || this->Node->tensor->tensor.is_empty()); }
    inline bool is_null() const { return (this->Node == nullptr || this->Node->tensor == nullptr || this->Node->tensor->tensor.is_null()); }
    inline int64_t contig_count() const { return this->detach().contig_count(); }
    inline std::vector<size_value_t> strides() const {
        return this->detach().strides();
    }
    inline std::vector<size_value_t> getChangedStrides() const {
        return this->detach().getChangedStrides();
    }

    inline void print() const { this->detach().print(); }
    inline void *data_ptr() { return this->detach().data_ptr(); }
    inline const void *data_ptr() const { return this->detach().data_ptr(); }
    inline void *data_ptr_end() { return this->detach().data_ptr_end(); }
    inline const void *data_ptr_end() const {
        return this->detach().data_ptr_end();
    }
    inline bool occupy_same_tensor_memory(const TensorGrad &tg) const noexcept {
        return this->detach().occupy_same_memory(tg.detach());
    }
    friend std::ostream &operator<<(std::ostream &out, const TensorGrad &);

    TensorGrad unsqueeze(size_value_t dim = 0) const;
    TensorGrad unsqueeze_as(const Tensor &) const;
    TensorGrad unsqueeze_as(const SizeRef &) const;
    TensorGrad squeeze(utils::optional_list list = nullptr) const;
    TensorGrad flatten(size_value_t, size_value_t) const;
    TensorGrad unflatten(size_value_t, size_value_t) const;
    TensorGrad permute(std::vector<size_value_t>) const;
    TensorGrad transpose(size_value_t, size_value_t) const;
    TensorGrad unfold(size_value_t dim, size_value_t size,
                      size_value_t step) const;
    TensorGrad split_axis(std::vector<range_>) const;
    TensorGrad split_axis(size_value_t) const;
    TensorGrad split_axis_1() const;
    TensorGrad div(size_value_t) const;
    TensorGrad real() const;
    TensorGrad imag() const;
    TensorGrad to_complex_from_real() const;
    TensorGrad to_complex_from_imag() const;
    TensorGrad sum(utils::optional_list list = nullptr,
                   bool keepdim = false) const;
    TensorGrad mean(utils::optional_list list = nullptr,
                    bool keepdim = false) const;
    result_types::max<TensorGrad, Tensor>
    max(utils::optional_list dim = nullptr, bool keepdim = false) const;
    result_types::max<TensorGrad, Tensor>
    min(utils::optional_list dim = nullptr, bool keepdim = false) const;
    TensorGrad exp() const;
    TensorGrad &exp_();
    TensorGrad pow(Scalar) const;
    TensorGrad &inverse_();
    TensorGrad inverse() const;
    TensorGrad clip(Scalar, Scalar) const;
    TensorGrad &clip_(Scalar, Scalar);
    TensorGrad pad(std::vector<size_value_t> p, const char *mode = "constant",
                   Scalar value = 0.0) const;
    TensorGrad unpad(std::vector<size_value_t> p) const;
    TensorGrad flip(utils::optional_list list = nullptr) const;
    TensorGrad flip_view(utils::optional_list list = nullptr) const;
    TensorGrad dilate(std::vector<size_value_t>) const;
    template<typename... Args>
    inline TensorGrad dilate(size_value_t i, Args&&... args) const {
        std::vector<size_value_t> vec;
        vec.reserve(sizeof...(Args) + 1);
        utils::collect_integers_impl(vec, i, std::forward<Args>(args)...);
        return this->dilate(std::move(vec)); 
    }
    TensorGrad undilate(std::vector<size_value_t>) const;
    template<typename... Args>
    inline TensorGrad undilate(size_value_t i, Args&&... args) const {
        std::vector<size_value_t> vec;
        vec.reserve(sizeof...(Args) + 1);
        utils::collect_integers_impl(vec, i, std::forward<Args>(args)...);
        return this->undilate(std::move(vec)); 
    }
    TensorGrad undilate_(std::vector<size_value_t>) const;

    template<typename... Args>
    inline TensorGrad undilate_(size_value_t i, Args&&... args) const {
        std::vector<size_value_t> vec;
        vec.reserve(sizeof...(Args) + 1);
        utils::collect_integers_impl(vec, i, std::forward<Args>(args)...);
        return this->undilate_(std::move(vec)); 
    }
    TensorGrad repeat_(size_value_t amt) const;
    TensorGrad repeat_(size_value_t dim, size_value_t amt) const;
    TensorGrad expand(SizeRef) const;
    TensorGrad to_dtype(DType) const;
    TensorGrad to_device(DeviceType) const;
    inline TensorGrad to(DType dt) const { return to_dtype(dt); }
    inline TensorGrad to(DeviceType dt) const { return to_device(dt); }
    TensorGrad clone() const;
    TensorGrad contiguous() const;

    // still need to implemenet:
    // currently none

    inline TensorGrad unsqueeze_as(const TensorGrad &tg) const {
        return this->unsqueeze_as(tg.detach());
    }
    inline TensorGrad expand_as(const Tensor &t) const {
        return this->expand(t.shape());
    }
    inline TensorGrad expand_as(const TensorGrad &tg) const {
        return this->expand(tg.shape());
    }

    inline static TensorGrad makeNullTensorArray(int64_t num) {
        return TensorGrad(Tensor::makeNullTensorArray(num));
    }
    
    // To be done potentially before backward is called:
    void ensure_grads_initialized();
    void zero_grad();
    void zero_self_grad_only();
    

    // below is a function to extract the autograd
    //      Note: While the function could be const,
    //            It is not because all uses it can do should not be
    // Validate makes sure or defines a gradient and 
    // backward function for every Node in the graph
    // Validate does not beed to be true when zeroing the gradients for example
    //      because this is done automatically
    // You can look at AutoGrad.h for more detail
    //      But quickly it does things like:
    //          - Automatically builds a backward path tree for which tensors to have the backward pass called
    //          - Can call the backward pass
    //          - Zero all tracked gradients
    grad::AutoGrad<std::unordered_set<intrusive_ptr<grad::utility::GraphNode>>> get_auto_grad(bool validate = true);
    void backward();
    void backward(const Tensor &);
    void accumulate_gradient(const Tensor &);
    void accumulate_gradient(Scalar);
    inline std::string backward_name() const {
        return (this->Node->backwardFunc != nullptr) ? this->Node->backwardFunc->get_name() : "None";
    }


    // To be done after backward is called:
    void update();         // updates current values based on gradient
    void update_mutable(); // updates current values based on gradient even if
                           // the tensor is immutable
    void _erase_graph();   // should only be called internally 

    static NEUROTENSOR_API void redefine_tracking(
        TensorGrad &, const TensorGrad &,
        std::function<void(const Tensor &, intrusive_ptr<TensorGrad> &)>,
        const char *func_name = __NT_FUNCTION_NAME__);
    // the underlying tensor, the function to update the gradient, the parents
    template <typename... Args>
    static TensorGrad make_tensor_grad(
        Tensor &,
        std::function<void(const Tensor &,
                           std::vector<intrusive_ptr<TensorGrad>> &)>,
        const TensorGrad &, Args &&...);                                           // TensorGrad.hpp
    template <typename OGFunc>
    static TensorGrad make_view_grad(Tensor &, const TensorGrad &, OGFunc &&);     // TensorGrad.hpp
};

Tensor &operator+=(Tensor &, const TensorGrad &);
Tensor &operator-=(Tensor &, const TensorGrad &);
Tensor &operator*=(Tensor &, const TensorGrad &);
Tensor &operator/=(Tensor &, const TensorGrad &);

} // namespace nt

#include "../functional/tensorgrad_get.h"
#include "TensorGrad.hpp"
#endif //_NT_TENSOR_GRAD_H_
