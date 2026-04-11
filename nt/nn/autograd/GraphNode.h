/*
 * This is a single node which contains the tensor, gradient, backward function, storage_ids and versioning
 * This is able to keep track of versions (in relation to writes) and hold all the information
 *  a tensorgrad may need
 *
 *  Currently, it's construction is pretty limited to very set rules and functions below
 *      This is by design, This will construct the creation to this design:
 *
 *      gradient is Tensor::Null() until initialized
 *          - gradient is nullptr when not being tracked
 *      backward_func is None as name and nullptr for function until initialized
 *          - backward_func is nullptr when not being tracked
 *      tensor is required at all instances
 *      version_ is required when tracking the gradient
 *          - version_ (a shared version) is nullptr when gradient isn't being tracked
 *      - then there are internal indexes to aid in performance/version tracking
 */

#ifndef NT_NN_AUTOGRAD_GRAPH_NODE_H__
#define NT_NN_AUTOGRAD_GRAPH_NODE_H__

namespace nt::grad::utility {
class GraphNode;
}

#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../../Tensor.h"
// #include "../../utils/tensor_holder.h"
#include "../../utils/type_traits.h"
#include "../../functional/functional.h"
#include "BackwardFunc.h"
#include "SharedVersion.h"
#include <vector>
#include <memory>
#include <functional>
#include <array>

namespace nt::grad::utility {

class GraphNode;
intrusive_ptr<GraphNode> makeNonTrackingGraphNode(const Tensor&);
intrusive_ptr<GraphNode> makeTrackingGraphNode(const Tensor&);

template<typename... Grads>
intrusive_ptr<GraphNode> makeReadGraphNode(
        const Tensor& output,
        Grads&&... grads);
template<typename... Grads>
intrusive_ptr<GraphNode>& makeReadGraphNode(
        intrusive_ptr<GraphNode>&,
        Grads&&... grads);
intrusive_ptr<GraphNode> makeReadGraphNode(
        const Tensor&,
        std::vector<intrusive_ptr<GraphNode>>&);
template<typename... Grads>
intrusive_ptr<GraphNode> makeWriteGraphNode(
        intrusive_ptr<GraphNode> current,
        Grads&&... grads);
template<typename BackFunc, typename... Grads>
inline intrusive_ptr<GraphNode> makeWriteGraphNode(
        BackFunc&& func, const char* name,
        intrusive_ptr<GraphNode> current, 
        Grads&&... other_grads);
intrusive_ptr<GraphNode> makeViewGraphNode(
        const Tensor& view, intrusive_ptr<GraphNode> current);


intrusive_ptr<GraphNode> get_current_grad(const intrusive_ptr<GraphNode>&) noexcept;


class NEUROTENSOR_API GraphNode : public intrusive_ptr_target {
    public:
    // Graph Tracking:
    std::vector<intrusive_ptr<GraphNode>> children;
    std::vector<std::pair<weak_intrusive_ptr<GraphNode>, uint64_t> > parents; // graph node and the current max version of that node
    
    private:

    // the tensor and the gradient are unique_ptr's because as of now, there is no reason
    // they should not be uniquely attached to their node
    // (subject to future change if absolutely necessary)
    // I would advise future versions to keep it this way for general saftey and to ensure
    // each node is holding exactly what it's other versions are, and there isn't sharing of the same grad/tensor
    std::unique_ptr<Tensor> tensor, grad;
    std::unique_ptr<backward_func> backwardFunc;

    // Version Tracking:
    intrusive_ptr<shared_version> version_;
    uint64_t snapshot_version_ = 0;
    int64_t current_grad_index_ = -1; // the latest write of same size
    
    GraphNode(std::unique_ptr<Tensor> tensor_,
            std::unique_ptr<Tensor> grad_,
            std::unique_ptr<backward_func> backwardFunc_,
            intrusive_ptr<shared_version> version,
            uint64_t snapshot_version,
            int64_t current_grad_index)
        :tensor(std::move(tensor_)), grad(std::move(grad_)),
        backwardFunc(std::move(backwardFunc_)),
        version_(std::move(version)),
        snapshot_version_(snapshot_version),
        current_grad_index_(current_grad_index)
    {if(!this->tensor){this->tensor = std::make_unique<Tensor>(Tensor::Null());}}
            
    
    // friend initialization and running nodes
    friend intrusive_ptr<GraphNode> makeNonTrackingGraphNode(const Tensor&);
    friend intrusive_ptr<GraphNode> makeTrackingGraphNode(const Tensor&);


    template<class TTarget, class DeleteOp, class NullType>
    friend class ::nt::intrusive_ptr;
    friend class  backward_func;
    
    friend intrusive_ptr<GraphNode> get_current_grad(const intrusive_ptr<GraphNode>&) noexcept;
    
    template<typename... Grads>
    friend intrusive_ptr<GraphNode> makeReadGraphNode(
        const Tensor& output,
        Grads&&... grads);
    template<typename... Grads>
    friend intrusive_ptr<GraphNode>& makeReadGraphNode(
        intrusive_ptr<GraphNode>&,
        Grads&&... grads);
    friend intrusive_ptr<GraphNode> makeReadGraphNode(
        const Tensor&,
        std::vector<intrusive_ptr<GraphNode>>&);
    template<typename... Grads>
    friend intrusive_ptr<GraphNode> makeWriteGraphNode(
        intrusive_ptr<GraphNode> current,
        Grads&&... grads);
    template<typename BackFunc, typename... Grads>
    friend intrusive_ptr<GraphNode> makeWriteGraphNode(
        BackFunc&& func, const char* name,
        intrusive_ptr<GraphNode> current,
        Grads&&... grads);
    friend intrusive_ptr<GraphNode> makeViewGraphNode(
            const Tensor&, intrusive_ptr<GraphNode> current);

    public:

    GraphNode() = delete;

    
    inline uint64_t storage_id() const noexcept { return bool(tensor) ? tensor->individual_storage_id() : 0; }
    inline uint64_t current_version() const noexcept { return version_->load(); } 
    inline const uint64_t& version_snapshot() const noexcept { return snapshot_version_; } 
    inline uint64_t increment_version() { return this->version_->increment_version(); }
    
    inline void release_resources() override { children.clear(); }
    
    void accumulate_gradient(const Tensor& in_grad);
    void accumulate_gradient(Scalar num);
    void zero_grad();
    void run_backward();
    void run_backward(const Tensor& _grad);
    inline const Tensor& detach() const noexcept { return *(this->tensor); }
    inline Tensor& detach() noexcept { return *(this->tensor); }
    inline const Tensor& gradient() const {
        utils::throw_exception(bool(this->grad),
                "Error, trying to get undefined gradient"
        );
        return *(this->grad);
    }
    inline Tensor& gradient() {
        utils::throw_exception(bool(this->grad),
                "Error, trying to get undefined gradient"
        );
        return *(this->grad);
    }
    inline const std::unique_ptr<Tensor>& raw_gradient() const noexcept { return this->grad; }
    inline std::unique_ptr<Tensor>& raw_gradient() noexcept { return this->grad; }
    
    // used in performance critical paths
    // only to be used when you know that the gradient
    // is defined (like after calling accumulate_gradient)
    inline const Tensor& gradient_() const noexcept { return *(this->grad); }
    inline Tensor& gradient_() { return *(this->grad); }
    
    inline void set_name(std::string name_) noexcept {
        if(!this->backwardFunc){
            this->backwardFunc = make_backward_func(EdgeType::Read, nullptr, std::move(name_));
            return;
        }
        this->backwardFunc->set_name(std::move(name_));
    }

    inline void set_func(backward_func_type func) noexcept {
        if(!this->backwardFunc){
            this->backwardFunc = make_backward_func(EdgeType::Read, func, "Unknown");
            return;
        }
        this->backwardFunc->set(std::move(func));
    }

    inline std::string name() const noexcept {
        return bool(this->backwardFunc) ?
                this->backwardFunc->get_name() :
                "NoGradTracked";
    }
    inline EdgeType edge_type() const {
        return bool(this->backwardFunc) ?
                this->backwardFunc->edge_type() :
                EdgeType::None;
    }

    inline bool is_tracking_grad() noexcept { return bool(grad); }
    inline void stop_grad_tracking() noexcept {
        this->grad = nullptr;
        this->backwardFunc = nullptr;
        // this->version_ = nullptr;
        this->snapshot_version_ = 0;
        this->current_grad_index_ = -1;
    }
    inline void start_grad_tracking() noexcept {
        this->ensure_initialization(true);
        if(!this->version_){
            this->version_ = make_intrusive<shared_version>();
            this->snapshot_version_ = 0;
            this->current_grad_index_ = -1;
        }
    }
    
    inline void ensure_initialization(bool zero_if_uninit = false){
        if(!this->tensor){
            this->tensor = std::make_unique<Tensor>(Tensor::Null());
            zero_if_uninit = false;
        }
        this->ensure_backward_initialization(
                zero_if_uninit && 
                !this->tensor->is_null());
    }

    // backward refers to the gradient and the backward function
    inline void ensure_backward_initialization(bool zero_if_uninit = false ) {
        //this is a function that can be used to make sure grad and backwardFunc are not nullptr
        if(!this->grad) 
            this->grad = std::make_unique<Tensor>(
                    zero_if_uninit ? ::nt::functional::zeros_like(*(this->tensor)) : 
                    Tensor::Null()
            );
         if(!this->backwardFunc) this->backwardFunc = make_backward_func(EdgeType::Read, nullptr, "None");
    }

    // void ensure_view_backward_initialization(bool zero_if_uninit = false );
    // void ensure_self_mod_backward_initialization(bool zero_if_uninit = false );
    // This function is to make sure the gradient is initialized
    // So, if !bool(this->grad) it will be initialized, or if this->grad->tensor.is_null()
    // it will also be initialized
    inline void ensure_gradient_init() {
        if(!bool(this->grad))
            this->grad = std::make_unique<Tensor>(nt::functional::zeros_like(*(this->tensor)));
        else if (this->grad->is_null())
            *(this->grad) = nt::functional::zeros_like(*(this->tensor));
    }

};

// Where this could really backfire:
//  - Storage id (which versions are tied to) is tracked internally inside of each Tensor
//  - However, if there is a detachment and that tensor is used to make another graph node
//      which is then put into the same graph, with brand new versions, this could (and would) entirely
//      screw up version tracking, so something like this would break it:
//
//
//  TensorGrad a; <- imaging already initialized
//  Tensor a_ = a.detach();
//  TensorGrad b(a_);
//  a += b; <- now there are multiple versions of the same tensor for the same storage id
//
//  NOTE:
//  For now, this is not handled, will probably handle it later
//      (known and currently unhandled potential if misused bug)
//
//  Fix:
//      put versioning at storage level id as well (this feels messy and re-writy given current state)

inline intrusive_ptr<GraphNode> makeNonTrackingGraphNode(const Tensor& tensor_){
    std::unique_ptr<Tensor> tensor = std::make_unique<Tensor>(tensor_);
    return make_intrusive<GraphNode>(
            std::move(tensor), /*grad = */ nullptr,
            /*backwardFunc = */ nullptr,
            /*version (shared) = */ nullptr,
            /*version (snapshot) = */ 0,
            /*current_grad_index = */ -1
    );
}

inline intrusive_ptr<GraphNode> makeTrackingGraphNode(const Tensor& tensor_){
    std::unique_ptr<Tensor> tensor = std::make_unique<Tensor>(tensor_);
    std::unique_ptr<Tensor> grad = std::make_unique<Tensor>(Tensor::Null());
    return make_intrusive<GraphNode>(
            std::move(tensor), std::move(grad),
            make_backward_func(EdgeType::Read, nullptr, "None"),
            /*version (shared) = */ make_intrusive<shared_version>(),
            /*version (snapshot) = */ 0,
            /*current_grad_index = */ -1
    );
}

inline intrusive_ptr<GraphNode> get_current_grad(const intrusive_ptr<GraphNode>& node) noexcept {
    return node->current_grad_index_ == -1 ? node : get_current_grad(node->children[node->current_grad_index_]); 
}

template<typename... Grads>
inline intrusive_ptr<GraphNode> makeReadGraphNode(
        const Tensor& output,
        Grads&&... grads){
    intrusive_ptr<GraphNode> node = makeTrackingGraphNode(output);
    makeReadGraphNode(node, std::forward<Grads&&>(grads)...);
    return std::move(node);
}
template<typename... Grads>
inline intrusive_ptr<GraphNode>& makeReadGraphNode(
        intrusive_ptr<GraphNode>& node,
        Grads&&... grads){
    // To execute a function for each argument: 
    ([&](auto& arg) { 
        static_assert(type_traits::is_same_v<type_traits::remove_cvref_t<decltype(arg)>, intrusive_ptr<GraphNode>>); 
        node->parents.emplace_back(get_current_grad(arg), arg->current_version()); 
        arg->children.push_back(node); 
    }(grads), ...); 
    return node; 

}

inline intrusive_ptr<GraphNode> makeReadGraphNode(
        const Tensor& output,
        std::vector<intrusive_ptr<GraphNode>>& parents){
    // To execute a function for each argument: 
    intrusive_ptr<GraphNode> node = makeTrackingGraphNode(output);
    for(auto& arg : parents){
        node->parents.emplace_back(get_current_grad(arg), arg->current_version());
        arg->children.push_back(node);
    }
    return std::move(node);
}

template<typename... Grads>
inline intrusive_ptr<GraphNode> makeWriteGraphNode(
        intrusive_ptr<GraphNode> current, 
        Grads&&... other_grads) {
    // so, the reason this is made before the new_self is made is the case where you have something
    // like: "a += a"
    // basically, what that will do is make it so that when dependencies are looked at after the new_self is
    // added to children, it will then return new_self
    // This is basically just really important when there is a write node, where the node that is writing to itself
    //  is also involved in the reading of itself (very specific edge case, but can come up)
    std::array<intrusive_ptr<GraphNode>, sizeof...(other_grads)> other_dependencies;
    std::array<uint64_t, sizeof...(other_grads)> other_versions;
    std::atomic<size_t> index = 0;
    ([&](auto& arg) { 
        static_assert(type_traits::is_same_v<type_traits::remove_cvref_t<decltype(arg)>, intrusive_ptr<GraphNode>>);
        other_dependencies[index] = get_current_grad(arg);
        other_versions[index] = arg->current_version();
        ++index;
    }(other_grads), ...);
    intrusive_ptr<GraphNode> self = get_current_grad(current); 
    uint64_t prev_version = self->increment_version(); // mark that a new version has been made

    intrusive_ptr<GraphNode> new_self = 
        make_intrusive<GraphNode>(
            std::make_unique<Tensor>(*(self->tensor)), 
            std::make_unique<Tensor>(Tensor::Null()),
            make_backward_func(EdgeType::Write, nullptr, "None"),
            /*version (shared) = */ self->version_,
            /*version (snapshot) = */ self->current_version(),
            /*current_grad_index = */ -1
    );
    new_self->parents.emplace_back(self, prev_version);
    self->children.emplace_back(new_self);
    self->current_grad_index_ = static_cast<int64_t>(self->children.size()-1);

    for(size_t i = 0; i < sizeof...(other_grads); ++i){
        new_self->parents.emplace_back(other_dependencies[i], other_versions[i]);
        other_dependencies[i]->children.emplace_back(new_self);
    }
    // if(new_self->children.size() > 0){
    //     std::cout << "POTENTIAL ERROR: NEW SELF CHILDREN SIZE IS " << new_self->children.size() << " FOR " << name<< std::endl;
    // }
    return new_self; 
}


template<typename BackFunc, typename... Grads>
inline intrusive_ptr<GraphNode> makeWriteGraphNode(
        BackFunc&& func, const char* name,
        intrusive_ptr<GraphNode> current, 
        Grads&&... other_grads) {
    // so, the reason this is made before the new_self is made is the case where you have something
    // like: "a += a"
    // basically, what that will do is make it so that when dependencies are looked at after the new_self is
    // added to children, it will then return new_self
    // This is basically just really important when there is a write node, where the node that is writing to itself
    //  is also involved in the reading of itself (very specific edge case, but can come up)
    std::array<intrusive_ptr<GraphNode>, sizeof...(other_grads)> other_dependencies;
    std::array<uint64_t, sizeof...(other_grads)> other_versions;
    std::atomic<size_t> index = 0;
    ([&](auto& arg) { 
        static_assert(type_traits::is_same_v<type_traits::remove_cvref_t<decltype(arg)>, intrusive_ptr<GraphNode>>);
        other_dependencies[index] = get_current_grad(arg);
        other_versions[index] = arg->current_version();
        ++index;
    }(other_grads), ...);
    intrusive_ptr<GraphNode> self = get_current_grad(current); 
    uint64_t prev_version = self->increment_version(); // mark that a new version has been made

    intrusive_ptr<GraphNode> new_self = 
        make_intrusive<GraphNode>(
            std::make_unique<Tensor>(*(self->tensor)), 
            std::make_unique<Tensor>(Tensor::Null()),
            make_backward_func(EdgeType::Write, std::forward<BackFunc>(func), name),
            /*version (shared) = */ self->version_,
            /*version (snapshot) = */ self->current_version(),
            /*current_grad_index = */ -1
    );
    new_self->parents.emplace_back(self, prev_version);
    self->children.emplace_back(new_self);
    self->current_grad_index_ = static_cast<int64_t>(self->children.size()-1);

    for(size_t i = 0; i < sizeof...(other_grads); ++i){
        new_self->parents.emplace_back(other_dependencies[i], other_versions[i]);
        other_dependencies[i]->children.emplace_back(new_self);
    }
    // if(new_self->children.size() > 0){
    //     std::cout << "POTENTIAL ERROR: NEW SELF CHILDREN SIZE IS " << new_self->children.size() << " FOR " << name<< std::endl;
    // }
    return new_self; 
}

inline intrusive_ptr<GraphNode> makeViewGraphNode(const Tensor& view, intrusive_ptr<GraphNode> current){
    intrusive_ptr<GraphNode> self = get_current_grad(current); 
    uint64_t prev_version = self->increment_version(); // mark that a new version has been created 
    intrusive_ptr<GraphNode> new_self = 
        make_intrusive<GraphNode>(
            std::make_unique<Tensor>(view), 
            std::make_unique<Tensor>(Tensor::Null()),
            make_backward_func(EdgeType::View, nullptr, "None"),
            /*version (shared) = */   self->version_,
            /*version (snapshot) = */ self->current_version(),
            /*current_grad_index = */ -1
    );
 

    new_self->parents.emplace_back(self, prev_version); 
    self->children.emplace_back(new_self); 
    return new_self; 
}

}


#endif
