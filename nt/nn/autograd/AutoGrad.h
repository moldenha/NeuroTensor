#ifndef NT_NN_AUTOGRAD_AUTO_GRAD_H__
#define NT_NN_AUTOGRAD_AUTO_GRAD_H__

#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../../Tensor.h"
#include "../../utils/unique_vector.hpp"
#include "GraphNode.h"
#include "BackwardFunc.h"

#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <utility>

namespace nt::grad{

template<class HashSet = std::unordered_set<intrusive_ptr<utility::GraphNode>> >
class AutoGrad : public intrusive_ptr_target{
    public:
    using node_type       = intrusive_ptr<utility::GraphNode>;
    using unique_vec_type = intrusive_ptr<unique_vector<node_type, HashSet>>;
    using list_type       = std::vector<node_type>;
    
    private:
 
    unique_vec_type tape;
    list_type       list;
    node_type       start_node;
    // this is meant to determine 2 things:
    //  a: has the list been validated
    //      if not -> validated_list == -1
    //      otherwise -> amount of non-dependent nodes in list
    //  b: faster set-node
    //      if validated_list is not -1, then that represents
    //      the number of independent nodes
    //      so it can easily be traversed, and make start_node
    //      at index 0 if it exists
    int64_t validated_list = -1;

    void collect_tape(); // makes sure all nodes are stored in tape
    void to_list_(); // internally stores the list variable
    public:
        AutoGrad()
            :tape(make_intrusive<typename unique_vec_type::target_type>()),
            start_node(nullptr),
            validated_list(-1)
        {}
        AutoGrad(const node_type& node)
            :tape(make_intrusive<typename unique_vec_type::target_type>()), 
            start_node(node),
            validated_list(-1)
        {}
        AutoGrad(AutoGrad&& other)
            :tape(std::move(other.tape)), list(std::move(other.list)),
            start_node(std::move(other.start_node)),
            validated_list(std::exchange(other.validated_list, -1))
        {}
        AutoGrad(const AutoGrad& other)
            :tape(other.tape), list(other.list),
            start_node(other.start_node),
            validated_list(other.validated_list)

        {}

        inline AutoGrad& operator=(AutoGrad&& other){
            this->tape = std::move(other.tape);
            this->list = std::move(other.list);
            this->start_node = std::move(other.start_node);
            this->validated_list = std::exchange(other.validated_list, -1);

            return *this;
        }

        inline AutoGrad& operator=(const AutoGrad& other){
            this->tape = other.tape;
            this->list = other.list;
            this->start_node = other.start_node;
            this->validated_list = other.validated_list;

            return *this;
        }

        inline list_type& to_list() {
            if(this->validated_list < 0)
                this->to_list_();
            return this->list;
        }
        inline void add_node(const node_type& node){
            this->tape->push_back(node);
            this->validated_list = -1;
        }
        // inline void merge(const AutoGrad& grad){
        //     this->tape.merge(grad.tape);
        //     this->validated_list = -1;
        // }
        inline void merge(intrusive_ptr<AutoGrad>& grad){
            unique_vec_type tape = (this->tape->size() > grad->tape->size() ? this->tape : grad->tape);
            tape->merge(this->tape->size() > grad->tape->size() ? *grad->tape : *this->tape);
            this->tape = tape;
            grad->tape = tape;
            this->validated_list = -1;
            grad->validated_list = -1;
            this->list.clear();
            grad->list.clear();
        }
        inline void erase() {
            this->tape->clear();
            this->list.clear();
            this->start_node.reset();
        }

        void backward(const Tensor&);
        void backward();
        void zero_grad();
        void set_start_node(const node_type& node);

};


}



#endif
