#include "AutoGrad.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../../Tensor.h"
#include "../../utils/unique_vector.hpp"
#include "GraphNode.h"
#include "BackwardFunc.h"

#include <unordered_set>
#include <unordered_map>
#include <deque>
#include <map>

namespace nt::grad {


namespace details{

template<typename HashSet, typename NodeType>
void tape_collector(
        const NodeType& node, 
        HashSet& visited,
        intrusive_ptr<unique_vector<NodeType, HashSet>>& vec){
    if(!visited.insert(node).second) return;
    vec->push_back(node);
    for(const auto& child : node->children)
        tape_collector(child, visited, vec);
    for(const auto& [weak_parent, _] : node->parents){
        if(auto parent = weak_parent.lock()) tape_collector(parent, visited, vec);
    }


}



/*
 * This is the functionality behind the algorithm that
 *  topologically tracks each node and it's dependencies
 *  properly with versioning
 */

template<typename NodeType>
void handle_child_increment(const NodeType& grad,
                            const std::unordered_map<uint64_t, std::map<uint64_t, NodeType>>& version_map,
                            std::unordered_map<NodeType, int32_t>& child_count){
    for(const auto& [weak_parent, version_cntr] : grad->parents){
        if(auto parent = weak_parent.lock()){
            auto it = child_count.find(parent);
            if(it == child_count.end()) continue;
            ++(it->second);
            if(version_cntr == parent->version_snapshot()) continue;
            auto v_map = version_map.find(parent->storage_id());
            if(v_map == version_map.end()) continue;
            for(const auto& [version, node] : v_map->second){
                if(node == grad) continue; // depending on self would have really bad result
                if(!(version <= version_cntr && version > parent->version_snapshot())) continue;
                auto sub_it = child_count.find(node);
                if(sub_it == child_count.end()) continue;
                ++(sub_it->second);
            }
        }
    }
}


template<typename NodeType>
void handle_child_decrement(const NodeType& grad,
                            const std::unordered_map<uint64_t, std::map<uint64_t, NodeType>>& version_map,
                            std::unordered_map<NodeType, int32_t>& child_count,
                            std::deque<NodeType>& queue){
    for(const auto& [weak_parent, version_cntr] : grad->parents){
        if(auto parent = weak_parent.lock()){
            auto it = child_count.find(parent);
            if(it == child_count.end()) continue;
            --(it->second);
            if(it->second == 0) queue.push_back(parent);
            if(version_cntr == parent->version_snapshot()) continue;
            auto v_map = version_map.find(parent->storage_id());
            if(v_map == version_map.end()) continue;
            for(const auto& [version, node] : v_map->second){
                if(node == grad) continue; // depending on self would have really bad result
                if(!(version <= version_cntr && version > parent->version_snapshot())) continue;
                auto sub_it = child_count.find(node);
                if(sub_it == child_count.end()) continue;
                --(sub_it->second);
                if(sub_it->second == 0) queue.push_back(node);
            }
        }
    }
}


}

template<typename HashSet>
void AutoGrad<HashSet>::collect_tape(){
    this->validated_list = false;
    HashSet visited;
    typename unique_vec_type::target_type original = *this->tape;
    this->tape->clear();
    this->list.clear();
    for(const auto& node : original)
        details::tape_collector<HashSet, node_type>(node, visited, this->tape);

}


struct BackwardValidator {
    std::unordered_set<utility::GraphNode*> executed;
    bool is_ready(utility::GraphNode* node){
        for (auto& child : node->children) {
            if (!executed.count(child.get()) && child->edge_type() == utility::EdgeType::Read) {
                return false;
            }
        }
        return true;
    }

    void check_node(utility::GraphNode* node, const std::unordered_map<nt::intrusive_ptr<utility::GraphNode>, int32_t>& child_count){
        std::string before_dep = "Node " + node->name() + " executed before dependencies!";
        if(!is_ready(node)){
            for(auto child : node->children){
                if(executed.count(child.get())) continue;
                std::cout << child->name() << " sub dependencies: " << std::endl;
                auto it = child_count.find(child);
                if(it == child_count.end()) std::cout << "child not in child count" << std::endl;
                else std::cout << "has " << it->second << " as current counter" << std::endl;
                for(auto child2 : child->children){
                    std::cout << "\t" << child2->name() << std::endl;
                }
            }
            throw std::runtime_error(before_dep);
        }
        assert(!executed.count(node) && "Node executed twice!");
        executed.insert(node);
    }
};

#ifdef NT_DEBUG_
#define NT_DEBUG_NODE_(node_name) back_validate.check_node(node_name.get(), child_count)
#else
#define NT_DEBUG_NODE_(node_name) 
#endif

// the whole point of seperating it into a function like this
// is so that the autograd doesn't have to iterate through everything
// more then once, when for example zero_grad is called on the autograd
// and then backward is called
template<typename HashSet>
void AutoGrad<HashSet>::to_list_() {
    this->collect_tape(); // clears list
    this->list.reserve(tape->size());
    this->validated_list = 0;
    std::unordered_map<node_type, int32_t> child_count;
    std::unordered_map<uint64_t, // storage_id
            std::map<uint64_t, node_type> // version -> node
        > version_map;
    for (auto& n : *this->tape) {
        version_map[n->storage_id()][n->version_snapshot()] = n;
    }


    for(auto& n : *this->tape) 
        child_count[n] = 0;  // just make it the number of children the node has
    for(auto& n : *this->tape) 
        details::handle_child_increment(n, version_map, child_count);

#ifdef NT_DEBUG_
    BackwardValidator back_validate;
#endif
    std::deque<node_type> queue;
    if(bool(this->start_node) && child_count[this->start_node] == 0){
        // if start node exists, and is ready to be executed
        NT_DEBUG_NODE_(this->start_node);
        details::handle_child_decrement(this->start_node, version_map, 
                child_count, queue);
        queue.clear();
        this->list.emplace_back(this->start_node);
        ++this->validated_list;
        for(const auto& it : child_count){
            if(it.second == 0 && it.first != this->start_node){
                queue.push_back(it.first);
                ++this->validated_list;
            }
        }
    }else{
        for(const auto& it : child_count){
            if(it.second == 0){
                queue.push_back(it.first);
                ++this->validated_list;
            }
        }
    }
 

    while(!queue.empty()){ 
        auto node = queue.front(); 
        queue.pop_front();
        NT_DEBUG_NODE_(node);
        this->list.emplace_back(node);
        details::handle_child_decrement(node, version_map, 
                child_count, queue);
    }
#ifndef NT_DEBUG_
    size_t failed_count = 0; 
    for(auto& [node, count] : child_count){ 
        if(count > 0) ++failed_count; 
    }
    if(failed_count > 0)
        throw std::runtime_error("Error, expected failed size to be 0");
#endif
    this->validated_list = true;

}

#undef NT_DEBUG_NODE_

template<typename HashSet>
void AutoGrad<HashSet>::set_start_node(const node_type& node){
    if(!bool(node)) this->start_node.reset();
    if(this->start_node == node) return; // if already set
    this->start_node = node;
    if(!this->tape->is_in(node)){
        this->add_node(node);
        return;
    }
    if(this->validated_list < 0) return;
    std::size_t index = 0;
    for(; index < this->validated_list; ++index)
        if(this->list[index] == node) break;
    std::swap(this->list[0], this->list[index]);
}

template<typename HashType>
void AutoGrad<HashType>::backward(const Tensor& initialGrad){
    list_type& traversed = this->to_list();
    utils::throw_exception(bool(this->start_node),
            "Error, when giving autograd the inital gradient "
            "the initial start node (last tensor) needs to be, "
            "defined, and there cannot be branching (not implemented yet)"
    );
    size_t start_index = 0;
    for(size_t i = 0; i < traversed.size(); ++i){
        if(traversed[i] == this->start_node){
            start_index = i;
            break;
        }
    }
    utils::throw_exception( 
            start_index <= this->validated_list,
            "Error, when giving autograd the inital gradient "
            "the initial start node (last tensor) needs to be, "
            "defined, and there cannot be branching (not implemented yet)"
    );

    this->start_node->accumulate_gradient(initialGrad);
    Tensor cpy_grad = this->start_node->gradient_().clone();
    this->start_node->zero_grad();
    this->start_node->run_backward(cpy_grad);
    this->start_node->accumulate_gradient(cpy_grad);
    for(size_t i = 0; i < start_index; ++i)
        traversed[i]->run_backward();
    for(size_t i = start_index+1; i < traversed.size(); ++i){
        traversed[i]->run_backward();
    }
}

template<class HashSet>
void AutoGrad<HashSet>::backward(){
    list_type& traversed = this->to_list();
    utils::throw_exception(
            bool(traversed[0]->raw_gradient())
            && !traversed[0]->raw_gradient()->is_null(),
           "Error, backward function not given an initial "
           "gradient is expected to already have a gradient defined"
    );
    
    // above ensures gradient is defined
    Tensor cpy_grad = traversed[0]->gradient_().clone();
    traversed[0]->zero_grad();
    traversed[0]->run_backward(cpy_grad);
    traversed[0]->accumulate_gradient(cpy_grad);
    for(size_t i = 1; i < traversed.size(); ++i){
        traversed[i]->run_backward();
    }
}

template<class HashSet>
void AutoGrad<HashSet>::zero_grad() {
    list_type& traversed = this->to_list();
    for(auto& node : traversed)
        node->zero_grad();
}

}



template class nt::grad::AutoGrad<std::unordered_set<nt::intrusive_ptr<nt::grad::utility::GraphNode>>>;

