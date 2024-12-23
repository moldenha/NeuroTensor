#ifndef _NT_LAYER_H_
#define _NT_LAYER_H_

#include "Module.h"
#include "../Tensor.h"
#include "../functional/functional.h"
#include "TensorGrad.h"
#include "layers.h"
#include <memory>
#include <iostream>
#include <functional>
#include <unordered_map>
#include <type_traits>
#include <variant>
#include <unordered_map>
#include "../dtype/compatible/DType_compatible.h"
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <optional>


namespace nt{

class Layer;
class LayerGraph;


struct LayerNode{
	mutable std::atomic<int64_t> refcount_;
	intrusive_ptr<TensorGrad> input, output;
	LayerNode() : refcount_(0), input(nullptr), output(nullptr) {}
};


//this is a class to track each individual string of grads to update
//this allows the layers to have custom backward functions
class LayerGraph : public intrusive_ptr_target{
	//probably going to be a tensor grad for most of them
	//it is just at the end of a layer, the input and output should be marked
	std::vector<Layer*> layers;
	std::vector<intrusive_ptr<LayerNode> > nodes;
	public:
		LayerGraph() = default;
		size_t mark_input(const intrusive_ptr<TensorGrad>& inp);
		void mark_output(const intrusive_ptr<TensorGrad>& inp, size_t);
		void mark_child_layerStart(const intrusive_ptr<TensorGrad>& inp);
		//this is called after mark_child_layerStart
		size_t mark_child_input(const intrusive_ptr<TensorGrad>& inp, Layer * l);		
		void mark_child_output(const intrusive_ptr<TensorGrad>& inp, Layer * l, size_t);
		void mark_child_layerEnd(const intrusive_ptr<TensorGrad>& inp);
		inline bool has_subs() const noexcept {return layers.size() > 1;}
		inline std::pair<Layer*, intrusive_ptr<LayerNode>& > operator[](const int64_t i){
			return std::pair<Layer*, intrusive_ptr<LayerNode>& >(layers.at(i), nodes.at(i));
		}
		inline std::pair<const Layer*, const intrusive_ptr<LayerNode>& > operator[](const int64_t i) const {
			return std::pair<const Layer*, const intrusive_ptr<LayerNode>& >(layers.at(i), nodes.at(i));
		}
		inline std::pair<Layer*, intrusive_ptr<LayerNode>& > back(){
			return std::pair<Layer*, intrusive_ptr<LayerNode>& >(layers.back(), nodes.back());
		}
		inline std::pair<const Layer*, const intrusive_ptr<LayerNode>& > back() const {
			return std::pair<const Layer*, const intrusive_ptr<LayerNode>& >(layers.back(), nodes.back());
		}

		inline std::pair<Layer*, intrusive_ptr<LayerNode> > pop_back(){
			std::pair<Layer*, intrusive_ptr<LayerNode>> outp(layers.back(), nodes.back());
			layers.pop_back();
			nodes.pop_back();
			return outp;
		}

		inline const size_t size() const noexcept {return layers.size();}
		void dump_graph();

};


class Layer{
	intrusive_ptr<Module> ptr_;
	intrusive_ptr<LayerGraph> MyGraph, ParentGraph;
	std::type_index type_;
	bool is_eval;
	void get_all_layers(reflect::detail::custom_typed_map<Layer>& map, std::string add);

	public:
		Layer() = delete;

		template<typename T, std::enable_if_t<std::is_base_of_v<Module, T>, bool> = true>
		Layer(intrusive_ptr<T> mod)
		:ptr_(mod), MyGraph(make_intrusive<LayerGraph>()), ParentGraph(nullptr),  type_(typeid(T)), is_eval(false)
		{this->register_sub_layers();}

		template<typename T, std::enable_if_t<std::is_base_of_v<Module, T>, bool> = true>
		Layer(const T& mod)
		:ptr_(make_intrusive<T>(mod)), MyGraph(make_intrusive<LayerGraph>()), ParentGraph(nullptr), type_(typeid(T)), is_eval(false)
		{this->register_sub_layers();}

		template<typename T, std::enable_if_t<std::is_base_of_v<Module, T>, bool> = true>
		Layer& operator=(intrusive_ptr<T> mod){
			ptr_ = mod;
			MyGraph = make_intrusive<LayerGraph>();
			ParentGraph = nullptr;
			type_ = typeid(T);
			is_eval = false;
			this->register_sub_layers();
			return *this;
		}
		void register_sub_layers() noexcept;


		template<typename T>
		inline reflect::detail::custom_typed_iterator<T> get_vars(){return reflect::detail::get_module_vars<T>(ptr_);}
		template<typename T>
		inline reflect::detail::custom_typed_map<T> get_named_vars(){return reflect::detail::get_named_module_vars<T>(ptr_);}
		inline std::string name() const noexcept {return reflect::detail::get_module_name(ptr_);}

		inline reflect::detail::custom_typed_map<Layer> get_all_named_layers(){
			reflect::detail::custom_typed_map<Layer> outp;
			this->get_all_layers(outp, name());
			return std::move(outp);
		}
		reflect::detail::custom_typed_iterator<Layer> get_all_layers();
		template<typename T>
		inline T* is_layer() noexcept {return dynamic_cast<T*>(ptr_.get());}
		template<typename T>
		inline const T* is_layer() const noexcept {return dynamic_cast<const T*>(ptr_.get());}
		Layer& eval();
		Layer& train();
		void update();

		//the idea is that each connecting gradient will have its gradient tracker in place
		//so if you notice, back in the LayerGraph class, everywhere where a new node is made, and an input is set, that tensor grad does not have a gradient graph associated with it
		//but the output does
		TensorGrad forward(TensorGrad _x);
		Tensor backward(Tensor grad); //returns dx
		inline TensorGrad operator()(TensorGrad _x){return forward(std::move(_x));}
		inline void dump_graph() noexcept {MyGraph->dump_graph();}
};

}


#include "Loss.h"
#endif //_NT_LAYER_H_
