#include "Layer.h"
#include "../utils/utils.h"

namespace nt{

void Layer::get_all_layers(reflect::detail::custom_typed_map<Layer>& map, std::string add){
	reflect::detail::custom_typed_map<Layer> my_layers = this->get_named_vars<Layer>();
	for(auto it : my_layers){
		const std::string& name = it.first;
		it.second.get_all_layers(map, add + '.' + name);
	}
	map.extend_unique(std::move(my_layers), add+'.');
}

void Layer::register_sub_layers() noexcept{
	reflect::detail::custom_typed_iterator<Layer> layers = this->get_vars<Layer>();
	for(auto& l : layers){
		utils::throw_exception(l.ParentGraph == nullptr, "One layer is being held by multiple parent layers, this is not allowed for the dynamic autograd graph");
		l.ParentGraph = MyGraph;
	}
}

reflect::detail::custom_typed_iterator<Layer> Layer::get_all_layers(){
	reflect::detail::custom_typed_iterator<Layer> outp = this->get_vars<Layer>();
	for(auto& l : this->get_vars<Layer>()){
		outp.extend(l.get_all_layers());
	}
	return std::move(outp);
}


Layer& Layer::eval(){
	bool overriden_eval = reflect::detail::backward_module_function_overriden(this->ptr_);
	this->is_eval = true;
	if(!overriden_eval){
		//the eval function should handle how to handle what grad is tracked and what is not
		for(auto& parameter : this->get_vars<TensorGrad>()){
			parameter.eval();
		}
	}
	for(auto& l : this->get_vars<Layer>()){
		l.eval();
	}
	return *this;
}

Layer& Layer::train(){
	bool overriden_eval = reflect::detail::backward_module_function_overriden(this->ptr_);
	this->is_eval = false;
	if(!overriden_eval){
		for(auto& parameter : this->get_vars<TensorGrad>()){
			parameter.train();
		}
	}
	for(auto& l : this->get_vars<Layer>()){
		l.train();
	}
	return *this;
}

void Layer::update(){
	for(auto& parameter : this->get_vars<TensorGrad>()){
		parameter.update();
	}
	for(auto& l : this->get_vars<Layer>()){
		l.update();
	}
}

//I want to have a way to track every input and output
//

TensorGrad Layer::forward(TensorGrad _x){
	if(this->is_eval){
		bool overriden_eval = reflect::detail::backward_module_function_overriden(this->ptr_);
		if(overriden_eval){
			return TensorGrad(this->ptr_->eval(_x.tensor));
		}
		return this->ptr_->forward(_x);
	}
	intrusive_ptr<TensorGrad> x = make_intrusive<TensorGrad>(_x.detach()); //modify the tensor grad to remove its gradient tracks before _x
	size_t grad_index = this->MyGraph->mark_input(x);
	size_t parent_index;
	if(this->ParentGraph){
		intrusive_ptr<TensorGrad> p_x = make_intrusive<TensorGrad>(_x); // have the gradient graph within this one
		this->ParentGraph->mark_child_layerStart(p_x);
		parent_index = this->ParentGraph->mark_child_input(x, this);

	}
	intrusive_ptr<TensorGrad> forward_out = make_intrusive<TensorGrad>(this->ptr_->forward(*x));
	this->MyGraph->mark_output(forward_out, grad_index);
	if(this->ParentGraph){
		intrusive_ptr<TensorGrad> ngrad_out = make_intrusive<TensorGrad>(forward_out->tensor);
		this->ParentGraph->mark_child_layerEnd(ngrad_out);
		this->ParentGraph->mark_child_output(forward_out, this, parent_index);
	}
	return *forward_out;
}

void get_grad_from_parents(intrusive_ptr<TensorGrad>& tgs, intrusive_ptr<TensorGrad>& input, Tensor& grad){
	for(auto& parent : tgs->parents){
		//a way to determine if they occupy the same memory
		if(parent->data_ptr() == input->data_ptr()){
		//TODO: look into the function below
		/* if(parent->occupy_same_tensor_memory(*input)){ */
			grad = std::move(parent->grad_value());
			return;
		}
		get_grad_from_parents(parent, input, grad);
	}
}

/* bool get_grad_from_parents_print(intrusive_ptr<TensorGrad>& tgs, intrusive_ptr<TensorGrad>& input){ */
/* 	for(auto& parent : tgs->parents){ */
/* 		std::cout << "parent shape: "<<parent->shape() << std::endl; */
/* 		if(parent->data_ptr() == input->data_ptr()){ */ 
/* 			//this is a way to determine if they occupy the same memory */
/* 			std::cout << "found input!"<<std::endl; */
/* 			return true; */
/* 		}else if(parent->shape() == input->shape()){ */
/* 			std::cout << "shapes were equal though..."<<std::endl; */
/* 		} */
/* 		return get_grad_from_parents_print(parent, input); */
/* 	} */
/* 	return false; */
/* } */


Tensor Layer::backward(Tensor grad){

	bool overriden_backward = reflect::detail::backward_module_function_overriden(this->ptr_);
	if(overriden_backward){
		this->MyGraph->back().second->input->grad = make_intrusive<tensor_holder>(this->ptr_->backward(grad));
		return this->MyGraph->back().second->input->grad_value();
	}
	while(this->MyGraph->size() > 0){
		//get the last pair of layers and the output and input tensor grad
		std::pair<Layer*, intrusive_ptr<LayerNode>> info = this->MyGraph->pop_back();
		//if it is this layer, then I know that the backward function has not been overriden
		//and then in that case just have the tensor grads update the path of grads themselves
		if(info.first == nullptr){
			info.second->output->backward(grad);
			get_grad_from_parents(info.second->output, info.second->input, grad);
			continue;
		}
		//otherwise calculate the gradient for the current layer...
		grad = std::move(info.first->backward(std::move(grad)));
	}
	return std::move(grad);
}

} //nt::
