#include "Layer.h"
#include "TensorGrad.h"
#include "../utils/utils.h"
#include <ios>

namespace nt{

void Layer::get_all_layers(reflect::detail::custom_typed_map<Layer>& map, std::string add){
	reflect::detail::custom_typed_map<Layer> my_layers = this->get_named_vars<Layer>();
	my_layers.extend(this->ptr_->_get_mapped_layer_names_wrapped());
	for(auto it : my_layers){
		const std::string& name = it.first;
		it.second.get_all_layers(map, add + '.' + name);
	}
	map.extend_unique(std::move(my_layers), add+'.');
}

/* void Layer::register_sub_layers() noexcept{ */
/* 	reflect::detail::custom_typed_iterator<Layer> layers = this->get_vars<Layer>(); */
/* 	for(auto& l : layers){ */
/* 		utils::throw_exception(l.ParentGraph == nullptr, "One layer is being held by multiple parent layers, this is not allowed for the dynamic autograd graph"); */
/* 		l.ParentGraph = MyGraph; */
/* 	} */
/* } */

reflect::detail::custom_typed_iterator<Layer> Layer::get_all_layers(){
	reflect::detail::custom_typed_map<Layer> named = this->get_all_named_layers();
	return reflect::detail::custom_typed_iterator<Layer>(named.get_references().begin(), named.get_references().end()); 
}


Layer& Layer::eval(){
	bool overriden_eval = reflect::detail::backward_module_function_overriden(this->ptr_);
	this->is_eval = true;
	if(!overriden_eval){
		//the eval function should handle how to handle what grad is tracked and what is not
		for(auto& parameter : this->get_vars<TensorGrad>()){
			// parameter.eval();
            parameter.track_grad_(false);
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
	for(auto& parameter : this->get_vars<TensorGrad>()){
		// parameter.train();
		parameter.track_grad_(true);
	}
	for(auto& l : this->get_vars<Layer>()){
		l.train();
	}
	return *this;
}


reflect::detail::custom_typed_iterator<TensorGrad> Layer::parameters(){
	reflect::detail::custom_typed_map<TensorGrad> named = this->named_parameters();
	return reflect::detail::custom_typed_iterator<TensorGrad>(named.get_references().begin(), named.get_references().end()); 
}

reflect::detail::custom_typed_map<TensorGrad> Layer::named_parameters(){
	reflect::detail::custom_typed_map<TensorGrad> start = this->get_named_vars<TensorGrad>();
	start.extend(this->ptr_->_get_mapped_grad_names_wrapped());

	for(const auto& l : this->get_named_vars<Layer>()){
		start.extend_unique(std::move(l.second.named_parameters()), '.'+l.first);
	}
	return std::move(start);
}

void Layer::update(){
	for(auto& parameter : this->parameters()){
		parameter.update_mutable();
	}
}


TensorGrad Layer::_run_forward(std::vector<utils::any_ref> _vec){
	if(this->is_eval){
		return reflect::detail::run_eval_function(this->ptr_, std::move(_vec));
	}
	return reflect::detail::run_forward_function(this->ptr_, std::move(_vec));
	
}



//I want to have a way to track every input and output
//this is old below, better way of doing it now
//now the backward function if it is overriden is just attached to the output tensor
//and the input tensor and the output tensor are just attached to each other

/* TensorGrad Layer::forward(TensorGrad _x){ */
/* 	if(this->is_eval){ */
/* 		bool overriden_eval = reflect::detail::backward_module_function_overriden(this->ptr_); */
/* 		if(overriden_eval){ */
/* 			return TensorGrad(this->ptr_->eval(_x.detach())); */
/* 		} */
/* 		return this->ptr_->forward(_x); */
/* 	} */
/* 	intrusive_ptr<TensorGrad> x = make_intrusive<TensorGrad>(_x.detach()); //modify the tensor grad to remove its gradient tracks before _x */
/* 	size_t grad_index = this->MyGraph->mark_input(x); */
/* 	size_t parent_index; */
/* 	if(this->ParentGraph){ */
/* 		intrusive_ptr<TensorGrad> p_x = make_intrusive<TensorGrad>(_x); // have the gradient graph within this one */
/* 		this->ParentGraph->mark_child_layerStart(p_x); */
/* 		parent_index = this->ParentGraph->mark_child_input(x, this); */

/* 	} */
/* 	intrusive_ptr<TensorGrad> forward_out = make_intrusive<TensorGrad>(this->ptr_->forward(*x)); */
/* 	this->MyGraph->mark_output(forward_out, grad_index); */
/* 	if(this->ParentGraph){ */
/* 		intrusive_ptr<TensorGrad> ngrad_out = make_intrusive<TensorGrad>(forward_out->tensor); */
/* 		this->ParentGraph->mark_child_layerEnd(ngrad_out); */
/* 		this->ParentGraph->mark_child_output(forward_out, this, parent_index); */
/* 	} */
/* 	return *forward_out; */
/* } */

/* void get_grad_from_parents(intrusive_ptr<TensorGrad>& tgs, intrusive_ptr<TensorGrad>& input, Tensor& grad){ */
/* 	for(auto& parent : tgs->parents){ */
/* 		//a way to determine if they occupy the same memory */
/* 		if(parent->data_ptr() == input->data_ptr() && parent->shape() == input->shape() && parent->tensor.dtype() == input->tensor.dtype() && (parent->tensor.arr_void().get_bucket().intrusive_strides() == input->tensor.arr_void().get_bucket().intrusive_strides()) && (parent->tensor.arr_void().get_bucket().intrusive_device() == input->tensor.arr_void().get_bucket().intrusive_device())){ */
/* 		//TODO: look into the function below */
/* 		/1* if(parent->occupy_same_tensor_memory(*input)){ *1/ */
/* 			/1* std::cout << "found true, checking other conditions:"<<std::endl; *1/ */
/* 			/1* std::cout <<std::boolalpha << "strides == strides: " << *1/ */
/* 			/1* 	(parent->tensor.arr_void().get_bucket().intrusive_strides() == input->tensor.arr_void().get_bucket().intrusive_strides()) << std::endl << *1/ */
/* 			/1* 	"devices == devices: " << *1/ */
/* 			/1* 	(parent->tensor.arr_void().get_bucket().intrusive_device() == input->tensor.arr_void().get_bucket().intrusive_device()) << std::endl *1/ */
/* 			/1* 	<< "occupy_same_tensor_memory: "<< parent->occupy_same_tensor_memory(*input) << *1/ */ 
/* 			/1* 	std::noboolalpha << std::endl; *1/ */
/* 			grad = std::move(parent->grad_value()); */
/* 			return; */
/* 		} */
/* 		get_grad_from_parents(parent, input, grad); */
/* 	} */
/* } */

/* /1* bool get_grad_from_parents_print(intrusive_ptr<TensorGrad>& tgs, intrusive_ptr<TensorGrad>& input){ *1/ */
/* /1* 	for(auto& parent : tgs->parents){ *1/ */
/* /1* 		std::cout << "parent shape: "<<parent->shape() << std::endl; *1/ */
/* /1* 		if(parent->data_ptr() == input->data_ptr()){ *1/ */ 
/* /1* 			//this is a way to determine if they occupy the same memory *1/ */
/* /1* 			std::cout << "found input!"<<std::endl; *1/ */
/* /1* 			return true; *1/ */
/* /1* 		}else if(parent->shape() == input->shape()){ *1/ */
/* /1* 			std::cout << "shapes were equal though..."<<std::endl; *1/ */
/* /1* 		} *1/ */
/* /1* 		return get_grad_from_parents_print(parent, input); *1/ */
/* /1* 	} *1/ */
/* /1* 	return false; *1/ */
/* /1* } *1/ */


/* Tensor Layer::backward(Tensor grad){ */

/* 	bool overriden_backward = reflect::detail::backward_module_function_overriden(this->ptr_); */
/* 	if(overriden_backward){ */
/* 		this->MyGraph->back().second->input->grad = make_intrusive<tensor_holder>(this->ptr_->backward(grad)); */
/* 		return this->MyGraph->back().second->input->grad_value(); */
/* 	} */
/* 	while(this->MyGraph->size() > 0){ */
/* 		//get the last pair of layers and the output and input tensor grad */
/* 		std::pair<Layer*, intrusive_ptr<LayerNode>> info = this->MyGraph->pop_back(); */
/* 		//if it is this layer, then I know that the backward function has not been overriden */
/* 		//and then in that case just have the tensor grads update the path of grads themselves */
/* 		if(info.first == nullptr){ */
/* 			info.second->output->backward(grad); */
/* 			get_grad_from_parents(info.second->output, info.second->input, grad); */
/* 			continue; */
/* 		} */
/* 		//otherwise calculate the gradient for the current layer... */
/* 		grad = std::move(info.first->backward(std::move(grad))); */
/* 	} */
/* 	return std::move(grad); */
/* } */

} //nt::
