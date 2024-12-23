#include "Layer.h"
#include "../utils/utils.h"

namespace nt{

size_t LayerGraph::mark_input(const intrusive_ptr<TensorGrad>& inp){
	layers.push_back(nullptr);
	intrusive_ptr<LayerNode> node = make_intrusive<LayerNode>();
	node->input = inp;
	nodes.push_back(node);
	return nodes.size()-1;
}

void LayerGraph::mark_output(const intrusive_ptr<TensorGrad>& inp, size_t index){
	utils::throw_exception(layers[index] == nullptr, "Error with logging, expected last vector to be a nullptr");
	nodes.back()->output = inp;
}

void LayerGraph::mark_child_layerStart(const intrusive_ptr<TensorGrad>& inp){
	utils::throw_exception(layers.back() == nullptr, "Error with logging, expected last vector to be a nullptr");
	nodes.back()->output = inp;
}

size_t LayerGraph::mark_child_input(const intrusive_ptr<TensorGrad>& inp, Layer * l){
	utils::throw_exception(layers.back() == nullptr, "Error with logging, expected last vector to be a nullptr");
	layers.push_back(l);
	intrusive_ptr<LayerNode> node = make_intrusive<LayerNode>();
	node->input = inp;
	nodes.push_back(node);
	return nodes.size()-1;
}


void LayerGraph::mark_child_output(const intrusive_ptr<TensorGrad>& inp, Layer * l, size_t index){
	utils::throw_exception(layers[index] == l, "Error with logging, expected to have tracked correct layer at index $", index);
	nodes[index]->output = inp;
}



void LayerGraph::mark_child_layerEnd(const intrusive_ptr<TensorGrad>& inp){
	layers.push_back(nullptr);
	intrusive_ptr<LayerNode> node = make_intrusive<LayerNode>();
	node->input = inp;
	nodes.push_back(node);
}

void LayerGraph::dump_graph(){
    std::cout << "LayerGraph Dump:" << std::endl;
    for (size_t i = 0; i < nodes.size(); ++i) {
        std::cout << "Node " << i << ": "
                  << "Layer = " << (layers[i] ? layers[i]->name() : "nullptr");
	if(nodes[i]->input){
		std::cout << ", Input = " << nodes[i]->input->shape();
	}else{
		std::cout << ", Input = None";
	}
	if(nodes[i]->output){
		std::cout << ", Output = "<<nodes[i]->output->shape();
	}else{
		std::cout << ", Output = None";
		
	}
	std::cout << std::endl;
    }
}


} //nt::
