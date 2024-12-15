#include "TensorGrad.h"
#include <functional>
#include <iostream>

namespace nt{


void TensorGrad::backward_self(const Tensor& grad, bool first){
	if(this->backwardFunc->is_valid()){
		if(first){
			this->backwardFunc->run(grad, this->parents, first);
		}else{
			this->backwardFunc->run(grad, this->parents);
		}
		this->backwardFunc->clear(); 
	}

	backward_parents(); //at this point each parent should have their own gradient
}

void TensorGrad::backward_child(const Tensor& grad, intrusive_ptr<intrusive_vector_tg>& vec, const int32_t& index){
	if(this->backwardFunc->is_valid()){
		this->backwardFunc->run(grad, this->parents);
		this->backwardFunc->clear();
	}
	auto cpy_parents = parents;
	//a lot of times, the only way the child is stored is within the vector it is about to be removed from
	//for this reason, the parents and the children need to be coppied over and then the backward pass can proceed
	//backward children before parents:
	if(this->children->size() > 0){
		for(int32_t i = this->children->size()-1; i > 0; --i){
			if(this->children->at(i) == vec->at(index)){continue;}
			this->children->at(i)->backward_child(children->at(i)->grad->tensor, children, i);
		}
	}
	vec->remove(index); //removes this child, preventing a continuous loop of doing the backward pass of this child
	
	//this is basically the child pass of the backward_parents is done, and then delete itself from 
	//any type of trace
	//and then afterwards proceede to the normal backward parents:
	for(auto& parent : cpy_parents){
		parent->backward_self(parent->grad->tensor);
	}
	//then done with this tensor :)
}

void TensorGrad::backward_parents(){
	if(this->children->size() > 0){
		for(int32_t i = this->children->size()-1; i > 0; --i){ //the first one is just an [] operator
			if(this->children->size() <= i){continue;} //because backward child will delete an index
			this->children->at(i)->backward_child(children->at(i)->grad->tensor, children, i);
		}
	
	}
	for(auto& parent : this->parents){
		parent->backward_self(parent->grad->tensor);
	}
	this->children->clear();
	/* this->parents.clear(); */
}



TensorGrad::TensorGrad(Tensor t, 
		intrusive_ptr<tensor_holder> g,
		intrusive_ptr<intrusive_back_func> f,
		std::vector<intrusive_ptr<TensorGrad> > p, 
		intrusive_ptr<intrusive_vector_tg> c)
	:tensor(t),
	do_track_grad(true),
	grad(g),
	backwardFunc(f),
	parents(std::move(p)),
	children(c)
{}



TensorGrad::TensorGrad(Scalar value) 
	: tensor(nt::Scalar(value)), 
	do_track_grad(true),
	grad(nullptr), 
	backwardFunc(make_intrusive<intrusive_back_func>()),
	parents({}),
	children(make_intrusive<intrusive_vector_tg>()) 
{}

TensorGrad::TensorGrad(const Tensor& t) 
	: tensor(t), 
	do_track_grad(true),
	grad(nullptr), 
	backwardFunc(make_intrusive<intrusive_back_func>()),
	parents({}),
	children(make_intrusive<intrusive_vector_tg>()) 
{}

TensorGrad::TensorGrad(Tensor&& t) 
	: tensor(t), 
	do_track_grad(true),
	grad(nullptr), 
	backwardFunc(make_intrusive<intrusive_back_func>()),
	parents({}),
	children(make_intrusive<intrusive_vector_tg>()) 
{}
TensorGrad::TensorGrad(TensorGrad&& tg)
	:tensor(std::move(tg.tensor)),
	do_track_grad(tg.do_track_grad),
	grad(std::move(tg.grad)),
	backwardFunc(std::move(tg.backwardFunc)),
	parents(std::move(tg.parents)),
	children(std::move(tg.children))
{}

TensorGrad::TensorGrad(const TensorGrad& tg)
	:tensor(tg.tensor),
	do_track_grad(tg.do_track_grad),
	grad(tg.grad),
	backwardFunc(tg.backwardFunc),
	parents(tg.parents),
	children(tg.children)
{}


void TensorGrad::swap(TensorGrad& tg){
	tensor.swap(tg.tensor);
	grad.swap(tg.grad);
	std::swap(backwardFunc, tg.backwardFunc);
	std::swap(parents, tg.parents);
	std::swap(children, tg.children);
	std::swap(do_track_grad, tg.do_track_grad);
}

TensorGrad& TensorGrad::operator=(const TensorGrad& tg){
	tensor = tg.tensor;
	grad = tg.grad;
	backwardFunc = tg.backwardFunc;
	parents = tg.parents;
	children = tg.children;
	do_track_grad = tg.do_track_grad;
	return *this;
}

TensorGrad& TensorGrad::operator=(TensorGrad&& tg){
	tensor = std::move(tg.tensor);
	grad = std::move(tg.grad);
	backwardFunc = std::move(tg.backwardFunc);
	parents = std::move(tg.parents);
	children = std::move(tg.children);
	do_track_grad = tg.do_track_grad;
	return *this;
}


bool TensorGrad::is_child() const noexcept {
	for(auto& parent : this->parents){
		if(parent->children->size() > 0){

			for(uint32_t i = 0; i < parent->children->size(); ++i){
				if(parent->children->at(i)->children == this->children){return true;}
			}
		}
	}
	return false;
}

void TensorGrad::unchild() noexcept {
	if(!is_child()){return;}
	for(auto& parent : this->parents){
		if(parent->children->size() > 0){
			for(uint32_t i = 0; i < parent->children->size(); ++i){
				if(parent->children->at(i)->children == this->children){
					parent->children->remove(i);
					return;
				}
			}
		}
	}
}

TensorGrad& TensorGrad::operator=(Scalar s){
	tensor = s;
	this->track_self_mod([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
				const size_t parent_index){
		parents[parent_index]->grad->tensor.fill_(0);
		
	});

	return *this;
}

TensorGrad& TensorGrad::set_(const Tensor& t){
	tensor.set_(t);
	this->track_self_mod([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
				const size_t parent_index){
		parents[parent_index]->grad->tensor.fill_(0);
		
	});
	return *this;
}

inline bool is_tracking_grad(const Tensor& t){return false;}

inline bool is_tracking_grad(const TensorGrad& t) noexcept {
	return t.do_track_grad;
}
template<typename... Args>
inline bool is_tracking_grad(const TensorGrad& t, const Args&... args) noexcept {
	if(t.do_track_grad){return true;}
	return is_tracking_grad(args...);
}


// Addition operation
TensorGrad TensorGrad::operator+(const TensorGrad& other) const {
	TensorGrad result(this->tensor + other.tensor);
	result.do_track_grad = is_tracking_grad(*this, other);
	result.track_tensors(*this, other);

	// Define backward function
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents) {
	
	    parents[0]->grad->tensor += grad;
	    parents[1]->grad->tensor += grad;
	});

	return std::move(result);
}


TensorGrad TensorGrad::operator+(const Scalar other) const {
	TensorGrad result(this->tensor + other);
	result.do_track_grad = is_tracking_grad(*this);
	result.track_tensors(*this);

	// Define backward function
	result.create_backward_function([other](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents) {
	    parents[0]->grad->tensor += grad + other; // Gradient of division w.r.t. the numerator
	});

	return std::move(result);
}

TensorGrad TensorGrad::operator+(const Tensor& other) const {
	TensorGrad result(this->tensor + other);
	result.do_track_grad = is_tracking_grad(*this);
	result.track_tensors(*this);

	// Define backward function
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents) {
	
	    parents[0]->grad->tensor += grad;
	});

	return std::move(result);
}

TensorGrad operator+(const Tensor& other, const TensorGrad& tg){
	TensorGrad result(tg.tensor + other);
	if(!tg.do_track_grad){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(tg);

	// Define backward function
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents) {
	
	    parents[0]->grad->tensor += grad;
	});
	return std::move(result);
	
}

TensorGrad operator+(const Scalar other, const TensorGrad& tg){
	TensorGrad result(tg.tensor + other);
	if(!tg.do_track_grad){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(tg);

	// Define backward function
	result.create_backward_function([other](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents) {
	    parents[0]->grad->tensor += grad + other; // Gradient of division w.r.t. the numerator
	});

	return std::move(result);
}

//This Addition operation
TensorGrad& TensorGrad::operator+=(const TensorGrad& other){
	this->tensor += other.tensor;
	this->track_self_mod([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
				const size_t parent_index, bool first){
		parents[parent_index+1]->grad->tensor += grad;
		if(!first)
			parents[parent_index]->grad->tensor += grad;
		
	}, other);

	return *this;
}

TensorGrad& TensorGrad::operator+=(const Tensor& other){
	this->tensor += other;
	this->track_self_mod([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
				const size_t parent_index, bool first){
		if(!first){
			parents[parent_index]->grad->tensor += grad;
		}
	});
	return *this;
}

TensorGrad& TensorGrad::operator+=(const Scalar other){
	this->tensor += other; //faster because I already know it is contiguous
	this->track_self_mod([other](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
				const size_t parent_index, bool first){
		if(!first){
			parents[parent_index]->grad->tensor += (grad + other);
		}
		else{
			parents[parent_index]->grad->tensor += other;
		}
	});
	return *this;
}
Tensor& operator+=(Tensor& t, const TensorGrad& tg){
	return t += tg.tensor;
}

// Subtraction operation
TensorGrad TensorGrad::operator-(const TensorGrad& other) const {
	TensorGrad result(this->tensor - other.tensor);
	if(!is_tracking_grad(*this, other)){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(*this, other);

	// Define backward function
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents) {
	
	    parents[0]->grad->tensor += grad;
	    parents[1]->grad->tensor -= grad;
	});

	return std::move(result);
}


TensorGrad TensorGrad::operator-(const Scalar other) const {
	TensorGrad result(this->tensor - other);
	if(!this->do_track_grad){
		result.do_track_grad = false;
		return std::move(result);
	}
	/* result.do_track_grad = is_tracking_grad(*this); */
	result.track_tensors(*this);

	// Define backward function
	result.create_backward_function([other](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents) {
	    parents[0]->grad->tensor += grad - other; // Gradient of division w.r.t. the numerator
	});

	return std::move(result);
}

TensorGrad TensorGrad::operator-(const Tensor& other) const {
	TensorGrad result(this->tensor - other);
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(*this);

	// Define backward function
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents) {
	
	    parents[0]->grad->tensor += grad;
	});

	return std::move(result);
}

TensorGrad operator-(const Tensor& other, const TensorGrad& tg){
	TensorGrad result(other - tg.tensor);
	if(!is_tracking_grad(tg)){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(tg);

	// Define backward function
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents) {
	
	    parents[0]->grad->tensor -= grad;
	});
	return std::move(result);
	
}

TensorGrad operator-(const Scalar other, const TensorGrad& tg){
	TensorGrad result(other - tg.tensor);
	if(!is_tracking_grad(tg)){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(tg);

	// Define backward function
	result.create_backward_function([other](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents) {
	    parents[0]->grad->tensor += other - grad; // Gradient of division w.r.t. the numerator
	});

	return std::move(result);
}



//This Subtraction operation
TensorGrad& TensorGrad::operator-=(const TensorGrad& other){
	this->tensor -= other.tensor;
	this->track_self_mod([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
				const size_t parent_index, bool first){
		parents[parent_index+1]->grad->tensor -= grad;
		if(!first)
			parents[parent_index]->grad->tensor += grad;
		
	}, other);
	return *this;
}

TensorGrad& TensorGrad::operator-=(const Tensor& other){
	this->tensor -= other;
	this->track_self_mod([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
				const size_t parent_index, bool first){
		if(!first)
			parents[parent_index]->grad->tensor += grad;
		
	});
	return *this;
}

TensorGrad& TensorGrad::operator-=(const Scalar other){
	this->tensor -= other; //faster because I already know it is contiguous
	
	this->track_self_mod([other](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
				const size_t parent_index, bool first){
		if(!first)
			parents[parent_index]->grad->tensor += grad - other;
		else
			parents[parent_index]->grad->tensor -= other;
		
	});
	return *this;
}

Tensor& operator-=(Tensor& t, const TensorGrad& tg){
	return t -= tg.tensor;
}

// Division operation
TensorGrad TensorGrad::operator/(const TensorGrad& other) const {
	TensorGrad result(this->tensor / other.tensor);
	if(!is_tracking_grad(*this, other)){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(*this, other);

	// Define backward function
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
				intrusive_ptr<tensor_holder> a, intrusive_ptr<tensor_holder> b) {
	
	    parents[0]->grad->tensor += grad / b->tensor; // Gradient of division w.r.t. the numerator
	    parents[1]->grad->tensor -= (a->tensor * grad) / b->tensor.pow(2); // Gradient of division w.r.t. the denominator
	}, *this, other);

	return std::move(result);
}
TensorGrad TensorGrad::operator/(const Scalar other) const {
	TensorGrad result(this->tensor / other);
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(*this);

	// Define backward function
	result.create_backward_function([other](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents) {
	    parents[0]->grad->tensor += grad / other; // Gradient of division w.r.t. the numerator
	});

	return std::move(result);
}

TensorGrad TensorGrad::operator/(const Tensor& other) const {
	TensorGrad result(this->tensor / other);
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(*this);

	// Define backward function
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
				intrusive_ptr<tensor_holder> b) {
	
	    parents[0]->grad->tensor += grad / b->tensor;
	}, other);

	return std::move(result);
}

TensorGrad operator/(const Tensor& other, const TensorGrad& tg){
	TensorGrad result(other / tg.tensor);
	if(!is_tracking_grad(tg)){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(tg);

	// Define backward function
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
				intrusive_ptr<tensor_holder> a, intrusive_ptr<tensor_holder> b) {
	
	    parents[0]->grad->tensor -= (a->tensor * grad) / b->tensor.pow(2);
	}, other, tg);
	return std::move(result);
	
}

TensorGrad operator/(const Scalar other, const TensorGrad& tg) {
    TensorGrad result(other / tg.tensor);
	if(!is_tracking_grad(tg)){
		result.do_track_grad = false;
		return std::move(result);
	}
    result.track_tensors(tg);

    // Define backward function
    result.create_backward_function([other](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
                                            intrusive_ptr<tensor_holder> b) {
        // Gradient of division w.r.t. the denominator
        parents[0]->grad->tensor -= (other * grad) / b->tensor.pow(2);
    }, tg);

    return std::move(result);
}

//This division operation
TensorGrad& TensorGrad::operator/=(const TensorGrad& other){
	if(!this->do_track_grad){
		this->tensor /= other.tensor;
		return *this;
	}
	intrusive_ptr<tensor_holder> this_clone = make_intrusive<tensor_holder>(this->tensor.clone());
	intrusive_ptr<tensor_holder> other_clone = make_intrusive<tensor_holder>(other.tensor.clone());
	this->tensor /= other_clone->tensor; //faster because I already know it is contiguous
	auto new_backward_func = [](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
			const size_t parent_index, bool first,
			intrusive_ptr<tensor_holder> a, intrusive_ptr<tensor_holder> b){
		if(!first){
			parents[parent_index]->grad->tensor += grad / b->tensor;
		}else{
			parents[parent_index]->grad->tensor /= b->tensor;
		}
		parents[parent_index+1]->grad->tensor -= (a->tensor * grad) / (b->tensor.pow(2));
	};
	this->track_self_mod(std::bind(new_backward_func, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, this_clone, other_clone), other);
	return *this;
}

TensorGrad& TensorGrad::operator/=(const Tensor& other){
	if(!this->do_track_grad){
		this->tensor /= other;
		return *this;
	}
	intrusive_ptr<tensor_holder> other_clone = make_intrusive<tensor_holder>(other.clone());
	this->tensor /= other_clone->tensor; //faster because I already know it is contiguous
	auto new_backward_func = [](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
			const size_t parent_index, bool first,
			intrusive_ptr<tensor_holder> b){
		if(!first){
			parents[parent_index]->grad->tensor += grad / b->tensor;
		}else{
			parents[parent_index]->grad->tensor /= b->tensor;
		}
	};
	this->track_self_mod(std::bind(new_backward_func, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, other_clone));
	return *this;	
}

TensorGrad& TensorGrad::operator/=(const Scalar other){
	this->tensor /= other; //faster because I already know it is contiguous
	this->track_self_mod([other](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
			const size_t parent_index, bool first){
		if(!first){
			parents[parent_index]->grad->tensor += grad / other;
		}else{
			parents[parent_index]->grad->tensor /= other;
		}
	});
	return *this;

}

Tensor& operator/=(Tensor& t, const TensorGrad& tg){
	return t /= tg.tensor;
}


// Multiplication operation
TensorGrad TensorGrad::operator*(const TensorGrad& other) const {
	TensorGrad result(this->tensor * other.tensor);
	if(!is_tracking_grad(*this, other)){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(*this, other);

	// Define backward function
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
								intrusive_ptr<tensor_holder> a, intrusive_ptr<tensor_holder> b) {

	    parents[0]->grad->tensor += b->tensor * grad;
	    parents[1]->grad->tensor += a->tensor * grad;
	}, *this, other);
	return std::move(result);
}



TensorGrad TensorGrad::operator*(const Scalar other) const {
	TensorGrad result(this->tensor * other);
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(*this);

	// Define backward function
	result.create_backward_function([other](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents) {
	    parents[0]->grad->tensor += grad * other; // Gradient of division w.r.t. the numerator
	});

	return std::move(result);
}

TensorGrad TensorGrad::operator*(const Tensor& other) const {
	TensorGrad result(this->tensor * other);
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(*this);

	// Define backward function
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
				intrusive_ptr<tensor_holder> b) {
	
	    parents[0]->grad->tensor += grad * b->tensor;
	}, other);

	return std::move(result);
}

TensorGrad operator*(const Tensor& other, const TensorGrad& tg){
	TensorGrad result(other * tg.tensor);
	if(!is_tracking_grad(tg)){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(tg);

	// Define backward function
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
				intrusive_ptr<tensor_holder> b) {
	
	    parents[0]->grad->tensor += grad * b->tensor;
	}, other);
	return std::move(result);
	
}

TensorGrad operator*(const Scalar other, const TensorGrad& tg) {
	TensorGrad result(other * tg.tensor);
	if(!is_tracking_grad(tg)){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(tg);

	// Define backward function
	result.create_backward_function([other](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents) {
		// Gradient of division w.r.t. the denominator
		parents[0]->grad->tensor -= other * grad;
	});

	return std::move(result);
}

//This multiplication operation
TensorGrad& TensorGrad::operator*=(const TensorGrad& other){
	if(!this->do_track_grad){
		this->tensor *= other.tensor;
		return *this;
	}
	intrusive_ptr<tensor_holder> this_clone = make_intrusive<tensor_holder>(this->tensor.clone());
	intrusive_ptr<tensor_holder> other_clone = make_intrusive<tensor_holder>(other.tensor.clone());
	this->tensor *= other_clone->tensor; //faster because I already know it is contiguous
	auto new_backward_func = [](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
			const size_t parent_index, bool first,
			intrusive_ptr<tensor_holder> a, intrusive_ptr<tensor_holder> b){
		if(!first){
			parents[parent_index]->grad->tensor += grad * b->tensor;
		}else{
			parents[parent_index]->grad->tensor *= b->tensor;
		}
		parents[parent_index+1]->grad->tensor += (a->tensor * grad);
	};
	this->track_self_mod(std::bind(new_backward_func, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, this_clone, other_clone), other);
	return *this;
}


//This multiplication operation
TensorGrad& TensorGrad::operator*=(const Tensor& other){
	if(!this->do_track_grad){
		this->tensor *= other;
		return *this;
	}
	intrusive_ptr<tensor_holder> other_clone = make_intrusive<tensor_holder>(other.clone());
	this->tensor *= other_clone->tensor; //faster because I already know it is contiguous
	auto new_backward_func = [](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
			const size_t parent_index, bool first,
			intrusive_ptr<tensor_holder> b){
		if(!first){
			parents[parent_index]->grad->tensor += grad * b->tensor;
		}else{
			parents[parent_index]->grad->tensor *= b->tensor;
		}
	};
	this->track_self_mod(std::bind(new_backward_func, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, other_clone));
	return *this;
}

TensorGrad& TensorGrad::operator*=(const Scalar other){
	this->tensor *= other; //faster because I already know it is contiguous
	const size_t other_indice = this->parents.size();
	this->track_self_mod([other](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
			const size_t parent_index, bool first){
		if(!first){
			parents[parent_index]->grad->tensor += grad * other;
		}else{
			parents[parent_index]->grad->tensor *= other;
		}
	});
	return *this;
}


Tensor& operator*=(Tensor& t, const TensorGrad& tg){
	return t *= tg.tensor;
}

std::ostream& operator<<(std::ostream &out, const TensorGrad& tg){return out << tg.tensor;}





TensorGrad TensorGrad::to_complex_from_real() const{
	TensorGrad result(this->tensor.to_complex_from_real());
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(*this);
	// Define backward function
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents) {
		parents[0]->grad->tensor += grad.real().to_dtype(parents[0]->grad->tensor.dtype);
	});
	return std::move(result);
}

TensorGrad TensorGrad::to_complex_from_imag() const{
	TensorGrad result(this->tensor.to_complex_from_imag());
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(*this);
	// Define backward function
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents) {
		parents[0]->grad->tensor += grad.imag().to_dtype(parents[0]->grad->tensor.dtype);
	});
	return std::move(result);
}

//need to make expand and expand_as before doing this:
TensorGrad TensorGrad::sum(size_value_t dim) const {
	//perform the forward sum operation
	TensorGrad result(this->tensor.sum(dim));
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}

	//track the current tensor in the result for backward computation
	result.track_tensors(*this);

	//define the backward function
	result.create_backward_function([dim](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents) {
		//repeat the gradient along the summed dimension
		parents[0]->grad->tensor += grad.unsqueeze(dim).expand_as(parents[0]->grad->tensor);
	});

	return std::move(result);
}

TensorGrad TensorGrad::sum() const{
	TensorGrad result(this->tensor.sum());
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}


	result.track_tensors(*this);

	//define the backward function
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents) {
		//repeat the gradient along the summed dimension
		parents[0]->grad->tensor += grad.expand_as(parents[0]->grad->tensor);
	});
	return std::move(result);
}


TensorGrad TensorGrad::mean(size_value_t dim) const {
	//perform the forward mean operation
	TensorGrad result(this->tensor.mean(dim));
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}


	//track the current tensor in the result for backward computation
	result.track_tensors(*this);
	size_value_t dim_size = shape()[dim];

	//define the backward function
	result.create_backward_function([dim, dim_size](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents) {
	//calculate the size of the dimension along which the mean was computed
	//this is dim_size

	//expand the gradient to the shape of the original tensor

	//divide the gradient by the size of the dimension to distribute it equally
	parents[0]->grad->tensor += grad.unsqueeze(dim).expand_as(parents[0]->grad->tensor) / dim_size;
	});

	return std::move(result);
}


TensorGrad TensorGrad::mean() const{
	TensorGrad result(this->tensor.mean());
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}

	result.track_tensors(*this);
	size_value_t dim_size = numel();

	result.create_backward_function([dim_size](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents) {
        //calculate the size of the dimension along which the mean was computed
        //this is dim_size

        //expand the gradient to the shape of the original tensor
        
        //divide the gradient by the size of the dimension to distribute it equally
        parents[0]->grad->tensor += grad / dim_size;
	});
	return std::move(result);
}

result_types::max<TensorGrad, Tensor> TensorGrad::max() const{
	result_types::max<Tensor, Tensor> o_a = this->tensor.max();
	TensorGrad result(o_a.values);
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return result_types::max<TensorGrad, Tensor>(std::move(result), std::move(o_a.indices));
	}

	result.track_grad(*this,
		[&o_a](Tensor& grad){return grad[o_a.indices];}
	);
	return result_types::max<TensorGrad, Tensor>(std::move(result), std::move(o_a.indices));
}

result_types::max<TensorGrad, Tensor> TensorGrad::max(size_value_t dim) const{
	result_types::max<Tensor, Tensor> o_a = this->tensor.max(dim);
	TensorGrad result(o_a.values);
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return result_types::max<TensorGrad, Tensor>(std::move(result), std::move(o_a.indices));
	}

	result.track_grad(*this,
		[&o_a](Tensor& grad){return grad[o_a.indices];}
	);
	return result_types::max<TensorGrad, Tensor>(std::move(result), std::move(o_a.indices));
}


TensorGrad TensorGrad::exp() const {
	//perform the forward exp operation
	TensorGrad result(this->tensor.exp());
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}

	//track the current tensor in the result for backward computation
	result.track_tensors(*this);

	//define the backward function
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
                                       intrusive_ptr<tensor_holder> a) {
		//compute the gradient of the exp function
		parents[0]->grad->tensor += a->tensor.exp() * grad;
	}, *this);

	return std::move(result);
}


TensorGrad& TensorGrad::exp_() {


	//apply the in-place exponential operation
	this->tensor.exp_();
	if(!this->do_track_grad){return *this;}

	//clone the tensor with the exp(tensor) function already applied
	//this will save computational time on the way backward
	//because the gradient of exp(x) is exp(x)
	intrusive_ptr<tensor_holder> this_clone = make_intrusive<tensor_holder>(this->tensor.clone());
	auto new_backward_func = [](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
			const size_t parent_index, bool first, intrusive_ptr<tensor_holder> a){
		if(!first){
			parents[parent_index]->grad->tensor += (a->tensor * grad);
		}else{
			parents[parent_index]->grad->tensor *= a->tensor;
		}
		
	};
	this->track_self_mod(std::bind(new_backward_func, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, this_clone));
	return *this;
}

TensorGrad TensorGrad::to_dtype(DType dt) const {
	TensorGrad result(this->tensor.to(dt));
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}

	result.track_tensors(*this);
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
		parents[0]->grad->tensor += grad.to(parents[0]->grad->tensor.dtype);
	});
	return std::move(result);
}

TensorGrad TensorGrad::to_device(DeviceType dt) const {
	TensorGrad result(this->tensor.to(dt));
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}

	result.track_tensors(*this);
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
		parents[0]->grad->tensor += grad.to(parents[0]->grad->tensor.device());
	});
	return std::move(result);
}

TensorGrad TensorGrad::contiguous() const{
	TensorGrad result(this->tensor.contiguous());
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}

	result.track_tensors(*this);
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
		parents[0]->grad->tensor += grad;
	});
	return std::move(result);

}

TensorGrad TensorGrad::clone() const{
	TensorGrad result(this->tensor.clone());
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}

	result.track_tensors(*this);
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
		parents[0]->grad->tensor += grad;
	});
	return std::move(result);

}


TensorGrad TensorGrad::pow(size_value_t exponent) const{
	TensorGrad result(this->tensor.pow(exponent));
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}

	result.track_tensors(*this);

	result.create_backward_function([&exponent](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> a){
		parents[0]->grad->tensor += exponent * a->tensor.pow(exponent - 1) * grad;
	}, *this);

	return std::move(result);
}


TensorGrad& TensorGrad::inverse_() {
	if(!this->do_track_grad){
		this->tensor.inverse_();
		return *this;
	}

	//clone the original tensor for backward computation
	// pow(2) to just off the bat save memory and computational time
	intrusive_ptr<tensor_holder> this_clone = make_intrusive<tensor_holder>(this->tensor.pow(2));

	// Apply the in-place inverse operation
	this->tensor.inverse_();
	auto new_backward_func = [](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
			const size_t parent_index, bool first, intrusive_ptr<tensor_holder> a){
		if(!first){
			parents[parent_index]->grad->tensor -= (grad / a->tensor);
		}else{
			parents[parent_index]->grad->tensor /= a->tensor;
		}
		
	};
	this->track_self_mod(std::bind(new_backward_func, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, this_clone));
	return *this;
}

TensorGrad TensorGrad::inverse() const {
	TensorGrad result(this->tensor.inverse());
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(*this);
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
				intrusive_ptr<tensor_holder> a){
		parents[0]->grad->tensor -= grad / a->tensor;
	}, make_intrusive<tensor_holder>(this->tensor.pow(2)));

	return std::move(result);
}

TensorGrad TensorGrad::clip(Scalar lower, Scalar higher) const{
	TensorGrad result(this->tensor.clip(lower, higher));
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(*this);
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents, intrusive_ptr<tensor_holder> a){
		parents[0]->grad->tensor[a->tensor] += grad[a->tensor];
	}, make_intrusive<tensor_holder>(((*this) >= lower) || ((*this) <= higher)));
	return std::move(result);
}


TensorGrad& TensorGrad::clip_(Scalar lower, Scalar higher) {
	(*this)[*this < lower] = lower;
	(*this)[*this > higher] = higher;
	return *this;
}

TensorGrad TensorGrad::pad(std::vector<size_value_t> p, const char* mode, double value) const{
	utils::throw_exception(p.size() % 2 == 0, "RuntimeError: The size of the pad must have 2 per dimension");
	utils::throw_exception((p.size() / 2) <= dims(), "RuntimeError: expected padding for at most $ dims but instead got $", dims(), int(p.size() / 2));

	std::vector<nt::my_range> ranges(dims());
	auto begin = p.cbegin();
	size_value_t start = dims() - size_value_t(p.size() / 2);
	for(size_value_t i = 0; i < dims(); ++i){
		if(i < (start)){
			ranges[i] = my_range(0, shape()[i]);
			continue;
		}
		ranges[i] = my_range(*begin, (-1)*size_value_t(*(begin + 1)));
		++begin;
		++begin;
	}
	TensorGrad result(this->tensor.pad(p, mode, value));
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(*this);
	result.create_backward_function([ranges](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
		parents[0]->grad->tensor += grad[ranges];
	});
	return std::move(result);
}

TensorGrad TensorGrad::flip(size_value_t dim) const{
	TensorGrad result(this->flip(dim));
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(*this);
	result.create_backward_function([dim](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
		parents[0]->grad->tensor += grad.flip(dim);
	});
	return std::move(result);
}

TensorGrad TensorGrad::flip() const{
	TensorGrad result(this->flip());
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(*this);
	result.create_backward_function([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
		parents[0]->grad->tensor += grad.flip();
	});
	return std::move(result);
}


TensorGrad TensorGrad::dilate(size_value_t dil) const{
	TensorGrad result(this->dilate(dil));
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(*this);
	result.create_backward_function([dil](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
		parents[0]->grad->tensor += grad.undilate_(dil);
	});
	return std::move(result);
	
}

TensorGrad TensorGrad::undilate(size_value_t dil) const{
	TensorGrad result(this->undilate(dil));
	if(!is_tracking_grad(*this)){
		result.do_track_grad = false;
		return std::move(result);
	}
	result.track_tensors(*this);
	result.create_backward_function([dil](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
		parents[0]->grad->tensor += grad.dilate(dil);
	});
	return std::move(result);
	
}



//these are all the operations where it is just the stride or view changed
//when that happens, the same thing cam just happen to the gradient that is being tracked
//and when that gradient is corrected, it will automatically update the gradient of the original tensor appropriately


#define COMBINE_PAIR(type, name) type name

#define COMBINE_RECURSIVE_0() 
#define COMBINE_RECURSIVE_1(type1, name1) COMBINE_PAIR(type1, name1)
#define COMBINE_RECURSIVE_2(type1, name1, type2, name2) COMBINE_PAIR(type1, name1), COMBINE_PAIR(type2, name2)
#define COMBINE_RECURSIVE_3(type1, name1, type2, name2, type3, name3) COMBINE_PAIR(type1, name1), COMBINE_PAIR(type2, name2), COMBINE_PAIR(type3, name3)
#define COMBINE_RECURSIVE_4(type1, name1, type2, name2, type3, name3, type4, name4) COMBINE_PAIR(type1, name1), COMBINE_PAIR(type2, name2), COMBINE_PAIR(type3, name3), COMBINE_PAIR(type4, name4)
// Add more as needed



#define COMBINE_SELECT_MACRO(_1, _1b, _2, _2b, _3, _3b, _4, _4b, NAME, ...) NAME
#define COMBINE_ARGUMENTS(...) COMBINE_SELECT_MACRO(__VA_ARGS__, COMBINE_RECURSIVE_4, COMBINE_RECURSIVE_4, COMBINE_RECURSIVE_3, COMBINE_RECURSIVE_3, COMBINE_RECURSIVE_2, COMBINE_RECURSIVE_2, COMBINE_RECURSIVE_1, COMBINE_RECURSIVE_0, COMBINE_RECURSIVE_0)(__VA_ARGS__)



#define	EXTRACT_ODD_PAIR(type, name) name
#define EXTRACT_ODD_RECURSIVE_0()
#define EXTRACT_ODD_RECURSIVE_1(type1, name1) name1
#define EXTRACT_ODD_RECURSIVE_2(type1, name1, type2, name2) name1, name2
#define EXTRACT_ODD_RECURSIVE_3(type1, name1, type2, name2, type3, name3) name1, name2, name3 
#define EXTRACT_ODD_RECURSIVE_4(type1, name1, type2, name2, type3, name3, type4, name4) name1, name2, name3, name4 
#define EXTRACT_SELECT_MACRO(_1, _1b, _2, _2b, _3, _3b, _4, _4b, NAME, ...) NAME



#define EXTRACT_ODD_ARGUMENTS(...) EXTRACT_SELECT_MACRO(__VA_ARGS__, EXTRACT_ODD_RECURSIVE_4, EXTRACT_ODD_RECURSIVE_4, EXTRACT_ODD_RECURSIVE_3, EXTRACT_ODD_RECURSIVE_3, EXTRACT_ODD_RECURSIVE_2, EXTRACT_ODD_RECURSIVE_2, EXTRACT_ODD_RECURSIVE_1, EXTRACT_ODD_RECURSIVE_0, EXTRACT_ODD_RECURSIVE_0)(__VA_ARGS__)


#define TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(op, ...) \
    TensorGrad TensorGrad::op(COMBINE_ARGUMENTS(__VA_ARGS__)) const { \
        TensorGrad result(tensor.op(EXTRACT_ODD_ARGUMENTS(__VA_ARGS__))); \
        result.track_grad(*this, \
            [EXTRACT_ODD_ARGUMENTS(__VA_ARGS__)](Tensor& grad){\
		    return grad.op(EXTRACT_ODD_ARGUMENTS(__VA_ARGS__)); } \
        ); \
        return std::move(result); \
    }

// these are all operations where the stride or view of the memory is changed
// (the actual values in the memory are not)
// for that reason, the same operation can just be done to track the gradient
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(view, SizeRef, s)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(operator[], size_value_t, i)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(operator[], my_range, i)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(operator[], std::vector<my_range>, i)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(operator[], Tensor, i)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(permute, std::vector<size_value_t>, v)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(transpose, size_value_t, a, size_value_t, b)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(unsqueeze, size_value_t, dim)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(unsqueeze_as, const Tensor&, dim)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(unsqueeze_as, const SizeRef&, dim)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(squeeze)    // No parameters for this function
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(flatten, size_value_t, a, size_value_t, b)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(unflatten, size_value_t, a, size_value_t, b)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(unfold, size_value_t, dim, size_value_t, size, size_value_t, step);
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(split_axis, std::vector<my_range>, i)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(split_axis, size_value_t, i)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(split_axis_1)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(div, size_value_t, i)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(real)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(imag)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(flip_)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(undilate_, size_value_t, dil)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(repeat_, size_value_t, amt)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(repeat_, size_value_t, dim, size_value_t, amt)
TENSORGRAD_CHANGE_STRIDE_VIEW_OPERATION(expand, SizeRef, s)






/* TensorGrad TensorGrad::operator[](TensorGrad::size_value_t&&) const; */
/* TensorGrad TensorGrad::operator[](Tensor&&) const; */
/* TensorGrad TensorGrad::operator[](my_range&&) const; */
/* TensorGrad TensorGrad::operator[](std::vector<my_range>&&) const; */

// Backward propagation
void TensorGrad::backward(const Tensor& initialGrad) {
	if(this->grad){
		this->grad->tensor += initialGrad;
	}
	else{
		this->grad = nt::make_intrusive<tensor_holder>(initialGrad);
	}
	this->backward_self(this->grad->tensor, true);

}

void TensorGrad::zero_grad() {
	if(this->grad){
		this->grad->tensor.fill_(0);
	}
	else{
		this->grad = nt::make_intrusive<tensor_holder>(nt::functional::zeros_like(tensor));
	}
	for(auto& parent : parents)
		parent->zero_grad();
}
Tensor TensorGrad::grad_value() const {
	if(this->grad){
		return this->grad->tensor;
	}
	return Tensor(Scalar(0.0));
}

void TensorGrad::update() {
	this->tensor -= this->grad->tensor;
}

}
