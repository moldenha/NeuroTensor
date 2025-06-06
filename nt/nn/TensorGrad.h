#ifndef _NT_TENSOR_GRAD_H_
#define _NT_TENSOR_GRAD_H_
namespace nt{
class TensorGrad; // Forward declaration
}


#include "../Tensor.h"
#include "../intrusive_ptr/intrusive_ptr.hpp"
#include "../utils/tensor_holder.h"
#include <vector>
#include <memory>
#include <iostream>
#include <functional>
#include <unordered_map>
#include <type_traits>
#include <variant>
#include <optional>
#include "../utils/optional_list.h"
#include <atomic>
#include "functional_class.h"
#include "../utils/name_func_macro.h"
#include "ScalarGrad.h"


namespace nt{


namespace functional{
template <typename T, typename... Args, 
    std::enable_if_t<std::is_same_v<std::decay_t<T>, TensorGrad>, int> = 0>
TensorGrad list(T &&first, Args &&...rest); //forward declaration
}

/* inline intrusive_ptr<tensor_holder>& operator=(intrusive_ptr<tensor_holder>& it, Tensor&& t) noexcept { */
/* 	it = make_intrusive<tensor_holder>(t); */
/* 	return it; */
/* } */

// class intrusive_vector_tg : public intrusive_ptr_target{
// 	std::vector<intrusive_ptr<TensorGrad> > vec;
// 	public:
// 		intrusive_vector_tg() = default;
// 		inline size_t size() const& {
// 			return vec.size();
// 		}
// 		inline intrusive_ptr<TensorGrad>& at(uint32_t i){return vec[i];}
// 		inline const intrusive_ptr<TensorGrad>& at(uint32_t i) const {return vec[i];}
// 		inline void push_back(intrusive_ptr<TensorGrad> val){
// 			vec.push_back(val);
// 		}
// 		inline void push_back(const TensorGrad& val){
// 			vec.push_back(make_intrusive<TensorGrad>(val));
// 		}
// 		inline void push_back(intrusive_ptr<TensorGrad> val) const {
// 			const_cast<std::vector<intrusive_ptr<TensorGrad> >&>(vec).push_back(val);
// 		}
// 		inline void push_back(const TensorGrad& t) const {
// 			const_cast<std::vector<intrusive_ptr<TensorGrad> >&>(vec).push_back(make_intrusive<TensorGrad>(t));
// 		}
// 		inline void clear() {vec.clear();}
// 		inline void remove(uint32_t i){
//             if(i < vec.size())
//                 vec.erase(vec.begin() + i);
// 		}
// 		inline std::vector<intrusive_ptr<TensorGrad> >::iterator begin() {return vec.begin();}
// 		inline std::vector<intrusive_ptr<TensorGrad> >::iterator end() {return vec.end();}
// 		inline std::vector<intrusive_ptr<TensorGrad> >::const_iterator begin() const {return vec.begin();}
// 		inline std::vector<intrusive_ptr<TensorGrad> >::const_iterator end() const {return vec.begin();}
// 		inline bool in(intrusive_ptr<TensorGrad>& t) const noexcept{
// 			for(auto& v : vec){
// 				if(v == t){return true;}
// 			}
// 			return false;
// 		}

// };

class tensor_grad_vec : public intrusive_ptr_target{
    std::vector<intrusive_ptr<TensorGrad>> vec;
public:
    using vec_type = typename std::vector<intrusive_ptr<TensorGrad>>;
    tensor_grad_vec() = default;
    ~tensor_grad_vec();
    inline vec_type& get() noexcept {return vec;}
    void clear();
    inline void push_back(intrusive_ptr<TensorGrad>& gr) {vec.push_back(gr);}
    template<typename... T>
    inline void emplace_back(T&&... items){
        vec.push_back(make_intrusive<TensorGrad>(std::forward<T&&>(items)...));
    }
    inline vec_type::size_type size() const {return vec.size();}
    //should be assessed in reverse in order to have correct calculations
    inline vec_type::reverse_iterator begin() {return vec.rbegin();}
    inline vec_type::reverse_iterator end() {return vec.rend();}
    inline intrusive_ptr<TensorGrad>& back() {return vec.back();}
    inline const intrusive_ptr<TensorGrad>& back() const {return vec.back();}

};


class intrusive_back_func : public intrusive_ptr_target{
	public:
		using function_type = std::function<void(const Tensor&, std::vector<intrusive_ptr<TensorGrad>>&)>;
		using function_type_b = std::function<void(const Tensor&, std::vector<intrusive_ptr<TensorGrad>>&, bool)>;
	private:
		std::variant<std::monostate, function_type, function_type_b> Func;
        mutable bool _used;
        std::string name;
	public:
		intrusive_back_func()
			:Func(std::monostate{}), _used(false), name("NoneBackward")
		{utils::throw_exception(Func.index() == 0, "Loaded a function type into backward function and index was expected to be 0 but got $", Func.index());}
		intrusive_back_func(std::string _name)
			:Func(std::monostate{}), _used(false), name(_name+"Backward")
		{
        utils::throw_exception(Func.index() == 0, "Loaded a function type into backward function and index was expected to be 0 but got $", Func.index());
        name[0] = std::toupper(name[0]);
        }
		intrusive_back_func(function_type func, std::string _name)
			:Func(func), _used(false), name(_name+"Backward")
		{
        utils::throw_exception(Func.index() == 1, "Loaded a function type into backward function and index was expected to be 1 but got $", Func.index());
        name[0] = std::toupper(name[0]);
        }
		intrusive_back_func(function_type_b func, std::string _name)
			:Func(func), _used(false), name(_name + "Backward")
		{
        utils::throw_exception(Func.index() == 2, "Loaded a function type into backward function and index was expected to be 2 but got $", Func.index());
        name[0] = std::toupper(name[0]);
        }
		/* inline function_type& get() noexcept {return Func;} */
		/* inline const function_type& get() const noexcept {return Func;} */
		inline void set(function_type func) noexcept {Func = func; _used = false;}
		inline void set(function_type_b func) noexcept {Func = func; _used = false;}
        inline void set(std::nullptr_t) noexcept {Func = std::monostate{}; _used = false;}
        inline void set_name(std::string _name) noexcept {name = _name + "Backward"; name[0] = std::toupper(name[0]);}
		inline const std::string& get_name() const noexcept {return name;}
        // inline void clear() noexcept {Func = std::monostate{}; _has_been_cleared = true;}
		inline size_t index() const noexcept {return Func.index();}
        inline const bool& used() const noexcept {return _used;}
        inline void un_use() const noexcept {_used = false;}
        inline void set_used() const noexcept {_used = true;}
		inline void run(const Tensor& t, std::vector<intrusive_ptr<TensorGrad>>& v){
            utils::throw_exception(_used == false, "Backward function already used, graph constructed improperly");
			if(std::monostate* f = std::get_if<std::monostate>(&Func)){
                _used = true;
                return;
				//utils::throw_exception(false, "Tried to run invalid function");
			}
			else if(function_type* f = std::get_if<function_type>(&Func)){
				utils::throw_exception(*f != nullptr, "Trying to run invalid function, was nullptr");
				(*f)(t, v);
			}
			else if(function_type_b* f = std::get_if<function_type_b>(&Func)){
				utils::throw_exception(*f != nullptr, "Trying to run invalid function, was nullptr");
				(*f)(t, v, false);
				
			}
			else{
				throw std::bad_variant_access();
			}
            _used = true;

		}
		inline void run(const Tensor& t, std::vector<intrusive_ptr<TensorGrad>>& v, bool b){
            utils::throw_exception(_used == false, "Backward function already used, graph constructed improperly");
			if(std::monostate* f = std::get_if<std::monostate>(&Func)){
                _used = true;
                return;
				//utils::throw_exception(false, "Tried to run invalid function");
			}
			else if(function_type* f = std::get_if<function_type>(&Func)){
				utils::throw_exception(*f != nullptr, "Trying to run invalid function, was nullptr");
				(*f)(t, v);
			}
			else if(function_type_b* f = std::get_if<function_type_b>(&Func)){
				utils::throw_exception(*f != nullptr, "Trying to run invalid function, was nullptr");
				(*f)(t, v, b);
				
			}
			else{
				throw std::bad_variant_access();
			}
            _used = true;
		}

		inline bool is_valid() const noexcept {
			if(const std::monostate* f = std::get_if<std::monostate>(&Func)){
				return false;
			}
			else if(const function_type* f = std::get_if<function_type>(&Func)){
				if(*f == nullptr){return false;}
				return true;
			}
			else if(const function_type_b* f = std::get_if<function_type_b>(&Func)){
				if(*f == nullptr){return false;}
				return true;
				
			}
			return false;
		}

};


class TensorGrad : public intrusive_ptr_target{
    friend class ScalarGrad;
	public:
		using size_value_t = Tensor::size_value_t;
        
        template <typename T, typename... Args, std::enable_if_t<std::is_same_v<std::decay_t<T>, TensorGrad>, int> >
        friend TensorGrad functional::list(T &&first, Args &&...rest);


	private:
    friend class functional::TensorGrad_Functional_Class;

	inline static intrusive_ptr<tensor_holder> make_tensor_holder(const TensorGrad& t){
		return nt::intrusive_ptr<tensor_holder>::make(t.tensor.conditional_mutate_clone());
	}
	inline static intrusive_ptr<tensor_holder> make_tensor_holder(intrusive_ptr<tensor_holder> t){return t;}
	inline static intrusive_ptr<tensor_holder> make_tensor_holder(const Tensor& t){
		return intrusive_ptr<tensor_holder>::make(t.conditional_mutate_clone());
	}
	
	// template<typename... Args>
	// void track_self_mod(std::function<void(const Tensor&, std::vector<intrusive_ptr<TensorGrad>>&, const size_t)> new_backward_func,
	// 	const Args&... args);
	// template<typename... Args>
	// void track_self_mod(std::function<void(const Tensor&, std::vector<intrusive_ptr<TensorGrad>>&, const size_t, bool)> new_backward_func,
	// 	const Args&... args);
	// void backward_self(const Tensor& grad, bool first=false);
	// void backward_child(const Tensor& grad, intrusive_ptr<intrusive_vector_tg>&, const int32_t&);
	// void backward_parents();

	// template<typename... Args>
	// void track_tensors(const TensorGrad&, const Args&... args);
	// void track_tensors(const TensorGrad&);

   //functions designed to track parents and children
    template<typename... Args>
    void track_tensors(const TensorGrad&, const Args&... args);
    void track_tensors(const TensorGrad&);
    void track_tensors(std::vector<TensorGrad>&);
    template<typename... Args>
    void track_self_mod_tensors(const TensorGrad&, const Args&... args);
    void track_self_mod_tensors();
    //function designed to track view and stride changes of gradient
    //[when memory remains unmodified]
    template<typename OutOperator>
    void track_grad(const TensorGrad& t, OutOperator&& op, const char* func_name = __NT_FUNCTION_NAME__);
    //create backward function for output
    template<typename backward_func>
    void create_backward_function(backward_func&& func, const char* func_name = __NT_FUNCTION_NAME__);
    template<typename backward_func, typename Arg1>
    void create_backward_function(backward_func&& func, Arg1&& arg1, const char* func_name = __NT_FUNCTION_NAME__);
    template<typename backward_func, typename Arg1, typename Arg2>
    void create_backward_function(backward_func&& func, Arg1&& arg1, Arg2&& arg2, const char* func_name = __NT_FUNCTION_NAME__);
    template<typename backward_func, typename Arg1, typename Arg2, typename Arg3>
    void create_backward_function(backward_func&& func, Arg1&& arg1, Arg2&& arg2, Arg3&& arg3, const char* func_name = __NT_FUNCTION_NAME__);
    template<typename backward_func, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
    void create_backward_function(backward_func&& func, Arg1&& arg1, Arg2&& arg2, Arg3&& arg3, Arg4&& arg4, const char* func_name = __NT_FUNCTION_NAME__);
    //for when knowing if this was the first tensor is important
    template<typename backward_func>
    void create_bool_backward_function(backward_func&& func, const char* func_name = __NT_FUNCTION_NAME__);
    template<typename backward_func, typename Arg1>
    void create_bool_backward_function(backward_func&& func, Arg1&& arg1, const char* func_name = __NT_FUNCTION_NAME__);
    template<typename backward_func, typename Arg1, typename Arg2>
    void create_bool_backward_function(backward_func&& func, Arg1&& arg1, Arg2&& arg2, const char* func_name = __NT_FUNCTION_NAME__);
    template<typename backward_func, typename Arg1, typename Arg2, typename Arg3>
    void create_bool_backward_function(backward_func&& func, Arg1&& arg1, Arg2&& arg2, Arg3&& arg3, const char* func_name = __NT_FUNCTION_NAME__);
    template<typename backward_func, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
    void create_bool_backward_function(backward_func&& func, Arg1&& arg1, Arg2&& arg2, Arg3&& arg3, Arg4&& arg4, const char* func_name = __NT_FUNCTION_NAME__);
    // template<typename backward_func, typename... Args, const char* func_name = __NT_FUNCTION_NAME__>
    // void create_bool_backward_function(backward_func&& func, Args&&... args);

    void delete_parents();
    void clear_parents();
    void delete_children();
    void clear_children();
    //the next 3 are related to handling children and branching
    //and making sure that the gradient is calculated in the correct order of branches
    void check_children();
    void check_self_done(weak_intrusive_ptr<TensorGrad>);
    void run_backward_self(weak_intrusive_ptr<TensorGrad>);
    //normal run_backward
    void run_backward(weak_intrusive_ptr<TensorGrad>);




	const bool grad_required;
    // this is a variable that is used to check how many children have to update its gradient before
    // this tensor updates it's parents gradient and so forth
	TensorGrad(Tensor t, intrusive_ptr<tensor_holder> _grad, intrusive_ptr<intrusive_back_func> back_func,
                    intrusive_ptr<tensor_grad_vec> _parents, intrusive_ptr<tensor_grad_vec> _children,
                    bool _grad_required);
	public:
		Tensor tensor;
		const DType& dtype;
		bool do_track_grad;
		mutable intrusive_ptr<tensor_holder> grad;
		intrusive_ptr<intrusive_back_func> backwardFunc;
        intrusive_ptr<tensor_grad_vec> children, parents;
        

		
		explicit TensorGrad(Scalar value, bool grad_required=true);
		explicit TensorGrad(const Tensor&, bool grad_required=true);
		explicit TensorGrad(Tensor&& t, bool grad_required=true);
		explicit TensorGrad(std::nullptr_t, bool grad_required = true);
        void release_resources() override;
		inline void eval() noexcept {if(grad_required){do_track_grad = false;}}
		inline void train() noexcept {if(grad_required){do_track_grad = true;}}
		TensorGrad(TensorGrad&& tg);
		TensorGrad(const TensorGrad& tg);
		TensorGrad& operator=(const TensorGrad& tg);
		TensorGrad& operator=(TensorGrad&& tg);
		TensorGrad& operator=(Scalar s);
		TensorGrad& operator=(const Tensor& t){return set_(t);}
		TensorGrad& set_(const Tensor& t);
        inline CommaOperator operator<<(Scalar s) {return tensor << s;}
		inline const DeviceType& device() const noexcept {return tensor.device();}
		void swap(TensorGrad&);
		// Addition operation
		TensorGrad operator+(const TensorGrad& other) const;
		TensorGrad operator+(const Scalar other) const;
		TensorGrad operator+(const Tensor& other) const;
		friend TensorGrad operator+(const Tensor&, const TensorGrad&);
		friend TensorGrad operator+(const Scalar, const TensorGrad&);
		//This Addition operation
		TensorGrad& operator+=(const TensorGrad& other);
		TensorGrad& operator+=(const Scalar other);
		TensorGrad& operator+=(const Tensor&);
		// Subtraction operation
		TensorGrad operator-(const TensorGrad& other) const;
		TensorGrad operator-(const Scalar other) const;
		TensorGrad operator-(const Tensor& other) const;
		friend TensorGrad operator-(const Tensor&, const TensorGrad&);
		friend TensorGrad operator-(const Scalar, const TensorGrad&);
		//This Subtraction operation
		TensorGrad& operator-=(const TensorGrad& other);
		TensorGrad& operator-=(const Tensor& other);
		TensorGrad& operator-=(const Scalar other);
		// Division operation
		TensorGrad operator/(const TensorGrad& other) const;
		TensorGrad operator/(const Scalar other) const;
		TensorGrad operator/(const Tensor& other) const;
		friend TensorGrad operator/(const Tensor&, const TensorGrad&);
		friend TensorGrad operator/(const Scalar, const TensorGrad&);
		//This division operation
		TensorGrad& operator/=(const TensorGrad& other);
		TensorGrad& operator/=(const Tensor& other);
		TensorGrad& operator/=(const Scalar other);
		// Multiplication operation
		TensorGrad operator*(const TensorGrad& other) const;
		TensorGrad operator*(const Scalar other) const;
		TensorGrad operator*(const Tensor& other) const;
		friend TensorGrad operator*(const Tensor&, const TensorGrad&);
		friend TensorGrad operator*(const Scalar, const TensorGrad&);
		//This multiplication operation
		TensorGrad& operator*=(const Tensor& other);
		TensorGrad& operator*=(const TensorGrad& other);
		TensorGrad& operator*=(const Scalar other);
		inline const nt::SizeRef& shape() const {return tensor.shape();}
		inline const size_t dims() const {return tensor.dims();}
		inline const size_value_t& numel() const {return tensor.numel();}
		template<typename... Args>
		inline TensorGrad view(int64_t i, Args&&... args) const{
			TensorGrad result(tensor.view(i, args...));
			result.track_grad(*this,
				[i, args...](nt::Tensor& grad){return grad.view(i, args...);}
			);
			return result;
		}
		TensorGrad view(SizeRef s) const;
		TensorGrad operator[](size_value_t) const;
		TensorGrad operator[](my_range) const;
		TensorGrad operator[](Tensor) const;
		TensorGrad operator[](std::vector<my_range>) const;
		TensorGrad& nullify();

		inline TensorGrad& operator++(){return *this += 1;}
		
		inline Tensor operator>=(const TensorGrad& t) const {return this->tensor >= t.tensor;}
		inline Tensor operator<=(const TensorGrad& t) const {return this->tensor <= t.tensor;}
		inline Tensor operator==(const TensorGrad& t) const {return this->tensor == t.tensor;}
		inline Tensor operator>=(const Tensor& t) const     {return this->tensor >= t;}
		inline Tensor operator<=(const Tensor& t) const     {return this->tensor <= t;}
		inline Tensor operator==(const Tensor& t) const     {return this->tensor == t;}
		inline Tensor operator>=(Scalar s) const            {return this->tensor >= s;}
		inline Tensor operator<=(Scalar s) const            {return this->tensor <= s;}
		inline Tensor operator==(Scalar s) const            {return this->tensor == s;}
		inline Tensor operator!=(Scalar s) const            {return this->tensor != s;}
		/* inline Tensor operator&&(Tensor t) const            {return this->tensor && t;} */
		/* inline Tensor operator||(Tensor t) const            {return this->tensor || t;} */

		inline Tensor operator>(const TensorGrad& t) const  {return this->tensor > t.tensor;}
		inline Tensor operator<(const TensorGrad& t) const  {return this->tensor < t.tensor;}
		inline Tensor operator>(const Tensor& t) const      {return this->tensor > t;}
		inline Tensor operator<(const Tensor& t) const      {return this->tensor < t;}
		inline Tensor operator>(Scalar s) const             {return this->tensor > s;}
		inline Tensor operator<(Scalar s) const             {return this->tensor < s;}
		inline TensorGrad operator-() const                 {return *this * -1;}

		inline TensorGrad& fill_(Scalar s)                  {return *this = s;}
		inline TensorGrad& fill_(const TensorGrad& val)     {return *this = val;}
		inline TensorGrad& fill_(const Tensor& val)         {return *this = val;}
		inline TensorGrad& add_(Scalar val)                 {return *this += val;}
		inline TensorGrad& add_(const TensorGrad& val)      {return *this += val;}
		inline TensorGrad& add_(const Tensor& val)          {return *this += val;}
		inline TensorGrad& subtract_(Scalar val)            {return *this -= val;}
		inline TensorGrad& subtract_(const TensorGrad& val) {return *this -= val;}
		inline TensorGrad& subtract_(const Tensor& val)     {return *this -= val;}
		inline TensorGrad& multiply_(Scalar val)            {return *this *= val;}
		inline TensorGrad& multiply_(const TensorGrad& val) {return *this *= val;}
		inline TensorGrad& multiply_(const Tensor& val)     {return *this *= val;}
		inline TensorGrad& divide_(Scalar val)              {return *this /= val;}
		inline TensorGrad& divide_(const TensorGrad& val)   {return *this /= val;}
		inline TensorGrad& divide_(const Tensor& val)       {return *this /= val;}
		inline Scalar toScalar() const                      {return this->tensor.toScalar();}
		template<typename T = Scalar>
		inline std::conditional_t< std::is_same_v<T, Scalar>, Scalar, T&> item() {
			if constexpr (std::is_same_v<T, Scalar>){
				return toScalar();
			}else{
				return this->tensor.item<T>();
			}
		}
		template<typename T = Scalar>
		inline std::conditional_t< std::is_same_v<T, Scalar>, Scalar, const T&> item() const {
			if constexpr (std::is_same_v<T, Scalar>){
				return toScalar();
			}else{
				return this->tensor.item<T>();
			}
		}
		inline bool is_contiguous() const                    {return this->tensor.is_contiguous();}
		inline bool is_empty() const                         {return this->tensor.is_empty();}
		inline bool is_null() const                          {return this->tensor.is_null();}
		inline int64_t contig_count() const                 {return this->tensor.contig_count();}
		inline std::vector<size_value_t> strides() const           {return this->tensor.strides();}
		inline std::vector<size_value_t> getChangedStrides() const {return this->tensor.getChangedStrides();}
		
		inline void print() const               {this->tensor.print();}
		inline void* data_ptr()                 {return this->tensor.data_ptr();}
		inline const void* data_ptr() const     {return this->tensor.data_ptr();}
		inline void* data_ptr_end()             {return this->tensor.data_ptr_end();}
		inline const void* data_ptr_end() const {return this->tensor.data_ptr_end();}
		inline bool occupy_same_tensor_memory(const TensorGrad& tg) const noexcept { 
			return this->tensor.occupy_same_memory(tg.tensor);
		}
		friend std::ostream& operator << (std::ostream &out, const TensorGrad&);
		

		TensorGrad unsqueeze(size_value_t dim = 0) const;
		TensorGrad unsqueeze_as(const Tensor&) const;
		TensorGrad unsqueeze_as(const SizeRef&) const;
		TensorGrad squeeze(utils::optional_list list = nullptr) const;
		TensorGrad flatten(size_value_t, size_value_t) const;
		TensorGrad unflatten(size_value_t, size_value_t) const;
		TensorGrad permute(std::vector<size_value_t>) const;
		TensorGrad transpose(size_value_t, size_value_t) const;
		TensorGrad unfold(size_value_t dim, size_value_t size, size_value_t step) const;
		TensorGrad split_axis(std::vector<my_range>) const;
		TensorGrad split_axis(size_value_t) const;
		TensorGrad split_axis_1() const;
		TensorGrad div(size_value_t) const;
		TensorGrad real() const;
		TensorGrad imag() const;
		TensorGrad to_complex_from_real() const;
		TensorGrad to_complex_from_imag() const;
		TensorGrad sum(utils::optional_list list = nullptr, bool keepdim=false) const;
		TensorGrad mean(utils::optional_list list = nullptr, bool keepdim=false) const;
		result_types::max<TensorGrad, Tensor> max() const;
		result_types::max<TensorGrad, Tensor> max(size_value_t dim) const;
		TensorGrad exp() const;
		TensorGrad& exp_();
		TensorGrad pow(Scalar) const;
		TensorGrad& inverse_();
		TensorGrad inverse() const;
		TensorGrad clip(Scalar, Scalar) const;
		TensorGrad& clip_(Scalar, Scalar);
		TensorGrad pad(std::vector<size_value_t> p, const char* mode = "constant", Scalar value = 0.0) const;
		TensorGrad unpad(std::vector<size_value_t> p) const;
        TensorGrad flip(utils::optional_list list = nullptr) const;
        TensorGrad flip_view(utils::optional_list list = nullptr) const;
		TensorGrad dilate(size_value_t) const;
		TensorGrad undilate(size_value_t) const;
		TensorGrad undilate_(size_value_t) const;
		TensorGrad repeat_(size_value_t amt) const;
		TensorGrad repeat_(size_value_t dim, size_value_t amt) const;
		TensorGrad expand(SizeRef) const;
		TensorGrad to_dtype(DType) const;
		TensorGrad to_device(DeviceType) const;
		inline TensorGrad to(DType dt) const {return to_dtype(dt);}
		inline TensorGrad to(DeviceType dt) const {return to_device(dt);}
		TensorGrad clone() const;
		TensorGrad contiguous() const;

		//still need to implemenet:
		//currently none

		inline Tensor& detach() noexcept {return this->tensor;}
		inline const Tensor& detach() const noexcept {return this->tensor;}
		inline TensorGrad unsqueeze_as(const TensorGrad& tg) const {return this->unsqueeze_as(tg.tensor);} 
		inline TensorGrad expand_as(const Tensor& t) const {return this->expand(t.shape());}
		inline TensorGrad expand_as(const TensorGrad& tg) const {return this->expand(tg.shape());}
	
		inline static TensorGrad makeNullTensorArray(int64_t num){ return TensorGrad(Tensor::makeNullTensorArray(num)); }
		void backward(const Tensor&);
		void backward();
		void zero_grad();
        void ensure_grads_initialized();
		Tensor grad_value() const;
		void update(); // updates current values based on gradient
		void update_mutable(); // updates current values based on gradient even if the tensor is immutable
        static void redefine_tracking(TensorGrad&, const TensorGrad&, std::function<void(const Tensor&, intrusive_ptr<TensorGrad>&)>, const char* func_name = __NT_FUNCTION_NAME__);
        //the underlying tensor, the function to update the gradient, the parents
        template<typename... Args>
        static TensorGrad make_tensor_grad(Tensor&, std::function<void(const Tensor&, std::vector<intrusive_ptr<TensorGrad>>&)>, const TensorGrad&, Args&&...);
        template<typename OGFunc>
        static TensorGrad make_view_grad(Tensor&, const TensorGrad&, OGFunc&&);
};

Tensor& operator+=(Tensor&, const TensorGrad&);
Tensor& operator-=(Tensor&, const TensorGrad&);
Tensor& operator*=(Tensor&, const TensorGrad&);
Tensor& operator/=(Tensor&, const TensorGrad&);


}//nt::


#include "TensorGrad.hpp"
#include "../functional/tensorgrad_get.h"
#endif //_NT_TENSOR_GRAD_H_
