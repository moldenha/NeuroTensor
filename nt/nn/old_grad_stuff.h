/* inline intrusive_ptr<tensor_holder>& operator=(intrusive_ptr<tensor_holder>&
 * it, Tensor&& t) noexcept { */
/* 	it = make_intrusive<tensor_holder>(t); */
/* 	return it; */
/* } */

// class NEUROTENSOR_API intrusive_vector_tg : public intrusive_ptr_target{
// 	std::vector<intrusive_ptr<TensorGrad> > vec;
// 	public:
// 		intrusive_vector_tg() = default;
// 		inline size_t size() const& {
// 			return vec.size();
// 		}
// 		inline intrusive_ptr<TensorGrad>& at(uint32_t i){return vec[i];}
// 		inline const intrusive_ptr<TensorGrad>& at(uint32_t i) const {return
// vec[i];} 		inline void push_back(intrusive_ptr<TensorGrad> val){
// 			vec.push_back(val);
// 		}
// 		inline void push_back(const TensorGrad& val){
// 			vec.push_back(make_intrusive<TensorGrad>(val));
// 		}
// 		inline void push_back(intrusive_ptr<TensorGrad> val) const {
// 			const_cast<std::vector<intrusive_ptr<TensorGrad>
// >&>(vec).push_back(val);
// 		}
// 		inline void push_back(const TensorGrad& t) const {
// 			const_cast<std::vector<intrusive_ptr<TensorGrad>
// >&>(vec).push_back(make_intrusive<TensorGrad>(t));
// 		}
// 		inline void clear() {vec.clear();}
// 		inline void remove(uint32_t i){
//             if(i < vec.size())
//                 vec.erase(vec.begin() + i);
// 		}
// 		inline std::vector<intrusive_ptr<TensorGrad> >::iterator begin() {return
// vec.begin();} 		inline std::vector<intrusive_ptr<TensorGrad> >::iterator end()
// {return vec.end();} 		inline std::vector<intrusive_ptr<TensorGrad>
// >::const_iterator begin() const {return vec.begin();} 		inline
// std::vector<intrusive_ptr<TensorGrad> >::const_iterator end() const {return
// vec.begin();} 		inline bool in(intrusive_ptr<TensorGrad>& t) const noexcept{
// 			for(auto& v : vec){
// 				if(v == t){return true;}
// 			}
// 			return false;
// 		}

// };

class NEUROTENSOR_API tensor_grad_vec : public intrusive_ptr_target {
    std::vector<intrusive_ptr<TensorGrad>> vec;

  public:
    using vec_type = typename std::vector<intrusive_ptr<TensorGrad>>;
    tensor_grad_vec() = default;
    ~tensor_grad_vec();
    inline vec_type &get() noexcept { return vec; }
    void clear();
    inline void push_back(intrusive_ptr<TensorGrad> &gr) { vec.push_back(gr); }
    template <typename... T> inline void emplace_back(T &&...items) {
        vec.push_back(make_intrusive<TensorGrad>(std::forward<T &&>(items)...));
    }
    inline vec_type::size_type size() const { return vec.size(); }
    // should be assessed in reverse in order to have correct calculations
    inline vec_type::reverse_iterator begin() { return vec.rbegin(); }
    inline vec_type::reverse_iterator end() { return vec.rend(); }
    inline intrusive_ptr<TensorGrad> &back() { return vec.back(); }
    inline const intrusive_ptr<TensorGrad> &back() const { return vec.back(); }
};



class NEUROTENSOR_API intrusive_back_func : public intrusive_ptr_target {
  public:
    using function_type = std::function<void(
        const Tensor &, std::vector<intrusive_ptr<TensorGrad>> &)>;
    using function_type_b = std::function<void(
        const Tensor &, std::vector<intrusive_ptr<TensorGrad>> &, bool)>;

  private:
    std::variant<std::monostate, function_type, function_type_b> Func;
    mutable bool _used;
    std::string name;

  public:
    intrusive_back_func()
        : Func(std::monostate{}), _used(false), name("NoneBackward") {
        utils::throw_exception(Func.index() == 0,
                               "Loaded a function type into backward function "
                               "and index was expected to be 0 but got $",
                               Func.index());
    }
    intrusive_back_func(std::string _name)
        : Func(std::monostate{}), _used(false), name(_name + "Backward") {
        utils::throw_exception(Func.index() == 0,
                               "Loaded a function type into backward function "
                               "and index was expected to be 0 but got $",
                               Func.index());
        name[0] = std::toupper(name[0]);
    }
    intrusive_back_func(function_type func, std::string _name)
        : Func(func), _used(false), name(_name + "Backward") {
        utils::throw_exception(Func.index() == 1,
                               "Loaded a function type into backward function "
                               "and index was expected to be 1 but got $",
                               Func.index());
        name[0] = std::toupper(name[0]);
    }
    intrusive_back_func(function_type_b func, std::string _name)
        : Func(func), _used(false), name(_name + "Backward") {
        utils::throw_exception(Func.index() == 2,
                               "Loaded a function type into backward function "
                               "and index was expected to be 2 but got $",
                               Func.index());
        name[0] = std::toupper(name[0]);
    }
    /* inline function_type& get() noexcept {return Func;} */
    /* inline const function_type& get() const noexcept {return Func;} */
    inline void set(function_type func) noexcept {
        Func = func;
        _used = false;
    }
    inline void set(function_type_b func) noexcept {
        Func = func;
        _used = false;
    }
    inline void set(std::nullptr_t) noexcept {
        Func = std::monostate{};
        _used = false;
    }
    inline void set_name(std::string _name) noexcept {
        name = _name + "Backward";
        name[0] = std::toupper(name[0]);
    }
    inline const std::string &get_name() const noexcept { return name; }
    // inline void clear() noexcept {Func = std::monostate{}; _has_been_cleared
    // = true;}
    inline size_t index() const noexcept { return Func.index(); }
    inline const bool &used() const noexcept { return _used; }
    inline void un_use() const noexcept { _used = false; }
    inline void set_used() const noexcept { _used = true; }
    inline void run(const Tensor &t,
                    std::vector<intrusive_ptr<TensorGrad>> &v) {
        utils::throw_exception(
            _used == false,
            "Backward function already used, graph constructed improperly");
        if (std::monostate *f = std::get_if<std::monostate>(&Func)) {
            _used = true;
            return;
            // utils::throw_exception(false, "Tried to run invalid function");
        } else if (function_type *f = std::get_if<function_type>(&Func)) {
            utils::throw_exception(
                *f != nullptr, "Trying to run invalid function, was nullptr");
            (*f)(t, v);
        } else if (function_type_b *f = std::get_if<function_type_b>(&Func)) {
            utils::throw_exception(
                *f != nullptr, "Trying to run invalid function, was nullptr");
            (*f)(t, v, false);

        } else {
            throw std::bad_variant_access();
        }
        _used = true;
    }
    inline void run(const Tensor &t, std::vector<intrusive_ptr<TensorGrad>> &v,
                    bool b) {
        utils::throw_exception(
            _used == false,
            "Backward function already used, graph constructed improperly");
        if (std::monostate *f = std::get_if<std::monostate>(&Func)) {
            _used = true;
            return;
            // utils::throw_exception(false, "Tried to run invalid function");
        } else if (function_type *f = std::get_if<function_type>(&Func)) {
            utils::throw_exception(
                *f != nullptr, "Trying to run invalid function, was nullptr");
            (*f)(t, v);
        } else if (function_type_b *f = std::get_if<function_type_b>(&Func)) {
            utils::throw_exception(
                *f != nullptr, "Trying to run invalid function, was nullptr");
            (*f)(t, v, b);

        } else {
            throw std::bad_variant_access();
        }
        _used = true;
    }

    inline bool is_valid() const noexcept {
        if (const std::monostate *f = std::get_if<std::monostate>(&Func)) {
            return false;
        } else if (const function_type *f = std::get_if<function_type>(&Func)) {
            if (*f == nullptr) {
                return false;
            }
            return true;
        } else if (const function_type_b *f =
                       std::get_if<function_type_b>(&Func)) {
            if (*f == nullptr) {
                return false;
            }
            return true;
        }
        return false;
    }
};


class weak_tensor_grad_vec;

struct NEUROTENSOR_API WeakTensorGrad : public intrusive_ptr_target{
    weak_intrusive_ptr<tensor_holder> tensor;
    weak_intrusive_ptr<tensor_holder> grad;
    weak_intrusive_ptr<intrusive_back_func> backwardFunc;
    weak_intrusive_ptr<tensor_grad_vec> parents;
    weak_intrusive_ptr<tensor_grad_vec> children;
    WeakTensorGrad(const TensorGrad& tg);
    intrusive_ptr<TensorGrad> to_intrusive_tg();
};


class weak_tensor_grad_vec : public intrusive_ptr_target {
    std::vector<intrusive_ptr<WeakTensorGrad>> vec;

  public:
    using vec_type = typename std::vector<intrusive_ptr<WeakTensorGrad>>;
    weak_tensor_grad_vec() = default;
    inline ~weak_tensor_grad_vec(){
        for(intrusive_ptr<WeakTensorGrad>& ptr : this->vec){
            ptr.reset();
            ptr = nullptr;
        } 
    }
    inline vec_type &get() noexcept { return vec; }
    inline void clear(){
        for(intrusive_ptr<WeakTensorGrad>& ptr : this->vec){
            ptr.reset();
            ptr = nullptr;
        } 
    }
    inline void push_back(intrusive_ptr<WeakTensorGrad> &gr) {
        vec.push_back(gr);
    }
    inline void emplace_back(const TensorGrad &gr) {
        vec.push_back(make_intrusive<WeakTensorGrad>(gr));
    }

    inline vec_type::size_type size() const { return vec.size(); }
    // should be assessed in reverse in order to have correct calculations
    inline vec_type::reverse_iterator begin() { return vec.rbegin(); }
    inline vec_type::reverse_iterator end() { return vec.rend(); }
    inline intrusive_ptr<WeakTensorGrad> &back() { return vec.back(); }
    inline const intrusive_ptr<WeakTensorGrad> &back() const {
        return vec.back();
    }
};

