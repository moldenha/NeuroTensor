//this is a class similar to std::any
//but dedicated to taking lvalue and rvalue references and handling them appropriately

#ifndef NT_ANY_REF_H__
#define NT_ANY_REF_H__
#include <iostream>
#include <memory>
#include <typeinfo>
#include <stdexcept>
#include <functional>
#include "type_traits.h"
#include "utils.h"
#include <vector>
#include "../dtype/Scalar.h"
#include "api_macro.h"

namespace nt{
namespace utils{
//this is a class specifically designed to handle referneces and non-references
//if used improperly, can be dangerous because it directly handles references
//if working with lvalue references, cannot go out of scope
//this would be illegal:
/*


//DO NOT DO THIS
any_ref return_lvalue(){
	float v;
	return any_ref(v); //automatically makes an lvalue
			   //would be returning a reference to a value out of scope
}

//THIS IS FINE
void process_vals(std::vector<any_ref> _vals){
	for(auto& val : _vals){
		val.cast<float&>() += 1.0;
	}
}

int main(){
	float a = 10;
	float b = 3;
	float c = 6;
	std::cout << "a: "<<a<<" b: "<<b<<" c: "<<c<<std::endl;
	std::vector<any_ref> _vals = make_any_ref_vector(a, b, c);
	process_vals(std::move(_vals));
	std::cout << "a: "<<a<<" b: "<<b<<" c: "<<c<<std::endl;
	return 0;
}


 */
class NEUROTENSOR_API any_ref {
private:
    struct BaseHolder {
        virtual ~BaseHolder() = default;
        inline virtual const std::type_info& type() const noexcept = 0;
        inline virtual std::unique_ptr<BaseHolder> clone() const = 0;
        inline virtual bool is_lvalue() const noexcept {return false;}
        inline virtual bool is_rvalue() const noexcept {return false;}
	inline virtual Scalar to_scalar() const noexcept {return Scalar();}
	inline virtual bool is_scalar() const noexcept {return false;}
    };

    template <typename T>
    struct RValHolder : BaseHolder {
        ::nt::type_traits::rvalue_wrapper<T> value;

        explicit RValHolder(::nt::type_traits::rvalue_wrapper<T> val) : value(val) {}
        explicit RValHolder(T&& v) : value(std::forward<T&&>(v)) {}
        RValHolder(T&) = delete;

        inline const std::type_info& type() const noexcept override {
            return typeid(T);
        }

        inline std::unique_ptr<BaseHolder> clone() const override {
            return std::make_unique<RValHolder<T>>(value);
        }
        inline bool is_rvalue() const noexcept override{return true;}
        inline Scalar to_scalar() const noexcept override{
            if constexpr (is_scalar_value_v<::nt::type_traits::remove_cvref_t<T>> || std::is_same_v<::nt::type_traits::remove_cvref_t<T>, Scalar>){
                return Scalar(value.t);
            }
            return Scalar();
        }
        inline bool is_scalar() const noexcept override{
            return is_scalar_value_v<::nt::type_traits::remove_cvref_t<T>> || std::is_same_v<::nt::type_traits::remove_cvref_t<T>, Scalar>;
        }
        
    };
    
    template <typename T>
    struct LValHolder : BaseHolder {
        std::reference_wrapper<T> value;

        explicit LValHolder(std::reference_wrapper<T> val) : value(val) {}
        explicit LValHolder(T& v) : value(this->to_ref(v)) {}
        LValHolder(T&&) = delete;

        inline const std::type_info& type() const noexcept override {
            return typeid(T);
        }

        inline std::unique_ptr<BaseHolder> clone() const override {
            return std::make_unique<LValHolder<T>>(value);
        }
        inline bool is_lvalue() const noexcept override{return true;}
        inline Scalar to_scalar() const noexcept override{
            if constexpr (is_scalar_value_v<::nt::type_traits::remove_cvref_t<T>> || std::is_same_v<::nt::type_traits::remove_cvref_t<T>, Scalar>){
                return Scalar(value.get());
            }
            return Scalar();
        }
        inline bool is_scalar() const noexcept override{
            return is_scalar_value_v<::nt::type_traits::remove_cvref_t<T>> || std::is_same_v<::nt::type_traits::remove_cvref_t<T>, Scalar>;
        }
        private:
        inline std::reference_wrapper<T> to_ref(T& val){
            if constexpr (std::is_const_v<T>){
                return std::cref(add_cv_like<const ::nt::type_traits::remove_cvref_t<T>&>(val));
            }else{
                return std::ref(add_cv_like<::nt::type_traits::remove_cvref_t<T>&>(val));
            }
        }
        
    };

    std::unique_ptr<BaseHolder> holder;

public:
    // Default constructor
    any_ref() = default;

    // Constructor to store a value
    template <typename T, std::enable_if_t<std::is_rvalue_reference_v<T> && std::is_const_v<T>, bool> = true>
    any_ref(T&& value) : holder(std::make_unique<RValHolder<const ::nt::type_traits::remove_cvref_t<T>> >(std::forward<T>(value))) {}
    
    template <typename T, std::enable_if_t<std::is_rvalue_reference_v<T> && !std::is_const_v<T>, bool> = true>
    any_ref(T&& value) : holder(std::make_unique<RValHolder<::nt::type_traits::remove_cvref_t<T>> >(std::forward<T>(value))) {}
    
    template <typename T, std::enable_if_t<std::is_lvalue_reference_v<T> && std::is_const_v<std::remove_reference_t<T>>, bool> = true>
    any_ref(T&& value) : holder(std::make_unique<LValHolder<const ::nt::type_traits::remove_cvref_t<T>> >(add_cv_like<const ::nt::type_traits::remove_cvref_t<T>&>(value))) {}
    
    template <typename T, std::enable_if_t<std::is_lvalue_reference_v<T> && !std::is_const_v<std::remove_reference_t<T>>, bool> = true>
    any_ref(T&& value) : holder(std::make_unique<LValHolder<::nt::type_traits::remove_cvref_t<T>> >(add_cv_like<::nt::type_traits::remove_cvref_t<T>&>(value))) {}
    
    template <typename T, std::enable_if_t<!std::is_reference_v<T>, bool> = true>
    any_ref(T&& value) : holder(std::make_unique<RValHolder<T> >(std::forward<T&&>(value))) {}
    

    // Copy constructor
    any_ref(const any_ref& other) : holder(other.holder ? other.holder->clone() : nullptr) {}

    // Move constructor
    any_ref(any_ref&& other) noexcept = default;

    ~any_ref() = default;

    inline any_ref& operator=(const any_ref& other) {
        if (this != &other) {
            holder = other.holder ? other.holder->clone() : nullptr;
        }
        return *this;
    }

    any_ref& operator=(any_ref&& other) noexcept = default;

    inline bool has_value() const noexcept {
        return holder != nullptr;
    }

    inline void reset() noexcept {
        holder.reset();
    }

    inline const std::type_info& type() const noexcept {
        return holder ? holder->type() : typeid(void);
    }
    
    inline bool is_lvalue() const noexcept {
        return holder ? holder->is_lvalue() : false;
    }
    inline bool is_rvalue() const noexcept {
        return holder ? holder->is_rvalue() : false;
    }

    inline bool is_scalar() const noexcept {
	return holder ? holder->is_scalar() : false;
    }

    inline Scalar to_scalar() const noexcept {
	return holder ? holder->to_scalar() : Scalar();
    }
    
    template<typename T>
    inline bool holds_type() const noexcept {
        return type() == typeid(T);
    }

    // Retrieve the value
    // Deals directly with references
    // lots of different cases, and handles scalars directly as well
    template <typename T>
    inline T cast() {
        if (!has_value()) {
            std::cerr << "any_ref does not hold a value" << std::endl;
            throw std::bad_cast();
            //throw std::bad_cast("any_ref does not hold a value");
        }
        if constexpr (std::is_lvalue_reference_v<T>){
            if(!holder->is_lvalue()){
                std::cerr << "trying to get lvalue reference from any_ref that does not hold an lvalue reference" << std::endl;
                throw std::bad_cast();
                //throw std::bad_cast("trying to get lvalue reference from any_ref that does not hold a reference");
            }
            if constexpr (std::is_const_v<T>){
                if(type() != typeid(::nt::type_traits::remove_cvref_t<T>) && type() != typeid(const ::nt::type_traits::remove_cvref_t<T>)){
                    std::cerr << "trying to get lvalue reference from any_ref that does not hold a reference of that type" << std::endl;
                    throw std::bad_cast();
                    //throw std::bad_cast("trying to get lvalue reference from any_ref that does not hold a reference of that type");
                }
                if(type() == typeid(::nt::type_traits::remove_cvref_t<T>)){
                    return static_cast<LValHolder<::nt::type_traits::remove_cvref_t<T> >*>(holder.get())->value.get();
                }
                else if(type() == typeid(const ::nt::type_traits::remove_cvref_t<T>)){
                    return static_cast<LValHolder<const ::nt::type_traits::remove_cvref_t<T> >*>(holder.get())->value.get();
                }
            }else{
                if(type() != typeid(::nt::type_traits::remove_cvref_t<T>)){
                    std::cerr << "trying to get a non-const lvalue reference from any_ref that does not hold a non-const reference of that type" << std::endl;
                    throw std::bad_cast();
                    //throw std::bad_cast("trying to get a non-const lvalue reference from any_ref that does not hold a non-const reference of that type");
                }
                return static_cast<LValHolder<::nt::type_traits::remove_cvref_t<T> >*>(holder.get())->value.get();
            }
        }else if constexpr (std::is_rvalue_reference_v<T>){
	    //a special case to handle scalars since r values will be coppied
	    if constexpr (is_scalar_value_v<::nt::type_traits::remove_cvref_t<T>> || std::is_same_v<Scalar,::nt::type_traits::remove_cvref_t<T>>){
		    if(!this->is_scalar()){
			std::cerr << "trying to get Scalar from any_ref that does not hold a Scalar" << std::endl;
                        throw std::bad_cast();
		    }
		    if constexpr (std::is_same_v<Scalar,::nt::type_traits::remove_cvref_t<T>>){
			return this->to_scalar();
		    }else{
			return this->to_scalar().to<::nt::type_traits::remove_cvref_t<T>>(); 
		    }
	    }
            if constexpr (!std::is_copy_constructible_v<::nt::type_traits::remove_cvref_t<T>>){
                if(!holder->is_rvalue()){
                    std::cerr << "trying to get rvalue reference from any_ref that does not hold an rvalue reference to a type that does not have a copy constructable" << std::endl;
                    throw std::bad_cast();
                    //throw std::bad_cast("trying to get lvalue reference from any_ref that does not hold a reference");
                }
            }
            if constexpr (std::is_const_v<T>){
                if(type() != typeid(::nt::type_traits::remove_cvref_t<T>) && type() != typeid(const ::nt::type_traits::remove_cvref_t<T>)){
                    std::cerr << "trying to get rvalue reference from any_ref that does not hold a reference of that type" << std::endl;
                    throw std::bad_cast();
                    //throw std::bad_cast("trying to get rvalue reference from any_ref that does not hold a reference of that type");
                }
                if(type() == typeid(::nt::type_traits::remove_cvref_t<T>)){
                    if constexpr (std::is_copy_constructible_v<::nt::type_traits::remove_cvref_t<T>>){
                        if(holder->is_lvalue()){
                            typename ::nt::type_traits::remove_cvref_t<T> val(static_cast<LValHolder<::nt::type_traits::remove_cvref_t<T> >*>(holder.get())->value.get());
                            if constexpr (std::is_move_constructible_v<::nt::type_traits::remove_cvref_t<T>>){
                                any_ref out(std::move(val));
                                return out.cast<T>();
                            }else{
                                any_ref out(std::forward<T&&>(val));
                                return out.cast<T>();
                            }
                        }
                    }
                    return static_cast<RValHolder<::nt::type_traits::remove_cvref_t<T> >*>(holder.get())->value();
                }
                else if(type() == typeid(const ::nt::type_traits::remove_cvref_t<T>)){
                    if constexpr (std::is_copy_constructible_v<::nt::type_traits::remove_cvref_t<T>>){
                        if(holder->is_lvalue()){
                            typename ::nt::type_traits::remove_cvref_t<T> val(static_cast<LValHolder<const ::nt::type_traits::remove_cvref_t<T> >*>(holder.get())->value.get());
                            if constexpr (std::is_move_constructible_v<::nt::type_traits::remove_cvref_t<T>>){
                                any_ref out(std::move(val));
                                return out.cast<T>();
                            }else{
                                any_ref out(std::forward<T&&>(val));
                                return out.cast<T>();
                            }
                        }
                    }
                    return static_cast<RValHolder<const ::nt::type_traits::remove_cvref_t<T> >*>(holder.get())->value();
                }
            }else{
                if constexpr (std::is_copy_constructible_v<::nt::type_traits::remove_cvref_t<T>>){
                    if(holder->is_lvalue()){
                        typename ::nt::type_traits::remove_cvref_t<T> val(static_cast<LValHolder<::nt::type_traits::remove_cvref_t<T> >*>(holder.get())->value.get());
                        if constexpr (std::is_move_constructible_v<::nt::type_traits::remove_cvref_t<T>>){
                            any_ref out(std::move(val));
                            return out.cast<T>();
                        }else{
                            any_ref out(std::forward<T&&>(val));
                            return out.cast<T>();
                        }
                    }
                }
                return static_cast<RValHolder<::nt::type_traits::remove_cvref_t<T> >*>(holder.get())->value();
            }
        }else{
            //just regular values
	    //a special case to handle scalars and copy them to regular values
	    // if constexpr (std::is_same_v<Scalar,::nt::type_traits::remove_cvref_t<T>>){
	    if constexpr (is_scalar_value_v<::nt::type_traits::remove_cvref_t<T>> || std::is_same_v<Scalar,::nt::type_traits::remove_cvref_t<T>>){
		    if(!this->is_scalar()){
			std::cerr << "trying to get Scalar from any_ref that does not hold a Scalar" << std::endl;
                        throw std::bad_cast();
		    }
		    if constexpr (std::is_same_v<Scalar,::nt::type_traits::remove_cvref_t<T>>){
			return this->to_scalar();
		    }else{
			return this->to_scalar().to<::nt::type_traits::remove_cvref_t<T>>(); 
		    }
	    }

            if(type() != typeid(::nt::type_traits::remove_cvref_t<T>) && type() != typeid(const ::nt::type_traits::remove_cvref_t<T>)){
                std::cerr << "trying to get a value from any_ref that does not hold a value of that type" << std::endl;
                throw std::bad_cast();
                //throw std::bad_cast("trying to get a value from any_ref that does not hold a value of that type");
            }
            if(type() == typeid(::nt::type_traits::remove_cvref_t<T>)){
                if(holder->is_lvalue()){
                    return static_cast<LValHolder<::nt::type_traits::remove_cvref_t<T> >*>(holder.get())->value.get();
                }else if(holder->is_rvalue()){
                    return static_cast<RValHolder<::nt::type_traits::remove_cvref_t<T> >*>(holder.get())->value();
                }
            }
            if(holder->is_lvalue()){
                return static_cast<LValHolder<const ::nt::type_traits::remove_cvref_t<T> >*>(holder.get())->value.get();
            }
            return static_cast<RValHolder<const ::nt::type_traits::remove_cvref_t<T> >*>(holder.get())->value();
        }
    }

    //this is a function to get an lvalue reference to the value inside of any_ref
    template<typename T>
    friend T& cast_lvalue(any_ref&);
    
    
};

template<typename... Args>
inline std::vector<any_ref> make_any_ref_vector(Args&&... vals){
	std::vector<any_ref> out;
	out.reserve(sizeof...(Args));
	((out.push_back(any_ref(std::forward<Args&&>(vals)))), ...);
	return std::move(out);
}

template<typename T>
inline T& cast_lvalue(any_ref& ref){
	if(ref.type() != typeid(::nt::type_traits::remove_cvref_t<T>) && ref.type() != typeid(const ::nt::type_traits::remove_cvref_t<T>)){
		std::cerr << "trying to get a value from any_ref that does not hold a value of that type" << std::endl;
        throw std::bad_cast();
	}
	if(ref.is_lvalue()){
		if(typeid(::nt::type_traits::remove_cvref_t<T>) == ref.type()){
			return static_cast<any_ref::LValHolder<::nt::type_traits::remove_cvref_t<T> >*>(ref.holder.get())->value.get();
		}
		return const_cast<::nt::type_traits::remove_cvref_t<T>&>(static_cast<any_ref::LValHolder<const ::nt::type_traits::remove_cvref_t<T> >*>(ref.holder.get())->value.get());
	}
	//ref is an r value
	if(typeid(::nt::type_traits::remove_cvref_t<T>) == ref.type()){
		return static_cast<any_ref::RValHolder<::nt::type_traits::remove_cvref_t<T> >*>(ref.holder.get())->value.t;
	}
	return const_cast<::nt::type_traits::remove_cvref_t<T>&>(static_cast<any_ref::RValHolder<const ::nt::type_traits::remove_cvref_t<T> >*>(ref.holder.get())->value.t);
}

}} //nt::utils::

#endif //_NT_ANY_REF_H_
