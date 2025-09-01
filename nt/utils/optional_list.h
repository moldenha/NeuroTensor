#ifndef NT_UTILS_OPTIONAL_LIST_H__
#define NT_UTILS_OPTIONAL_LIST_H__

#include "../intrusive_ptr/intrusive_ptr.hpp"
#include "../refs/intrusive_list.h"
#include "optional_except.h"
#include <initializer_list>
#include "type_traits.h"
#include "api_macro.h"
#include <vector>
/* #include "../dtype/Scalar.h" //more compatible with every type as a scalar */


namespace nt{
namespace utils{


//this is an optional specifically to wrap either a list of elements, a single element, or no elements
class NEUROTENSOR_API optional_list{
	intrusive_ptr<intrusive_list<int64_t> > list;
	public:
		optional_list(const optional_list&);
		optional_list(optional_list&&);
		optional_list(const intrusive_ptr<intrusive_list<int64_t> >&);
		optional_list(intrusive_ptr<intrusive_list<int64_t> >&&);
        template<typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
		optional_list(std::initializer_list<T> l)
        :list(make_intrusive<intrusive_list<int64_t>>(l.size()))
        {
            int64_t* out_ptr = list->ptr();
            const T* begin = l.begin();
            const T* end = l.end();
            for(;begin != end; ++begin, ++out_ptr){
                *out_ptr = static_cast<int64_t>(*begin);
            }
        }
        template<typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
		optional_list(std::vector<T> l)
        :list(make_intrusive<intrusive_list<int64_t>>(l.size()))
        {
            int64_t* out_ptr = list->ptr();
            auto begin = l.cbegin();
            auto end = l.cend();
            for(;begin != end; ++begin, ++out_ptr){
                *out_ptr = static_cast<int64_t>(*begin);
            }
        }
		template<typename T, std::enable_if_t<std::is_same_v<T, std::nullptr_t>, bool> = true>
		optional_list(T element)
		:list(nullptr)
		{}
		template<typename T, std::enable_if_t<!std::is_same_v<T, std::nullptr_t> && std::is_convertible_v<T, int64_t>, bool> = true>
        optional_list(T element)
        : optional_list({static_cast<int64_t>(element)}) {}

		optional_list();
		optional_list& operator=(const optional_list&);
		optional_list& operator=(optional_list&&);
		optional_list& operator=(std::initializer_list<int64_t>);
		optional_list& operator=(const intrusive_ptr<intrusive_list<int64_t> > &);
		optional_list& operator=(intrusive_ptr<intrusive_list<int64_t> >&&);
		template<typename T, std::enable_if_t<std::is_same_v<T, std::nullptr_t>, bool> = true>
		inline optional_list& operator=(T element){
			if(*this){this->list.reset();}
			return *this;
		}
		template<typename T, std::enable_if_t<!std::is_same_v<T, std::nullptr_t> && std::is_convertible_v<T, int64_t>, bool> = true>
		inline optional_list& operator=(T element){
			return *this = {static_cast<int64_t>(element)};
		}
		
		inline bool has_value() const noexcept {return bool(list);}
		inline explicit operator bool() const noexcept {return has_value();}
		inline const intrusive_list<int64_t>* operator->() const noexcept{
			if(!has_value()){return nullptr;}
			return list.get();
		}
		inline intrusive_list<int64_t>* operator->() noexcept{
			if(!has_value()){return nullptr;}
			return list.get();
		}
		inline const intrusive_list<int64_t>& operator*() const& noexcept{
			return *list;
		}
		inline intrusive_list<int64_t>& operator*() & noexcept{
			return *list;
		}

		inline const int64_t& operator[](int64_t i) const& noexcept {
			return (*list)[i];
		}

		inline int64_t& operator[](int64_t i) & noexcept {
			return (*list)[i];
		}

		inline const intrusive_list<int64_t>& value() const&{
			if(!has_value()){throw bad_optional_access();}
			return *list;
		}
		inline intrusive_list<int64_t>& value() &{
			if(!has_value()){throw bad_optional_access();}
			return *list;
		}

		inline const int64_t& at(int64_t i) const& {
			if(!has_value()){throw bad_optional_access();}
            if(list->size() >= i){throw bad_optional_size_access();}
			return (*list)[i];
		}
		
		inline int64_t& at(int64_t i) & {
			if(!has_value()){throw bad_optional_access();}
            if(list->size() >= i){throw bad_optional_size_access();}
			return (*list)[i];
		}


		template<class U>
		inline intrusive_list<int64_t> value_or(U&& default_value) const& {
			return bool(*this) ? **this : intrusive_list<int64_t>(std::forward<U>(default_value));
		}
		template<class U>
		inline intrusive_list<int64_t> value_or(U&& default_value) && {
			return bool(*this) ? std::move(**this) : intrusive_list<int64_t>(std::forward<U>(default_value));
		}

		template<class F>
		inline auto and_then(F&& f) &{
			if (*this)
			    return std::invoke(std::forward<F>(f), **this);
			else
			    return ::nt::type_traits::remove_cvref_t<std::invoke_result_t<F, Tensor&>>{};
		}

		template<class F>
		inline auto and_then(F&& f) const&{
			if (*this)
			    return std::invoke(std::forward<F>(f), **this);
			else
			    return ::nt::type_traits::remove_cvref_t<std::invoke_result_t<F, const Tensor&>>{};
		}

		template<class F>
		inline auto and_then(F&& f) &&{
			if (*this)
			    return std::invoke(std::forward<F>(f), std::move(**this));
			else
			    return ::nt::type_traits::remove_cvref_t<std::invoke_result_t<F, intrusive_list<int64_t>>>{};
		}

		template<class F>
		inline auto and_then(F&& f) const&&{
			if (*this)
			    return std::invoke(std::forward<F>(f), std::move(**this));
			else
			    return ::nt::type_traits::remove_cvref_t<std::invoke_result_t<F, intrusive_list<int64_t>>>{};
		}
		inline void swap(optional_list& op) noexcept {op.list.swap(list);}
		inline void reset(){list.reset(); list = nullptr;}
		template<typename... Args>
		inline intrusive_list<int64_t>& emplace(Args&&... args){
			if(*this){
				reset();
				return emplace(std::forward<Args&&>(args)...);
			}
			this->list = make_intrusive<intrusive_list<int64_t> >(std::forward<Args&&>(args)...);
			return *this->list;
		}
		inline int64_t* begin() & noexcept {
			return list->begin();
		}
		inline const int64_t* begin() const& noexcept {
			return list->begin();
		}
		inline const int64_t* cbegin() const& noexcept {
			return list->cbegin();
		}

		inline int64_t* end() & noexcept {
			return list->end();
		}
		
		inline const int64_t* end() const& noexcept {
			return list->end();
		}

		inline const int64_t* cend() const& noexcept{
			return list->cend();
		}

		inline bool is_scalar() const& noexcept{
			return list->size() == 1;
		}

		inline const int64_t& get_scalar() const& noexcept {
			return (*this)[0];
		}
        
        inline std::vector<int64_t> to_vector() const& noexcept {
            if(!this->has_value()) return std::vector<int64_t>{};
            return std::vector<int64_t>(this->cbegin(), this->cend());
        }
        
        inline std::vector<int64_t> to_repeat_vector(int64_t i) const& {
			if(!has_value()){throw bad_optional_access();}
            if(!(is_scalar() || list->size() == i)){throw bad_optional_size_access();}
            if(is_scalar()){
                return std::vector<int64_t>(i, this->get_scalar());
            }
            return std::vector<int64_t>(this->cbegin(), this->cend());
        }
	
};


}} //nt::utils::

namespace std{
inline void swap(::nt::utils::optional_list& a, ::nt::utils::optional_list& b) noexcept {a.swap(b);}
} //std::

#endif
