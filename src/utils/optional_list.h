#ifndef _NT_UTILS_OPTIONAL_LIST_H_
#define _NT_UTILS_OPTIONAL_LIST_H_

#include "../intrusive_ptr/intrusive_ptr.hpp"
#include "../refs/intrusive_list.h"
#include "optional_except.h"
#include <initializer_list>
#include "type_traits.h"
/* #include "../dtype/Scalar.h" //more compatible with every type as a scalar */


namespace nt{
namespace utils{


//this is an optional specifically to wrap either a list of elements, a single element, or no elements
class optional_list{
	intrusive_ptr<intrusive_list<int64_t> > list;
	public:
		optional_list(const optional_list&);
		optional_list(optional_list&&);
		optional_list(const intrusive_ptr<intrusive_list<int64_t> >&);
		optional_list(intrusive_ptr<intrusive_list<int64_t> >&&);
		optional_list(std::initializer_list<int64_t>);
		template<typename T, std::enable_if_t<std::is_same_v<T, std::nullptr_t>, bool> = true>
		optional_list(T element)
		:list(nullptr)
		{}
		template<typename T, std::enable_if_t<!std::is_same_v<T, std::nullptr_t>, bool> = true>
		optional_list(T element)
		:optional_list({static_cast<int64_t>(element)})
		{}

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
		template<typename T, std::enable_if_t<!std::is_same_v<T, std::nullptr_t>, bool> = true>
		inline optional_list& operator=(T element){
			return *this = {static_cast<int64_t>(element)};
		}
		
		inline constexpr bool has_value() const noexcept {return bool(list);}
		inline constexpr explicit operator bool() const noexcept {return has_value();}
		inline constexpr const intrusive_list<int64_t>* operator->() const noexcept{
			if(!has_value()){return nullptr;}
			return list.get();
		}
		inline constexpr intrusive_list<int64_t>* operator->() noexcept{
			if(!has_value()){return nullptr;}
			return list.get();
		}
		inline constexpr const intrusive_list<int64_t>& operator*() const& noexcept{
			return *list;
		}
		inline constexpr intrusive_list<int64_t>& operator*() & noexcept{
			return *list;
		}

		inline const int64_t& operator[](int64_t i) const& noexcept {
			return (*list)[i];
		}

		inline int64_t& operator[](int64_t i) & noexcept {
			return (*list)[i];
		}

		inline constexpr const intrusive_list<int64_t>& value() const&{
			if(!has_value()){throw bad_optional_access();}
			return *list;
		}
		inline constexpr intrusive_list<int64_t>& value() &{
			if(!has_value()){throw bad_optional_access();}
			return *list;
		}

		inline const int64_t& at(int64_t i) const& {
			if(!has_value()){throw bad_optional_access();}
			return (*list)[i];
		}
		
		inline int64_t& at(int64_t i) & {
			if(!has_value()){throw bad_optional_access();}
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
		inline constexpr auto and_then(F&& f) &{
			if (*this)
			    return std::invoke(std::forward<F>(f), **this);
			else
			    return std::remove_cvref_t<std::invoke_result_t<F, Tensor&>>{};
		}

		template<class F>
		inline constexpr auto and_then(F&& f) const&{
			if (*this)
			    return std::invoke(std::forward<F>(f), **this);
			else
			    return std::remove_cvref_t<std::invoke_result_t<F, const Tensor&>>{};
		}

		template<class F>
		inline constexpr auto and_then(F&& f) &&{
			if (*this)
			    return std::invoke(std::forward<F>(f), std::move(**this));
			else
			    return std::remove_cvref_t<std::invoke_result_t<F, intrusive_list<int64_t>>>{};
		}

		template<class F>
		inline constexpr auto and_then(F&& f) const&&{
			if (*this)
			    return std::invoke(std::forward<F>(f), std::move(**this));
			else
			    return std::remove_cvref_t<std::invoke_result_t<F, intrusive_list<int64_t>>>{};
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
		inline constexpr int64_t* begin() & noexcept {
			return list->begin();
		}
		inline constexpr const int64_t* begin() const& noexcept {
			return list->begin();
		}
		inline constexpr const int64_t* cbegin() const& noexcept {
			return list->cbegin();
		}

		inline constexpr int64_t* end() & noexcept {
			return list->end();
		}
		
		inline constexpr const int64_t* end() const& noexcept {
			return list->end();
		}

		inline constexpr const int64_t* cend() const& noexcept{
			return list->cend();
		}

		inline constexpr bool is_scalar() const& noexcept{
			return list->size() == 1;
		}

		inline constexpr const int64_t& get_scalar() const& noexcept {
			return (*this)[0];
		}
	
};


}} //nt::utils::

namespace std{
inline void swap(nt::utils::optional_list& a, nt::utils::optional_list& b) noexcept {a.swap(b);}
} //std::

#endif
