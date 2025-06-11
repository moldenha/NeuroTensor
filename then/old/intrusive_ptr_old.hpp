#ifndef INTRUSIVE_PTR_HPP
#define INTRUSIVE_PTR_HPP

#include <_types/_uint32_t.h>
#include <_types/_uint8_t.h>
#include <atomic>
#include <memory>
#include <sys/_types/_key_t.h>
#include <type_traits>
#include <cassert>
#include <functional>
#include <cstdlib>
#include <iostream>
#include <sys/shm.h>
#include <sys/ipc.h>
#include "../utils/utils.h"
#include <cstdlib> // For std::aligned_alloc
#include <immintrin.h>

#ifdef __AVX512F__
#define ALIGN_BYTE_SIZE 64
#elif defined(__AVX2__)
#define ALIGN_BYTE_SIZE 32
#elif defined(__AVX__)
#define ALIGN_BYTE_SIZE 32
#else
#define ALIGN_BYTE_SIZE 16
#endif

/*

template<typename T>
class target{
	T* ptr;
	mutable std::atomic<uint32_t> count;
	public:
		target() : ptr(nullptr), count(0) {}
		target(T* p) : ptr(p), count(1) {}
};

*/


namespace nt{

class intrusive_ptr_target;

namespace detail{
	template <class TTarget>
	struct intrusive_target_default_null_type final{
		static constexpr TTarget* singleton() noexcept{
			return nullptr;
		}
	};
	struct DontIncreaseRefCount {};
	struct DontOrderStrides {};
	//increment needs to be aquire-release to make use_count() and
	//unique() reliable
	inline uint32_t atomic_refcount_increment(std::atomic<int64_t>& refcount){
		return refcount.fetch_add(1, std::memory_order_acq_rel) + 1;
	}

	inline uint32_t atomic_refcount_decrement(std::atomic<int64_t>& refcount){
		return refcount.fetch_sub(1, std::memory_order_acq_rel) - 1;
	}
	
	template<typename To, typename From>
	inline std::function<void(To*)> change_function_input(const std::function<void(From*)>& func){
		return std::function<void(To*)>([&func](To* ptr){func(reinterpret_cast<From*>(ptr));});
	}

	template<class TTarget, class ToNullType, class FromNullType>
	inline TTarget* assign_ptr_(TTarget* rhs){
		if(FromNullType::singleton() == rhs){
			return ToNullType::singleton();
		}
		return rhs;
	}

	/* template<class TTarget, class FromTarget, class ToNullType, class FromNullType> */
	/* inline TTarget* */ 
	constexpr uint32_t kImpracticallyHugeReferenceCount = 0x0FFFFFFF;
	namespace intrusive_ptr{
		inline void incref(intrusive_ptr_target* self);
	}

	template<typename T>
	struct is_pointer { static constexpr bool value = false; };

	template<typename T>
	struct is_pointer<T*> { static constexpr bool value = true; };
	enum class Device{
		SharedCPU,
		CPU
	}; // to come will be cuda, and mlx
}


class intrusive_ptr_target{
	mutable std::atomic<int64_t> refcount_;

	template<typename T, typename TTarget, typename NullType, typename TNullType>
	friend class intrusive_ptr;
	friend inline void detail::intrusive_ptr::incref(intrusive_ptr_target *self);

	
	virtual void release_resources() {}

	protected:
	
	virtual ~intrusive_ptr_target(){
		utils::throw_exception(refcount_.load() == 0 || refcount_.load() >= detail::kImpracticallyHugeReferenceCount,
				"needed refcount to be too high or at 0 in order to release resources");
		release_resources();
	}
	
	constexpr intrusive_ptr_target() : refcount_(0) {}

	// intrusive_ptr_target supports copy and move: but refcount and weakcount
	// don't participate (since they are intrinsic properties of the memory
	// location)
	intrusive_ptr_target(intrusive_ptr_target&& rhs) noexcept
		{}

	intrusive_ptr_target(const intrusive_ptr_target& rhs) noexcept
		{}

	intrusive_ptr_target& operator=(const intrusive_ptr_target& rhs) noexcept {
		return *this;
	}

	intrusive_ptr_target& operator=(intrusive_ptr_target&& rhs) noexcept {
		return *this;
	}

};



//that way both the pointer and refcount can be defined
template<
	class T, 
	class TTarget = intrusive_ptr_target,
	class NullType = detail::intrusive_target_default_null_type<T>,
	class TNullType = detail::intrusive_target_default_null_type<TTarget> >
class intrusive_ptr final{
	template<class T2, class TTarget2, class NullType2, class TNullType2>
	friend class intrusive_ptr;

#ifndef _WIN32
	static_assert(
			NullType::singleton() == NullType::singleton(),
			"NullType must have a constexpr singleton method"
		     );
	static_assert(
			TNullType::singleton() == TNullType::singleton(),
			"NullType must have a constexpr singleton method"
		     );
#endif

	static_assert(
			std::is_base_of_v<TTarget,
				std::remove_pointer_t<decltype(TNullType::singleton())>>,
				"TNullType::singleton() must return an target_type* pointer");

	TTarget* target_;
	T* ptr_;
	std::function<void(T*)> deallocate_;
	detail::Device device_;

	//as a way to protect from , these are the onl 
	inline bool is_null(){
		return ptr_ == NullType::singleton() || target_ == TNullType::singleton();
	}
	inline bool both_null(){
		return ptr_ == NullType::singleton() && target_ == TNullType::singleton();
	}
	inline bool one_null(){
		return !both_null() && is_null();
	}
	inline void null_self() noexcept{
		target_ = TNullType::singleton();
		ptr_ = NullType::singleton();
	}
	inline void check_make_null(){
		if(is_null()){
			if(both_null())
				return;
			else if(ptr_ == NullType::singleton())
				delete target_;
			else if(target_ == TNullType::singleton())
				deallocate_(ptr_);
			null_self();
		}
	}
	void reset_(){
		if(target_ != TNullType::singleton()
				&& detail::atomic_refcount_decrement(target_->refcount_) == 0){
			if(ptr_ != NullType::singleton())
				deallocate_(ptr_);
			delete target_;
			return;
		}
		null_self();
	}
	inline void retain_(){
		if(target_ != TNullType::singleton()){
			uint32_t new_count = detail::atomic_refcount_increment(target_->refcount_);
			utils::throw_exception(new_count != 1, "intrusive_ptr: cannot increase refcount after it reaches zero");
		}
	}

	
	explicit intrusive_ptr(T* ptr, TTarget* target, std::function<void(T*)> dealc = [](T* ptr){std::free(ptr);}, detail::Device dev = detail::Device::CPU)
		: intrusive_ptr(ptr, target, dealc, dev, detail::DontIncreaseRefCount{}) 
	{
		check_make_null();
		if(!is_null()){
			target_->refcount_.store(1, std::memory_order_relaxed);
		}
	}

	explicit intrusive_ptr(T* ptr)
		:target_(ptr == NullType::singleton() ? TNullType::singleton() : new TTarget()), 
		ptr_(ptr),
		deallocate_([](T* ptr){std::free(ptr);}),
		device_(detail::Device::CPU)
		{
			check_make_null();
			if(!is_null()){
				target_->refcount_.store(1, std::memory_order_relaxed);
			}
		}

	public:
		using element_type = T;
		using target_type = TTarget;
		intrusive_ptr() noexcept
			:intrusive_ptr(NullType::singleton(), 
					TNullType::singleton(), 
					[](T* ptr){std::free(ptr);}, 
					detail::Device::CPU, detail::DontIncreaseRefCount{})
			{}

		intrusive_ptr(std::nullptr_t) noexcept
			:intrusive_ptr(NullType::singleton(), 
					TNullType::singleton(), 
					[](T* ptr){std::free(ptr);}, 
					detail::Device::CPU, detail::DontIncreaseRefCount{})
			{}
		
		

		explicit intrusive_ptr(T* ptr, 
				TTarget* target, 
				std::function<void(T*)> dealc, 
				detail::Device dev, 
				detail::DontIncreaseRefCount) noexcept
			:target_(target), ptr_(ptr), deallocate_(dealc), device_(dev)
		{check_make_null();}
		
		explicit intrusive_ptr(uint32_t amt)
			:intrusive_ptr(static_cast<T*>(std::malloc(amt * sizeof(T))))
		{utils::throw_exception(ptr_ != NullType::singleton(), "Failure to allocate ptr_, T*");}

		explicit intrusive_ptr(std::unique_ptr<T> rhs) noexcept
			: intrusive_ptr(rhs.release()) {}
		
		template<class From>
		explicit intrusive_ptr(std::unique_ptr<From> rhs) noexcept
			:intrusive_ptr(reinterpret_cast<T*>(rhs.release()))
		{}

		inline ~intrusive_ptr() noexcept { reset_();}
		
		inline intrusive_ptr(intrusive_ptr&& rhs) noexcept
			:target_(rhs.target_), ptr_(rhs.ptr_), deallocate_(rhs.deallocate_), device_(rhs.device_)
			{rhs.null_self();}


		template<typename From, typename FromTarget, typename FromNull, typename FromNullT>
		intrusive_ptr(intrusive_ptr<From, FromTarget, FromNull, FromNullT>&& rhs) noexcept
			:target_(detail::assign_ptr_<TTarget, TNullType, FromNullT>(rhs.target_)),
			ptr_(reinterpret_cast<T*>(rhs.ptr_)),
			deallocate_(detail::change_function_input<T, From>(rhs.deallocate_)),
			device_(rhs.device_)
		{
			static_assert(
				std::is_convertible<FromTarget*, TTarget*>::value,
				"Move warning, intrusive_ptr got pointer of wrong type");
			rhs.null_self();
		}



		inline intrusive_ptr(const intrusive_ptr& other) 
			: target_(other.target_), ptr_(other.ptr_), deallocate_(other.deallocate_), device_(other.device_) 
		{
			retain_();
		}

		template<typename From, typename FromTarget, typename FromNullType, typename FromNullT>
		intrusive_ptr(const intrusive_ptr<From, FromTarget, FromNullType, FromNullT>& rhs) noexcept
			:target_(detail::assign_ptr_<TTarget, TNullType, FromNullT>(rhs.target_)),
			ptr_(reinterpret_cast<T*>(rhs.ptr_)),
			deallocate_(detail::change_function_input<T, From>(rhs.deallocate_)),
			device_(rhs.device_)

		{
			static_assert(
				std::is_convertible<FromTarget*, TTarget*>::value,
				"Move warning, intrusive_ptr got pointer of wrong type");
			retain_();

		}

		intrusive_ptr& operator=(intrusive_ptr&& rhs) & noexcept {
			//std::cout << "move = operator called for intrusive_ptr<T>"<<std::endl;
			target_ = rhs.target_;
			ptr_ = rhs.ptr_;
			deallocate_ = rhs.deallocate_;
			device_ = rhs.device_;
			rhs.null_self();
			/* return operator= <TTarget, NullType>(std::move(rhs)); */
			return *this;
		}

		template<typename From, typename FromTarget, typename FromNullType, typename FromNullT>
		intrusive_ptr& operator=(intrusive_ptr<From, FromTarget, FromNullType, FromNullT>&& rhs) & noexcept{
			//std::cout << "move = operator called for intrusive_ptr<others->T>"<<std::endl;
			static_assert(
				std::is_convertible<FromTarget*, TTarget*>::value,
				"Move warning, intrusive_ptr got pointer of wrong type");

			intrusive_ptr tmp(std::move(rhs));
			swap(tmp);
			return *this;
		}
		


		intrusive_ptr& operator=(const intrusive_ptr& rhs) & noexcept{
			//std::cout << "copy = operator called for intrusive_ptr<T>"<<std::endl;
			if(this != &rhs){
				reset_();
				target_ = rhs.target_;
				ptr_ = rhs.ptr_;
				device_ = rhs.device_;
				retain_();
			}
			return *this;
		}
		
		template<typename From, typename FromTarget, typename FromNullType, typename FromNullT>
		intrusive_ptr& operator=(const intrusive_ptr<From, FromTarget, FromNullType, FromNullT>& rhs) & noexcept{
			//std::cout << "copy = operator called for intrusive_ptr<others->T>"<<std::endl;
			static_assert(
				std::is_convertible<FromTarget*, TTarget*>::value,
				"Copy warning, intrusive_ptr got pointer of wrong type");
			intrusive_ptr tmp(rhs);
			swap(tmp);
			return *this;
		}

		inline T* operator->() const noexcept{
			return ptr_;
		}
		inline T& operator*() const noexcept{
			return *ptr_;
		}
		inline T* get() const noexcept{
			return ptr_;
		}
		inline T& operator[](const std::ptrdiff_t i) const noexcept{
			return ptr_[i];
		}
		inline const detail::Device& device() const noexcept{ return device_;}
		inline const bool is_shared() const noexcept {return device_ == detail::Device::SharedCPU;}
		inline const bool is_cpu() const noexcept {return device_ == detail::Device::CPU;}
		void reset() noexcept{
			reset_();
		}
		
		template<class Y>
		inline void reset(Y* ptr){
			if(ptr_ != reinterpret_cast<T*>(ptr)){
				reset_();
				ptr_ = reinterpret_cast<T*>(ptr);
				if(ptr_ != NullType::singleton()){
					target_ = new TTarget();
					target_->refcount_.store(1, std::memory_order_relaxed);
				}
			}
		}
		inline int64_t use_count() const noexcept{
			if (target_ == TNullType::singleton())
				return 0;
			return target_->refcount_.load(std::memory_order_acquire);
		}
		
		inline bool defined() const noexcept {return !both_null();}

		inline bool is_unique() const noexcept{
			return use_count() == 1;
		}
		
		inline void swap(intrusive_ptr& r) noexcept{
			std::swap(r.target_, target_);
			std::swap(r.ptr_, ptr_);
			std::swap(r.device_, device_);
			std::swap(r.deallocate_, deallocate_);
		}
		operator bool() const noexcept{
			return !both_null();
		}
		
		template<class... Args>
		static intrusive_ptr make(Args&&... args){
			return intrusive_ptr(new T(std::forward<Args>(args)...));
		}

		static intrusive_ptr unsafe_make_from_raw_new(T* ptr){
			return intrusive_ptr(ptr);	
		}

		static intrusive_ptr make_aligned(const std::size_t amt, const std::size_t align_byte = ALIGN_BYTE_SIZE){
			utils::throw_exception((amt * sizeof(T)) % align_byte == 0, "Cannot align $ bytes", amt * sizeof(T));
			return intrusive_ptr(static_cast<T*>(std::aligned_alloc(align_byte, amt * sizeof(T))));
		}

		static intrusive_ptr make_shared(const std::size_t amt, key_t key = IPC_PRIVATE){
			const uint32_t n_size = amt * sizeof(T);
			utils::throw_exception(utils::get_shared_memory_max() >= n_size, "Expected to allocate at most $ bytes of shared memory, but was asked to allocate $ bytes of shared memory", utils::get_shared_memory_max(), n_size);
			int shmid = shmget(key, n_size, IPC_CREAT | 0666);
			utils::throw_exception(shmid != -1, "Making segment ID failed for shared memory (shmget)");
			void* sharedArray = shmat(shmid, nullptr, 0);
			utils::throw_exception(sharedArray != (void*)-1, "Making shared memory failed (shmat)");
			return intrusive_ptr((T*)sharedArray,
					new TTarget(),
					[shmid](T* ptr){
						shmdt(ptr);
						shmctl(shmid, IPC_RMID, nullptr);
					},
					detail::Device::SharedCPU);
		}

		static intrusive_ptr to_cpu(const intrusive_ptr& ptr, const std::size_t amt){
			if(ptr.is_cpu())
				return ptr;
			intrusive_ptr outp(amt);
			T* optr = outp.get();
			T* iptr = ptr.get();
			T* iptr_end = iptr + amt;
			for(;iptr != iptr_end; ++iptr, ++optr)
				*optr = *iptr;
			return outp;
		}

		static intrusive_ptr to_shared(const intrusive_ptr& ptr, const std::size_t amt, key_t key = IPC_PRIVATE){
			if(ptr.is_shared())
				return ptr;
			intrusive_ptr outp = intrusive_ptr::make_shared(amt, key);
			T* optr = outp.get();
			T* iptr = ptr.get();
			T* iptr_end = iptr + amt;
			for(;iptr != iptr_end; ++iptr, ++optr)
				*optr = *iptr;
			return outp;
		}
};


//the point of the intrusive_ptr<ptr*> classes are to act as a stride class for intrusive_ptr<ptr>
//therefore they must be initialized by an intrusive_ptr<ptr> class
template<
	class T, 
	class TTarget,
	class NullType,
	class TNullType>
class intrusive_ptr<T*, TTarget, NullType, TNullType> final{
	template<class T2, class TTarget2, class NullType2, class TNullType2>
	friend class intrusive_ptr;

#ifndef _WIN32
	static_assert(
			NullType::singleton() == NullType::singleton(),
			"NullType must have a constexpr singleton method"
		     );
	static_assert(
			TNullType::singleton() == TNullType::singleton(),
			"NullType must have a constexpr singleton method"
		     );
#endif

	static_assert(
			std::is_base_of_v<TTarget,
				std::remove_pointer_t<decltype(TNullType::singleton())>>,
				"TNullType::singleton() must return an target_type* pointer");

	intrusive_ptr<std::remove_pointer_t<T>> original;
	TTarget* target_;
	T** ptr_;

	//as a way to protect from , these are the onl 
	inline bool is_null(){
		return ptr_ == NullType::singleton() || target_ == TNullType::singleton();
	}
	inline bool both_null(){
		return ptr_ == NullType::singleton() && target_ == TNullType::singleton();
	}
	inline bool one_null(){
		return !both_null() && is_null();
	}
	inline void null_self() noexcept{
		target_ = TNullType::singleton();
		ptr_ = NullType::singleton();
	}
	inline void check_make_null(){
		if(is_null()){
			if(both_null())
				return;
			else if(ptr_ == NullType::singleton())
				delete target_;
			else if(target_ == TNullType::singleton())
				std::free(ptr_);
			null_self();
		}
	}
	void reset_(){
		original.reset_();
		if(target_ != TNullType::singleton()
				&& detail::atomic_refcount_decrement(target_->refcount_) == 0){
			if(ptr_ != NullType::singleton())
				std::free(ptr_);
			delete target_;
			return;
		}
		null_self();
	}
	inline void retain_(){
		if(target_ != TNullType::singleton()){
			uint32_t new_count = detail::atomic_refcount_increment(target_->refcount_);
			utils::throw_exception(new_count != 1, "intrusive_ptr: cannot increase refcount after it reaches zero");
		}
	}

	
	explicit intrusive_ptr(T** ptr, TTarget* target, intrusive_ptr<std::remove_pointer_t<T>> orig)
		: intrusive_ptr(ptr, target, std::move(orig), detail::DontIncreaseRefCount{}) 
	{
		check_make_null();
		if(!is_null()){
			target_->refcount_.store(1, std::memory_order_relaxed);
		}
	}



	public:
		using element_type = T;
		using target_type = TTarget;
		intrusive_ptr() noexcept
			:intrusive_ptr(NullType::singleton(), 
					TNullType::singleton(), 
					intrusive_ptr<std::remove_pointer_t<T>>(), 
					detail::DontIncreaseRefCount{})
			{}

		intrusive_ptr(std::nullptr_t) noexcept
			:intrusive_ptr(NullType::singleton(), 
					TNullType::singleton(), 
					intrusive_ptr<std::remove_pointer_t<T>>(), 
					detail::DontIncreaseRefCount{})
			{}
		
		

		explicit intrusive_ptr(T** ptr, TTarget* target, intrusive_ptr<std::remove_pointer_t<T>> orig, detail::DontIncreaseRefCount) noexcept
			:original(std::move(orig)), target_(target), ptr_(ptr)
		{check_make_null();}
		
		explicit intrusive_ptr(uint32_t amt, detail::Device dev = detail::Device::CPU, key_t key = IPC_PRIVATE)
			:original(
				dev == detail::Device::CPU ? intrusive_ptr<std::remove_pointer_t<T>>(amt)
							: intrusive_ptr<std::remove_pointer_t<T>>::make_shared(amt, key)),
			target_(new TTarget()),
			ptr_((T**)std::malloc(amt * sizeof(T*)))
		{
			check_make_null();
			utils::throw_exception(!is_null(), "Error making target or ptr for strides");
			target_->refcount_.store(1, std::memory_order_relaxed);
			T** mptr_ = ptr_;
			T* optr_ = original.get();
			T* optr_end = optr_ + amt;
			for(;optr_ != optr_end; ++optr_, ++mptr_)
				*mptr_ = optr_;
		}

		inline ~intrusive_ptr() noexcept { reset_();}
		
		inline intrusive_ptr(intrusive_ptr&& rhs) noexcept
			:original(std::move(rhs.original)), target_(rhs.target_), ptr_(rhs.ptr_)
			{rhs.null_self();}


		template<typename From, typename FromTarget, typename FromNull, typename FromNullT,
			std::enable_if_t<detail::is_pointer<From>::value, bool> = true>
		intrusive_ptr(intrusive_ptr<From, FromTarget, FromNull, FromNullT>&& rhs) noexcept
			:original(std::move(rhs.original)),
			target_(detail::assign_ptr_<TTarget, TNullType, FromNullT>(rhs.target_)),
			ptr_(reinterpret_cast<T**>(rhs.ptr_))
		{
			static_assert(
				std::is_convertible<FromTarget*, TTarget*>::value,
				"Move warning, intrusive_ptr got pointer of wrong type");
			rhs.null_self();
		}

		template<typename From, typename FromTarget, typename FromNull, typename FromNullT,
			std::enable_if_t<!detail::is_pointer<From>::value && !std::is_same_v<From, void>, bool> = true>
		intrusive_ptr(intrusive_ptr<From, FromTarget, FromNull, FromNullT>&& rhs, const std::size_t amt) noexcept
			:original(std::move(rhs)),
			target_(new TTarget()),
			ptr_((T**)std::malloc(amt * sizeof(T**)))
		{
			static_assert(sizeof(From) == sizeof(std::remove_pointer_t<T>),
				"Move warning, when making stride for intrusive_ptr<From> to intrusive_ptr<T*> the size of T and From must be the same");
			static_assert(
				std::is_convertible<FromTarget*, TTarget*>::value,
				"Move warning, intrusive_ptr got pointer of wrong type");
			check_make_null();
			utils::throw_exception(!is_null(), "Error making target or ptr for strides");
			target_->refcount_.store(1, std::memory_order_relaxed);
			T** mptr_ = ptr_;
			T* optr_ = original.get();
			T* optr_end = optr_ + amt;
			for(;optr_ != optr_end; ++optr_, ++mptr_)
				*mptr_ = optr_;
		}



		inline intrusive_ptr(const intrusive_ptr& other) 
			: original(other.original), target_(other.target_), ptr_(other.ptr_)
		{
			retain_();
		}

		template<typename From, typename FromTarget, typename FromNullType, typename FromNullT,
			std::enable_if_t<detail::is_pointer<From>::value, bool> = true>
		intrusive_ptr(const intrusive_ptr<From, FromTarget, FromNullType, FromNullT>& rhs) noexcept
			:original(rhs.original),
			target_(detail::assign_ptr_<TTarget, TNullType, FromNullT>(rhs.target_)),
			ptr_(reinterpret_cast<T**>(rhs.ptr_))
		{
			static_assert(
				std::is_convertible<From*, TTarget*>::value,
				"Move warning, intrusive_ptr got pointer of wrong type");
			retain_();

		}
		
		template<typename From, typename FromTarget, typename FromNullType, typename FromNullT,
			std::enable_if_t<!detail::is_pointer<From>::value && !std::is_same_v<From, void>, bool> = true>
		intrusive_ptr(const intrusive_ptr<From, FromTarget, FromNullType, FromNullT>& rhs, const std::size_t amt) noexcept
			:original(rhs),
			target_(new TTarget()),
			ptr_((T**)std::malloc(amt * sizeof(T*)))
		{
			static_assert(sizeof(From) == sizeof(std::remove_pointer_t<T>),
				"Move warning, when making stride for intrusive_ptr<From> to intrusive_ptr<T*> the size of T and From must be the same");
			static_assert(
				std::is_convertible<From*, TTarget*>::value,
				"Move warning, intrusive_ptr got pointer of wrong type");
			check_make_null();
			utils::throw_exception(!is_null(), "Error making target or ptr for strides");
			target_->refcount_.store(1, std::memory_order_relaxed);
			T** mptr_ = ptr_;
			T* optr_ = original.get();
			T* optr_end = optr_ + amt;
			for(;optr_ != optr_end; ++optr_, ++mptr_)
				*mptr_ = optr_;

		}


		intrusive_ptr& operator=(intrusive_ptr&& rhs) & noexcept {
			original = std::move(rhs.original);
			target_ = rhs.target_;
			ptr_ = rhs.ptr_;
			rhs.null_self();
			/* return operator= <TTarget, NullType>(std::move(rhs)); */
			return *this;
		}
	

		template<typename From, typename FromTarget, typename FromNullType, typename FromNullT,
				std::enable_if_t<detail::is_pointer<From>::value, bool> = true>
		intrusive_ptr& operator=(intrusive_ptr<From, FromTarget, FromNullType, FromNullT>&& rhs) & noexcept{
			static_assert(
				std::is_convertible<FromTarget*, TTarget*>::value,
				"Move warning, intrusive_ptr got pointer of wrong type");

			intrusive_ptr tmp(std::move(rhs));
			swap(tmp);
			return *this;
		}
		


		intrusive_ptr& operator=(const intrusive_ptr& rhs) & noexcept{
			if(this !=
rhs){
				reset_();
				target_ = rhs.target_;
				ptr_ = rhs.ptr_;
				retain_();
			}
			return *this;
		}
		
		template<typename From, typename FromTarget, typename FromNullType, typename FromNullT,
			std::enable_if_t<detail::is_pointer<From>::value, bool> = true>
		intrusive_ptr& operator=(const intrusive_ptr<From, FromTarget, FromNullType, FromNullT>& rhs) & noexcept{
			static_assert(
				std::is_convertible<FromTarget*, TTarget*>::value,
				"Copy warning, intrusive_ptr got pointer of wrong type");
			intrusive_ptr tmp(rhs);
			swap(tmp);
			return *this;
		}

		inline T** operator->() const noexcept{
			return ptr_;
		}
		inline T* operator*() const noexcept{
			return *ptr_;
		}
		inline T** get() const noexcept{
			return ptr_;
		}
		inline T* operator[](std::ptrdiff_t i) const noexcept{
			return ptr_[i];
		}
		inline intrusive_ptr<std::remove_pointer_t<T>>& vals_() const noexcept{
			return original;	
		}
		void reset() noexcept{
			reset_();
		}
		

		inline int64_t use_count() const noexcept{
			if (target_ == TNullType::singleton())
				return 0;
			return target_->refcount_.load(std::memory_order_acquire);
		}
		
		inline bool defined() const noexcept {return !both_null();}

		inline bool is_unique() const noexcept{
			return use_count() == 1;
		}
		
		inline void swap(intrusive_ptr& r) noexcept{
			r.swap(original);
			std::swap(r.target_, target_);
			std::swap(r.ptr_, ptr_);
		}
		operator bool() const noexcept{
			return !both_null();
		}
		inline const detail::Device& device() const noexcept{ return original.device_;}
		inline const bool is_shared() const noexcept {return original.device_ == detail::Device::SharedCPU;}
		inline const bool is_cpu() const noexcept {return original.device == detail::Device::CPU;}
		inline void manual_usecount(const int64_t c){target_->refcount_.store(1, std::memory_order_relaxed);}


		static intrusive_ptr make_shared(const std::size_t amt, key_t key = IPC_PRIVATE){
			return intrusive_ptr(amt, detail::Device::SharedCPU, key);
		}

		static intrusive_ptr to_cpu(const intrusive_ptr& ptr, const std::size_t amt){
			if(ptr.is_cpu())
				return ptr;
			return intrusive_ptr(intrusive_ptr<std::remove_pointer_t<T>>::to_cpu(ptr.original, amt), amt);
		}

		static intrusive_ptr to_shared(const intrusive_ptr& ptr, const std::size_t amt, key_t key = IPC_PRIVATE){
			if(ptr.is_shared())
				return ptr;
			return intrusive_ptr(intrusive_ptr<std::remove_pointer_t<T>>::to_shared(ptr.original, amt, key), amt);
		}
	
};



//that way both the pointer and refcount can be defined
template<
	class TTarget,
	class NullType,
	class TNullType>
class intrusive_ptr<void, TTarget, NullType, TNullType> final{
	template<class T2, class TTarget2, class NullType2, class TNullType2>
	friend class intrusive_ptr;
	using T = void;
	/* using NullType = detail::intrusive_target_default_null_type<void>; */ 
	
#ifndef _WIN32
	static_assert(
			NullType::singleton() == NullType::singleton(),
			"NullType must have a constexpr singleton method"
		     );
	static_assert(
			TNullType::singleton() == TNullType::singleton(),
			"NullType must have a constexpr singleton method"
		     );
#endif

	static_assert(
			std::is_base_of_v<TTarget,
				std::remove_pointer_t<decltype(TNullType::singleton())>>,
				"TNullType::singleton() must return an target_type* pointer");

	TTarget* target_;
	T* ptr_;
	std::function<void(void*)> deallocate_;
	detail::Device device_;
	//as a way to protect from accidental deallocations or bad null checks, these are the only functions that can reall interact with the pointer past a copy
	inline bool is_null(){
		return ptr_ == NullType::singleton() || target_ == TNullType::singleton();
	}
	inline bool both_null(){
		return ptr_ == NullType::singleton() && target_ == TNullType::singleton();
	}
	inline bool one_null(){
		return !both_null() && is_null();
	}
	inline void null_self() noexcept{
		target_ = TNullType::singleton();
		ptr_ = NullType::singleton();
	}
	inline void check_make_null(){
		if(is_null()){
			if(both_null())
				return;
			else if(ptr_ == NullType::singleton())
				delete target_;
			else if(target_ == TNullType::singleton())
				deallocate_(ptr_);
			null_self();
		}
	}
	void reset_(){
		if(target_ != TNullType::singleton()
				&& detail::atomic_refcount_decrement(target_->refcount_) <= 0){
			if(ptr_ != NullType::singleton()){
				deallocate_(ptr_);
			}
			delete target_;
			return;
		}
		null_self();
	}
	inline void retain_(){
		if(target_ != TNullType::singleton()){
			int64_t new_count = detail::atomic_refcount_increment(target_->refcount_);
			utils::throw_exception(new_count != 1, "intrusive_ptr: cannot increase refcount after it reaches zero got refcount of void*");
		}
	}

	
	explicit intrusive_ptr(T* ptr, TTarget* target, std::function<void(void*)> dealc, detail::Device dev = detail::Device::CPU)
		: intrusive_ptr(ptr, target, dealc, dev, detail::DontIncreaseRefCount{}) 
	{
		check_make_null();
		if(!is_null()){
			target_->refcount_.store(1, std::memory_order_relaxed);
		}
	}
	
	template<typename From>
	explicit intrusive_ptr(From* ptr)
		:target_(ptr == detail::intrusive_target_default_null_type<From>::singleton() ? TNullType::singleton() : new TTarget()), 
		ptr_(reinterpret_cast<void*>(ptr)),
		deallocate_([](void* p){From* pt = reinterpret_cast<From*>(p);std::free(pt);}),
		device_(detail::Device::CPU)
		{
			check_make_null();
			if(!is_null()){
				target_->refcount_.store(1, std::memory_order_relaxed);
			}
		}

	public:
		using element_type = T;
		using target_type = TTarget;
		intrusive_ptr() noexcept
			:target_(TNullType::singleton()),
			ptr_(NullType::singleton()),
			deallocate_([](void* p){;}),
			device_(detail::Device::CPU)
			{}

		intrusive_ptr(std::nullptr_t) noexcept
			:target_(TNullType::singleton()),
			ptr_(NullType::singleton()),
			deallocate_([](void* p){;}),
			device_(detail::Device::CPU)
			{}
		
		explicit intrusive_ptr(T* ptr, TTarget* target, std::function<void(void*)> dealc, detail::Device dev, detail::DontIncreaseRefCount) noexcept
			:target_(target),
			ptr_(ptr),
			deallocate_(dealc),
			device_(dev)
		{check_make_null();}

		explicit intrusive_ptr(std::size_t amt, std::size_t byte_size)
			:target_(new TTarget()),
			ptr_(std::malloc(amt * byte_size)),
			deallocate_([](void* pt){std::free(pt);}),
			device_(detail::Device::CPU)
		{
			check_make_null();
			utils::throw_exception(ptr_ != NullType::singleton(), "Failure to allocate ptr_, void*");
			if(!is_null()){
				target_->refcount_.store(1, std::memory_order_relaxed);
			}

		}


		template<class From>
		explicit intrusive_ptr(std::unique_ptr<From> rhs) noexcept
			:intrusive_ptr(rhs.release())
		{}

		inline ~intrusive_ptr() noexcept { reset_();}
		
		inline intrusive_ptr(intrusive_ptr&& rhs) noexcept
			:target_(rhs.target_), ptr_(rhs.ptr_), deallocate_(rhs.deallocate_), device_(rhs.device_)
			{rhs.null_self();}


		template<typename From, typename FromTarget, typename FromNull, typename FromNullT>
		intrusive_ptr(intrusive_ptr<From, FromTarget, FromNull, FromNullT>&& rhs) noexcept
			:target_(detail::assign_ptr_<TTarget, TNullType, FromNullT>(rhs.target_)),
			ptr_(reinterpret_cast<void*>(rhs.ptr_)),
			deallocate_(detail::change_function_input<void*, From>(rhs.deallocate_)),
			device_(rhs.device_)
		{
			static_assert(
				std::is_convertible<FromTarget*, TTarget*>::value,
				"Move warning, intrusive_ptr got pointer of wrong type");
			rhs.null_self();
		}



		inline intrusive_ptr(const intrusive_ptr& other) 
			: target_(other.target_), 
			ptr_(other.ptr_),
			deallocate_(other.deallocate_),
			device_(other.device_)
		{
			retain_();
		}

		template<typename From, typename FromTarget, typename FromNullType, typename FromNullT>
		intrusive_ptr(const intrusive_ptr<From, FromTarget, FromNullType, FromNullT>& rhs) noexcept
			:target_(detail::assign_ptr_<TTarget, TNullType, FromNullT>(rhs.target_)),
			ptr_(reinterpret_cast<void*>(rhs.ptr_)),
			deallocate_(detail::change_function_input<void, From>(rhs.deallocate_)),
			device_(rhs.device_)


		{
			static_assert(
				std::is_convertible<FromTarget*, TTarget*>::value,
				"Move warning, intrusive_ptr got pointer of wrong type");
			retain_();
		}

		intrusive_ptr& operator=(intrusive_ptr&& rhs) & noexcept {
			reset_();
			target_ = rhs.target_;
			ptr_ = rhs.ptr_;
			deallocate_ = rhs.deallocate_;
			device_ = rhs.device_;
			rhs.null_self();
			/* return operator= <TTarget, NullType>(std::move(rhs)); */
			return *this;
		}

		template<typename From, typename FromTarget, typename FromNullType, typename FromNullT>
		intrusive_ptr& operator=(intrusive_ptr<From, FromTarget, FromNullType, FromNullT>&& rhs) & noexcept{

			static_assert(
				std::is_convertible<FromTarget*, TTarget*>::value,
				"Move warning, intrusive_ptr got pointer of wrong type");

			reset_();
			target_ = detail::assign_ptr_<TTarget, TNullType, FromNullT>(rhs.target_);
			ptr_ = reinterpret_cast<void*>(rhs.ptr_);
			deallocate_ = detail::change_function_input<void, From>(rhs.deallocate_);
			device_ = rhs.device_;
			rhs.null_self();
			return *this;
		}
		


		intrusive_ptr& operator=(const intrusive_ptr& rhs) & noexcept{
			if(this != &rhs){
				reset_();
				deallocate_ = rhs.deallocate_;
				target_ = rhs.target_;
				ptr_ = rhs.ptr_;
				device_ = rhs.device_;
				retain_();
			}
			return *this;
		}
		
		template<typename From, typename FromTarget, typename FromNullType, typename FromNullT>
		intrusive_ptr& operator=(const intrusive_ptr<From, FromTarget, FromNullType, FromNullT>& rhs) & noexcept{
			static_assert(
				std::is_convertible<From*, TTarget*>::value,
				"Copy warning, intrusive_ptr got pointer of wrong type");
			reset_();
			target_ = detail::assign_ptr_<TTarget, TNullType, FromNullT>(rhs.target_);
			ptr_ = reinterpret_cast<void*>(rhs.ptr_);
			deallocate_ = detail::change_function_input<void, From>(rhs.deallocate_);
			device_ = rhs.device_;
			retain_();
			return *this;
		}

		inline T* operator->() const noexcept{
			return ptr_;
		}

		inline T* get() const noexcept{
			return ptr_;
		}
		void reset() noexcept{
			reset_();
		}
		
		//only viable on cpu versions, will probably be depreciated in future versions
		template<class Y>
		inline void reset(Y* ptr){
			if(ptr_ != reinterpret_cast<T*>(ptr)){
				reset_();
				ptr_ = reinterpret_cast<T*>(ptr);
				if(ptr_ != NullType::singleton()){
					target_ = new TTarget();
					target_->refcount_.store(1, std::memory_order_relaxed);
					deallocate_ = ([](void* ptr){Y* p = reinterpret_cast<Y*>(ptr); delete p;});
					device_ = detail::Device::CPU;
				}
			}
		}
		inline int64_t use_count() const noexcept{
			if (target_ == TNullType::singleton())
				return 0;
			return target_->refcount_.load(std::memory_order_acquire);
		}
		inline void manual_usecount(const int64_t c){target_->refcount_.store(1, std::memory_order_relaxed);}
		
		inline bool defined() const noexcept {return !both_null();}

		inline bool is_unique() const noexcept{
			return use_count() == 1;
		}
		
		inline void swap(intrusive_ptr& r) noexcept{
			std::swap(r.target_, target_);
			std::swap(r.ptr_, ptr_);
			std::swap(r.deallocate_, deallocate_);
			std::swap(r.device_, device_);
		}
		operator bool() const noexcept{
			return !both_null();
		}
		
		static intrusive_ptr make_intrusive(void* ptr, std::function<void(void*)> dealc){
			return intrusive_ptr(ptr, new TTarget(), dealc);	
		}
		inline const detail::Device& device() const noexcept{ return device_;}
		inline const bool is_shared() const noexcept {return device_ == detail::Device::SharedCPU;}
		inline const bool is_cpu() const noexcept {return device_ == detail::Device::CPU;}
		
		static intrusive_ptr make_aligned(const std::size_t amt, std::size_t type_size, const std::size_t align_byte = ALIGN_BYTE_SIZE){
			utils::throw_exception((amt * type_size) % align_byte == 0, "Cannot align $ bytes", amt * type_size);
			return intrusive_ptr(static_cast<T*>(std::aligned_alloc(align_byte, amt * type_size)));
		}

		static intrusive_ptr make_shared(const std::size_t amt, const std::size_t byte_size, key_t key = IPC_PRIVATE){
			const uint32_t n_size = amt * byte_size;
			utils::throw_exception(utils::get_shared_memory_max() >= n_size, "Expected to allocate at most $ bytes of shared memory, but was asked to allocate $ bytes of shared memory void*", utils::get_shared_memory_max(), n_size);
			int shmid = shmget(key, n_size, IPC_CREAT | 0666);
			utils::throw_exception(shmid != -1, "Making segment ID failed for shared memory (shmget)");
			void* sharedArray = shmat(shmid, nullptr, 0);
			utils::throw_exception(sharedArray != (void*)-1, "Making shared memory failed (shmat)");
			return intrusive_ptr(sharedArray,
					new TTarget(),
					[shmid](void* ptr){
						shmdt(ptr);
						shmctl(shmid, IPC_RMID, nullptr);
					},
					detail::Device::SharedCPU);
		}

		static intrusive_ptr to_cpu(const intrusive_ptr& ptr, const std::size_t amt, const std::size_t byte_size){
			if(ptr.is_cpu())
				return ptr;
			intrusive_ptr outp(amt, byte_size);
			const std::size_t n_size = amt * byte_size;
			const std::size_t end = n_size / sizeof(uint64_t);
			const std::size_t r = (n_size % sizeof(uint64_t)) / sizeof(uint8_t);
			uint64_t* optr = reinterpret_cast<uint64_t*>(outp.get());
			uint64_t* iptr = reinterpret_cast<uint64_t*>(ptr.get());
			uint64_t* iptr_end = iptr + end;
			for(;iptr != iptr_end; ++iptr, ++optr)
				*optr = *iptr;
			if(r != 0){
				uint8_t* iptr_a = reinterpret_cast<uint8_t*>(iptr_end);
				uint8_t* iptra_end = iptr_a + r;
				uint8_t* optr_a = reinterpret_cast<uint8_t*>(optr);
				for(;iptr != iptr_end; ++iptr, ++optr)
					*optr = *iptr;
			}
			return outp;
		}

		static intrusive_ptr to_shared(const intrusive_ptr& ptr, const std::size_t amt, const std::size_t byte_size, key_t key = IPC_PRIVATE){
			if(ptr.is_shared())
				return ptr;
			intrusive_ptr outp = intrusive_ptr::make_shared(amt, byte_size, key);
			const std::size_t n_size = amt * byte_size;
			const std::size_t end = n_size / sizeof(uint64_t);
			const std::size_t r = (n_size % sizeof(uint64_t)) / sizeof(uint8_t);
			uint64_t* optr = reinterpret_cast<uint64_t*>(outp.get());
			uint64_t* iptr = reinterpret_cast<uint64_t*>(ptr.get());
			uint64_t* iptr_end = iptr + end;
			for(;iptr != iptr_end; ++iptr, ++optr)
				*optr = *iptr;
			if(r != 0){
				uint8_t* iptr_a = reinterpret_cast<uint8_t*>(iptr_end);
				uint8_t* iptra_end = iptr_a + r;
				uint8_t* optr_a = reinterpret_cast<uint8_t*>(optr);
				for(;iptr != iptr_end; ++iptr, ++optr)
					*optr = *iptr;
			}
			return outp;
		}
};

//the point of the intrusive_ptr<ptr*> classes are to act as a stride class for intrusive_ptr<ptr>
//therefore they must be initialized by an intrusive_ptr<ptr> class
template< 
	class TTarget,
	class NullType,
	class TNullType>
class intrusive_ptr<void*, TTarget, NullType, TNullType> final{
	template<class T2, class TTarget2, class NullType2, class TNullType2>
	friend class intrusive_ptr;
	using T = void*;

#ifndef _WIN32
	static_assert(
			NullType::singleton() == NullType::singleton(),
			"NullType must have a constexpr singleton method"
		     );
	static_assert(
			TNullType::singleton() == TNullType::singleton(),
			"NullType must have a constexpr singleton method"
		     );
#endif

	static_assert(
			std::is_base_of_v<TTarget,
				std::remove_pointer_t<decltype(TNullType::singleton())>>,
				"TNullType::singleton() must return an target_type* pointer");

	intrusive_ptr<std::remove_pointer_t<T>> original;
	TTarget* target_;
	T* ptr_;

	//as a way to protect from , these are the onl 
	inline bool is_null(){
		return ptr_ == NullType::singleton() || target_ == TNullType::singleton();
	}
	inline bool both_null(){
		return ptr_ == NullType::singleton() && target_ == TNullType::singleton();
	}
	inline bool one_null(){
		return !both_null() && is_null();
	}
	inline void null_self() noexcept{
		target_ = TNullType::singleton();
		ptr_ = NullType::singleton();
	}
	inline void check_make_null(){
		if(is_null()){
			if(both_null())
				return;
			else if(ptr_ == NullType::singleton())
				delete target_;
			else if(target_ == TNullType::singleton())
				std::free(ptr_);
			null_self();
		}
	}
	void reset_(){
		if(target_ != TNullType::singleton()
				&& detail::atomic_refcount_decrement(target_->refcount_) == 0){
			if(ptr_ != NullType::singleton())
				std::free(ptr_);
			delete target_;
			return;
		}
		null_self();
	}
	inline void retain_(){
		if(target_ != TNullType::singleton()){
			int64_t new_count = detail::atomic_refcount_increment(target_->refcount_);
			utils::throw_exception(new_count != 1, "intrusive_ptr: cannot increase refcount after it reaches zero void**");
		}
	}

	
	explicit intrusive_ptr(T* ptr, TTarget* target, intrusive_ptr<std::remove_pointer_t<T>> orig)
		: intrusive_ptr(ptr, target, std::move(orig), detail::DontIncreaseRefCount{}) 
	{
		check_make_null();
		if(!is_null()){
			target_->refcount_.store(1, std::memory_order_relaxed);
		}
	}



	public:
		using element_type = T*;
		using target_type = TTarget;
		intrusive_ptr() noexcept
			:intrusive_ptr(NullType::singleton(), TNullType::singleton(), intrusive_ptr<std::remove_pointer_t<T>>(), detail::DontIncreaseRefCount{})
			{}

		intrusive_ptr(std::nullptr_t) noexcept
			:intrusive_ptr(NullType::singleton(), TNullType::singleton(), intrusive_ptr<std::remove_pointer_t<T>>(), detail::DontIncreaseRefCount{})
			{}
		
		

		explicit intrusive_ptr(T* ptr, TTarget* target, intrusive_ptr<std::remove_pointer_t<T>> orig, detail::DontIncreaseRefCount) noexcept
			:original(std::move(orig)), target_(target), ptr_(ptr)
		{check_make_null();}
		
		explicit intrusive_ptr(const uint32_t amt, const uint32_t byte_size, const uint32_t ptr_byte_size, detail::Device dev = detail::Device::CPU, key_t key = IPC_PRIVATE)
			:original(dev == detail::Device::CPU ? intrusive_ptr<void>(amt, byte_size)
							: intrusive_ptr<void>::make_shared(amt, byte_size, key)),
			target_(new TTarget()),
			ptr_((void**)std::malloc(amt * ptr_byte_size))
		{
			check_make_null();
			utils::throw_exception(!is_null(), "Error making target or ptr for strides void**");
			target_->refcount_.store(1, std::memory_order_relaxed);
			uint8_t** mptr_ = reinterpret_cast<uint8_t**>(ptr_);
			uint8_t* optr_ = reinterpret_cast<uint8_t*>(original.get());
			const std::size_t mptr_add = ptr_byte_size / sizeof(uint8_t*);
			for(uint32_t i = 0; i < amt; ++i, mptr_ += mptr_add, optr_ += byte_size){
				*(mptr_) = optr_;
			}
		}
		
		

		inline ~intrusive_ptr() noexcept { reset_();}
		
		inline intrusive_ptr(intrusive_ptr&& rhs) noexcept
			:original(std::move(rhs.original)), target_(rhs.target_), ptr_(rhs.ptr_)
			{rhs.null_self();}
		
		inline intrusive_ptr<void*> copy_strides(const uint32_t amt, const uint32_t ptr_byte_size) const{
			return intrusive_ptr((void**)std::malloc(amt * ptr_byte_size), new TTarget(), original);
		}

		template<typename From, typename FromTarget, typename FromNull, typename FromNullT,
			std::enable_if_t<detail::is_pointer<From>::value, bool> = true>
		intrusive_ptr(intrusive_ptr<From, FromTarget, FromNull, FromNullT>&& rhs) noexcept
			:original(std::move(rhs.original)),
			target_(detail::assign_ptr_<TTarget, TNullType, FromNullT>(rhs.target_)),
			ptr_(reinterpret_cast<T*>(rhs.ptr_))
		{
			static_assert(
				std::is_convertible<FromTarget*, TTarget*>::value,
				"Move warning, intrusive_ptr got pointer of wrong type");
			rhs.null_self();
		}

		template<typename From, typename FromTarget, typename FromNull, typename FromNullT,
			std::enable_if_t<!detail::is_pointer<From>::value && !std::is_same_v<From, void>, bool> = true>
		intrusive_ptr(intrusive_ptr<From, FromTarget, FromNull, FromNullT>&& rhs, const std::size_t amt) noexcept
			:original(std::move(rhs)),
			target_(new TTarget()),
			ptr_((void**)std::malloc(amt * sizeof(From*)))
		{
			static_assert(
				std::is_convertible<FromTarget*, TTarget*>::value,
				"Move warning, intrusive_ptr got pointer of wrong type");
			check_make_null();
			utils::throw_exception(!is_null(), "Error making target or ptr for strides");
			target_->refcount_.store(1, std::memory_order_relaxed);
			From** mptr_ = reinterpret_cast<From**>(ptr_);
			From* optr_ = reinterpret_cast<From*>(original.get());
			for(uint32_t i = 0; i < amt; ++i)
				mptr_[i] = &optr_[i];
		}

		template<typename From, typename FromTarget, typename FromNull, typename FromNullT,
			std::enable_if_t<std::is_same_v<From, void>, bool> = true>
		intrusive_ptr(intrusive_ptr<From, FromTarget, FromNull, FromNullT>&& rhs, const std::size_t amt, const std::size_t byte_size, const std::size_t ptr_byte_size) noexcept
			:original(std::move(rhs)),
			target_(new TTarget()),
			ptr_((void**)std::malloc(amt * ptr_byte_size))
		{
			static_assert(
				std::is_convertible<FromTarget*, TTarget*>::value,
				"Move warning, intrusive_ptr got pointer of wrong type");
			check_make_null();
			utils::throw_exception(!is_null(), "Error making target or ptr for strides");
			target_->refcount_.store(1, std::memory_order_relaxed);
			uint8_t** mptr_ = reinterpret_cast<uint8_t**>(ptr_);
			uint8_t* optr_ = reinterpret_cast<uint8_t*>(original.get());
			const std::size_t mptr_add = ptr_byte_size / sizeof(uint8_t*);
			for(uint32_t i = 0; i < amt; ++i, mptr_ += mptr_add, optr_ += byte_size){
				*(mptr_) = optr_;
			}
		}

		template<typename From, typename FromTarget, typename FromNull, typename FromNullT,
			std::enable_if_t<std::is_same_v<From, void>, bool> = true>
		intrusive_ptr(intrusive_ptr<From, FromTarget, FromNull, FromNullT>&& rhs, const std::size_t amt, const std::size_t byte_size, const std::size_t ptr_byte_size, detail::DontOrderStrides) noexcept
			:original(std::move(rhs)),
			target_(new TTarget()),
			ptr_((void**)std::malloc(amt * ptr_byte_size))
		{
			static_assert(
				std::is_convertible<FromTarget*, TTarget*>::value,
				"Move warning, intrusive_ptr got pointer of wrong type");
			check_make_null();
			utils::throw_exception(!is_null(), "Error making target or ptr for strides");
			target_->refcount_.store(1, std::memory_order_relaxed);
		}



		inline intrusive_ptr(const intrusive_ptr& other) 
			: original(other.original), target_(other.target_), ptr_(other.ptr_)
		{
			retain_();
			//std::cout << use_count() << std::endl;
		}

		template<typename From, typename FromTarget, typename FromNullType, typename FromNullT,
			std::enable_if_t<detail::is_pointer<From>::value, bool> = true>
		intrusive_ptr(const intrusive_ptr<From, FromTarget, FromNullType, FromNullT>& rhs) noexcept
			:original(rhs.original),
			target_(detail::assign_ptr_<TTarget, TNullType, FromNullT>(rhs.target_)),
			ptr_(detail::assign_ptr_<T, NullType, FromNullType>(rhs.ptr_))
		{
			static_assert(
				std::is_convertible<FromTarget*, TTarget*>::value,
				"Move warning, intrusive_ptr got pointer of wrong type");
			retain_();

		}
		
		template<typename From, typename FromTarget, typename FromNullType, typename FromNullT,
			std::enable_if_t<!detail::is_pointer<From>::value && !std::is_same_v<From, void>, bool> = true>
		intrusive_ptr(const intrusive_ptr<From, FromTarget, FromNullType, FromNullT>& rhs, const std::size_t amt) noexcept
			:original(rhs),
			target_(new TTarget()),
			ptr_((void**)std::malloc(amt * sizeof(From*)))
		{
			static_assert(
				std::is_convertible<FromTarget*, TTarget*>::value,
				"Move warning, intrusive_ptr got pointer of wrong type");
			check_make_null();
			utils::throw_exception(!is_null(), "Error making target or ptr for strides");
			target_->refcount_.store(1, std::memory_order_relaxed);
			From* optr_ = reinterpret_cast<From*>(original.get());
			From** mptr_ = reinterpret_cast<From**>(ptr_);
			for(uint32_t i = 0; i < amt; ++i, ++optr_, ++mptr_)
				*(mptr_) = optr_;
		}

		template<typename From, typename FromTarget, typename FromNullType, typename FromNullT,
			std::enable_if_t<std::is_same_v<From, void>, bool> = true>
		intrusive_ptr(const intrusive_ptr<From, FromTarget, FromNullType, FromNullT>& rhs, const std::size_t amt, const std::size_t byte_size, const std::size_t ptr_byte_size) noexcept
			:original(rhs),
			target_(new TTarget()),
			ptr_((void**)std::malloc(amt * ptr_byte_size))
		{
			static_assert(
				std::is_convertible<FromTarget*, TTarget*>::value,
				"Move warning, intrusive_ptr got pointer of wrong type");
			check_make_null();
			utils::throw_exception(!is_null(), "Error making target or ptr for strides");
			target_->refcount_.store(1, std::memory_order_relaxed);
			uint8_t** mptr_ = reinterpret_cast<uint8_t**>(ptr_);
			uint8_t* optr_ = reinterpret_cast<uint8_t*>(original.get());
			const std::size_t mptr_add = ptr_byte_size / sizeof(uint8_t*);
			for(uint32_t i = 0; i < amt; ++i, mptr_ += mptr_add, optr_ += byte_size){
				*(mptr_) = optr_;
			};
		}

		template<typename From, typename FromTarget, typename FromNullType, typename FromNullT,
			std::enable_if_t<std::is_same_v<From, void>, bool> = true>
		intrusive_ptr(const intrusive_ptr<From, FromTarget, FromNullType, FromNullT>& rhs, const std::size_t amt, const std::size_t byte_size, const std::size_t ptr_byte_size, detail::DontOrderStrides) noexcept
			:original(rhs),
			target_(new TTarget()),
			ptr_((void**)std::malloc(amt * ptr_byte_size))
		{
			static_assert(
				std::is_convertible<FromTarget*, TTarget*>::value,
				"Move warning, intrusive_ptr got pointer of wrong type");
			check_make_null();
			utils::throw_exception(!is_null(), "Error making target or ptr for strides");
			target_->refcount_.store(1, std::memory_order_relaxed);
		}	

		intrusive_ptr& operator=(intrusive_ptr<void*, TTarget, NullType, TNullType>&& rhs) & noexcept {
			original = std::move(rhs.original);
			target_ = rhs.target_;
			ptr_ = rhs.ptr_;
			rhs.null_self();
			/* return operator= <TTarget, NullType>(std::move(rhs)); */
			return *this;
		}
	

		template<typename From, typename FromTarget, typename FromNullType, typename FromNullT,
				std::enable_if_t<detail::is_pointer<From>::value, bool> = true>
		intrusive_ptr& operator=(intrusive_ptr<From, FromTarget, FromNullType, FromNullT>&& rhs) & noexcept{
			static_assert(
				std::is_convertible<FromTarget*, TTarget*>::value,
				"Move warning, intrusive_ptr got pointer of wrong type");

			intrusive_ptr tmp(std::move(rhs));
			swap(tmp);
			return *this;
		}
		


		intrusive_ptr& operator=(const intrusive_ptr<void*, TTarget, NullType, TNullType>& rhs){
			if(this != &rhs){
				reset_();
				target_ = rhs.target_;
				ptr_ = rhs.ptr_;
				retain_();
			}
			return *this;
		}
		
		template<typename From, typename FromTarget, typename FromNullType, typename FromNullT,
			std::enable_if_t<detail::is_pointer<From>::value, bool> = true>
		intrusive_ptr& operator=(const intrusive_ptr<From, FromTarget, FromNullType, FromNullT>& rhs) & noexcept{
			static_assert(
				std::is_convertible<FromTarget*, TTarget*>::value,
				"Copy warning, intrusive_ptr got pointer of wrong type");
			intrusive_ptr tmp(rhs);
			swap(tmp);
			return *this;
		}

		inline T* operator->() const noexcept{
			return ptr_;
		}
		inline T& operator*() const noexcept{
			return *ptr_;
		}
		inline T* get() const noexcept{
			return ptr_;
		}
		inline T& operator[](const std::ptrdiff_t i) const noexcept{
			return ptr_[i];
		}
		inline const intrusive_ptr<std::remove_pointer_t<T>>& vals_() const noexcept{
			return original;	
		}
		inline intrusive_ptr<std::remove_pointer_t<T>>& vals_() noexcept{
			return original;	
		}
		void reset() noexcept{
			reset_();
		}
		

		inline int64_t use_count() const noexcept{
			if (target_ == TNullType::singleton())
				return 0;
			return target_->refcount_.load(std::memory_order_acquire);
		}
		inline void manual_usecount(const int64_t c){target_->refcount_.store(1, std::memory_order_relaxed);}
		
		inline bool defined() const noexcept {return !both_null();}

		inline bool is_unique() const noexcept{
			return use_count() == 1;
		}
		
		inline void swap(intrusive_ptr& r) noexcept{
			r.swap(original);
			std::swap(r.target_, target_);
			std::swap(r.ptr_, ptr_);
		}
		operator bool() const noexcept{
			return !both_null();
		}
		
		inline const detail::Device& device() const noexcept{ return original.device_;}
		inline const bool is_shared() const noexcept {return original.device_ == detail::Device::SharedCPU;}
		inline const bool is_cpu() const noexcept {return original.device_ == detail::Device::CPU;}


		static intrusive_ptr make_shared(const std::size_t amt, const std::size_t byte_size, const std::size_t ptr_byte_size, key_t key = IPC_PRIVATE){
			return intrusive_ptr(amt, byte_size, ptr_byte_size, detail::Device::SharedCPU, key);
		}

		static intrusive_ptr to_cpu(const intrusive_ptr& ptr, const std::size_t amt, const std::size_t byte_size, const std::size_t ptr_byte_size){
			if(ptr.is_cpu())
				return ptr;
			return intrusive_ptr(intrusive_ptr<void>::to_cpu(ptr.original, amt, byte_size), amt, byte_size, ptr_byte_size);
		}
		static intrusive_ptr to_cpu(const intrusive_ptr& ptr, const std::size_t amt, const std::size_t byte_size, const std::size_t ptr_byte_size, detail::DontOrderStrides){
			if(ptr.is_cpu())
				return ptr;
			 return intrusive_ptr(intrusive_ptr<void>::to_cpu(ptr.original, amt, byte_size), amt, byte_size, ptr_byte_size, detail::DontOrderStrides{});
		}

		static intrusive_ptr to_shared(const intrusive_ptr& ptr, const std::size_t amt, const std::size_t byte_size, const std::size_t ptr_byte_size, key_t key = IPC_PRIVATE){
			if(ptr.is_shared())
				return ptr;
			return intrusive_ptr(intrusive_ptr<void>::to_shared(ptr.original, amt, byte_size, key), amt, byte_size, ptr_byte_size);
		}

		static intrusive_ptr to_shared(const intrusive_ptr& ptr, const std::size_t amt, const std::size_t byte_size, const std::size_t ptr_byte_size, key_t key, detail::DontOrderStrides){
			if(ptr.is_shared())
				return ptr;
			return intrusive_ptr(intrusive_ptr<void>::to_shared(ptr.original, amt, byte_size, key), amt, byte_size, ptr_byte_size, detail::DontOrderStrides{});
		}


		
};


}
#endif
