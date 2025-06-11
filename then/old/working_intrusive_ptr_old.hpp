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



//basically any class that uses an intrusive_ptr, must inherit from intrusive_ptr_target, that way the refcount can be made internally

class intrusive_ptr_target{
	mutable std::atomic<int64_t> refcount_;

	template<typename T, typename TTarget, typename NullType, typename TNullType>
	friend class intrusive_ptr;
	friend inline void detail::intrusive_ptr::incref(intrusive_ptr_target *self);

	
	virtual void release_resources() {}

	protected:
	
	virtual ~intrusive_ptr_target(){
		/* utils::throw_exception(refcount_.load() == 0 || refcount_.load() >= detail::kImpracticallyHugeReferenceCount, */
		/* 		"needed refcount to be too high or at 0 in order to release resources"); */
		/* release_resources(); */
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
	

	inline T* handle_amt_ptr(uint64_t amt){
		if(amt == 0)
			return NullType::singleton();
		return new T[amt];

	}
	//as a way to protect from , these are the onl 
	inline bool is_null() const{
		return ptr_ == NullType::singleton() || target_ == TNullType::singleton();
	}
	inline bool both_null() const{
		return ptr_ == NullType::singleton() && target_ == TNullType::singleton();
	}
	inline bool one_null() const{
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
			else if(target_ == TNullType::singleton()){
				deallocate_(ptr_);
			}
			null_self();
		}
	}
	void reset_(){
		if(target_ != TNullType::singleton()
				&& detail::atomic_refcount_decrement(target_->refcount_) == 0){
			/* std::cout << "deallocating T*"<<std::endl; */
			if(ptr_ != NullType::singleton())
				deallocate_(ptr_);
			/* std::cout << "deallocating target T*"<<std::endl; */
			delete target_;
			return;
		}
		null_self();
	}
	inline void retain_() const {
		if(target_ != TNullType::singleton()){
			uint32_t new_count = detail::atomic_refcount_increment(target_->refcount_);
			utils::throw_exception(new_count != 1, "intrusive_ptr: cannot increase refcount after it reaches zero");
		}
	}

	
	explicit intrusive_ptr(T* ptr, TTarget* target, std::function<void(T*)> dealc, detail::Device dev = detail::Device::CPU)
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
		deallocate_([](T* ptr){if(ptr != NullType::singleton()){delete[] ptr;}}),
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
					[](T* ptr){if(ptr != NullType::singleton()){delete[] ptr;}}, 
					detail::Device::CPU, detail::DontIncreaseRefCount{})
			{}

		intrusive_ptr(std::nullptr_t) noexcept
			:intrusive_ptr(NullType::singleton(), 
					TNullType::singleton(), 
					[](T* ptr){if(ptr != NullType::singleton()){delete[] ptr;}}, 
					detail::Device::CPU, detail::DontIncreaseRefCount{})
			{}
		
		

		explicit intrusive_ptr(T* ptr, 
				TTarget* target, 
				std::function<void(T*)> dealc, 
				detail::Device dev, 
				detail::DontIncreaseRefCount) noexcept
			:target_(target), ptr_(ptr), deallocate_(dealc), device_(dev)
		{check_make_null();}
		
		explicit intrusive_ptr(uint64_t amt)
			:intrusive_ptr(handle_amt_ptr(amt))
		{
			if(amt == 0){null_self();}
			deallocate_ = [amt](T* ptr){if(ptr != NullType::singleton()){/*std::cout << "deleting of size "<<amt<<std::endl;*/delete[] ptr;}};
			utils::throw_exception(ptr_ != NullType::singleton() || amt == 0, "Failure to allocate ptr_, T*");}

		explicit intrusive_ptr(std::unique_ptr<T> rhs) noexcept
			: intrusive_ptr(rhs.release()) {}
		
		template<class From>
		explicit intrusive_ptr(std::unique_ptr<From> rhs) noexcept
			:intrusive_ptr(reinterpret_cast<T*>(rhs.release()))
		{}

		inline ~intrusive_ptr() noexcept { /*std::cout << "going to reset"<<std::endl;*/reset_();}
		
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
		inline intrusive_ptr<T> operator+(const uint64_t i) const noexcept{
			retain_();
			return intrusive_ptr(ptr_ + i,
					target_,
					deallocate_,
					device_,
					detail::DontIncreaseRefCount{});
		}
		inline const detail::Device& device() const noexcept{ return device_;}
		inline const bool is_shared() const noexcept {return device_ == detail::Device::SharedCPU;}
		inline const bool is_cpu() const noexcept {return device_ == detail::Device::CPU;}
		inline void nullify() noexcept{
			null_self();
		}
		inline T* release() noexcept{
			T* returning = ptr_;
			null_self();
			return returning;
		}
		inline void reset() noexcept{
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
			return intrusive_ptr(new T(std::forward<Args&&>(args)...));
		}

		static intrusive_ptr unsafe_make_from_raw_new(T* ptr){
			return intrusive_ptr(ptr);	
		}

		static intrusive_ptr make_aligned(const uint64_t amt, const std::size_t align_byte = ALIGN_BYTE_SIZE){
			uint64_t size = amt * sizeof(T);
			if (size % align_byte != 0) size += align_byte - (size % align_byte);
			/* utils::throw_exception((amt * sizeof(T)) % align_byte == 0, "Cannot align $ bytes", amt * sizeof(T)); */
			return intrusive_ptr(static_cast<T*>(std::aligned_alloc(align_byte, size)), 
					new TTarget(), 
					([](T* ptr){if(!ptr){return;}std::free(ptr);}));
		}

#ifdef USE_PARALLEL
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
#endif

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
#ifdef USE_PARALLEL
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
#endif
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
	
	inline bool both_null() const {
		return ptr_ == NullType::singleton() && target_ == TNullType::singleton();
	}
	inline bool one_null() const {
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
		/* std::cout << "reset_() called"<<std::endl; */
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
	inline void retain_() const{
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
		deallocate_([](void* p){From* pt = reinterpret_cast<From*>(p);if(pt != nullptr){std::free(pt);}}),
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
		
		template<typename From>
		explicit intrusive_ptr(From* ptr, std::function<void(void*)> dealc) noexcept
			:target_(ptr == detail::intrusive_target_default_null_type<From>::singleton() ? TNullType::singleton() : new TTarget()),
			ptr_(reinterpret_cast<void*>(ptr)),
			deallocate_(dealc),
			device_(detail::Device::CPU)
		{
			check_make_null();
			if(!is_null()){
				target_->refcount_.store(1, std::memory_order_relaxed);
			}
		}

		explicit intrusive_ptr(T* ptr, TTarget* target, std::function<void(void*)> dealc, detail::Device dev, detail::DontIncreaseRefCount) noexcept
			:target_(target),
			ptr_(ptr),
			deallocate_(dealc),
			device_(dev)
		{check_make_null();}

		explicit intrusive_ptr(int64_t amt, std::size_t byte_size)
			:target_(new TTarget()),
			ptr_(std::malloc(amt * byte_size)),
			deallocate_([](void* pt){if(pt != nullptr){std::free(pt);}}),
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
			deallocate_(detail::change_function_input<void, From>(rhs.deallocate_)),
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
		inline intrusive_ptr operator+(const uint64_t i) const noexcept{
			utils::throw_exception(!is_null(), "Was null for +");
			retain_();
			return intrusive_ptr((void*)(reinterpret_cast<uint8_t*>(ptr_) + i),
					target_,
					deallocate_,
					device_,
					detail::DontIncreaseRefCount{});
		}
		inline bool is_null() const {
			return ptr_ == NullType::singleton() || target_ == TNullType::singleton();
		}
		inline void nullify() noexcept{
			null_self();
		}
		inline T* release() noexcept{
			T* returning = ptr_;
			null_self();
			return returning;
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
		
		static intrusive_ptr make_aligned(const uint64_t amt, std::size_t type_size, const std::size_t align_byte = ALIGN_BYTE_SIZE, std::function<void(void*)> deallocate = ([](void* ptr){std::free(ptr);})){
			uint64_t size = amt * type_size;
			if (size % align_byte != 0) size += align_byte - (size % align_byte);
			/* utils::throw_exception((amt * sizeof(T)) % align_byte == 0, "Cannot align $ bytes", amt * sizeof(T)); */
			return intrusive_ptr(std::aligned_alloc(align_byte, size), 
					[](void* ptr){std::free(ptr);});
		}

#ifdef USE_PARALLEL
		static intrusive_ptr make_shared(const uint64_t amt, const std::size_t byte_size, key_t key = IPC_PRIVATE){
			const uint64_t n_size = amt * byte_size;
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
#endif

		static intrusive_ptr to_cpu(const intrusive_ptr& ptr, const uint64_t amt, const std::size_t byte_size){
			if(ptr.is_cpu())
				return ptr;
			intrusive_ptr outp(amt, byte_size);
			const uint64_t n_size = amt * byte_size;
			const uint64_t end = n_size / sizeof(uint64_t);
			const uint64_t r = (n_size % sizeof(uint64_t)) / sizeof(uint8_t);
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
#ifdef USE_PARALLEL
		static intrusive_ptr to_shared(const intrusive_ptr& ptr, const uint64_t amt, const std::size_t byte_size, key_t key = IPC_PRIVATE){
			if(ptr.is_shared())
				return ptr;
			intrusive_ptr outp = intrusive_ptr::make_shared(amt, byte_size, key);
			const uint64_t n_size = amt * byte_size;
			const uint64_t end = n_size / sizeof(uint64_t);
			const uint64_t r = (n_size % sizeof(uint64_t)) / sizeof(uint8_t);
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
#endif
};

//that way both the pointer and refcount can be defined
template<class T,
	class TTarget,
	class NullType,
	class TNullType>
class intrusive_ptr<T*, TTarget, NullType, TNullType> final{
	template<class T2, class TTarget2, class NullType2, class TNullType2>
	friend class intrusive_ptr;
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
	T** ptr_;
	std::function<void(T**)> deallocate_;
	detail::Device device_;
	//as a way to protect from accidental deallocations or bad null checks, these are the only functions that can reall interact with the pointer past a copy
	inline bool is_null() const {
		return ptr_ == NullType::singleton() || target_ == TNullType::singleton();
	}
	inline bool both_null() const {
		return ptr_ == NullType::singleton() && target_ == TNullType::singleton();
	}
	inline bool one_null() const {
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
		/* std::cout << "reset_() called"<<std::endl; */
		if(target_ != TNullType::singleton()
				&& detail::atomic_refcount_decrement(target_->refcount_) <= 0){
			if(ptr_ != NullType::singleton()){
				/* std::cout << "going to deallocate"<<std::endl; */
				/* std::cout << "deallocating T**"<<std::endl; */
				deallocate_(ptr_);
				/* std::cout << "deallocated"<<std::endl; */
			}
			/* std::cout << "deallocating target T**"<<std::endl; */
			delete target_;
			return;
		}
		null_self();
	}
	inline void retain_() const{
		if(target_ != TNullType::singleton()){
			int64_t new_count = detail::atomic_refcount_increment(target_->refcount_);
			utils::throw_exception(new_count != 1, "intrusive_ptr: cannot increase refcount after it reaches zero got refcount of void*");
		}
	}

	
	explicit intrusive_ptr(T** ptr, TTarget* target, std::function<void(T**)> dealc, detail::Device dev = detail::Device::CPU)
		: intrusive_ptr(ptr, target, dealc, dev, detail::DontIncreaseRefCount{}) 
	{
		check_make_null();
		if(!is_null()){
			target_->refcount_.store(1, std::memory_order_relaxed);
		}
	}
	
	template<typename From>
	explicit intrusive_ptr(From** ptr)
		:target_(ptr == detail::intrusive_target_default_null_type<From>::singleton() ? TNullType::singleton() : new TTarget()), 
		ptr_(reinterpret_cast<T**>(ptr)),
		deallocate_([](T** p){From** pt = reinterpret_cast<From**>(p);if(pt != nullptr){delete[] pt;}}),
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
			deallocate_([](T** p){;}),
			device_(detail::Device::CPU)
			{}

		intrusive_ptr(std::nullptr_t) noexcept
			:target_(TNullType::singleton()),
			ptr_(NullType::singleton()),
			deallocate_([](T** p){;}),
			device_(detail::Device::CPU)
			{}
		
		template<typename From>
		explicit intrusive_ptr(From** ptr, std::function<void(T**)> dealc) noexcept
			:target_(ptr == detail::intrusive_target_default_null_type<From>::singleton() ? TNullType::singleton() : new TTarget()),
			ptr_(reinterpret_cast<T**>(ptr)),
			deallocate_(dealc),
			device_(detail::Device::CPU)
		{
			check_make_null();
			if(!is_null()){
				target_->refcount_.store(1, std::memory_order_relaxed);
			}
		}

		explicit intrusive_ptr(T** ptr, TTarget* target, std::function<void(T**)> dealc, detail::Device dev, detail::DontIncreaseRefCount) noexcept
			:target_(target),
			ptr_(ptr),
			deallocate_(dealc),
			device_(dev)
		{check_make_null();}

		explicit intrusive_ptr(int64_t amt)
			:target_(new TTarget()),
			ptr_(new T*[amt]),
			deallocate_([](T** pt){if(pt != nullptr){delete[] pt;}}),
			device_(detail::Device::CPU)
		{
			check_make_null();
			utils::throw_exception(ptr_ != NullType::singleton(), "Failure to allocate ptr_, void*");
			if(!is_null()){
				target_->refcount_.store(1, std::memory_order_relaxed);
			}

		}


		template<class From>
		explicit intrusive_ptr(std::unique_ptr<From*> rhs) noexcept
			:intrusive_ptr(rhs.release())
		{}

		inline ~intrusive_ptr() noexcept { reset_();}
		
		inline intrusive_ptr(intrusive_ptr&& rhs) noexcept
			:target_(rhs.target_), ptr_(rhs.ptr_), deallocate_(rhs.deallocate_), device_(rhs.device_)
			{rhs.null_self();}


		template<typename From, typename FromTarget, typename FromNull, typename FromNullT>
		intrusive_ptr(intrusive_ptr<From*, FromTarget, FromNull, FromNullT>&& rhs) noexcept
			:target_(detail::assign_ptr_<TTarget, TNullType, FromNullT>(rhs.target_)),
			ptr_(reinterpret_cast<T**>(rhs.ptr_)),
			deallocate_(detail::change_function_input<T*, From*>(rhs.deallocate_)),
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
		intrusive_ptr(const intrusive_ptr<From*, FromTarget, FromNullType, FromNullT>& rhs) noexcept
			:target_(detail::assign_ptr_<TTarget, TNullType, FromNullT>(rhs.target_)),
			ptr_(reinterpret_cast<T**>(rhs.ptr_)),
			deallocate_(detail::change_function_input<T*, From*>(rhs.deallocate_)),
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
		intrusive_ptr& operator=(intrusive_ptr<From*, FromTarget, FromNullType, FromNullT>&& rhs) & noexcept{

			static_assert(
				std::is_convertible<FromTarget*, TTarget*>::value,
				"Move warning, intrusive_ptr got pointer of wrong type");

			reset_();
			target_ = detail::assign_ptr_<TTarget, TNullType, FromNullT>(rhs.target_);
			ptr_ = reinterpret_cast<T**>(rhs.ptr_);
			deallocate_ = detail::change_function_input<T*, From*>(rhs.deallocate_);
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
		intrusive_ptr& operator=(const intrusive_ptr<From*, FromTarget, FromNullType, FromNullT>& rhs) & noexcept{
			static_assert(
				std::is_convertible<From*, TTarget*>::value,
				"Copy warning, intrusive_ptr got pointer of wrong type");
			reset_();
			target_ = detail::assign_ptr_<TTarget, TNullType, FromNullT>(rhs.target_);
			ptr_ = reinterpret_cast<T**>(rhs.ptr_);
			deallocate_ = detail::change_function_input<T*, From*>(rhs.deallocate_);
			device_ = rhs.device_;
			retain_();
			return *this;
		}

		inline T** operator->() const noexcept{
			return ptr_;
		}

		inline T** get() const noexcept{
			return ptr_;
		}
		inline T*& operator[](const std::ptrdiff_t size) noexcept{return ptr_[size];}
		inline const T* operator[](const std::ptrdiff_t size) const noexcept { return ptr_[size]; }
			inline intrusive_ptr operator+(const uint64_t i) const noexcept{
			utils::throw_exception(!is_null(), "Was null for +");
			retain_();
			return intrusive_ptr(ptr_ + i,
					target_,
					deallocate_,
					device_,
					detail::DontIncreaseRefCount{});
		}
		inline void nullify() noexcept{
			null_self();
		}
		inline T** release() noexcept{
			T** returning = ptr_;
			null_self();
			return returning;
		}
		void reset() noexcept{
			reset_();
		}
		
		//only viable on cpu versions, will probably be depreciated in future versions
		/* template<class Y> */
		/* inline void reset(Y** ptr){ */
		/* 	if(ptr_ != reinterpret_cast<T**>(ptr)){ */
		/* 		reset_(); */
		/* 		ptr_ = reinterpret_cast<T**>(ptr); */
		/* 		if(ptr_ != NullType::singleton()){ */
		/* 			target_ = new TTarget(); */
		/* 			target_->refcount_.store(1, std::memory_order_relaxed); */
		/* 			deallocate_ = ([](void* ptr){Y* p = reinterpret_cast<Y*>(ptr); delete p;}); */
		/* 			device_ = detail::Device::CPU; */
		/* 		} */
		/* 	} */
		/* } */
		inline int64_t use_count() const noexcept{
			if (target_ == TNullType::singleton())
				return 0;
			return target_->refcount_.load(std::memory_order_acquire);
		}
		/* inline void manual_usecount(const int64_t c){target_->refcount_.store(1, std::memory_order_relaxed);} //<- very unsafe, removing */
		
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
		
		static intrusive_ptr make_intrusive(T** ptr, std::function<void(T**)> dealc){
			return intrusive_ptr(ptr, new TTarget(), dealc);	
		}
		inline const detail::Device& device() const noexcept{ return device_;}
		inline const bool is_shared() const noexcept {return device_ == detail::Device::SharedCPU;}
		inline const bool is_cpu() const noexcept {return device_ == detail::Device::CPU;}
		
};



}
#endif
