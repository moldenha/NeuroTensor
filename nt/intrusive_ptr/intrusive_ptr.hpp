#ifndef NT_INTRUSIVE_PTR_HPP__
#define NT_INTRUSIVE_PTR_HPP__

// heavily based off of
// https://github.com/pytorch/pytorch/blob/main/c10/util/intrusive_ptr.h

#include "../utils/utils.h"
#include "../memory/meta_allocator.h" // MetaAlloc
#include <atomic>
#include <cassert>
#include <cstdlib> // For std::aligned_alloc
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <type_traits>

#ifdef _WIN32
#include <malloc.h>  // for _aligned_malloc and _aligned_free
#endif

#define _NT_ALIGN_BYTE_SIZE_ 32

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

namespace nt {

class intrusive_ptr_target;

template <typename T> class intrusive_ptr_target_array;

namespace detail {
template <class TTarget> struct intrusive_target_default_null_type final {
    static constexpr TTarget *singleton() noexcept { return nullptr; }
};

template<class TTarget>
struct intrusive_ptr_default_deleter final {
    inline void operator()(void* ptr) {if(ptr != nullptr){MetaFree<TTarget>(reinterpret_cast<TTarget*>(ptr));}}
};

template<class TTarget>
struct intrusive_ptr_default_deleter_arr {
    inline void operator()(void* ptr) {if(ptr != nullptr){MetaFreeArr<TTarget>(reinterpret_cast<TTarget*>(ptr));}}
};

template<class TTarget>
struct intrusive_ptr_default_deleter_arr<TTarget[]> {
    inline void operator()(void* ptr) {if(ptr != nullptr){MetaFreeArr<TTarget>(reinterpret_cast<TTarget*>(ptr));}}
};

struct intrusive_ptr_dont_delete final {
    inline void operator()(void* ptr) {;}
};



template<typename T>
struct intrusive_ptr_is_default_deleter : std::false_type {};
template<typename T>
struct intrusive_ptr_is_default_deleter<intrusive_ptr_default_deleter<T>> : std::true_type {};
template<typename T>
struct intrusive_ptr_is_default_deleter<intrusive_ptr_default_deleter_arr<T>> : std::true_type {};
template<typename T>
struct intrusive_ptr_is_default_deleter<intrusive_ptr_default_deleter_arr<T[]>> : std::true_type {};


struct DontIncreaseRefCount {};
struct DontOrderStrides {};
// increment needs to be aquire-release to make use_count() and
// unique() reliable
inline int64_t atomic_refcount_increment(std::atomic<int64_t> &refcount) {
    return refcount.fetch_add(1, std::memory_order_acq_rel) + 1;
}

inline int64_t atomic_weakcount_increment(std::atomic<int64_t> &weakcount) {
    return weakcount.fetch_add(1, std::memory_order_relaxed) + 1;
}


inline int64_t atomic_refcount_decrement(std::atomic<int64_t> &refcount) {
    return refcount.fetch_sub(1, std::memory_order_acq_rel) - 1;
}

inline int64_t atomic_weakcount_decrement(std::atomic<int64_t> &weakcount) {
    return weakcount.fetch_sub(1, std::memory_order_acq_rel) - 1;
}

inline int64_t atomic_refcount_fetch(const std::atomic<int64_t> &refcount) {
    return refcount.load(std::memory_order_acquire);
}

inline int64_t atomic_weakcount_fetch(const std::atomic<int64_t> &weakcount) {
    return weakcount.load(std::memory_order_acquire);
}


inline int64_t atomic_refcount_increment(std::atomic<int64_t> *refcount) {
    return refcount->fetch_add(1, std::memory_order_acq_rel) + 1;
}

inline int64_t atomic_weakcount_increment(std::atomic<int64_t> *weakcount) {
    return weakcount->fetch_add(1, std::memory_order_acq_rel) + 1;
}

inline int64_t atomic_refcount_decrement(std::atomic<int64_t> *refcount) {
    return refcount->fetch_sub(1, std::memory_order_acq_rel) - 1;
}

inline int64_t atomic_weakcount_decrement(std::atomic<int64_t> *weakcount) {
    return weakcount->fetch_sub(1, std::memory_order_acq_rel) - 1;
}


inline int64_t atomic_refcount_fetch(const std::atomic<int64_t> *refcount) {
    return refcount->load(std::memory_order_acquire);
}

inline int64_t atomic_weakcount_fetch(const std::atomic<int64_t> *weakcount) {
    return weakcount->load(std::memory_order_acquire);
}


template <typename To, typename From>
inline std::function<void(To *)>
change_function_input(const std::function<void(From *)> &func) {
    return std::function<void(To *)>(
        [&func](To *ptr) { func(reinterpret_cast<From *>(ptr)); });
}

template <typename U, typename T>
struct ConvertFunctionWrapper {
    static void (*stored_func)(U*);

    static void wrapper(T* ptr) {
        stored_func(static_cast<U*>(ptr));
    }
};

template <typename U, typename T>
void (*ConvertFunctionWrapper<U, T>::stored_func)(U*) = nullptr;

template <typename U, typename T>
void (*convert_function_input(void (*func_a)(U*)))(T*) {
    ConvertFunctionWrapper<U, T>::stored_func = func_a;
    return &ConvertFunctionWrapper<U, T>::wrapper;
}


template <class TTarget, class ToNullType, class FromNullType>
inline TTarget *assign_ptr_(TTarget *rhs) {
    if (FromNullType::singleton() == rhs) {
        return ToNullType::singleton();
    }
    return rhs;
}

/* template<class TTarget, class FromTarget, class ToNullType, class
 * FromNullType> */
/* inline TTarget* */
constexpr uint32_t kImpracticallyHugeReferenceCount = 0x0FFFFFFF;
namespace intrusive_ptr {
inline void incref(intrusive_ptr_target *self);
template <typename T> inline void incref(intrusive_ptr_target_array<T> *self);
} // namespace intrusive_ptr
namespace weak_intrusive_ptr{
inline void incref(intrusive_ptr_target* self);
} //namespace weak_intrusive_ptr

template <typename T> struct is_pointer {
    static constexpr bool value = false;
};

template <typename T> struct is_pointer<T *> {
    static constexpr bool value = true;
};
// enum class Device { SharedCPU, CPU }; // to come will be cuda, and mlx

template <typename, typename = void>
struct has_subscript_operator : std::false_type {};

template <typename T>
struct has_subscript_operator<
    T, std::void_t<decltype(std::declval<T &>()[std::declval<int64_t>()])>>
    : std::true_type {};

template <typename T, typename NullType>
inline void defaultCPPArrayDeallocator(T *ptr) {
    if (ptr != NullType::singleton()) {
        MetaFreeArr<T>(ptr);
    }
}

template<typename T>
inline void defaultAlignedDeallocator(T* ptr){
    if(!ptr) return;
    MetaAlignedFree(ptr);
}

template <typename TTarget> inline void defaultIntrusiveDeleter(void *ptr) {
    MetaFree<TTarget>(static_cast<TTarget *>(ptr));
}

inline void passIntrusiveDeleter(void *ptr) { ; }


} // namespace detail

class intrusive_ptr_target {
    mutable std::atomic<int64_t> refcount_;
    mutable std::atomic<int64_t> weakcount_;

    // Friend declaration for primary template intrusive_ptr
    template <class TTargetCK, class DeleteOpCK, class NullTypeCK> friend class intrusive_ptr;
    template <class TTargetCK, class DeleteOpCK, class NullTypeCK> friend class weak_intrusive_ptr;

    friend void detail::intrusive_ptr::incref(intrusive_ptr_target *self);

    friend void detail::weak_intrusive_ptr::incref(intrusive_ptr_target *self);

    template<typename TTargetCK>
    friend void MetaFreeArr(TTargetCK*);
    template<typename TTargetCK>
    friend void MetaFree(TTargetCK*);
    template<typename TTargetCK>
    friend TTargetCK* MetaNewArr_(int64_t, const char*, int);
    template<typename TTargetCK, typename... TargetArgs>
    friend TTargetCK* MetaNew_(const char*, int, TargetArgs&&...);


    virtual void release_resources() {}

  protected:
    virtual ~intrusive_ptr_target() {
         // utils::throw_exception(refcount_.load() == 0 || refcount_.load() >=
         // detail::kImpracticallyHugeReferenceCount,
         // 		"needed refcount to be too high or at 0 in order to release
         // resources");
         // utils::throw_exception(weakcount_.load() == 0 || weakcount_.load() == 1 ||
         //                       weakcount_.load() == detail::kImpracticallyHugeReferenceCount - 1 ||
         //                       weakcount_.load() == detail::kImpracticallyHugeReferenceCount,
         //        "Tried to destruct intrusive_ptr that has weak_intrusive_ptr");
        release_resources();
    }

    constexpr intrusive_ptr_target() : refcount_(0), weakcount_(0) {}

    // intrusive_ptr_target supports copy and move: but refcount and weakcount
    // don't participate (since they are intrinsic properties of the memory
    // location)
    intrusive_ptr_target(intrusive_ptr_target &&rhs) noexcept { ; }

    intrusive_ptr_target(const intrusive_ptr_target &rhs) noexcept {}

    intrusive_ptr_target &operator=(const intrusive_ptr_target &rhs) noexcept {
        return *this;
    }

    intrusive_ptr_target &operator=(intrusive_ptr_target &&rhs) noexcept {
        return *this;
    }
};

// Primary template declarations
template <class TTarget, class DeleteOp, class NullType>
class weak_intrusive_ptr;

template <class TTarget,
          class DeleteOp = detail::intrusive_ptr_default_deleter<TTarget>,
          class NullType = detail::intrusive_target_default_null_type<TTarget>>
class intrusive_ptr;


// basically any class that uses an intrusive_ptr, must inherit from
// intrusive_ptr_target, that way the refcount can be made internally

template <class TTarget, class DeleteOp, class NullType> 
class intrusive_ptr final {


  private:
    template <class TTarget2, class DeleteOp2, class NullType2> friend class intrusive_ptr;
    friend class weak_intrusive_ptr<TTarget, DeleteOp, NullType>;

    template<typename TTargetCK>
    friend void MetaFreeArr(TTargetCK*);
    template<typename TTargetCK>
    friend void MetaFree(TTargetCK*);
    template<typename TTargetCK>
    friend TTargetCK* MetaNewArr_(int64_t, const char*, int);
    template<typename TTargetCK, typename... TargetArgs>
    friend TTargetCK* MetaNew_(const char*, int, TargetArgs&&...);


#ifndef _WIN32
    static_assert(NullType::singleton() == NullType::singleton(),
                  "NullType must have a constexpr singleton method");
#endif

    static_assert(
        std::is_base_of_v<
            TTarget, std::remove_pointer_t<decltype(NullType::singleton())>>,
        "NullType::singleton() must return an target_type* pointer");

    TTarget *target_;
    // DeleterFunc dealc;
    inline bool is_null() const { return target_ == NullType::singleton(); }

    inline void null_self() { target_ = NullType::singleton(); }
    inline void reset_() {
        if (target_ != NullType::singleton() &&
            detail::atomic_refcount_decrement(target_->refcount_) == 0) {
            //weakcount is always at least 1 by design
            bool _dealloc = detail::atomic_weakcount_fetch(target_->weakcount_) == 1;
            if(!_dealloc){
                std::remove_const_t<TTarget>* tar = const_cast<std::remove_const_t<TTarget>*>(target_);
                tar->release_resources();
                _dealloc =
                    detail::atomic_weakcount_decrement(target_->weakcount_) == 0;
            }
            if(_dealloc){
                DeleteOp{}(target_);
                return;
            }
        }
        null_self();
    }

    inline void retain_() const {
        if (target_ != NullType::singleton()) {
            int64_t new_count =
                detail::atomic_refcount_increment(target_->refcount_);
            utils::throw_exception(new_count != 1,
                                   "intrusive_ptr: cannot increase refcount "
                                   "after it reaches zero");
        }
    }

    explicit intrusive_ptr(TTarget *ptr)
        : target_(ptr) {
        if (!is_null()) {
            if constexpr (std::is_pointer_v<decltype(target_->refcount_)>) {
                target_->refcount_->store(1, std::memory_order_relaxed);
                target_->weakcount_->store(1, std::memory_order_relaxed);
            } else {
                target_->refcount_.store(1, std::memory_order_relaxed);
                target_->weakcount_.store(1, std::memory_order_relaxed);
            }
        }
    }

  public:
    using target_type = TTarget;
    intrusive_ptr() noexcept
        : intrusive_ptr(NullType::singleton(),
                        detail::DontIncreaseRefCount{}) {}

    intrusive_ptr(std::nullptr_t) noexcept
        : intrusive_ptr(NullType::singleton(),
                        detail::DontIncreaseRefCount{}) {}

    /* intrusive_ptr(nullptr) noexcept */
    /* 	:intrusive_ptr(NullType::singleton(), &detail::passIntrusiveDeleter,
     * detail::DontIncreaseRefCount{}) */
    /* 	{} */

    explicit intrusive_ptr(TTarget *target,
                           detail::DontIncreaseRefCount) noexcept
        : target_(target) {}


    explicit intrusive_ptr(std::unique_ptr<TTarget, DeleteOp> rhs) noexcept
        : intrusive_ptr(rhs.release()) {}

    explicit intrusive_ptr(intrusive_ptr &&rhs) noexcept
        : target_(rhs.target_) {
        rhs.target_ = NullType::singleton();
    }
    inline ~intrusive_ptr() noexcept { reset_(); }

    template <typename FromTarget, typename FromDeleteOp, typename FromNull>
    inline intrusive_ptr(intrusive_ptr<FromTarget, FromDeleteOp, FromNull> &&rhs) noexcept
        : target_(
              detail::assign_ptr_<TTarget, NullType, FromNull>(rhs.target_)) {
        static_assert(std::is_convertible<FromTarget *, TTarget *>::value,
                      "Move warning, intrusive_ptr got pointer of wrong type");
        static_assert(std::is_same_v<FromDeleteOp, DeleteOp> ||
                      (detail::intrusive_ptr_is_default_deleter<FromDeleteOp>::value
                       && detail::intrusive_ptr_is_default_deleter<DeleteOp>::value),
                      "Move warning, intrusive_ptr got deleter of wrong type");

        rhs.null_self();
    }


    inline intrusive_ptr(const intrusive_ptr &other)
        : target_(other.target_) {
        retain_();
    }

    template <typename FromTarget, typename FromDeleteOp, typename FromNullType>
    inline intrusive_ptr(
        const intrusive_ptr<FromTarget, FromDeleteOp, FromNullType> &rhs) noexcept
        : target_(detail::assign_ptr_<TTarget, NullType, FromNullType>(
              rhs.target_)) {
        static_assert(std::is_convertible<FromTarget *, TTarget *>::value,
                      "Move warning, intrusive_ptr got pointer of wrong type");
        static_assert(std::is_same_v<FromDeleteOp, DeleteOp> ||
                      (detail::intrusive_ptr_is_default_deleter<FromDeleteOp>::value
                       && detail::intrusive_ptr_is_default_deleter<DeleteOp>::value),
                      "Copy warning, intrusive_ptr got deleter of wrong type");
        retain_();
    }




    inline intrusive_ptr &operator=(intrusive_ptr &&rhs) & noexcept {
        return this->template operator= <TTarget, DeleteOp, NullType>(std::move(rhs));
    }

    template <typename FromTarget, typename FromDeleteOp, typename FromNullType>
    inline intrusive_ptr &
    operator=(intrusive_ptr<FromTarget, FromDeleteOp, FromNullType> &&rhs) & noexcept {
        // std::cout << "move = operator called for
        // intrusive_ptr<others->T>"<<std::endl;
        static_assert(std::is_convertible<FromTarget *, TTarget *>::value,
                      "Move warning, intrusive_ptr got pointer of wrong type");
        static_assert(std::is_same_v<FromDeleteOp, DeleteOp> ||
                      (detail::intrusive_ptr_is_default_deleter<FromDeleteOp>::value
                       && detail::intrusive_ptr_is_default_deleter<DeleteOp>::value),
                      "Move warning, intrusive_ptr got deleter of wrong type");


        intrusive_ptr tmp(std::move(rhs));
        swap(tmp);
        return *this;
    }

    inline intrusive_ptr& operator=(const intrusive_ptr& rhs) & noexcept {
        return this->template operator= <TTarget, DeleteOp, NullType>(rhs);
    }

    template <typename FromTarget, typename FromDeleteOp, typename FromNullType>
    inline intrusive_ptr &
    operator=(const intrusive_ptr<FromTarget, FromDeleteOp, FromNullType> &rhs) & noexcept {
        // std::cout << "copy = operator called for
        // intrusive_ptr<others->T>"<<std::endl;
        static_assert(std::is_convertible<FromTarget *, TTarget *>::value,
                      "Copy warning, intrusive_ptr got pointer of wrong type");
        static_assert(std::is_same_v<FromDeleteOp, DeleteOp> ||
                      (detail::intrusive_ptr_is_default_deleter<FromDeleteOp>::value
                       && detail::intrusive_ptr_is_default_deleter<DeleteOp>::value),
                      "Copy warning, intrusive_ptr got deleter of wrong type");

        intrusive_ptr tmp(rhs);
        swap(tmp);
        return *this;
    }

    inline TTarget *operator->() const noexcept { return target_; }
    inline TTarget &operator*() const noexcept { return *target_; }
    inline TTarget *get() const noexcept { return target_; }

    // Enable operator[] only if TTarget has an operator[], support currenly not
    // enabled
    /* template<typename IntegerType, typename
     * std::enable_if<std::is_integral<IntegerType>::value &&
     * detail::has_subscript_operator<TTarget>::value, int>::type = 0> */
    /* inline auto operator[](const int64_t i) const noexcept ->
     * std::enable_if_t<detail::has_subscript_operator<TTarget>::value,
     * decltype((*target_)[i])> { */
    /* return (*target_)[i]; */
    /* } */

    /* template<typename IntegerType, typename
     * std::enable_if<std::is_integral<IntegerType>::value &&
     * detail::has_subscript_operator<TTarget>::value, int>::type = 0> */
    /* inline auto operator[](const int64_t i) noexcept ->
     * std::enable_if_t<detail::has_subscript_operator<TTarget>::value,
     * decltype((*target_)[i])> { */
    /* return (*target_)[i]; */
    /* } */
    template <typename IntegerType,
              typename std::enable_if<
                  std::is_integral<IntegerType>::value &&
                      detail::has_subscript_operator<TTarget>::value,
                  int>::type = 0>
    inline auto operator[](const IntegerType i) const noexcept {
        return (*target_)[i];
    }

    template <typename IntegerType,
              typename std::enable_if<
                  std::is_integral<IntegerType>::value &&
                      detail::has_subscript_operator<TTarget>::value,
                  int>::type = 0>
    inline auto operator[](const IntegerType i) noexcept {
        return (*target_)[i];
    }

    // Fallback for when TTarget does not have an operator[]
    template <typename IntegerType,
              typename std::enable_if<
                  std::is_integral<IntegerType>::value &&
                      !detail::has_subscript_operator<TTarget>::value,
                  int>::type = 0>
    inline auto operator[](const IntegerType i) const noexcept {
        static_assert(detail::has_subscript_operator<TTarget>::value,
                      "TTarget does not support operator[]");
        return typename std::remove_reference<decltype((*target_))>::type();
    }

    template <typename IntegerType,
              typename std::enable_if<
                  std::is_integral<IntegerType>::value &&
                      !detail::has_subscript_operator<TTarget>::value,
                  int>::type = 0>
    inline auto operator[](const IntegerType i) noexcept {
        static_assert(detail::has_subscript_operator<TTarget>::value,
                      "TTarget does not support operator[]");
        return typename std::remove_reference<decltype((*target_))>::type();
    }

    //UNSAFE
    inline void nullify() noexcept {
        target_ = NullType::singleton();
    }
    
    //UNSAFE
    inline TTarget *release() noexcept {
        TTarget *returning = target_;
        target_ = NullType::singleton();
        return returning;
    }
    inline void reset() noexcept {
        reset_();
        target_ = NullType::singleton();
    }

    inline operator bool() const noexcept {
        return target_ != NullType::singleton();
    }

    inline bool operator==(TTarget *ptr) const noexcept {
        if (ptr == nullptr) {
            return target_ == nullptr || !(*this);
        }
        return target_ == ptr;
    }

    inline bool operator==(const intrusive_ptr &ptr) const noexcept {
        if (is_null() && ptr.is_null()) {
            return true;
        }
        if (is_null() || ptr.is_null()) {
            return false;
        }
        return target_ == ptr.target_;
    }

    inline bool operator!=(TTarget *ptr) const noexcept {
        return !(*this == ptr);
    }

    inline bool operator!=(const intrusive_ptr &ptr) const noexcept {
        return !(*this == ptr);
    }

    inline void swap(intrusive_ptr &ptr) noexcept {
        std::swap(target_, ptr.target_);
    }

    inline bool defined() const noexcept {
        return target_ != NullType::singleton();
    }

    inline int64_t use_count() const noexcept {
        if (target_ == NullType::singleton()) {
            return 0;
        }
        if constexpr (std::is_pointer_v<decltype(target_->refcount_)>) {
            return target_->refcount_->load(std::memory_order_acquire);
        } else {
            return target_->refcount_.load(std::memory_order_acquire);
        }
    }

    inline int64_t weak_use_count() const noexcept {
        if(target_ == NullType::singleton()){
            return 0;
        }
        if constexpr (std::is_pointer_v<decltype(target_->weakcount_)>) {
            return target_->weakcount_->load(std::memory_order_acquire);
        } else {
            return target_->weakcount_.load(std::memory_order_acquire);
        }

    }

    inline bool unique() const noexcept { return use_count() == 1; }

    template <class... Args> inline static intrusive_ptr make(Args &&...args) {
        return intrusive_ptr(MetaNew(TTarget, std::forward<Args &&>(args)...));
    }

    inline static intrusive_ptr unsafe_steal_from_new(TTarget *target) {
        return intrusive_ptr<TTarget>(target);
    }

    // this is basically just making it a container to view the pointer that it
    // has inherited but once this intrusive_ptr goes out of scope, dont delete
    // it or anything
    inline static intrusive_ptr<TTarget, detail::intrusive_ptr_dont_delete, NullType> unsafe_act_as_view(TTarget *target) {
        return intrusive_ptr<TTarget, detail::intrusive_ptr_dont_delete, NullType>(target);
    }
};

// this is a way to hold memory intrusively, but, in an array
// so the usage will be intrusive_ptr<T[]>
// the first thing is to make an intrusive_ptr_array_target

template <class TTarget,
          class DeleteOp = detail::intrusive_ptr_default_deleter<TTarget>,
          class NullType = detail::intrusive_target_default_null_type<TTarget>,
          class... Args>
inline intrusive_ptr<TTarget, DeleteOp, NullType> make_intrusive(Args &&...args) {
    return intrusive_ptr<TTarget, DeleteOp, NullType>::make(std::forward<Args>(args)...);
}

template <class TTarget, class DeleteOp, class NullType>
inline void swap(intrusive_ptr<TTarget, DeleteOp, NullType> &lhs,
                 intrusive_ptr<TTarget, DeleteOp, NullType> &rhs) noexcept {
    lhs.swap(rhs);
}


template <typename TTarget, class DeleteOp = detail::intrusive_ptr_default_deleter<TTarget>, class NullType = detail::intrusive_target_default_null_type<TTarget>>
class weak_intrusive_ptr final{

  private:
    template <class TTarget2, class DeleteOp2, class NullType2> friend class intrusive_ptr;

    template <class TTarget2, class DeleteOp2, class NullType2> friend class weak_intrusive_ptr;

#ifndef _WIN32
    static_assert(NullType::singleton() == NullType::singleton(),
                  "NullType must have a constexpr singleton method");
#endif

    static_assert(
        std::is_base_of_v<
            TTarget, std::remove_pointer_t<decltype(NullType::singleton())>>,
        "NullType::singleton() must return an target_type* pointer");

    TTarget *target_;
    inline void null_self() { target_ = NullType::singleton(); }
    inline bool is_null() const { return target_ == NullType::singleton(); }
    
    template <class TTarget2, class DeleteOp2, class NullType2>
    friend class weak_intrusive_ptr;

    inline void retain_(){
        if (target_ != NullType::singleton()) {
            int64_t new_count = detail::atomic_weakcount_increment(target_->weakcount_);
            utils::throw_exception(
                new_count != 1,
                "weak_intrusive_ptr: cannot increment count after reaching 0");
        }
    }

    inline void reset_() noexcept {
        if(target_ != NullType::singleton()
            && detail::atomic_weakcount_decrement(target_->weakcount_) == 0){
            DeleteOp{}(target_);
        }
        this->null_self();
    }
    
    constexpr explicit weak_intrusive_ptr(TTarget *target) 
        :target_(target)
    {}


public:
    using target_type = TTarget;

    explicit weak_intrusive_ptr(const intrusive_ptr<TTarget, DeleteOp, NullType>& ptr)
    :weak_intrusive_ptr(ptr.get()) {
        this->retain_();
    }

    // weak ptr copy and move constructors

    weak_intrusive_ptr(weak_intrusive_ptr&& rhs) noexcept : target_(rhs.target_) {
        rhs.null_self();
    }

    template <typename FromTarget, typename FromDeleteOp, typename FromNull>
    weak_intrusive_ptr(weak_intrusive_ptr<FromTarget, FromDeleteOp, FromNull>&& rhs) noexcept 
        : target_(
              detail::assign_ptr_<TTarget, NullType, FromNull>(rhs.target_)){
        static_assert(std::is_convertible<FromTarget *, TTarget *>::value,
                      "Move warning, intrusive_ptr got pointer of wrong type");
        static_assert(std::is_same_v<FromDeleteOp, DeleteOp> ||
                      (detail::intrusive_ptr_is_default_deleter<FromDeleteOp>::value
                       && detail::intrusive_ptr_is_default_deleter<DeleteOp>::value),
                      "Move warning, intrusive_ptr got deleter of wrong type");
        rhs.null_self();
    }

    weak_intrusive_ptr(const weak_intrusive_ptr& rhs) : target_(rhs.target_) {
        retain_();
    }

    template <typename FromTarget, typename FromDeleteOp, typename FromNull>
    weak_intrusive_ptr(const weak_intrusive_ptr<FromTarget, FromDeleteOp, FromNull>& rhs) 
    : target_(
        detail::assign_ptr_<TTarget, NullType, FromNull>(rhs.target_)){
        static_assert(
            std::is_convertible_v<FromTarget*, TTarget*>,
            "Type mismatch. weak_intrusive_ptr copy constructor got pointer of wrong type.");
        static_assert(std::is_same_v<FromDeleteOp, DeleteOp> ||
                      (detail::intrusive_ptr_is_default_deleter<FromDeleteOp>::value
                       && detail::intrusive_ptr_is_default_deleter<DeleteOp>::value),
                      "Copy warning, intrusive_ptr got deleter of wrong type");

        retain_();
    }

    
    ~weak_intrusive_ptr() noexcept {reset_();}
    

    inline weak_intrusive_ptr& operator=(weak_intrusive_ptr&& rhs) & noexcept{
        return this->template operator= <TTarget, DeleteOp, NullType>(std::move(rhs));
    }

    template <typename FromTarget, typename FromDeleteOp, typename FromNullType>
    weak_intrusive_ptr& operator=(
      weak_intrusive_ptr<FromTarget, FromDeleteOp, FromNullType>&& rhs) & noexcept {
        static_assert(std::is_convertible<FromTarget *, TTarget *>::value,
            "Move warning, weak_intrusive_ptr got pointer of wrong type");
        static_assert(std::is_same_v<FromDeleteOp, DeleteOp> ||
                      (detail::intrusive_ptr_is_default_deleter<FromDeleteOp>::value
                       && detail::intrusive_ptr_is_default_deleter<DeleteOp>::value),
                      "Move warning, intrusive_ptr got deleter of wrong type");
        weak_intrusive_ptr tmp(std::move(rhs));
        swap(tmp);
        return *this;
    }

    weak_intrusive_ptr& operator=(const weak_intrusive_ptr& rhs) & noexcept {
        if(this == &rhs) return *this;
        return this->template operator= <TTarget, DeleteOp, NullType>(rhs);
    }

    template <typename FromTarget, typename FromDeleteOp, typename FromNullType>
    weak_intrusive_ptr& operator=(
      const weak_intrusive_ptr<FromTarget, FromDeleteOp, FromNullType>& rhs) & noexcept {
        static_assert(std::is_convertible<FromTarget *, TTarget *>::value,
            "Copy warning, weak_intrusive_ptr got pointer of wrong type");
        static_assert(std::is_same_v<FromDeleteOp, DeleteOp> ||
                      (detail::intrusive_ptr_is_default_deleter<FromDeleteOp>::value
                       && detail::intrusive_ptr_is_default_deleter<DeleteOp>::value),
                      "Move warning, intrusive_ptr got deleter of wrong type");

        weak_intrusive_ptr tmp(rhs);
        swap(tmp);
        return *this;
    }

    inline void reset() noexcept {
        reset_();
    }
    
    inline void swap(weak_intrusive_ptr& ptr) noexcept {
        std::swap(target_, ptr.target_);

    }
    
    inline TTarget* _unsafe_get_target() const noexcept {
        return target_;
    }

   
    inline int64_t use_count() const noexcept {
        if (target_ == NullType::singleton()) {
            return 0;
        }
        if constexpr (std::is_pointer_v<decltype(target_->refcount_)>) {
            return target_->refcount_->load(std::memory_order_acquire);
        } else {
            return target_->refcount_.load(std::memory_order_acquire);
        }
    }

    inline int64_t weak_use_count() const noexcept {
        if(target_ == NullType::singleton()){
            return 0;
        }
        if constexpr (std::is_pointer_v<decltype(target_->weakcount_)>) {
            return target_->weakcount_->load(std::memory_order_acquire);
        } else {
            return target_->weakcount_.load(std::memory_order_acquire);
        }
    }

    inline bool expired() const noexcept {
        return use_count() == 0;
    }
    
    intrusive_ptr<TTarget, DeleteOp, NullType> lock() const noexcept {
        if(expired()) return intrusive_ptr<TTarget, DeleteOp, NullType>();
        if constexpr (std::is_pointer_v<decltype(target_->refcount_)>) {
            auto refcount = target_->refcount_->load(std::memory_order_acquire);
            do{
                if(refcount == 0){
                    //no strong references left
                    return intrusive_ptr<TTarget, DeleteOp, NullType>();
                }
            }while(
                !target_->refcount_->compare_exchange_weak(refcount, refcount + 1));
            return intrusive_ptr<TTarget, DeleteOp, NullType>(
                target_, detail::DontIncreaseRefCount{});
        } else {
            auto refcount = target_->refcount_.load(std::memory_order_acquire);
            do{
                if(refcount == 0){
                    //no strong references left
                    return intrusive_ptr<TTarget, DeleteOp, NullType>();
                }
            }while(
                !target_->refcount_.compare_exchange_weak(refcount, refcount + 1));
            return intrusive_ptr<TTarget, DeleteOp, NullType>(
                target_, detail::DontIncreaseRefCount{});
        }
    }

    // retrurns the owning pointer
    // but is weakly referenced
    //  [*] must be put back into weak_intrusive_ptr
    // if handled improperly is unsafe
    //
    inline TTarget *release() noexcept {
        TTarget *returning = target_;
        target_ = NullType::singleton();
        return returning;
    }

    //ptr must have come from weak_intrusive_ptr::release().
    inline static weak_intrusive_ptr reclaim(TTarget* owning_weak_ptr) {
        if constexpr (std::is_pointer_v<decltype(owning_weak_ptr->refcount_)>) {
            utils::throw_exception(
                owning_weak_ptr == NullType::singleton() ||
                owning_weak_ptr->weakcount_->load() > 1 ||
                (owning_weak_ptr->refcount_->load() == 0 &&
                owning_weak_ptr->weakcount_->load() > 0),
                "Pointer passed into weak_intrusive_ptr::reclaim must"
                "have come from weak_intrusive_ptr::release");
        }
        else {
            utils::throw_exception(
                owning_weak_ptr == NullType::singleton() ||
                owning_weak_ptr->weakcount_.load() > 1 ||
                (owning_weak_ptr->refcount_.load() == 0 &&
                owning_weak_ptr->weakcount_.load() > 0),
                "Pointer passed into weak_intrusive_ptr::reclaim must"
                "have come from weak_intrusive_ptr::release");
        }
        return weak_intrusive_ptr(owning_weak_ptr);
    }

    
    // raw ptr retains ownership
    // new weak_intrusive_ptr representing a new weak reference
    static weak_intrusive_ptr reclaim_copy(TTarget* owning_ptr) {
        weak_intrusive_ptr out = weak_intrusive_ptr::reclaim(owning_ptr);
        out.retain_();
        return out;
    }
    
};



namespace detail{

namespace intrusive_ptr {
inline void incref(intrusive_ptr_target *self){
    if(self){
        atomic_refcount_increment(self->refcount_);
    }
}
template <typename T> inline void incref(intrusive_ptr_target_array<T> *self){
    if(self){
        atomic_refcount_increment(self->refcount_);
    }
}

} // namespace intrusive_ptr
namespace weak_intrusive_ptr{
inline void incref(intrusive_ptr_target* self){
    if(self){
        atomic_weakcount_increment(self->weakcount_);
    }
}
} //namespace weak_intrusive_ptr

} //namespace detail

namespace utils {

template <typename T> struct is_intrusive_ptr : std::false_type {};

// partial specialization for intrusive_ptr
template <typename U>
struct is_intrusive_ptr<intrusive_ptr<U>> : std::true_type {};

// helper variable template
template <typename T>
inline constexpr bool is_intrusive_ptr_v = is_intrusive_ptr<T>::value;

} // namespace utils

} // namespace nt


#include <utility> // std::hash

namespace std{
template<typename TTarget, typename NullType>
inline void swap(::nt::intrusive_ptr<TTarget, NullType>& lhs, ::nt::intrusive_ptr<TTarget, NullType>& rhs){
    lhs.swap(rhs);
}
template<typename TTarget, typename NullType>
inline void swap(::nt::weak_intrusive_ptr<TTarget, NullType>& lhs, ::nt::weak_intrusive_ptr<TTarget, NullType>& rhs){
    lhs.swap(rhs);
}

// To allow intrusive_ptr and weak_intrusive_ptr inside std::unordered_map or
// std::unordered_set, there needs to be a hash
template <class TTarget, class DeleteOp, class NullType>
struct hash<nt::intrusive_ptr<TTarget, DeleteOp, NullType>> {
  size_t operator()(const nt::intrusive_ptr<TTarget, DeleteOp, NullType>& x) const {
    return std::hash<TTarget*>()(x.get());
  }
};
template <class TTarget, class DeleteOp, class NullType>
struct hash<nt::weak_intrusive_ptr<TTarget, DeleteOp, NullType>> {
  size_t operator()(const nt::weak_intrusive_ptr<TTarget, DeleteOp, NullType>& x) const {
    return std::hash<TTarget*>()(x._unsafe_get_target());
  }
};

}

#endif //NT_INTRUSIVE_PTR_HPP__
