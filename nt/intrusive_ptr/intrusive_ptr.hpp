#ifndef _NT_INTRUSIVE_PTR_HPP_
#define _NT_INTRUSIVE_PTR_HPP_

// heavily based off of
// https://github.com/pytorch/pytorch/blob/main/c10/util/intrusive_ptr.h

#include "../utils/utils.h"
// #include <_types/_uint32_t.h>
// #include <_types/_uint8_t.h>
#include <atomic>
#include <cassert>
#include <cstdlib> // For std::aligned_alloc
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
// #include <sys/_types/_key_t.h>
// #include <sys/ipc.h>
// #include <sys/shm.h>
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
        delete[] ptr;
    }
}

template <typename T> inline void defaultCStyleDeallocator(T *ptr) {
    if (!ptr) {
        return;
    }
    std::free(ptr);
}

template <typename TTarget> inline void defaultIntrusiveDeleter(void *ptr) {
    delete static_cast<TTarget *>(ptr);
}

inline void passIntrusiveDeleter(void *ptr) { ; }

inline void* portable_aligned_alloc(std::size_t alignment, std::size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}


} // namespace detail

class intrusive_ptr_target {
    mutable std::atomic<int64_t> refcount_;
    mutable std::atomic<int64_t> weakcount_;

    // Friend declaration for primary template intrusive_ptr
    template <class TTargetCK, class NullTypeCK> friend class intrusive_ptr;
    template <class TTargetCK, class NullTypeCK> friend class weak_intrusive_ptr;

    friend inline void
    detail::intrusive_ptr::incref(intrusive_ptr_target *self);

    friend inline void
    detail::weak_intrusive_ptr::incref(intrusive_ptr_target *self);

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
template <class TTarget, class NullType>
class weak_intrusive_ptr;

template <class TTarget,
          class NullType = detail::intrusive_target_default_null_type<TTarget>>
class intrusive_ptr;

// Partial specialization for arrays
template <class T>
class intrusive_ptr<T[], detail::intrusive_target_default_null_type<T[]>>;

// basically any class that uses an intrusive_ptr, must inherit from
// intrusive_ptr_target, that way the refcount can be made internally



//intrusive_ptr<T[]> does not currently have weak_intrusive_ptr support
template <typename T> class intrusive_ptr_target_array {
    friend class intrusive_ptr<T[],
                               detail::intrusive_target_default_null_type<T[]>>;
    mutable std::atomic<int64_t> refcount_;
    // mutable std::atomic<int64_t> weakcount_;
    friend inline void
    detail::intrusive_ptr::incref<>(intrusive_ptr_target_array<T> *self);
    // friend inline void
    // detail::weak_intrusive_ptr::incref(intrusive_ptr_target_array<T> *self);

    virtual void release_resources() {}

  protected:
    virtual ~intrusive_ptr_target_array() {
        /* utils::throw_exception(refcount_.load() == 0 || refcount_.load() >=
         * detail::kImpracticallyHugeReferenceCount, */
        /* 		"needed refcount to be too high or at 0 in order to release
         * resources"); */
        /* release_resources(); */
    }

    constexpr intrusive_ptr_target_array() : refcount_(0) {}

    // intrusive_ptr_target supports copy and move: but refcount and weakcount
    // don't participate (since they are intrinsic properties of the memory
    // location)
    intrusive_ptr_target_array(intrusive_ptr_target_array &&rhs) noexcept {}

    intrusive_ptr_target_array(const intrusive_ptr_target_array &rhs) noexcept {
    }

    intrusive_ptr_target_array &
    operator=(const intrusive_ptr_target_array &rhs) noexcept {
        return *this;
    }

    intrusive_ptr_target_array &
    operator=(intrusive_ptr_target_array &&rhs) noexcept {
        return *this;
    }
};

template <typename T> class intrusive_variable : public intrusive_ptr_target {
    T var;

  public:
    using type = T;
    intrusive_variable(T val) : var(val) {}

    template <typename... Args>
    intrusive_variable(Args &&...args) : var(std::forward<Args>(args)...) {}

    T &get() { return var; }
    const T &get() const { return var; }
};

// redo of intrusive_ptr:
template <class TTarget, class NullType> class intrusive_ptr final {
  public:
    using DeleterFunc = void (*)(void *);

  private:
    template <class TTarget2, class NullType2> friend class intrusive_ptr;
    friend class weak_intrusive_ptr<TTarget, NullType>;

#ifndef _WIN32
    static_assert(NullType::singleton() == NullType::singleton(),
                  "NullType must have a constexpr singleton method");
#endif

    static_assert(
        std::is_base_of_v<
            TTarget, std::remove_pointer_t<decltype(NullType::singleton())>>,
        "NullType::singleton() must return an target_type* pointer");

    TTarget *target_;
    DeleterFunc dealc;
    inline bool is_null() const { return target_ == NullType::singleton(); }

    inline void null_self() { target_ = NullType::singleton(); }
    inline void reset_() {
        if (target_ != NullType::singleton() &&
            detail::atomic_refcount_decrement(target_->refcount_) == 0) {
            //weakcount is always at least 1 by design
            bool _dealloc = detail::atomic_weakcount_fetch(target_->weakcount_) == 1;
            if(!_dealloc){
                const_cast<std::remove_const_t<TTarget>*>(target_)->release_resources();
                _dealloc =
                    detail::atomic_weakcount_decrement(target_->weakcount_) == 0;
            }
            if(_dealloc){
                dealc(target_);
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
        : target_(ptr), dealc(&detail::defaultIntrusiveDeleter<TTarget>) {
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
    explicit intrusive_ptr(TTarget *ptr, DeleterFunc func)
        : target_(ptr), dealc(func) {

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
        : intrusive_ptr(NullType::singleton(), &detail::passIntrusiveDeleter,
                        detail::DontIncreaseRefCount{}) {}

    intrusive_ptr(std::nullptr_t) noexcept
        : intrusive_ptr(NullType::singleton(), &detail::passIntrusiveDeleter,
                        detail::DontIncreaseRefCount{}) {}

    /* intrusive_ptr(nullptr) noexcept */
    /* 	:intrusive_ptr(NullType::singleton(), &detail::passIntrusiveDeleter,
     * detail::DontIncreaseRefCount{}) */
    /* 	{} */

    explicit intrusive_ptr(TTarget *target,
                           detail::DontIncreaseRefCount) noexcept
        : target_(target), dealc(&detail::defaultIntrusiveDeleter<TTarget>) {}

    explicit intrusive_ptr(TTarget *target, DeleterFunc func,
                           detail::DontIncreaseRefCount) noexcept
        : target_(target), dealc(func) {}

    explicit intrusive_ptr(std::unique_ptr<TTarget> rhs) noexcept
        : intrusive_ptr(rhs.release()) {}

    explicit intrusive_ptr(intrusive_ptr &&rhs) noexcept
        : target_(rhs.target_), dealc(rhs.dealc) {
        rhs.target_ = NullType::singleton();
    }
    inline ~intrusive_ptr() noexcept { reset_(); }

    template <typename FromTarget, typename FromNull>
    inline intrusive_ptr(intrusive_ptr<FromTarget, FromNull> &&rhs) noexcept
        : target_(
              detail::assign_ptr_<TTarget, NullType, FromNull>(rhs.target_)),
          dealc(rhs.dealc) {
        static_assert(std::is_convertible<FromTarget *, TTarget *>::value,
                      "Move warning, intrusive_ptr got pointer of wrong type");
        rhs.null_self();
    }

    inline intrusive_ptr(const intrusive_ptr &other)
        : target_(other.target_), dealc(other.dealc) {
        retain_();
    }

    template <typename FromTarget, typename FromNullType>
    inline intrusive_ptr(
        const intrusive_ptr<FromTarget, FromNullType> &rhs) noexcept
        : target_(detail::assign_ptr_<TTarget, NullType, FromNullType>(
              rhs.target_)),
          dealc(rhs.dealc) {
        static_assert(std::is_convertible<FromTarget *, TTarget *>::value,
                      "Move warning, intrusive_ptr got pointer of wrong type");
        retain_();
    }

    inline intrusive_ptr &operator=(intrusive_ptr &&rhs) & noexcept {
        // std::cout << "move = operator called for
        // intrusive_ptr<T>"<<std::endl;
        target_ = rhs.target_;
        dealc = rhs.dealc;
        rhs.null_self();
        return *this;
    }

    template <typename FromTarget, typename FromNullType>
    inline intrusive_ptr &
    operator=(intrusive_ptr<FromTarget, FromNullType> &&rhs) & noexcept {
        // std::cout << "move = operator called for
        // intrusive_ptr<others->T>"<<std::endl;
        static_assert(std::is_convertible<FromTarget *, TTarget *>::value,
                      "Move warning, intrusive_ptr got pointer of wrong type");

        intrusive_ptr tmp(std::move(rhs));
        swap(tmp);
        return *this;
    }

    inline intrusive_ptr &operator=(const intrusive_ptr &rhs) & noexcept {
        // std::cout << "copy = operator called for
        // intrusive_ptr<T>"<<std::endl;
        if (this != &rhs) {
            reset_();
            target_ = rhs.target_;
            dealc = rhs.dealc;
            retain_();
        }
        return *this;
    }

    template <typename FromTarget, typename FromNullType>
    inline intrusive_ptr &
    operator=(const intrusive_ptr<FromTarget, FromNullType> &rhs) & noexcept {
        // std::cout << "copy = operator called for
        // intrusive_ptr<others->T>"<<std::endl;
        static_assert(std::is_convertible<FromTarget *, TTarget *>::value,
                      "Copy warning, intrusive_ptr got pointer of wrong type");
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
        dealc = &detail::passIntrusiveDeleter;
    }
    
    //UNSAFE
    inline TTarget *release() noexcept {
        TTarget *returning = target_;
        target_ = NullType::singleton();
        dealc = &detail::passIntrusiveDeleter;
        return returning;
    }
    inline void reset() noexcept {
        reset_();
        target_ = NullType::singleton();
        dealc = &detail::passIntrusiveDeleter;
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
        std::swap(dealc, ptr.dealc);
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
        if constexpr (std::is_pointer_v<decltype(target_->refcount_)>) {
            return target_->refcount_->load(std::memory_order_acquire);
        } else {
            return target_->refcount_.load(std::memory_order_acquire);
        }

    }

    inline bool unique() const noexcept { return use_count() == 1; }

    template <class... Args> inline static intrusive_ptr make(Args &&...args) {
        return intrusive_ptr(new TTarget(std::forward<Args &&>(args)...));
    }

    inline static intrusive_ptr unsafe_steal_from_new(TTarget *target) {
        return intrusive_ptr<TTarget>(target);
    }

    // this is basically just making it a container to view the pointer that it
    // has inherited but once this intrusive_ptr goes out of scope, dont delete
    // it or anything
    inline static intrusive_ptr unsafe_act_as_view(TTarget *target) {
        return intrusive_ptr<TTarget>(target, &detail::passIntrusiveDeleter);
    }
};

// this is a way to hold memory intrusively, but, in an array
// so the usage will be intrusive_ptr<T[]>
// the first thing is to make an intrusive_ptr_array_target

template <class TTarget,
          class NullType = detail::intrusive_target_default_null_type<TTarget>,
          class... Args>
inline intrusive_ptr<TTarget, NullType> make_intrusive(Args &&...args) {
    return intrusive_ptr<TTarget, NullType>::make(std::forward<Args>(args)...);
}

template <class TTarget, class NullType>
inline void swap(intrusive_ptr<TTarget, NullType> &lhs,
                 intrusive_ptr<TTarget, NullType> &rhs) noexcept {
    lhs.swap(rhs);
}

// this is an intrusive_ptr hybrid designed for arrays
// it to an extent follows a shared_ptr where the refcount is internal to the
// class, where it is alocated
template <class T>
class intrusive_ptr<T[], detail::intrusive_target_default_null_type<T[]>>
    final {
  public:
    using DeleterFnArr = void (*)(T *);

  private:
    template <class T2, class NullType2> friend class intrusive_ptr;
    using NullType = detail::intrusive_target_default_null_type<T>;
    using TTarget = intrusive_ptr_target_array<T>;
    using TNullType = detail::intrusive_target_default_null_type<TTarget>;

#ifndef _WIN32
    static_assert(NullType::singleton() == NullType::singleton(),
                  "NullType must have a constexpr singleton method");

#endif

    static_assert(
        std::is_base_of_v<
            T, std::remove_pointer_t<decltype(NullType::singleton())>> ||
            std::is_same_v<
                T, std::remove_pointer_t<decltype(NullType::singleton())>>,
        "NullType::singleton() must return an T[] pointer");

    TTarget *target_;
    T *ptr_; // the reason there are 2 pointers, is because of the add operation
             // this operation allows the pointer to be easily freed
    T *original_;
    DeleterFnArr deallocate_;

    inline T *handle_amt_ptr(uint64_t amt) {
        if (amt == 0)
            return NullType::singleton();
        return new T[amt];
    }
    // as a way to protect from , these are the onl
    inline bool is_null() const {
        return ptr_ == NullType::singleton() ||
               target_ == TNullType::singleton() ||
               original_ == NullType::singleton();
    }
    inline bool both_null() const {
        return (ptr_ == NullType::singleton() ||
                original_ == NullType::singleton()) &&
               target_ == TNullType::singleton();
    }
    inline bool one_null() const { return !both_null() && is_null(); }
    inline void null_self() noexcept {
        target_ = TNullType::singleton();
        ptr_ = NullType::singleton();
        original_ = NullType::singleton();
    }
    inline void check_make_null() {
        if (is_null()) {
            if (both_null()) {
                null_self();
                return;
            } else if (ptr_ != NullType::singleton() &&
                       original_ == NullType::singleton())
                ptr_ = NullType::singleton();
            else if (ptr_ == NullType::singleton())
                delete target_;
            else if (target_ == TNullType::singleton()) {
                deallocate_(original_);
            }
            null_self();
        }
    }

    inline void reset_() {
        if (target_ != TNullType::singleton() &&
            detail::atomic_refcount_decrement(target_->refcount_) == 0) {
            /* std::cout << "deallocating T*"<<std::endl; */
            if (original_ != NullType::singleton())
                deallocate_(original_);
            /* std::cout << "deallocating target T*"<<std::endl; */
            delete target_;
        }
        null_self();
    }
    inline void retain_() const {
        if (target_ != TNullType::singleton()) {
            uint32_t new_count =
                detail::atomic_refcount_increment(target_->refcount_);
            utils::throw_exception(new_count != 1,
                                   "intrusive_ptr: cannot increase refcount "
                                   "after it reaches zero");
        }
    }

    explicit intrusive_ptr(T *ptr, TTarget *target, DeleterFnArr dealc)
        : intrusive_ptr(ptr, target, dealc, detail::DontIncreaseRefCount{}) {
        check_make_null();
        if (!is_null()) {
            target_->refcount_.store(1, std::memory_order_relaxed);
        }
    }

    explicit intrusive_ptr(T *ptr, T *original, TTarget *target,
                           DeleterFnArr dealc)
        : target_(target), ptr_(ptr), original_(original), deallocate_(dealc) {
        check_make_null();
        if (!is_null()) {
            target_->refcount_.store(1, std::memory_order_relaxed);
        }
    }

    explicit intrusive_ptr(T *ptr)
        : target_(ptr == NullType::singleton() ? TNullType::singleton()
                                               : new TTarget()),
          ptr_(ptr), original_(ptr),
          deallocate_(&detail::defaultCPPArrayDeallocator<T, NullType>) {
        check_make_null();
        if (!is_null()) {
            target_->refcount_.store(1, std::memory_order_relaxed);
        }
    }

  public:
    using element_type = T;
    using target_type = TTarget;
    using deleter_function_type = void (*)(T *);
    intrusive_ptr() noexcept
        : intrusive_ptr(NullType::singleton(), TNullType::singleton(),
                        &detail::defaultCPPArrayDeallocator<T, NullType>,
                        detail::DontIncreaseRefCount{}) {}

    intrusive_ptr(std::nullptr_t) noexcept
        : intrusive_ptr(NullType::singleton(), TNullType::singleton(),
                        &detail::defaultCPPArrayDeallocator<T, NullType>,
                        detail::DontIncreaseRefCount{}) {}

    explicit intrusive_ptr(T *ptr, TTarget *target, DeleterFnArr dealc,
                           detail::DontIncreaseRefCount) noexcept
        : target_(target), ptr_(ptr), original_(ptr), deallocate_(dealc) {
        check_make_null();
    }

    explicit intrusive_ptr(T *ptr, T *original, TTarget *target,
                           DeleterFnArr dealc,
                           detail::DontIncreaseRefCount) noexcept
        : target_(target), ptr_(ptr), original_(original), deallocate_(dealc) {
        check_make_null();
    }

    explicit intrusive_ptr(uint64_t amt) : intrusive_ptr(handle_amt_ptr(amt)) {
        if (amt == 0) {
            null_self();
        }
        deallocate_ = [](T *ptr) {
            if (ptr != NullType::singleton()) {
                delete[] ptr;
            }
        };
        utils::throw_exception(ptr_ != NullType::singleton() || amt == 0,
                               "Failure to allocate ptr_, T*");
    }

    explicit intrusive_ptr(std::unique_ptr<T[]> rhs) noexcept
        : intrusive_ptr(rhs.release()) {}

    template <class From>
    explicit intrusive_ptr(std::unique_ptr<From> rhs) noexcept
        : intrusive_ptr(reinterpret_cast<T *>(rhs.release())) {}

    inline ~intrusive_ptr() noexcept { reset_(); }

    inline intrusive_ptr(intrusive_ptr &&rhs) noexcept
        : target_(rhs.target_), ptr_(rhs.ptr_), original_(rhs.original_),
          deallocate_(rhs.deallocate_) {
        rhs.null_self();
    }

    template <typename From, typename FromNull>
    intrusive_ptr(intrusive_ptr<From[], FromNull> &&rhs) noexcept
        : target_(rhs.target_), ptr_(reinterpret_cast<T *>(rhs.ptr_)),
          original_(reinterpret_cast<T *>(rhs.original_)),
          deallocate_(
              detail::convert_function_input<T, From>(rhs.deallocate_)) {
        static_assert(std::is_convertible<From *, T *>::value,
                      "Move warning, intrusive_ptr got pointer of wrong type");
        rhs.null_self();
    }

    inline intrusive_ptr(const intrusive_ptr &other)
        : target_(other.target_), ptr_(other.ptr_), original_(other.original_),
          deallocate_(other.deallocate_) {
        retain_();
    }

    template <typename From, typename FromNullType>
    intrusive_ptr(const intrusive_ptr<From[], FromNullType> &rhs) noexcept
        : target_(rhs.target_), ptr_(reinterpret_cast<T *>(rhs.ptr_)),
          original_(reinterpret_cast<T *>(rhs.original_)),
          deallocate_(
              detail::convert_function_input<T, From>(rhs.deallocate_)) {
        static_assert(std::is_convertible<From *, T *>::value,
                      "Move warning, intrusive_ptr got pointer of wrong type");
        retain_();
    }

    intrusive_ptr &operator=(intrusive_ptr &&rhs) & noexcept {
        // std::cout << "move = operator called for
        // intrusive_ptr<T>"<<std::endl;
        target_ = rhs.target_;
        original_ = rhs.original_;
        ptr_ = rhs.ptr_;
        deallocate_ = rhs.deallocate_;
        rhs.null_self();
        /* return operator= <TTarget, NullType>(std::move(rhs)); */
        return *this;
    }

    template <typename From, typename FromNullType>
    intrusive_ptr &
    operator=(intrusive_ptr<From[], FromNullType> &&rhs) & noexcept {
        // std::cout << "move = operator called for
        // intrusive_ptr<others->T>"<<std::endl;
        static_assert(std::is_convertible<T *, From *>::value,
                      "Move warning, intrusive_ptr got pointer of wrong type");

        intrusive_ptr tmp(std::move(rhs));
        swap(tmp);
        return *this;
    }

    intrusive_ptr &operator=(const intrusive_ptr &rhs) & noexcept {
        // std::cout << "copy = operator called for
        // intrusive_ptr<T>"<<std::endl;
        if (this != &rhs) {
            reset_();
            target_ = rhs.target_;
            ptr_ = rhs.ptr_;
            original_ = rhs.original_;
            deallocate_ = rhs.deallocate_;
            retain_();
        }
        return *this;
    }

    template <typename From, typename FromNullType>
    intrusive_ptr &
    operator=(const intrusive_ptr<From[], FromNullType> &rhs) & noexcept {
        // std::cout << "copy = operator called for
        // intrusive_ptr<others->T>"<<std::endl;
        static_assert(std::is_convertible<From *, T *>::value,
                      "Copy warning, intrusive_ptr got pointer of wrong type");
        intrusive_ptr tmp(rhs);
        swap(tmp);
        return *this;
    }

    inline DeleterFnArr dealc_func() const { return deallocate_; }

    inline T *operator->() const noexcept { return ptr_; }

    template <typename U = T,
              typename = std::enable_if_t<!std::is_void<U>::value>>
    inline U &operator*() const noexcept {
        return *ptr_;
    }
    inline T *get() const noexcept { return ptr_; }
    template <typename IntegerType, typename U = T,
              typename = std::enable_if_t<!std::is_void<U>::value &&
                                          std::is_integral<IntegerType>::value>>
    inline U &operator[](const IntegerType i) const noexcept {
        return ptr_[i];
    }

    template <typename IntegerType,
              typename std::enable_if<std::is_integral<IntegerType>::value,
                                      int>::type = 0>
    inline intrusive_ptr operator+(const IntegerType i) const noexcept {
        retain_();
        if constexpr (!std::is_void_v<T>) {
            return intrusive_ptr(ptr_ + i, original_, target_, deallocate_,
                                 detail::DontIncreaseRefCount{});
        } else {
            return intrusive_ptr(reinterpret_cast<uint8_t *>(ptr_) + i,
                                 original_, target_, deallocate_,
                                 detail::DontIncreaseRefCount{});
        }
    }

    inline void nullify() noexcept { null_self(); }
    inline T *release() noexcept {
        T *returning = ptr_;
        null_self();
        return returning;
    }
    inline void reset() noexcept { reset_(); }

    template <class Y> inline void reset(Y *ptr) {
        if (original_ != reinterpret_cast<T *>(ptr)) {
            reset_();
            original_ = reinterpret_cast<T *>(ptr);
            ptr_ = original_;
            if (ptr_ != NullType::singleton()) {
                target_ = new TTarget();
                target_->refcount_.store(1, std::memory_order_relaxed);
            }
        }
    }
    inline int64_t use_count() const noexcept {
        if (target_ == TNullType::singleton())
            return 0;
        return target_->refcount_.load(std::memory_order_acquire);
    }

    inline bool defined() const noexcept { return !both_null(); }

    inline bool is_unique() const noexcept { return use_count() == 1; }

    inline void swap(intrusive_ptr &r) noexcept {
        std::swap(r.target_, target_);
        std::swap(r.ptr_, ptr_);
        std::swap(r.original_, original_);
        std::swap(r.deallocate_, deallocate_);
    }
    operator bool() const noexcept { return !both_null(); }

    static intrusive_ptr unsafe_make_from_raw_new(T *ptr) {
        return intrusive_ptr(ptr);
    }

    static intrusive_ptr
    make_aligned(const uint64_t amt,
                 const std::size_t align_byte = _NT_ALIGN_BYTE_SIZE_) {
        uint64_t size = amt * sizeof(T);
        if (size % align_byte != 0)
            size += align_byte - (size % align_byte);
        /* utils::throw_exception((amt * sizeof(T)) % align_byte == 0, "Cannot
         * align $ bytes", amt * sizeof(T)); */
        return intrusive_ptr(
            static_cast<T *>(detail::portable_aligned_alloc(align_byte, size)),
            new TTarget(), &detail::defaultCStyleDeallocator<T>);
    }

};


template <typename TTarget, class NullType = detail::intrusive_target_default_null_type<TTarget>>
class weak_intrusive_ptr final{
  public:
    using DeleterFunc = void (*)(void *);

  private:
    template <class TTarget2, class NullType2> friend class intrusive_ptr;

    template <class TTarget2, class NullType2> friend class weak_intrusive_ptr;

#ifndef _WIN32
    static_assert(NullType::singleton() == NullType::singleton(),
                  "NullType must have a constexpr singleton method");
#endif

    static_assert(
        std::is_base_of_v<
            TTarget, std::remove_pointer_t<decltype(NullType::singleton())>>,
        "NullType::singleton() must return an target_type* pointer");

    TTarget *target_;
    DeleterFunc dealc;
    inline void null_self() { target_ = NullType::singleton(); }
    inline bool is_null() const { return target_ == NullType::singleton(); }
    
    template <class TTarget2, class NullType2>
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
            dealc(target_);
        }
        this->null_self();
    }
    
    constexpr explicit weak_intrusive_ptr(TTarget *target) 
        :target_(target), dealc(&detail::defaultIntrusiveDeleter<TTarget>)
    {}

    constexpr explicit weak_intrusive_ptr(TTarget *target, DeleterFunc func)
    :target_(target), dealc(func)
    {}

public:
    using target_type = TTarget;

    explicit weak_intrusive_ptr(const intrusive_ptr<TTarget, NullType>& ptr)
    :weak_intrusive_ptr(ptr.get()) {
        this->retain_();
    }

    

    template <typename FromTarget, typename FromNull>
    weak_intrusive_ptr(const intrusive_ptr<FromTarget, FromNull> &rhs) noexcept
        : target_(
              detail::assign_ptr_<TTarget, NullType, FromNull>(rhs.target_)),
          dealc(rhs.dealc) {
        static_assert(std::is_convertible<FromTarget *, TTarget *>::value,
                      "Move warning, intrusive_ptr got pointer of wrong type");
        retain_();
    }
 

    template <typename FromTarget, typename FromNull>
    weak_intrusive_ptr(intrusive_ptr<FromTarget, FromNull> &&rhs) noexcept
        : target_(
              detail::assign_ptr_<TTarget, NullType, FromNull>(rhs.target_)),
          dealc(rhs.dealc) {
        static_assert(std::is_convertible<FromTarget *, TTarget *>::value,
                      "Move warning, weak_intrusive_ptr got pointer of wrong type");
        rhs.null_self();
    }
    
    weak_intrusive_ptr(const weak_intrusive_ptr& rhs) : target_(rhs.target_), dealc(rhs.dealc) {
        retain_();
    }

    inline weak_intrusive_ptr& operator=(weak_intrusive_ptr&& rhs) & noexcept{
        target_ = rhs.target_;
        dealc = rhs.dealc;
        rhs.null_self();
        return *this;
    }

    template <typename FromTarget, typename FromNullType>
    weak_intrusive_ptr& operator=(
      weak_intrusive_ptr<FromTarget, FromNullType>&& rhs) & noexcept {
        static_assert(std::is_convertible<FromTarget *, TTarget *>::value,
            "Move warning, weak_intrusive_ptr got pointer of wrong type");
        weak_intrusive_ptr tmp(std::move(rhs));
        swap(tmp);
        return *this;
    }

    weak_intrusive_ptr& operator=(const weak_intrusive_ptr& rhs) & noexcept {
        if(this == &rhs) return *this;
        reset_();
        target_ = rhs.target_;
        dealc = rhs.dealc;
        retain_();
    }

    template <typename FromTarget, typename FromNullType>
    weak_intrusive_ptr& operator=(
      const weak_intrusive_ptr<FromTarget, FromNullType>& rhs) & noexcept {
        static_assert(std::is_convertible<FromTarget *, TTarget *>::value,
            "Copy warning, weak_intrusive_ptr got pointer of wrong type");
        weak_intrusive_ptr tmp(rhs);
        swap(tmp);
        return *this;
    }

    inline void reset() noexcept {
        reset_();
    }
    
    inline void swap(weak_intrusive_ptr& ptr) noexcept {
        std::swap(target_, ptr.target_);
        std::swap(dealc, ptr.dealc);

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
        if (target_ == NullType::singleton()) {
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
    
    intrusive_ptr<TTarget, NullType> lock() const noexcept {
        if(expired()) return intrusive_ptr<TTarget, NullType>();
        if constexpr (std::is_pointer_v<decltype(target_->refcount_)>) {
            auto refcount = target_->refcount_->load(std::memory_order_acquire);
            do{
                if(refcount == 0){
                    //no strong references left
                    return intrusive_ptr<TTarget, NullType>();
                }
            }while(
                !target_->refcount_->compare_exchange_weak(refcount, refcount + 1));
            return intrusive_ptr<TTarget, NullType>(
                target_, dealc, detail::DontIncreaseRefCount{});
        } else {
            auto refcount = target_->refcount_.load(std::memory_order_acquire);
            do{
                if(refcount == 0){
                    //no strong references left
                    return intrusive_ptr<TTarget, NullType>();
                }
            }while(
                !target_->refcount_.compare_exchange_weak(refcount, refcount + 1));
            return intrusive_ptr<TTarget, NullType>(
                target_, dealc, detail::DontIncreaseRefCount{});
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
        dealc = &detail::passIntrusiveDeleter;
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

    //ptr must have come from weak_intrusive_ptr::release()
    inline static weak_intrusive_ptr reclaim(TTarget* owning_weak_ptr, DeleterFunc _dealc) {
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
        return weak_intrusive_ptr(owning_weak_ptr, _dealc);
    }
    
    // raw ptr retains ownership
    // new weak_intrusive_ptr representing a new weak reference
    static weak_intrusive_ptr reclaim_copy(TTarget* owning_ptr) {
        weak_intrusive_ptr out = weak_intrusive_ptr::reclaim(owning_ptr);
        out.retain_();
        return out;
    }
    
};


// T should be the parent
template <typename T>
class intrusive_parent_ptr_array : public intrusive_ptr_target {
    T *ptr;
    T *original;
    std::atomic<int64_t> *internal_refcount_;
    const std::size_t child_bytes;
    template <typename U> friend class intrusive_parent_ptr_array;

    template <typename TargetA, typename NullA> friend class intrusive_ptr;
    // std::function<void(T*)> deleter; = delete[] ptr;

    inline void null_self() {
        ptr = nullptr;
        original = nullptr;
        internal_refcount_ = nullptr;
    }
    inline void releaseMemory() {
        if (original &&
            detail::atomic_refcount_decrement(*internal_refcount_) == 0) {
            delete[] original;
            delete internal_refcount_;
        }
        original = nullptr;
        ptr = nullptr;
        internal_refcount_ = nullptr;
    }

    inline void retain_() const {
        if (original != nullptr) {
            int64_t new_count =
                nt::detail::atomic_refcount_increment(*internal_refcount_);
            utils::throw_exception(new_count != 1,
                                   "IntrusivePtrArray: cannot increase "
                                   "refcount after it reaches zero");
        }
    }
    explicit intrusive_parent_ptr_array(T *_ptr, T *_original,
                                        std::atomic<int64_t> *refcount_,
                                        std::size_t bytes)
        : ptr(_ptr), original(_original), internal_refcount_(refcount_),
          child_bytes(bytes) {}

  public:
    intrusive_parent_ptr_array()
        : ptr(nullptr), original(nullptr), internal_refcount_(nullptr),
          child_bytes(0) {}

    ~intrusive_parent_ptr_array() { releaseMemory(); }

    template <typename From>
    intrusive_parent_ptr_array(
        const intrusive_parent_ptr_array<From> &rhs) noexcept
        : ptr(rhs.ptr), original(rhs.original),
          internal_refcount_(rhs.internal_refcount_),
          child_bytes(rhs.child_bytes) {
        static_assert(std::is_convertible<From *, T *>::value,
                      "Move warning, intrusive_ptr got pointer of wrong type");
        retain_();
    }

    template <typename From>
    inline intrusive_parent_ptr_array &
    operator=(const intrusive_parent_ptr_array<From> &rhs) noexcept {
        static_assert(std::is_convertible<From *, T *>::value,
                      "Move warning, intrusive_ptr got pointer of wrong type");
        ptr = rhs.ptr;
        original = rhs.original;
        internal_refcount_ = rhs.internal_refcount_;
        child_bytes = rhs.child_bytes;
        retain_();
        return *this;
    }

    template <typename From>
    intrusive_parent_ptr_array(
        intrusive_parent_ptr_array<From> &&rhs) noexcept
        : ptr(rhs.ptr), original(rhs.original),
          internal_refcount_(rhs.internal_refcount_),
          child_bytes(rhs.child_bytes) {
        static_assert(std::is_convertible<From *, T *>::value,
                      "Move warning, intrusive_ptr got pointer of wrong type");
        rhs.null_self();
    }

    template <typename From>
    inline intrusive_parent_ptr_array &
    operator=(intrusive_parent_ptr_array<From> &&rhs) noexcept {

        static_assert(std::is_convertible<From *, T *>::value,
                      "Move warning, intrusive_ptr got pointer of wrong type");
        ptr = rhs.ptr;
        original = rhs.original;
        internal_refcount_ = rhs.internal_refcount_;
        child_bytes = rhs.child_bytes;
        rhs.null_self();
        return *this;
    }

    intrusive_parent_ptr_array(const intrusive_parent_ptr_array &rhs) noexcept
        : ptr(rhs.ptr), original(rhs.original),
          internal_refcount_(rhs.internal_refcount_),
          child_bytes(rhs.child_bytes) {
        retain_();
    }

    intrusive_parent_ptr_array(intrusive_parent_ptr_array &&rhs) noexcept
        : ptr(rhs.ptr), original(rhs.original),
          internal_refcount_(rhs.internal_refcount_),
          child_bytes(rhs.child_bytes) {
        rhs.null_self();
    }

    inline T *getMemory() { return ptr; }
    inline T *getMemory() const { return ptr; }

    inline T &operator[](const std::ptrdiff_t i) {
        return *reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(ptr) +
                                      (child_bytes * i));
    }
    inline T &operator[](const std::ptrdiff_t i) const {
        return *reinterpret_cast<const T *>(
            reinterpret_cast<const uint8_t *>(ptr) + (child_bytes * i));
    }

    inline intrusive_ptr<intrusive_parent_ptr_array>
    operator+(const std::ptrdiff_t i) const {
        if (!original) {
            return intrusive_ptr<intrusive_parent_ptr_array<T>>::
                unsafe_steal_from_new(
                    const_cast<intrusive_parent_ptr_array *>(this));
        }
        retain_();
        return intrusive_ptr<intrusive_parent_ptr_array>::make(
            reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(ptr) +
                                  (child_bytes * i)),
            original, internal_refcount_, child_bytes);
    }

    template <typename Parent, typename Child>
    inline static intrusive_ptr<intrusive_parent_ptr_array<Parent>>
    make_children(int64_t i) {
        Parent *original = new Child[i];
        std::atomic<int64_t> *counter = new std::atomic<int64_t>(1);
        counter->store(1, std::memory_order_relaxed);
        return intrusive_ptr<intrusive_parent_ptr_array<Parent>>::make(
            original, original, counter, sizeof(Child));
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


namespace std{
template<typename TTarget, typename NullType>
inline void swap(::nt::intrusive_ptr<TTarget, NullType>& lhs, ::nt::intrusive_ptr<TTarget, NullType>& rhs){
    lhs.swap(rhs);
}
template<typename TTarget, typename NullType>
inline void swap(::nt::weak_intrusive_ptr<TTarget, NullType>& lhs, ::nt::weak_intrusive_ptr<TTarget, NullType>& rhs){
    lhs.swap(rhs);
}
}

#endif //_NT_INTRUSIVE_PTR_HPP_
