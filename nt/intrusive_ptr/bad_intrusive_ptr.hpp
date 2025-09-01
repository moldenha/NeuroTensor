namespace nt{


// Partial specialization for arrays
template <class T>
class intrusive_ptr<T[], detail::intrusive_ptr_default_deleter_arr<T>, detail::intrusive_target_default_null_type<T[]>>;


//intrusive_ptr<T[]> does not currently have weak_intrusive_ptr support
template <typename T> class intrusive_ptr_target_array {
    friend class intrusive_ptr<T[],
                                detail::intrusive_ptr_default_deleter_arr<T>,
                               detail::intrusive_target_default_null_type<T[]>>;
    mutable std::atomic<int64_t> refcount_;
    // mutable std::atomic<int64_t> weakcount_;
    friend void detail::intrusive_ptr::incref<>(intrusive_ptr_target_array<T> *self);
    // friend void
    // detail::weak_intrusive_ptr::incref(intrusive_ptr_target_array<T> *self);

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


// this is an intrusive_ptr hybrid designed for arrays
// it to an extent follows a shared_ptr where the refcount is internal to the
// class, where it is alocated
template <class T>
class intrusive_ptr<T[], detail::intrusive_ptr_default_deleter_arr<T>, detail::intrusive_target_default_null_type<T[]>>
    final {
  private:
    template <class T2, class DeleteOp2, class NullType2> friend class intrusive_ptr;
    using NullType = detail::intrusive_target_default_null_type<T>;
    using DeleteArrOp = detail::intrusive_ptr_default_deleter_arr<T>; 
    using TTarget = intrusive_ptr_target_array<T>;
    using DeleteOp = detail::intrusive_ptr_default_deleter<TTarget>;
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

    inline T *handle_amt_ptr(uint64_t amt) {
        if (amt == 0)
            return NullType::singleton();
        return MetaNewArr(T, amt);
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
                DeleteOp{}(target_);
            else if (target_ == TNullType::singleton()) {
                DeleteArrOp{}(original_);
            }
            null_self();
        }
    }

    inline void reset_() {
        if (target_ != TNullType::singleton() &&
            detail::atomic_refcount_decrement(target_->refcount_) == 0) {
            /* std::cout << "deallocating T*"<<std::endl; */
            if (original_ != NullType::singleton())
                DeleteArrOp{}(original_);
            /* std::cout << "deallocating target T*"<<std::endl; */
            DeleteOp{}(target_);
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

    explicit intrusive_ptr(T *ptr, TTarget *target)
        : intrusive_ptr(ptr, target, detail::DontIncreaseRefCount{}) {
        check_make_null();
        if (!is_null()) {
            target_->refcount_.store(1, std::memory_order_relaxed);
        }
    }

    explicit intrusive_ptr(T *ptr, T *original, TTarget *target)
        : target_(target), ptr_(ptr), original_(original) {
        check_make_null();
        if (!is_null()) {
            target_->refcount_.store(1, std::memory_order_relaxed);
        }
    }

    explicit intrusive_ptr(T *ptr)
        : target_(ptr == NullType::singleton() ? TNullType::singleton()
                                               : MetaNew(TTarget, )),
          ptr_(ptr), original_(ptr) {
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
                        detail::DontIncreaseRefCount{}) {}

    intrusive_ptr(std::nullptr_t) noexcept
        : intrusive_ptr(NullType::singleton(), TNullType::singleton(),
                        detail::DontIncreaseRefCount{}) {}

    explicit intrusive_ptr(T *ptr, TTarget *target,
                           detail::DontIncreaseRefCount) noexcept
        : target_(target), ptr_(ptr), original_(ptr) {
        check_make_null();
    }

    explicit intrusive_ptr(T *ptr, T *original, TTarget *target,
                           detail::DontIncreaseRefCount) noexcept
        : target_(target), ptr_(ptr), original_(original) {
        check_make_null();
    }

    explicit intrusive_ptr(uint64_t amt) : intrusive_ptr(handle_amt_ptr(amt)) {
        if (amt == 0) {
            null_self();
        }
        utils::throw_exception(ptr_ != NullType::singleton() || amt == 0,
                               "Failure to allocate ptr_, T*");
    }

    explicit intrusive_ptr(std::unique_ptr<T[], DeleteArrOp> rhs) noexcept
        : intrusive_ptr(rhs.release()) {}

    template <class From>
    explicit intrusive_ptr(std::unique_ptr<From> rhs) noexcept
        : intrusive_ptr(reinterpret_cast<T *>(rhs.release())) {}

    inline ~intrusive_ptr() noexcept { reset_(); }


    explicit intrusive_ptr(intrusive_ptr &&rhs) noexcept
        : target_(rhs.target_), ptr_(rhs.ptr_), original_(rhs.original_){
        rhs.null_self();
    }

    template <typename FromTarget, typename FromDeleteArrOp, typename FromNull>
    intrusive_ptr(intrusive_ptr<FromTarget[], FromDeleteArrOp, FromNull> &&rhs) noexcept
        : target_(rhs.target_), ptr_(reinterpret_cast<T *>(rhs.ptr_)),
          original_(reinterpret_cast<T *>(rhs.original_)) {
        static_assert(std::is_convertible<FromTarget *, T *>::value,
                      "Move warning, intrusive_ptr got pointer of wrong type");
        static_assert(std::is_same_v<FromDeleteArrOp, DeleteArrOp> ||
                      (detail::intrusive_ptr_is_default_deleter<FromDeleteArrOp>::value
                       && detail::intrusive_ptr_is_default_deleter<DeleteArrOp>::value),
                      "Move warning, intrusive_ptr got deleter of wrong type");
        rhs.null_self();
    }


    inline intrusive_ptr(const intrusive_ptr &other)
        : target_(other.target_), ptr_(other.ptr_), original_(other.original_){
        retain_();
    }

    template <typename From, typename FromDeleteArrOp, typename FromNullType>
    intrusive_ptr(const intrusive_ptr<From[], FromDeleteArrOp, FromNullType> &rhs) noexcept
        : target_(rhs.target_), ptr_(reinterpret_cast<T *>(rhs.ptr_)),
          original_(reinterpret_cast<T *>(rhs.original_)) {
        static_assert(std::is_convertible<From *, T *>::value,
                      "Copy warning, intrusive_ptr got pointer of wrong type");
        static_assert(std::is_same_v<FromDeleteArrOp, DeleteArrOp> ||
                      (detail::intrusive_ptr_is_default_deleter<FromDeleteArrOp>::value
                       && detail::intrusive_ptr_is_default_deleter<DeleteArrOp>::value),
                      "Copy warning, intrusive_ptr got deleter of wrong type");
        retain_();
    }



    inline intrusive_ptr &operator=(intrusive_ptr &&rhs) & noexcept {
        return this->template operator= <T, DeleteArrOp, detail::intrusive_target_default_null_type<T[]>>(std::move(rhs));
    }

    template <typename From, typename FromDeleteArrOp, typename FromNullType>
    intrusive_ptr &
    operator=(intrusive_ptr<From[], FromDeleteArrOp, FromNullType> &&rhs) & noexcept {
        // std::cout << "move = operator called for
        // intrusive_ptr<others->T>"<<std::endl;
        static_assert(std::is_convertible<T *, From *>::value,
                      "Move warning, intrusive_ptr got pointer of wrong type");
        static_assert(std::is_same_v<FromDeleteArrOp, DeleteArrOp> ||
                      (detail::intrusive_ptr_is_default_deleter<FromDeleteArrOp>::value
                       && detail::intrusive_ptr_is_default_deleter<DeleteArrOp>::value),
                      "Move warning, intrusive_ptr got deleter of wrong type");

        intrusive_ptr tmp(std::move(rhs));
        swap(tmp);
        return *this;
    }
    
    inline intrusive_ptr &operator=(const intrusive_ptr &rhs) & noexcept {
        // std::cout << "copy = operator called for
        // intrusive_ptr<T>"<<std::endl;
        return this->template operator= <T, DeleteArrOp, detail::intrusive_target_default_null_type<T[]>>(rhs);
    }

    template <typename From, typename FromDeleteArrOp, typename FromNullType>
    intrusive_ptr &
    operator=(const intrusive_ptr<From[], FromDeleteArrOp, FromNullType> &rhs) & noexcept {
        // std::cout << "copy = operator called for
        // intrusive_ptr<others->T>"<<std::endl;
        static_assert(std::is_convertible<From *, T *>::value,
                      "Copy warning, intrusive_ptr got pointer of wrong type");
        static_assert(std::is_same_v<FromDeleteArrOp, DeleteArrOp> ||
                      (detail::intrusive_ptr_is_default_deleter<FromDeleteArrOp>::value
                       && detail::intrusive_ptr_is_default_deleter<DeleteArrOp>::value),
                      "Copy warning, intrusive_ptr got deleter of wrong type");
        intrusive_ptr tmp(rhs);
        swap(tmp);
        return *this;
    }

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
            return intrusive_ptr(ptr_ + i, original_, target_,
                                 detail::DontIncreaseRefCount{});
        } else {
            return intrusive_ptr(reinterpret_cast<uint8_t *>(ptr_) + i,
                                 original_, target_,
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
                target_ = MetaNew(TTarget, );
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
    }
    operator bool() const noexcept { return !both_null(); }

    static intrusive_ptr unsafe_make_from_raw_new(T *ptr) {
        return intrusive_ptr(ptr);
    }

    // static intrusive_ptr
    // make_aligned(const uint64_t amt,
    //              const std::size_t align_byte = _NT_ALIGN_BYTE_SIZE_) {
    //     uint64_t size = amt * sizeof(T);
    //     if (size % align_byte != 0)
    //         size += align_byte - (size % align_byte);
    //     /* utils::throw_exception((amt * sizeof(T)) % align_byte == 0, "Cannot
    //      * align $ bytes", amt * sizeof(T)); */
    //     return intrusive_ptr(
    //         static_cast<T *>(MetaAlignedAlloc(align_byte, size)),
    //         MetaNew(TTarget, ), &detail::defaultAlignedDeallocator<T>);
    // }

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
            MetaFree<std::atomic<int64_t>>(internal_refcount_);
            MetaFreeArr<T>(original);
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
        Parent *original = MetaNewArr(Child, i);
        std::atomic<int64_t> *counter = MetaNew(std::atomic<int64_t>, 1);
        counter->store(1, std::memory_order_relaxed);
        return intrusive_ptr<intrusive_parent_ptr_array<Parent>>::make(
            original, original, counter, sizeof(Child));
    }
};

}
