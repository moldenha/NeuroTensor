#include <memory>
#include <immintrin.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <cstdlib> // For std::aligned_alloc
#include "../utils/utils.h"

#ifdef __AVX512F__
#define ALIGN_BYTE_SIZE 64
#elif defined(__AVX2__)
#define ALIGN_BYTE_SIZE 32
#elif defined(__AVX__)
#define ALIGN_BYTE_SIZE 32
#else
#define ALIGN_BYTE_SIZE 16
#endif

namespace nt{

namespace detail{
	enum class Device{
		SharedCPU,
		CPU
	}; // to come will be cuda, and mlx
}

class intrusive_void;
class intrusive_bucket;
class intrusive_voids;

class intrusive_void{
	std::shared_ptr<void> ptr_;
	void* n_ptr;
	detail::Device device_;
	intrusive_void(std::shared_ptr<void> ptr, void* nptr)
		:ptr_(ptr),
		n_ptr(nptr)
	{}
	public:
		explicit intrusive_void(std::shared_ptr<void> ptr, detail::Device device = detail::Device::CPU)
			:ptr_(std::move(ptr)),
			n_ptr(nullptr),
			device_(device)
		{n_ptr = ptr_.get();}

		explicit intrusive_void()
			:ptr_(nullptr),
			n_ptr(nullptr),
			device_(detail::Device::CPU)
		{}

		intrusive_void(const intrusive_void& other)
			:ptr_(other.ptr_),
			n_ptr(other.n_ptr),
			device_(other.device_)
		{}
		intrusive_void(intrusive_void&& other)
			:ptr_(std::move(other.ptr_)),
			n_ptr(other.n_ptr),
			device_(other.device_)
		{}
		intrusive_void& operator=(const intrusive_void& other){
			n_ptr = nullptr;
			ptr_ = other.ptr_;
			n_ptr = other.n_ptr;
			device_ = other.device_;
			return *this;
		}
		intrusive_void& operator=(intrusive_void&& other){
			reset();
			ptr_ = std::move(other.ptr_);
			n_ptr = other.n_ptr;
			device_ = other.device_;
			return *this;
		}
		/*
		 * td::shared_ptr<void> ptr(
        new float[10], // Pointer to the dynamically allocated array
        [](void* p) { delete[] static_cast<float*>(p); } // Custom deleter to delete the array
    );*/
		inline void* get() noexcept {return n_ptr;}
		inline const void* get() const noexcept {return n_ptr;}
		inline void* operator->() {return n_ptr;}
		inline void reset() {n_ptr = nullptr; ptr_.reset();}
		~intrusive_void() {reset();}
		inline intrusive_void operator+(int64_t i) const {
			return intrusive_void(ptr_, reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(n_ptr) + i));
		}
		inline long use_count() const noexcept {
			return ptr_.use_count();
		}
		inline void swap( intrusive_void& r ) noexcept {
			std::swap(r.n_ptr, n_ptr);
			std::swap(r.ptr_, ptr_);
		}
		inline const detail::Device& device() const {return device_;}
		static intrusive_void make_aligned(const std::size_t amt, std::size_t type_size, const std::size_t align_byte = ALIGN_BYTE_SIZE){
			utils::throw_exception((amt * type_size) % align_byte == 0, "Cannot align $ bytes", amt * type_size);
			void* ptr = std::aligned_alloc(align_byte, amt * type_size);
			utils::throw_exception(ptr, "Was unable to make alligned memory");
			return intrusive_void(std::shared_ptr<void>(ptr, [](void* ptr){std::free(ptr);}));
		}
		inline const bool is_shared() const noexcept {return device_ == detail::Device::SharedCPU;}
		inline const bool is_cpu() const noexcept {return device_ == detail::Device::CPU;}
#ifdef USE_PARALLEL
		static intrusive_void make_shared(const uint64_t amt, const std::size_t byte_size, key_t key = IPC_PRIVATE){
			const uint64_t n_size = amt * byte_size;
			utils::throw_exception(utils::get_shared_memory_max() >= n_size, "Expected to allocate at most $ bytes of shared memory, but was asked to allocate $ bytes of shared memory void*", utils::get_shared_memory_max(), n_size);
			int shmid = shmget(key, n_size, IPC_CREAT | 0666);
			utils::throw_exception(shmid != -1, "Making segment ID failed for shared memory (shmget)");
			void* sharedArray = shmat(shmid, nullptr, 0);
			utils::throw_exception(sharedArray != (void*)-1, "Making shared memory failed (shmat)");
			return intrusive_void(std::shared_ptr(sharedArray,
					[shmid](void* ptr){
						shmdt(ptr);
						shmctl(shmid, IPC_RMID, nullptr);
					}),
					detail::Device::SharedCPU);
		}
#endif
		static intrusive_void to_cpu(const intrusive_void& ptr, const uint64_t amt, const std::size_t byte_size){
			if(ptr.is_cpu())
				return ptr;
			intrusive_void outp(std::shared_ptr<void>(std::malloc(amt * byte_size), [](void* ptr){std::free(ptr);}), detail::Device::CPU);
			const uint64_t n_size = amt * byte_size;
			const uint64_t end = n_size / sizeof(uint64_t);
			const uint64_t r = (n_size % sizeof(uint64_t)) / sizeof(uint8_t);
			uint64_t* optr = reinterpret_cast<uint64_t*>(outp.get());
			const uint64_t* iptr = reinterpret_cast<const uint64_t*>(ptr.get());
			const uint64_t* iptr_end = iptr + end;
			for(;iptr != iptr_end; ++iptr, ++optr)
				*optr = *iptr;
			if(r != 0){
				const uint8_t* iptr_a = reinterpret_cast<const uint8_t*>(iptr_end);
				const uint8_t* iptra_end = iptr_a + r;
				uint8_t* optr_a = reinterpret_cast<uint8_t*>(optr);
				for(;iptr != iptr_end; ++iptr, ++optr)
					*optr = *iptr;
			}
			return outp;
		}
#ifdef USE_PARALLEL
		static intrusive_void to_shared(const intrusive_void& ptr, const uint64_t amt, const std::size_t byte_size, key_t key = IPC_PRIVATE){
			if(ptr.is_shared())
				return ptr;
			intrusive_void outp = intrusive_ptr::make_shared(amt, byte_size, key);
			const uint64_t n_size = amt * byte_size;
			const uint64_t end = n_size / sizeof(uint64_t);
			const uint64_t r = (n_size % sizeof(uint64_t)) / sizeof(uint8_t);
			uint64_t* optr = reinterpret_cast<uint64_t*>(outp.get());
			const uint64_t* iptr = reinterpret_cast<const uint64_t*>(ptr.get());
			const uint64_t* iptr_end = iptr + end;
			for(;iptr != iptr_end; ++iptr, ++optr)
				*optr = *iptr;
			if(r != 0){
				const uint8_t* iptr_a = reinterpret_cast<const uint8_t*>(iptr_end);
				const uint8_t* iptra_end = iptr_a + r;
				uint8_t* optr_a = reinterpret_cast<uint8_t*>(optr);
				for(;iptr != iptr_end; ++iptr, ++optr)
					*optr = *iptr;
			}
			return outp;
		}
#endif

};

class intrusive_bucket {

	    std::shared_ptr<intrusive_void[]> ptr_;
	    intrusive_void* n_ptr;

	    explicit intrusive_bucket(std::shared_ptr<intrusive_void[]> ptr, intrusive_void* nptr)
		: ptr_(ptr),
		n_ptr(nptr)
	    {}
	public:
		explicit intrusive_bucket(std::shared_ptr<intrusive_void[]> ptr)
			:ptr_(std::move(ptr)),
			n_ptr(nullptr)
		{n_ptr = ptr_.get();}
		
		explicit intrusive_bucket(int64_t i)
			:ptr_(new intrusive_void[i]),
			n_ptr(nullptr)
		{n_ptr = ptr_.get();}

		explicit intrusive_bucket()
			:ptr_(nullptr),
			n_ptr(nullptr)
		{}

		intrusive_bucket(const intrusive_bucket& other)
			:ptr_(other.ptr_),
			n_ptr(other.n_ptr)
		{}
		intrusive_bucket(intrusive_bucket&& other)
			:ptr_(std::move(other.ptr_)),
			n_ptr(other.n_ptr)
		{}
		intrusive_bucket& operator=(const intrusive_bucket& other){
			n_ptr = nullptr;
			ptr_ = other.ptr_;
			n_ptr = other.n_ptr;
			return *this;
		}
		intrusive_bucket& operator=(intrusive_bucket&& other){
			reset();
			ptr_ = std::move(other.ptr_);
			n_ptr = other.n_ptr;
			return *this;
		}
		/*
		 * td::shared_ptr<void> ptr(
        new float[10], // Pointer to the dynamically allocated array
        [](void* p) { delete[] static_cast<float*>(p); } // Custom deleter to delete the array
    );*/
		inline intrusive_void* get() noexcept {return n_ptr;}
		inline const intrusive_void* get() const noexcept {return n_ptr;}
		inline intrusive_void* operator->() const noexcept {return n_ptr;}
		inline intrusive_void& operator[](std::ptrdiff_t idx){
			return n_ptr[idx];
		}
		inline const intrusive_void& operator[](std::ptrdiff_t idx) const {
			return n_ptr[idx];
		}
		inline void reset() {n_ptr = nullptr; ptr_.reset();}
		~intrusive_bucket() {reset();}
		inline intrusive_bucket operator+(int64_t i) const {
			return intrusive_bucket(ptr_, n_ptr + i);
		}
		inline long use_count() const noexcept {
			return ptr_.use_count();
		}
		inline void swap( intrusive_bucket& r ) noexcept {
			std::swap(r.n_ptr, n_ptr);
			std::swap(r.ptr_, ptr_);
		}

};


class intrusive_voids {

    std::shared_ptr<void*> ptr_;
    void** n_ptr;

    explicit intrusive_voids(std::shared_ptr<void*> ptr, void** nptr)
        : ptr_(ptr),
        n_ptr(nptr)
    {}
public:
    explicit intrusive_voids(std::shared_ptr<void*> ptr)
        : ptr_(std::move(ptr)),
        n_ptr(nullptr)
    { n_ptr = ptr_.get(); }

    explicit intrusive_voids(int64_t i)
        : ptr_(new void*[i]),
        n_ptr(nullptr)
    { n_ptr = ptr_.get(); }

    explicit intrusive_voids()
        : ptr_(nullptr),
        n_ptr(nullptr)
    {}

    intrusive_voids(const intrusive_voids& other)
        : ptr_(other.ptr_),
        n_ptr(other.n_ptr)
    {}
    intrusive_voids(intrusive_voids&& other)
        : ptr_(std::move(other.ptr_)),
        n_ptr(other.n_ptr)
    {}
    intrusive_voids& operator=(const intrusive_voids& other){
        n_ptr = nullptr;
        ptr_ = other.ptr_;
        n_ptr = other.n_ptr;
        return *this;
    }
    intrusive_voids& operator=(intrusive_voids&& other){
        reset();
        ptr_ = std::move(other.ptr_);
        n_ptr = other.n_ptr;
        return *this;
    }

    inline void** get() const noexcept { return n_ptr; }
    inline void** operator->() const noexcept { return n_ptr; }
    inline void*& operator[](std::ptrdiff_t idx){
        return n_ptr[idx];
    }
    inline const void* operator[](std::ptrdiff_t idx) const {
        return n_ptr[idx];
    }
    inline void reset() { n_ptr = nullptr; ptr_.reset(); }
    ~intrusive_voids() { reset(); }
    inline intrusive_voids operator+(int64_t i) const {
        return intrusive_voids(ptr_, n_ptr + i);
    }
    inline long use_count() const noexcept {
        return ptr_.use_count();
    }
    inline void swap(intrusive_voids& r) noexcept {
        std::swap(r.n_ptr, n_ptr);
        std::swap(r.ptr_, ptr_);
    }

};


}

namespace std{
	void swap(::nt::intrusive_void& r, ::nt::intrusive_void& l) noexcept{
		r.swap(l);
	}
	void swap(::nt::intrusive_bucket& r, ::nt::intrusive_bucket& l) noexcept {
		r.swap(l);
	}
	void swap(::nt::intrusive_voids& r, ::nt::intrusive_voids& l) noexcept {
		r.swap(l);
	}
}
