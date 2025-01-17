#ifndef _NT_UTILS_H_
#define _NT_UTILS_H_
#include <iostream>
#include <sstream>
#include <string_view>

#include <cstddef>
#include <string>
#include <stdexcept>
#include <complex>
#include "../types/Types.h"
#include <initializer_list>
#include <type_traits>
#include <vector>
#include <tuple>
#include <variant>
#include <sstream>
#include <fstream>
#include <cstring>
#include <utility>

#ifdef USE_PARALLEL
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/blocked_range3d.h>
#include <thread>
#include <sys/ioctl.h>
#endif
#include "../memory/DeviceEnum.h"
#include <cstdint>
#include <limits>

namespace nt{
template<class T>struct tag_t{};
template<class T>constexpr tag_t<T> tag{};

namespace utils{

/* template <typename T> */
/* struct IsInitializerList : std::false_type {}; */

/* template <typename T> */
/* struct IsInitializerList<std::initializer_list<T>> : std::true_type {}; */

/* template<typename T> */
/* constexpr bool IsInitializerList_t = IsInitializerList<T>::value; */


/* template <typename T> */
/* struct InitializerList_size{ */
/* 	using type = std::conditional_t<IsInitializerList_t<T>, T::value_type, T>; */
/* 	std::size_t value = !IsInitializerList_t<T> ? 0 : 1 + InitializerList_size<type>::value; */
/* }; */




template<typename T, std::size_t N>
struct NestedInitializerLists_type{
	using type = typename std::initializer_list<typename NestedInitializerLists_type<T, N-1>::type>;
};

template<typename T>
struct NestedInitializerLists_type<T, 1>{
	using type = typename std::initializer_list<T>;
};

// Base template for counting the number of arguments in the parameter pack
template<typename... Args>
struct VariadicArgumentCount {
    static const size_t value = sizeof...(Args);
};

// Recursive template specialization to count arguments
template<typename T, typename... Rest>
struct VariadicArgumentCount<T, Rest...> {
    static const size_t value = 1 + VariadicArgumentCount<Rest...>::value;
};




// Recursive case: when there are at least two template parameters
template<typename T, typename U, typename... Args>
struct SameType {
    static constexpr bool value = std::is_same_v<T, U> && SameType<T, Args...>::value;
};


template<typename T>
struct SameType<T, T>{
    static constexpr bool value = true;
};

template<typename T, std::size_t N, std::enable_if_t<N == 1, bool> = true>
inline void get_shape(const typename NestedInitializerLists_type<T, N>::type& v, std::vector<int64_t>& s){
	s.back() = v.size();
}

template<typename T, std::size_t N, std::enable_if_t<N != 1, bool> = true>
inline void get_shape(const typename NestedInitializerLists_type<T, N>::type& v, std::vector<int64_t>& s){
	s[s.size() - N] = v.size();
	get_shape<T, N-1>(*v.begin(), s);

}


template<typename T, std::size_t N, std::enable_if_t<N == 1, bool> = true>
inline bool validate_shape(const typename NestedInitializerLists_type<T, N>::type& v, std::vector<int64_t>& s){
	return s.back() == v.size();
}

template<typename T, std::size_t N, std::enable_if_t<N != 1, bool> = true>
inline bool validate_shape(const typename NestedInitializerLists_type<T, N>::type& v, std::vector<int64_t>& s){
	if(s[s.size() - N] != v.size())
		return false;
	for(auto begin = v.begin(); begin != v.end(); ++begin){
		if(!validate_shape<T, N-1>(*begin, s))
			return false;
	}
	return true;
}

 
inline bool endsWith(const char* str, const char* suffix) {
    size_t str_len = strlen(str);
    size_t suffix_len = strlen(suffix);

    // Check if the string is shorter than the suffix
    if (str_len < suffix_len) {
        return false;
    }

    // Compare the end of the string with the suffix
    return strncmp(str + str_len - suffix_len, suffix, suffix_len) == 0;
}

template<typename T>
inline void print_format_inner(std::ostringstream& out, std::string_view& str, size_t& last_pos, const T& var){
	size_t pos = str.substr(last_pos, str.size()).find('$');
	if(pos == std::string_view::npos){
		if(str.size() == last_pos){
			return;
		}
		out << str.substr(last_pos, str.size());
		last_pos = str.size();
		return;
	}
	out << str.substr(last_pos, pos);
	out << var;
	last_pos += pos + 1;
}

inline void print_format_inner(std::ostringstream& out, std::string_view& str, size_t& last_pos){
	size_t pos = str.substr(last_pos, str.size()).find('$');
	if(pos == std::string_view::npos){
		if(str.size() <= last_pos){
			return;
		}
		out << str.substr(last_pos, str.size());
		last_pos = str.size();
		return;
	}
	out << str.substr(last_pos, pos);
	last_pos += pos + 1;
}

template<typename... Args>
inline void throw_exception(const bool err, std::string_view str, const Args &...arg){
	if(err)
		return;
	std::ostringstream out;
	size_t last_pos = 0;
	(print_format_inner(out, str, last_pos, arg), ...);
	while(last_pos < str.size())
		print_format_inner(out, str, last_pos, "");
	throw std::invalid_argument(out.str());
}

template<typename... Args>
inline void throw_exception_2(const bool err, std::string_view str, const char* file, int line, const Args &...arg){
	if(err)
		return;
	std::ostringstream out;
	out << "Error at line " << line << " in file " << file << ": ";
	size_t last_pos = 0;
	(print_format_inner(out, str, last_pos, arg), ...);
	while(last_pos < str.size())
		print_format_inner(out, str, last_pos, "");
	throw std::invalid_argument(out.str());
}
#define THROW_EXCEPTION(err, message, ...) \
    throw_exception_2((err), (message), __FILE__, __LINE__, ##__VA_ARGS__)



template<typename T, std::size_t N>
inline std::vector<int64_t> aquire_shape(const typename NestedInitializerLists_type<T, N>::type& v){
	std::vector<int64_t> shape(N);
	get_shape<T, N>(v, shape);
	if(!validate_shape<T, N>(v, shape))
		throw std::invalid_argument("\nRuntime Error: Got inconsistent shape");
	return std::move(shape);
}

template<typename T, std::size_t N, typename UnaryFunction, std::enable_if_t<N == 1, bool> = true>
inline void flatten_func(const typename NestedInitializerLists_type<T, N>::type& v, UnaryFunction&& unary_op){
	for(auto begin = v.begin(); begin != v.end(); ++begin){
		unary_op(*begin);
	}
}

template<typename T, std::size_t N, typename UnaryFunction, std::enable_if_t<N != 1, bool> = true>
inline void flatten_func(const typename NestedInitializerLists_type<T, N>::type& v, UnaryFunction&& unary_op){
	for(auto begin = v.begin(); begin != v.end(); ++begin){
		flatten_func<T, N-1, UnaryFunction>(*begin, std::forward<UnaryFunction&&>(unary_op));
	}
}

void beginDualProgressBar(uint32_t total_a, uint32_t total_b, uint32_t width=50);
void printDualProgressBar(uint32_t progress_a, uint32_t total_a, uint32_t progress_b, uint32_t total_b, uint32_t width=50);
void endDualProgressBar(uint32_t total_a, uint32_t total_b, uint32_t width=50);
void printProgressBar(uint32_t progress, uint32_t total, std::string add = "", uint32_t width = 50);

#ifdef USE_PARALLEL
void printThreadingProgressBar(uint32_t progress, uint32_t total, std::string add = "", uint32_t width = 50);

bool isPipeReadable(int pipefd);
inline bool pid_still_running(pid_t pid){
	int status;
        pid_t result = waitpid(pid, &status, WNOHANG);
	if (result == -1) {
		return false;
	}
	else if (result == 0) {
		return true;
	}
	return false;
}

inline bool pids_still_running(const std::vector<pid_t>& pids){
	for(auto begin = pids.cbegin(); begin != pids.cend(); ++begin)
		if(pid_still_running(*begin)){return true;}
	return false;
}


inline ssize_t getPipeReadableBytes(int fd) {
     ssize_t bytesAvailable = 0;
    if (ioctl(fd, FIONREAD, &bytesAvailable) == -1) {
	    std::cout << "ioctl error"<<std::endl;
	    return -1; // Return -1 to indicate error
    }
    return bytesAvailable;
}

#ifdef __linux__
#include <unistd.h>
// Function to get the number of threads per core on Linux
inline unsigned int getThreadsPerCore() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    unsigned int threadsPerCore = 0;

    while (std::getline(cpuinfo, line)) {
        if (line.find("siblings") != std::string::npos) {
            std::istringstream iss(line);
            std::string key, value;
            iss >> key >> value;
            
            if (key == "siblings") {
                threadsPerCore = std::stoi(value);
                break;
            }
        }
    }

    return threadsPerCore;
}


inline void get_shared_memory_max(){return SHMMAX;}


#elif _WIN32
#include <windows.h>
// Function to get the number of threads per core on Windows
inline unsigned int getThreadsPerCore() {
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    
    return sysInfo.dwNumberOfProcessors / sysInfo.dwNumberOfProcessors;
}

inline uint64_t get_shared_memory_max() {
    // On Windows, the maximum shared memory size can be retrieved via the `GetSystemInfo` or `GlobalMemoryStatusEx` function.
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    GlobalMemoryStatusEx(&statex);

    // Return a portion of the available physical memory for shared memory use (e.g., 75% of available memory).
    // this would be the bottom number * 0.75, but just going to return all of it
    return static_cast<uint64_t>(statex.ullTotalPhys);
}

#elif __APPLE__
// Function to get the number of threads per core on macOS
#include <sys/sysctl.h>

inline unsigned int getThreadsPerCore() {
    int threadsPerCore = 0;
    size_t size = sizeof(threadsPerCore);

    sysctlbyname("hw.physicalcpu", &threadsPerCore, &size, NULL, 0);

    return threadsPerCore;
}

inline uint32_t get_shared_memory_max(){
	const char* command = "sysctl -n kern.sysv.shmmax";
	FILE* pipe = popen(command, "r");
	throw_exception(pipe, "Error executing command.");
	uint32_t shmmax;
	throw_exception(fscanf(pipe, "%u", &shmmax)  == 1, "Error reading command output.");
	pclose(pipe);
	return shmmax;
}

#else
#error "Unsupported platform"
#endif


inline int getNumCores() {
    int numCores = 0;

#ifdef __linux__
    numCores = sysconf(_SC_NPROCESSORS_ONLN);
#elif _WIN32
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    numCores = sysInfo.dwNumberOfProcessors;
#elif __APPLE__
    int mib[4];
    size_t len = sizeof(numCores);
    mib[0] = CTL_HW;
    mib[1] = HW_AVAILCPU;
    sysctl(mib, 2, &numCores, &len, nullptr, 0);
#endif

    return numCores;
}

inline tbb::blocked_range<int64_t> calculateGrainSize1D(int64_t dataSize){
	return tbb::blocked_range<int64_t>(0, dataSize, std::max<int64_t>(1, dataSize / (getThreadsPerCore() * 10))); // will adjust as needed
}

inline tbb::blocked_range<int64_t> calculateGrainSize1D(int64_t start, int64_t end){
	int64_t dataSize = end - start;
	return tbb::blocked_range<int64_t>(start, end, std::max<int64_t>(1, dataSize / (getThreadsPerCore() * 10))); // will adjust as needed
}

inline tbb::blocked_range2d<int64_t> calculateGrainSize2D(int64_t rows, int64_t cols){
	unsigned int threadsPerCore = getThreadsPerCore();
	int64_t grainSizeRows = std::max<int64_t>(1, rows / (threadsPerCore * 5));
	int64_t grainSizeCols = std::max<int64_t>(1, cols / (threadsPerCore * 5));
	//make sure the grain size isn't too big
	grainSizeRows = std::min<int64_t>(grainSizeRows, rows);
	grainSizeCols = std::min<int64_t>(grainSizeCols, cols);
	return tbb::blocked_range2d<int64_t>(0, rows, grainSizeRows, 0, cols, grainSizeCols);
}

inline tbb::blocked_range2d<int64_t> calculateGrainSize2D(int64_t rows_start, int64_t rows_end, int64_t cols_start, int64_t cols_end){
	int64_t cols = cols_end - cols_start;
	int64_t rows = rows_end - rows_start;
	unsigned int threadsPerCore = getThreadsPerCore();
	int64_t grainSizeRows = std::max<int64_t>(1, rows / (threadsPerCore * 5));
	int64_t grainSizeCols = std::max<int64_t>(1, cols / (threadsPerCore * 5));
	//make sure the grain size isn't too big
	grainSizeRows = std::min<int64_t>(grainSizeRows, rows);
	grainSizeCols = std::min<int64_t>(grainSizeCols, cols);
	return tbb::blocked_range2d<int64_t>(rows_start, rows_end, grainSizeRows, cols_start, cols_end, grainSizeCols);
}

inline tbb::blocked_range3d<int64_t> calculateGrainSize3D(int64_t pages, int64_t rows, int64_t cols){
	unsigned int threadsPerCore = getThreadsPerCore();
	/* std::cout << "threadsPerCore: "<<threadsPerCore<< "total concurrency available: "<< std::thread::hardware_concurrency()<<std::endl; */
	/* std::cout << "number of cores: "<<getNumCores() <<std::endl; */
	int64_t grainSizeRows = std::max<int64_t>(1, rows / (threadsPerCore));
	int64_t grainSizeCols = std::max<int64_t>(1, cols / (threadsPerCore));
	int64_t grainSizePages = std::max<int64_t>(1, pages / (threadsPerCore));
	//make sure the grain size isn't too big
	grainSizeRows = std::min<int64_t>(grainSizeRows, rows);
	grainSizeCols = std::min<int64_t>(grainSizeCols, cols);
	grainSizePages = std::min<int64_t>(grainSizePages, pages);
	/* std::cout << "grain sizes: "<< grainSizePages<<"," << grainSizeRows<<", "<<grainSizeCols << "("<<pages<<","<<rows<<","<<cols<<")"<<std::endl; */
	return tbb::blocked_range3d<int64_t>(0, pages, grainSizePages, 0, rows, grainSizeRows, 0, cols, grainSizeCols);
}

inline tbb::blocked_range3d<int64_t> calculateGrainSize3D(int64_t pages_start, int64_t pages_end, int64_t rows_start, int64_t rows_end, int64_t cols_start, int64_t cols_end){
	int64_t cols = cols_end - cols_start;
	int64_t rows = rows_end - rows_start;
	int64_t pages = pages_end - pages_start;
	unsigned int threadsPerCore = getThreadsPerCore();
	int64_t grainSizeRows = std::max<int64_t>(1, rows / (threadsPerCore));
	int64_t grainSizeCols = std::max<int64_t>(1, cols / (threadsPerCore));
	int64_t grainSizePages = std::max<int64_t>(1, pages / (threadsPerCore));
	//make sure the grain size isn't too big
	grainSizeRows = std::min<int64_t>(grainSizeRows, rows);
	grainSizeCols = std::min<int64_t>(grainSizeCols, cols);
	grainSizePages = std::min<int64_t>(grainSizePages, pages);
	/* std::cout << "grain sizes: "<< grainSizePages<<"," << grainSizeRows<<", "<<grainSizeCols << "("<<pages<<","<<rows<<","<<cols<<")"<<std::endl; */
	return tbb::blocked_range3d<int64_t>(pages_start, pages_end, grainSizePages, rows_start, rows_end, grainSizeRows, cols_start, cols_end, grainSizeCols);
}


#endif


class my_tuple{
	int64_t first, second;
	public:
		inline my_tuple(const int64_t a, const int64_t b)
			:first(a), second(b)
		{}

		inline my_tuple(const int64_t a)
			:first(a), second(a)
		{}

		inline my_tuple(std::tuple<int64_t, int64_t> t)
			:first(std::get<0>(t)), second(std::get<1>(t))
		{}

		inline my_tuple& operator=(int64_t x){first = x; second = x; return *this;}
		inline my_tuple& operator=(std::tuple<int64_t, int64_t> x){first = std::get<0>(x); second = std::get<1>(x); return *this;}
		inline const int64_t& operator[](const int64_t x) const {if(x == 0){return first;}return second;}
		inline bool operator==(const int64_t x) const {return (first == x && second == x);}
		inline bool operator!=(const int64_t x) const {return (first != x && second != x);}
		inline bool operator > (const int64_t x) const {return (first > x && second > x);}
		inline bool operator < (const int64_t x) const {return (first < x && second < x);}
		inline bool operator >= (const int64_t x) const {return (first >= x && second >= x);}
		inline bool operator <= (const int64_t x) const {return (first <= x && second <= x);}
};

inline std::ostream& operator<<(std::ostream& out, const my_tuple& t){
	return out << '(' << t[0] << ',' << t[1] << ')';
}




// Template meta-programming to generate a tuple type containing a specified number of elements of a given type
template<typename T, size_t N, typename... REST>
struct generate_tuple_type_n {
    typedef typename generate_tuple_type_n<T, N-1, T, REST...>::type type;
};

template<typename T, typename... REST>
struct generate_tuple_type_n<T, 0, REST...> {
    typedef std::tuple<REST...> type;
};

template<size_t N>
class my_n_tuple{
	std::array<int64_t, N> arr;

	inline void set_array(int64_t index){}
	inline void set_array(int64_t index, const int64_t& a){
		arr[index] = a;
	}

	template<typename... Args>
	inline void set_array(int64_t index, const int64_t& a, const Args&... args){
		arr[index] = a;
		set_array(index+1, args...);
	}

	template<typename T, T... ints>
	inline void set_array_tuple(const generate_tuple_type_n<int64_t, N>& t, std::integer_sequence<T, ints...> int_seq){
		((arr[ints] = std::get<ints>(t)), ...);
	}
	public:
		template<typename... Args>
		inline my_n_tuple(const int64_t a, const int64_t b, const Args... args)
		{
			static_assert(N > 2 ? sizeof...(Args) + 2 == N : sizeof...(Args) == 0, "Expected to get N arguments for tuple");
			static_assert(N >= 2, "Expected to get N arguments for tuple");
			set_array(0, a, b, args...);
		}

		inline my_n_tuple(const int64_t a)
		{
			for(size_t i = 0; i < N; ++i)
				arr[i] = a;
		}

		inline my_n_tuple(generate_tuple_type_n<int64_t, N> t){
			set_array_tuple(t, std::make_index_sequence<N>{});
		}

		inline my_n_tuple& operator=(int64_t x){
			for(size_t i = 0; i < N; ++i)
				arr[i] = x;
			return *this;
		}
		inline my_tuple& operator=(generate_tuple_type_n<int64_t, N> t){
			set_array_tuple(t, std::make_index_sequence<N>{});
		}
		inline const int64_t& operator[](const int64_t x) const {return arr[x];}
		inline bool operator==(const int64_t x) const {
			for(size_t i = 0; i < N; ++i){
				if(arr[i] != x){return false;}}
			return true;
		}
		inline bool operator!=(const int64_t x) const {
			for(size_t i = 0; i < N; ++i){
				if(arr[i] == x){return false;}}
			return true;
		}
		inline bool operator > (const int64_t x) const {
			for(size_t i = 0; i < N; ++i){
				if(arr[i] <= x){return false;}}
			return true;
		}
		inline bool operator < (const int64_t x) const {
			for(size_t i = 0; i < N; ++i){if(arr[i] >= x){return false;}}
			return true;
		}
		inline bool operator >= (const int64_t x) const {
			for(size_t i = 0; i < N; ++i){if(arr[i] < x){return false;}}
			return true;
		}
		inline bool operator <= (const int64_t x) const {
			for(size_t i = 0; i < N; ++i){if(arr[i] > x){return false;}}
			return true;
		}
};

template<std::size_t N>
inline std::ostream& operator<<(std::ostream& out, const my_n_tuple<N>& t){
	out << '(';
	for(uint32_t i = 0; i < N-1; ++i)
		out << t[i] << ',';
	return out << t[N-1] << ')';
}

class tuple_or_int{
	std::variant<int64_t, std::tuple<int64_t, int64_t>> val;
	public:
		inline tuple_or_int(int64_t x)
			:val(x)
		{}

		inline tuple_or_int(std::tuple<int64_t, int64_t> x)
			:val(x)
		{}

		inline tuple_or_int& operator=(int64_t x){val = x; return *this;}
		inline tuple_or_int& operator=(std::tuple<int64_t, int64_t> x){val = x; return *this;}
		inline std::tuple<int64_t, int64_t> to_tuple() const{
			if(std::holds_alternative<std::tuple<int64_t, int64_t>>(val)){
				return std::get<1>(val);
			}
			return std::tuple<int64_t, int64_t>(std::get<0>(val), std::get<0>(val));
		}
		inline operator std::tuple<int64_t, int64_t>() const {return to_tuple();}
		inline my_tuple to_my_tuple() const {
			if(std::holds_alternative<std::tuple<int64_t, int64_t>>(val)){
				return my_tuple(std::get<1>(val));
			}
			return my_tuple(std::get<0>(val), std::get<0>(val));
	
		}
};


namespace memory_details{
extern int64_t cpu_memory_allocated;
extern int64_t shared_cpu_memory_allocated;
extern int64_t meta_memory_allocated;

}

inline int64_t& AllocatedMemory(DeviceType dt){
	switch(dt){
		case DeviceType::META:
			return memory_details::meta_memory_allocated;
		case DeviceType::CPU:
			return memory_details::cpu_memory_allocated;
		case DeviceType::CPUShared:
			return memory_details::shared_cpu_memory_allocated;
		default:
			return memory_details::meta_memory_allocated;
	}
}

inline int64_t MaxMemory(DeviceType dt){
	switch(dt){
		case DeviceType::META:
			return -1;
		case DeviceType::CPU:
			return std::numeric_limits<int64_t>::max();
		case DeviceType::CPUShared:
			return get_shared_memory_max();

	}
}

inline void CheckAllocation(DeviceType dt, int64_t bytes){
	int64_t& curMem = AllocatedMemory(dt);
	int64_t maxMem = MaxMemory(dt);
	throw_exception((maxMem - curMem - bytes) >= 0,
			"Trying to allocate $ bytes of memory on $, but already allocated $ and there is a max of $ bytes to allocate", bytes, dt, curMem, maxMem);
	curMem += bytes;
}

inline void DeallocateMemory(DeviceType dt, int64_t bytes){
	AllocatedMemory(dt) -= bytes;
}

//struct to reverse an index_sequence
template <typename Seq, std::size_t N>
struct reverse_index_sequence_impl;

template <std::size_t... Is, std::size_t N>
struct reverse_index_sequence_impl<std::index_sequence<Is...>, N> {
    using type = std::index_sequence<(N - 1 - Is)...>;
};

template <std::size_t N>
using reverse_index_sequence = typename reverse_index_sequence_impl<std::make_index_sequence<N>, N>::type;


template<typename T, typename Arg>
inline constexpr decltype(auto) add_cv_like(Arg& arg) noexcept {
    if constexpr (std::is_const<T>::value && std::is_volatile<T>::value) {
        return const_cast<const volatile Arg&>(arg);
    }
    else if constexpr (std::is_const<T>::value) {
        return const_cast<const Arg&>(arg);
    }
    else if constexpr (std::is_volatile<T>::value) {
        return const_cast<volatile Arg&>(arg);
    }
    else {
        return const_cast<Arg&>(arg);
    }
}

template <typename C, typename T, typename R, typename... Types>
inline constexpr bool is_class_function_return_type(R(T::*f)(Types...)){return std::is_same_v<C, R>;}


//repeat types
//std::tuple<repeat_types_t<int, 4> > is the same as std::tuple<int, int, int, int>

template <typename T, std::size_t N, typename... Ts>
struct repeat_types_helper {
    using type = typename repeat_types_helper<T, N - 1, T, Ts...>::type;
};

// Base case: N == 0
template <typename T, typename... Ts>
struct repeat_types_helper<T, 0, Ts...> {
    using type = std::tuple<Ts...>;
};

// Alias for easier use
template <typename T, std::size_t N>
using repeat_types_t = typename repeat_types_helper<T, N>::type;

//primary template for variadic template check
template <typename T, typename Arg1, typename... Args>
struct is_all_same {
    static constexpr bool value = std::is_same_v<T, Arg1> && is_all_same<T, Args...>::value;
};

//specialization for the base case
template <typename T>
struct is_all_same<T, T> : std::true_type {};

template<typename T, typename... Args>
inline static constexpr bool is_all_same_v = is_all_same<T, Args...>::value;

}} //nt::utils::


namespace nt{
namespace detect_number {
	template<class T, class...Ts>
	constexpr bool is_charlike(tag_t<T>, Ts&&...){ return false; }
	constexpr bool is_charlike( tag_t<double> ){ return true; }
	constexpr bool is_charlike( tag_t<float> ){ return true; }
	constexpr bool is_charlike( tag_t<int32_t> ){ return true; }
	constexpr bool is_charlike( tag_t<uint32_t> ){ return true;}
	constexpr bool is_charlike( tag_t<int64_t> ){return true;}
	constexpr bool is_charlike( tag_t<int16_t> ){ return true; }
	constexpr bool is_charlike( tag_t<uint16_t> ){ return true; }
	constexpr bool is_charlike( tag_t<int8_t> ){ return true; }
	constexpr bool is_charlike( tag_t<uint8_t> ){ return true; }
	template<class T>
	constexpr bool detect=is_charlike(tag<T>);
};

namespace detect_number_pc {
	template<class T, class...Ts>
	constexpr bool is_charlike(tag_t<T>, Ts&&...){ return false; }
	constexpr bool is_charlike( tag_t<double> ){ return true; }
	constexpr bool is_charlike( tag_t<float> ){ return true; }
	constexpr bool is_charlike( tag_t<complex_128 >){ return true; }
	constexpr bool is_charlike( tag_t<complex_64 >){ return true; }
	constexpr bool is_charlike( tag_t<int32_t> ){ return true; }
	constexpr bool is_charlike( tag_t<uint32_t> ){ return true;}
	constexpr bool is_charlike( tag_t<int64_t> ){return true;}
	constexpr bool is_charlike( tag_t<int16_t> ){ return true; }
	constexpr bool is_charlike( tag_t<uint16_t> ){ return true; }
	constexpr bool is_charlike( tag_t<int8_t> ){ return true; }
	constexpr bool is_charlike( tag_t<uint8_t> ){ return true; }
	template<class T>
	constexpr bool detect=is_charlike(tag<T>);
};





}

#endif // _NT_UTILS_H_
