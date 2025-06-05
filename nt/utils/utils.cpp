#include "utils.h"

#include <cstddef>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <string>
#include <string_view>

#ifdef USE_PARALLEL
#include <tbb/mutex.h>
#if defined(_WIN32)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #include <cstdint>
  #include <io.h>  // for _get_osfhandle
  // #ifndef pid_t
  //   using pid_t = DWORD;
  // #endif
#elif defined(__APPLE__) || defined(__linux__) || defined(__unix__)
  #include <sys/types.h>
  #include <signal.h>
  #include <errno.h>
  #include <sys/ioctl.h>
  #include <unistd.h>

#endif

#endif


/* #include "../refs/SizeRef.h" */
/* #include "../dtype/DType_enum.h" */
/* #include "../dtype/DType.h" */
namespace nt{
namespace utils{


/* template<typename T> */
/* void print_format_inner(std::ostringstream& out, std::string_view& str, size_t& last_pos, const T& var){ */
/* 	size_t pos = str.substr(last_pos, str.size()).find('$'); */
/* 	if(pos == std::string_view::npos){ */
/* 		if(str.size() == last_pos) */
/* 			return; */
/* 		out << str.substr(last_pos, str.size()); */
/* 		last_pos = str.size(); */
/* 		return; */
/* 	} */
/* 	out << str.substr(last_pos, pos); */
/* 	out << var; */
/* 	last_pos += pos + 1; */
/* } */

/* template<> void print_format_inner<int>(std::ostringstream&, std::string_view&, size_t&, const int&); */
/* template<> void print_format_inner<float>(std::ostringstream&, std::string_view&, size_t&, const float&); */
/* template<> void print_format_inner<double>(std::ostringstream&, std::string_view&, size_t&, const double&); */
/* template<> void print_format_inner<uint32_t>(std::ostringstream&, std::string_view&, size_t&, const uint32_t&); */
/* template<> void print_format_inner<SizeRef>(std::ostringstream&, std::string_view&, size_t&, const SizeRef&); */
/* template<> void print_format_inner<DType>(std::ostringstream&, std::string_view&, size_t&, const DType&); */


namespace memory_details{
int64_t cpu_memory_allocated = 0;
int64_t shared_cpu_memory_allocated = 0;
int64_t meta_memory_allocated = 0;
}

#ifdef USE_PARALLEL

#if defined(_WIN32)

// On Windows, we'll treat the PID as a DWORD.
// If you spawn a process with CreateProcess, you get its PID as a DWORD.
// This inline function returns true if the process is still running.
bool pid_still_running(DWORD pid) {
  // Try to open the process with minimal rights (QUERY_INFORMATION is enough
  // to call GetExitCodeProcess). If OpenProcess fails, assume the process
  // does not exist (or we have no permission → treat as “not running”).
  HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
  if (h == nullptr) {
      // ERROR_INVALID_PARAMETER or ERROR_ACCESS_DENIED both imply "not running"
      return false;
  }

  DWORD exitCode = 0;
  if (!GetExitCodeProcess(h, &exitCode)) {
      // Unable to query exit code—assume it's not running
      CloseHandle(h);
      return false;
  }

  CloseHandle(h);
  // STILL_ACTIVE (259) means the process has not exited yet.
  return (exitCode == STILL_ACTIVE);
}

ssize_t getPipeReadableBytes(int fd) {
    HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (h == INVALID_HANDLE_VALUE) {
        std::cerr << "Invalid file descriptor\n";
        return -1;
    }

    DWORD bytesAvailable = 0;
    BOOL success = PeekNamedPipe(h, nullptr, 0, nullptr, &bytesAvailable, nullptr);
    if (!success) {
        std::cerr << "PeekNamedPipe failed\n";
        return -1;
    }

    return static_cast<ssize_t>(bytesAvailable);
}

#elif (defined(__APPLE__) || defined(__linux__) || defined(__unix__)) && !defined(_WIN32)

// On POSIX (Linux/macOS), pid_t is the native process‐ID type.
// Sending signal 0 does not actually send a signal; it merely performs error
// checking: if kill(pid, 0) returns 0 (or errno==EPERM), the process exists.
bool pid_still_running(pid_t pid) {
  if (pid <= 0) {
      return false;
  }
  // kill(pid, 0) → 
  //   0: pid exists & we can signal it
  //  -1 with errno==EPERM: pid exists but we lack permissions
  //  -1 with errno==ESRCH: pid does not exist
  int err = kill(pid, 0);
  if (err == 0) {
      return true;
  }
  if (err == -1 && errno == EPERM) {
      return true;
  }
  return false;
}

ssize_t getPipeReadableBytes(int fd) {
     ssize_t bytesAvailable = 0;
    if (ioctl(fd, FIONREAD, &bytesAvailable) == -1) {
	    std::cout << "ioctl error"<<std::endl;
	    return -1; // Return -1 to indicate error
    }
    return bytesAvailable;
}
#elif defined(_WIN32)

ssize_t getPipeReadableBytes(int fd) {
    HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (h == INVALID_HANDLE_VALUE) {
        std::cerr << "Invalid file descriptor\n";
        return -1;
    }

    DWORD bytesAvailable = 0;
    BOOL success = PeekNamedPipe(
        h,                // handle to the pipe
        nullptr,          // buffer (not reading actual data)
        0,                // buffer size
        nullptr,          // bytes read
        &bytesAvailable,  // bytes available
        nullptr           // bytes left this message (not needed)
    );

    if (!success) {
        std::cerr << "PeekNamedPipe failed with error code: " << GetLastError() << "\n";
        return -1;
    }

    return static_cast<ssize_t>(bytesAvailable);
}


#else
  #error "Unsupported platform for pid_still_running(...)"
#endif


#ifdef _WIN32


bool isPipeReadable(int pipefd){
    return getPipeReadableBytes(pipefd) > 0; 
}

#else
bool isPipeReadable(int pipefd) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(pipefd, &rfds);

    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 0;

    // Use select to check if the file descriptor is ready for reading
    int ready = select(pipefd + 1, &rfds, nullptr, nullptr, &timeout);
    if (ready == -1) {
        // Error occurred
        perror("select");
        return false;
    } else if (ready == 0) {
        // No data to read
        return false;
    } else {
        // Data is available to read
        return true;
    }
}

#endif


void printThreadingProgressBar(uint32_t progress, uint32_t total, std::string add, uint32_t width) {
    float percentage = static_cast<float>(progress) / total;
    int numChars = static_cast<int>(percentage * width);

    static tbb::mutex printMutex;
    tbb::mutex::scoped_lock lock(printMutex);
    std::cout << "\r[";
    for (int i = 0; i < numChars; ++i) {
        std::cout << "=";
    }
    for (int i = numChars; i < width; ++i) {
        std::cout << " ";
    }
    std::cout << "] " << static_cast<int>(percentage * 100.0) << "% "<<progress<<'/'<<total << add;
    std::cout.flush();
    lock.release();
}



#endif


void printProgressBar(uint32_t progress, uint32_t total, std::string add, uint32_t width) {
    float percentage = static_cast<float>(progress) / total;
    int numChars = static_cast<int>(percentage * width);

    std::cout << "\r[";
    for (int i = 0; i < numChars; ++i) {
        std::cout << "=";
    }
    for (int i = numChars; i < width; ++i) {
        std::cout << " ";
    }
    std::cout << "] " << static_cast<int>(percentage * 100.0) << "% "<<progress<<'/'<<total << add;
    std::cout.flush();
}

void beginDualProgressBar(uint32_t total_a, uint32_t total_b, uint32_t width){
	std::cout << "[";
	for(int i = 0; i < width; ++i)
		std::cout << ' ';
	std::cout << "] 0% 0/"<<total_a << "\n";
	std::cout << "[";
	for(int i = 0; i < width; ++i)
		std::cout << ' ';
	std::cout << "] 0% 0/"<<total_b;
	std::cout.flush();
}
void printDualProgressBar(uint32_t progress_a, uint32_t total_a, uint32_t progress_b, uint32_t total_b, uint32_t width){
	std::cout << "\033[F";
	printProgressBar(progress_a, total_a, "", width);
	std::cout << "\n";
	printProgressBar(progress_b, total_b, "", width);

}
void endDualProgressBar(uint32_t total_a, uint32_t total_b, uint32_t width){
	std::cout << "\033[F";
	printProgressBar(total_a, total_a, "", width);
	std::cout << "\n";
	printProgressBar(total_b, total_b, "", width);
	
}

tuple_or_int::tuple_or_int(int64_t x)
    :val(x)
{}


tuple_or_int::tuple_or_int(std::tuple<int64_t, int64_t> x)
    :val(x)
{}

tuple_or_int& tuple_or_int::operator=(int64_t x){val = x; return *this;}
tuple_or_int& tuple_or_int::operator=(std::tuple<int64_t, int64_t> x){val = x; return *this;}

std::tuple<int64_t, int64_t> tuple_or_int::to_tuple() const{
    if(std::holds_alternative<std::tuple<int64_t, int64_t>>(val)){
        return std::get<1>(val);
    }
    return std::tuple<int64_t, int64_t>(std::get<0>(val), std::get<0>(val));
}

tuple_or_int::operator std::tuple<int64_t, int64_t>() const {return this->to_tuple();}

my_tuple tuple_or_int::to_my_tuple() const {
    if(std::holds_alternative<std::tuple<int64_t, int64_t>>(val)){
        return my_tuple(std::get<1>(val));
    }
    return my_tuple(std::get<0>(val), std::get<0>(val));

}

my_tuple::my_tuple(const int64_t a, const int64_t b)
    :first(a), second(b)
{}

my_tuple::my_tuple(const int64_t a)
    :first(a), second(a)
{}

my_tuple::my_tuple(std::tuple<int64_t, int64_t> t)
    :first(std::get<0>(t)), second(std::get<1>(t))
{}

my_tuple& my_tuple::operator=(int64_t x){first = x; second = x; return *this;}
my_tuple& my_tuple::operator=(std::tuple<int64_t, int64_t> x){first = std::get<0>(x); second = std::get<1>(x); return *this;}
const int64_t& my_tuple::operator[](const int64_t x) const {if(x == 0){return first;}return second;}
bool my_tuple::operator==(const int64_t x) const {return (first == x && second == x);}
bool my_tuple::operator!=(const int64_t x) const {return (first != x && second != x);}
bool my_tuple::operator > (const int64_t x) const {return (first > x && second > x);}
bool my_tuple::operator < (const int64_t x) const {return (first < x && second < x);}
bool my_tuple::operator >= (const int64_t x) const {return (first >= x && second >= x);}
bool my_tuple::operator <= (const int64_t x) const {return (first <= x && second <= x);}

std::ostream& operator<<(std::ostream& out, const my_tuple& t){
	return out << '(' << t[0] << ',' << t[1] << ')';
}

template<size_t N>
my_n_tuple<N>::my_n_tuple(const int64_t a)
{
    for(size_t i = 0; i < N; ++i)
        arr[i] = a;
}

template<size_t N>
my_n_tuple<N>& my_n_tuple<N>::operator=(int64_t x){
    for(size_t i = 0; i < N; ++i)
        arr[i] = x;
    return *this;
}


template<size_t N>
const int64_t& my_n_tuple<N>::operator[](const int64_t x) const {return arr[x];}
template<size_t N>
bool my_n_tuple<N>::operator==(const int64_t x) const {
    for(size_t i = 0; i < N; ++i){
        if(arr[i] != x){return false;}}
    return true;
}
template<size_t N>
bool my_n_tuple<N>::operator!=(const int64_t x) const {
    for(size_t i = 0; i < N; ++i){
        if(arr[i] == x){return false;}}
    return true;
}
template<size_t N>
bool my_n_tuple<N>::operator > (const int64_t x) const {
    for(size_t i = 0; i < N; ++i){
        if(arr[i] <= x){return false;}}
    return true;
}
template<size_t N>
bool my_n_tuple<N>::operator < (const int64_t x) const {
    for(size_t i = 0; i < N; ++i){if(arr[i] >= x){return false;}}
    return true;
}
template<size_t N>
bool my_n_tuple<N>::operator >= (const int64_t x) const {
    for(size_t i = 0; i < N; ++i){if(arr[i] < x){return false;}}
    return true;
}
template<size_t N>
bool my_n_tuple<N>::operator <= (const int64_t x) const {
    for(size_t i = 0; i < N; ++i){if(arr[i] > x){return false;}}
    return true;
}


template class my_n_tuple<2>;
template class my_n_tuple<3>;
template class my_n_tuple<4>;

template <>
std::ostream& operator<<(std::ostream& out, const my_n_tuple<2>& t){
	return out << '(' << t[0] << ',' << t[1]  << ')';
}

template <>
std::ostream& operator<<(std::ostream& out, const my_n_tuple<3>& t){
	return out << '(' << t[0] << ',' << t[1] << ',' << t[2] << ')';
}

template <>
std::ostream& operator<<(std::ostream& out, const my_n_tuple<4>& t){
	return out << '(' << t[0] << ',' << t[1] << ',' << t[2] << ',' << t[3] << ')';
}


}

}




/* namespace nt{ */
/* namespace utils{ */


/* /1* template<> void throw_exception<DType, DType>(const bool, std::string_view, const DType&, const DType&); *1/ */
/* /1* template<> void throw_exception<int, int>(const bool, std::string_view, const int&, const int&); *1/ */
/* /1* template<> void throw_exception<uint32_t, uint32_t>(const bool, std::string_view, const uint32_t&, const uint32_t&); *1/ */
/* /1* template<> void throw_exception<float, float>(const bool, std::string_view, const float&, const float&); *1/ */
/* /1* template<> void throw_exception<double, double>(const bool, std::string_view, const double&, const double&); *1/ */
/* /1* template<> void throw_exception<SizeRef, SizeRef>(const bool, std::string_view, const SizeRef&, const SizeRef&); *1/ */


/* } */
/* } */
