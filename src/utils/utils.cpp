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
