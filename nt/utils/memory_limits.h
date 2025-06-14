#ifndef _NT_UTILS_MEMORY_LIMITS_H_
#include "utils.h"

namespace nt::utils{

#ifndef USE_PARALLEL
inline unsigned int getThreadsPerCore() { return 1; }
inline uint64_t get_shared_memory_max() { return 0; }

#else
#ifdef __linux__
#include <unistd.h>
#include <cstdint>
#include <fstream>
#include <string>
#include <cstdio>
#include <string>
#include <thread>
// Function to get the number of threads per core on Linux
inline unsigned int getThreadsPerCore() {
    FILE *pipe = popen("lscpu | grep 'Thread(s) per core:'", "r");
    if (!pipe) {
        return static_cast<unsigned int>(std::thread::hardware_concurrency());
    }
    char buffer[128];
    std::string result = "";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    size_t pos = result.find(":");
    if (pos != std::string::npos) {
        std::string threads_per_core_str = result.substr(pos + 2);
        try{
            int threads_per_core = std::stoi(threads_per_core_str);
            return  threads_per_core <= 1 ?  static_cast<unsigned int>(std::thread::hardware_concurrency()) :  static_cast<unsigned int>(threads_per_core); 
        } catch (const std::invalid_argument& e) {
            return static_cast<unsigned int>(std::thread::hardware_concurrency());
        }
    }
    return static_cast<unsigned int>(std::thread::hardware_concurrency());
}

inline uint64_t get_shared_memory_max() {
    std::ifstream shmmax_file("/proc/sys/kernel/shmmax");
    uint64_t shmmax = 0;
    if (shmmax_file.is_open()) {
        shmmax_file >> shmmax;
    }
    return shmmax;
}


#elif defined(_WIN32)
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

#elif defined(__APPLE__)
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
#endif //Platform
#endif //USE_PARALLEL

} //nt::utils::

#endif
