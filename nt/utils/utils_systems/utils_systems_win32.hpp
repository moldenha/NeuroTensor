#ifdef _WIN32

#ifndef NT_UTILS_UTILS_SYSTEMS_WIN32_HPP__
#define NT_UTILS_UTILS_SYSTEMS_WIN32_HPP__

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdint>
#include <io.h>  // for _get_osfhandle

namespace nt::utils{

#ifdef USE_PARALLEL
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

int32_t getPipeReadableBytes(int fd) {
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

    return static_cast<int32_t>(bytesAvailable);
}

bool isPipeReadable(int pipefd){
    return getPipeReadableBytes(pipefd) > 0; 
}

#else  // not defined USE_PARALLEL
#endif //USE_PARALLEL

} //nt::utils::

#endif // NT_UTILS_UTILS_SYSTEMS_WIN32_HPP__ 
#endif // _WIN32
