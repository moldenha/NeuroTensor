#if (defined(__APPLE__) || defined(__linux__) || defined(__unix__)) && !defined(_WIN32)

#ifndef NT_UTILS_UTILS_SYSTEMS_POSIX_HPP__
#define NT_UTILS_UTILS_SYSTEMS_POSIX_HPP__

#include <sys/types.h>
#include <signal.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <unistd.h>


namespace nt::utils{

#ifdef USE_PARALLEL
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

#else  // not defined USE_PARALLEL
#endif //USE_PARALLEL

} //nt::utils::

#endif // NT_UTILS_UTILS_SYSTEMS_POSIX_HPP__ 
#endif // (defined(__APPLE__) || defined(__linux__) || defined(__unix__)) && !defined(_WIN32)
