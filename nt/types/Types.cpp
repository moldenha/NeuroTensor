#include "Types.h"
#include <ostream>

#include <type_traits>
#include <valarray>
#include "../convert/Convert.h"

namespace nt{



std::ostream& operator<<(std::ostream& os, const int128_t i){
  std::ostream::sentry s(os);
  if (s) {
    nt::uint128_t tmp = i < 0 ? -i : i;
    char buffer[128];
    char *d = std::end(buffer);
    do {
      --d;
      *d = "0123456789"[tmp % 10];
      tmp /= 10;
    } while (tmp != 0);
    if (i < 0) {
      --d;
      *d = '-';
    }
    int len = std::end(buffer) - d;
    if (os.rdbuf()->sputn(d, len) != len) {
      os.setstate(std::ios_base::badbit);
    }
  }
  return os;
}
std::ostream& operator<<(std::ostream& os, const uint128_t i){
  std::ostream::sentry s(os);
  if (s) {
    nt::uint128_t tmp = i;
    char buffer[128];
    char *d = std::end(buffer);
    do {
      --d;
      *d = "0123456789"[tmp % 10];
      tmp /= 10;
    } while (tmp != 0);
    int len = std::end(buffer) - d;
    if (os.rdbuf()->sputn(d, len) != len) {
      os.setstate(std::ios_base::badbit);
    }
  }
  return os;

}

}
