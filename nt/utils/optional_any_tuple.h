#ifndef NT_UTILS_OPTIONAL_ANY_TUPLE_H__
#define NT_UTILS_OPTIONAL_ANY_TUPLE_H__

#include <type_traits>
#include <vector>
#include "api_macro.h"

namespace nt {
namespace utils {

template <typename T> class NEUROTENSOR_API optional_any_tuple {
    std::vector<T> vals;

  public:
    template <typename G,
              std::enable_if_t<std::is_convertible_v<G, T>, bool> = true>
    optional_any_tuple(G val) : vals({T(val)}) {}

    template <typename G, typename... Args>
    optional_any_tuple(G val, Args... vals) : vals({T(val), T(vals)...}) {}

    optional_any_tuple(std::initializer_list<T> list) : vals(list) {}

    template <typename G,
              std::enable_if_t<std::is_convertible_v<G, T>, bool> = true>
    optional_any_tuple(std::initializer_list<G> list) {
        vals.reserve(list.size());
        for (auto l : list) {
            vals.emplace_back(l);
        }
    }

    inline const std::vector<T> &get_vals() const { return vals; }
    inline int64_t size() const { return static_cast<int64_t>(vals.size()); }
    inline const T &operator[](const size_t s) const { return vals[s]; }
};

} // namespace utils
} // namespace nt

#endif
