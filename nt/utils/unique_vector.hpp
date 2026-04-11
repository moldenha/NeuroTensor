#ifndef NT_UTILS_UNIQUE_VECTOR_HPP__
#define NT_UTILS_UNIQUE_VECTOR_HPP__

#include <unordered_set>
#include <vector>
#include "../intrusive_ptr/intrusive_ptr.hpp"

namespace nt{


template<typename T, typename HashSet = std::unordered_set<T>>
class unique_vector : public intrusive_ptr_target {
    HashSet tracker;
    std::vector<T> vals;
    public:
        using const_iterator = typename std::vector<T>::const_iterator;
        using iterator = typename std::vector<T>::iterator;
        using const_reverse_iterator = typename std::vector<T>::const_reverse_iterator;
        using reverse_iterator = typename std::vector<T>::reverse_iterator;

        unique_vector() = default;
        unique_vector(const unique_vector&) = default;
        unique_vector(unique_vector&&) = default;
        inline unique_vector& operator=(const unique_vector& vec){
            tracker = vec.tracker;
            vals = vec.vals;
            return *this;
        }
        inline unique_vector& operator=(unique_vector&& vec){
            tracker = std::move(vec.tracker);
            vals = std::move(vec.vals);
            return *this;
        }

        inline void push_back(const T& val){
            auto p = tracker.insert(val);
            if(!p.second) return;
            vals.push_back(val);
        }

        inline const_iterator cbegin() const noexcept           { return vals.cbegin(); }
        inline const_iterator cend() const noexcept             { return vals.cend();   }
        inline const_iterator begin() const noexcept            { return vals.begin();  }
        inline const_iterator end() const noexcept              { return vals.end();    }
        inline iterator begin() noexcept                        { return vals.begin();  }
        inline iterator end() noexcept                          { return vals.end();    }
        inline reverse_iterator rbegin() noexcept               { return vals.rbegin(); }
        inline reverse_iterator rend() noexcept                 { return vals.rend();   }
        inline const_reverse_iterator crbegin() const noexcept  { return vals.crbegin();}
        inline const_reverse_iterator crend() const noexcept    { return vals.crend();  }

        inline void merge(const unique_vector& vec){
            for(const auto& val : vec){
                this->push_back(val);
            }
        }
        inline std::size_t size() const noexcept { return this->vals.size(); }
        inline void clear() noexcept {
            this->tracker.clear();
            this->vals.clear();
        }
        bool is_in(const T& val) const noexcept {
            return (this->tracker.find(val) != this->tracker.end());
        }
};


}

#endif
