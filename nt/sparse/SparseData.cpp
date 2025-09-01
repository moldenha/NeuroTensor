#include "SparseData.h"

namespace nt{
namespace sparse_details{

void SparseMemoryData::push_back(int64_t index, const void* data) {
    if (size >= max_size) {
        reserve(max_size > 0 ? max_size * 2 : 1);
    }
    indices.push_back(index);
    std::memcpy(static_cast<char*>(memory) + (size * type_size), data, type_size);
    size++;
}

void* SparseMemoryData::access(int64_t index) {
    for (int64_t i = 0; i < size; i++) {
        if (indices[i] == index) {
            return static_cast<char*>(memory) + (i * type_size);
        }
    }
    return nullptr; // Not found
}

void SparseMemoryData::reserve(int64_t new_max_size) {
    if (new_max_size <= max_size) return;
    
    indices.reserve(new_max_size);
    void* new_memory = MetaMalloc(type_size * new_max_size);

    if (memory) {
        std::memcpy(new_memory, memory, type_size * size);
        MetaCStyleFree(memory);
    }

    memory = new_memory;
    max_size = new_max_size;
}

template <typename T, typename Compare>
std::vector<std::size_t> sort_permutation(
    const std::vector<T>& vec,
    Compare& compare)
{
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(),
        [&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
    return p;
}


template <typename T>
void apply_permutation_in_place(
    std::vector<T>& vec,
    const std::vector<std::size_t>& p)
{
    std::vector<bool> done(vec.size());
    for (std::size_t i = 0; i < vec.size(); ++i)
    {
        if (done[i])
        {
            continue;
        }
        done[i] = true;
        std::size_t prev_j = i;
        std::size_t j = p[i];
        while (i != j)
        {
            std::swap(vec[prev_j], vec[j]);
            done[j] = true;
            prev_j = j;
            j = p[j];
        }
    }
}

// void SparseMemoryData::sort(){
    
    
// }

}

}
