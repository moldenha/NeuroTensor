
#include <iostream>
#include <numeric>
#include <vector>
#include <memory.h>

namespace nt{
namespace permute{

class Permuter{
	void** strides_old;
	const std::vector<int64_t>& strides;
	const std::vector<int64_t>& shape;
	std::vector<int64_t> shape_accumulate;
	public:
		Permuter(void**, const std::vector<int64_t>&, const std::vector<int64_t>&);
		int64_t get_index(int64_t) const;
		void* get_ptr(int64_t);

};


void Permute(void** ar, void** n_str, uint32_t size, const std::vector<int64_t>& new_shape, const std::vector<int64_t>& new_strides);

}
}
