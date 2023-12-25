#include <_types/_uint32_t.h>
#include <iostream>
#include <numeric>
#include <vector>
#include <memory.h>

namespace nt{
namespace permute{

class Permuter{
	void** strides_old;
	const std::vector<uint32_t>& strides;
	const std::vector<uint32_t>& shape;
	std::vector<uint32_t> shape_accumulate;
	public:
		Permuter(void**, const std::vector<uint32_t>&, const std::vector<uint32_t>&);
		uint32_t get_index(uint32_t) const;
		void* get_ptr(uint32_t);

};


void Permute(void** ar, void** n_str, uint32_t size, const std::vector<uint32_t>& new_shape, const std::vector<uint32_t>& new_strides);

}
}
