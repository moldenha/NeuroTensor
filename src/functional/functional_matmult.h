#include "../types/Types.h"
#include <cstddef>


namespace nt{
/* float16_t dot_product_16(const float16_t*, const float16_t*, size_t size); */ // <- does not work 
/* float dot_product_8(const float*, const float*, size_t size); */
/* double dot_product_4(const double*, const double*, size_t size); */
//for anything other than floats and doubles, the dot product needs to be done manually
//there is also the libbf16 library which if it could be installed, would be able great, there is an intel version that can be used to do all the operations at the same time

}
