/*
 * I don't generally like to do this, but
 *      if there are going to be a lot of header files that include the same things
 *      and that for what ever reason ever needs to change, it would be better to do that here
*/

#include "utils.h"
    // float128_t
    // float16_t
    // uint128_t
    // int128_t
    // my_complex<T>
#include "../../types/Types.h"
#include "../../utils/always_inline_macro.h"
#include "../../utils/type_traits.h"
#include "../../convert/Convert.h"
#include "../../dtype/compatible/DType_compatible.h"
#include <cmath>

