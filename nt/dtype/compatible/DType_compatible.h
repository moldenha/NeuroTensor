#ifndef NT_DTYPE_DTYPE_COMPATIBLE_H__
#define NT_DTYPE_DTYPE_COMPATIBLE_H__

#include "../../types/Types.h"
#include "compatible_macro.h"
#include "DTypeDeclareMacros.h"


#define X(type, dtype_enum_a, dtype_enum_b)\
    NT_REGISTER_FLOATING_TYPE(type, k##dtype_enum_a)
NT_GET_X_FLOATING_DTYPES_
#undef X

#define X(type, dtype_enum_a, dtype_enum_b)\
    NT_REGISTER_COMPLEX_TYPE(type, k##dtype_enum_a)
NT_GET_X_COMPLEX_DTYPES_
#undef X

#define X(type, dtype_enum_a, dtype_enum_b)\
    NT_REGISTER_INTEGER_TYPE(type, k##dtype_enum_a)
NT_GET_X_SIGNED_INTEGER_DTYPES_
#undef X

#define X(type, dtype_enum_a, dtype_enum_b)\
    NT_REGISTER_UNSIGNED_TYPE(type, k##dtype_enum_a)
NT_GET_X_UNSIGNED_INTEGER_DTYPES_
#undef X

#define X(type, dtype_enum_a, dtype_enum_b)\
    NT_REGISTER_OTHER_TYPE(type, k##dtype_enum_a)
NT_GET_X_OTHER_DTYPES_
#undef X

NT_FINISH_DTYPE_REGISTER()





#endif //NT_DTYPE_DTYPE_COMPATIBLE_H__
