#ifndef NT_DTYPE_ENUM_H__
#define NT_DTYPE_ENUM_H__

#include "../types/Types.h"
#include "compatible/DTypeDeclareMacros.h"


namespace nt{

#define X(type, dtype_enum_a, dtype_enum_b)\
    dtype_enum_a,\
    dtype_enum_b = dtype_enum_a,\

enum class DType{
   NT_GET_X_FLOATING_DTYPES_ 
   NT_GET_X_COMPLEX_DTYPES_
   NT_GET_X_SIGNED_INTEGER_DTYPES_
   NT_GET_X_UNSIGNED_INTEGER_DTYPES_
   NT_GET_X_OTHER_DTYPES_
};

enum class DTypeShared{
   NT_GET_X_FLOATING_DTYPES_ 
   NT_GET_X_COMPLEX_DTYPES_
   NT_GET_X_SIGNED_INTEGER_DTYPES_
   NT_GET_X_UNSIGNED_INTEGER_DTYPES_
   NT_GET_X_OTHER_DTYPES_
};

#undef X

// Declaring constexpr dtypes
// Used in compatible to make macro declaration and other uses easier

#define X(type, dtype_enum_a, dtype_enum_b)\
static constexpr DType k##dtype_enum_a = DType::dtype_enum_a;\
static constexpr DType k##dtype_enum_b = DType::dtype_enum_b;\

NT_GET_X_FLOATING_DTYPES_ 
NT_GET_X_COMPLEX_DTYPES_
NT_GET_X_SIGNED_INTEGER_DTYPES_
NT_GET_X_UNSIGNED_INTEGER_DTYPES_
NT_GET_X_OTHER_DTYPES_

#undef X

#define X(type, dtype_enum_a, dtype_enum_b)\
    case DTypeShared::dtype_enum_a: return DType::dtype_enum_a;\

inline DType DTypeShared_DType(const DTypeShared& dt){
	switch(dt){
       NT_GET_X_FLOATING_DTYPES_ 
       NT_GET_X_COMPLEX_DTYPES_
       NT_GET_X_SIGNED_INTEGER_DTYPES_
       NT_GET_X_UNSIGNED_INTEGER_DTYPES_
       NT_GET_X_OTHER_DTYPES_
       default:
            return DType::Bool;
	}
}

#undef X

#define X(type, dtype_enum_a, dtype_enum_b)\
    case DType::dtype_enum_a: return DTypeShared::dtype_enum_a;\


inline DTypeShared DType_DTypeShared(const DType& dt){
	switch(dt){
       NT_GET_X_FLOATING_DTYPES_ 
       NT_GET_X_COMPLEX_DTYPES_
       NT_GET_X_SIGNED_INTEGER_DTYPES_
       NT_GET_X_UNSIGNED_INTEGER_DTYPES_
       NT_GET_X_OTHER_DTYPES_
       default:
            return DTypeShared::Bool;
	}
}

#undef X

inline const char* toString(const DType& dt){
    switch(dt){
#define X(type, dtype_enum_a, dtype_enum_b)\
    case DType::dtype_enum_a : return "DType" #dtype_enum_a;\

       NT_GET_X_FLOATING_DTYPES_ 
       NT_GET_X_COMPLEX_DTYPES_
       NT_GET_X_SIGNED_INTEGER_DTYPES_
       NT_GET_X_UNSIGNED_INTEGER_DTYPES_
       NT_GET_X_OTHER_DTYPES_
       default:
            return "UnknownType";
    }
#undef X
}


}

#endif //NT_DTYPE_ENUM_H__
