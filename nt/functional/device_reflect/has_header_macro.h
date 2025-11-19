// This is a header file for macros to use to determine if a header file exists for a device
// if it does, then it can be used

#ifndef NT_DEVICE_HAS_HEADER_MACRO_H__
#define NT_DEVICE_HAS_HEADER_MACRO_H__

#ifndef NT_STR_
#define NT_STR_(n) #n
#endif

#define NT_DEVICE_HEADER_PATH(device, op) "../"#device"/"#op".h"

#ifdef __has_include
#define NT_CHECK_HEADER_PATH(device, op)\
    __has_include(NT_STR_(NT_DEVICE_HEADER_PATH(devce, op)))
#else
#define NT_CHECK_HEADER_PATH(device, op) 1 // true
#endif





#endif // NT_DEVICE_HAS_HEADER_MACRO_H__
