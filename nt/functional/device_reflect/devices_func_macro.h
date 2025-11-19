// macro for a function to be applied for all different devices

#ifndef NT_DEVICES_FUNCTIONAL_REFLECT_FUNC_MACRO_H__
#define NT_DEVICES_FUNCTIONAL_REFLECT_FUNC_MACRO_H__

#define NT_GET_DEVICES_FUNCTIONAL_FUNC(func, ...)
    func(cpu, __VA_ARGS__)\
    func(mkl, __VA_ARGS__)\
    func(cuda, __VA_ARGS__)



#endif // NT_DEVICES_FUNCTIONAL_REFLECT_FUNC_MACRO_H__

