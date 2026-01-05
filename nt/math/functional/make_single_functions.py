import os

def make_file(fname, force = True):
    file_name = fname + '.hpp'
    if (os.path.exists(file_name) and force is False): 
        return
    lines = ["#ifndef NT_MATH_FUNCTIONAL_" + fname.upper() + "_HPP__\n",
             "#define NT_MATH_FUNCTIONAL_" + fname.upper() + "_HPP__\n",
             "\n",
             '#include "utils.h"\n',
             '#include "general_include.h"\n',
             '#include NT_MAKE_FLOAT128_MATH_FUNCTION_ACCESSIBLE__('+fname+')\n\n'
             '\n',
             '// namespace nt::math{ \n',
             '\n',
             'NT_MAKE_SINGULAR_FUNCTION_(' + fname + ')\n',
             '\n',
             '// nt::math:: \n',
             '\n',
             '#endif'
    ]

    with open(file_name, 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    make_file('sqrt', force = True)
    make_file('exp')
    make_file('log')
    make_file('tanh')
    make_file('cosh')
    make_file('sinh')
    make_file('asinh')
    make_file('acosh')
    make_file('atanh')
    make_file('atan')
    make_file('asin')
    make_file('acos')
    make_file('tan')
    make_file('sin')
    make_file('cos')
