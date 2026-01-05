import os

def make_file(fname, force = False):
    file_name = fname + '_decl.h'
    if (os.path.exists(file_name) and force is False): 
        return
    lines = ["#ifndef NT_MATH_FUNCTIONAL_" + fname.upper() + "_DECL_H__\n",
             "#define NT_MATH_FUNCTIONAL_" + fname.upper() + "_DECL_H__\n",
             "\n",
             '#include "decl.h"\n',
             'NT_MATH_KMATH_DECL('+fname+')\n',
             '\n',
             '\n',
             '#endif'
    ]

    with open(file_name, 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    make_file('abs', force = True)
    make_file('ceil', force = True)
    make_file('copysign')
    make_file('floor')
    make_file('frexp')
    make_file('ldexp')
    make_file('signbit')
    make_file('trunc')

