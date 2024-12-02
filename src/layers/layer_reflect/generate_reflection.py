#this is a way to generate the reflection library parts

import copy

#this adds a character to every string in the list
def add_atBeginList(l, ch):
    for i in range(len(l)):
        l[i] = ch + l[i]
    return l

#creates a list of alphabet characters of n variables long
def alphabet_at(num):
    alphabet = list(map(chr, range(ord('a'), ord('z')+1))) + list(map(chr, range(ord('A'), ord('Z')+1)))
    if(num < len(alphabet)):
        return alphabet[0:num]
    copy_alphabet = copy.deepcopy(alphabet)
    n_alphabet = copy.deepcopy(alphabet)
    index = 0
    while(len(n_alphabet)<num):
        distance = ord('z') - ord('a')
        if(index < distance):
            ch = chr(index+ord('a'))
        elif(index < distance*2):
            ch = chr(index-distance+ord('A'))
        else:
            index = 0
            alphabet = add_atBeginList(alphabet, 'a')
            copy_alphabet = copy.deepcopy(alphabet)
            ch = 'a'
        n_alphabet = n_alphabet + add_atBeginList(copy_alphabet, ch)
        copy_alphabet = copy.deepcopy(alphabet)
        index += 1
    return n_alphabet[0:num]


#print a list with a max number of variables
def print_list(l, max_variables):
    index = 0
    print("[", end='')
    for i in range(len(l)):
        if(i+1 != len(l)):
            print(l[i],end=', ')
        else:
            print(l[i],end='')
        if(index >= max_variables):
            print('')
            print('\t\t\t',end='')
            index=0
        index += 1
    print(']', end='')

        

#creates the tie structure function given the correct number of arguments
def create_tie_structure(num):
    print("template<typename T>")
    print("inline constexpr auto tie_structure(T& val, detail::size_t_<"+str(num)+">){")
    print("\tauto& ", end='')
    alphabet = alphabet_at(num)
    print_list(alphabet, 20)
    print(' = const_cast<std::remove_cv_t<T>&>(val);')
    print("\treturn detail::make_tuple_of_references(")
    for i in range(len(alphabet)-1):
        print("\t\t\tdetail::workaround_cast<T, decltype({})>,".format(alphabet[i]))
    print("\t\t\tdetail::workaround_cast<T, decltype({})>);".format(alphabet[-1]))
    print("}")




from contextlib import redirect_stdout
    
def print_all_tie_structures(num):
    with open('tie_structure.txt', 'w') as f:
        with redirect_stdout(f):
            for i in range(num):
                create_tie_structure(i+1)




def variables_to_strings_macro(num):
    print("#define _NT_VARIABLE_STRING_0_() {}")
    for i in range(1, num):
        print("#define _NT_VARIABLE_STRING_{}_(".format(i), end='')
        for j in range(i-1):
            print("name{}".format(j+1), end=', ')
        print("name{}".format(i), end=") ")
        print("{",end='')
        for j in range(i-1):
            print("#name{}".format(j+1), end=', ')
        print("#name{}".format(i), end='')
        print("}")

def variable_to_string_select_macro(num):
    print("#define _NT_VARIABLE_STRING_SELECT_MACRO_(",end='')
    for i in range(num):
        print('_{}'.format(i+1), end=', ')
    print(", NAME, ...) NAME)")
    print("#define _NT_VARIABLE_STRING_(...) _NT_VARIABLE_STRING_SELECT_MACRO_(__VA_ARGS__, ",end='')
    for i in reversed(range(num)):
        print("_NT_VARIABLE_STRING_{}_".format(i+1), end=", ")
    print("_NT_VARIABLE_STRING_0_)(__VA_ARGS__)")

   

def cls_ptr_macro(num):
    print("#define _NT_CLS_PTR_VAR_0_(ptr, cls)")
    for i in range(1, num):
        print("#define _NT_CLS_PTR_VAR_{}_(ptr, cls, ".format(i), end='')
        for j in range(i-1):
            print("name{}".format(j+1), end=', ')
        print("name{}".format(i), end=") ")
        for j in range(i-1):
            print("::nt::reflect::detail::workaround_cast<cls, decltype(ptr->name{})>(ptr->name{})".format(j+1, j+1), end=", ")
        print("::nt::reflect::detail::workaround_cast<cls, decltype(ptr->name{})>(ptr->name{})".format(i, i))


def cls_ptr_macro_select(num):
    print("#define _NT_CLS_TO_ITERATOR_(ptr, cls, ...) _NT_VARIABLE_SELECT_MACRO_(__VA_ARGS__",end='')
    for i in reversed(range(num)):
        print("_NT_CLS_PTR_VAR_{}_".format(i+1), end=', ')
    print("_NT_CLS_PTR_VAR_0_)(ptr, cls, __VA_ARGS__)")


def num_args_macro(num):
    print("#define _NT_NUMARGS_HELPER_(", end='')
    for i in range(num):
        print('_{}'.format(i+1), end=', ')
    print("N, ...) N")
    print("#define _NT_NUMARGS_(...) _NT_NUMARGS_HELPER_(__VA_ARGS__, ", end='')
    for i in reversed(range(num)):
        print(i+1, end=", ")
    print("0)")

if __name__ == '__main__':
    #do the tie structure output
    #print_all_tie_structures(100)
    #print variables to string macro
    #variables_to_strings_macro(100)
    # variable_to_string_select_macro(100)
    #print the num args macro
    #num_args_macro(100)
    cls_ptr_macro_select(100)
    # with open('cls_ptr_macro.txt', 'w') as f:
    #     with redirect_stdout(f):
            # cls_ptr_macro(100)
