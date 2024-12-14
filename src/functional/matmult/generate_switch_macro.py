from contextlib import redirect_stdout


begin_comments = '''
//this is a file that has macros to auto generate switch statements
//this way the matmult function can remain to have a templated type
//and a variable tile size and pack size depending on the type
'''

#these are other essential macros
begin_macros = '''
#define _NT_MATMULT_MIN_(x, y) x < y ? x : y
#define _NT_MATMULT_MAX_(x, y) x > y ? x : y


#define _NT_MATMULT_ENSURE_ALIGNMENT_(type, align_byte, amt) (((amt * sizeof(type)) % align_byte != 0) ? (amt * sizeof(type)) + align_byte - ((amt * sizeof(type)) % align_byte) : amt * sizeof(type))


#define _NT_MATMULT_NTHREADS_ 21
#define _NT_MATMULT_DECLARE_STATIC_BLOCK_(type) static type blockA_packed_##type[_NT_MATMULT_ENSURE_ALIGNMENT_(type, 64, tile_size_v<type> * _NT_MATMULT_NTHREADS_ * tile_size_v<type> * _NT_MATMULT_NTHREADS_)] __attribute((aligned(64)));\
				   static type blockB_packed_##type[_NT_MATMULT_ENSURE_ALIGNMENT_(type, 64, tile_size_v<type> * _NT_MATMULT_NTHREADS_ * tile_size_v<type> * _NT_MATMULT_NTHREADS_)] __attribute((aligned(64)));

#define DETER_NT_MATMULT_MIN_E_SIZE(length, addition) length % addition == 0 ? length / addition : int64_t(length / addition) + 1

#define _NT_MATMULT_DETERMINE_SIZE_(length, addition) length % addition == 0 ? length / addition : int64_t(length / addition) + 1

'''

def print_file_start():
    print(begin_comments)
    print("#ifndef _NT_MATMULT_MACROS_H_")
    print("#define _NT_MATMULT_MACROS_H_")
    print(begin_macros)


def print_single_case_macro(num):
    if num == 0:
        print("#define _NT_MATMULT_CASE_{}_(max, case_macro)\\".format(num))
        print("\tdefault:\\")
        print("\t\tcase_macro(max, 0, 0)")
        return
    print("#define _NT_MATMULT_CASE_{}_(max, case_macro)\\".format(num))
    print("\tcase {}:\\".format(num))
    print("\t\tcase_macro(max, {}, {})\\".format(num, num-1))
    print("\t_NT_MATMULT_CASE_{}_(max, case_macro)".format(num-1))


def print_case_macros(num):
    for i in range(num):
        print_single_case_macro(i)
        print(' ')
    print("#define _NT_MATMULT_SWITCH_(max, var_name, case_macro)\\")
    print("\tswitch(var_name){\\")
    print("\t\t_NT_MATMULT_CASE_##max##_(max, case_macro)\\")
    print("\t}")



def print_single_tile_kif(num):
    if num % 2 != 0:
        return
    if num == 0:
        return
    if num == 2:
        print("#define _NT_MATMULT_TILE_KIF_2_(start, var_name, case_macro)\\")
        print("\tstart if constexpr (tile_size == 2){\\")
        print("\t\t_NT_MATMULT_SWITCH_(2, var_name, case_macro);\\")
        print("\t}")
        return
    print("#define _NT_MATMULT_TILE_KIF_{}_(start, var_name, case_macro)\\".format(num))
    print("\tstart if constexpr (tile_size == "+str(num)+"){\\")
    print("\t\t_NT_MATMULT_SWITCH_({}, var_name, case_macro);\\".format(num))
    print("\t}\\")
    print("\t_NT_MATMULT_TILE_KIF_{}_(else, var_name, case_macro)".format(num-2))



def print_single_tile_kif_list(num, next_num):
    if next_num == 0:
        print("#define _NT_MATMULT_TILE_KIF_{}_(start, var_name, case_macro)\\".format(num))
        print("\tstart if constexpr (tile_size == "+str(num)+"){\\")
        print("\t\t_NT_MATMULT_SWITCH_("+str(num)+", var_name, case_macro);\\")
        print("\t}")
        return
    print("#define _NT_MATMULT_TILE_KIF_{}_(start, var_name, case_macro)\\".format(num))
    print("\tstart if constexpr (tile_size == "+str(num)+"){\\")
    print("\t\t_NT_MATMULT_SWITCH_({}, var_name, case_macro);\\".format(num))
    print("\t}\\")
    print("\t_NT_MATMULT_TILE_KIF_{}_(else, var_name, case_macro)".format(next_num))

def print_tile_macros(num):
    for i in range(num):
        print_single_tile_kif(i)
        print(' ')
    print("#define _NT_MATMULT_SWITCHES_(var_name, case_macro) _NT_MATMULT_TILE_KIF_{}_(, var_name, case_macro)".format(num-1))

def print_tile_macros_list(nums):
    for i in reversed(range(len(nums))):
        if i == 1:
            print_single_tile_kif_list(nums[i], 0)
            print(' ')
            break
        print_single_tile_kif_list(nums[i], nums[i-1])
        print(' ')
    print("#define _NT_MATMULT_SWITCHES_(var_name, case_macro) _NT_MATMULT_TILE_KIF_{}_(, var_name, case_macro)".format(max(nums)))


# print_tile_macros_list([4, 8, 16, 32, 64])


def print_end_file():
    print()
    print()
    print()
    print("#endif //_NT_MATMULT_MACROS_H_")


def print_file(nums):
    print_file_start()
    print_case_macros(max(nums)+1)
    print_tile_macros_list(nums)
    print_end_file()

#takes the bits of variables that are going to be worked with
def bits_to_tilesizes(nums):
    avx = []
    avx2 = []
    for num in nums:
        avx.append(int(128/num) * 2)
        avx2.append(int(256/num) * 2)
    together = avx
    together.extend(avx2)
    out = set(together)
    out = list(out)
    out.sort()
    return out

if __name__ == '__main__':
    nums = bits_to_tilesizes([8, 16, 32, 64])
    with open('nt_matmult_macros.h', 'w') as f:
        with redirect_stdout(f):
            print_file(nums)
            
