//this is a header file containing macros to determine the number of arguments in a __VA_ARGS__ macro expansion
//and also if there are none
//this is really useful and used a few different times
#ifndef NT_NUMARGS_MACRO_IS_EMPTY_MACRO_HEADER__
#define NT_NUMARGS_MACRO_IS_EMPTY_MACRO_HEADER__

namespace nt{

//get true number of __VA_ARGS__ including 0
//this is expanded from https://stackoverflow.com/questions/66556552/a-way-to-count-the-number-of-va-args-arguments-including-0-without-compile
//The author Jens Gustedt, specifically, from the blog post Detect empty macro arguments 
//this was implemented to basically be able to detect if __VA_ARGS__ is 0 or 1
//this was a big issue for layers who have 0 parameters or sublayers, such as the identity layer
/* NOTE: In these macros, "1" means true, and "0" means false. */



#define _NT_EXPAND_(x) x

#define __NT_GLUE(X,Y) X##Y
#define _NT_GLUE_(X,Y) __NT_GLUE(X,Y)

/* Returns the 100th argument. */
#define _NT_ARG_100_(_,\
   _100,_99,_98,_97,_96,_95,_94,_93,_92,_91,_90,_89,_88,_87,_86,_85,_84,_83,_82,_81, \
   _80,_79,_78,_77,_76,_75,_74,_73,_72,_71,_70,_69,_68,_67,_66,_65,_64,_63,_62,_61, \
   _60,_59,_58,_57,_56,_55,_54,_53,_52,_51,_50,_49,_48,_47,_46,_45,_44,_43,_42,_41, \
   _40,_39,_38,_37,_36,_35,_34,_33,_32,_31,_30,_29,_28,_27,_26,_25,_24,_23,_22,_21, \
   _20,_19,_18,_17,_16,_15,_14,_13,_12,_11,_10,_9,_8,_7,_6,_5,_4,_3,_2,X_,...) X_

/* Returns whether __VA_ARGS__ has a comma (up to 100 arguments). */
#define _NT_HAS_COMMA_(...) _NT_EXPAND_(_NT_ARG_100_(__VA_ARGS__, \
   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, \
   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0))

/* Produces a comma if followed by a parenthesis. */
#define _NT_TRIGGER_PARENTHESIS_(...) ,
#define _NT_PASTE5_(_0, _1, _2, _3, _4) _0 ## _1 ## _2 ## _3 ## _4
#define _NT_IS_EMPTY_CASE_0001 ,
/* Returns true if inputs expand to (false, false, false, true) */
#define __NT_IS_EMPTY__(_0, _1, _2, _3) _NT_HAS_COMMA_(_NT_PASTE5_(_NT_IS_EMPTY_CASE_, _0, _1, _2, _3))
/* Returns whether __VA_ARGS__ is empty. */
#define _NT_IS_EMPTY_(...)                                               \
   __NT_IS_EMPTY__(                                                       \
      /* Testing for an argument with a comma                       \
         e.g. "ARG1, ARG2", "ARG1, ...", or "," */                  \
      _NT_HAS_COMMA_(__VA_ARGS__),                                       \
      /* Testing for an argument around parenthesis                 \
         e.g. "(ARG1)", "(...)", or "()" */                         \
      _NT_HAS_COMMA_(_NT_TRIGGER_PARENTHESIS_ __VA_ARGS__),                 \
      /* Testing for a macro as an argument, which will             \
         expand the parenthesis, possibly generating a comma. */    \
      _NT_HAS_COMMA_(__VA_ARGS__ (/*empty*/)),                           \
      /* If all previous checks are false, __VA_ARGS__ does not     \
         generate a comma by itself, nor with _TRIGGER_PARENTHESIS_ \
         behind it, nor with () after it.                           \
         Therefore, "_TRIGGER_PARENTHESIS_ __VA_ARGS__ ()"          \
         only generates a comma if __VA_ARGS__ is empty.            \
         So, this tests for an empty __VA_ARGS__ (given the         \
         previous conditionals are false). */                       \
      _NT_HAS_COMMA_(_NT_TRIGGER_PARENTHESIS_ __VA_ARGS__ (/*empty*/))      \
   )

#define _NT_VAR_COUNT_EMPTY_1(...) 0
#define _NT_VAR_COUNT_EMPTY_0(...) _NT_EXPAND_(_NT_ARG_100_(__VA_ARGS__, \
   100,99,98,97,96,95,94,93,92,91,90,89,88,87,86,85,84,83,82,81, \
   80,79,78,77,76,75,74,73,72,71,70,69,68,67,66,65,64,63,62,61, \
   60,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41, \
   40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21, \
   20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1))
#define _NT_NUMARGS_(...) _NT_GLUE_(_NT_VAR_COUNT_EMPTY_, _NT_IS_EMPTY_(__VA_ARGS__))(__VA_ARGS__)

//example usage:
/*

1 means empty
#define _NT_HANDLE_NULL_TENSORS_EMPTY_1(...) throw_exception(!is_null(), "Cannot perform operation"
                                                                         __NT_FUNCTION_NAME__
                                                                        "on a null tensor")
0 means non-empty
#define _NT_HANDLE_NULL_TENSORS_EMPTY_0(...) _NT_HANDLE_NULL_TENSORS_NON_EMPTY_SELECT_(__VA_ARGS__, _NT_HANDLE_NULL_TENSORS_NON_EMPTY_4, _NT_HANDLE_NULL_TENSORS_NON_EMPTY_3, _NT_HANDLE_NULL_TENSORS_NON_EMPTY_2, _NT_HANDLE_NULL_TENSORS_NON_EMPTY_1, _NT_HANDLE_NULL_TENSORS_NON_EMPTY_0)(__VA_ARGS__)
#define _NT_HANDLE_NULL_TENSORS_(...) _NT_GLUE_(_NT_HANDLE_NULL_TENSORS_EMPTY_, _NT_IS_EMPTY_(__VA_ARGS__))(__VA_ARGS__)

*/

}

#endif  
