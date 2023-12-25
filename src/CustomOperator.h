
//************************************************************
// CustomOperator.h
//
// (c) 2008, John Crenshaw
//
// This code is available under the terms of The Code Project
// Open License. You should have received a copy of the license
// (CPOL.html) with the code.
//************************************************************

#pragma once

//************************************************************
// Technically speaking, you are not allowed to declare custom
// keywords or operators in C++. In practice, C++ has a macro
// language that is rich enough to make this not only possible,
// but relatively easy. This file declares the macros and
// classes required to make this work easily.
//************************************************************

//************************************************************
//************************************************************
// Things to note
//************************************************************
//************************************************************
//------------------------------------------------------------
// There is currently no support for Unary Prefix operators.
// The reasons are threefold. First, there was no good way to
// add these without creating a debugging nightmare. Second,
// the syntactic benefits of adding these are dubious at best.
// Finally, if you really want to, you can emulate these readily
// using something like the following:
//
//    class myoperator_dummy_left_side {public: myoperator_dummy_left_side(){}};
//    #define myoperator myoperator_dummy_left_side BinaryOperatorDefinition(_op_myoperator, *)
//
// The reason we didn't do something like this in the first
// place is because it creates a fatal order of operations
// problem. Consider the following:
//
//    a * myoperator b
//
// With the workaround, the above code is incorrectly interpretd as:
//
//    (a * myoperator_dummy_left_side) * internalstuff * b
// 
// Since a * myoperator_dummy_left_side is an undefined operation, you would
// get a weird cryptic error until you write:
//
//    a * (myoperator b)
//
// This is no better than:
//
//    a * myoperator(b)
//
// So this portion was dropped.
//
//------------------------------------------------------------
//
// Unary postfix operators have a slightly different order of
// operations than you might expect. They are handled as part
// of the multiplication group, rather than with the other
// postfix operators. Normal postfix operators evaluate BEFORE
// the prefix operators, however Custom postfix operators are
// evaluated AFTER the prefix operators. If you place multiple
// postfix operators on the right, they will still be evaluated
// from left to right however, as expected.
//
//------------------------------------------------------------
//
// Since an operator basically reserves a keyword at the global
// level, you should use caution before defining a new one. Ask
// yourself if this action would be better served by creating a
// member function, global function, or some other mechanism.
// The ability to add your own operators has a place in C++,
// without a doubt; but, that place is not everywhere.
//
//------------------------------------------------------------
//
// There is currently no support for adding custom operators to
// a class, where the function executed is a class member. You
// could work around this easily enough either by adding Bind
// macros that are class aware, or by using the global operator
// function as a stub and calling the member function there.
//
//------------------------------------------------------------
//
// There is currently no support for template operators. This
// could be added, but there are some complexities to consider
// when caching the left hand type for binary operators.
//
//------------------------------------------------------------
// 
// Ternary and Multary (Polyadic or n-ary) operators should
// also be possible, but are not yet implemented. These ARE
// planned, as I have some very specific intentions for the
// Multary operators.
//
//------------------------------------------------------------
// 
// Operators currently have to be composed of a-zA-Z0-9_
// This is a restriction of the way C++ handles things. I CAN
// work around this if I'm willing to put together a file
// preprocessor, but that seems like a bit much for just a bit
// of syntax sugar, and it would make this complicated for
// people to integrate.
//
//************************************************************

//************************************************************
// Usage:
// To create a new operator, you must:
// 1. Define the operator (#define to hook it up)
// 2. Declare it
// 3. Declare any left hand types you will use (Binary only)
// 4. Implement functions form(s) of the operator
// 5. Bind the operator to the function form(s)
//
// Typically, you should place the declarations inside of a *.h
// file. The implementation can either be inline in the *.h file,
// or split, with a declaration in the *.h file and the actual
// implementation in the *.cpp file (just like any other global
// function.)
//
// Macros are provided to simplify stages 1, 2, 3, and 5.
//
// Stage 4 is handled entirely by hand, because this allows for
// greater transparency and readability of the code. This also
// makes it clear what function name you use for this operator,
// in case you ever want/need to use the function instead.
//
// For examples of how to set up and implement your operators,
// see the included sample files.
//************************************************************

//************************************************************
// Defining the operators
//
// When everything else is done, a #define is required to point
// the operator to our new handler. This macro makes things a
// bit easier by hiding some of the bizarre syntax.
//
// Parameters:
// opname = the name passed to DeclareXXXXXXOperator
// precedence = operator that has the same precedence as your
//              new operator should. For example, if you are
//              creating a MATCH operator, which runs a regex
//              on a string, the operator would be expected to
//              have the same precedence as ==.
// precedence = the precedence passed to DeclareXXXXXXOperator
//
// NOTE: Unlike the others, this shouldn't be used directly.
// Instead, it should be used as the second half of a #define:
// #define {opname} BinaryOperatorDefinition({opname}, {precedence})
//************************************************************
#define BinaryOperatorDefinition(opname, precedence) precedence CCustomOperatorHelper_##opname() precedence
#define UnaryPostOperatorDefinition(opname) * CCustomOperatorUnaryPostHelper_##opname()


//************************************************************
// Declaring an operator
//
// Operators need to be declared with one of the functions
// below. This creates a few utility classes that are used
// to hijack operators without making a mess.
//
// Parameters:
// opname = the name of your operator's function counterpart
//************************************************************
#define DeclareBinaryOperator(opname) _macro_DeclareBinaryOperator(opname)
#define DeclareUnaryPostOperator(opname) _macro_DeclareUnaryPostOperator(opname)
// TODO: Ternary should technically be possible. Make it work
// Ternary operators ALWAYS get lowest priority, because otherwise
// you would end up with two halves that make no sense. Consider the
// following:
//
// a ? b, c : d
//
// (a ? b), (c : d) makes no sense whatsoever, so this MUST be interpreted
// as (a) ? (b, c) : (d). This is true of all ternary operators. 
// #define DeclareTernaryOperator(opname, opname2) _macro_DeclareTrinaryOperator(opname, opname2)
// TODO: Multary (Polyadic or n-ary) operators should also be possible. Do it.

//************************************************************
// Declaring left types
//
// We force the macros to evaluate the operators from left to
// right, with a fancy class in the middle. An operator needs
// to be defined to handle the relationship between the left
// hand value, and the class in the middle.
//
// Parameters:
// opname = the name passed to DeclareXXXXXXOperator
// precedence = the precedence passed to XXXXXXOperatorDefinition
// lefttype = C++ data or object type to accept on the left.
//
// Which one to use?
// use the *Ref version of this function if you want the type
// to be treated as a reference. This is faster, and will also
// circumvent the copy constructor, if the type doesn't have
// one. The trade off is that you lose implicit type conversions.
//************************************************************
#define DeclareOperatorLeftType(opname, precedence, lefttype) _macro_DeclareOperatorLeftType(opname, precedence, lefttype)
#define DeclareOperatorLeftTypeRef(opname, precedence, lefttype) _macro_DeclareOperatorLeftTypeRef(opname, precedence, lefttype)


//************************************************************
// Binding the operator(s) and function(s)
//
// We force the macros to evaluate the operators from left to
// right, with a fancy class in the middle. An operator needs
// to be defined to handle the relationship between the left
// hand value, and the class in the middle. BindBinaryOperator
// will create all the magical stuff, and call the function
// designated by opname.
//
// Parameters:
// rettype = return type for the operator
// opname = the name passed to DeclareXXXXXXOperator
// precedence = the precedence passed to XXXXXXOperatorDefinition
// lefttype = C++ data or object type to accept on the left.
// righttype = C++ data or object type to accept on the left.
// leftarg = left parameter (type and name) for the function
// rightarg = right parameter (type and name) for the function
//************************************************************
#define BindBinaryOperator(rettype, opname, precedence, lefttype, righttype) _macro_BindBinaryOperator(rettype, opname, precedence, lefttype, righttype)
#define BindUnaryPostOperator(rettype, opname, lefttype) _macro_BindUnaryPostOperator(rettype, opname, lefttype)



//************************************************************
// We use a class to do some fancy things with the parameters
// in certain operators. This class is the base for that one so
// we can also do some fancy things with deletion in yet a third
// class. The result is a significant reduction in object copying.
// and a substantial savings at runtime. After all, what good is
// syntactic sugar if it makes your app slow?
class CCustomOperator_parambase
{
public:
   virtual ~CCustomOperator_parambase(){}
};



// // squared operator
// class CCustomOperatorUnaryPostHelper_squared
// {
// public:
//    virtual ~CCustomOperatorUnaryPostHelper_squared(){}
// };
// // This would make a fantastic Unary postfix operator, except that
// // the principle only works from inside a class. If you have direct
// // access to the class, this makes little sense.
// // double operator[](double l, CCustomOperatorUnaryPostHelper_squared& r)
// inline double operator*(double l, CCustomOperatorUnaryPostHelper_squared& r)
// {
//    return l*l;
// }
// #define squared * CCustomOperatorUnaryPostHelper_squared()


//************************************************************
// macro implementations (this is where it gets ugly.)


#define _macro_DeclareBinaryOperator(opname)                                                     \
template<class T_left> class CCustomOperatorHelper_##opname##_leftparam_T : public CCustomOperator_parambase   \
{                                                                                                            \
public:                                                                                                      \
   CCustomOperatorHelper_##opname##_leftparam_T(T_left l){m_l_buf = l;m_pl = &m_l_buf;}                            \
   T_left m_l_buf; /* buffer, since we have to copy the left side value */                                   \
   T_left* m_pl; /* pointer to the value on the left */                                                      \
};                                                                                                            \
                                                                                                            \
template<class T_left> class CCustomOperatorHelper_##opname##_leftparamref_T : public CCustomOperator_parambase                                                                                                            \
{                                                                                                            \
public:                                                                                                            \
   CCustomOperatorHelper_##opname##_leftparamref_T(T_left& l){m_pl = &l;}                                                                                                            \
   T_left* m_pl; /* pointer to the value on the left */                                                                                                            \
};                                                                                                            \
                                                                                                            \
class CCustomOperatorHelper_##opname                                                                                                            \
{                                                                                                            \
public:                                                                                                            \
   CCustomOperatorHelper_##opname(){m_pLeft = NULL;}                                                                                                            \
   ~CCustomOperatorHelper_##opname(){delete m_pLeft;}                                                                                                            \
   CCustomOperator_parambase* m_pLeft;                                                                                                            \
};                                                                                                            \

//************************************************************
// END _macro_DeclareBinaryOperator MACRO
//************************************************************


#define _macro_DeclareUnaryPostOperator(opname)                                                                                                            \
class CCustomOperatorUnaryPostHelper_##opname                                                                                                            \
{                                                                                                            \
public:                                                                                                            \
   virtual ~CCustomOperatorUnaryPostHelper_##opname(){}                                                                                                            \
};                                                                                                            \

//************************************************************
// END _macro_DeclareUnaryPostOperator MACRO
//************************************************************


#define _macro_DeclareOperatorLeftType(opname, precedence, lefttype)                                                                                                            \
inline CCustomOperatorHelper_##opname##_leftparam_T<lefttype>& operator precedence (lefttype l, CCustomOperatorHelper_##opname& r)                                                                                                            \
{                                                                                                            \
   return *(CCustomOperatorHelper_##opname##_leftparam_T<lefttype>*)(r.m_pLeft = new CCustomOperatorHelper_##opname##_leftparam_T<lefttype>(l));                                                                                                            \
}                                                                                                            \

//************************************************************
// END _macro_DeclareOperatorLeftType MACRO
//************************************************************

#define _macro_DeclareOperatorLeftTypeRef(opname, precedence, lefttype)                                                                                                            \
inline CCustomOperatorHelper_##opname##_leftparamref_T<lefttype>& operator precedence (lefttype& l, CCustomOperatorHelper_##opname& r)                                                                                                            \
{                                                                                                            \
   return *(CCustomOperatorHelper_##opname##_leftparamref_T<lefttype>*)(r.m_pLeft = new CCustomOperatorHelper_##opname##_leftparamref_T<lefttype>(l));                                                                                                            \
}                                                                                                            \

//************************************************************
// END _macro_DeclareOperatorLeftTypeRef MACRO
//************************************************************


#define _macro_BindBinaryOperator(rettype, opname, precedence, lefttype, righttype)                                                                                                            \
inline rettype operator precedence (CCustomOperatorHelper_##opname##_leftparam_T<lefttype>& l, righttype r)                                                                                                            \
{                                                                                                            \
   return opname(*l.m_pl, r);                                                                                                            \
}                                                                                                            \
   inline rettype operator precedence (CCustomOperatorHelper_##opname##_leftparamref_T<lefttype>& l, righttype r)                                                                                                            \
{                                                                                                            \
   return opname(*l.m_pl, r);                                                                                                            \
}                                                                                                            \

//************************************************************
// END _macro_BindBinaryOperator MACRO
//************************************************************


#define _macro_BindUnaryPostOperator(rettype, opname, lefttype)                                                                                                            \
inline lefttype operator * (lefttype l, CCustomOperatorUnaryPostHelper_##opname& r)                                                                                                            \
{                                                                                                            \
   return opname(l);                                                                                                            \
}                                                                                                            \

//************************************************************
// END _macro_BindBinaryOperator MACRO
//************************************************************

/*
template<class T_left> class CCustomOperatorHelper_like_leftparam_T : public CCustomOperator_leftparambase
{
public:
   CCustomOperatorHelper_like_leftparam_T(T_left l)
   {
      m_l_buf = l;
      m_pl = &m_l_buf;
   }

   T_left m_l_buf; // buffer, since we have to copy the left side value
   T_left* m_pl; // pointer to the value on the left
};

template<class T_left> class CCustomOperatorHelper_like_leftparamref_T : public CCustomOperator_leftparambase
{
public:
   CCustomOperatorHelper_like_leftparamref_T(T_left l)
   {
      m_pl = &l;
   }
   T_left* m_pl; // pointer to the value on the left
};


class CCustomOperatorHelper_like
{
public:
   CCustomOperatorHelper_like()
   {
      m_pLeft = NULL;
   }
   ~CCustomOperatorHelper_like()
   {
      delete m_pLeft;
   }

   CCustomOperator_leftparambase* m_pLeft;

//    template<class T_right> CCustomOperatorHelper_like_leftparam_T<T_right> operator * (T_right r)
//    {
//       return CCustomOperatorHelper_like_leftparam_T<T_right>(r);
//    }
//    template<class T_right> CCustomOperatorHelper_like_leftparam_T<T_right> operator * (T_right& r)
//    {
//       return CCustomOperatorHelper_like_leftparam_T<T_right>(r);
//    }
};
*/

/*
inline CCustomOperatorHelper_like_leftparam_T<CString>& operator * (CString l, CCustomOperatorHelper_like& _like)
{
   return *(CCustomOperatorHelper_like_leftparam_T<CString>*)(_like.m_pLeft = new CCustomOperatorHelper_like_leftparam_T<CString>(l));
}

// inline CCustomOperatorHelper_like_leftparam_T<int>& operator * (int l, CCustomOperatorHelper_like& _like)
// {
//    return *(CCustomOperatorHelper_like_leftparam_T<int>*)(_like.m_pLeft = new CCustomOperatorHelper_like_leftparam_T<int>(l));
// }
*/
/*
bool operator_like (CString _left, CString _right);
bool operator_like (CString _left, int _right);
inline bool operator == (CCustomOperatorHelper_like_leftparam_T<CString>& _like, CString s)
{
   return operator_like(*_like.m_pl, s);
}

inline bool operator == (CCustomOperatorHelper_like_leftparam_T<CString>& _like, int i)
{
   return operator_like(*_like.m_pl, i);
}

inline bool operator_like (CString _left, CString _right)
{
   return !!_left.Find(_right);
}

inline bool operator_like (CString _left, int _right)
{
   TCHAR a[20];
   return !!_left.Find(_itot_s(_right, a, 10));
}
*/

/*
#define like * CCustomOperatorHelper_like() ==
*/

