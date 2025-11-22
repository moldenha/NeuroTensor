# Type Info

Some important information:

-- certain types are not necessarily defined on all systems, for example: `float16, float128, int128, uint128`
-- while there are macros that can be used to derive if the type is system defined or not, 
    when writing code, try to avoid using macros when possible and use the built-in `if constexpr`

-- `float16_t` : `nt::type_traits::simde_float16` <- when `true` simde_float16 used otherwise `half::half_float`
-- `int128_t`  : `nt::type_traits::system_int128` <- when `true` the system int128 otherwise `boost::multiprecision::int128`
-- `uint128_t` : `nt::type_traits::system_int128` <- when `true` the system uint128 otherwise the `uint128_t` class
-- `f;pat128_t`: `nt::type_traits::system_float128` <- when `true` the system float128 is used otherwise the `boost::multiprecision::cpp_bin_float_quad` is used
