# KMath
kmath is an extension of neurotensor designed to ensure the following functions for real floating types are constexpr. This is similar to the `boost::math::ccmath` library. However, there are a few different caveates:

- The functions always evaluate to constexpr
    - `boost::math::ccmath` will check for constexpr function use first and then use a constexpr implementation
    - This will always use the constexpr implementation
    - This inevitably means that if you use these functions in a non-constexpr way, they will be slower
