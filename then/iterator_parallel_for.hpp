/*
 *
 * the entire point of this header file is to take an iterator pointing to a multidimensional object (a tensor) and optimize for loops on it
 * because this library has 3 different iterator types, and of course different strides are supported
 *  -regular pointer (T*)
 *  -list all bucketed pointer (BucketIterator_list<T>, T**)
 *  -and an iterator when it is multiple buckets of contiguous memory (BucketIterator_blocked<T>)
 *
 * BucketIterator_blocked<T> has a lot of overhead when compared to T*,
 * this is meant to reduce that
 * a lot of this logic can be translated to handling cuda kernels especially when memory is bucketed
 * will probably make a similar wrapper of this when cuda support is added
 * will make an std only wrapper (which is just having to make a task group as I already have an std::parallel_for equivalent to tbb::parallel_for
 * also going to make an accelerate wrapper for Accelerate, for Apple ARM cores
 *
 * In the future I will adapt this to add 2d, 3d, and 4d support (you can see commented out lines where there was untested 2d support)
 *
 * I would also like to add dual iterator support
 * this just happens to not be at the top of my to-do list right now
 * and this was honestly a little one day code side quest
 *
 *
 */


#ifdef USE_PARALLEL
#include "iterator_parallel_for_parallel.hpp"
#else
#include "iteraotor_parallel_for_no_parallel.hpp"
#endif

