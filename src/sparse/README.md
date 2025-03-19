## Sparse Data

There are plans to add sparse matrix and sparse tensor representations. There is already a sparse tensor representation, but it is not represented memory-wise how I want it to be. If you look at the sparse memory data header, that is more synonymous of how I want the Spase Matrix and Sparse Tensor to work. Where memory is only filled in where needed, for example in cuda there would only be sparse amounts of data filled in at specific indexes, which would help with GPU usage for sparse data. This will mainly be used for linear algebra functions. For now though, it is not entirely supported, and in the beggining stages of being supported and finished.

So far this is experimental, the only fully functioning class is the SparseTensor
