#include "SimplexRadi.h"
#include "SimplexConstruct.h"
#include "../dtype/ArrayVoid.hpp"

namespace nt {
namespace tda {

Tensor compute_circumradii(const Tensor &simplicies) {
    /*
     Computes the associated circum radius with each simplex

     Args:
        simplicies: {B, N+1, D}
            B: batches of simplicies
            N+1: N+1 verticies for each simplex
            D: dimensionality of each point
     Returns:
        Tensor: {B}, dtype float64
            B: the radius associated with each simplex

    */
    if (simplicies.dtype == DType::TensorObj) {
        Tensor out = Tensor::makeNullTensorArray(simplicies.numel());
        Tensor *begin = reinterpret_cast<Tensor *>(out.data_ptr());
        for (auto t : simplicies) {
            *begin = compute_circumradii(t);
            ++begin;
        }
        return std::move(out);
    }
    utils::THROW_EXCEPTION(
        simplicies.dims() == 3,
        "Expected to get simplicies of dims 3, corresponding to batches, "
        "verticies, point dim but got $",
        simplicies.dims());
    const int64_t &B = simplicies.shape()[0];  // Batches
    const int64_t Np1 = simplicies.shape()[1]; // N+1 verticies
    const int64_t &D = simplicies.shape()[2];  // Dims

    Tensor f_simplicies = simplicies.to(DType::Float64);
    Tensor centroid =
        f_simplicies.mean(-2, true); // the center of the simplexes
    centroid = centroid.repeat_(-2, 3);

    Tensor avg_distances = f_simplicies - centroid;
    avg_distances.pow_(2);
    avg_distances = avg_distances.sum({-1, -2});
    // it is divided by 4 because when the simplexes were formed, it was dist_sq
    // < (2 * radius)^2
    avg_distances /= (Np1 * 4);
    return functional::sqrt(avg_distances);
}

Tensor compute_circumradii(const Tensor &index_simplicies,
                           const Tensor &points) {
    /*
     Computes the associated circum radius with each simplex

     Args:
        simplicies: {B, N+1}
            B: batches of simplicies
            N+1: N+1 verticies for each simplex
     Returns:
        Tensor: {B}, dtype float64
            B: the radius associated with each simplex

    */
    if (index_simplicies.dtype == DType::TensorObj) {
        Tensor out =
            Tensor::makeNullTensorArray(index_simplicies.numel());
        Tensor *begin = reinterpret_cast<Tensor *>(out.data_ptr());
        const Tensor *p_begin =
            reinterpret_cast<const Tensor *>(points.data_ptr());
        for (auto t : index_simplicies) {
            *begin = compute_circumradii(t, *p_begin);
            ++begin;
            ++p_begin;
        }
        return std::move(out);
    }
    Tensor simplicies =
        from_index_simplex_to_point_simplex(index_simplicies, points);
    utils::THROW_EXCEPTION(
        simplicies.dims() == 3,
        "Expected to get simplicies of dims 3, corresponding to batches, "
        "verticies, point dim but got $",
        simplicies.dims());
    const int64_t &B = simplicies.shape()[0];  // Batches
    const int64_t Np1 = simplicies.shape()[1]; // N+1 verticies
    const int64_t &D = simplicies.shape()[2];  // Dims

    Tensor f_simplicies = simplicies.to(DType::Float64);
    Tensor centroid =
        f_simplicies.mean(-2, true); // the center of the simplexes
    centroid = centroid.repeat_(-2, 3);

    Tensor avg_distances = f_simplicies - centroid;
    avg_distances.pow_(2);
    avg_distances = avg_distances.sum({-1, -2});
    // it is divided by 4 because when the simplexes were formed, it was dist_sq
    // < (2 * radius)^2
    avg_distances /= (Np1 * 4);
    return functional::sqrt(avg_distances);
}

Tensor compute_point_radii(Tensor simplicies) {
    /*
     Computes the associated point radius with each simplex

     Args:
        simplicies: {B, N+1, D}
            B: batches of simplicies
            N+1: N+1 verticies for each simplex
            D: dimensionality of each point
     Returns:
        Tensor: {B}, dtype float64
            B: the radius associated with each simplex

    */
    if (simplicies.dtype == DType::TensorObj) {
        Tensor out = Tensor::makeNullTensorArray(simplicies.numel());
        Tensor *begin = reinterpret_cast<Tensor *>(out.data_ptr());
        for (auto t : simplicies) {
            *begin = compute_point_radii(t);
            ++begin;
        }
        return std::move(out);
    }
    utils::THROW_EXCEPTION(
        simplicies.dims() == 3,
        "Expected to get simplicies of dims 3, corresponding to batches, "
        "verticies, point dim but got $",
        simplicies.dims());
    const int64_t &B = simplicies.shape()[0];  // Batches
    const int64_t Np1 = simplicies.shape()[1]; // N+1 verticies
    const int64_t &D = simplicies.shape()[2];  // Dims

    Tensor f_simplicies =
        simplicies.to(DType::Float64).view(B, 1, Np1, D);
    Tensor f_simplicies_2 = f_simplicies.view(B, Np1, 1, D);
    // diff: (B, Np1, 1, D) - (B, 1, Np1, D) -> (B, Np1, Np1, D)
    Tensor diff = f_simplicies_2 - f_simplicies;

    // Compute squared Euclidean distances: sum across last dimension (D)
    // diff_sq: (B, Np1 * Np1)
    Tensor dist_sq = diff.pow(2).sum(-1).view(B, Np1 * Np1);

    Tensor max_radi = dist_sq.max(-1).values;

    return functional::sqrt(max_radi) / 4;
}

Tensor compute_point_radii(const Tensor &index_simplicies,
                           const Tensor &points) {
    /*
     Computes the associated point radius with each simplex

     Args:
        simplicies: {B, N+1}
            B: batches of simplicies
            N+1: N+1 verticies for each simplex
     Returns:
        Tensor: {B}, dtype float64
            B: the radius associated with each simplex

    */
    if (index_simplicies.dtype == DType::TensorObj) {
        Tensor out =
            Tensor::makeNullTensorArray(index_simplicies.numel());
        Tensor *begin = reinterpret_cast<Tensor *>(out.data_ptr());
        const Tensor *p_begin =
            reinterpret_cast<const Tensor *>(points.data_ptr());
        for (auto t : index_simplicies) {
            *begin = compute_point_radii(t, *p_begin);
            ++begin;
            ++p_begin;
        }
        return std::move(out);
    }
    Tensor simplicies =
        from_index_simplex_to_point_simplex(index_simplicies, points);
    // utils::THROW_EXCEPTION(simplicies.dims() == 3, "Expected to get
    // simplicies of dims 3, corresponding to batches, verticies, point dim but
    // got $", simplicies.dims());
    const int64_t &B = simplicies.shape()[0];  // Batches
    const int64_t Np1 = simplicies.shape()[1]; // N+1 verticies
    const int64_t &D = simplicies.shape()[2];  // Dims

    Tensor f_simplicies =
        simplicies.to(DType::Float64).view(B, 1, Np1, D);
    Tensor f_simplicies_2 = f_simplicies.view(B, Np1, 1, D);
    // diff: (B, Np1, 1, D) - (B, 1, Np1, D) -> (B, Np1, Np1, D)
    Tensor diff = f_simplicies_2 - f_simplicies;

    // Compute squared Euclidean distances: sum across last dimension (D)
    // diff_sq: (B, Np1 * Np1)
    Tensor dist_sq = diff.pow(2).sum(-1).view(B, Np1 * Np1);

    Tensor max_radi = dist_sq.max(-1).values;

    return functional::sqrt(max_radi) / 4;
}

Tensor compute_point_radii(const Tensor &index_simplicies,
                           const BasisOverlapping &balls, int64_t batch) {

    const int64_t &B = index_simplicies.shape()[0];  // Batches
    const int64_t Np1 = index_simplicies.shape()[1]; // N+1 verticies

    Tensor distances = functional::sqrt(balls.get_distances(batch)) / 2;
    const double *radi[distances.shape()[0]];
    const double *begin =
        reinterpret_cast<const double *>(distances.data_ptr());
    const double *end = begin + distances.numel();
    const int64_t &cols = distances.shape()[1];
    int64_t c = 0;
    for (; begin != end; begin += cols, ++c)
        radi[c] = begin;

    Tensor out_radi = functional::zeros({B}, DType::Float64);
    double *o_begin = reinterpret_cast<double *>(out_radi.data_ptr());
    double *o_end = reinterpret_cast<double *>(out_radi.data_ptr_end());
    const int64_t *current =
        reinterpret_cast<const int64_t *>(index_simplicies.data_ptr());
    const int64_t *current_end =
        reinterpret_cast<const int64_t *>(index_simplicies.data_ptr_end());
    int64_t i, j;
    for (; o_begin != o_end; ++o_begin, current += Np1) {
        for (i = 0; i < Np1; ++i) {
            for (j = 0; j < Np1; ++j) {
                if (j == i)
                    continue;
                *o_begin = std::max(*o_begin, radi[current[i]][current[j]]);
            }
        }
    }

    return std::move(out_radi);
}

std::pair<Tensor, Tensor> compute_point_grad_radii(const Tensor& index_simplicies,
                           const Tensor& distances){
    utils::throw_exception(distances.is_contiguous(), 
                           "Expected to compute point radii from contiguous distance matrix");
    const int64_t &B = index_simplicies.shape()[0];  // Batches
    const int64_t Np1 = index_simplicies.shape()[1]; // N+1 verticies

    const float *radi[distances.shape()[0]];
    const float *begin =
        reinterpret_cast<const float *>(distances.data_ptr());
    const float *end = begin + distances.numel();
    const int64_t &cols = distances.shape()[1];
    int64_t c = 0;
    for (; begin != end; begin += cols, ++c)
        radi[c] = begin;
    
    Tensor out_radi = functional::zeros({B}, DType::Float32);
    Tensor out_indexes({B}, DType::int64);
    float *o_begin = reinterpret_cast<float *>(out_radi.data_ptr());
    float *o_end = reinterpret_cast<float *>(out_radi.data_ptr_end());
    int64_t* o_indexes = reinterpret_cast<int64_t*>(out_indexes.data_ptr());
    int64_t* o_indexes_end = reinterpret_cast<int64_t*>(out_indexes.data_ptr_end());
    utils::THROW_EXCEPTION((o_indexes_end-o_indexes) == (o_end - o_begin), "INTERNAL LOGIC ERROR PTR LENGTH");
    const int64_t *current =
        reinterpret_cast<const int64_t *>(index_simplicies.data_ptr());
    const int64_t *current_end =
        reinterpret_cast<const int64_t *>(index_simplicies.data_ptr_end());
    int64_t i, j, cur_i, cur_j;
    for (; o_begin != o_end; ++o_begin, ++o_indexes, current += Np1) {
        cur_i = 0; cur_j = 0;
        for (i = 0; i < Np1; ++i) {
            for (j = 0; j < Np1; ++j) {
                if (j == i)
                    continue;
                const float& cur_r = radi[current[i]][current[j]];
                if(cur_r > *o_begin){
                    *o_begin = cur_r;
                    cur_i = i; cur_j = j;
                }
                // *o_begin = std::max(*o_begin, radi[current[i]][current[j]]);
            }
        }
        *o_indexes = (current[cur_i] * cols + current[cur_j]);

    }
    if(o_indexes != o_indexes_end){
        *o_indexes = (current[cur_i] * cols + current[cur_j]);
    }
    return {out_radi, out_indexes};

    
}

void sort_simplex_on_radi(Tensor &simplicies, Tensor &simplex_radi) {
    if (simplicies.dtype != DType::TensorObj) {
        utils::THROW_EXCEPTION(
            simplex_radi.dtype != DType::TensorObj,
            "Expected if simplicies are not batched, neither are the radi");
        auto [sorted, indices] = get<2>(functional::sort(simplex_radi));
        Tensor n_simplex = simplicies[indices];
        simplex_radi.swap(sorted);
        simplicies.swap(n_simplex);
        return;
    }
    utils::THROW_EXCEPTION(
        simplex_radi.dtype == DType::TensorObj,
        "Expected if simplicies were batched so were the radi");
    Tensor *radi_begin =
        reinterpret_cast<Tensor *>(simplex_radi.data_ptr());
    Tensor *simplex_begin =
        reinterpret_cast<Tensor *>(simplicies.data_ptr());
    Tensor *simplex_end =
        reinterpret_cast<Tensor *>(simplicies.data_ptr_end());
    for (; simplex_begin != simplex_end; ++simplex_begin, ++radi_begin) {
        auto [sorted, indices] = get<2>(functional::sort(*radi_begin));
        Tensor n_simplex = (*simplex_begin)[indices];
        radi_begin->swap(sorted);
        simplex_begin->swap(n_simplex);
    }
}

void sort_simplex_on_radi(Tensor &simplicies, Tensor &simplex_radi, Tensor& grad_indexes) {
    utils::THROW_EXCEPTION(
        simplex_radi.dtype != DType::TensorObj,
        "Expected if simplicies are not batched, neither are the radi");
    auto [sorted, indices] = get<2>(functional::sort(simplex_radi));
    Tensor n_simplex = simplicies[indices];
    Tensor n_grad_indexes = grad_indexes[indices];
    simplex_radi.swap(sorted);
    simplicies.swap(n_simplex);
    grad_indexes.swap(n_grad_indexes);
}



std::set<double> get_radi_set(const Tensor &simplex_radi, int64_t batch) {
    if (simplex_radi.dtype == DType::TensorObj) {
        std::set<double> rSimplex;
        utils::THROW_EXCEPTION(
            simplex_radi[batch].item<Tensor>().dtype == DType::Float64,
            "Expected dtype of radi to be double but got $",
            simplex_radi[batch].item<Tensor>().dtype);
        simplex_radi[batch]
            .item<Tensor>()
            .arr_void()
            .cexecute_function<
                WRAP_DTYPES<DTypeEnum<DType::Float64>>>(
                [&rSimplex](auto begin, auto end) {
                    for (; begin != end; ++begin) {
                        rSimplex.insert(*begin);
                    }
                });
        return std::move(rSimplex);
    }
    utils::THROW_EXCEPTION(simplex_radi.dtype == DType::Float64,
                               "Expected dtype of radi to be double but got $",
                               simplex_radi.dtype);
    std::set<double> rSimplex;
    simplex_radi.arr_void()
        .cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Float64>>>(
            [&rSimplex](auto begin, auto end) {
                for (; begin != end; ++begin) {
                    rSimplex.insert(*begin);
                }
            });
    return std::move(rSimplex);
}

} // namespace tda
} // namespace nt
