#include "SimplexConstruct.h"
#include "SimplexRadi.h"
#include <set>
#include <vector>

namespace nt {
namespace tda {

std::vector<std::vector<int64_t>>
search_cliques(const Tensor &within_radius, const Tensor &overlay_mask,
               std::vector<int64_t> candidates, int64_t depth, int64_t K,
               std::vector<int64_t> &item_vec) {
    if (depth == K) {
        return {std::move(candidates)};
    }
    std::vector<std::vector<int64_t>> cliques;
    const int64_t last = candidates.back();

    candidates.push_back(0);
    auto c_begin = candidates.cbegin();
    auto c_end = candidates.cend();
    --c_end;
    if (within_radius[last].item<Tensor>().is_null()) {
        return std::move(cliques);
    }
    Tensor neighbors = within_radius[last].item<Tensor>().item<Tensor>();
    if (neighbors.is_null() || neighbors.is_empty()) {
        return std::move(cliques);
    }
    // std::cout << "neighbors: "<<neighbors<<std::endl;
    const int64_t *n_begin =
        reinterpret_cast<const int64_t *>(neighbors.data_ptr());
    const int64_t *n_end =
        reinterpret_cast<const int64_t *>(neighbors.data_ptr_end());
    for (; n_begin != n_end && *n_begin < last; ++n_begin) {
        ;
    }
    // while(*n_begin < last && n_begin < n_end){++n_begin;}
    if (n_begin >= n_end) {
        return std::move(cliques);
    }
    for (; n_begin < n_end; ++n_begin) {
        bool stop = false;
        for (; c_begin != c_end; ++c_begin) {
            item_vec[0] = *n_begin;
            item_vec[1] = *c_begin;
            if (!overlay_mask.item<bool>(item_vec)) {
                stop = true;
                break;
            }
        }
        c_begin = candidates.cbegin();
        if (stop)
            continue;
        candidates.back() = *n_begin;
        auto n_cliques = search_cliques(within_radius, overlay_mask, candidates,
                                        depth + 1, K, item_vec);
        cliques.insert(cliques.cend(), n_cliques.begin(), n_cliques.end());
    }
    return std::move(cliques);
}

Tensor construct_simplexes(const Tensor &points, const Tensor &initial,
                           const std::vector<std::vector<int64_t>> &cliques,
                           int64_t N) {
    std::vector<Tensor> final(cliques.size());
    int64_t point_dim = initial.shape()[-1];
    auto cliques_b = cliques.cbegin();
    auto cliques_e = cliques.cend();
    auto f_begin = final.begin();
    for (; cliques_b != cliques_e; ++cliques_b, ++f_begin) {
        std::vector<Tensor> outputs(N + 1);
        outputs[0] = initial;
        auto o_begin = outputs.begin();
        ++o_begin;
        auto c_begin = cliques_b->cbegin();
        auto c_end = cliques_b->cend();
        int64_t pos = 1;
        for (; c_begin != c_end; ++c_begin, ++o_begin, ++pos) {
            *o_begin = points[*c_begin];
        }
        // a lack of checks in terms of shapes makes the following faster
        // when I already know it will work
        *f_begin = functional::cat_unordered(outputs).view(N + 1, point_dim);
    }
    return functional::stack(std::move(final));
}

Tensor _sub_find_all_simplicies(int64_t simplicies_amt, const Tensor &points,
                                const Tensor &ball_masks, double r,
                                int64_t index) {
    // std::cout << "point[0] is "<<points[0] << std::endl;
    Tensor initial = points[index];
    Tensor mask = ball_masks[index];
    if (functional::count(mask) < simplicies_amt) {
        // std::cout << "was less than simplicies_amt"<<std::endl;
        return Tensor::Null();
    }
    Tensor potentials = points[mask];
    if (simplicies_amt == 2) {
        std::vector<Tensor> stacking(potentials.shape()[0]);
        for (int64_t i = 0; i < potentials.shape()[0]; ++i) {
            std::vector<Tensor> output({initial, potentials[i]});
            stacking[i] = functional::cat(std::move(output)).view(2, -1);
        }
        return functional::stack(std::move(stacking));
    }
    // std::cout << "potential points: "<<potentials<<std::endl;
    int64_t D = potentials.shape()[-1];
    int64_t N = potentials.shape()[-2];
    Tensor cpy_potentials =
        potentials.view(1, -1, D).expand({N, N, D}).transpose(0, 1).clone();
    Tensor diff = potentials.view(-1, 1, D) - cpy_potentials;
    Tensor dist_sq = diff.pow(2).sum(-1); // pairwise distance
    double r_sq = (r * r);
    Tensor overlap_mask = (dist_sq <= r_sq);
    // std::cout << "simplex overlap mask before: "<<overlap_mask << std::endl;
    overlap_mask.fill_diagonal_(false);
    // std::cout << "simplex overlap mask after: "<<overlap_mask << std::endl;
    Tensor split = overlap_mask.split_axis(0);
    // std::cout << "split is: "<<split<<std::endl;
    Tensor within_radius = functional::where(split);
    // std::cout << "simplicies amt is "<<simplicies_amt<<std::endl;
    // std::cout << "got within radius: "<<within_radius <<std::endl;
    if (N >= simplicies_amt) {
        std::vector<std::vector<int64_t>> cliques;
        std::vector<int64_t> item_vec({0, 0});
        for (int64_t i = 0; i < N; ++i) {
            auto n_cliques = search_cliques(within_radius, overlap_mask, {i}, 1,
                                            simplicies_amt - 1, item_vec);
            cliques.reserve(n_cliques.size());
            for (const auto &cliq : n_cliques) {
                cliques.push_back(cliq);
            }
        }
        if (cliques.size() == 0) {
            return Tensor::Null();
        }
        // std::cout << "cliques ("<<cliques.size()<<"): {"<<std::endl;
        // for(size_t i = 0; i < cliques.size()-1; ++i){
        //     std::cout << '(';
        //     for(size_t j = 0; j < cliques[i].size()-1; ++j)
        //         std::cout << cliques[i][j] << ',';
        //     std::cout << cliques[i].back()<<")"<<std::endl;
        // }
        // for(size_t j = 0; j < cliques.back().size()-1; ++j)
        //     std::cout << cliques.back()[j] << ',';
        // std::cout << cliques.back().back()<<")}"<<std::endl;
        // construct simplicies based on cliques
        return construct_simplexes(potentials, initial, cliques,
                                   simplicies_amt - 1);
        // std::cout << "simplicies: "<<simplicies<<std::endl;
    }

    return Tensor::Null();
}

void _sub_find_all_simplex_indexes(int64_t simplicies_amt, const Tensor &points,
                                   const Tensor &ball_masks, double r,
                                   int64_t index,
                                   std::vector<int64_t> &cliques) {
    // std::cout << "point[0] is "<<points[0] << std::endl;
    Tensor initial = points[index];
    Tensor mask = ball_masks[index];

    if (functional::count(mask) < simplicies_amt) {
        // std::cout << "was less than simplicies_amt"<<std::endl;
        return;
    }
    Tensor potentials = points[mask];
    if (simplicies_amt == 2) {
        int64_t amt = functional::count(mask);
        cliques.reserve(amt * simplicies_amt);
        const bool *start = reinterpret_cast<const bool *>(mask.data_ptr());
        const bool *stop = reinterpret_cast<const bool *>(mask.data_ptr_end());
        for (int64_t i = 0; start != stop; ++i, ++start) {
            if (i == index) {
                continue;
            }
            if (*start) {
                cliques.push_back(index);
                cliques.push_back(i);
            }
        }
        return;
    }
    // std::cout << "potential points: "<<potentials<<std::endl;
    int64_t D = potentials.shape()[-1];
    int64_t N = potentials.shape()[-2];
    if (N < simplicies_amt)
        return;
    Tensor cpy_potentials =
        potentials.view(1, -1, D).expand({N, N, D}).transpose(0, 1).clone();
    Tensor diff = potentials.view(-1, 1, D) - cpy_potentials;
    Tensor dist_sq = diff.pow(2).sum(-1); // pairwise distance
    double r_sq = (r * r);
    Tensor overlap_mask = (dist_sq <= r_sq);
    // std::cout << "simplex overlap mask before: "<<overlap_mask << std::endl;
    overlap_mask.fill_diagonal_(false);
    // std::cout << "simplex overlap mask after: "<<overlap_mask << std::endl;
    Tensor split = overlap_mask.split_axis(0);
    // std::cout << "split is: "<<split<<std::endl;
    Tensor within_radius = functional::where(split);
    // std::cout << "simplicies amt is "<<simplicies_amt<<std::endl;
    // std::cout << "got within radius: "<<within_radius <<std::endl;
    Tensor fix_mask = functional::where(mask)[0].item<Tensor>();
    const int64_t *arr = reinterpret_cast<const int64_t *>(fix_mask.data_ptr());
    // std::vector<std::vector<int64_t>> cliques;
    std::vector<int64_t> item_vec({0, 0});
    for (int64_t i = 0; i < N; ++i) {
        auto n_cliques = search_cliques(within_radius, overlap_mask, {i}, 1,
                                        simplicies_amt - 1, item_vec);
        cliques.reserve(n_cliques.size() * simplicies_amt);
        for (const auto &cliq : n_cliques) {
            cliques.push_back(index);
            for (const auto &sub_index : cliq)
                cliques.push_back(arr[sub_index]);
        }
    }
}

Tensor find_all_simplicies_indexes(int64_t simplicies_amt, const Tensor &points,
                                   const BasisOverlapping &balls,
                                   double radius) {
    Tensor overlap_mask = balls.adjust_radius(radius);
    std::vector<Tensor> out_batches(points.shape()[0]);
    for (int64_t b = 0; b < points.shape()[0]; ++b) {
        Tensor pts_ = points[b].item<Tensor>();
        if (simplicies_amt == 1) {
            // std::cout << "pts has shape "<<pts_.shape() << std::endl;
            out_batches[b] =
                functional::arange(pts_.shape()[0], DType::int64).view(-1, 1);
            // out_batches[b] = pts_.view(pts_.shape()[0], 1, -1);
            // std::cout << "out_batches[b] has shape "<<out_batches[b].shape()
            // << std::endl;
            continue;
        }
        Tensor masks_ = overlap_mask[b].item<Tensor>();
        std::vector<int64_t> cliques;
        for (int64_t i = 0; i < pts_.shape()[0]; ++i) {
            _sub_find_all_simplex_indexes(simplicies_amt, pts_, masks_, radius,
                                          i, cliques);
        }
        if (cliques.size() == 0) {
            out_batches[b] = Tensor::Null();
            continue;
        }
        Tensor indexes =
            functional::vector_to_tensor(cliques).view(-1, simplicies_amt);
        Tensor sorted = functional::sort(indexes, -1, false, true,
                                         false); // return sorted only
        Tensor unique =
            functional::unique(sorted.view(-1, simplicies_amt), -1, true, false)
                .contiguous(); // return unique only
        out_batches[b] = std::move(unique);
    }
    return functional::vectorize(std::move(out_batches));
    // outs.reserve(points.shape()[0]);
    // for(int64_t i = 0; i <
    // return _sub_find_all_simplicies(simplicies_amt, points[0].item<Tensor>(),
    // overlap_mask[0].item<Tensor>(), radius);
}

Tensor from_index_simplex_to_point_simplex(const Tensor &indexes,
                                           const Tensor &points) {
    if (indexes.dtype() == DType::TensorObj || points.dtype() == DType::TensorObj) {
        utils::throw_exception(indexes.dtype() == points.dtype(),
                               "Expected if both tensors are dtype tensor they "
                               "are both but got indexes $ and points $",
                               indexes.dtype(), points.dtype());
        utils::throw_exception(indexes.numel() == points.numel(),
                               "Expected if both tensors are dtype tensor they "
                               "are the sanem numel indexes $ and points $",
                               indexes.numel(), points.numel());
        Tensor out = Tensor::makeNullTensorArray(indexes.numel());
        Tensor *o_begin = reinterpret_cast<Tensor *>(out.data_ptr());
        const Tensor *i_begin =
            reinterpret_cast<const Tensor *>(indexes.data_ptr());
        const Tensor *p_begin =
            reinterpret_cast<const Tensor *>(points.data_ptr());
        const Tensor *i_end = i_begin + indexes.numel();
        for (; i_begin != i_end; ++i_begin, ++p_begin, ++o_begin) {
            *o_begin = from_index_simplex_to_point_simplex(*i_begin, *p_begin);
        }
        return std::move(out);
    }
    utils::throw_exception(
        indexes.dims() == 2,
        "Expected indexes to have a dimension of 2 but got $", indexes.dims());
    const int64_t &N = indexes.shape()[-1];
    const int64_t &D = points.shape()[-1];
    Tensor flattened = indexes.flatten(0, -1).contiguous();
    return points[flattened].view(-1, N, D);
}

// Function to generate all unique combinations of size N
void generateCombinations(const int64_t &num_size, int64_t N,
                          std::vector<int64_t> &result,
                          std::vector<int64_t> &current, int64_t start) {
    if (current.size() == N) {
        result.insert(result.end(), current.begin(), current.end());
        return;
    }

    for (int64_t i = start; i < num_size; ++i) {
        current.push_back(i);
        generateCombinations(num_size, N, result, current, i + 1);
        current.pop_back();
    }
}

size_t binomialCoeff(int n, int k) {
    if (k > n - k)
        k = n - k;
    size_t res = 1;
    for (int i = 0; i < k; ++i) {
        res *= (n - i);
        res /= (i + 1);
    }
    return res;
}

// Wrapper function
std::vector<int64_t>
getCombinationsWithoutSelfContainment(const int64_t &num_size, int N) {
    size_t numCombinations = binomialCoeff(num_size, N) * N;
    std::vector<int64_t> combinations;
    combinations.reserve(numCombinations); // Pre-allocate memory
    std::vector<int64_t> current;
    current.reserve(N);
    generateCombinations(num_size, N, combinations, current, 0);
    return std::move(combinations);
}

void getCombinationsWithoutSelfContainment(std::vector<int64_t> &result,
                                           const int64_t &num_size, int64_t N,
                                           int64_t start) {
    std::vector<int64_t> current;
    current.reserve(N);
    generateCombinations(num_size, N, result, current, start);
}

// finds all simplexes
// sorts them based on their radii
// returns a tensor containing 2 tensors:
//   t1: tensor of batches, with simplicies
//           depending on indexes only true: {B, N} DType::int64
//                                     false: {B, N, D} DType::int64
//                                     B: number of simplicies
//                                     N: simplicies_amt
//                                     D: dimension of points
//   t2: tensor of batches, with radi: out {B, DType::Float64}
//   it should be noted, the main slow down on this function stems from
//   calculating the radi and distance between each point which makes sense
//   beacuse there are so many simplicies (thousands) and multiple slow
//   operations such as a sqrt function and squaring the size of the matricies
//   consider making a function to take the distance in relation to the
//   BasisOverlapping as there is already a dist_sq matrix that has been
//   calculated
Tensor find_all_simplicies(int64_t simplicies_amt, const Tensor &points,
                           const BasisOverlapping &balls,
                           bool indexes_only) {
    std::vector<Tensor> out_batches(points.shape()[0]);
    std::vector<Tensor> out_radi(points.shape()[0]);
    std::map<int64_t, std::vector<int64_t>> batch_map;
    int64_t c = 0;
    for (const auto point : points) {
        if (batch_map.count(point.shape()[0])) {
            batch_map[point.shape()[0]].push_back(c);
        } else {
            batch_map[point.shape()[0]] = {c};
        }
        ++c;
    }
    for (const auto [num_elements, batches] : batch_map) {
        // auto start = std::chrono::high_resolution_clock::now();
        std::vector<int64_t> combinations =
            getCombinationsWithoutSelfContainment(num_elements, simplicies_amt);
        if (combinations.size() == 0) {
            for (const auto b : batches) {
                out_batches[b] = Tensor::Null();
                out_radi[b] = Tensor::Null();
            }
            continue;
        }
        Tensor indexes =
            functional::vector_to_tensor(combinations).view(-1, simplicies_amt);
        Tensor sorted = functional::sort(indexes, -1, false, true,
                                         false); // return sorted only
        Tensor unique = functional::unique(sorted.view(-1, simplicies_amt), -1,
                                           true, false); // return unique only
        for (const auto b : batches) {
            Tensor n_unique = unique.clone();
            Tensor radi = compute_point_radii(n_unique, balls, b);
            out_batches[b] = std::move(n_unique);
            out_radi[b] = std::move(radi);
        }
    }
    Tensor simplex_complex = functional::vectorize(std::move(out_batches));
    Tensor radii = functional::vectorize(std::move(out_radi));
    sort_simplex_on_radi(simplex_complex, radii);
    if (indexes_only)
        return functional::list(simplex_complex, radii);
    return functional::list(
        from_index_simplex_to_point_simplex(simplex_complex, points), radii);
}

//for num points, it is only indexes only
std::pair<Tensor, Tensor> find_all_simplicies(int64_t simplicies_amt, const int64_t num_points,
                           const Tensor &distance_matrix, double max_radi, bool sort) {
    //distance matrix gives the distance between index i and j as distance_matrix[i][j];
    utils::throw_exception(num_points > 0,
                           "Cannot find simplex complex from 0 points");
    utils::throw_exception(simplicies_amt > 0,
                           "Cannot get simplex comples of -1-simplex");
    utils::throw_exception(num_points > simplicies_amt,
                           "Cannot find $-simplex with only $ points", 
                           simplicies_amt-1, num_points);
    utils::THROW_EXCEPTION(distance_matrix.dtype() != nt::DType::TensorObj, 
                           "INTERNAL LOGIC ERROR: Got batches of distance matrices but did not detect batches $", 
                           distance_matrix.dtype());
    utils::throw_exception(distance_matrix.dtype() == nt::DType::Float32, 
                           "Expected distance matrix to be a dtype of float32 got $", 
                           distance_matrix.dtype());
    std::vector<int64_t> combinations =
            getCombinationsWithoutSelfContainment(num_points, simplicies_amt);
    utils::THROW_EXCEPTION(combinations.size() != 0,
                           "INTERNAL LOGIC ERROR: Unable to find simplices");
    Tensor indexes =
            functional::vector_to_tensor(combinations).view(-1, simplicies_amt);
    Tensor sorted = functional::sort(indexes, -1, false, true,
                                     false); // return sorted only
    Tensor simplex_complex = functional::unique(sorted.view(-1, simplicies_amt), -1,
                                       true, false).contiguous(); // return unique only
    std::pair<Tensor, Tensor> radi_gr = compute_point_grad_radii(simplex_complex, distance_matrix);
    if(sort)
        sort_simplex_on_radi(simplex_complex, radi_gr.first, radi_gr.second);

    if(max_radi < 0)
        return {functional::list(simplex_complex, radi_gr.first), radi_gr.second};
    Tensor less_than = radi_gr.first <= max_radi;
    utils::throw_exception(!functional::none(less_than), "No simplex radi is less than $", max_radi);
    return {functional::list(simplex_complex[less_than], radi_gr.first[less_than]), radi_gr.second[less_than]};

}

} // namespace tda
} // namespace nt
