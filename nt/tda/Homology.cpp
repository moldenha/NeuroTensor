#include "Homology.h"
#include "../dtype/ArrayVoid.hpp"
#include "../linalg/linalg.h"
#include "Boundaries.h"
#include "MatrixReduction.h"
#include "cpu/MatrixReduction.h"
#include "SimplexConstruct.h"
#include "SimplexRadi.h"
#include <queue>
#include "../utils/macros.h"

namespace nt {
namespace tda {

// this is to make maps
std::map<double, int64_t> make_simplex_radi_map(const Tensor &simplex_radi) {
    utils::THROW_EXCEPTION(simplex_radi.dtype() == DType::Float64,
                           "Expected to get simplex radi with dtype "
                           "Float64 but got $ when making map",
                           simplex_radi.dtype());
    std::map<double, int64_t> out;
    simplex_radi.arr_void()
        .cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Float64>>>(
            [&out](auto begin, auto end) {
                int64_t c = 0;
                double r = -1.0;
                for (; begin != end; ++begin, ++c) {
                    if (*begin != r) {
                        r = *begin;
                        while (begin != end && *begin == r) {
                            ++begin;
                            ++c;
                        }
                        out[r] = c;
                        if (begin == end)
                            break;
                    }
                }
            });
    return std::move(out);
}

std::map<double, std::array<int64_t, 2>>
make_simplex_radi_map(const Tensor &r1, const Tensor &r2) {
    utils::THROW_EXCEPTION(
        r1.dtype() == r2.dtype(),
        "Expected radi for both tensors r1 ($) and r2 ($) to be the same",
        r1.dtype(), r2.dtype());
    utils::THROW_EXCEPTION(r1.dtype() == DType::Float64,
                           "Expected to get simplex radi with dtype "
                           "Float64 but got $ when making map",
                           r1.dtype());
    std::map<double, int64_t> s1 = make_simplex_radi_map(r1);
    std::map<double, int64_t> s2 = make_simplex_radi_map(r2);
    std::map<double, std::array<int64_t, 2>> out;
    auto begin1 = s1.begin();
    auto begin2 = s2.begin();
    auto end1 = s1.end();
    auto end2 = s2.end();
    // first iteration
    if (begin1->first < begin2->first) {
        out[begin1->first] = {begin1->second, 0};
        ++begin1;
    } else if (begin2->first < begin1->first) {
        out[begin2->first] = {0, begin2->second};
        ++begin2;
    }

    while (begin1 != end1 && begin2 != end2) {
        if (begin1->first == begin2->first) {
            out[begin1->first] = {begin1->second, begin2->second};
            ++begin1;
            ++begin2;
        } else if (begin1->first < begin2->first) {
            out[begin1->first] = {begin1->second, out.rbegin()->second[1]};
            ++begin1;
        } else if (begin2->first < begin1->first) {
            out[begin2->first] = {out.rbegin()->second[0], begin2->second};
            ++begin2;
        }
    }
    return std::move(out);
}

std::map<double, std::array<int64_t, 2>>
make_simplex_radi_map(const Tensor &simplex_radi, int64_t input) {
    utils::THROW_EXCEPTION(simplex_radi.dtype() == DType::Float64,
                           "Expected to get simplex radi with dtype "
                           "Float64 but got $ when making map",
                           simplex_radi.dtype());
    std::map<double, std::array<int64_t, 2>> out;
    simplex_radi.arr_void()
        .cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Float64>>>(
            [&out, &input](auto begin, auto end) {
                int64_t c = 0;
                double r = -1.0;
                for (; begin != end; ++begin, ++c) {
                    if (*begin != r) {
                        r = *begin;
                        while (begin != end && *begin == r) {
                            ++begin;
                            ++c;
                        }
                        out[r] = {c, input};
                        if (begin == end)
                            break;
                    }
                }
            });
    return std::move(out);
}

std::map<double, std::array<int64_t, 2>>
make_simplex_radi_map(int64_t input, const Tensor &simplex_radi) {
    utils::THROW_EXCEPTION(simplex_radi.dtype() == DType::Float64,
                           "Expected to get simplex radi with dtype "
                           "Float64 but got $ when making map",
                           simplex_radi.dtype());
    std::map<double, std::array<int64_t, 2>> out;
    simplex_radi.arr_void()
        .cexecute_function<WRAP_DTYPES<DTypeEnum<DType::Float64>>>(
            [&out, &input](auto begin, auto end) {
                int64_t c = 0;
                double r = -1.0;
                for (; begin != end; ++begin, ++c) {
                    if (*begin != r) {
                        r = *begin;
                        while (begin != end && *begin == r) {
                            ++begin;
                            ++c;
                        }
                        out[r] = {input, c};
                        if (begin == end)
                            break;
                    }
                }
            });
    return std::move(out);
}

SparseTensor boundary_to_radius(std::map<double, std::array<int64_t, 2>> &map,
                                SparseTensor &boundary, double radius) {
    if (radius >= map.rbegin()->first) {
        return boundary;
    }
    auto begin = map.rbegin();
    auto end = map.rend();
    for (; begin != end; ++begin) {
        if (begin->first <= radius)
            break;
    }
    begin = std::prev(begin);
    const int64_t &x_max = begin->second[0];
    const int64_t &y_max = begin->second[1];
    utils::THROW_EXCEPTION(x_max > 0 && y_max > 0,
                           "Too low of radius, got 0 as an index");
    return boundary(range> x_max, range> y_max);
}

SparseTensor boundary_to_radius(const std::array<int64_t, 2> &keys,
                                SparseTensor &boundary) {
    const int64_t &x_max = keys[0];
    const int64_t &y_max = keys[1];
    utils::THROW_EXCEPTION(x_max > 0 && y_max > 0,
                           "Too low of radius, got 0 as an index");
    return boundary(range> x_max, range> y_max);
}

Tensor extract_columns(const Tensor &space, const Tensor &cols) {
    // cols is a dtype bool that tells what columns from space to extract
    std::vector<int64_t> extracting;
    extracting.reserve(space.shape()[-1]);
    int64_t count = 0;
    const uint_bool_t *start =
        reinterpret_cast<const uint_bool_t *>(cols.data_ptr());
    const uint_bool_t *stop =
        reinterpret_cast<const uint_bool_t *>(cols.data_ptr_end());
    for (; start != stop; ++start, ++count) {
        if (!(start->value == 1))
            continue;
        extracting.push_back(count);
    }
    count = extracting.size();
    if (count == 0) {
        return Tensor::Null();
    }
    Tensor out({space.shape()[0], count}, space.dtype());
    float *ptr = reinterpret_cast<float *>(out.data_ptr());

    const int64_t &num_cols = space.shape()[-1];
    const float *begin = reinterpret_cast<const float *>(space.data_ptr());
    const float *end = reinterpret_cast<const float *>(space.data_ptr_end());
    if (count == 1) {
        const int64_t &val = extracting[0];
        for (; begin < end; begin += num_cols, ++ptr) {
            *ptr = begin[val];
        }
        return std::move(out);
    }
    auto col_begin = extracting.cbegin();
    auto col_end = extracting.cend();
    auto col_cpy = col_begin;
    for (; begin < end; begin += num_cols) {
        for (; col_begin != col_end; ++col_begin, ++ptr) {
            *ptr = begin[*col_begin];
        }
    }
    return std::move(out);
}

Tensor extract_columns(const Tensor &space, Tensor &cols, int64_t min) {
    int64_t *col_begin = reinterpret_cast<int64_t *>(cols.data_ptr());
    int64_t *col_end = reinterpret_cast<int64_t *>(cols.data_ptr_end());
    std::for_each(col_begin, col_end, [&min](int64_t &val) { val -= min; });
    col_begin = reinterpret_cast<int64_t *>(cols.data_ptr());
    while (col_begin != col_end && *col_begin < 0) {
        ++col_begin;
    }
    if (col_begin == col_end) {
        return Tensor::Null();
    }
    int64_t count = col_end - col_begin;
    Tensor out({space.shape()[0], count}, space.dtype());
    float *ptr = reinterpret_cast<float *>(out.data_ptr());
    const int64_t &num_cols = space.shape()[-1];
    const float *begin = reinterpret_cast<const float *>(space.data_ptr());
    const float *end = reinterpret_cast<const float *>(space.data_ptr_end());
    if (count == 1) {
        const int64_t &val = col_begin[0];
        for (; begin < end; begin += num_cols, ++ptr) {
            *ptr = begin[val];
        }
        return std::move(out);
    }
    int64_t *col_cpy = col_begin;

    for (; begin < end; begin += num_cols) {
        for (; col_begin != col_end; ++col_begin, ++ptr) {
            // std::cout << "getting "<<begin[*col_begin] << std::endl;
            *ptr = begin[*col_begin];
        }
        col_begin = col_cpy;
    }
    return std::move(out);
}

std::size_t GeneratorHash::operator()(const Tensor &vec) const {
    std::size_t hash = 0;
    const int64_t *begin = reinterpret_cast<const int64_t *>(vec.data_ptr());
    const int64_t *end = begin + vec.numel();
    for (; begin != end; ++begin) {
        hash ^= std::hash<int64_t>{}(*begin) + 0x9e3779b9 + (hash << 6) +
                (hash >> 2);
    }
    return hash;
}

bool GeneratorEqual::operator()(const Tensor &a, const Tensor &b) const {
    if (a.is_null() || b.is_null())
        return false;
    if (a.numel() != b.numel())
        return false;
    std::size_t hash = 0;
    const int64_t *a_begin = reinterpret_cast<const int64_t *>(a.data_ptr());
    const int64_t *a_end = reinterpret_cast<const int64_t *>(a.data_ptr_end());
    const int64_t *b_begin = reinterpret_cast<const int64_t *>(b.data_ptr());
    for (; a_begin != a_end; ++a_begin, ++b_begin) {
        if (*a_begin != *b_begin) {
            return false;
        }
    }
    return true;
}


/*
 * I am keeping the code below in comments because it explicitly points out how to get betti
 * numbers and generators from a boundary matrix, and is good for educational purposes
int64_t PersistentHomology::get_betti_number(SparseTensor &boundary_k,
                                             SparseTensor &boundary_kp1) {
    // A: a tensor representing the transpose of the reduction of boundary_k
    // B: a tensor representing the reduction of boundary_kp1
    auto [A, B] = get<2>(simultaneousReduce(boundary_k, boundary_kp1));
    finishRowReducing(B);
    int64_t dimKChains = boundary_k.shape()[1]; // Number of k-chains
    int64_t rank_k = numPivotRows(A);
    int64_t kernelDim = dimKChains - rank_k; // Nullity of boundary_k
    int64_t rank_kp1 = numPivotRows(B);
    // int64_t imageDim = rank_kp1
    int64_t betti = kernelDim - rank_kp1;
    return betti;
}

std::tuple<int64_t, Tensor>
PersistentHomology::get_generator(SparseTensor &boundary_k,
                                  SparseTensor &boundary_kp1) {
    // A: a tensor representing the transpose of the reduction of boundary_k
    // B: a tensor representing the reduction of boundary_kp1
    auto [A, B] = get<2>(simultaneousReduce(boundary_k, boundary_kp1));
    finishRowReducing(B);
    int64_t dimKChains = boundary_k.shape()[1]; // Number of k-chains
    int64_t rank_k = numPivotRows(A);
    int64_t kernelDim = dimKChains - rank_k; // Nullity of boundary_k
    int64_t rank_kp1 = numPivotRows(B);
    // int64_t imageDim = rank_kp1
    int64_t betti = kernelDim - rank_kp1;
    utils::THROW_EXCEPTION(betti >= 0, "INTERNAL LOGIC ERROR NEGATIVE BETTI");
    if (betti == 0) {
        return std::make_tuple(static_cast<int64_t>(0), Tensor::Null());
    }
    // std::cout << "boundary shapes: "<<boundary_k.shape() << ',' <<
    // boundary_kp1.shape() << std::endl;
    Tensor kernel = linalg::null_space(
        boundary_k.underlying_tensor().to(DType::Float32), "lu");
    if (kernel.is_null()) {
        return std::make_tuple(static_cast<int64_t>(0), Tensor::Null());
    }
    Tensor col_space = linalg::col_space(boundary_kp1.underlying_tensor(), B)
                           .to(DType::Float32);
    Tensor M = functional::cat(col_space, kernel, 1);
    Tensor reduced = rowReduce(M);
    Tensor pivots = linalg::pivot_cols(reduced);
    Tensor generator = extract_columns(kernel, pivots, col_space.shape()[-1]);
    if (generator.is_null()) {
        return std::make_tuple(static_cast<int64_t>(0), Tensor::Null());
    }
    return std::make_tuple(betti, std::move(generator));
}

*/
inline void radius_to_adjacency(double radius, double *begin, double *end,
                                bool *out) {
    std::transform(begin, end, out,
                   [&radius](const double &var) { return var <= radius; });
}

void bfs(int64_t start, bool** adj, std::vector<bool>& visited, 
         std::vector<int64_t>& components, const int64_t& N, int64_t index){
    std::queue<int64_t> q;
    q.push(start);
    visited[start] = true;
    components[start] = index;
    while (!q.empty()) {
        int64_t node = q.front();
        q.pop();

        for (int64_t neighbor = 0; neighbor < N; ++neighbor) {
            if (adj[node][neighbor] && !visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
                components[neighbor] = index;
            }
        }
    }
}

int64_t getConnectedComponents(
        bool** adj, const int64_t& N, std::vector<bool>& visited, std::vector<int64_t>& components){
    // std::vector<std::vector<int64_t>> components;
    int64_t count = 0;
    for (int64_t i = 0; i < N; ++i) {
        if (!visited[i]) {
            bfs(i, adj, visited, components, N, count);
            ++count;
        }
    }
    std::fill(visited.begin(), visited.end(), false);
    return count;
}

void bfs(int64_t start, bool** adj, std::vector<bool>& visited, const int64_t& N){
    std::queue<int64_t> q;
    q.push(start);
    visited[start] = true;
    while (!q.empty()) {
        int64_t node = q.front();
        q.pop();

        for (int64_t neighbor = 0; neighbor < N; ++neighbor) {
            if (adj[node][neighbor] && !visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}


int64_t countConnectedComponents(
        bool** adj, const int64_t& N, std::vector<bool>& visited){
    std::vector<std::vector<int64_t>> components;
    int64_t count = 0;
    for (int64_t i = 0; i < N; ++i) {
        if (!visited[i]) {
            bfs(i, adj, visited, N);
            ++count;
        }
    }
    std::fill(visited.begin(), visited.end(), false);
    return count;
}


void getUniqueRadiVec(std::vector<double>& vec) {
    // Sort the vector
    std::sort(vec.begin(), vec.end());

    // Use std::unique to move duplicates to the end
    auto it = std::unique(vec.begin(), vec.end());

    // Erase the duplicates from the end
    vec.erase(it, vec.end());
    
}

std::map<double, int64_t> PersistentHomology::GetH0BettiNumbers() {
    Tensor distances = functional::sqrt(balls.get_distances(0)) / 2;
    const int64_t &N = distances.shape()[0];
    Tensor adjacency(distances.shape(), DType::Bool);
    adjacency = false;
    double *dist_begin = reinterpret_cast<double *>(distances.data_ptr());
    double *dist_end = reinterpret_cast<double *>(distances.data_ptr_end());
    bool *adjacency_begin = reinterpret_cast<bool *>(adjacency.data_ptr());
    NT_VLA(bool*, adj, N);
    // bool *adj[N];
    for (int64_t i = 0; i < N; ++i, adjacency_begin += N) {
        adj[i] = adjacency_begin;
    }
    adjacency_begin = reinterpret_cast<bool *>(adjacency.data_ptr());
    std::vector<double> independent_distances(dist_begin, dist_end);
    getUniqueRadiVec(independent_distances); 
    dist_begin = reinterpret_cast<double *>(distances.data_ptr());
    dist_end = reinterpret_cast<double *>(distances.data_ptr_end());
    std::vector<bool> visited(N, false);
    std::map<double, int64_t> out;
    out[0.0] = N;
    int64_t last = N;
    for(const auto& r : independent_distances){
        if(r == 0) continue;
        radius_to_adjacency(r, dist_begin, dist_end, adjacency_begin);
        adjacency_begin = reinterpret_cast<bool *>(adjacency.data_ptr());
        dist_begin = reinterpret_cast<double *>(distances.data_ptr());
        dist_end = reinterpret_cast<double *>(distances.data_ptr_end());
        int64_t amt_connected = countConnectedComponents(adj, N, visited);
        if(amt_connected == 1){
            out[r] = amt_connected;
            return out;
        }
        // std::vector<std::vector<int64_t>> connected = getConnectedComponents(adj, N, visited);
        if(amt_connected != last){
            out[r] = amt_connected;
            last = amt_connected;
        }
    }
    NT_VLA_DEALC(adj);
    return out;
    
    // example:
    // radius_to_adjacency(3.4, dis_begin, dist_end, adjacency_begin)
}

std::tuple<std::map<double, int64_t>, std::map<double, Tensor>>
PersistentHomology::GetH0Generators() {
    Tensor distances = functional::sqrt(balls.get_distances(0)) / 2;
    const int64_t &N = distances.shape()[0];
    Tensor adjacency(distances.shape(), DType::Bool);
    adjacency = false;
    double *dist_begin = reinterpret_cast<double *>(distances.data_ptr());
    double *dist_end = reinterpret_cast<double *>(distances.data_ptr_end());
    bool *adjacency_begin = reinterpret_cast<bool *>(adjacency.data_ptr());
    NT_VLA(bool*, adj, N);
    // bool *adj[N];
    for (int64_t i = 0; i < N; ++i, adjacency_begin += N) {
        adj[i] = adjacency_begin;
    }
    adjacency_begin = reinterpret_cast<bool *>(adjacency.data_ptr());
    std::vector<double> independent_distances(dist_begin, dist_end);
    getUniqueRadiVec(independent_distances); 
    dist_begin = reinterpret_cast<double *>(distances.data_ptr());
    dist_end = reinterpret_cast<double *>(distances.data_ptr_end());
    std::vector<bool> visited(N, false);
    std::map<double, int64_t> out;
    std::map<double, Tensor> generators;
    out[0.0] = N;
    generators[0.0] = nt::linalg::eye(N, 0, DType::int8);
    std::vector<int64_t> components(N, 0);
    int64_t last = N;
    bool did_one = false;
    for(const auto& r : independent_distances){
        if(r == 0) continue;
        radius_to_adjacency(r, dist_begin, dist_end, adjacency_begin);
        adjacency_begin = reinterpret_cast<bool *>(adjacency.data_ptr());
        dist_begin = reinterpret_cast<double *>(distances.data_ptr());
        dist_end = reinterpret_cast<double *>(distances.data_ptr_end());
        int64_t amt_connected = getConnectedComponents(adj, N, visited, components);
        if(amt_connected == 0) continue;
        if(amt_connected == 1){
            if(did_one) continue;
            did_one = true;
            generators[r] = functional::ones({N, 1}, DType::int8);
            out[r] = 1;
        }
        else if(amt_connected != last){
            out[r] = amt_connected;
            Tensor gen = functional::zeros({N, amt_connected}, DType::int8);
            int8_t* g_mat = reinterpret_cast<int8_t*>(gen.data_ptr());
            for(int64_t i = 0; i < N; ++i, g_mat += amt_connected){
                g_mat[components[i]] = 1;
            }
            last = amt_connected;
            generators[r] = gen;
        }
    }
    NT_VLA_DEALC(adj);
    return std::make_tuple(std::move(out), std::move(generators));
}

std::map<double,
    std::tuple<int64_t, int64_t, int64_t>
    > construct_boundary_radi_map(
        const std::map<double, std::array<int64_t, 2>>& map_a,
        const std::map<double, std::array<int64_t, 2>>& map_b){
    auto begin_a = map_a.begin();
    auto end_a = map_a.end();
    auto begin_b = map_b.begin();
    auto end_b = map_b.end();
    std::map<double,
        std::tuple<int64_t, int64_t, int64_t>
        > out_map;
    //iterate until none of the indices are 0
    while(begin_a != end_a && (begin_a->second[0] == 0 || begin_a->second[1] == 0)) ++begin_a;
    while(begin_b != end_b && (begin_b->second[0] == 0 || begin_b->second[1] == 0)) ++begin_b;
    if(begin_a == end_a || begin_b == end_b){
        return out_map;
    }
    for(;begin_a != end_a; ++begin_a){
        while(begin_b != end_b && begin_b->second[0] < begin_a->second[1]) ++begin_b;
        if(begin_a->second[1] < begin_b->second[0]) continue;
        while(begin_b != end_b && begin_b->second[0] == begin_a->second[1] ){
            out_map[begin_b->first] = 
                std::make_tuple(begin_a->second[0], begin_a->second[1], begin_b->second[1]);
            ++begin_b;
        }
    }
    return out_map;
}

std::map<double,
        std::tuple<int64_t, int64_t, int64_t> > construct_boundary_radi_map(
                        const std::map<double, std::array<int64_t, 2>>& map_a,
                        const std::map<double, std::array<int64_t, 2>>& map_b,
                        const std::set<double>& radii){
    std::map<double,
        std::tuple<int64_t, int64_t, int64_t> > out_map;
    // std::cout << "maps: "<<std::endl;
    // for(const auto [r, arr] : map_a){
    //     std::cout << r<<": {"<<arr[0]<<','<<arr[1]<<"} ";
    // }
    // std::cout << std::endl;
    // for(const auto [r, arr] : map_b){
    //     std::cout << r<<": {"<<arr[0]<<','<<arr[1]<<"} ";
    // }
    // std::cout << std::endl;
    // std::cout << "set: ";
    // for(const auto& r : radii){
    //     std::cout << r << ' ';
    // }
    // std::cout << std::endl;

    for(const double& r : radii){
        // std::cout << "getting radius "<<r<<std::endl;
        auto _m1 = map_a.find(r);
        auto _m2 = map_b.find(r);
        if(_m1 == map_a.end() || _m2 == map_b.end())
            continue;
        const std::array<int64_t, 2> &M1 = _m1->second;
        const std::array<int64_t, 2> &M2 = _m2->second;
        if (M1[0] == 0 || M1[1] == 0 || M2[0] == 0 || M2[1] == 0)
            continue;
        //M1[1] == M2[0]
        out_map[r] = std::make_tuple(M1[0], M1[1], M2[1]);
    }
    return std::move(out_map);
}

std::map<double, int64_t> PersistentHomology::GetHKBettiNumbers(int64_t k,
                                                                double max) {
    std::map<double,
        std::tuple<int64_t, int64_t, int64_t> > boundary_map = 
            construct_boundary_radi_map(SigmaMaps[k-1],
                                        SigmaMaps[k],
                                        Radii[k]);

    std::map<double, int64_t> betti_numbers = ::nt::tda::cpu::getBettiNumbers(BoundaryMatricies[k-1], BoundaryMatricies[k], boundary_map, max, true);
    return std::move(betti_numbers);
    // for (const auto &r : Radii[k]) {
    //     if (r > max)
    //         break;

    //     const std::array<int64_t, 2> &M1 = SigmaMaps[k - 1][r];
    //     const std::array<int64_t, 2> &M2 = SigmaMaps[k][r];
    //     if (M1[0] == 0 || M1[1] == 0 || M2[0] == 0 || M2[1] == 0)
    //         continue;
    //     // last = r;
    //     SparseTensor s_1 = boundary_to_radius(M1, BoundaryMatricies[k - 1]);
    //     SparseTensor s_2 = boundary_to_radius(M2, BoundaryMatricies[k]);
    //     int64_t betti_number = get_betti_number(s_1, s_2);
    //     betti_numbers[r] = betti_number;
    // }
    // return std::move(betti_numbers);
}

std::map<double, int64_t> PersistentHomology::GetHKBettiNumbers(int64_t k) {
    return GetHKBettiNumbers(k, -1.0);
}

std::tuple<std::map<double, int64_t>, std::map<double, Tensor>>
PersistentHomology::GetHKGenerators(int64_t k, double max) {
    std::map<double,
        std::tuple<int64_t, int64_t, int64_t> > boundary_map = 
            construct_boundary_radi_map(SigmaMaps[k-1],
                                        SigmaMaps[k],
                                        Radii[k]);
    
    auto [betti_numbers, col_spaces] = ::nt::tda::cpu::getBettiNumbersColSpace(BoundaryMatricies[k-1], BoundaryMatricies[k], boundary_map, max, true);
    std::map<double, Tensor> generators;
    for(const auto& val : col_spaces){
        double radius = val.first;
        Tensor col_space = Tensor(val.second).to(DType::Float32);
        auto [km1_size, k_size, kp1_size] = boundary_map[radius];

        SparseMatrix boundary_k = BoundaryMatricies[k-1].block(0, km1_size, 0, k_size);
        Tensor kernel = linalg::null_space(
            Tensor(boundary_k).to(DType::Float32), "lu");
        if (kernel.is_null()) {
            continue;
        }

        Tensor M = functional::cat(col_space, kernel, 1);
        Tensor reduced = rowReduce(M);
        Tensor pivots = linalg::pivot_cols(reduced);
        Tensor generator = extract_columns(kernel, pivots, col_space.shape()[-1]);
        generators[radius] = generator;

    }
    return std::make_tuple(betti_numbers, generators);
    // for (const auto &r : Radii[k]) {
    //     if (r > max)
    //         break;
    //     const std::array<int64_t, 2> &M1 = SigmaMaps[k - 1][r];
    //     const std::array<int64_t, 2> &M2 = SigmaMaps[k][r];
    //     if (M1[0] == 0 || M1[1] == 0 || M2[0] == 0 || M2[1] == 0)
    //         continue;

    //     // last = r;
    //     SparseTensor s_1 = boundary_to_radius(M1, BoundaryMatricies[k - 1]);
    //     SparseTensor s_2 = boundary_to_radius(M2, BoundaryMatricies[k]);
    //     auto [betti_number, generator] = get_generator(s_1, s_2);
    //     betti_numbers[r] = betti_number;
    //     if (!generator.is_null() && betti_number != 0) {
    //         generators[r] = std::move(generator);
    //     }
    // }
    // return std::make_tuple(std::move(betti_numbers), std::move(generators));
}

std::tuple<std::map<double, int64_t>, std::map<double, Tensor>>
PersistentHomology::GetHKGenerators(int64_t k) {
    return GetHKGenerators(k, -1.0);
}

double PersistentHomology::get_next_radius(double radius, int64_t k) {
    std::set<double> &radii = Radii[k];
    auto begin = radii.begin();
    auto end = radii.end();
    for (; begin != end; ++begin) {
        if (*begin > radius) {
            return *begin;
        }
    }
    return -1.0;
}

PersistentHomology::PersistentHomology(Tensor _pts)
    : balls(_pts), points(_pts) {
    if (balls.is_batched()) {
        std::cout << "This construct has gotten batched graphs to process, "
                     "but this is not a BatchedConstruct class, only a "
                     "Construct class, will only process the first graph"
                  << std::endl;
    }
}

void PersistentHomology::constructGroups(int64_t max_homologies) {
    if (max_homologies == -1) {
        max_homologies = this->dims();
    }
    SimplexConstruct.clear();
    SimplexRadi.clear();
    Radii.clear();
    SimplexConstruct.reserve(max_homologies + 1);
    SimplexRadi.reserve(max_homologies + 1);
    Radii.reserve(max_homologies);
    for (int64_t i = 0; i < max_homologies + 1; ++i) {
        auto [aSimplexK, RKs] =
            get<2>(find_all_simplicies(i + 1, points, balls, true));
        SimplexConstruct.emplace_back(std::move(aSimplexK));
        std::set<double> rSimplexK = get_radi_set(RKs);
        SimplexRadi.emplace_back(std::move(RKs));
        Radii.emplace_back(std::move(rSimplexK));
    }

    // now construct the assocated boundary matricies
    BoundaryMatricies.clear();
    BoundaryMatricies.reserve(max_homologies);
    SigmaMaps.clear();
    SigmaMaps.reserve(max_homologies);
    for (int64_t i = 1; i < max_homologies + 1; ++i) {
        // returns boundary matrix mapping simplex_(i-1) (rows) -> simplex_i
        // (cols
        SparseMatrix Boundary =
            compute_boundary_sparse_matrix_index(SimplexConstruct[i][0].item<Tensor>(),
                                    SimplexConstruct[i - 1][0].item<Tensor>());
        BoundaryMatricies.emplace_back(std::move(Boundary));
        if (i == 1) {
            std::map<double, std::array<int64_t, 2>> sigma_map =
                make_simplex_radi_map(
                    SimplexConstruct[0][0].item<Tensor>().shape()[0],
                    SimplexRadi[1][0].item<Tensor>());
            SigmaMaps.emplace_back(std::move(sigma_map));
            continue;
        }
        std::map<double, std::array<int64_t, 2>> sigma_map =
            make_simplex_radi_map(SimplexRadi[i - 1][0].item<Tensor>(),
                                  SimplexRadi[i][0].item<Tensor>());
        SigmaMaps.emplace_back(std::move(sigma_map));
    }
}

void PersistentHomology::applyMaxRadius(double r) {
    for (auto &Set : Radii) {
        for (auto it = Set.begin(); it != Set.end();) {
            if (*it > r) {
                it = Set.erase(it);
            } else {
                ++it;
            }
        }
    }

    for (auto &Map : SigmaMaps) {
        for (auto it = Map.begin(); it != Map.end();) {
            if (it->first > r) {
                it = Map.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void PersistentHomology::findHomology(int64_t k) {
    if (k == -1) {
        k = this->dims();
    }
    utils::THROW_EXCEPTION(
        SimplexRadi.size() >= k,
        "Have not constructed simplex complexes and groups up to $", k);
    utils::THROW_EXCEPTION(
        k >= 0,
        "Cannot construct less than 0 homology groups but asked to "
        "construct $ homology groups",
        k);
    BettiNumbers.clear();
    BettiNumbers.reserve(k);
    auto [H0BettiNumbers, H0Generators] = GetH0Generators();
    BettiNumbers.emplace_back(std::move(H0BettiNumbers));
    Generators.clear();
    Generators.reserve(k);
    Generators.emplace_back(std::move(H0Generators));
    double max_radi = -1.0;
    // the above is the last radi when B0 is 0
    // max_radi *= max_radi; // to account for radi instead of diameter squared
    for (int64_t i = 1; i < k; ++i) {
        auto [HkBettiNumbers, HkGenerators] = GetHKGenerators(i, max_radi);
        BettiNumbers.emplace_back(std::move(HkBettiNumbers));
        Generators.emplace_back(std::move(HkGenerators));
        auto begin = BettiNumbers.back().rbegin();
        auto end = BettiNumbers.back().rend();
        for (; begin != end; ++begin) {
            if (begin->second > 0) {
                max_radi = begin->first;
                break;
            }
        }
    }
}

std::unordered_map<Tensor, std::vector<std::pair<double, int64_t>>,
                   GeneratorHash, GeneratorEqual>
PersistentHomology::getGeneratorMap(int64_t k) {
    std::map<double, Tensor> &r_map = this->Generators.at(k);
    std::unordered_map<Tensor, std::vector<std::pair<double, int64_t>>,
                       GeneratorHash, GeneratorEqual>
        o_map;
    for (const auto [radius, generator] : r_map) {
        Tensor gen = generator.shape()[-1] == 1 ? generator.view(-1)
                                                : generator.transpose(-1, -2);
        if (gen.dims() == 1) {
            Tensor coords = functional::where(gen != 0).item<Tensor>();
            auto [it, inserted] = o_map.insert(
                {coords,
                 std::vector<std::pair<double, int64_t>>({{radius, 0}})});
            if (!inserted) {
                auto &vec = it->second;
                vec.push_back({radius, 0});
            }
        } else {
            Tensor split = gen.split_axis(-2);
            int64_t cntr = 0;
            for (const auto t : split) {
                Tensor coords = functional::where(t != 0).item<Tensor>();
                auto [it, inserted] = o_map.insert(
                    {coords, std::vector<std::pair<double, int64_t>>(
                                 {{radius, cntr}})});
                if (!inserted) {
                    auto &vec = it->second;
                    vec.push_back({radius, cntr});
                }
                ++cntr;
            }
        }
    }
    return std::move(o_map);
}

std::vector<std::tuple<Tensor, double, double>>
PersistentHomology::getHomologyGroup(int64_t k) {
    if (k == 0) {
        std::vector<std::tuple<Tensor, double, double>> out;
        std::map<double, Tensor> &r_map = this->Generators.at(0);
        out.reserve(r_map.size());
        auto begin = r_map.begin();
        auto end = r_map.end();
        while (begin != end){
            auto prev = begin++;
            if(begin == end)
                break;
            Tensor split = prev->second.split_axis(-1);
            for(const auto t : split){
                Tensor coords = functional::where(t != 0).item<Tensor>();
                out.emplace_back(std::move(coords), 0.0, begin->first);
            }
        }
        auto last = r_map.rbegin();
        out.emplace_back(functional::arange(r_map[0.0].shape()[0], DType::int64), 0.0, -1.0);
        return std::move(out);
    }
    std::vector<std::tuple<Tensor, double, double>> out;
    auto map = this->getGeneratorMap(k);
    out.reserve(map.size());
    for (auto [coords, vec] : map) {
        out.emplace_back(coords, vec[0].first,
                         this->get_next_radius(vec.back().first, k));
    }
    return std::move(out);
}

std::vector<std::vector<std::tuple<Tensor, double, double>>>
PersistentHomology::getHomologyGroups() {
    size_t k = BettiNumbers.size();
    std::vector<std::vector<std::tuple<Tensor, double, double>>> out;
    out.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        out.push_back(this->getHomologyGroup(i));
    }
    return std::move(out);
}

Tensor PersistentHomology::generatorToSimplexComplex(Tensor generator,
                                                     int64_t k,
                                                     bool to_points) {
    if (!functional::any(generator > 1)) {
        if (generator.numel() != 2) {
            generator = functional::where(generator != 0);
        }
    }
    // std::cout <<
    if (!to_points) {
        return SimplexConstruct[k][0].item<Tensor>()[generator];
    }
    return from_index_simplex_to_point_simplex(
        SimplexConstruct[k][0].item<Tensor>()[generator].contiguous(),
        this->points[0].item<Tensor>());
}

void PersistentHomology::add_weight(Tensor weight){
    this->balls.add_weight(weight);
}

} // namespace tda
} // namespace nt
