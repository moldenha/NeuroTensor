#ifndef _NT_TDA_PERSISTENT_HOMOLOGY_H_
#define _NT_TDA_PERSISTENT_HOMOLOGY_H_

#include "../Tensor.h"
#include "../sparse/SparseTensor.h"
#include "BasisOverlapping.h"
#include "PlotDiagrams.h"
#include "Points.h"
#include <map>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace nt {
namespace tda {

struct GeneratorHash {
    std::size_t operator()(const Tensor &vec) const;
};

struct GeneratorEqual {
    bool operator()(const Tensor &a, const Tensor &b) const;
};

std::map<double, int64_t> make_simplex_radi_map(const Tensor &simplex_radi);
std::map<double, std::array<int64_t, 2>>
    make_simplex_radi_map(const Tensor &r1, const Tensor &r2);
std::map<double, std::array<int64_t, 2>>
    make_simplex_radi_map(const Tensor &simplex_radi, int64_t input);
std::map<double, std::array<int64_t, 2>>
    make_simplex_radi_map(int64_t input, const Tensor &simplex_radi);

class PersistentHomology {
    BasisOverlapping balls;
    Tensor points;
    // boundary matricies are in terms of indexes
    // index 0: sigma_0 which maps simplex_0->simplex_1,
    // index 1: sigma_1 which makes simplex_1->simplicies_2
    std::vector<SparseTensor> BoundaryMatricies;
    // radi associated with each level of simplex
    // for example the radi associated with 2_simplexes are Radii[2]
    std::vector<std::set<double>> Radii;
    // index 0: sigma_0 which maps simplex_0->simplex_1,
    // index 1: sigma_1 which makes simplex_1->simplicies_2
    std::vector<std::map<double, std::array<int64_t, 2>>> SigmaMaps;

    // betti numbers associated with Hi, and radi
    // for example
    // H1 Betti numbers at radius 5.6 is:
    // BettiNumbers[1][5.6]
    // granted as long as 5.6 is a key
    // which can be checked by deciding if 5.6 is is Radii[1]
    std::vector<std::map<double, int64_t>> BettiNumbers;
    std::vector<std::map<double, Tensor>> Generators;

    // simplex_0 is SimplexConstruct[0]
    // and the associated radi are SimplexRadi[0]
    std::vector<Tensor> SimplexConstruct;
    std::vector<Tensor> SimplexRadi;
    int64_t get_betti_number(SparseTensor &boundary_k,
                             SparseTensor &boundary_kp1);
    // returns the betti number and the generator
    // if the betti number is 0, then the generator is null
    // this is because there are 0 kth dimensional holes
    std::tuple<int64_t, Tensor>
    get_generator(SparseTensor &boundary_k, SparseTensor &boundary_kp1);

    std::map<double, int64_t> GetH0BettiNumbers();
    std::tuple<std::map<double, int64_t>, std::map<double, Tensor>>
    GetH0Generators();
    std::map<double, int64_t> GetHKBettiNumbers(int64_t k, double max);
    std::map<double, int64_t> GetHKBettiNumbers(int64_t k);
    std::tuple<std::map<double, int64_t>, std::map<double, Tensor>>
    GetHKGenerators(int64_t k, double max);
    std::tuple<std::map<double, int64_t>, std::map<double, Tensor>>
    GetHKGenerators(int64_t k);
    double get_next_radius(double radius, int64_t k);

  public:
    PersistentHomology() = delete;
    PersistentHomology(Tensor _pts);
    inline int64_t dims() { return points.item<Tensor>().shape()[-1]; }
    
    void add_weight(Tensor weight);
    
    // constructs simplexes and boundary matricies
    // and associated radi with each simplex
    void constructGroups(int64_t max_homologies = -1);
    void applyMaxRadius(double r);
    // this constructs the betti numbers up to a certain k
    void findHomology(int64_t k = -1);

    inline const std::vector<std::map<double, int64_t>> &
    getBettiNumbers() const noexcept {
        return this->BettiNumbers;
    }

    inline const std::vector<std::map<double, Tensor>> &
    getGenerators() const noexcept {
        return this->Generators;
    }

    // each vector will already be sorted by radius because of the use of
    // std::map
    std::unordered_map<Tensor, std::vector<std::pair<double, int64_t>>,
                       GeneratorHash, GeneratorEqual>
    getGeneratorMap(int64_t k);

    // this gets the simplex-complex indexes, the birth, and the death
    std::vector<std::tuple<Tensor, double, double>>
    getHomologyGroup(int64_t k);

    std::vector<std::vector<std::tuple<Tensor, double, double>>>
    getHomologyGroups();
    Tensor generatorToSimplexComplex(Tensor generator, int64_t k,
                                         bool to_points = true);

    inline static PersistentHomology
    FromPointCloud(Tensor &cloud, int8_t point = 1, int64_t dims = -1) {
        dims = dims == -1 ? cloud.dims() : dims;
        return PersistentHomology(
            extract_points_from_cloud(cloud, point, dims));
    }
};

} // namespace tda
} // namespace nt

#endif
