#include "../src/tda/Shapes.h"
#include "../src/Tensor.h"
#include "../src/functional/functional.h"
#include "../src/images/image.h"
#include "../src/tda/KDTree.h"
#include "../src/tda/BatchPoints.h"
#include "../src/tda/BatchBasis.h"
#include "../src/convert/std_convert.h"
#include "matplot/matplot.h" //visualization of the graph
#include <string>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string_view>
#include <filesystem>
#include <map>
#include <set>
#include <cmath>
#include <cstdlib>



std::vector<std::pair<nt::tda::Simplex, int>> boundary_operator(const nt::tda::Simplex& simplex){
    std::vector<std::pair<nt::tda::Simplex, int>> boundary;

    // Boundary of an n-simplex is a sum of its (n-1)-faces
    int64_t n = simplex.size(); // Number of vertices
    int64_t k = simplex.dims(); // Dimension of the space
    boundary.reserve(n);


    for (int64_t i = 0; i < n; ++i) {
        // Create a new (n-1)-simplex by omitting the i-th vertex
        nt::tda::Simplex face(n - 1, k);

        // Copy all vertices except the i-th
        auto face_iter = face.begin();
        for (int64_t j = 0; j < n; ++j) {
            if (j == i) continue; // Skip the i-th vertex
            *face_iter = simplex[j];
            ++face_iter;
        }

        // Determine the orientation (+1 or -1)
        int orientation = (i % 2 == 0) ? 1 : -1;

        // Add the face to the boundary
        boundary.emplace_back(std::make_pair(face, orientation));
    }

    return boundary;
}

nt::Tensor boundary_operator(const nt::Tensor& simplicies){
    nt::utils::throw_exception(simplicies.dtype == nt::DType::int64, "Expected to get simplicies of dtype int64 but got $", simplicies.dtype);
    nt::utils::throw_exception(simplicies.dims() == 3, "Expected simplicies to have dims of 3 (b, N, D) but got $", simplicies.dims());
    // Boundary of an n-simplex is a sum of its (n-1)-faces
    int64_t N = simplicies.shape()[-2]; // Number of vertices
    int64_t D = simplicies.shape()[-1]; // Dimension of the space
    nt::Tensor input = simplicies.transpose(0, 1);

    nt::Tensor indices = nt::functional::arange(N, nt::DType::int64, 0);
    std::vector<nt::Tensor> faces;
    faces.reserve(N);
    
    for(int64_t i = 0; i < N; ++i){
        nt::Tensor mask = indices != i;
        faces.emplace_back(input[mask].transpose(0,1)); // shape (b, N-1, D)
    }
    
    //stack results and apply alternating signs
    nt::Tensor boundary = nt::functional::stack(std::move(faces), 1).to(nt::DType::Float32); // shape (b, N, N-1, D)
    nt::Tensor signs = nt::functional::arange(N, nt::DType::Float32).view(1, N, 1, 1).inverse_();  //Shape (1, N, 1, 1)
    
    return boundary * signs; // shape (b, N, N-1, D)
}


//creates a sparse matrix representing bounds as a boundary operator
nt::Tensor sparse_boundary_operator(const nt::Tensor& simplicies){
    nt::utils::throw_exception(simplicies.dtype == nt::DType::int64, "Expected to get simplicies of dtype int64 but got $", simplicies.dtype);
    nt::utils::throw_exception(simplicies.dims() == 3, "Expected simplicies to have dims of 3 (b, N, D) but got $", simplicies.dims());
    nt::utils::throw_exception(simplicies.is_contiguous(), "Expected to perform sparse boundary operator on a contiguous tensor");
    // Boundary of an n-simplex is a sum of its (n-1)-faces
    int64_t N = simplicies.shape()[-2]; // Number of vertices
    int64_t D = simplicies.shape()[-1]; // Dimension of the space
    int64_t b = simplicies.shape()[0];

    nt::Tensor out = nt::functional::zeros({b * N, b}, nt::DType::int64); // sparse matrix of shape (b * N, b)
    int64_t* arr = reinterpret_cast<int64_t*>(out.data_ptr());


    for(int64_t simplex_idx = 0; simplex_idx < b; ++simplex_idx){
        for(int64_t vertex_idx = 0; vertex_idx < N; ++vertex_idx){
            int64_t row = simplex_idx * N + vertex_idx;  //unique row index per (N-1)-simplex
            int64_t col = simplex_idx;  //column index refers to original simplex
            int64_t sign = (vertex_idx % 2 == 0) ? 1 : -1;  // alternating signs
            arr[(row * b) + col] = sign;
        }
    }

    return std::move(out);
}

bool verify_chain_complex(const nt::tda::Simplex& simplex) {
    using BoundaryResult = std::vector<std::pair<nt::tda::Simplex, int>>;

    // Compute the first boundary
    BoundaryResult first_boundary = boundary_operator(simplex);

    // Compute the second boundary: apply the boundary operator to each face in the first boundary
    std::map<nt::tda::Simplex, int> second_boundary_map;

    for (const auto& [face, orientation] : first_boundary) {
        // Compute the boundary of the current face
        BoundaryResult second_boundary_faces = boundary_operator(face);

        for (const auto& [subface, sub_orientation] : second_boundary_faces) {
            // Add the subface to the second boundary, accounting for orientation
            int combined_orientation = orientation * sub_orientation;
            if (second_boundary_map.count(subface)) {
                second_boundary_map[subface] += combined_orientation;
            } else {
                second_boundary_map[subface] = combined_orientation;
            }
        }
    }

    // Verify if the second boundary is zero (all coefficients should cancel out)
    for (const auto& [subface, coefficient] : second_boundary_map) {
        if (coefficient != 0) {
            return false; // The chain complex property is violated
        }
    }

    return true; // The chain complex property holds
}


class ChainComplex {
public:
    // Chain groups: each dimension has a collection of simplices
    std::map<int, std::unordered_set<nt::tda::Simplex>> chain_groups;

    // Boundary maps: maps from k-simplices to (k-1)-simplices
    std::map<int, std::unordered_map<nt::tda::Simplex, std::vector<std::pair<nt::tda::Simplex, int>>>> boundary_maps;
    
    ChainComplex() = default;

    // Add a simplex to the chain complex
    void add_simplex(const nt::tda::Simplex& simplex) {
        int64_t dim = simplex.dims();
        chain_groups[dim].insert(simplex);
    }

    //Add a list of simplexes of a single dimension
    void add_simplexes(const nt::tda::Simplexes& simplexes){
        int64_t dim = simplexes.simplex_dims();
        chain_groups[dim].reserve(simplexes.size());
        for(const auto& simplex : simplexes){
            chain_groups[dim].insert(simplex);
        }
    }


    // Assemble the chain complex by computing boundary maps
    void assemble() {
        for (const auto& [dim, simplices] : chain_groups) {
            if (dim == 1) continue; // 0-simplices don't have boundaries
            for (const nt::tda::Simplex& simplex : simplices) {
                // Compute the boundary of the current simplex
                auto boundary = boundary_operator(simplex);
                // Store the boundary in the boundary map
                for (const auto& [face, orientation] : boundary) {
                    boundary_maps[dim][simplex].emplace_back(face, orientation);
                }
            }
        }
    }

    bool all_zero(){
        for (const auto& [dim, simplices] : chain_groups) {
            if(simplices.size() != 0){return false;}
        }
        return true;
    }

    const std::unordered_set<nt::tda::Simplex>& getChains(int k) const {
        auto out = chain_groups.find(k);
        nt::utils::throw_exception(out != chain_groups.cend(), "cannot get a chain group that does not exist");
        return out->second;
    }

    std::unordered_set<nt::tda::Simplex> get_chains(int k) const {
        auto out = chain_groups.find(k);
        if(out == chain_groups.end())
            return {};
        return out->second;
    }



    // Verify the chain complex property
    bool verify_chain_complex() {
        for (const auto& [dim, boundary_map] : boundary_maps) {
            if (dim == 1) continue; // No boundary map for 0-simplices

            for (const auto& [simplex, faces] : boundary_map) {
                std::unordered_map<nt::tda::Simplex, int> second_boundary_map;

                // Compute the second boundary
                for (const auto& [face, orientation] : faces) {
                    auto second_boundary = boundary_maps[dim - 1][face];

                    for (const auto& [subface, sub_orientation] : second_boundary) {
                        int combined_orientation = orientation * sub_orientation;
                        second_boundary_map[subface] += combined_orientation;
                    }
                }

                // Check if all coefficients cancel out
                for (const auto& [subface, coefficient] : second_boundary_map) {
                    if (coefficient != 0) {
                        return false; // Chain complex property violated
                    }
                }
            }
        }
        return true;
    }

    // Print the chain complex structure
    void print() const {
        for (const auto& [dim, simplices] : chain_groups) {
            if(simplices.size() == 1){continue;}
            std::cout << "C_" << dim << ": ";
            for (const auto& simplex : simplices) {
                std::cout << simplex << " ";
            }
            std::cout << std::endl;
        }
    }

    // Retrieve the boundary of a given simplex
    std::vector<std::pair<nt::tda::Simplex, int>> boundary(const nt::tda::Simplex& simplex, int k) const {
        // int k = simplex.dims(); // Get the dimension of the simplex

        // Check if the boundary map for dimension k exists
        auto it = boundary_maps.find(k);
        if (it != boundary_maps.end()) {
            // Check if the simplex exists in the boundary map
            auto simplex_it = it->second.find(simplex);
            if (simplex_it != it->second.end()) {
                // Return the boundary (vector of face-orientation pairs)
                return simplex_it->second;
            }
        }

        // If no boundary is found, return an empty vector
        return {};
    }


};

struct Homology{
    std::vector<nt::tda::Simplex> representativeCycles;
    Homology() = default;
    Homology(const Homology& h)
    :representativeCycles(h.representativeCycles)
    {}
    Homology(const nt::tda::Simplex& simp)
    :representativeCycles({simp})
    {}
    Homology(const nt::tda::Point& pt)
    :representativeCycles({nt::tda::Simplex(1, pt.dims())})
    {representativeCycles[0][0] = pt;}

    inline bool operator==(const Homology& h) const noexcept {
        if(representativeCycles.size() != h.representativeCycles.size()){return false;}
        for(size_t i = 0; i < this->representativeCycles.size(); ++i){
            if(this->representativeCycles[i] != h.representativeCycles[i]){return false;}
        }
        return true;
    }
    inline bool operator!=(const Homology& h) const noexcept {
        if(representativeCycles.size() != h.representativeCycles.size()){return true;}
        for(size_t i = 0; i < this->representativeCycles.size(); ++i){
            if(this->representativeCycles[i] == h.representativeCycles[i]){return false;}
        }
        return true;
    }
    // inline bool operator<(const Homology& h) const noexcept {return representativeCycles < h.representativeCycles;}
    std::vector<nt::tda::Simplex>::iterator find(const nt::tda::Simplex& simplex) {
        auto begin = representativeCycles.begin();
        auto end = representativeCycles.end();
        for(;begin != end; ++begin){
            if(*begin == simplex){return begin;}
        }
        return end;
    }
    std::vector<nt::tda::Simplex>::const_iterator find(const nt::tda::Simplex& simplex) const {
        auto begin = representativeCycles.cbegin();
        auto end = representativeCycles.cend();
        for(;begin != end; ++begin){
            if(*begin == simplex){return begin;}
        }
        return end;
    }
    std::vector<nt::tda::Simplex>::iterator begin() {return representativeCycles.begin();} 
    std::vector<nt::tda::Simplex>::iterator end() {return representativeCycles.end();}
    std::vector<nt::tda::Simplex>::const_iterator begin() const {return representativeCycles.begin();} 
    std::vector<nt::tda::Simplex>::const_iterator end() const {return representativeCycles.end();}
    std::vector<nt::tda::Simplex>::const_iterator cbegin() const {return representativeCycles.cbegin();} 
    std::vector<nt::tda::Simplex>::const_iterator cend() const {return representativeCycles.cend();}
    void reserve(size_t i){
        representativeCycles.reserve(i);
    }
    void push_back(nt::tda::Simplex &&simplex){
        representativeCycles.push_back(std::forward<nt::tda::Simplex>(simplex));
    }
    void push_back(const nt::tda::Simplex &simplex){
        representativeCycles.push_back(simplex);
    }
    void emplace_back(const nt::tda::Simplex &simplex){
        representativeCycles.emplace_back(simplex);
    }

    nt::tda::Simplex& operator[](size_t i) noexcept {return representativeCycles[i];}
    const nt::tda::Simplex& operator[](size_t i) const noexcept {return representativeCycles[i];}
    nt::tda::Simplex& at(size_t i){return representativeCycles.at(i);}
    const nt::tda::Simplex& at(size_t i) const {return representativeCycles.at(i);}
    nt::tda::Simplex& back() {return representativeCycles.back();}
    const nt::tda::Simplex& back() const {return representativeCycles.back();}
    Homology& insert(std::unordered_set<nt::tda::Simplex>& simplexes){
        this->reserve(simplexes.size());
        for(const auto& simplex : simplexes)
            this->emplace_back(simplex);
        return *this;
    }
    
    inline int64_t size() const {return static_cast<int64_t>(representativeCycles.size());}
    //this is how many points are in each simplex
    inline size_t simplex_size() const noexcept {return representativeCycles.size() == 0 ? 0 : representativeCycles[0].size();}
    //this is for example a 0-simplex which holds 1 point
    inline size_t simplex_dims() const noexcept {return representativeCycles.size() == 0 ? 0 : representativeCycles[0].size()-1;}
    //this is the dimension of the point, so exactly how many coordinates
    inline size_t point_dims() const noexcept {return representativeCycles.size() == 0 ? 0 : representativeCycles[0].dims();}
    

};

//this is just used to hold homologies and corresponding births and deaths
struct HomologyMap{
    std::vector<Homology> homologies;
    std::vector<std::pair<double, double>> logs;
    class iterator{
        public:
            explicit iterator(std::vector<Homology>::iterator h_it, std::vector<std::pair<double, double>>::iterator l_it) : _h(h_it), _l(l_it) {}
            inline const std::pair<Homology&, std::pair<double, double>&> operator*() {return {*_h, *_l};}
            inline const iterator& operator++() {++_h; ++_l; return *this;}
            inline bool operator!=(const iterator& other) const {return _h != other._h;}
            inline bool operator==(const iterator& other) const {return _h == other._h;}
        private:
            std::vector<Homology>::iterator _h;
            std::vector<std::pair<double, double>>::iterator _l;
    };
    class const_iterator{
        public:
            explicit const_iterator(std::vector<Homology>::const_iterator h_it, std::vector<std::pair<double, double>>::const_iterator l_it) : _h(h_it), _l(l_it) {}
            inline const std::pair<const Homology&, const std::pair<double, double>&> operator*() {return {*_h, *_l};}
            inline const const_iterator& operator++() {++_h; ++_l; return *this;}
            inline bool operator!=(const const_iterator& other) const {return _h != other._h;}
            inline bool operator==(const const_iterator& other) const {return _h == other._h;}
        private:
            std::vector<Homology>::const_iterator _h;
            std::vector<std::pair<double, double>>::const_iterator _l;
    };
    inline int64_t size() const {return homologies.size();}
    inline void add(const Homology& homology, std::pair<double, double> log){
        homologies.emplace_back(homology);
        logs.emplace_back(log);
    }
    inline void add(const nt::tda::Simplex simp, std::pair<double, double> log){
        homologies.emplace_back(simp);
        logs.emplace_back(log);
    }
    inline void add(const nt::tda::Point pt, std::pair<double, double> log){
        homologies.emplace_back(pt);
        logs.emplace_back(log);
    }
    inline std::pair<double, double>& operator[](const Homology& homology){
        for(int64_t i = 0; i < homologies.size(); ++i){
            if(homologies[i] == homology){return logs[i];}
        }
        this->add(homology, {0,0});
        return logs.back();
    }
    iterator begin() {
        return iterator(homologies.begin(), logs.begin());
    }
    iterator end() {
        return iterator(homologies.end(), logs.end());
    }
    const_iterator begin() const {
        return const_iterator(homologies.begin(), logs.begin());
    }
    const_iterator end() const {
        return const_iterator(homologies.end(), logs.end());
    }
    const_iterator cbegin() const {
        return const_iterator(homologies.cbegin(), logs.cbegin());
    }
    const_iterator cend() const {
        return const_iterator(homologies.cend(), logs.cend());
    }

    iterator find(const Homology& h){
        for(auto b = begin(); b != end(); ++b){
            if((*b).first == h){return b;}
        }
        return end();
    }
    const_iterator find(const Homology& h) const {
        for(auto b = begin(); b != end(); ++b){
            if((*b).first == h){return b;}
        }
        return end();
    }

};

std::ostream& operator<<(std::ostream& os, const Homology& h){
    os << "[";
    for(int64_t i = 0; i < h.size()-1; ++i){
        os << h[i] << ',';
    }
    if(h.size() == 0){return os << ']';}
    return os << h.back() << ']';
}

class Homologies {
public:
    // Compute H_k = ker(∂_k) / im(∂_{k+1})
    static Homology computeHomology(const ChainComplex& chainComplex, int k) {
        // Get the chain groups and boundary operators
        std::unordered_set<nt::tda::Simplex> kChains = chainComplex.get_chains(k);
        std::unordered_set<nt::tda::Simplex> kPlus1Chains = chainComplex.get_chains(k + 1);

        // Compute ker(∂_k)
        std::unordered_set<nt::tda::Simplex> kernel = computeKernel(chainComplex, k, kChains);

        // Compute im(∂_{k+1})
        std::unordered_set<nt::tda::Simplex> image = computeImage(chainComplex, k + 1, kPlus1Chains);

        // Compute ker(∂_k) / im(∂_{k+1})
        Homology homologyGroup;
        for (const auto& simplex : kernel) {
            if (image.find(simplex) == image.end()) {
                homologyGroup.push_back(simplex);
            }
        }

        return homologyGroup;
    }
    
    static void computeHomology(std::vector<Homology>& homologies, const ChainComplex& chainComplex, int k) {
        // Get the chain groups and boundary operators
        std::unordered_set<nt::tda::Simplex> kChains = chainComplex.get_chains(k);
        if(kChains.size() == 0){
            return;
        }
        std::unordered_set<nt::tda::Simplex> kPlus1Chains = chainComplex.get_chains(k + 1);

        // Compute ker(∂_k)
        std::unordered_set<nt::tda::Simplex> kernel = computeKernel(chainComplex, k, kChains);
        if(kernel.size() == 0){return;}
        if(kPlus1Chains.size() == 0){
            Homology homologyGroup;
            homologyGroup.insert(kernel);
            homologies.push_back(homologyGroup);
            return;
 
        }

        // Compute im(∂_{k+1})
        std::unordered_set<nt::tda::Simplex> image = computeImage(chainComplex, k + 1, kPlus1Chains);
        if(image.size() == 0){
            Homology homologyGroup;
            homologyGroup.insert(kernel);
            homologies.push_back(homologyGroup);
            return;
 
        }

        // Compute ker(∂_k) / im(∂_{k+1})
        Homology homologyGroup;
        homologyGroup.reserve(kernel.size());
        for (const auto& simplex : kernel) {
            if (image.find(simplex) == image.end()) {
                homologyGroup.push_back(simplex);
            }
        }
        if(homologyGroup.size() == 0){return;}
        homologies.push_back(homologyGroup);

    }


private:
    static bool isZero(const std::vector<std::pair<nt::tda::Simplex, int>>& boundary) {
        std::map<nt::tda::Simplex, int> simplex_sums;

        // Sum the orientations for each simplex
        for (const auto& [simplex, orientation] : boundary) {
            simplex_sums[simplex] += orientation;
        }

        // Check if all summed coefficients are zero
        for (const auto& [simplex, sum] : simplex_sums) {
            if (sum != 0) {
                return false;
            }
        }

        return true;
    }
    static std::unordered_set<nt::tda::Simplex> computeKernel(const ChainComplex& chainComplex, int k, const std::unordered_set<nt::tda::Simplex>& kChains) {
        std::unordered_set<nt::tda::Simplex> kernel;
        for (const auto& chain : kChains) {
            std::vector<std::pair<nt::tda::Simplex, int>> boundary_result = chainComplex.boundary(chain, k);
            if (isZero(boundary_result)) {  // Check if boundary sums to zero
                kernel.insert(chain);
            }
        }
        return kernel;
    }

    static std::unordered_set<nt::tda::Simplex> computeImage(const ChainComplex& chainComplex, int kPlus1, const std::unordered_set<nt::tda::Simplex>& kPlus1Chains) {
        std::unordered_set<nt::tda::Simplex> image;
        for (const auto& chain : kPlus1Chains) {
            std::vector<std::pair<nt::tda::Simplex, int> > boundaries = chainComplex.boundary(chain, kPlus1);
            for(const auto& boundary : boundaries){
                image.insert(boundary.first);
            }
        }
        return image;
    }
};


void handle_births_deaths(std::vector<Homology>& cur_homologies, HomologyMap& birth_deaths, const double& radi){
    for(const auto& homology : cur_homologies){
        if(birth_deaths.find(homology) == birth_deaths.end()){
            birth_deaths[homology] = {radi, radi};
        }
    }
    for(auto& [homology, log] : birth_deaths){
        if(log.second != log.first){continue;}
        bool found = false;
        for(const auto& c_h : cur_homologies){
            if(c_h == homology){found = true;break;}
        }
        if(found) continue;
        log.second = radi;
    }
}


HomologyMap construct_homologies(nt::Tensor cloud, int64_t dims, int64_t point = 1, double increment=0.2){
    if(dims == cloud.dims()){
        cloud = cloud.unsqueeze(0);
    }
    nt::tda::BatchPoints pts(cloud, point, dims); 
    //this is going to control the radius of each basis around a point
    nt::tda::BatchBasises balls(pts);
    std::unordered_map<nt::tda::Point, std::pair<double, double>> h0_homologies;
    const std::vector<nt::tda::Point>& _gen_pts = pts.generatePoints(0);
    for(const auto& pt : _gen_pts){
        auto [it, worked] = h0_homologies.insert({pt, {0.0, 0.0}});
        nt::utils::throw_exception(worked, "Point $ was unable to be tracked [internal error]", pt);
    }
        // nt::tda::Simplex simp(1, dims); //hold 1 point of size dims;
        // simp[0] = pt;
        // Homology out(simp);
        // h0_homologies[out] = {0.0, 0.0};
    // }
    double radi = increment;
    balls.radius_to(radi, false);


    double max = 0;
    for(int64_t i = 0; i < dims; ++i){
        max = std::max(max, static_cast<double>(cloud.shape()[-1 + (-1 * i)]));
    }

    

    HomologyMap births_deaths;
    int64_t init_size = _gen_pts.size();
    while(radi < max){
        // nt::utils::printProgressBar(radi*10, max*10, " " + std::to_string(balls.get_balls(0).size()));
        radi += increment;
        balls.radius_to(radi);
        //handle the h0 homology group
        if(balls.get_balls(0).size() != init_size){
            for(const auto& ball : balls.get_balls(0)){
                if(ball.points.size() == 1){continue;}
                for(const auto& pt : ball.points){
                    if(h0_homologies[pt].second == 0.0) h0_homologies[pt].second = radi;
                }
            }
            init_size = balls.get_balls(0).size();
        }
        std::vector<Homology> homologies;
        for(const auto& ball : balls.get_balls(0)){ // only works for batch size of 1
            auto start = std::chrono::high_resolution_clock::now();
            ChainComplex chain;
            for(int i = 1; i <= dims; ++i){
                auto start_inner = std::chrono::high_resolution_clock::now();
                nt::tda::Simplexes simps(ball, i);
                auto end_inner = std::chrono::high_resolution_clock::now();
                auto duration_inner = std::chrono::duration_cast<std::chrono::milliseconds>(end_inner - start_inner);
                std::cout << "simplex creation took: " << duration_inner.count() << " milliseconds" << std::endl;
                start_inner = std::chrono::high_resolution_clock::now();
                chain.add_simplexes(simps);
                end_inner = std::chrono::high_resolution_clock::now();
                duration_inner = std::chrono::duration_cast<std::chrono::milliseconds>(end_inner - start_inner);
                std::cout << "simplex addition took: " << duration_inner.count() << " milliseconds" << std::endl;
                // if(i == 2){std::cout << (chain.chain_groups.find(i) == chain.chain_groups.end() ? 0 : chain.chain_groups.find(i)->second.size()) << std::endl;}
            }
            // std::cout << "going to assemble"<<std::endl;
            chain.assemble();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            if(duration.count() != 0){
                std::cout << "chain assembly took: " << duration.count() << " milliseconds" << std::endl;
            }
            // if(!chain.verify_chain_complex()){continue;}
            
            // Compute homology groups
            // std::cout << "going to compute homologies"<<std::endl;
            start = std::chrono::high_resolution_clock::now();
            for(int i = 1; i <= dims; ++i){
                Homologies::computeHomology(homologies, chain, i+1); //i+1 is the amount of points in a simplex
            }
            end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            if(duration.count() != 0)
                std::cout << "Homology computation took: " << duration.count() << " milliseconds" << std::endl;

        }
        handle_births_deaths(homologies, births_deaths, radi);
        if(balls.get_balls(0).size() == 1){break;}
    }
    std::vector<Homology> homologies;
    handle_births_deaths(homologies, births_deaths, radi);

    for(const auto& [pt, log] : h0_homologies){
        births_deaths.add(pt, log);
    }

    std::cout << "ending amount of homologies: "<< births_deaths.size()<<std::endl; 
    return std::move(births_deaths);
}

void plotPersistentDiagram(const HomologyMap& homologyData, int64_t dims, bool _do_show=true) {
    using namespace matplot;

    //blue, orange, green, red, magenta
    std::vector<std::array<float, 3>> colors = {
        {0.12f, 0.46f, 0.7f},
        {1.0f, 0.49f, 0.055f} , 
        {0.17f, 0.63f, 0.17f}, 
        {1.0f, 0.0f, 0.0f}, 
        {1.0f, 0.0f, 1.0f} 
    };
    while(colors.size() < dims+1){
        colors.push_back({static_cast <float> (rand()) / static_cast <float> (RAND_MAX), static_cast <float> (rand()) / static_cast <float> (RAND_MAX), static_cast <float> (rand()) / static_cast <float> (RAND_MAX)});
    }
    // colororder(colors);
    double max_radi = 0;
    for (const auto& [homology, radii] : homologyData) {
        max_radi = std::max({max_radi, radii.first, radii.second});
    }
    std::vector<std::string> names;
    figure();
    hold(on); // Ensure multiple scatter plots are overlaid
    std::vector<double> x_vals = {0, max_radi * 1.1};
    std::vector<double> y_vals = {0, max_radi * 1.1};
    std::vector<double> x_vals2 = {0, max_radi * 1.1};
    std::vector<double> y_vals2 = {max_radi, max_radi};
    auto diag_line = plot(x_vals, y_vals, "k--");  // "k--" means black dotted line
    diag_line->line_width(2.0);  
    names.push_back("∞"); 
    for(int i = 0; i <= dims; ++i){
        std::vector<double> birth, death;
        for (const auto& [homology, radii] : homologyData) {
            if(homology.simplex_dims() != i){continue;}
            birth.push_back(radii.first);
            death.push_back(radii.second);
        }
        if(birth.size() == 0){continue;}
        std::cout << "for homology "<<i<<" have "<<birth.size()<<" points"<<std::endl;
        auto sc = scatter(birth, death, 10); // Scatter plot
        sc->marker_color(colors[i]);
        // sc->marker("o");
        sc->marker_face_color(colors[i]);
        names.push_back("H_{" + std::to_string(i)+"}");
        //sc->display_name("H" + std::to_string(i));
    }


    auto straight_line = plot(x_vals2, y_vals2, "k--");
    straight_line->line_width(2.0);

    xlabel("Birth");
    ylabel("Death");
    title("Persistence Diagram");
    axis({-0.5, max_radi * 1.1, -0.5, max_radi * 1.1});
    grid(false);
    auto l = matplot::legend(names); // Show legend
    l->location(legend::general_alignment::bottomright);
    l->font_size(15);

    save("homology_diagram.png");
    if(_do_show){show();}
}

void plot_point_cloud(nt::Tensor cloud, uint8_t point, int64_t dims){
    nt::utils::throw_exception(dims == 1 || dims == 2 || dims == 3, "Can only work with dimensions 1, 2, or 3 but got $", dims);
    using namespace matplot;
    nt::Tensor where = nt::functional::where(cloud == point);
    figure();
    if(dims == 1){
        nt::Tensor y = where[where.numel()-1].item<nt::Tensor>().to(nt::DType::Double);

        std::vector<double> y_dots(reinterpret_cast<double*>(y.data_ptr()), reinterpret_cast<double*>(y.data_ptr_end()));
        std::vector<double> x_dots(y_dots.size(), 1);
        scatter(x_dots, y_dots, 7);
        axis({-0.1, static_cast<double>(cloud.shape()[-1]) * 1.1, -0.1, static_cast<double>(cloud.shape()[-1]) * 1.1});
    }
    if(dims == 2){
        nt::Tensor y = where[where.numel()-1].item<nt::Tensor>().to(nt::DType::Double);
        nt::Tensor x = where[where.numel()-2].item<nt::Tensor>().to(nt::DType::Double);
        std::vector<double> y_dots(reinterpret_cast<double*>(y.data_ptr()), reinterpret_cast<double*>(y.data_ptr_end()));
        std::vector<double> x_dots(reinterpret_cast<double*>(x.data_ptr()), reinterpret_cast<double*>(x.data_ptr_end()));
        scatter(x_dots, y_dots, 7);
        axis({-0.1, static_cast<double>(cloud.shape()[-1]) * 1.1, 
            -0.1, static_cast<double>(cloud.shape()[-2]) * 1.1});
    }
    if(dims == 3){
        nt::Tensor y = where[where.numel()-1].item<nt::Tensor>().to(nt::DType::Double);
        nt::Tensor x = where[where.numel()-2].item<nt::Tensor>().to(nt::DType::Double);
        nt::Tensor z = where[where.numel()-3].item<nt::Tensor>().to(nt::DType::Double);
        std::vector<double> y_dots(reinterpret_cast<double*>(y.data_ptr()), reinterpret_cast<double*>(y.data_ptr_end()));
        std::vector<double> x_dots(reinterpret_cast<double*>(x.data_ptr()), reinterpret_cast<double*>(x.data_ptr_end()));
        std::vector<double> z_dots(reinterpret_cast<double*>(z.data_ptr()), reinterpret_cast<double*>(z.data_ptr_end()));
        scatter3(z_dots, x_dots, y_dots);
        // s_axis = {-0.1, static_cast<double>(cloud.shape()[-1]) * 1.1, 
        //     -0.1, static_cast<double>(cloud.shape()[-2]) * 1.1,
        //     -0.1, static_cast<double>(cloud.shape()[-3]) * 1.1};
    }
    
    title("Point Cloud");
    grid(false);
    save("point_cloud.png");
    show();



}

class Point{
    nt::Tensor tensor;
	public:
		Point()
			:tensor()
		{}
		explicit Point(int64_t n)
			:tensor({n}, nt::DType::int64)
		{}
		Point(std::initializer_list<nt::Scalar> l)
			:tensor(nt::Tensor::FromInitializer<1>(l, nt::DType::int64))
		{}
        Point(nt::Tensor t)
        :tensor(t)
        {}
		Point(Point&& p)
			:tensor(std::move(p.tensor))
		{}
		Point(const Point& p)
			:tensor(p.tensor)
		{}
		Point& operator=(const Point& p){
            tensor = p.tensor;
			return *this;
		}
		Point& operator=(Point&& p){
			tensor = std::move(p.tensor);
			return *this;
		}
		//explicit function to share the memory
		inline Point& share(const Point& p) noexcept{
			tensor = p.tensor;
			return *this;
		}
		inline int64_t& operator[](int64_t n) noexcept {return reinterpret_cast<int64_t*>(tensor.data_ptr())[n];}
		inline const int64_t& operator[](int64_t n) const noexcept {return reinterpret_cast<const int64_t*>(tensor.data_ptr())[n];}
		inline const int64_t& size() const noexcept {return tensor.numel();}
		inline const int64_t& dims() const noexcept {return tensor.numel();}
		inline int64_t* begin() noexcept {return reinterpret_cast<int64_t*>(tensor.data_ptr());}
		inline int64_t* end() noexcept {return reinterpret_cast<int64_t*>(tensor.data_ptr_end());}
		inline const int64_t* begin() const noexcept {return reinterpret_cast<const int64_t*>(tensor.data_ptr());}
		inline const int64_t* end() const noexcept {return reinterpret_cast<const int64_t*>(tensor.data_ptr_end());}
		inline const int64_t* cbegin() const noexcept {return reinterpret_cast<const int64_t*>(tensor.data_ptr());}
		inline const int64_t* cend() const noexcept {return reinterpret_cast<const int64_t*>(tensor.data_ptr_end());}
		inline Point clone() const noexcept {return Point(tensor.clone());}
		inline Point operator+(int64_t element) const noexcept{
            return Point(tensor + element);
		}
		inline Point& operator+=(int64_t element) noexcept{
            tensor += element;
			return *this;
		}
		inline const int64_t& back() const noexcept {return (*this)[size()-1];}
		inline int64_t& back() noexcept {return (*this)[size()-1];}
		inline const bool operator==(const Point& p) const noexcept{return nt::functional::all(tensor == p.tensor);}
		inline const bool operator!=(const Point& p) const noexcept{return !((*this) == p);}
        inline nt::Tensor& detach() {return tensor;}
        inline const nt::Tensor& detach() const {return tensor;}


};

nt::Tensor extract_points_from_cloud(nt::Tensor cloud, uint8_t point, int64_t dims){
    nt::utils::throw_exception(cloud.dims() >= dims, "Expected to process cloud with dims of at least $ but got $", dims, cloud.dims());
    if(cloud.dims() == dims){cloud = cloud.unsqueeze(0);}
    nt::Tensor points_w = nt::functional::where((cloud == 1).split_axis((-1) * (dims+1)));
    for(int64_t i = 0; i < points_w.shape()[0]; ++i){
        points_w[i].item<nt::Tensor>() = nt::functional::stack(points_w[i].item<nt::Tensor>());
    }
    nt::Tensor points = points_w.RowColSwap_Tensors();
    return std::move(points);
}

nt::Tensor get_pts_in_r(nt::Tensor points, Point p, double radius, int64_t dims){
    int64_t batches = points.numel();
    nt::Tensor point = p.detach();
    point = point.repeat_(batches).view(-1, dims).split_axis(0);
    nt::Tensor dist_sq = std::pow((points - point), 2).sum(1);
    nt::Tensor mask = (dist_sq <= (radius*radius));
    nt::Tensor filtered = points[mask];
    return std::move(filtered);
}

void get_points_in_r_test(){
   int64_t dims = 3;
    nt::Tensor cloud = nt::functional::zeros({6, 30, 30, 30}, nt::DType::uint8);
    nt::Tensor bools = nt::functional::randbools(cloud.shape(), 0.05); //fill 5% with 1's
    cloud[bools] = 1;
    std::cout << "the cloud has "<<nt::functional::count(cloud == 1) << " ones (points) with a size of "<<cloud.numel() << std::endl;
    nt::Tensor points = extract_points_from_cloud(cloud, 1, 3);
    Point p({15, 10, 5});
    double radius = 5.0;
    nt::Tensor filtered = get_pts_in_r(points, p, radius, dims); 
}


class BasisOverlapping{
    nt::Tensor dist_sq;
public:
    BasisOverlapping() = default;
    BasisOverlapping(nt::Tensor points){
        nt::utils::throw_exception(points.dtype == nt::DType::TensorObj, "Expected to get batches of tensors in terms of tensor objects but got $", points.dtype);
        nt::utils::throw_exception(points[0].item<nt::Tensor>().dtype == nt::DType::int64, "Expected to get the coordinates of the points in terms of int64, but got $",
                                    points[0].item<nt::Tensor>().dtype);
        nt::utils::throw_exception(points[0].item<nt::Tensor>().dims() == 2, "Expected to get the dims of the points as 2, but got $",
                                    points[0].item<nt::Tensor>().dims());
        
        int D = points[0].item<nt::Tensor>().shape()[1];
        nt::Tensor cpy_pts = nt::Tensor::makeNullTensorArray(points.numel());
        // nt::Tensor cpy_pts = points.view_Tensors(-1, 1, D).transpose_Tensors(-1, -2); 
        for(int64_t i = 0; i < cpy_pts.numel(); ++i){
            auto sh = points[i].item<nt::Tensor>().shape();
            cpy_pts[i] = points[i].item<nt::Tensor>().view(1, -1, D).expand({sh[0], sh[0], D}).clone();
            // std::cout << "cpy_pts[i]: "<<cpy_pts[i].item<nt::Tensor>() << std::endl;
        }
        // Expand points: (N, 1, D) - (1, N, D) -> (N, N, D)
        nt::Tensor diff = points.view_Tensors(-1, 1, D) - cpy_pts;

        // Compute squared Euclidean distances: sum across last dimension (D)
        dist_sq = diff.pow(2).sum(-1);

    }

    nt::Tensor adjust_radius(double r) const {
        // Compute squared radius sum (r + r)^2 = (2r)^2
        double r_sq = 4 * (r * r);

        // Compute mask: dist_sq <= (2r)^2
        nt::Tensor overlap_mask = (dist_sq <= r_sq);

        // Remove self-overlap (diagonal elements)
        overlap_mask.fill_diagonal_(false);

        // Return indices of overlapping pairs
        //return overlap_mask.nonzero();
        return std::move(overlap_mask);
 
    }

};

nt::Tensor find_overlapping_bases(const nt::Tensor& points, double r) {
    int D = points[0].item<nt::Tensor>().shape()[1];

    nt::Tensor cpy_pts = nt::Tensor::makeNullTensorArray(points.numel());
    // nt::Tensor cpy_pts = points.view_Tensors(-1, 1, D).transpose_Tensors(-1, -2); 
    for(int64_t i = 0; i < cpy_pts.numel(); ++i){
        auto sh = points[i].item<nt::Tensor>().shape();
        cpy_pts[i] = points[i].item<nt::Tensor>().view(1, -1, D).expand({sh[0], sh[0], D}).clone();
    }
    // Expand points: (N, 1, D) - (1, N, D) -> (N, N, D)
    nt::Tensor diff = points.view_Tensors(-1, 1, D) - cpy_pts;

    // Compute squared Euclidean distances: sum across last dimension (D)
    nt::Tensor dist_sq = diff.pow(2).sum(-1);

    // Compute squared radius sum (r + r)^2 = (2r)^2
    double r_sq = 4 * (r * r);

    // Compute mask: dist_sq <= (2r)^2
    nt::Tensor overlap_mask = (dist_sq <= r_sq).transpose_Tensors(0, -1);


    // Remove self-overlap (diagonal elements)
    overlap_mask.fill_diagonal_(false);

    // Return indices of overlapping pairs
    //return overlap_mask.nonzero();
    return std::move(overlap_mask);
}

std::vector<std::vector<int64_t>> search_cliques(const nt::Tensor& within_radius, const nt::Tensor& overlay_mask, std::vector<int64_t> candidates, int64_t depth, int64_t K, std::vector<int64_t>& item_vec){
    if(depth == K){return {std::move(candidates)};}
    std::vector<std::vector<int64_t> > cliques;
    const int64_t last = candidates.back();

    candidates.push_back(0);
    auto c_begin = candidates.cbegin();
    auto c_end = candidates.cend();
    --c_end; 
    if(within_radius[last].item<nt::Tensor>().is_null()){
        return std::move(cliques);
    }
    nt::Tensor neighbors = within_radius[last].item<nt::Tensor>().item<nt::Tensor>();
    if(neighbors.is_null() || neighbors.is_empty()){return std::move(cliques);}
    // std::cout << "neighbors: "<<neighbors<<std::endl;
    const int64_t* n_begin = reinterpret_cast<const int64_t*>(neighbors.data_ptr());
    const int64_t* n_end = reinterpret_cast<const int64_t*>(neighbors.data_ptr_end());
    for(;n_begin != n_end && *n_begin < last; ++n_begin){;}
    // while(*n_begin < last && n_begin < n_end){++n_begin;}
    if(n_begin >= n_end){return std::move(cliques);} 
    for(;n_begin < n_end; ++n_begin){
        bool stop = false;
        for(;c_begin != c_end; ++c_begin){
            item_vec[0] = *n_begin;
            item_vec[1] = *c_begin;
            if(!overlay_mask.item<bool>(item_vec)){
                stop = true;
                break;
            }
        }
        c_begin = candidates.cbegin();
        if(stop) continue;
        candidates.back() = *n_begin;
        auto n_cliques = search_cliques(within_radius, overlay_mask, candidates, depth+1, K, item_vec);
        cliques.insert(cliques.cend(), n_cliques.begin(), n_cliques.end());
    }
    return std::move(cliques);
}

nt::Tensor construct_simplexes(const nt::Tensor& points, const nt::Tensor& initial, const std::vector<std::vector<int64_t> >& cliques, int64_t N){
    std::vector<nt::Tensor> final(cliques.size());
    int64_t point_dim = initial.shape()[-1];
    auto cliques_b = cliques.cbegin();
    auto cliques_e = cliques.cend();
    auto f_begin = final.begin();
    for(; cliques_b != cliques_e; ++cliques_b, ++f_begin){
        std::vector<nt::Tensor> outputs(N+1);
        outputs[0] = initial;
        auto o_begin = outputs.begin();
        ++o_begin;
        auto c_begin = cliques_b->cbegin();
        auto c_end = cliques_b->cend();
        int64_t pos = 1;
        for(;c_begin != c_end; ++c_begin, ++o_begin, ++pos){
            *o_begin = points[*c_begin];
        }
        //a lack of checks in terms of shapes makes the following faster
        //when I already know it will work
        *f_begin = nt::functional::cat_unordered(outputs).view(N+1, point_dim);
    }
    return nt::functional::stack(std::move(final));
}

nt::Tensor _sub_find_all_simplicies(int64_t simplicies_amt, const nt::Tensor& points, const nt::Tensor& ball_masks, double r, int64_t index){
    // std::cout << "point[0] is "<<points[0] << std::endl;
    nt::Tensor initial = points[index];
    nt::Tensor mask = ball_masks[index];
    if(nt::functional::count(mask) < simplicies_amt){
        // std::cout << "was less than simplicies_amt"<<std::endl;
        return nt::Tensor::Null();
    }
    nt::Tensor potentials = points[mask];
    if(simplicies_amt == 2){
        std::vector<nt::Tensor> stacking(potentials.shape()[0]);
        for(int64_t i = 0; i < potentials.shape()[0]; ++i){
            std::vector<nt::Tensor> output({initial, potentials[i]});
            stacking[i] = nt::functional::cat(std::move(output));
        }
        return nt::functional::stack(std::move(stacking));
    }
    // std::cout << "potential points: "<<potentials<<std::endl;
    int64_t D = potentials.shape()[-1];
    int64_t N = potentials.shape()[-2];
    nt::Tensor cpy_potentials = potentials.view(1, -1, D).expand({N, N, D}).transpose(0, 1).clone();
    nt::Tensor diff = potentials.view(-1, 1, D) - cpy_potentials;
    nt::Tensor dist_sq = diff.pow(2).sum(-1); //pairwise distance
    double r_sq = (r * r);
    nt::Tensor overlap_mask = (dist_sq <= r_sq);
    // std::cout << "simplex overlap mask before: "<<overlap_mask << std::endl;
    overlap_mask.fill_diagonal_(false);
    // std::cout << "simplex overlap mask after: "<<overlap_mask << std::endl;
    nt::Tensor split = overlap_mask.split_axis(0);
    // std::cout << "split is: "<<split<<std::endl;
    nt::Tensor within_radius = nt::functional::where(split);
    // std::cout << "simplicies amt is "<<simplicies_amt<<std::endl;
    // std::cout << "got within radius: "<<within_radius <<std::endl;
    if(N >= simplicies_amt){ 
        std::vector<std::vector<int64_t>> cliques;
        std::vector<int64_t> item_vec({0,0});
        for(int64_t i = 0; i < N; ++i){
            auto n_cliques = search_cliques(within_radius, overlap_mask, {i}, 1, simplicies_amt-1, item_vec);
            cliques.reserve(n_cliques.size());
            for(const auto& cliq : n_cliques){
                cliques.push_back(cliq);

            }
        }
        if(cliques.size() == 0){return nt::Tensor::Null();}
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
        //construct simplicies based on cliques
        return construct_simplexes(potentials, initial, cliques, simplicies_amt-1);
        // std::cout << "simplicies: "<<simplicies<<std::endl;
    }

    return nt::Tensor::Null();    
}


//for example a 0-simpex would require 1 point in it
nt::Tensor find_all_simplicies(int64_t simplicies_amt, const nt::Tensor& points, const BasisOverlapping& balls, double radius){
    nt::Tensor overlap_mask = balls.adjust_radius(radius);
    std::vector<nt::Tensor> out_batches(points.shape()[0]);
    for(int64_t b = 0; b < points.shape()[0]; ++b){
        nt::Tensor pts_ = points[b].item<nt::Tensor>();
        nt::Tensor masks_ = overlap_mask[b].item<nt::Tensor>();
        std::vector<nt::Tensor> simplicies;
        simplicies.reserve(pts_.shape()[0]);
        for(int64_t i = 0; i < pts_.shape()[0]; ++i){
            nt::Tensor i_simplicies = _sub_find_all_simplicies(simplicies_amt, pts_, masks_, radius, i);
            if(!i_simplicies.is_null()){
                simplicies.emplace_back(std::move(i_simplicies));
            }
        }
        if(simplicies.size() > 0){
            nt::Tensor pre_process = nt::functional::cat(std::move(simplicies));
            // if(b == 0){
            //     std::cout << "before pre-process shape: "<<pre_process.shape() << std::endl;
            //     std::cout << "after split shape: "<<pre_process.split_axis(-2).shape() << std::endl;
            //     std::cout << "after split: "<<pre_process.split_axis(-2).view(pre_process.shape()[0], -1)[0] << std::endl;
            // }
            //going to sort, and then find the unique simplexes
            nt::Tensor sorted = nt::functional::coordsort(pre_process, -2, false, true, false); //return sorted only
            // auto [sorted, indices_sort] = nt::get<2>(nt::functional::coordsort(pre_process, -2));
            // if(b == 0){std::cout << sorted.view(sorted.shape()[0], -1) << std::endl; std::cout << pre_process.view(pre_process.shape()[0], -1) << std::endl; std::cout << indices_sort << std::endl;}
            nt::Tensor unique = nt::functional::unique(sorted.view(sorted.shape()[0], -1), -1, true, false); //return unique only
            // nt::Tensor unique_parts = nt::functional::unique(sorted.view(sorted.shape()[0], -1), -1);
            // auto [unique, indices_unique] = nt::get<2>(unique_parts);
            out_batches[b] = unique.view(-1, sorted.shape()[-2], sorted.shape()[-1]);
            // std::cout << "preprocess shape after: "<< out_batches[b].shape() << std::endl; 
        }else{
            out_batches[b] = nt::Tensor::Null();
        }
    }
    return nt::functional::vectorize(std::move(out_batches));
    // outs.reserve(points.shape()[0]);
    // for(int64_t i = 0; i < 
    // return _sub_find_all_simplicies(simplicies_amt, points[0].item<nt::Tensor>(), overlap_mask[0].item<nt::Tensor>(), radius);
}




//this is a function where all simplicies for all radi can be first created
//then this function assigns radi values to each simplex as in when they showed up
nt::Tensor compute_circumradii(const nt::Tensor& simplicies){
    /*
     Computes the associated radius with each simplex
    
     Args:
        simplicies: {B, N+1, D} 
            B: batches of simplicies
            N+1: N+1 verticies for each simplex
            D: dimensionality of each point
     Returns:
        Tensor: {B}, dtype float64
            B: the radius associated with each simplex
        
    */
    if(simplicies.dtype == nt::DType::TensorObj){
        nt::Tensor out = nt::Tensor::makeNullTensorArray(simplicies.numel());
        nt::Tensor* begin = reinterpret_cast<nt::Tensor*>(out.data_ptr());
        for(auto t : simplicies){
            *begin = compute_circumradii(t);
            ++begin;
        }
        return std::move(out);

    }
    nt::utils::throw_exception(simplicies.dims() == 3, "Expected to get simplicies of dims 3, corresponding to batches, verticies, point dim but got $", simplicies.dims());
    const int64_t& B = simplicies.shape()[0]; //Batches
    const int64_t Np1 = simplicies.shape()[1]; //N+1 verticies
    const int64_t& D = simplicies.shape()[2]; //Dims
    
    nt::Tensor f_simplicies = simplicies.to(nt::DType::Float64);
    nt::Tensor centroid = f_simplicies.mean(-2, true); //the center of the simplexes
    centroid = centroid.repeat_(-2, 3);

    nt::Tensor avg_distances = f_simplicies - centroid;
    avg_distances.pow_(2);
    avg_distances = avg_distances.sum({-1, -2});
    //it is divided by 4 because when the simplexes were formed, it was dist_sq < (2 * radius)^2 
    avg_distances /= (Np1 * 4);
    return nt::functional::sqrt(avg_distances);
}


template <typename T>
struct NumericVectorHash {

    std::size_t operator()(const nt::Tensor& vec) const {
        return vec.arr_void().cexecute_function<nt::WRAP_DTYPES<nt::DTypeEnum<nt::DTypeFuncs::type_to_dtype<T> > > >([](auto begin, auto end){
            std::size_t hash = 0;
            for(;begin != end; ++begin){
                if constexpr (std::is_same_v<nt::my_complex<nt::float16_t>, T>){
                    hash ^= std::hash<float>{}(_NT_FLOAT16_TO_FLOAT32_(begin->real())) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                    hash ^= std::hash<float>{}(_NT_FLOAT16_TO_FLOAT32_(begin->imag())) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr (std::is_same_v<nt::my_complex<float>, T>){
                    hash ^= std::hash<float>{}(begin->real()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                    hash ^= std::hash<float>{}(begin->imag()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr (std::is_same_v<nt::my_complex<double>, T>){
                    hash ^= std::hash<double>{}(begin->real()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                    hash ^= std::hash<double>{}(begin->imag()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr (std::is_same_v<nt::float16_t, T>){
                    hash ^= std::hash<float>{}(_NT_FLOAT16_TO_FLOAT32_(*begin)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr(std::is_same_v<nt::uint_bool_t, T>){
                    hash ^= std::hash<float>{}(*begin ? float(1) : float(0)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
#ifdef __SIZEOF_INT128__
                else if constexpr(std::is_same_v<nt::uint128_t, T>){
                    hash ^= std::hash<int64_t>{}(nt::convert::convert<int64_t, nt::uint128_t>(*begin)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr(std::is_same_v<nt::int128_t, T>){
                    hash ^= std::hash<int64_t>{}(nt::convert::convert<int64_t, nt::int128_t>(*begin)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
#endif
                else{
                    hash ^= std::hash<T>{}(*begin) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
            }
            return hash;
        });
    }
};

template<typename T>
struct NumericVectorEqual {
    bool operator()(const nt::Tensor& a, const nt::Tensor& b) const {
        if(a.numel() != b.numel() || a.dtype != b.dtype){return false;}
        if(a.is_null() || b.is_null()){return false;}
        const nt::ArrayVoid& arr_v = b.arr_void();
        return a.arr_void().cexecute_function<nt::DTypeFuncs::type_to_dtype<T> >([&arr_v](auto begin, auto end) -> bool{
            using value_t = nt::utils::IteratorBaseType_t<decltype(begin)>;
            return arr_v.cexecute_function<nt::DTypeFuncs::type_to_dtype<value_t>>([&begin, &end](auto second, auto s_end) -> bool{
                return std::equal(begin, end, second);
            });
        });
    }
};


nt::SparseTensor compute_boundary_matrix(const nt::Tensor& simplicies){
    nt::utils::throw_exception(simplicies.dims() == 3, "Expected to get simplicies of dims 3, corresponding to batches, verticies, point dim but got $", simplicies.dims());
    nt::utils::throw_exception(simplicies.dtype == nt::DType::int64, "Expected simplicies dtype to be int64 but got $", simplicies.dtype);
    const int64_t& B = simplicies.shape()[0]; //Batches
    const int64_t Np1 = simplicies.shape()[1]; //N+1 verticies
    const int64_t& D = simplicies.shape()[2]; //Dims
    // nt::Tensor faces = nt::functional::combinations(nt::functional::arange(Np1, nt::DType::int64), Np1-1).contiguous().split_axis(0);
    // nt::Tensor* f_begin = reinterpret_cast<nt::Tensor*>(faces.data_ptr());
    // nt::Tensor* f_end = reinterpret_cast<nt::Tensor*>(faces.data_ptr_end()); 
    // nt::Tensor B_dim = nt::functional::arange(B, nt::DType::int64).view(B, 1).repeat_(-1, D*(Np1-1)).flatten(0, -1).contiguous();
    // nt::Tensor point_dim = nt::functional::arange(D, nt::DType::int64).repeat_((Np1-1) * B).flatten(0, -1).contiguous();


    std::unordered_map<nt::Tensor, std::vector<std::pair<int64_t, int8_t>>, NumericVectorHash<int64_t>, NumericVectorEqual<int64_t> > simplex_map;
    for(int64_t i = 0; i < Np1; ++i){
        nt::Tensor parted_simplicies = simplicies.index_except(1, i);
        nt::Tensor split_faces = parted_simplicies.split_axis(0);
        int64_t counter = 0;
        nt::Tensor* split_begin = reinterpret_cast<nt::Tensor*>(split_faces.data_ptr());
        nt::Tensor* split_end = reinterpret_cast<nt::Tensor*>(split_faces.data_ptr_end());
        int8_t boundary = (i % 2 == 0) ? 1 : -1;
        for(;split_begin != split_end; ++split_begin, ++counter){
            auto [it, inserted] = simplex_map.insert({*split_begin, {{counter, boundary}}});
            if(!inserted){it->second.emplace_back(counter, boundary);}
        }
    }
    


    std::vector<int64_t> x;
    x.reserve(simplex_map.size());
    std::vector<int64_t> y;
    y.reserve(simplex_map.size());
    std::vector<int8_t> boundaries;
    boundaries.reserve(simplex_map.size());
    std::vector<int64_t> my_ints(2);
    my_ints[0] = 0;
    my_ints[1] = 1;
    int64_t r = 2;
    for(const auto [t, vec] : simplex_map){
        if(vec.size() == 1){continue;}
        int64_t n = vec.size();
        auto first = my_ints.begin();
        auto last = my_ints.end();
        while((*first) != n-r){
            auto mt = last;
            --mt; // Ensure mt is decremented before use
            while (*mt == n - int64_t(last - mt)) {
                --mt;
            }
            (*mt)++;
            while (++mt != last) *mt = *(mt-1)+1;
            x.push_back(vec[my_ints[0]].first);
            y.push_back(vec[my_ints[1]].first);
            boundaries.push_back(vec[my_ints[0]].second);
        }
        my_ints[0] = 0;
        my_ints[1] = 1;
    }

    nt::Tensor X = nt::functional::vector_to_tensor(x);
    nt::Tensor Y = nt::functional::vector_to_tensor(y);
    nt::Tensor Boundaries = nt::functional::vector_to_tensor(boundaries);
    return nt::SparseTensor(nt::functional::list(X, Y), Boundaries, {B, B}, nt::DType::int8, 0);
}

void cohomology_reduce(nt::SparseTensor& boundary_matrix) {
    nt::utils::throw_exception(boundary_matrix.dtype() == nt::DType::int8, 
                               "Expected boundary matrix to have a dtype of int8 but got $", 
                               boundary_matrix.dtype());
    nt::utils::throw_exception(boundary_matrix.dims() == 2,
                               "Expected boundary matrix to have a dimensionality of 2 but got $",
                               boundary_matrix.dims());
    nt::utils::throw_exception(boundary_matrix.shape()[0] == boundary_matrix.shape()[1],
                               "Expected boundary matrix to be a square matrix but got shape $",
                               boundary_matrix.shape());

    auto shape = boundary_matrix.shape();
    int64_t B = shape[0];
    
    std::vector<int64_t> pivot_col(B, -1);

    for(int64_t j = 0; j < B; ++j){
        auto col = boundary_matrix[j]; 
        //this is the same as getting a collumn of a transposed boundary matrix
        //just a lot more efficient
        int64_t pivot = -1;
        col.underlying_tensor().arr_void().cexecute_function<nt::WRAP_DTYPES<nt::DTypeEnum<nt::DType::int8> > >(
        [&pivot](auto begin, auto end){
            int64_t i = 0;
            for(;begin != end; ++begin, ++i){
                if(*begin != 0){
                    pivot = i;
                    break;
                }
            }
        });
        if(pivot == -1){continue;}
        pivot_col[pivot] = j;
        auto pivot_col = boundary_matrix[pivot];
        pivot_col.underlying_tensor().arr_void().cexecute_function<nt::WRAP_DTYPES<nt::DTypeEnum<nt::DType::int8> > >(
        [&boundary_matrix, &j, &pivot, &col](auto begin, auto end){
            begin += (j+1);
            int64_t k = j+1;
            for(;begin != end; ++begin, ++k){
                if(*begin != 0){
                    boundary_matrix[k] ^= col;
                }
            }
        });
    }

}

//takes a reduced boundary matrix
std::vector<std::pair<int64_t, int64_t>> extract_persistence_cohomology(const nt::SparseTensor reduced){
    nt::utils::throw_exception(reduced.dtype() == nt::DType::int8, 
                               "Expected boundary matrix to have a dtype of int8 but got $", 
                               reduced.dtype());
    nt::utils::throw_exception(reduced.dims() == 2,
                               "Expected boundary matrix to have a dimensionality of 2 but got $",
                               reduced.dims());
    nt::utils::throw_exception(reduced.shape()[0] == reduced.shape()[1],
                               "Expected boundary matrix to be a square matrix but got shape $",
                               reduced.shape());

    auto shape = reduced.shape();
    int64_t B = shape[0];
    
    std::unordered_map<int64_t, int64_t> birth_time;
    std::vector<std::pair<int64_t, int64_t>> persistence_pairs;

    reduced.underlying_tensor().arr_void().cexecute_function<nt::WRAP_DTYPES<nt::DTypeEnum<nt::DType::int8> > >(
    [&shape, &B, &birth_time, &persistence_pairs](auto begin, auto end){
    for(int64_t j = 0; j < B; ++j){
        for(int64_t i = 0; i < B; ++i){
            if(begin[j * B + i] != 0){
                if(birth_time.find(i) == birth_time.end()){
                    birth_time[i] = j;
                }else{
                    persistence_pairs.push_back({birth_time[i], j});
                    birth_time.erase(i);
                }
            }
        }
    }
    });
    return std::move(persistence_pairs);
}


void persistent_diagram_test(){
    //std::vector<std::pair<int64_t, int64_t>> cords({{20,10}, {30,40}, {32,23}, {4,5}, {2, 11}, {21, 28}, {17, 26}, {12, 13}, {20, 8}, {14, 4}, {32, 33}, {10, 27}, {36, 17}, {1, 18}, {10, 11}, {9, 17}, {13, 29}, {5, 34}, {24, 32}, {20, 32}, {18, 20}, {22, 24}, {7, 36}, {32, 13}});
    ////some random points to make a point cloud
    
    //nt::Tensor simps = nt::Tensor::FromInitializer<3>({ { {20, 10}, {20, 5}, {19, 3},
    //                                                        {19, 3}, {20, 10}, {20, 5} } }).view(2,3,2);
    //std::cout << "simps: "<<simps<<std::endl;
    //auto [sorted, indices] = nt::get<2>(nt::functional::coordsort(simps, -2));
    //std::cout << "sorted: "<<sorted<<std::endl;
    //std::cout << "sorted: "<<sorted.view(2, -1)<<std::endl;
    //auto u_out = nt::functional::unique(sorted.view(2, -1), -1);
    //auto [unique, indices_u] = nt::get<2>(u_out);
    //std::cout << "unique: "<<unique<<std::endl;
    //nt::Tensor split = simps.split_axis(0);
    //nt::Tensor a = split[0].item<nt::Tensor>();
    //nt::Tensor b = split[1].item<nt::Tensor>();
    //std::cout << "a: "<<a<<"b: "<<b<<std::endl;
    //const nt::ArrayVoid& arr = b.arr_void();
    //a.arr_void().cexecute_function<nt::WRAP_DTYPES<nt::NumberTypesL> >([](auto begin, auto end, auto second){
    //    using namespace nt;
    //    for(;begin != end; ++begin, ++second){
    //        std::cout << *begin << ',' << *second << std::endl;
    //    }
    //}, arr);



    int64_t dims = 2;
    nt::Tensor cloud = nt::functional::zeros({6, 30, 30}, nt::DType::uint8);
    // for(const auto& cord : cords)
    //     cloud[0][cord.first][cord.second] = 1;
    nt::Tensor bools = nt::functional::randbools(cloud.shape(), 0.03); //fill 5% with 1's
    uint8_t point = 1;
    cloud[bools] = 1;
    std::cout << "the cloud has "<<nt::functional::count(cloud == 1) << " ones (points) with a size of "<<cloud.numel() << std::endl;
    nt::Tensor points = extract_points_from_cloud(cloud, point, dims);
    Point p({15, 10, 5});
    double radius = 5.0;
    BasisOverlapping balls(points);
    //std::cout << "overlapping basis: "<<balls.adjust_radius(radius)[0].item<nt::Tensor>() << std::endl;
    nt::Tensor simplicies = find_all_simplicies(3, points, balls, radius);
    std::cout << "simplicies: "<<std::endl;
    for(auto t : simplicies){
        std::cout << t.shape() << std::endl;
    }
    nt::Tensor simplex_radi = compute_circumradii(simplicies);
    nt::SparseTensor boundary = compute_boundary_matrix(simplicies[0].item<nt::Tensor>());
    cohomology_reduce(boundary);
    auto pairs = extract_persistence_cohomology(boundary);
    nt::Tensor& radii_0 = simplex_radi[0].item<nt::Tensor>();
    int64_t total = 0;
    for(auto& pair : pairs){
        double first_r = radii_0[pair.first].item<double>();
        double second_r = radii_0[pair.second].item<double>();
        if(first_r > second_r){continue;}
        ++total;
        std::cout << "{("<<pair.first<<',' << first_r<<"), ("<<pair.second << ", " << second_r<<")} ";
    }
    std::cout << "total: "<<total <<  std::endl;
    // std::cout << simplicies[0].item<nt::Tensor>() << std::endl;
    // nt::Tensor filtered = get_pts_in_r(points, p, radius, dims);
    // std::cout << "filtered: "<<filtered<<std::endl;
    //auto diagram = construct_homologies(cloud.clone(), 2);
    // bool more = false;
    // for(const auto& [homology, log] : diagram){
    //     if(homology.simplex_dims() != 0){std::cout << "found a non-zero! "; more=true;}
    //     else{continue;}
    //     std::cout << "Homology H"<<homology.simplex_dims()<<": {birth: "<<log.first<<", death: "<<log.second<<", homology: "<< homology << "}"<<std::endl;
    // }
    // std::cout << "a total of "<<diagram.size()<<" homologies"<<std::endl;
    // if(!more){return;}
    //plotPersistentDiagram(diagram, 3);
    //plot_point_cloud(cloud, 1, 2);
}


//this function is used to map out how a persistence diagram would be made
//void persistent_diagram_test(){
//    std::vector<std::pair<int64_t, int64_t>> cords({{20,10}, {30,40}, {32,23}, {4,5}, {2, 11}, {21, 28}, {17, 26}, {12, 13}, {20, 8}, {14, 4}, {32, 33}, {10, 27}, {36, 17}, {1, 18}, {10, 11}, {9, 17}, {13, 29}, {5, 34}, {24, 32}, {20, 32}, {18, 20}, {22, 24}, {7, 36}, {32, 13}});
//    //some random points to make a point cloud
//    nt::Tensor cloud = nt::functional::zeros({1, 41,41}, nt::DType::uint8);
//    for(const auto& cord : cords)
//        cloud[0][cord.first][cord.second] = 1;
//    std::cout << "point cloud: " << cloud.to(nt::DType::uint32)<<std::endl;
//    //the point cloud, the point it is looking for, the dimensions to process
//    nt::tda::BatchPoints pts(cloud, 1, 2); 
//    //this is going to control the radius of each basis around a point
//    nt::tda::BatchBasises balls(pts);
    
//    const double radi_increment = 0.2; // this is the radius to increment by
//    double cur_radi = 1;
//    balls.radius_to(cur_radi, false);


//    const int64_t point_size = cords.size();
//    int64_t cur_basis_amt = balls.get_balls(0).size();
//    std::cout << std::endl << "current amount of seperate Basises that are not overlapping: "<< cur_basis_amt << std::endl;
//    std::cout << "number of points: "<<point_size << std::endl;
//    int64_t last_h0 = 0;
//    int64_t last_h1 = 0;
//    int64_t last_h2 = 0;
//    //homology, 
//    std::map<Homology, std::pair<double, double> > birth_deaths;
//    while(cur_radi < std::max(cloud.shape()[-1], cloud.shape()[-2])){
//        cur_radi += radi_increment;
//        balls.radius_to(cur_radi);
//        if(balls.get_balls(0).size() != cur_basis_amt){
//            cur_basis_amt = balls.get_balls(0).size();
//            // std::cout << "at radius "<<cur_radi << " the number of basis's overlapping = "<<cur_basis_amt << std::endl;
//        }
//        std::vector<Homology> homologies;
//        for(const auto& ball : balls.get_balls(0)){
//            nt::tda::Simplexes S0(ball, 0); //0D simplexes (a single point)
//            nt::tda::Simplexes S1(ball, 1); //1D simplexes (2 points)
//            nt::tda::Simplexes S2(ball, 2); //2D simplexes (3 points)
            
//            ChainComplex chain;
//            chain.add_simplexes(S0);
//            chain.add_simplexes(S1);
//            chain.add_simplexes(S2);
//            chain.assemble();
//            // if(!chain.verify_chain_complex()){continue;}
            
//            // Compute homology groups
//            Homologies::computeHomology(homologies, chain, 1); //1 is the amount of points in the simplex
//            Homologies::computeHomology(homologies, chain, 2);  
//            Homologies::computeHomology(homologies, chain, 3);
//        } 
//        handle_births_deaths(homologies, birth_deaths, cur_radi);
//        if(cur_basis_amt == 1 && homologies.size() == 0){break;}
//    }
//    for(const auto& [homology, log] : birth_deaths){
//        std::cout << "Homology H"<<homology.simplex_dims()<<": {birth: "<<log.first<<", death: "<<log.second<<", homology: "<< homology << "}"<<std::endl;
//    }

//}
