#include <nt/Tensor.h>
#include <nt/functional/functional.h>
#include <nt/tda/Homology.h>
#include <nt/tda/PlotDiagrams.h>
#include <nt/tda/SimplexConstruct.h>
#include <nt/tda/SimplexRadi.h>
#include <nt/tda/Boundaries.h>
#include <nt/tda/MatrixReduction.h>
#include <nt/tda/nn/PH.h>
#include <nt/nn/Loss.h>
#include <algorithm>
#include <random>
#include <vector>
#include <limits>


void persistent_dist_mat_gradient(){
    
    int64_t dims = 2;
    // nt::Tensor cloud = nt::functional::zeros({6, 30, 30}, nt::DType::uint8);
    // nt::Tensor bools = nt::functional::randbools(cloud.shape(), 0.03); //fill 3% with 1's
    int8_t point = 1;
    // cloud[bools] = 1;
    nt::TensorGrad cloud(nt::Tensor({9, 9}, nt::DType::Float32));
    cloud.detach()
          << 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0;

    
    nt::TensorGrad dist_mat = nt::tda::cloudToDist(cloud, 1);
    std::cout << "dist mat: "<<dist_mat<<std::endl;
    nt::Tensor wanted = nt::functional::rand(0.3, 2.5, dist_mat.shape(), dist_mat.dtype());
    nt::Tensor grad = dist_mat.detach() - wanted;
    dist_mat.backward(grad);
    std::cout << "cloud: " << cloud << std::endl;
    std::cout << "cloud grad: " << cloud.grad() << std::endl;
    cloud.update();
    std::cout << "cloud: " << cloud << std::endl;

}



//tests gettingg the gradient of a simplex complex
void persistent_simplex_complex_gradient(){
    nt::TensorGrad cloud(nt::Tensor({9, 9}, nt::DType::Float32));
    cloud.detach()
          << 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0;
    nt::TensorGrad dist_mat = nt::tda::cloudToDist(cloud, 1);
    auto [simplex_complex, radi] = nt::tda::VRfiltration(dist_mat, 3);
    std::cout <<"radi: "<< radi << std::endl;
    std::cout << "simplex complex: "<<simplex_complex<<std::endl;
    //negative values increase the radii
    //positive values decrease the radii
    nt::Tensor wanted = nt::functional::rand(1.0, 6.0, radi.shape(), nt::DType::Float32);
    nt::ScalarGrad loss = nt::tda::loss::filtration_loss(radi, wanted);
    loss.backward();
    std::cout << "cloud: " << cloud << std::endl;
    std::cout << "cloud grad: " << cloud.grad() << std::endl;
    cloud.update();
    std::cout << "cloud: " << cloud << std::endl;
    std::cout << "wanted: "<<wanted<<std::endl;
    std::cout << "loss: "<<loss<<std::endl;
    dist_mat = nt::tda::cloudToDist(cloud, 1);
    auto [n_simplex_complex, n_radi] = nt::tda::VRfiltration(dist_mat, 3);
    std::cout << "n_radi: "<<n_radi<<std::endl;
    std::cout << "n_simplex_complex: "<<n_simplex_complex<<std::endl;
    std::cout << std::boolalpha << n_radi.is_contiguous() << std::noboolalpha << std::endl;
}


//tests gettingg the gradient of a simplex complex
void persistent_boundary_gradient(){
    nt::TensorGrad cloud(nt::Tensor({9, 9}, nt::DType::Float32));
    cloud.detach()
          << 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0;
    nt::TensorGrad dist_mat = nt::tda::cloudToDist(cloud, 1);
    auto [simplex_complex_3, radi_3] = nt::tda::VRfiltration(dist_mat, 3);
    auto [simplex_complex_2, radi_2] = nt::tda::VRfiltration(dist_mat, 2);
    std::cout <<"radi: "<< radi_3 << std::endl;
    std::cout <<"radi: "<< radi_2 << std::endl;
    std::cout << std::boolalpha << radi_3.is_contiguous() << std::noboolalpha << std::endl;
    float alpha = 10.0;
    nt::TensorGrad sig_radi_3 = nt::functional::sigmoid(alpha * radi_3);
    nt::TensorGrad sig_radi_2 = nt::functional::sigmoid(alpha * radi_2);
    std::cout << "sig_radi_3: "<<sig_radi_3 << std::endl;
    std::cout << "sig_radi_2: "<<sig_radi_2 << std::endl;
    auto [x_indexes, y_indexes, boundaries] = 
            nt::tda::compute_differentiable_boundary_sparse_matrix_index(simplex_complex_3, simplex_complex_2,
                                                                         sig_radi_3.detach(), sig_radi_2.detach());

    nt::utils::throw_exception(x_indexes.size() == y_indexes.size()
                           && x_indexes.size() == boundaries.size(),
                            "wrong sizes, $, $, $ ", x_indexes.size(), y_indexes.size(), boundaries.size());
    // for(size_t i = 0; i < x_indexes.size(); ++i){
    //     std::cout << "("<<x_indexes[i]<<','<<y_indexes[i]<<"): {"<<boundaries[i]<<','<<(boundaries[i] < 0 ? -1 : 1)<<'}'<<std::endl;
    // }
    nt::TensorGrad boundary_grad = sig_radi_2.view(-1, 1) * sig_radi_3;
    std::cout << "boundary_grad: "<<boundary_grad<<std::endl;

    nt::Tensor mult = nt::functional::zeros(boundary_grad.shape(), nt::DType::Float32);
    const int64_t& cols = mult.shape()[-1];
    float* access = reinterpret_cast<float*>(mult.data_ptr());
    for(size_t i = 0; i < x_indexes.size(); ++i){
        access[x_indexes[i] * cols + y_indexes[i]] = boundaries[i];
    }
    boundary_grad *= mult;
    std::cout << "boundary_grad: "<<boundary_grad<<std::endl;
}


//n is the number of points
std::vector<std::vector<int64_t>> findAllPaths(const nt::Tensor& _laplacian, int64_t startNode) {
    int64_t n = _laplacian.shape()[0];
    const float* laplacian = reinterpret_cast<const float*>(_laplacian.data_ptr());
    std::vector<bool> visited(n, false);
    std::vector<std::vector<int64_t>> allPaths;  // Vector to store all paths
    std::queue<std::vector<int64_t>> q;  // Queue to store the current path during BFS
    visited[startNode] = true;
    q.push({startNode});
    
    while (!q.empty()) {
        // Get the current path
        std::vector<int64_t> currentPath = q.front();
        q.pop();
        int64_t currentNode = currentPath.back();
        
        // Check all neighbors of the current node
        for (int i = 0; i < n; ++i) {
            if (!visited[i] && std::abs(laplacian[currentNode * n + i]) > 0.1) {
                // Mark the node as visited and extend the path
                visited[i] = true;
                std::vector<int64_t> newPath = currentPath;
                newPath.push_back(i);
                q.push(newPath);
                
                // If there are no more unvisited neighbors, save the path
                allPaths.push_back(newPath);
            }
        }
    }
    
    return allPaths;
}


/*
nt::TensorGrad boundary_kp1_ = sig_radi_2.view(-1, 1) * sig_radi_3;
    nt::TensorGrad boundary_k_ = sig_radi_1.view(-1, 1) * sig_radi_2;

    nt::Tensor mult_kp1 = nt::functional::zeros(boundary_kp1_.shape(), nt::DType::Float32);
    const int64_t& cols_kp1 = mult_kp1.shape()[-1];
    float* access_kp1 = reinterpret_cast<float*>(mult_kp1.data_ptr());
    for(size_t i = 0; i < x_indexes_kp1.size(); ++i){
        access_kp1[x_indexes_kp1[i] * cols_kp1 + y_indexes_kp1[i]] = (boundaries_kp1[i] < 1 ? -1 : 1.0);
    }
    nt::TensorGrad boundary_kp1 = boundary_kp1_ * mult_kp1;

    nt::Tensor mult = nt::functional::zeros(boundary_k_.shape(), nt::DType::Float32);
    const int64_t& cols_k = mult.shape()[-1];
    float* access = reinterpret_cast<float*>(mult.data_ptr());
    for(size_t i = 0; i < x_indexes_k.size(); ++i){
        access[x_indexes_k[i] * cols_k + y_indexes_k[i]] = (boundaries_k[i] < 1 ? -1 : 1.0);
    }
    nt::TensorGrad boundary_k = boundary_k_ * mult;
*/

//the hodge 1d laplacian works a little differently
void hodge_laplacian_1d_gradient(){
    nt::TensorGrad cloud(nt::tda::generate_random_cloud({9, 9, 9}).to(nt::DType::Float32));
    nt::TensorGrad dist_mat = nt::tda::cloudToDist(cloud, 1);
    nt::TensorGrad weighted_distance(nt::functional::randn(dist_mat.shape()));
    dist_mat *= weighted_distance;
    auto [simplex_complex_2, radi_2] = nt::tda::VRfiltration(dist_mat, 2, 3.0);
    auto [simplex_complex_1, radi_1] = nt::tda::VRfiltration(dist_mat, 1);
    radi_1 = 1.0; //should be all zeros
    float alpha = 2.0;
    nt::TensorGrad sig_radi_2 = nt::functional::sigmoid(alpha * radi_2);
    nt::TensorGrad sig_radi_1 = nt::functional::sigmoid(alpha * radi_1);
    
    auto [x_indexes_k, y_indexes_k, boundaries_k] = 
            nt::tda::compute_differentiable_boundary_sparse_matrix_index(simplex_complex_2, simplex_complex_1,
                                                                         sig_radi_2.detach(), sig_radi_1.detach());
    
    nt::TensorGrad boundary_k = sig_radi_1.view(-1, 1) * sig_radi_2;
   
    nt::Tensor mult = nt::functional::ones(boundary_k.shape(), nt::DType::Float32);
    nt::Tensor mult_2 = nt::functional::zeros(boundary_k.shape(), nt::DType::Float32);
    const int64_t& cols = mult.shape()[-1];
    float* access = reinterpret_cast<float*>(mult.data_ptr());
    float* access2 = reinterpret_cast<float*>(mult_2.data_ptr());
    for(size_t i = 0; i < x_indexes_k.size(); ++i){
        access[x_indexes_k[i] * cols + y_indexes_k[i]] = (boundaries_k[i] < 1 ? -1 : 1.0);
        access2[x_indexes_k[i] * cols + y_indexes_k[i]] = 1;
    }
    boundary_k.detach() *= mult_2;
    boundary_k *= mult;
    
    auto hodge_laplacian = nt::functional::matmult(boundary_k, boundary_k, false, true);

    std::cout << "hodge laplacian shape: "<<hodge_laplacian.shape()<<std::endl;
    std::vector<std::vector<int64_t> > wanted_paths({ {1, 2, 5, 6},
                                        {0, 8, 11},
                                        {9, 3, 10, 4, 5, 7, 8, 1} });

    nt::Tensor wanted_laplacian = nt::functional::zeros(hodge_laplacian.shape(), nt::DType::Float32);
    float* begin = reinterpret_cast<float*>(wanted_laplacian.data_ptr());
    for(const auto& path : wanted_paths){
        for(size_t i = 0; i < path.size(); ++i){
            for(size_t j = 0; j < path.size(); ++j){
                if(j == i) continue;
                begin[path[i] * wanted_laplacian.shape()[0] + path[j]] = 1;
                begin[path[j] * wanted_laplacian.shape()[0] + path[i]] = 1;
            }
        }
        
    }
    // std::cout << "wanted laplacian: "<<wanted_laplacian<<std::endl;
    nt::Tensor gradient = std::pow(wanted_laplacian - hodge_laplacian.detach(), 2) / wanted_laplacian.numel();
    hodge_laplacian.backward(-gradient);
    // std::cout << "boundary k: "<<boundary_k<<std::endl;
    // std::cout << "boundary_k gradient: "<<boundary_k.grad->tensor << std::endl;
    
    // std::cout << "sig_radi_2 grad: "<<sig_radi_2.grad->tensor<<std::endl;
     
    std::cout << "distance matrix gradient: "<<dist_mat.grad()<<std::endl;
    std::cout << "weighted distance matrx gradient: "<< weighted_distance.grad(); 
    weighted_distance.update();
    std::cout << weighted_distance << std::endl;
    std::cout << simplex_complex_1 << std::endl; 

}

void hodge_laplacian_gradient(){
    nt::TensorGrad cloud(nt::tda::generate_random_cloud({9, 9, 9}).to(nt::DType::Float32));
    nt::TensorGrad dist_mat = nt::tda::cloudToDist(cloud, 1);
    nt::TensorGrad weighted_distance(nt::functional::randn(dist_mat.shape()));
    dist_mat *= weighted_distance;
    //max radius of 6.0
    auto [simplex_complex_3, radi_3] = nt::tda::VRfiltration(dist_mat, 3, 3.0);
    auto [simplex_complex_2, radi_2] = nt::tda::VRfiltration(dist_mat, 2, 3.0);
    auto [simplex_complex_1, radi_1] = nt::tda::VRfiltration(dist_mat, 1);
    // std::cout <<"radi: "<< radi_3 << std::endl;
    // std::cout <<"radi: "<< radi_2 << std::endl;
    // std::cout <<"radi: "<< radi_1 << std::endl;
    radi_1 = 1.0; //should be all zeros
    std::cout << std::boolalpha << radi_3.is_contiguous() << std::noboolalpha << std::endl;
    float alpha = 2.0;
    nt::TensorGrad sig_radi_3 = nt::functional::sigmoid(alpha * radi_3);
    nt::TensorGrad sig_radi_2 = nt::functional::sigmoid(alpha * radi_2);
    nt::TensorGrad sig_radi_1 = nt::functional::sigmoid(alpha * radi_1);
    // std::cout << "sig_radi_3: "<<sig_radi_3 << std::endl;
    // std::cout << "sig_radi_2: "<<sig_radi_2 << std::endl;
    // std::cout << "sig_radi_1: "<<sig_radi_1 << std::endl;
    auto [x_indexes_kp1, y_indexes_kp1, boundaries_kp1] = 
            nt::tda::compute_differentiable_boundary_sparse_matrix_index(simplex_complex_3, simplex_complex_2,
                                                                         sig_radi_3.detach(), sig_radi_2.detach());

   auto [x_indexes_k, y_indexes_k, boundaries_k] = 
            nt::tda::compute_differentiable_boundary_sparse_matrix_index(simplex_complex_2, simplex_complex_1,
                                                                         sig_radi_2.detach(), sig_radi_1.detach());
    


    // nt::SparseTensor old_boundary_kp1 = nt::tda::compute_boundary_matrix_index(simplex_complex_3, simplex_complex_2);
    // nt::SparseTensor old_boundary_k = nt::tda::compute_boundary_matrix_index(simplex_complex_2, simplex_complex_1);

    // for(size_t i = 0; i < x_indexes.size(); ++i){
    //     std::cout << "("<<x_indexes[i]<<','<<y_indexes[i]<<"): {"<<boundaries[i]<<','<<(boundaries[i] < 0 ? -1 : 1)<<'}'<<std::endl;
    // }
    // int64_t km1 = 6;
    // int64_t kp1 = 15;
    // int64_t k = 14;
    nt::TensorGrad boundary_kp1 = sig_radi_2.view(-1, 1) * sig_radi_3;
    nt::TensorGrad boundary_k = sig_radi_1.view(-1, 1) * sig_radi_2;

    {
        nt::Tensor mult = nt::functional::ones(boundary_kp1.shape(), nt::DType::Float32);
        nt::Tensor mult_2 = nt::functional::zeros(boundary_kp1.shape(), nt::DType::Float32);
        const int64_t& cols = mult.shape()[-1];
        float* access = reinterpret_cast<float*>(mult.data_ptr());
        float* access2 = reinterpret_cast<float*>(mult_2.data_ptr());
        for(size_t i = 0; i < x_indexes_kp1.size(); ++i){
            access[x_indexes_kp1[i] * cols + y_indexes_kp1[i]] = (boundaries_kp1[i] < 1 ? -1 : 1.0);
            access2[x_indexes_kp1[i] * cols + y_indexes_kp1[i]] = 1;
        }
        boundary_kp1.detach() *= mult_2;
        boundary_kp1 *= mult;
    }

    {
        nt::Tensor mult = nt::functional::ones(boundary_k.shape(), nt::DType::Float32);
        nt::Tensor mult_2 = nt::functional::zeros(boundary_k.shape(), nt::DType::Float32);
        const int64_t& cols = mult.shape()[-1];
        float* access = reinterpret_cast<float*>(mult.data_ptr());
        float* access2 = reinterpret_cast<float*>(mult_2.data_ptr());
        for(size_t i = 0; i < x_indexes_k.size(); ++i){
            access[x_indexes_k[i] * cols + y_indexes_k[i]] = (boundaries_k[i] < 1 ? -1 : 1.0);
            access2[x_indexes_k[i] * cols + y_indexes_k[i]] = 1;
        }
        boundary_k.detach() *= mult_2;
        boundary_k *= mult;
    }
    // boundary_kp1 = boundary_kp1[{nt::range_(0, k), nt::range_(0, kp1)}];
    // boundary_k = boundary_k[{nt::range_(0, km1), nt::range_(0, k)}];

    
    
    
    // obk -= -1;
    // obkp1 -= -1;
    auto delta_up = nt::functional::matmult(boundary_kp1, boundary_kp1, false, true);
    auto delta_down = nt::functional::matmult(boundary_k, boundary_k, true);
    auto hodge_laplacian = delta_up + delta_down;
    // std::cout << hodge_laplacian << std::endl;
    // for(int64_t i = 0; i < hodge_laplacian.shape()[0]; ++i){
    //     auto all_paths = findAllPaths(hodge_laplacian.detach(), i);
    //     for(const auto& path : all_paths){
    //         std::cout << "path: ";
    //         for(const auto& element : path)
    //             std::cout << element<< ' ';
    //         std::cout << std::endl;
    //     }
    // }
    
    //example wanted paths:
    std::vector<std::vector<int64_t> > wanted_paths({ {1, 2, 5, 6},
                                        {0, 8, 11, 12},
                                        {9, 3, 10, 4, 5, 7, 8, 1} });

    nt::Tensor wanted_laplacian = nt::functional::zeros(hodge_laplacian.shape(), nt::DType::Float32);
    float* begin = reinterpret_cast<float*>(wanted_laplacian.data_ptr());
    for(const auto& path : wanted_paths){
        for(size_t i = 0; i < path.size(); ++i){
            for(size_t j = 0; j < path.size(); ++j){
                if(j == i) continue;
                begin[path[i] * wanted_laplacian.shape()[0] + path[j]] = 1;
                begin[path[j] * wanted_laplacian.shape()[0] + path[i]] = 1;
            }
        }
        
    }
    // std::cout << "wanted laplacian: "<<wanted_laplacian<<std::endl;
    nt::Tensor gradient = std::pow(wanted_laplacian - hodge_laplacian.detach(), 2) / wanted_laplacian.numel();
    hodge_laplacian.backward(gradient);
    std::cout << "boundary kp1: "<<boundary_kp1<<std::endl;
    std::cout << "boundary_kp1 gradient: "<<boundary_kp1.grad() << std::endl;
    
    std::cout << "sig_radi_2 grad: "<<sig_radi_2.grad()<<std::endl;
    std::cout << "sig_radi_3 grad: "<<sig_radi_3.grad()<<std::endl;
     
    std::cout << "distance matrix gradient: "<<dist_mat.grad()<<std::endl;
    std::cout << "weighted distance matrx gradient: "<< weighted_distance.grad(); 
}



std::vector<int64_t> generateUniqueRandom(int64_t n, int64_t min, int64_t max) {
    if (n > (max - min + 1)) {
        throw std::invalid_argument("Cannot generate more unique numbers than the range allows.");
    }
    if (min > max) {
        throw std::invalid_argument("Invalid range: min must be less than or equal to max.");
    }
    if (n <= 0) {
        return {}; // Return an empty vector for non-positive n
    }

    std::vector<int64_t> pool(max - min + 1);
    for (int i = 0; i < pool.size(); ++i) {
        pool[i] = min + i;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(pool.begin(), pool.end(), gen);

    std::vector<int64_t> result(pool.begin(), pool.begin() + n);
    return result;
}

nt::Tensor generate_random_paths(int64_t i, int64_t total, int64_t max_depth){
    nt::Tensor output = nt::Tensor::makeNullTensorArray(i);
    nt::Tensor* begin = reinterpret_cast<nt::Tensor*>(output.data_ptr());
    nt::Tensor* end = reinterpret_cast<nt::Tensor*>(output.data_ptr_end());
    std::default_random_engine generator;
    std::uniform_int_distribution<int64_t> distribution(1, max_depth);
    for(;begin != end; ++begin){
        *begin = nt::functional::vector_to_tensor(generateUniqueRandom(distribution(generator), 0, total));
    }
    return std::move(output);
}

void nn_laplacian_1d_test(){
    nt::TensorGrad cloud(nt::tda::generate_random_cloud({20, 20}).to(nt::DType::Float32));
    nt::TensorGrad weight(nt::Tensor::Null());
    nt::Tensor target_paths = nt::Tensor::Null();
    double lr = 0.01;
    
    for(int i = 0; i < 100; ++i){
        nt::TensorGrad dist_mat = nt::tda::cloudToDist(cloud, 1);
        if(weight.detach().is_null()){
            weight.detach() = nt::functional::randn( dist_mat.shape());
            std::cout << "weight: "<<weight<<std::endl;
        }
        dist_mat *= weight;
        dist_mat = nt::functional::sigmoid(dist_mat);
        //distance matrix, k = 2, max radius = 6.0, alpha = 5.0, sigmoid already applied
        auto [laplacian, simplexes] = nt::tda::hodge_laplacian(dist_mat, 1, 6.0);
        nt::TensorGrad paths = nt::tda::findAllPaths(laplacian);
        if(target_paths.is_null()){
            target_paths = generate_random_paths(10, laplacian.shape()[0]-1, 3);
            // std::cout << paths<<std::endl;
            // std::cout << "target: "<<target_paths<<std::endl;
            std::cout << laplacian << std::endl;

        }
        auto loss = nt::tda::loss::path_loss(paths, target_paths);
        std::cout << "loss: "<< loss << " with "<<paths.numel()<<" paths "<<std::endl;
        loss.backward();
        if(i == 0){
            // std::cout << paths.grad->tensor<<std::endl;
            std::cout << laplacian.grad() <<std::endl;
            std::cout << weight.grad() << std::endl;
            std::cout << simplexes << std::endl;
        }
        // weight.grad->tensor *= 40;
        weight.update();
        if(i > 50) lr = 0.1;
        if(i == 99){
            // std::cout << paths << std::endl;
            std::cout << laplacian<<std::endl;
            std::cout << simplexes<<std::endl;
        }
    }

    std::cout << weight << std::endl;

}

void nn_laplacian_2_test(){
    nt::Tensor cloud(nt::tda::generate_random_cloud({15, 15}).to(nt::DType::Float32));
    nt::Tensor dist_mat(nt::tda::cloudToDist(cloud, 1));

    nt::TensorGrad weight1(nt::Tensor::Null());
    nt::TensorGrad bias1(nt::Tensor::Null());
    nt::TensorGrad weight2(nt::Tensor::Null());
    nt::TensorGrad bias2(nt::Tensor::Null());
    nt::Tensor target_paths = nt::Tensor::Null();
    //target simplex path is needed because the actual simplex order is going to move quite a bit
    nt::Tensor target_simplex_path = nt::Tensor::Null();
    double lr = 0.001;
    double cur_max_radi = -1.0;
    int64_t iterations = 20;
    for(int i = 0; i < iterations; ++i){
        std::cout << "[ITERATION  "<<i<<"]:"<<std::endl;
        // nt::TensorGrad dist_mat = nt::tda::cloudToDist(cloud, 1);
        if(weight1.detach().is_null()){
            weight1 = nt::TensorGrad(nt::functional::randn(dist_mat.shape()));
            bias1 = nt::TensorGrad(nt::functional::randn(dist_mat.shape()));
            weight2 = nt::TensorGrad(nt::functional::randn(dist_mat.shape()));
            bias2 = nt::TensorGrad(nt::functional::randn(dist_mat.shape()));
        }
        // nt::TensorGrad temp_mult1 = nt::functional::matmult(dist_mat, weight1);
        // nt::TensorGrad redefine_temp1(temp_mult1.detach());
        // nt::TensorGrad::redefine_tracking(redefine_temp1, temp_mult1,
        //                                   [](const nt::Tensor& grad, nt::intrusive_ptr<nt::TensorGrad>& parent){
	    	                            // std::cout << "[!] -> -> operator+ redefined tracking backward called [!] [!]" <<std::endl;
        //                                   parent->grad->tensor = grad;
    	//    });

        nt::TensorGrad dist_mat1 = nt::functional::matmult(dist_mat, weight1) + bias1;
        dist_mat1 = nt::functional::tanh(dist_mat1);
        nt::TensorGrad dist_mat2 = nt::functional::matmult(dist_mat, weight2) + bias2;
        dist_mat2 = nt::functional::tanh(dist_mat2);
        // dist_mat = nt::functional::softmax(dist_mat);
        //distance matrix, k = 2, max radius = 6.0, alpha = 5.0
        auto [laplacian, simplexes] = nt::tda::hodge_laplacian(dist_mat1, dist_mat2, 2, cur_max_radi);
        nt::TensorGrad paths = nt::tda::findAllPaths(laplacian);
        if(target_paths.is_null()){
            target_paths = generate_random_paths(10, laplacian.shape()[0]-1, 5);
            target_simplex_path = nt::Tensor::makeNullTensorArray(target_paths.numel());
            nt::Tensor* pth_b = reinterpret_cast<nt::Tensor*>(target_paths.data_ptr());
            nt::Tensor* pth_e = reinterpret_cast<nt::Tensor*>(target_paths.data_ptr_end());
            nt::Tensor* out_b = reinterpret_cast<nt::Tensor*>(target_simplex_path.data_ptr());
            nt::Tensor split = simplexes.split_axis(-2);
            nt::Tensor* scx_b = reinterpret_cast<nt::Tensor*>(split.data_ptr());
            int64_t k = 2;
            for(;pth_b != pth_e; ++pth_b, ++out_b){
                nt::Tensor cur({pth_b->numel(), k}, nt::DType::int64);
                int64_t* begin = reinterpret_cast<int64_t*>(pth_b->data_ptr());
                int64_t* end = reinterpret_cast<int64_t*>(pth_b->data_ptr_end());
                int64_t* begin_o = reinterpret_cast<int64_t*>(cur.data_ptr());
                for(;begin != end; ++begin){
                    const nt::Tensor& complex = scx_b[*begin];
                    const int64_t* cpy_b = reinterpret_cast<const int64_t*>(complex.data_ptr());
                    for(int64_t i = 0; i < k; ++i, ++begin_o, ++cpy_b){
                        *begin_o = *cpy_b;
                    }
                }
                *out_b = cur;
            }
            // std::cout << paths<<std::endl;
            // std::cout << "target: "<<target_paths<<std::endl;
            // std::cout << "laplacian: "<<std::endl;
            // std::cout << laplacian << std::endl;
            // std::cout << simplexes << std::endl;
            // std::cout << target_paths << std::endl;
            // std::cout << target_simplex_path << std::endl;

        }else{
            //update the target paths based on the target simplex paths
            nt::Tensor split = simplexes.split_axis(-2);
            target_paths = nt::Tensor::makeNullTensorArray(target_simplex_path.numel());
            nt::Tensor* pth_b = reinterpret_cast<nt::Tensor*>(target_paths.data_ptr());
            nt::Tensor* pth_e = reinterpret_cast<nt::Tensor*>(target_paths.data_ptr_end());
            nt::Tensor* tsx_b = reinterpret_cast<nt::Tensor*>(target_simplex_path.data_ptr());
            nt::Tensor* scx_b = reinterpret_cast<nt::Tensor*>(split.data_ptr());
            nt::Tensor* scx_e = reinterpret_cast<nt::Tensor*>(split.data_ptr_end());
            for(;pth_b != pth_e; ++pth_b, ++tsx_b){
                const int64_t& depth = tsx_b->shape()[0];
                const int64_t& k = tsx_b->shape()[1];
                nt::Tensor out({depth}, nt::DType::int64);
                int64_t* out_b = reinterpret_cast<int64_t*>(out.data_ptr());
                nt::Tensor target_split = tsx_b->split_axis(-2);
                nt::Tensor* target_b = reinterpret_cast<nt::Tensor*>(target_split.data_ptr());
                for(int64_t i = 0; i < depth; ++i){
                    auto cpy_b = scx_b;
                    for(int64_t j = 0; cpy_b != scx_e; ++cpy_b, ++j){
                        if(nt::functional::all(*cpy_b == target_b[i])){
                            out_b[i] = j;
                            break;
                        }
                    }
                }
                *pth_b = out;
            }
        }
        auto loss = nt::tda::loss::path_loss(paths, target_paths);
        std::cout << "loss: "<<loss << " with "<<paths.numel()<<" paths "<<std::endl;
        loss.backward();
        // if(i == -1){
        //     std::cout << "grads: "<<std::endl;
        //     // std::cout << paths.grad->tensor<<std::endl;
        //     std::cout << laplacian.grad->tensor<<std::endl;
        //     std::cout << weight1.grad->tensor << std::endl;
        //     std::cout << weight2.grad->tensor << std::endl;
        //     std::cout << bias2.grad->tensor << std::endl;
        //     std::cout << dist_mat1.grad->tensor << std::endl;
        //     std::cout << dist_mat2.grad->tensor << std::endl;
        // }
        if(i < 75){
        weight1.grad() *= lr;
        weight1.update();
        bias1.grad() *= lr;
        bias1.update();
        weight2.grad() *= lr;
        weight2.update();
        bias2.grad() *= lr;
        bias2.update();
        // if(i > 50) lr = 0.1;
        }
        if(i == iterations-1){
            nt::Tensor wanted_laplacian = nt::functional::zeros(laplacian.shape(), nt::DType::Float32);
            float* begin = reinterpret_cast<float*>(wanted_laplacian.data_ptr());
            // std::cout << paths << std::endl;
            const nt::Tensor* paths_begin = reinterpret_cast<const nt::Tensor*>(target_paths.data_ptr());
            const nt::Tensor* paths_end = paths_begin + target_paths.numel();
            for(;paths_begin != paths_end; ++paths_begin){
                // std::cout << "working with path "<<paths_begin->dtype<<std::endl;
                // std::cout << std::boolalpha << paths_begin->is_null() << std::endl;
                // std::cout << std::boolalpha << paths_begin->l() << std::endl;
                const int64_t* path_b = reinterpret_cast<const int64_t*>(paths_begin->data_ptr());
                const int64_t* path_e = reinterpret_cast<const int64_t*>(paths_begin->data_ptr_end());
                const int64_t size = path_e-path_b;
                for(size_t i = 0; i < size; ++i){
                    for(size_t j = 0; j < size; ++j){
                        if(j == i) continue;
                        if(path_b[i] > wanted_laplacian.shape()[0] || path_b[j] > wanted_laplacian.shape()[0]) continue;
                        begin[path_b[i] * wanted_laplacian.shape()[0] + path_b[j]] = 1;
                        begin[path_b[j] * wanted_laplacian.shape()[0] + path_b[i]] = 1;
                    }
                }
                
            }
            std::cout << "wanted laplacian: "<<wanted_laplacian<<std::endl;
            std::cout << "laplacian: "<<std::endl;
            std::cout << laplacian<<std::endl;
            std::cout << simplexes << std::endl;
            std::cout << "where != 0:"<<std::endl;
            laplacian.detach()[wanted_laplacian != 0].print();
            std::cout << nt::functional::count(laplacian.detach()[wanted_laplacian != 0] != 0) << std::endl;
            std::cout << dist_mat1.grad() << std::endl;
        }
        // break;
    }

    std::cout << weight1 << std::endl;
    std::cout << weight2 << std::endl;

}


/*

dL: Tensor([[0,1,0,1,0,0,1,0,0,0,0,0,1,0,0],
         [1,0,0,0,1,1,1,1,0,0,1,1,1,0,1],
         [0,0,0,0,0,0,0,1,1,0,0,1,0,1,0],
         [1,0,0,0,1,0,1,0,1,1,0,1,0,0,1],
         [0,1,0,1,0,0,0,0,0,0,0,1,0,0,1],
         [0,1,0,0,0,0,0,0,0,0,1,0,0,0,0],
         [1,1,0,1,0,0,0,0,1,0,0,0,0,0,0],
         [0,1,1,0,0,0,0,0,1,0,0,1,1,1,0],
         [0,0,1,1,0,0,1,1,0,0,0,1,0,1,0],
         [0,0,0,1,0,0,0,0,0,0,0,0,0,0,1],
         [0,1,0,0,0,1,0,0,0,0,0,0,0,0,0],
         [0,1,1,1,1,0,0,1,1,0,0,0,0,1,1],
         [1,1,0,0,0,0,0,1,0,0,0,0,0,1,0],
         [0,0,1,0,0,0,0,1,1,0,0,1,1,0,0],
         [0,1,0,1,1,0,0,0,0,1,0,1,0,0,0]], {15,15})

dL: Tensor([[0,1,0,0,1,0,0,1,0,0,0,0,0,1,0],
         [1,0,0,0,0,1,1,1,1,0,0,1,1,1,1],
         [0,0,0,0,1,0,0,0,1,1,1,0,1,1,1],
         [0,0,0,0,0,0,0,0,1,1,0,0,1,0,0],
         [1,0,1,0,0,1,0,1,0,1,1,0,1,0,1],
         [0,1,0,0,1,0,0,0,0,0,0,0,1,0,1],
         [0,1,0,0,0,0,0,0,0,0,0,1,0,0,0],
         [1,1,0,0,1,0,0,0,0,1,0,0,0,0,0],
         [0,1,1,1,0,0,0,0,0,1,0,0,1,1,0],
         [0,0,1,1,1,0,0,1,1,0,0,0,1,0,0],
         [0,0,1,0,1,0,0,0,0,0,0,0,0,0,1],
         [0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],
         [0,1,1,1,1,1,0,0,1,1,0,0,0,0,1],
         [1,1,1,0,0,0,0,0,1,0,0,0,0,0,0],
         [0,1,1,0,1,1,0,0,0,0,1,0,1,0,0]], {15,15})
*/

void nn_laplacian_2_test_sub(){
    nt::TensorGrad cloud(nt::tda::generate_random_cloud({15, 15}).to(nt::DType::Float32));
    nt::TensorGrad dist_mat = nt::tda::cloudToDist(cloud, 1);
    auto [laplacian, simplexes] = nt::tda::hodge_laplacian(dist_mat, 2, -1.0);
    std::cout << simplexes<<std::endl;
    nt::Tensor wanted_paths = generate_random_paths(10, laplacian.shape()[0]-1, 3);
    
    nt::Tensor wanted_laplacian = nt::functional::zeros(laplacian.shape(), nt::DType::Float32);
    float* begin = reinterpret_cast<float*>(wanted_laplacian.data_ptr());
    // std::cout << "wanted paths: "<<wanted_paths<<std::endl;
    const nt::Tensor* paths_begin = reinterpret_cast<const nt::Tensor*>(wanted_paths.data_ptr());
    const nt::Tensor* paths_end = paths_begin + wanted_paths.numel();
    for(;paths_begin != paths_end; ++paths_begin){
        // std::cout << "working with path "<<paths_begin->dtype<<std::endl;
        // std::cout << std::boolalpha << paths_begin->is_null() << std::endl;
        // std::cout << std::boolalpha << paths_begin->l() << std::endl;
        const int64_t* path_b = reinterpret_cast<const int64_t*>(paths_begin->data_ptr());
        const int64_t* path_e = reinterpret_cast<const int64_t*>(paths_begin->data_ptr_end());
        const int64_t size = path_e-path_b;
        for(size_t i = 0; i < size; ++i){
            for(size_t j = 0; j < size; ++j){
                if(j == i) continue;
                if(path_b[i] > wanted_laplacian.shape()[0] || path_b[j] > wanted_laplacian.shape()[0]) continue;
                begin[path_b[i] * wanted_laplacian.shape()[0] + path_b[j]] = 1;
                begin[path_b[j] * wanted_laplacian.shape()[0] + path_b[i]] = 1;
            }
        }
        
    }
    //setting the diagonal of the wanted laplacian
    const int64_t N = wanted_laplacian.shape()[0];
    for (int64_t i = 0; i < N; ++i) {
        float degree = 0.0f;
        for (int64_t j = 0; j < N; ++j) {
            if (i != j) {
                degree += begin[i * N + j];
            }
        }
        begin[i * N + i] = degree;
    }
    
    laplacian.detach()[laplacian.detach() > 1] = 1;
    laplacian.detach()[laplacian.detach() < -1] = -1;
    std::cout << "laplacian: "<<laplacian<<std::endl<<"wanted laplacian: "<<wanted_laplacian<<std::endl;
    nt::Tensor dL = std::pow(std::abs(laplacian.detach()) - wanted_laplacian, 2);
    dL.fill_diagonal_(0.0);
    std::cout << "dL: "<<dL<<std::endl;
    laplacian.backward(dL);
    std::cout << "ran backward"<<std::endl;
    
    // auto [laplacian_2, simplexes_2] = nt::tda::hodge_laplacian(dist_mat, 2, -1.0, 5.0, true);
    // std::cout  << simplexes_2<<std::endl;
}


std::pair<int64_t, int64_t> num_row_cols(const nt::Tensor& t){
    const float* begin = reinterpret_cast<const float*>(t.data_ptr());
    const int64_t& rows = t.shape()[0];
    const int64_t& cols = t.shape()[1];
    const float* r_begin = begin;
    const float* r_end = r_begin + cols;
    int64_t row_cntr = 0;
    for(;r_begin != r_end; ++r_begin){
        if(*r_begin != 0)
            ++row_cntr;
    }
    int64_t col_cntr = 0;
    for(int64_t i = 0; i < rows; ++i, begin += cols){
        if(*begin != 0) ++col_cntr;
    }
    return {row_cntr, col_cntr};
}



void nn_boundary_test(){
    // nt::Tensor cloud = nt::tda::generate_random_cloud({15, 15});
    // std::cout << cloud << std::endl;
    // std::cout << nt::functional::count(cloud == 1) << std::endl;
    nt::Tensor _cloud({15, 15}, nt::DType::int8);
    _cloud <<    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,
                 0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,
                 0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;   
    nt::Tensor wanted_boundary({15, 20}, nt::DType::Float32);
    wanted_boundary <<   0,0,0,1,0,0,0,0,0,-1,0,0,1,0,0,0,0,0,-1,0,
                         -1,0,0,0,0,0,0,-1,0,0,0,-1,0,0,1,0,0,-1,0,0,
                         -1,0,0,0,0,0,-1,0,0,0,0,0,-1,0,0,0,0,0,-1,0,
                         0,0,0,-1,0,0,0,0,-1,0,0,0,0,-1,0,0,-1,0,0,0,
                         0,0,0,0,0,-1,0,0,0,0,-1,0,0,-1,0,0,0,0,0,0,
                         0,0,-1,0,0,-1,0,0,0,0,0,0,-1,0,-1,0,0,0,0,0,
                         -1,0,0,0,-1,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,
                         0,-1,0,0,-1,0,0,-1,-1,0,0,0,0,0,0,1,0,0,0,0,
                         0,0,-1,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,
                         0,0,0,1,0,-1,-1,0,1,0,0,0,0,0,0,0,0,0,0,0,
                         0,0,0,0,0,0,1,-1,0,0,-1,0,0,0,0,0,-1,0,0,0,
                         1,0,0,0,0,0,0,0,-1, -1,0,0,0,0,0,-1,0,-1,0,0,
                         0,0,0,0,0,0,0,0,0,0,-1,-1,-1,0,0,0,0,0,0,-1,
                         0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,0,0,0,-1,
                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1;

    nt::TensorGrad cloud(_cloud.to(nt::DType::Float32));
    nt::TensorGrad _dist_mat = nt::tda::cloudToDist(cloud, 1);
    nt::TensorGrad weight(nt::functional::randn(_dist_mat.shape()));
    nt::TensorGrad dist_mat = _dist_mat * weight;

    auto [simplex_complex_3, radi_3] = nt::tda::VRfiltration(dist_mat, 3, -1.0);
    auto [simplex_complex_2, radi_2] = nt::tda::VRfiltration(dist_mat, 2, -1.0);
    std::cout << "radi 2: "<<radi_2<<std::endl;
    std::cout << "radi_3: "<<radi_3 << std::endl;
    std::cout << "simplex_complex_2: " << simplex_complex_2 << std::endl;
    std::cout << "simplex_complex_3: " << simplex_complex_3 << std::endl;
    // auto [simplex_complex_1, radi_1] = nt::tda::VRfiltration(dist_mat, 1);

    nt::TensorGrad boundary = nt::tda::BoundaryMatrix(simplex_complex_3,  simplex_complex_2, radi_3,radi_2);  
    std::cout << "boundary: " << boundary<<std::endl; 
    
    nt::Tensor grad = boundary.detach() - wanted_boundary;
    boundary.backward(grad);
    std::cout << "old weight: "<<weight<<std::endl;
    weight.update();
    std::cout << "new weight: "<<weight<<std::endl;
    
    nt::TensorGrad dist_mat_2 = _dist_mat * weight;
    auto [simplex_complex_3_2, radi_3_2] = nt::tda::VRfiltration(dist_mat_2, 3, -1.0);
    auto [simplex_complex_2_2, radi_2_2] = nt::tda::VRfiltration(dist_mat_2, 2, -1.0);
    std::cout << "radi 2_2: "<<radi_2_2<<std::endl;

    nt::TensorGrad boundary_2 = nt::tda::BoundaryMatrix(simplex_complex_3_2,  simplex_complex_2_2, radi_3_2,radi_2_2);  
    std::cout << "boundary_2: " << boundary_2<<std::endl;
    std::cout << nt::functional::where(boundary_2 != 0) << std::endl;
    std::cout << nt::functional::where(boundary != 0) << std::endl;

}



inline nt::TensorGrad pairwise_l2(const nt::TensorGrad& x, const nt::TensorGrad& y){
    return (x.view(-1, 1) - y.view(1, -1)).pow(2);
}

inline nt::Tensor sample_gumbel(const nt::TensorGrad& logits){
    nt::Tensor u = nt::functional::rand(0, 1, logits.shape(), logits.dtype()); //uniform (0,1)
    return -nt::functional::log(-nt::functional::log(u + 1e-10));       // Gumbel(0,1)
}

nt::TensorGrad gumbel_softmax(const nt::TensorGrad& logits, float tau, bool hard = false) {
    nt::Tensor gumbel_noise = sample_gumbel(logits);
    // std::cout << "gumble noise is "<<nt::noprintdtype<<gumbel_noise<<nt::printdtype<<std::endl;
    // std::cout << "logits are: "<<logits<<std::endl;
    nt::TensorGrad y = ((logits*100) + gumbel_noise) / tau;
    std::cout << "y is "<<nt::noprintdtype<<y<<nt::printdtype<<std::endl;
    y = nt::functional::softmax(y, -1); // apply softmax along last dim
    std::cout << "y is "<<nt::noprintdtype<<y<<nt::printdtype<<std::endl;

    if (hard) {
        // Straight-through: make y_hard one-hot
        nt::Tensor y_hard = nt::functional::one_hot(nt::functional::argmax(y.detach(), -1), y.shape()[-1]).to(y.dtype());
        // Use straight-through estimator
        return (y_hard - y).detach() + y;
    }

    return y;
}


nt::TensorGrad perform_row_swap(const nt::Tensor& matrix, const nt::TensorGrad& row_vec){
    nt::TensorGrad logits_r = pairwise_l2(row_vec, row_vec);
    nt::TensorGrad P_row = gumbel_softmax(logits_r, 1.0, true);
    return nt::functional::matmult(P_row, matrix);


}

//this is a function that experimentally checks if rows and columns can learn to be swapped
void row_swap_test(){
    

    auto critereon = nt::loss::raw_error;
    nt::Tensor matrix({6, 15}, nt::DType::Float32);
    matrix << 0,0,0,0,0,0,0,-1,0,-1,0,-1,-1,0,-1,
              0,0,0,-1,-1,-1,0,0,0,0,0,0,0,-1,1,
              0,-1,-1,0,0,-1,0,0,0,0,-1,0,1,0,0,
              0,0,0,0,1,0,-1,0,-1,1,1,0,0,0,0,
              -1,0,-1,-1,0,0,1,1,0,0,0,0,0,0,0,
              -1,-1,0,0,0,0,0,0,1,0,0,1,0,-1,0;

    nt::Tensor wanted_matrix({6, 15}, nt::DType::Float32);
    wanted_matrix <<  0,0,0,0,0,0,0,-1,0,-1,0,-1,-1,0,-1,
                      -1,0,-1,-1,0,0,1,1,0,0,0,0,0,0,0,
                      0,0,0,0,1,0,-1,0,-1,1,1,0,0,0,0,
                      0,-1,-1,0,0,-1,0,0,0,0,-1,0,1,0,0,
                      0,0,0,-1,-1,-1,0,0,0,0,0,0,0,-1,1,
                      -1,-1,0,0,0,0,0,0,1,0,0,1,0,-1,0;

    nt::TensorGrad row_vec(nt::functional::rand(0.1, 2.3, {matrix.shape()[0]}, nt::DType::Float32));
    std::cout << nt::noprintdtype << row_vec << nt::printdtype << std::endl;
    int iterations = 10;
    float lr = 1.0;
    for(int i = 0; i < iterations; ++i){
        nt::TensorGrad swapped = perform_row_swap(matrix, nt::functional::tanh(row_vec));
        if(i == 0 || i == iterations-1) std::cout << swapped << std::endl;
        nt::ScalarGrad loss = critereon(swapped, wanted_matrix);
        std::cout << "loss: "<<loss << std::endl;
        loss.backward();
        // std::cout << row_vec.grad->tensor  << std::endl;
        // row_vec.grad->tensor *= lr;
        row_vec.update();
        std::cout << nt::noprintdtype << row_vec << nt::printdtype << std::endl;
    }
    std::cout << nt::noprintdtype << row_vec << nt::printdtype << std::endl;

    
}
