#include "laplacian.h"
#include "filtration.h"
#include "boundaries.h"

#include "../../nn/functional.h"
#include "../../functional/functional.h" //zeros, abs, vector_to_tensor
#include "../Boundaries.h"

namespace nt {
namespace tda {

// this takes k-1 radi, k radi, and k+1 radi and returns a laplacian
TensorGrad hodge_laplacian(TensorGrad radi_1, TensorGrad radi_2,
                           TensorGrad radi_3, Tensor simplex_complex_1,
                           Tensor simplex_complex_2, Tensor simplex_complex_3){
    auto [x_indexes_kp1, y_indexes_kp1, boundaries_kp1] = 
            compute_differentiable_boundary_sparse_matrix_index(simplex_complex_3, simplex_complex_2,
                                                                         radi_3.tensor, radi_2.tensor);
    auto [x_indexes_k, y_indexes_k, boundaries_k] = 
            compute_differentiable_boundary_sparse_matrix_index(simplex_complex_2, simplex_complex_1,
                                                                         radi_2.tensor, radi_1.tensor);
    TensorGrad boundary_kp1 = radi_2.view(-1, 1) * radi_3;
    TensorGrad boundary_k = radi_1.view(-1, 1) * radi_2;
    {
        Tensor mult = functional::zeros(boundary_kp1.shape(), DType::Float32);
        const int64_t& cols = mult.shape()[-1];
        float* access = reinterpret_cast<float*>(mult.data_ptr());
        for(size_t i = 0; i < x_indexes_kp1.size(); ++i){
            access[x_indexes_kp1[i] * cols + y_indexes_kp1[i]] = (boundaries_kp1[i] < 1 ? -1 : 1.0);
        }
        boundary_kp1 *= mult.to(boundary_kp1.dtype);
    }
    {
        Tensor mult = functional::zeros(boundary_k.shape(), DType::Float32);
        const int64_t& cols = mult.shape()[-1];
        float* access = reinterpret_cast<float*>(mult.data_ptr());
        for(size_t i = 0; i < x_indexes_k.size(); ++i){
            access[x_indexes_k[i] * cols + y_indexes_k[i]] = (boundaries_k[i] < 1 ? -1 : 1.0);
        }
        boundary_k *= mult.to(DType::Float32);
    }
    TensorGrad delta_up = functional::matmult(boundary_kp1, boundary_kp1, false, true);
    TensorGrad delta_down = functional::matmult(boundary_k, boundary_k, true);
    TensorGrad _hodge_laplacian = delta_up + delta_down;
    return std::move(_hodge_laplacian);

}
// this takes a distance matrix, and returns a hodge laplacian differentiable
//returns the gradient laplacian, and the simplex which corresponds to points
std::tuple<TensorGrad, Tensor> hodge_laplacian(TensorGrad distance_matrix, int64_t k, double max_radi){
    utils::throw_exception(k >= 1, "Expected k for hodge laplacian to be greater than or equal to 1, got $", k);
    //this is the 0-hodge laplacian
    //the k numbering may need to be updated to reflect the actual mathematics better
    //it is just boundary_1 * boundary_1.transpose(-1,-2)
    if(k == 1){
        //single points
        TensorGrad radi_1(functional::ones({distance_matrix.shape()[0]}, DType::Float));
        Tensor simplex_complex_1 = functional::arange(distance_matrix.shape()[0], DType::int64).view(-1, 1);
        auto [simplex_complex_2, radi_2] = VRfiltration(distance_matrix, 2, max_radi, false);
        TensorGrad boundary = BoundaryMatrix(simplex_complex_2, simplex_complex_1,
                                             radi_2, radi_1);
        TensorGrad _hodge_laplacian = functional::matmult(boundary, boundary, false, true);
        return {std::move(_hodge_laplacian), simplex_complex_2};
    }

    auto [simplex_complex_kp1, radi_kp1] = VRfiltration(distance_matrix, k+1, max_radi, false);
    auto [simplex_complex_k, radi_k] = VRfiltration(distance_matrix, k, max_radi, false);
    auto [simplex_complex_km1, radi_km1] = VRfiltration(distance_matrix, k-1, max_radi, false);
    if(k == 2)
        radi_km1 = 1.0; //instead of all 0's
    // TensorGrad sig_radi_kp1 = apply_sigmoid ? functional::sigmoid(alpha * radi_kp1) : radi_kp1;
    // TensorGrad sig_radi_k = apply_sigmoid ? functional::sigmoid(alpha * radi_k) : radi_k;
    // TensorGrad sig_radi_km1 = apply_sigmoid ? functional::sigmoid(alpha * radi_km1) : radi_km1;
    
    TensorGrad boundary_kp1 = BoundaryMatrix(simplex_complex_kp1, simplex_complex_k,
                                                radi_kp1, radi_k);
    TensorGrad boundary_k = BoundaryMatrix(simplex_complex_k, simplex_complex_km1,
                                                radi_k, radi_km1);
    // nt::Tensor printing = boundary_k.tensor.clone();
    // printing[printing < 0] = -1;
    // printing[printing > 0] = 1;
    // std::cout << "boundary_k: "<<printing<<std::endl;
    // std::cout << "boundary k: "<<boundary_k<<std::endl;
    

    TensorGrad delta_up = functional::matmult(boundary_kp1, boundary_kp1, false, true);
    TensorGrad delta_down = functional::matmult(boundary_k, boundary_k, true);
    TensorGrad _hodge_laplacian = delta_up + delta_down;
    
    // Tensor printing = _hodge_laplacian.tensor.clone();

    return {std::move(_hodge_laplacian), simplex_complex_k};
}



// this takes a distance matrix, and returns a hodge laplacian differentiable
//returns the gradient laplacian, and the simplex which corresponds to points
std::tuple<TensorGrad, Tensor> hodge_laplacian(TensorGrad distance_matrix1, TensorGrad distance_matrix2, int64_t k, double max_radi){
    utils::throw_exception(k >= 1, "Expected k for hodge laplacian to be greater than or equal to 1, got $", k);
    //this is the 0-hodge laplacian
    //the k numbering may need to be updated to reflect the actual mathematics better
    //it is just boundary_1 * boundary_1.transpose(-1,-2)
    if(k == 1){
        //single points
        TensorGrad radi_1(functional::ones({distance_matrix1.shape()[0]}, DType::Float));
        TensorGrad radi_1_2(functional::ones({distance_matrix1.shape()[0]}, DType::Float));
        Tensor simplex_complex_1 = functional::arange(distance_matrix1.shape()[0], DType::int64).view(-1, 1);
        auto [simplex_complex_2, radi_2, radi_2_2] = VRfiltration(distance_matrix1, distance_matrix2, 2, max_radi, false);
        TensorGrad boundary = BoundaryMatrix(simplex_complex_2, simplex_complex_1,
                                             radi_2, radi_1, radi_2_2, radi_1_2);
        TensorGrad _hodge_laplacian = functional::matmult(boundary, boundary, false, true);
        return {std::move(_hodge_laplacian), simplex_complex_2};
    }

    auto [simplex_complex_kp1, radi_kp1, radi_kp1_2] = VRfiltration(distance_matrix1, distance_matrix2, k+1, max_radi, false);
    auto [simplex_complex_k, radi_k, radi_k_2] = VRfiltration(distance_matrix1, distance_matrix2, k, max_radi, false);
    auto [simplex_complex_km1, radi_km1, radi_km1_2] = VRfiltration(distance_matrix1, distance_matrix2, k-1, max_radi, false);
    if(k == 2)
        radi_km1 = 1.0; //instead of all 0's
    // TensorGrad sig_radi_kp1 = apply_sigmoid ? functional::sigmoid(alpha * radi_kp1) : radi_kp1;
    // TensorGrad sig_radi_k = apply_sigmoid ? functional::sigmoid(alpha * radi_k) : radi_k;
    // TensorGrad sig_radi_km1 = apply_sigmoid ? functional::sigmoid(alpha * radi_km1) : radi_km1;
    
    TensorGrad boundary_kp1 = BoundaryMatrix(simplex_complex_kp1, simplex_complex_k,
                                                radi_kp1, radi_k,
                                                radi_kp1_2, radi_k_2);
    TensorGrad boundary_k = BoundaryMatrix(simplex_complex_k, simplex_complex_km1,
                                                radi_k, radi_km1,
                                                radi_k_2, radi_km1_2);
    // nt::Tensor printing = boundary_k.tensor.clone();
    // printing[printing < 0] = -1;
    // printing[printing > 0] = 1;
    // std::cout << "boundary_k: "<<printing<<std::endl;
    // std::cout << "boundary k: "<<boundary_k<<std::endl;
    

    TensorGrad delta_up = functional::matmult(boundary_kp1, boundary_kp1, false, true);
    TensorGrad delta_down = functional::matmult(boundary_k, boundary_k, true);
    TensorGrad _hodge_laplacian = delta_up + delta_down;
    
    // Tensor printing = _hodge_laplacian.tensor.clone();

    return {std::move(_hodge_laplacian), simplex_complex_k};
}

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
            //EPSILON DEFINED HERE PER THE FUNCTION BELOW
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


inline double sigmoid(double x){
    return 1 / (1 + std::exp(-x));
}

TensorGrad findAllPaths(const TensorGrad& laplacian){
    std::vector<std::vector<int64_t>> all_paths;
    for(int64_t i = 0; i < laplacian.shape()[0]; ++i){
        std::vector<std::vector<int64_t>> paths = findAllPaths(laplacian.tensor, i);
        all_paths.insert(all_paths.end(), paths.begin(), paths.end());
    }
    Tensor out = Tensor::makeNullTensorArray(all_paths.size());
    Tensor* begin = reinterpret_cast<Tensor*>(out.data_ptr());
    Tensor* end = reinterpret_cast<Tensor*>(out.data_ptr_end());
    auto path_begin = all_paths.cbegin();
    for(;begin != end; ++begin, ++path_begin){
        *begin = functional::vector_to_tensor(*path_begin);
    }
    if(!laplacian.do_track_grad){
        return TensorGrad(out, false);
    }

    TensorGrad path(out, true);
    Tensor _lap = laplacian.tensor.clone();
    TensorGrad::redefine_tracking(path, laplacian,
    [_lap](const Tensor& wanted_paths, intrusive_ptr<TensorGrad>& parent){
        Tensor wanted_laplacian = functional::zeros(_lap.shape(), DType::Float32);
        float* begin = reinterpret_cast<float*>(wanted_laplacian.data_ptr());
        const Tensor* paths_begin = reinterpret_cast<const Tensor*>(wanted_paths.data_ptr());
        const Tensor* paths_end = paths_begin + wanted_paths.numel();
        for(;paths_begin != paths_end; ++paths_begin){
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
        for(int64_t i = 0; i < wanted_laplacian.shape()[0]; ++i){
            int64_t N = 0;
            for(int64_t c = 0; c < wanted_laplacian.shape()[1]; ++c){
                if(c == i) continue;
                N += begin[i * wanted_laplacian.shape()[0] + c] == 1 ? 1 : 0;
            }
            begin[i * wanted_laplacian.shape()[0] + i] = N;
        }
        
        Tensor present_lap = (std::abs(_lap) > 0.1).to(DType::Float32);
        int64_t false_positives = functional::count((wanted_laplacian == 0) && (present_lap == 1));
        int64_t false_negatives = functional::count((wanted_laplacian == 1) && (present_lap == 0));
        double total = static_cast<double>(std::max(false_positives + false_negatives, int64_t(1)));
        double alpha, beta;
        //so in the following, I am going to weight the false positives and false negatives based on what I am seeing
        alpha = sigmoid(static_cast<double>(false_negatives) / total);
        beta = sigmoid(static_cast<double>(false_positives) / total);


        Tensor weight = functional::ones(wanted_laplacian.shape(), wanted_laplacian.dtype);
        //  false negatives weight:
        // weight[wanted_laplacian == 1 && present_lap != 1] = 1.0;
        // false positives weight:
        weight[present_lap == 1 && wanted_laplacian != 1] = 0.001;
        weight.fill_diagonal_(0.0);
        // wanted_laplacian.fill_diagonal_(1.0);
        // wanted_laplacian.fill_diagonal_(0.0);
        //epsilon is 0.1 [LOOK AT FIND ALL PATHS FUNCTION]
        //in reality we only care if there is a 1 or not, so this works

        // std::cout << present_lap << std::endl;
        // Tensor sig = functional::sigmoid(_lap);
        // Tensor diff = wanted_laplacian - present_lap;
        // Tensor diff = functional::sigmoid(_lap) - wanted_laplacian;
        // diff[diff < 0] = 0;
        // Tensor dL = (diff) / wanted_laplacian.numel();
        Tensor dL = std::pow(present_lap - wanted_laplacian, 2) / wanted_laplacian.numel();
        dL.fill_diagonal_(0.0);

        // std::cout << "present laplacian: "<<present_lap<<std::endl;
        // std::cout << "wanted laplacian: "<<wanted_laplacian<<std::endl;
        // std::cout << present_lap - wanted_laplacian << std::endl;
        // std::cout << "dL: "<<dL<<std::endl;
        // Tensor dL = weight * std::pow(std::abs(_lap) - wanted_laplacian, 2);
        // Tensor dL = std::pow(_lap - wanted_laplacian, 2);
        // dL.fill_diagonal_(0);
        // if(nt::functional::all(dL == 0)){std::cout << "dL was all zeros"<<std::endl;}
        std::cout << "laplacian loss: "<<dL.sum().toScalar().to<float>()<< std::endl;
        // dL *= weight;
        // std::cout << "not equal to zero: "<<nt::functional::count(present_lap[wanted_laplacian != 0] != 0) << " of " << functional::count(wanted_laplacian != 0)<< std::endl;
        // std::cout << "equal to zero: "<<nt::functional::count(present_lap[wanted_laplacian == 0] == 0) << std::endl;
        // std::cout << "total equal to zero: "<<nt::functional::count(_lap == 0) << std::endl;
        // std::cout << "total negative: "<<nt::functional::count(diff < 0) << std::endl;
        // std::cout << "total positive: "<<nt::functional::count(diff > 0) << std::endl;
        std::cout << "false positives: "<<false_positives << std::endl;
        std::cout << "false negatives: " << false_negatives << std::endl;
        parent->grad->tensor = -dL;
    });
    return std::move(path);
}


}
} // namespace nt
