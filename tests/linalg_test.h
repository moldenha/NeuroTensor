#include <nt/Tensor.h>
#include <nt/linalg/linalg.h>


void svd_test(){
    std::cout << "SVD Test: " << std::endl;
    nt::Tensor A = nt::functional::randn({20, 11}, nt::DType::Float64);
    auto [S, V, D] = nt::get<3>(nt::linalg::SVD(A));
    std::cout << S << ',' << V << ',' << D << std::endl;
}

void qr_test(){
    std::cout << "QR Test: " << std::endl;
    nt::Tensor A = nt::functional::randn({20, 11}, nt::DType::Float64);
    auto [Q, R] = nt::get<2>(nt::linalg::QR(A));
    std::cout << Q << ',' << R << std::endl;
}


void inv_test(){
    std::cout << "Inv test: "<<std::endl;
    nt::Tensor A = nt::functional::randn({20, 20}, nt::DType::Float64);
    nt::Tensor inv = nt::linalg::inv(A);
    std::cout << inv << std::endl;
}

void pinv_test(){
    std::cout << "pseudo Inv test: "<<std::endl;
    nt::Tensor A = nt::functional::randn({20, 11}, nt::DType::Float64);
    nt::Tensor inv = nt::linalg::pinv(A);
    std::cout << inv << std::endl;
}

void eye_test(){
    std::cout << "identity matrix of 30, with dtype int8:" << std::endl;
    nt::Tensor i = nt::linalg::eye(30, 0, nt::DType::int8);
    std::cout << i << std::endl;
}


