#include <nt/nt.h>
#include <nt/math/math.h>
#include <nt/bit/bit.h>

inline nt::float128_t custom_abs(const nt::float128_t& a) noexcept {
    using U = nt::uint128_t;
    U bits = nt::bit_cast<U>(a);
    nt::bitset<128, nt::uint128_t> set(bits);
    std::cout << set << std::endl;
    bits &= ~(U(1) << 127);
    return nt::bit_cast<nt::float128_t>(bits);
}

void float_128_test(){
    nt::Tensor check_close = nt::rand(-3, 3, {5}, nt::DType::Float128);
    std::cout << check_close << std::endl;
    std::cout << nt::abs(check_close) << std::endl;
    nt::float128_t a = -1.2342;
    std::cout << nt::math::abs(a) << std::endl;
    std::cout << custom_abs(a) << std::endl;
}
