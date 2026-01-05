#include <iostream>
#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <cmath>

namespace nt{

namespace details{
template<class Type>
constexpr std::size_t bitset_bits_per_word = sizeof(Type) * CHAR_BIT;

template<std::size_t N, class Type>
constexpr std::size_t bitset_words = (N + bitset_bits_per_word<Type> - 1) / bitset_bits_per_word<Type>;


}

namespace type_traits{
template<typename T>
using remove_cvref_t = typename std::remove_reference_t<std::remove_cv_t<T>>;
}

template<std::size_t N, class Type = uint64_t>
class bitset{
    static_assert(std::is_integral_v<Type>, "Error bitset must use an integer type");
    static constexpr std::size_t BITS_PER_WORD = details::bitset_bits_per_word<Type>;
    static constexpr std::size_t WORDS = details::bitset_words<N, Type>;
    Type data[WORDS]{};
    
public:
    
    constexpr bitset() = default;
    constexpr bitset(Type low_word)
    {
        if constexpr (WORDS > 0){
            data[0] = low_word;
        }
    }
    
    constexpr bool test(std::size_t pos) const {
        std::size_t word = pos / BITS_PER_WORD;
        std::size_t bit  = pos % BITS_PER_WORD;
        return (data[word] >> bit) & 1ULL;
    }
    
    template<bool value = true>
    constexpr void set(std::size_t pos) {
        std::size_t word = pos / BITS_PER_WORD;
        std::size_t bit  = pos % BITS_PER_WORD;
        Type mask = Type(1) << bit;
        if constexpr (value)
            data[word] |= mask;
        else
            data[word] &= ~mask;
    }

    constexpr void reset(std::size_t pos) {
        set<false>(pos);
    }

    constexpr bitset& operator|=(const bitset& other) {
        for (std::size_t i = 0; i < WORDS; ++i)
            data[i] |= other.data[i];
        return *this;
    }

    constexpr bitset& operator&=(const bitset& other) {
        for (std::size_t i = 0; i < WORDS; ++i)
            data[i] &= other.data[i];
        return *this;
    }

    constexpr Type lo_type() const {
        static_assert(WORDS >= 1, "bitset too small");
        return data[0];
    }
    
    constexpr Type hi_type() const {
        static_assert(WORDS >= 2, "bitset must be >= 128 bits");
        return data[1];
    }

    constexpr const Type* raw() const { return data; }

};



// designed to take a constexpr bitset<32, uint32_t> and turn it into a any_bitset<float> for example
// template<class T, std::size_t N, class Type>
// constexpr T bitset_to_any(const bitset<N, Type>& bits)
// {
//     static_assert(std::is_trivially_copyable_v<T>,
//                   "T must be trivially copyable");
//     static_assert(std::is_trivially_copyable_v<Type>,
//                   "bitset storage type must be trivially copyable");

//     // Total bytes in the bitset storage
//     constexpr std::size_t bytes = sizeof(bits.raw()[0]) * details::bitset_words<N,Type>;
//     constexpr std::size_t total_bytes = sizeof(T) * details::bitset_words<N,Type>;

//     static_assert(sizeof(T) == bytes,
//                   "Size mismatch: bitset bytes != output type bytes");

//     union u__{Type src[details::bitset_words<N, Type>]; T dst;};
//     constexpr u__ u = {bits.raw()};
//     return u.dst;
    
//     // T* ptr_out = &out;
//     // unsigned char* out_bytes = (unsigned char*)ptr_out;
//     // // unsigned char* out_bytes  = reinterpret_cast<unsigned char*>(&out);
//     // const unsigned char* data_bytes = reinterpret_cast<const unsigned char*>(bits.raw());

//     // for (std::size_t i = 0; i < total_bytes; ++i)
//     //     out_bytes[i] = data_bytes[i];
//     // return out;

// }

template<class To, class From_>
inline std::enable_if_t<
    sizeof(To) == sizeof(From_) &&
    std::is_trivially_copyable_v<From_> &&
    std::is_trivially_copyable_v<To>,
    To>
// constexpr support needs compiler magic
bit_cast(const From_& src) noexcept
{
    using From = type_traits::remove_cvref_t<std::decay_t<From_>>;
    // the union is trivially constructible
    // Therefore From_ and To don't need to be
    union u__{u__(){}; char bits[sizeof(From)]; type_traits::remove_cvref_t<To> dst;} u;
    std::memcpy(&u.dst, &src, sizeof(From));
    return u.dst;
}


}

constexpr nt::bitset<16, uint16_t> make_float16_qnan()
{
    nt::bitset<16, uint16_t> b{};

    // exponent = 11111 (bits 10..14)
    for (int i = 10; i <= 14; ++i)
        b.set(i);

    // quiet bit (mantissa MSB, bit 9)
    b.set(9);

    return b;
}

constexpr nt::bitset<16, uint16_t> FLOAT16_QNAN_BITS = make_float16_qnan();

struct float16_t{
    uint16_t var{};
    constexpr float16_t(uint16_t val_) : var(val_) {}
};


// constexpr float16_t* fp16_qnan_ptr = FLOAT16_QNAN_BITS.n_raw<float16_t>(); 

// const float16_t FLOAT16_QNAN = nt::bit_cast<float16_t>(FLOAT16_QNAN_BITS); 
// constexpr float16_t FLOAT16_QNAN = float16_t(make_float16_qnan().lo_type());


using uint128_t = __uint128_t; 
constexpr nt::bitset<128, uint128_t> make_float128_qnan()
{
    nt::bitset<128, uint128_t> b{};

    // exponent = 111111111111111 (bits 112..126)
    for (int i = 112; i <= 126; ++i)
        b.set(i);

    // quiet NaN mantissa MSB (bit 111)
    b.set(111);

    return b;
}

constexpr nt::bitset<128, uint128_t> FLOAT128_QNAN_BITS = make_float128_qnan();



struct float128_t{
    uint128_t a;
};



std::ostream& operator<<(std::ostream& os, const uint128_t& i){
  std::ostream::sentry s(os);
  if (s) {
    uint128_t tmp = i;
    char buffer[128];
    char *d = std::end(buffer);
    do {
      --d;
      *d = "0123456789"[tmp % 10];
      tmp /= 10;
    } while (tmp != 0);
    int len = std::end(buffer) - d;
    if (os.rdbuf()->sputn(d, len) != len) {
      os.setstate(std::ios_base::badbit);
    }
  }
  return os;

}

std::ostream& operator<<(std::ostream& os, const float16_t& f){
    return os << f.var;
}

std::ostream& operator<<(std::ostream& os, const float128_t& f){
    return os << f.a;
}

template<typename T>
T nan() noexcept;

template<>
inline float16_t nan() noexcept {
    return nt::bit_cast<float16_t>(FLOAT16_QNAN_BITS);
}

template<>
inline float nan() noexcept {
    return std::numeric_limits<float>::quiet_NaN();
}

template<>
inline double nan() noexcept {
    return std::numeric_limits<double>::quiet_NaN();
}

template<>
inline float128_t nan() noexcept {
    return nt::bit_cast<float128_t>(FLOAT128_QNAN_BITS);
}

namespace nt{

inline bool isnan(const float& val){return std::isnan(val);}
inline bool isnan(const double& val){return std::isnan(val);}
inline bool isnan(const float16_t& val){
    uint16_t bits = ::nt::bit_cast<uint16_t>(val);

    constexpr uint16_t exponent_mask = 0x7C00;  // 11111 0000000000
    constexpr uint16_t mantissa_mask = 0x03FF;  // 00000 1111111111

    // NaN: exponent all 1s AND mantissa != 0
    return (bits & exponent_mask) == exponent_mask &&
           (bits & mantissa_mask) != 0;
}
inline bool isnan(const float128_t& val){
    uint128_t bits = ::nt::bit_cast<uint128_t>(val);
    return (bits & (uint128_t(0x7FFF) << 112)) == (uint128_t(0x7FFF) << 112) &&
       (bits & ((uint128_t(1) << 112) - 1)) != 0;
}

}


constexpr nt::bitset<16, uint16_t> make_float16_pos_inf()
{
    nt::bitset<16, uint16_t> b{};

    // exponent = 11111 (bits 10..14)
    for (int i = 10; i <= 14; ++i)
        b.set(i);

    // mantissa = 0
    // sign = 0 (bit 15 left at default 0)

    return b;
}

constexpr nt::bitset<16, uint16_t> FLOAT16_POS_INF_BITS = make_float16_pos_inf();

// -infinity (float16)
constexpr nt::bitset<16, uint16_t> make_float16_neg_inf()
{
    nt::bitset<16, uint16_t> b{};

    // exponent = 11111
    for (int i = 10; i <= 14; ++i)
        b.set(i);

    // sign bit = 1
    b.set(15);

    return b;
}

constexpr nt::bitset<16, uint16_t> FLOAT16_NEG_INF_BITS = make_float16_neg_inf();


// +infinity (float128)
constexpr nt::bitset<128, uint128_t> make_float128_pos_inf()
{
    nt::bitset<128, uint128_t> b{};

    // exponent bits 112..126 = all 1s
    for (int i = 112; i <= 126; ++i)
        b.set(i);

    // mantissa = 0
    // sign = 0 (bit 127 default)

    return b;
}

constexpr nt::bitset<128, uint128_t> FLOAT128_POS_INF_BITS = make_float128_pos_inf();


// -infinity (float128)
constexpr nt::bitset<128, uint128_t> make_float128_neg_inf()
{
    nt::bitset<128, uint128_t> b{};

    // exponent bits 112..126 = all 1s
    for (int i = 112; i <= 126; ++i)
        b.set(i);

    // sign = 1
    b.set(127);

    return b;
}

constexpr nt::bitset<128, uint128_t> FLOAT128_NEG_INF_BITS = make_float128_neg_inf();


// really just returns the MIN
template<typename T>
constexpr T make_inf_like_integer(){
    static_assert(std::is_integral_v<T>, "Can only make inf like integer for function");
    nt::bitset<sizeof(T) * CHAR_BIT, T> b{};
    for(int i = 0; i < sizeof(T) * CHAR_BIT; ++i)
        b.set(i);
    if constexpr (std::is_signed_v<T>){
        b.reset(sizeof(T) * CHAR_BIT - 1);
    }
    return b.lo_type(); 
}

template<typename T>
constexpr T make_neg_inf_like_integer(){
    static_assert(std::is_integral_v<T>, "Can only make inf like integer for function");
    if constexpr (std::is_unsigned_v<T>){
        return 0;
    }
    nt::bitset<sizeof(T) * CHAR_BIT, T> b{};
    b.set(sizeof(T) * CHAR_BIT - 1);
    return b.lo_type();
}

template<typename T>
T inf() noexcept;

template<>
inline float16_t inf() noexcept {
    return nt::bit_cast<float16_t>(FLOAT16_POS_INF_BITS);
}

template<>
inline float inf() noexcept {
    return std::numeric_limits<float>::infinity();
}

template<>
inline double inf() noexcept {
    return std::numeric_limits<double>::infinity();
}

template<>
inline float128_t inf() noexcept {
    return nt::bit_cast<float128_t>(FLOAT128_POS_INF_BITS);
}


template<typename T>
T neg_inf() noexcept;

template<>
inline float16_t neg_inf() noexcept {
    return nt::bit_cast<float16_t>(FLOAT16_NEG_INF_BITS);
}

template<>
inline float neg_inf() noexcept {
    return - std::numeric_limits<float>::infinity();
}

template<>
inline double neg_inf() noexcept {
    return - std::numeric_limits<double>::infinity();
}

template<>
inline float128_t neg_inf() noexcept {
    return nt::bit_cast<float128_t>(FLOAT128_NEG_INF_BITS);
}

namespace nt{
inline bool isinf(const float& val) { return std::isinf(val); }
inline bool isinf(const double& val) {return std::isinf(val);}
inline bool isinf(const float128_t& val){
    uint128_t bits = ::nt::bit_cast<uint128_t>(val);
    return (bits & (uint128_t(0x7FFF) << 112)) == (uint128_t(0x7FFF) << 112) &&
       (bits & ((uint128_t(1) << 112) - 1)) == 0;
}
inline bool isinf(const float16_t& val){
    uint16_t bits = ::nt::bit_cast<uint16_t>(val);
    constexpr uint16_t exponent_mask = 0x7C00; // 11111 0000000000
    constexpr uint16_t mantissa_mask = 0x03FF; // 00000 1111111111
    return (bits & exponent_mask) == exponent_mask &&
           (bits & mantissa_mask) == 0;
}

}

int main(){
    std::cout << nan<float16_t>() << ' '
              << nan<float>() << ' '
              << nan<double>() << ' '
              << nan<float128_t>() << std::endl;

    std::cout << inf<float16_t>() << ' '
              << inf<float>() << ' '
              << inf<double>() << ' '
              << inf<float128_t>() << std::endl;

    std::cout << neg_inf<float16_t>() << ' '
              << neg_inf<float>() << ' '
              << neg_inf<double>() << ' '
              << neg_inf<float128_t>() << std::endl;

    std::cout << std::boolalpha << nt::isnan(nan<float16_t>()) << ' '
              << nt::isnan(nan<float>()) << ' '
              << nt::isnan(nan<double>()) << ' '
              << nt::isnan(nan<float128_t>()) << std::noboolalpha << std::endl;

    std::cout << std::boolalpha << nt::isinf(inf<float16_t>()) << ' '
              << nt::isinf(inf<float>()) << ' '
              << nt::isinf(inf<double>()) << ' '
              << nt::isinf(inf<float128_t>()) << std::noboolalpha << std::endl;
    
    std::cout << std::boolalpha << nt::isinf(neg_inf<float16_t>()) << ' '
              << nt::isinf(neg_inf<float>()) << ' '
              << nt::isinf(neg_inf<double>()) << ' '
              << nt::isinf(neg_inf<float128_t>()) << std::noboolalpha << std::endl;
    
    std::cout << make_inf_like_integer<uint64_t>() << ' ' << make_neg_inf_like_integer<uint64_t>() << std::endl;
    std::cout << make_inf_like_integer<int64_t>() << ' ' << make_neg_inf_like_integer<int64_t>() << std::endl;
    std::cout << sizeof(float128_t) << std::endl;
    std::cout << sizeof(FLOAT128_QNAN_BITS) << std::endl;
    return 0;
}
