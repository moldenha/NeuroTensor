#ifndef NT_FMRI_STL_MESH_UTILS_HPP_
#define NT_FMRI_STL_MESH_UTILS_HPP_

#include <vector>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <limits>
#include <cmath>
#include <fstream>
#include <iostream>
#include "../../utils/type_traits.h"
#include "../../utils/always_inline_macro.h"
#include "../../math/math.h"
#include "../../math/kmath.h"

namespace nt::fmri::mesh::utils{
// Vec3T is for a triangular mesh
template<typename T>
struct Vec3T {
    static_assert(type_traits::is_arithmetic_v<T>);
    using value_type = T;
    T x, y, z;
    constexpr Vec3T(): x(0), y(0), z(0) {}
    constexpr Vec3T(float X, float Y, float Z): x(X), y(Y), z(Z) {}
    constexpr Vec3T(const Vec3T&) = default;
    constexpr Vec3T(Vec3T&&) = default;
    constexpr Vec3T operator+(const Vec3T& o) const { return {x+o.x, y+o.y, z+o.z}; }
    constexpr Vec3T operator-(const Vec3T& o) const { return {x-o.x, y-o.y, z-o.z}; }
    constexpr Vec3T operator*(float s) const { return {x*s, y*s, z*s}; }
    constexpr Vec3T operator/(float s) const { return {x/s, y/s, z/s}; }
};


using Vec3 = typename Vec3T<float>;
using Vec3i = typename Vec3T<int32_t>;

template<typename T, std::enable_if_t<type_traits::is_floating_point_v<T>, bool> = true>
NT_ALWAYS_INLINE constexpr T dot(const Vec3T<T>& a, const Vec3T<T>& b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
template<typename T, std::enable_if_t<type_traits::is_floating_point_v<T>, bool> = true>
NT_ALWAYS_INLINE constexpr Vec3T<T> cross(const Vec3T<T>& a,const Vec3T<T>& b){ return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x}; }
template<typename T, std::enable_if_t<type_traits::is_floating_point_v<T>, bool> = true>
NT_ALWAYS_INLINE T norm(const Vec3<T>& v){ return ::nt::math::sqrt(dot(v,v)); }


/// Small symmetric 4x4 "quadric" matrix used by QEM (only store upper tri)
struct Mat4Sym{
    // store as array index mapping: 0:00,1:01,2:02,3:03,4:11,5:12,6:13,7:22,8:23,9:33
    double d[10]; 
    // used instead of std::array for constexpr option
    // will consider making an nt::array that has constexpr properties in the future
    constexpr Mat4Sym() {
        for(int i = 0; i < 10; ++i)
            d[i] = 0.0;
    }

    inline constexpr void add_plane(double a,double b,double c,double d0,double w=1.0){
        // plane ax+by+cz+d0=0 with weight w
        // outer product [a b c d0]^T [a b c d0]
        double p[4] = {a* w, b* w, c* w, d0* w};
        // accumulate symmetric entries
        d[0] += p[0]*p[0]; // 00
        d[1] += p[0]*p[1]; // 01
        d[2] += p[0]*p[2]; // 02
        d[3] += p[0]*p[3]; // 03
        d[4] += p[1]*p[1]; // 11
        d[5] += p[1]*p[2]; // 12
        d[6] += p[1]*p[3]; // 13
        d[7] += p[2]*p[2]; // 22
        d[8] += p[2]*p[3]; // 23
        d[9] += p[3]*p[3]; // 33
    }

    inline constexpr Mat4Sym& operator+=(const Mat4Sym& o){
        for(int i=0;i<10;++i) d[i]+=o.d[i];
        return *this;
    }

};


/// Solve for optimal vertex from Q = Q1+Q2 by minimizing v^T * Q * v with v = [x y z 1]
/// For QEM you try to solve 3x3 linear system: [ Q00 Q01 Q02 ] [x] = -[Q03 Q13 Q23]^T
/// where Q elements pulled from Mat4Sym. Returns (success, Vec3). If fail -> midpoint fallback recommended.
inline constexpr bool solve_optimal_vertex(const Mat4Sym& Q, Vec3& out){
    // build symmetric 3x3
    double m00 = Q.d[0], m01 = Q.d[1], m02 = Q.d[2];
    double m11 = Q.d[4], m12 = Q.d[5];
    double m22 = Q.d[7];

    double b0 = -Q.d[3], b1 = -Q.d[6], b2 = -Q.d[8]; // negatives of 03,13,23

    // Determinant
    double det = m00*(m11*m22 - m12*m12) - m01*(m01*m22 - m12*m02) + m02*(m01*m12 - m11*m02);
    if (::nt::math::kmath::abs(det) < 1e-12) return false;

    // Cramer's rule
    double inv_det = 1.0/det;

    double dx =  (b0*(m11*m22 - m12*m12) - m01*(b1*m22 - m12*b2) + m02*(b1*m12 - m11*b2)) * inv_det;
    double dy = -(m00*(b1*m22 - m12*b2) - b0*(m01*m22 - m12*m02) + m02*(m01*b2 - b1*m02)) * inv_det;
    double dz =  (m00*(m11*b2 - b1*m12) - m01*(m01*b2 - b1*m02) + b0*(m01*m12 - m11*m02)) * inv_det;

    out.x = float(dx);
    out.y = float(dy);
    out.z = float(dz);
    return true;
}

/// Edge key (unordered pair) helper
struct EdgeKey {
    int a,b;
    constexpr EdgeKey(int i,int j){ if(i<j){a=i;b=j;} else {a=j;b=i;} }
    constexpr bool operator==(EdgeKey const& o) const { return a==o.a && b==o.b; }
};



} // nt::fmri::mesh::utils

namespace std{
    template<> struct hash<EdgeKey> {
        size_t operator()(EdgeKey const& e) const noexcept {
            return ( (size_t)e.a * 73856093u ) ^ ((size_t)e.b * 19349663u);
        }
    };
} // std::


namespace nt::fmri::mesh::utils {

/// Priority entry
struct EdgeEntry {
    int v1, v2;
    double error;
    Vec3 pos; // optimal or fallback
    constexpr EdgeEntry(int a=0,int b=0,double e=0.0, Vec3 p=Vec3()): v1(a),v2(b),error(e),pos(p){}
};

struct EdgeCmp {
    constexpr bool operator()(const EdgeEntry& A, const EdgeEntry& B) const {
        return A.error > B.error;
    }
};

/// Build vertex normals and per-face normals (returns per-vertex normal averaged)
inline std::vector<Vec3> compute_vertex_normals(const std::vector<Vec3>& V, const std::vector<Vec3i>& F){
    std::vector<Vec3> normals(V.size(), Vec3(0,0,0));
    for(auto &t : F){
        Vec3 v0 = V[t.b] - V[t.a];
        Vec3 v1 = V[t.c] - V[t.a];
        Vec3 fn = cross(v0, v1);
        // accumulate
        normals[t.a] = normals[t.a] + fn;
        normals[t.b] = normals[t.b] + fn;
        normals[t.c] = normals[t.c] + fn;
    }
    // normalize
    for(auto &n : normals){
        float L = norm(n);
        if (L>1e-12) { n = n / L; }
    }
    return normals;
}

/// Build per-vertex quadric Q from triangle planes
inline void build_vertex_quadrics(const std::vector<Vec3>& V, const std::vector<Vec3i>& F, std::vector<Mat4Sym>& outQ){
    size_t nv = V.size();
    outQ.assign(nv, Mat4Sym());
    auto vnorms = compute_vertex_normals(V, F);
    // For each face compute plane (a,b,c,d) from normalized normal and one point
    for(auto &t : F){
        Vec3 p0 = V[t.a], p1 = V[t.b], p2 = V[t.c];
        Vec3 fn = cross(p1 - p0, p2 - p0);
        float fnL = norm(fn);
        if (fnL < 1e-12f) continue;
        Vec3 n = fn / fnL; // face normal (not averaged)
        // plane eq: n.x * x + n.y * y + n.z * z + d = 0
        double d0 = - (n.x * p0.x + n.y * p0.y + n.z * p0.z);
        // Add same plane to each vertex of triangle
        outQ[t.a].add_plane(n.x, n.y, n.z, d0);
        outQ[t.b].add_plane(n.x, n.y, n.z, d0);
        outQ[t.c].add_plane(n.x, n.y, n.z, d0);
    }
}

NT_ALWAYS_INLINE EdgeEntry compute_edge_error(const std::vector<Mat4Sym>& Qs, const std::vector<Vec3>& V, int v1, int v2){
    Mat4Sym Q = Qs[v1];
    Q += Qs[v2];
    Vec3 opt;
    bool ok = solve_optimal_vertex(Q, opt);
    if (!ok) { // fallback to midpoint
        opt = (V[v1] + V[v2]) * 0.5f;
    }
    // compute error = v^T * Q * v (with homogeneous coordinate 1)
    double x = opt.x, y = opt.y, z = opt.z;
    // evaluate [x y z 1] * Q * [x y z 1]^T using symmetric storage
    double vtv = x*(x*Q.d[0] + y*Q.d[1] + z*Q.d[2] + Q.d[3])
               + y*(x*Q.d[1] + y*Q.d[4] + z*Q.d[5] + Q.d[6])
               + z*(x*Q.d[2] + y*Q.d[5] + z*Q.d[7] + Q.d[8])
               + (x*Q.d[3] + y*Q.d[6] + z*Q.d[8] + Q.d[9]);
    return EdgeEntry(v1, v2, vtv, opt);
}

void simplify_mesh_qem(std::vector<Vec3>& V, std::vector<Vec3i>& F, int target_face_count = 20000);
/// Taubin smoothing: iterative non-shrinking Laplacian smoothing
void taubin_smooth(std::vector<Vec3>& V, const std::vector<Vec3i>& F, int iterations=10, float lambda=0.5f, float mu=-0.53f);

} // nt::fmri::mesh::utils


#endif

