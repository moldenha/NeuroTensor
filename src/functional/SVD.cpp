//most of the algorithm was obtained from:
//https://stackoverflow.com/questions/3856072/single-value-decomposition-implementation-c

#include <vector>
#include <cassert>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "../utils/utils.h"
#include "functional.h"
#include "linalg.h"
#include "../Tensor.h"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/blocked_range3d.h>
#include <tbb/global_control.h>
#include <tbb/task_group.h>

#define U(i,j) U_[(i)*dim[0]+(j)]
#define S(i,j) S_[(i)*dim[1]+(j)]
#define V(i,j) V_[(i)*dim[1]+(j)]

template <typename T>
void GivensL(T* S_, const int64_t dim[2], int64_t m, T a, T b){
    T r=std::sqrt(a*a+b*b);
    T c=a/r;
    T s=-b/r;
    
    tbb::parallel_for(tbb::blocked_range<int64_t>(0, dim[1]),
    [&](const tbb::blocked_range<int64_t>& r){
        for(int64_t i = r.begin(); i != r.end(); ++i){
            T S0=S(m+0,i);
            T S1=S(m+1,i);
            S(m  ,i)+=S0*(c-1);
            S(m  ,i)+=S1*(-s );

            S(m+1,i)+=S0*( s );
            S(m+1,i)+=S1*(c-1);
 
        }
    });
}

template <typename T>
void GivensR(T* S_, const int64_t dim[2], int64_t m, T a, T b){
    T r=std::sqrt(a*a+b*b);
    T c=a/r;
    T s=-b/r;

    tbb::parallel_for(tbb::blocked_range<int64_t>(0, dim[0]),
    [&](const tbb::blocked_range<int64_t>& r){
    for(int64_t i=r.begin();i != r.end(); ++i){
        T S0=S(i,m+0);
        T S1=S(i,m+1);
        S(i,m  )+=S0*(c-1);
        S(i,m  )+=S1*(-s );

        S(i,m+1)+=S0*( s );
        S(i,m+1)+=S1*(c-1);
    }
    });
}

template <typename T>
void SVD(const int64_t dim[2], T* U_, T* S_, T* V_, T eps=-1){
    // nt::utils::throw_exception(dim[0] >= dim[1], "When performing SVD, rows ($) are expected to be greater than or equal to the number of collumns ($)", dim[0], dim[1]);
  assert(dim[0]>=dim[1]);

    { // Bi-diagonalization
        int64_t n=std::min(dim[0],dim[1]);
        std::vector<T> house_vec(std::max(dim[0],dim[1]));
        for(int64_t i=0;i<n;i++){
            // Column Householder
            {
                T x1=S(i,i);
                if(x1<0) x1=-x1;

                T x_inv_norm=0;
                for(int64_t j=i;j<dim[0];j++){
                    x_inv_norm+=S(j,i)*S(j,i);
                }
                if(x_inv_norm>0) x_inv_norm=1/std::sqrt(x_inv_norm);

                T alpha=std::sqrt(1+x1*x_inv_norm);
                T beta=x_inv_norm/alpha;

                house_vec[i]=-alpha;
                for(int64_t j=i+1;j<dim[0];j++){
                    house_vec[j]=-beta*S(j,i);
                }
                if(S(i,i)<0) for(int64_t j=i+1;j<dim[0];j++){
                    house_vec[j]=-house_vec[j];
                }
            }
            tbb::parallel_for(tbb::blocked_range<int64_t>(i, dim[1]),
            [&](const tbb::blocked_range<int64_t>& r){
            for(int64_t k = r.begin(); k != r.end(); ++k){
                T dot_prod = 0;
                for(int64_t j = i; j < dim[0]; ++j){
                    dot_prod += S(j,k)*house_vec[j];
                }
                for(int64_t j = i; j < dim[0]; ++j){
                    S(j,k) -= dot_prod*house_vec[j];
                }
            }});
            tbb::parallel_for(tbb::blocked_range<int64_t>(0, dim[0]),
            [&](const tbb::blocked_range<int64_t>& r){
            for(int64_t k = r.begin(); k != r.end(); ++k){
                T dot_prod = 0;
                for(int64_t j = i; j < dim[0]; ++j){
                    dot_prod+=U(k,j)*house_vec[j];
                }
                for(int64_t j = i; j < dim[0]; ++j){
                    U(k,j) -= dot_prod*house_vec[j];
                }
            }});

            // Row Householder
            if(i>=n-1) continue;
            {
                T x1=S(i,i+1);
                if(x1<0) x1=-x1;
                T x_inv_norm=0;
                for(int64_t j=i+1;j<dim[1];++j){
                  x_inv_norm+=S(i,j)*S(i,j);
                }
                if(x_inv_norm>0) x_inv_norm=1/std::sqrt(x_inv_norm);

                T alpha=std::sqrt(1+x1*x_inv_norm);
                T beta=x_inv_norm/alpha;

                house_vec[i+1]=-alpha;
                for(int64_t j=i+2;j<dim[1];j++){
                  house_vec[j]=-beta*S(i,j);
                }
                if(S(i,i+1)<0) for(int64_t j=i+2;j<dim[1];j++){
                  house_vec[j]=-house_vec[j];
                }
            }
            tbb::parallel_for(tbb::blocked_range<int64_t>(i, dim[0]),
            [&](const tbb::blocked_range<int64_t>& r){
            for(int64_t k = r.begin(); k != r.end(); ++k){
                T dot_prod=0;
                for(int64_t j=i+1;j<dim[1];j++){
                  dot_prod+=S(k,j)*house_vec[j];
                }
                for(int64_t j=i+1;j<dim[1];j++){
                  S(k,j)-=dot_prod*house_vec[j];
                }
            }});
            tbb::parallel_for(tbb::blocked_range<int64_t>(0, dim[1]),
            [&](const tbb::blocked_range<int64_t>& r){
            for(int64_t k = r.begin(); k != r.end(); ++k){
                T dot_prod=0;
                for(int64_t j=i+1;j<dim[1];j++){
                  dot_prod+=V(j,k)*house_vec[j];
                }
                for(int64_t j=i+1;j<dim[1];j++){
                  V(j,k)-=dot_prod*house_vec[j];
                }
            }});
        }
        int64_t k0=0;
        if(eps<0){
            eps=1.0;
            while(eps+(T)1.0>1.0) eps*=0.5;
            eps*=64.0;
        }
        while(k0<dim[1]-1){ // Diagonalization
            T S_max=0.0;
            for(int64_t i=0;i<dim[1];i++) S_max=(S_max>S(i,i)?S_max:S(i,i));

            while(k0<dim[1]-1 && std::abs(S(k0,k0+1))<=eps*S_max) k0++;
            if(k0==dim[1]-1) continue;

            int64_t n=k0+2;
            while(n<dim[1] && std::abs(S(n-1,n))>eps*S_max) n++;

            T alpha=0;
            T beta=0;
            { // Compute mu
              T C[2][2];
              C[0][0]=S(n-2,n-2)*S(n-2,n-2);
              if(n-k0>2) C[0][0]+=S(n-3,n-2)*S(n-3,n-2);
              C[0][1]=S(n-2,n-2)*S(n-2,n-1);
              C[1][0]=S(n-2,n-2)*S(n-2,n-1);
              C[1][1]=S(n-1,n-1)*S(n-1,n-1)+S(n-2,n-1)*S(n-2,n-1);

              T b=-(C[0][0]+C[1][1])/2;
              T c=  C[0][0]*C[1][1] - C[0][1]*C[1][0];
              T d=0;
              if(b*b-c>0) d=std::sqrt(b*b-c);
              else{
                T b=(C[0][0]-C[1][1])/2;
                T c=-C[0][1]*C[1][0];
                if(b*b-c>0) d=std::sqrt(b*b-c);
              }

              T lambda1=-b+d;
              T lambda2=-b-d;

              T d1=lambda1-C[1][1]; d1=(d1<0?-d1:d1);
              T d2=lambda2-C[1][1]; d2=(d2<0?-d2:d2);
              T mu=(d1<d2?lambda1:lambda2);

              alpha=S(k0,k0)*S(k0,k0)-mu;
              beta=S(k0,k0)*S(k0,k0+1);
            }

            for(int64_t k=k0;k<n-1;k++)
            {
              int64_t dimU[2]={dim[0],dim[0]};
              int64_t dimV[2]={dim[1],dim[1]};
              GivensR(S_,dim ,k,alpha,beta);
              GivensL(V_,dimV,k,alpha,beta);

              alpha=S(k,k);
              beta=S(k+1,k);
              GivensL(S_,dim ,k,alpha,beta);
              GivensR(U_,dimU,k,alpha,beta);

              alpha=S(k,k+1);
              beta=S(k,k+2);
            }

            { // Make S bi-diagonal again
              for(int64_t i0=k0;i0<n-1;i0++){
                for(int64_t i1=0;i1<dim[1];i1++){
                  if(i0>i1 || i0+1<i1) S(i0,i1)=0;
                }
              }
              for(int64_t i0=0;i0<dim[0];i0++){
                for(int64_t i1=k0;i1<n-1;i1++){
                  if(i0>i1 || i0+1<i1) S(i0,i1)=0;
                }
              }
              for(int64_t i=0;i<dim[1]-1;i++){
                if(std::abs(S(i,i+1))<=eps*S_max){
                  S(i,i+1)=0;
                }
              }
            }
        }
    }
}

#undef U
#undef S
#undef V

template<typename T>
inline void svd(char *JOBU, char *JOBVT, int64_t *M, int64_t *N, T *A, int64_t *LDA,
    T *S, T *U, int64_t *LDU, T *VT, int64_t *LDVT, T *WORK, int64_t *LWORK,
    int64_t *INFO){
  assert(*JOBU=='S');
  assert(*JOBVT=='S');
  const int64_t dim[2]={std::max(*N,*M), std::min(*N,*M)};
  T* U_=new T[dim[0]*dim[0]]; memset(U_, 0, dim[0]*dim[0]*sizeof(T));
  T* V_=new T[dim[1]*dim[1]]; memset(V_, 0, dim[1]*dim[1]*sizeof(T));
  T* S_=new T[dim[0]*dim[1]];

  const int64_t lda=*LDA;
  const int64_t ldu=*LDU;
  const int64_t ldv=*LDVT;

  if(dim[1]==*M){
    for(int64_t i=0;i<dim[0];i++)
    for(int64_t j=0;j<dim[1];j++){
      S_[i*dim[1]+j]=A[i*lda+j];
    }
  }else{
    for(int64_t i=0;i<dim[0];i++)
    for(int64_t j=0;j<dim[1];j++){
      S_[i*dim[1]+j]=A[j*lda+i];
    }
  }
  for(int64_t i=0;i<dim[0];i++){
    U_[i*dim[0]+i]=1;
  }
  for(int64_t i=0;i<dim[1];i++){
    V_[i*dim[1]+i]=1;
  }

  SVD<T>(dim, U_, S_, V_, (T)-1);

  for(int64_t i=0;i<dim[1];i++){ // Set S
    S[i]=S_[i*dim[1]+i];
  }
  if(dim[1]==*M){ // Set U
    for(int64_t i=0;i<dim[1];i++)
    for(int64_t j=0;j<*M;j++){
      U[j+ldu*i]=V_[j+i*dim[1]]*(S[i]<0.0?-1.0:1.0);
    }
  }else{
    for(int64_t i=0;i<dim[1];i++)
    for(int64_t j=0;j<*M;j++){
      U[j+ldu*i]=U_[i+j*dim[0]]*(S[i]<0.0?-1.0:1.0);
    }
  }
  if(dim[0]==*N){ // Set V
    for(int64_t i=0;i<*N;i++)
    for(int64_t j=0;j<dim[1];j++){
      VT[j+ldv*i]=U_[j+i*dim[0]];
    }
  }else{
    for(int64_t i=0;i<*N;i++)
    for(int64_t j=0;j<dim[1];j++){
      VT[j+ldv*i]=V_[i+j*dim[1]];
    }
  }
  for(int64_t i=0;i<dim[1];i++){
    S[i]=S[i]*(S[i]<0.0?-1.0:1.0);
  }

  delete[] U_;
  delete[] S_;
  delete[] V_;
  *INFO=0;
}

// int example(){
//   typedef double Real_t;
//   int n1=45, n2=27;

//   // Create matrix
//   Real_t* M =new Real_t[n1*n2];
//   for(size_t i=0;i<n1*n2;i++) M[i]=drand48();

//   int m = n2;
//   int n = n1;
//   int k = (m<n?m:n);
//   Real_t* tU =new Real_t[m*k];
//   Real_t* tS =new Real_t[k];
//   Real_t* tVT=new Real_t[k*n];

//   { // Compute SVD
//     int INFO=0;
//     char JOBU  = 'S';
//     char JOBVT = 'S';
//     int wssize = 3*(m<n?m:n)+(m>n?m:n);
//     int wssize1 = 5*(m<n?m:n);
//     wssize = (wssize>wssize1?wssize:wssize1);
//     Real_t* wsbuf = new Real_t[wssize];
//     svd(&JOBU, &JOBVT, &m, &n, &M[0], &m, &tS[0], &tU[0], &m, &tVT[0], &k, wsbuf, &wssize, &INFO);
//     delete[] wsbuf;
//   }

//   { // Check Error
//     Real_t max_err=0;
//     for(size_t i0=0;i0<m;i0++)
//     for(size_t i1=0;i1<n;i1++){
//       Real_t E=M[i1*m+i0];
//       for(size_t i2=0;i2<k;i2++) E-=tU[i2*m+i0]*tS[i2]*tVT[i1*k+i2];
//       if(max_err<std::abs(E)) max_err=std::abs(E);
//     }
//     std::cout<<max_err<<'\n';
//   }

//   delete[] tU;
//   delete[] tS;
//   delete[] tVT;
//   delete[] M;

//   return 0;
// }



namespace nt{
namespace linalg{

template<DType dt = DType::Integer>
Tensor handle_svd_matrix(Tensor& t){
    if(dt != t.dtype){return handle_svd_matrix<DTypeFuncs::next_dtype_it<dt> >(t);}
    if constexpr (DTypeFuncs::is_dtype_floating_v<dt>){
        using type_t = DTypeFuncs::dtype_to_type_t<dt>;
        int64_t n1 = t.shape()[0];
        int64_t n2 = t.shape()[1];
        int64_t m = n2;
        int64_t n = n1;
        int64_t k = (m<n?m:n);
        Tensor tU({m, k}, t.dtype);
        Tensor tS({k}, t.dtype);
        Tensor tVT({k, n}, t.dtype);
        int64_t INFO=0;
        char JOBU  = 'S';
        char JOBVT = 'S';
        int64_t wssize = 3*(m<n?m:n)+(m>n?m:n);
        int64_t wssize1 = 5*(m<n?m:n);
        wssize = (wssize>wssize1?wssize:wssize1);
        type_t* wsbuf = new type_t[wssize];
        svd<type_t>(&JOBU, &JOBVT, &m, &n, reinterpret_cast<type_t*>(t.data_ptr()), &m, reinterpret_cast<type_t*>(tS.data_ptr()),
            reinterpret_cast<type_t*>(tU.data_ptr()), &m, reinterpret_cast<type_t*>(tVT.data_ptr()),
            &k, wsbuf, &wssize, &INFO);
        delete[] wsbuf;
        return functional::list(tU, tS, tVT);

    }
	else{
		utils::THROW_EXCEPTION(false, "Expected dtype to be flaoting but got dtype $", t.dtype);
		return Tensor();
	}
}

Tensor SVD_matrix(Tensor& t){
    return handle_svd_matrix<DType::Integer>(t); 
}

template<DType dt = DType::Float>
Tensor handle_svd_batched(Tensor& t){
    if(dt != t.dtype){return handle_svd_matrix<DTypeFuncs::next_dtype_it<dt> >(t);}
    if constexpr (DTypeFuncs::is_dtype_floating_v<dt>){
        using type_t = DTypeFuncs::dtype_to_type_t<dt>;
        int64_t n1 = t.shape()[1];
        int64_t n2 = t.shape()[2];
        int64_t batches = t.shape()[0];
        int64_t mat_size = n1 * n2;
        int64_t m = n2;
        int64_t n = n1;
        int64_t k = (m<n?m:n);
        Tensor tU({batches, m, k}, t.dtype);
        Tensor tS({batches, k}, t.dtype);
        Tensor tVT({batches, k, n}, t.dtype);
        int64_t INFO=0;
        char JOBU  = 'S';
        char JOBVT = 'S';
        int64_t wssize = 3*(m<n?m:n)+(m>n?m:n);
        int64_t wssize1 = 5*(m<n?m:n);
        wssize = (wssize>wssize1?wssize:wssize1);
        type_t* wsbuf = new type_t[wssize];
        for(int64_t b = 0; b < batches; ++b){
            svd<type_t>(&JOBU, &JOBVT, &m, &n, &reinterpret_cast<type_t*>(t.data_ptr())[b * mat_size], &m, &reinterpret_cast<type_t*>(tS.data_ptr())[b * k],
                &reinterpret_cast<type_t*>(tU.data_ptr())[b * m * k], &m, &reinterpret_cast<type_t*>(tVT.data_ptr())[b * k * n],
                &k, wsbuf, &wssize, &INFO);
            INFO = 0;
            JOBU = 'S';
            JOBVT = 'S';
            int64_t wssize = 3*(m<n?m:n)+(m>n?m:n);
            int64_t wssize1 = 5*(m<n?m:n);
            wssize = (wssize>wssize1?wssize:wssize1);
        }
        delete[] wsbuf;
        return functional::list(tU, tS, tVT);

    }
	else{
		utils::THROW_EXCEPTION(false, "Expected dtype to be flaoting but got dtype $", t.dtype);
		return Tensor();
	}
}

Tensor SVD(Tensor _t){
    Tensor t = _t.contiguous();
    if(t.dims() == 2){return SVD_matrix(t);}
    t = t.flatten(0, -3);
    assert(t.dims() == 3);
    return handle_svd_batched(t);
}

}}
