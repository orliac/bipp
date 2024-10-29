#include "host/kernels/gemmexp.hpp"

#include <cmath>
#include <complex>
#include <vector>

#include "host/omp_definitions.hpp"

#ifdef BIPP_VC
#include <Vc/Vc>
#endif

namespace bipp {
namespace host {

template <typename T>
auto gemmexp(std::size_t nEig, std::size_t nPixel, std::size_t nAntenna, T alpha,
             const std::complex<T>* __restrict__ vUnbeam, std::size_t ldv,
             const T* __restrict__ xyz, std::size_t ldxyz, const T* __restrict__ pixelX,
             const T* __restrict__ pixelY, const T* __restrict__ pixelZ, T* __restrict__ out,
             std::size_t ldout) -> void {
#ifdef BIPP_VC

  using simdType = Vc::Vector<T>;
  constexpr std::size_t simdSize = simdType::size();

  const simdType alphaVec = alpha;
  const typename simdType::IndexType indexComplex([](auto&& n) { return 2 * n; });

  BIPP_OMP_PRAGMA("omp parallel for schedule(static)")
  for (std::size_t idxPix = 0; idxPix < nPixel; ++idxPix) {
    const simdType pX = pixelX[idxPix];
    const simdType pY = pixelY[idxPix];
    const simdType pZ = pixelZ[idxPix];

    for (std::size_t idxEig = 0; idxEig < nEig; ++idxEig) {
      simdType sumReal = 0;
      simdType sumImag = 0;

      std::size_t idxAnt = 0;
      for (; idxAnt + simdSize <= nAntenna; idxAnt += simdSize) {
        simdType x(xyz + idxAnt, Vc::Unaligned);
        simdType y(xyz + ldxyz + idxAnt, Vc::Unaligned);
        simdType z(xyz + 2 * ldxyz + idxAnt, Vc::Unaligned);

        const auto imag = alphaVec * Vc::fma(pX, x, Vc::fma(pY, y, pZ * z));

        simdType cosValue, sinValue;
        Vc::sincos(imag, &sinValue, &cosValue);

        const T* vUnbeamScalarPtr = reinterpret_cast<const T*>(vUnbeam + idxEig * ldv + idxAnt);
        simdType vValueReal(vUnbeamScalarPtr, indexComplex);
        simdType vValueImag(vUnbeamScalarPtr + 1, indexComplex);

        sumReal += vValueReal * cosValue - vValueImag * sinValue;
        sumImag += Vc::fma(vValueReal, sinValue, vValueImag * cosValue);
      }

      const auto tail = nAntenna - idxAnt;
      if (tail) {
        simdType x, y, z;
        x.setZero();
        y.setZero();
        z.setZero();
        for (std::size_t i = 0; i < tail; ++i) {
          x[i] = xyz[idxAnt + i];
          y[i] = xyz[idxAnt + ldxyz + i];
          z[i] = xyz[idxAnt + 2 * ldxyz + i];
        }
        const auto imag = alphaVec * Vc::fma(pX, x, Vc::fma(pY, y, pZ * z));

        simdType cosValue, sinValue;
        Vc::sincos(imag, &sinValue, &cosValue);

        simdType vValueReal;
        simdType vValueImag;
        for (std::size_t i = 0; i < tail; ++i) {
          const auto vValue = vUnbeam[idxEig * ldv + idxAnt + i];
          vValueReal[i] = vValue.real();
          vValueImag[i] = vValue.imag();
        }
        auto tailSumReal = vValueReal * cosValue - vValueImag * sinValue;
        auto tailSumImag = Vc::fma(vValueReal, sinValue, vValueImag * cosValue);

        for (std::size_t i = 0; i < tail; ++i) {
          sumReal[i] += tailSumReal[i];
          sumImag[i] += tailSumImag[i];
        }
      }

      const T sumRealScalar = sumReal.sum();
      const T sumImagScalar = sumImag.sum();
      out[idxEig * ldout + idxPix] = sumRealScalar * sumRealScalar + sumImagScalar * sumImagScalar;
    }
  }

#else

  BIPP_OMP_PRAGMA("omp parallel") {
    std::vector<std::complex<T> > pixSumVec(nEig);

    BIPP_OMP_PRAGMA("omp for schedule(static)")
    for (std::size_t idxPix = 0; idxPix < nPixel; ++idxPix) {
      const auto pX = pixelX[idxPix];
      const auto pY = pixelY[idxPix];
      const auto pZ = pixelZ[idxPix];
      for (std::size_t idxAnt = 0; idxAnt < nAntenna; ++idxAnt) {
        const auto imag =
            alpha * (pX * xyz[idxAnt] + pY * xyz[idxAnt + ldxyz] + pZ * xyz[idxAnt + 2 * ldxyz]);
        //const std::complex<T> ie{std::cos(imag), std::sin(imag)};
        const std::complex<T> cim_part(0, imag);
        const std::complex<T> ie = std::exp(cim_part);
        for (std::size_t idxEig = 0; idxEig < nEig; ++idxEig) {
          pixSumVec[idxEig] += vUnbeam[idxEig * ldv + idxAnt] * ie;
        }
      }
      for (std::size_t idxEig = 0; idxEig < nEig; ++idxEig) {
        const auto pv = pixSumVec[idxEig];
        pixSumVec[idxEig] = 0;
        out[idxEig * ldout + idxPix] = pv.real() * pv.real() + pv.imag() * pv.imag();
      }
    }
  }

#endif
}

template <typename T>
auto gemmexp_eo(const std::size_t nPixel, const std::size_t nAntenna, const T alpha) -> void {
    printf("hi from gemmexp_eo\n");
}

const std::size_t M_BLOCK_SIZE = 10000;
const std::size_t N_BLOCK_SIZE = 10000;

/*
*  Special gemm with vectorized exponentiation
*/
template <typename T>
auto gemmexp_(const std::size_t M,
              const std::size_t N,
              const std::size_t K,
              const T           alpha,
              const T* __restrict__ A,
              const std::size_t lda,
              const T* __restrict__ B,
              const std::size_t ldb,
              std::complex<T>* __restrict__ C,
              const std::size_t ldc) -> void {
  
  assert(K == 3);
  
  for (std::size_t i=0; i<M; i++) {
    
    T a0 = A[i];
    T a1 = A[i + lda];
    T a2 = A[i + 2 * lda];
    
    for (std::size_t j=0; j<N; j++) {
      
      T b0 = B[j];
      T b1 = B[j + ldb];
      T b2 = B[j + 2*ldb];
      
      T im_part = alpha * (a0*b0 + a1*b1 + a2*b2);
      std::complex<T> cim_part(0, im_part);
      std::complex<T> c = std::exp(cim_part);
      
      C[j * M + i] = c;
    }
  }
}
  
/*
*  Special gemm with vectorized exponentiation
*/
template <typename T>
auto gemmexp_original(const std::size_t M,
                      const std::size_t N,
                      const std::size_t K,
                      const T           alpha,
                      const T* __restrict__ A,
                      const std::size_t lda,
                      const T* __restrict__ B,
                      const std::size_t ldb,
                      std::complex<T>* __restrict__ C,
                      const std::size_t ldc) -> void {
  
  assert(K == 3);
  
  T sin_, cos_;
  
  const T zero = 0.0;
  
  std::size_t idx_c = 0;
  std::size_t idx_b = 0;
  
  for (std::size_t ib = 0; ib < M; ib += M_BLOCK_SIZE ) {
    
    std::size_t Mb = std::min(M_BLOCK_SIZE, M - ib);
    
    for (std::size_t jb = 0; jb < N; jb += N_BLOCK_SIZE) {

      std::size_t Nb = std::min(N_BLOCK_SIZE, N - jb);

      for (std::size_t j = 0; j < Nb; j++) {

        //idx_b = jb*3 + j*K;
        idx_b = jb + j;
        idx_c = (j + jb) * ldc + ib;
        
#pragma vector always
        for (std::size_t i = 0; i < Mb; i = i + 1) {
          T a0 = A[ib + i];
          T a1 = A[ib + i + lda];
          T a2 = A[ib + i + 2 * lda];
          //T b0 = B[idx_b];
          //T b1 = B[idx_b + 1];
          //T b2 = B[idx_b + 2];
          T b0 = B[idx_b];
          T b1 = B[idx_b + ldb];
          T b2 = B[idx_b + 2 * ldb];
          //printf("idx_b: %ld %ld %ld\n", idx_b, idx_b + ldb, idx_b + 2 * ldb);
          //printf("%.6f %.6f %.6f\n", a0, a1, a2);
          //printf("%.6f %.6f %.6f\n", b0, b1, b2);
          //printf("check = %.6f\n", (a0*b0 + a1*b1 + a2*b2));
          T im_part = alpha * (a0*b0 + a1*b1 + a2*b2);
          //printf("im_part = %.6f\n", im_part);
          
          std::complex<T> cim_part(zero, im_part);
          C[idx_c + i] = std::exp(cim_part);
        }
      }
    }
  }
}

template auto gemmexp_<float>(const std::size_t M,
                              const std::size_t N,
                              const std::size_t K,
                              const float           alpha,
                              const float* __restrict__ A,
                              const std::size_t lda,
                              const float* __restrict__ B,
                              const std::size_t ldb,
                              std::complex<float>* __restrict__ C,
                              const std::size_t ldc) -> void;
    
template auto gemmexp_<double>(const std::size_t M,
                               const std::size_t N,
                               const std::size_t K,
                               const double           alpha,
                               const double* __restrict__ A,
                               const std::size_t lda,
                               const double* __restrict__ B,
                               const std::size_t ldb,
                               std::complex<double>* __restrict__ C,
                               const std::size_t ldc) -> void;
  
template auto gemmexp_original<float>(const std::size_t M,
                                      const std::size_t N,
                                      const std::size_t K,
                                      const float           alpha,
                                      const float* __restrict__ A,
                                      const std::size_t lda,
                                      const float* __restrict__ B,
                                      const std::size_t ldb,
                                      std::complex<float>* __restrict__ C,
                                      const std::size_t ldc) -> void;
  
template auto gemmexp_original<double>(const std::size_t M,
                                       const std::size_t N,
                                       const std::size_t K,
                                       const double           alpha,
                                       const double* __restrict__ A,
                                       const std::size_t lda,
                                       const double* __restrict__ B,
                                       const std::size_t ldb,
                                       std::complex<double>* __restrict__ C,
                                       const std::size_t ldc) -> void;


template auto gemmexp<float>(std::size_t nEig, std::size_t nPixel, std::size_t nAntenna,
                             float alpha, const std::complex<float>* __restrict__ vUnbeam,
                             std::size_t ldv, const float* __restrict__ xyz, std::size_t ldxyz,
                             const float* __restrict__ pixelX, const float* __restrict__ pixelY,
                             const float* __restrict__ pixelZ, float* __restrict__ out,
                             std::size_t ldout) -> void;

template auto gemmexp<double>(std::size_t nEig, std::size_t nPixel, std::size_t nAntenna,
                              double alpha, const std::complex<double>* __restrict__ vUnbeam,
                              std::size_t ldv, const double* __restrict__ xyz, std::size_t ldxyz,
                              const double* __restrict__ pixelX, const double* __restrict__ pixelY,
                              const double* __restrict__ pixelZ, double* __restrict__ out,
                              std::size_t ldout) -> void;

// eo former implementation
template auto gemmexp_eo<float>(const std::size_t nPixel, const std::size_t nAntenna, const float alpha) -> void;

template auto gemmexp_eo<double>(const std::size_t nPixel, const std::size_t nAntenna, const double alpha) -> void;


}  // namespace host
}  // namespace bipp
