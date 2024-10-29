#include "host/standard_synthesis.hpp"

#include <algorithm>
#include <complex>
#include <limits>
#include <cassert>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "context_internal.hpp"
#include "host/blas_api.hpp"
#include "host/eigensolver.hpp"
#include "host/gram_matrix.hpp"
#include "host/kernels/gemmexp.hpp"
#include "memory/allocator.hpp"
#include "memory/copy.hpp"
#include "memory/array.hpp"

#include <iostream>
#include <chrono>
#include <omp.h>


void exp_array(float* __restrict__ in, std::complex<float>* __restrict__ out,
          const std::size_t start, const std::size_t N) {
#pragma omp simd
  for (std::size_t i=start; i<start+N; i++) {
    float cos_; float sin_;
    sincosf(in[i], &sin_, &cos_);
    out[i].real(cos_);
    out[i].imag(sin_);
  }
}

void exp_array(double* __restrict__ in, std::complex<double>* __restrict__ out,
          const std::size_t start, const std::size_t N) {
#pragma omp simd
  for (std::size_t i=start; i<start+N; i++) {
    double cos_; double sin_;
    sincos(in[i], &sin_, &cos_);
    out[i].real(cos_);
    out[i].imag(sin_);
  }
}


namespace bipp {
namespace host {

template <typename T>
StandardSynthesis<T>::StandardSynthesis(std::shared_ptr<ContextInternal> ctx,
                                        StandardSynthesisOptions opt, std::size_t nImages,
                                        ConstHostView<T, 1> pixelX, ConstHostView<T, 1> pixelY,
                                        ConstHostView<T, 1> pixelZ)
    : ctx_(std::move(ctx)),
      opt_(opt),
      nImages_(nImages),
      nPixel_(pixelX.size()),
      count_(0),
      totalVisibilityCount_(0),
      pixel_(ctx_->host_alloc(), {pixelX.size(), 3}),
      img_(ctx_->host_alloc(), {nPixel_, nImages_}) {
  assert(pixelX.size() == pixelY.size());
  assert(pixelX.size() == pixelZ.size());
  //printf("nPixel_ = %ld\n", nPixel_);
  //printf("nImages = %ld\n", nImages_);
  copy(pixelX, pixel_.slice_view(0));
  copy(pixelY, pixel_.slice_view(1));
  copy(pixelZ, pixel_.slice_view(2));
  
  img_.zero();
}

template <typename T>
auto StandardSynthesis<T>::process(CollectorInterface<T>& collector) -> void {
  auto data = collector.get_data();

  for (const auto& s : data) {
    totalVisibilityCount_ += s.nVis;
    this->process_single(s.wl, s.nVis, s.v, s.dMasked, s.xyzUvw, s.w);
  }
}

template <typename T>
auto StandardSynthesis<T>::process_single(T wl, const std::size_t nVis, ConstView<std::complex<T>, 2> vView,
                                          ConstHostView<T, 2> dMasked, ConstView<T, 2> xyzUvwView,
                                          ConstView<std::complex<T>, 2> wView)
    -> void {
  HostArray<std::complex<T>, 2> v(ctx_->host_alloc(),vView.shape());
  copy(ConstHostView<std::complex<T>, 2>(vView), v);
  HostArray<std::complex<T>, 2> w(ctx_->host_alloc(),wView.shape());
  copy(ConstHostView<std::complex<T>, 2>(wView), w);
  ConstHostView<T, 2> xyz(xyzUvwView);

  const auto nEig     = dMasked.shape(0);
  const auto nAntenna = v.shape(0);
  const auto nBeam    = w.shape(0);
  
  assert(xyz.shape(1) == 3);
  assert(v.shape(0) == xyz.shape(0));
  assert(v.shape(1) == dMasked.shape(0));
  assert(img_.shape(1) == dMasked.shape(1));
  assert(w.shape(1) == nAntenna);

  printf("================== process_single input ===================\n");
  printf("xyz     = %4ld x %4ld = nAntenna x 3\n", xyz.shape(0), xyz.shape(1));
  printf("v       = %4ld x %4ld = nBeam    x nEig == dMasked.shape(0)\n", v.shape(0), v.shape(1));
  printf("w       = %4ld x %4ld = nBeam    x nAntenna\n", w.shape(0), w.shape(1));
  printf("dMasked = %4ld x %4ld = nEig     x nImage/nLevels\n", dMasked.shape(0), dMasked.shape(1));
  printf("===========================================================\n");


  HostArray<T, 2> dMaskedArray(ctx_->host_alloc(), dMasked.shape());
  copy(dMasked, dMaskedArray);

  auto dCount = HostArray<short, 1>(ctx_->host_alloc(), dMasked.shape(0));
  dCount.zero();
  for (std::size_t idxLevel = 0; idxLevel < nImages_; ++idxLevel) {
    auto mask = dMaskedArray.slice_view(idxLevel);
    for(std::size_t i = 0; i < mask.size(); ++i) {
      dCount[i] |= mask[i] != 0;
    }
  }

  // remove any eigenvalue that is zero for all level
  // by copying forward
  std::size_t nEigRemoved = 0;
  for (std::size_t i = 0; i < nEig; ++i) {
    if(dCount[i]) {
      if(nEigRemoved) {
        copy(v.slice_view(i), v.slice_view(i - nEigRemoved));
        for (std::size_t idxLevel = 0; idxLevel < nImages_; ++idxLevel) {
          dMaskedArray[{i - nEigRemoved, idxLevel}] = dMaskedArray[{i, idxLevel}];
        }
      }
    } else {
      ++nEigRemoved;
    }
  }

  const auto nEigMasked = nEig - nEigRemoved;
  printf("nEig = %ld, nEigRemoved = %ld -> nEigMasked = %ld\n", nEig, nEigRemoved, nEigMasked);

  auto unlayeredStats = HostArray<T, 2>(ctx_->host_alloc(), {nPixel_, nEigMasked});
  auto unlay = HostArray<T, 2>(ctx_->host_alloc(), {nPixel_, nEigMasked});
  auto unlay2 = HostArray<T, 2>(ctx_->host_alloc(), {nPixel_, nEigMasked});
  auto dMaskedReduced = dMaskedArray.sub_view({0, 0}, {nEigMasked, dMaskedArray.shape(1)});

  printf("v.strides(1) = %ld\n", v.strides(1));

  T alpha = 2.0 * M_PI / wl;

#if 0

  printf("@@@ sf gemmexp @@@\n");
  printf("ldout = %ld (unlayeredStats.strides(1))\n", unlayeredStats.strides(1));
  auto start = std::chrono::high_resolution_clock::now();
  gemmexp(nEigMasked, nPixel_, nAntenna, alpha, v.data(), v.strides(1), xyz.data(), xyz.strides(1),
          &pixel_[{0, 0}], &pixel_[{0, 1}], &pixel_[{0, 2}], unlayeredStats.data(),
          unlayeredStats.strides(1));
  ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "gemmexp", unlayeredStats);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> gemmexpms = end - start;
  printf("gemmexp = %.6f\n", gemmexpms.count());

  // cluster eigenvalues / vectors based on mask
  auto sc = std::chrono::high_resolution_clock::now();
  for (std::size_t idxLevel = 0; idxLevel < nImages_; ++idxLevel) {
    auto dMaskedSlice = dMaskedReduced.slice_view(idxLevel).sub_view(0, nEigMasked);

    auto imgCurrent = img_.slice_view(idxLevel);
    for (std::size_t idxEig = 0; idxEig < dMaskedSlice.size(); ++idxEig) {
      if (dMaskedSlice[idxEig]) {
        const auto scale = dMaskedSlice[idxEig];
        ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG, "Assigning (rescaled by nVis = {}) eigenvalue {} to bin {}",
                           nVis, dMaskedSlice[{idxEig}] * nVis, idxLevel);
        blas::axpy(nPixel_, scale, &unlayeredStats[{0, idxEig}], 1, imgCurrent.data(), 1);
      }
    }
  }
  auto ec = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> dc = ec - sc;
  printf("dc (clustering)     = %.6f\n", dc.count());
#endif
  
#if 1
  printf("@@@ eo gemmexp @@@\n");

  printf("nEigMasked = %ld\n", nEigMasked);
  std::size_t nWidth = std::sqrt(nPixel_);
  
  auto grid = pixel_.data();

  auto sn = std::chrono::high_resolution_clock::now();

  const std::size_t mt = static_cast<std::size_t>(omp_get_max_threads());
  printf("mt = %ld\n", mt);

  auto tmp    = HostArray<T, 3>(ctx_->host_alloc(), {mt, nAntenna, nWidth});
  auto tmpexp = HostArray<std::complex<T>, 3>(ctx_->host_alloc(), {mt, nAntenna, nWidth});
  auto tmpexp2 = HostArray<std::complex<T>, 3>(ctx_->host_alloc(), {mt, nAntenna, nWidth});
  auto tmpe   = HostArray<std::complex<T>, 3>(ctx_->host_alloc(), {mt, nEigMasked, nWidth});
  auto tmpi   = HostArray<T, 3>(ctx_->host_alloc(), {mt, nEigMasked, nWidth});
  
  auto sn2 = std::chrono::high_resolution_clock::now();
  const T zero = 0;
  auto unlay_ = unlay.data();
  auto tmp_ = tmp.data();
  auto tmpexp_ = tmpexp.data();
  auto tmpexp2_ = tmpexp2.data();

#pragma omp parallel
{
  const int tid = omp_get_thread_num();

#pragma omp for

  for (std::size_t i = 0; i<nWidth; i++) {
    //for (std::size_t i = 0; i<1; i++) {
      
    // Indices in global inputs (grid & pixel)
    const size_t igrid  = i * nWidth;
    const size_t ipixel = i * nWidth * nAntenna;
    const std::size_t itp  = tid * nAntenna * nWidth;
    const std::size_t ite  = tid * nEigMasked * nWidth;

#if 1
    //auto sg = std::chrono::high_resolution_clock::now();
    blas::gemm(CblasColMajor, CblasNoTrans, CblasTrans,
               nAntenna, nWidth, 3, alpha,
               xyz.data(), nAntenna,
               &grid[igrid], nPixel_,
               0,
               &tmp_[tid * nAntenna * nWidth], nAntenna);
    //auto eg = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> dg = eg - sg;
    //printf("-T- gemm i_w %4ld in %.6f\n", i, dg.count());
    
    // Compute exponential using sincos (needs -O3 --fast-math to vectorize)
    //auto se = std::chrono::high_resolution_clock::now();
    exp_array(tmp.data(), tmpexp.data(), itp, nAntenna * nWidth);
    //auto ee = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> de = ee - se;
#endif
    
    //printf("-T- i_w %4ld gemm %8.6f, exp %8.6f\n", i, dg.count(), de.count());
    /*
    printf("M==nEigMasked = %ld, N==nWidth = %ld, K==nAntenna = %ld\n",
           nEigMasked, nWidth, nAntenna);
    printf("A==V.shape = [%ld x %ld]\n", v.shape(0), v.shape(1));
    printf("B=tmpexp(1t) = [%ld x %ld]\n", nAntenna, nWidth);
    */

    // compare with gemmexp_original gemm + exp
    //auto se2 = std::chrono::high_resolution_clock::now();

    /*
    gemmexp_original(nAntenna, nWidth, 3, alpha,
                     xyz.data(), nAntenna,
                     &grid[igrid], nPixel_,
                     &tmpexp_[tid * nAntenna * nWidth], nAntenna);
    */
    /*
    gemmexp_(nAntenna, nWidth, 3, alpha,
             xyz.data(), nAntenna,
             &grid[igrid], nPixel_,
             &tmpexp_[tid * nAntenna * nWidth], nAntenna);
    */
    
    //auto ee2 = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> de2 = ee2 - se2;
    //printf("-T- i_w %4ld gemm %8.6f, exp2 %8.6f\n", i, dg.count(), de2.count());
    /*
    for (int i=tid * nAntenna * nWidth; i<tid * nAntenna * nWidth + 600; i++) {
      printf("-D- tmpexp vs tmpexp2 on %3d: (%10.6f, %10.6f) vs (%10.6f, %10.6f)\n",
             i, tmpexp_[i].real(), tmpexp_[i].imag(), tmpexp2_[i].real(), tmpexp2_[i].imag());
    }
    */
    //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    blas::gemm(CblasColMajor, CblasTrans, CblasNoTrans,
               //nEigMasked, nWidth, nAntenna, {1, 0},
               nEigMasked, nWidth, nAntenna, {1, 0},
               //v.data(), nEigMasked,
               v.data(), nEig,
               &tmpexp.data()[itp], nAntenna, 
               {0, 0},
               &tmpe.data()[ite], nEigMasked);

    //exit(1);
    /*
    if (i < 3)
      for (int ii=0; ii<3; ii++)
        printf("tid %d: tmpe[%d] = %12.6f + %12.6fi\n", tid, ii, tmpe.data()[ii].real(), tmpe.data()[ii].imag());
    fflush(stdout);
    */
    
    /// Fill unlayered stats from thread-wise temporary E
    //
    for (size_t j=0; j<nWidth; j++) {
#pragma omp simd
      for (size_t k=0; k<nEigMasked; k++) {
        auto idx_e = tid * nEigMasked * nWidth + j * nEigMasked + k;
        //auto idx_u = k *   nWidth * nWidth + i * nWidth + j;
        //unlay_[idx_u] = std::norm(tmpe.data()[idx_e]);
        tmpi.data()[idx_e] = std::norm(tmpe.data()[idx_e]);
        /*
        if (i < 2 && j < 3 && k < 3) {
          printf("ijk = %d, %d, %d: unlay_[%6ld] =  %10.6f (tid=%ld)\n",
                 i, j, k, idx_u, unlay_[idx_u], tid);
        }
        */
      }
    }
    
    // Transpose and copy out of place in unlay2
    blas::omatcopy(CblasColMajor, CblasTrans,
                   nEigMasked, nWidth,
                   1,
                   &tmpi.data()[tid * nEigMasked * nWidth], nEigMasked,
                   &unlay2.data()[i * nWidth], nPixel_);
    
  }
 }

  fflush(stdout);
  auto en2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> dn2 = en2 - sn2;
  printf("dn2 (gemm + exp + ...) = %.3f s\n", dn2.count());

  // gemmexp_original

  

  // cluster eigenvalues / vectors based on mask
  auto sc2 = std::chrono::high_resolution_clock::now();
  for (std::size_t idxLevel = 0; idxLevel < nImages_; ++idxLevel) {
    auto dMaskedSlice = dMaskedReduced.slice_view(idxLevel).sub_view(0, nEigMasked);

    auto imgCurrent = img_.slice_view(idxLevel);
    for (std::size_t idxEig = 0; idxEig < dMaskedSlice.size(); ++idxEig) {
      if (dMaskedSlice[idxEig]) {
        const auto scale = dMaskedSlice[idxEig];
        ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG, "Assigning (rescaled by nVis = {}) eigenvalue {} to bin {}",
                           nVis, dMaskedSlice[{idxEig}] * nVis, idxLevel);
        //blas::axpy(nPixel_, scale, &unlay[{0, idxEig}], 1, imgCurrent.data(), 1);
        blas::axpy(nPixel_, scale, &unlay2[{0, idxEig}], 1, imgCurrent.data(), 1);
      }
    }
  }
  auto ec2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> dc2 = ec2 - sc2;
  printf("dc (clustering)     = %.3f\n", dc2.count());
#endif  
  ++count_;
}

template <typename T>
auto StandardSynthesis<T>::get(View<T, 2> out) -> void {
  assert(out.shape(0) == nPixel_);
  assert(out.shape(1) == nImages_);

  HostView<T, 2> outHost(out);

  if (opt_.normalizeImage) {
    for (std::size_t i = 0; i < nImages_; ++i) {
      const T* __restrict__ localImg = &img_[{0, i}];
      T* __restrict__ outputImg = &outHost[{0, i}];

      const T scale = count_ ? static_cast<T>(1.0 / static_cast<double>(count_)) : 0;

      ctx_->logger().log(BIPP_LOG_LEVEL_DEBUG,
                         "StandardSynthesis<T>::get (host) totalVisibilityCount_ = {}, totalCollectCount_ = {}, scale = {}",
                         totalVisibilityCount_, count_, scale);

      for (std::size_t j = 0; j < nPixel_; ++j) {
        outputImg[j] = localImg[j] * scale;
      }
    }
  } else {
    copy(img_, outHost);
  }

  for (std::size_t i = 0; i < nImages_; ++i) {
    ctx_->logger().log_matrix(BIPP_LOG_LEVEL_DEBUG, "image output", out.slice_view(i));
  }
}

template class StandardSynthesis<double>;

template class StandardSynthesis<float>;

}  // namespace host
}  // namespace bipp
