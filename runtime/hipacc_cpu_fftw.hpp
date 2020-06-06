#include <fftw3.h>
#include <omp.h>

#include "hipacc_fft_helper.hpp"

#include <assert.h>
#include <iostream>
#include <type_traits>

#include <cmath>
#include <complex>
#include <string>

template <class P> struct FFT {};
// fftw wrapper functinos for single precision
template <> struct FFT<float> {
  template <typename... TArgs> static void *fft_malloc(TArgs... args) {
    return fftwf_malloc(args...);
  }
  template <typename... TArgs>
  static fftwf_plan fft_plan_dft_r2c_2d(TArgs... args) {
    return fftwf_plan_dft_r2c_2d(args...);
  }
  template <typename... TArgs>
  static fftwf_plan fft_plan_dft_c2r_2d(TArgs... args) {
    return fftwf_plan_dft_c2r_2d(args...);
  }
  template <typename... TArgs> static void fft_execute(TArgs... args) {
    fftwf_execute(args...);
  }
  template <typename... TArgs> static void fft_destroy_plan(TArgs... args) {
    fftwf_destroy_plan(args...);
  }
  template <typename... TArgs> static void fft_free(TArgs... args) {
    fftwf_free(args...);
  }
  template <typename... TArgs> static void fft_flops(TArgs... args) {
    fftwf_flops(args...);
  }
  template <typename... TArgs> static void fft_init_threads(TArgs... args) {
    fftwf_init_threads(args...);
  }
  template <typename... TArgs> static void fft_plan_with_nthreads(TArgs... args) {
    fftwf_plan_with_nthreads(args...);
  }
  template <typename... TArgs> static void fft_cleanup_threads(TArgs... args) {
    fftwf_cleanup_threads(args...);
  }
};
// fftw wrapper functinos for double precision
template <> struct FFT<double> {
  template <typename... TArgs> static void *fft_malloc(TArgs... args) {
    return fftw_malloc(args...);
  }
  template <typename... TArgs>
  static fftw_plan fft_plan_dft_r2c_2d(TArgs... args) {
    return fftw_plan_dft_r2c_2d(args...);
  }
  template <typename... TArgs>
  static fftw_plan fft_plan_dft_c2r_2d(TArgs... args) {
    return fftw_plan_dft_c2r_2d(args...);
  }
  template <typename... TArgs> static void fft_execute(TArgs... args) {
    fftw_execute(args...);
  }
  template <typename... TArgs> static void fft_destroy_plan(TArgs... args) {
    fftw_destroy_plan(args...);
  }
  template <typename... TArgs> static void fft_free(TArgs... args) {
    fftw_free(args...);
  }
  template <typename... TArgs> static void fft_flops(TArgs... args) {
    fftw_flops(args...);
  }
  template <typename... TArgs> static void fft_init_threads(TArgs... args) {
    fftw_init_threads(args...);
  }
  template <typename... TArgs> static void fft_plan_with_nthreads(TArgs... args) {
    fftw_plan_with_nthreads(args...);
  }
  template <typename... TArgs> static void fft_cleanup_threads(TArgs... args) {
    fftw_cleanup_threads(args...);
  }
};

template <class TPrecision, class V>
void fftConvolve(TPrecision *in, int width, int height, V *k, int k_w, int k_h,
                 TPrecision *out) {
  typedef
      typename std::conditional<std::is_same<TPrecision, float>::value,
                                fftwf_complex, fftw_complex>::type fft_complex;
  typedef typename std::conditional<std::is_same<TPrecision, float>::value,
                                    fftwf_plan, fftw_plan>::type fft_plan;
  bool floatPrecision = false;
  if (std::is_same<TPrecision, float>::value) {
    floatPrecision = true;
  }

  FFT<TPrecision>::fft_init_threads();
  FFT<TPrecision>::fft_plan_with_nthreads(4);

  int image_width = width;
  int image_height = height;
  int matsize = image_width * image_height;
  int intermediateSize = image_height * (image_width / 2 + 1);

  // setup buffers
  TPrecision *kernel =
      (TPrecision *)FFT<TPrecision>::fft_malloc(sizeof(TPrecision) * matsize);
  fft_complex *image_fft = (fft_complex *)FFT<TPrecision>::fft_malloc(
      sizeof(fft_complex) * intermediateSize);
  fft_complex *kernel_fft = (fft_complex *)FFT<TPrecision>::fft_malloc(
      sizeof(fft_complex) * intermediateSize);

  memset(kernel, 0, sizeof(TPrecision) * matsize);

  // prepare kernel
  putKernel(k, kernel, k_w, k_h, image_width, image_height);

  // create plans
  fft_plan plan_image = FFT<TPrecision>::fft_plan_dft_r2c_2d(
      image_height, image_width, in, image_fft, FFTW_ESTIMATE);
  fft_plan plan_kernel = FFT<TPrecision>::fft_plan_dft_r2c_2d(
      image_height, image_width, kernel, kernel_fft, FFTW_ESTIMATE);
  fft_plan plan_inverse = FFT<TPrecision>::fft_plan_dft_c2r_2d(
      image_height, image_width, image_fft, out, FFTW_ESTIMATE);

  // start
  auto start_time = std::chrono::system_clock::now();

  //#pragma omp parallel sections
  {
    //#pragma omp section
    { FFT<TPrecision>::fft_execute(plan_image); }
    //#pragma omp section
    { FFT<TPrecision>::fft_execute(plan_kernel); }
  }

// Pointwise Multiplication in Frequency Domain
#pragma omp parallel for
  for (int ind = 0; ind < intermediateSize; ++ind) {
    reinterpret_cast<std::complex<TPrecision> *>(image_fft)[ind] *=
        reinterpret_cast<std::complex<TPrecision> *>(kernel_fft)[ind];
  }

  FFT<TPrecision>::fft_execute(plan_inverse);

// scale
#pragma omp parallel for
  for (int ind = 0; ind < matsize; ++ind) {
    out[ind] /= (image_height * image_width);
  }

  // stop
  auto end_time = std::chrono::system_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  std::cout << "h: " << height << " w: " << width << std::endl;
  std::cout << (float)elapsed_time.count() / 1000.0 << " ms" << std::endl;

  // time estimation
  double add_i, add_k, add_inv, mul_i, mul_k, mul_inv, fma_i, fma_k, fma_inv;
  FFT<TPrecision>::fft_flops(plan_image, &add_i, &mul_i, &fma_i);
  FFT<TPrecision>::fft_flops(plan_kernel, &add_k, &mul_k, &fma_k);
  FFT<TPrecision>::fft_flops(plan_inverse, &add_inv, &mul_inv, &fma_inv);
  double FLOPMS = 4000000;
  double simpleFFTcost =
      (add_i + mul_i + 2 * fma_i) +
      (add_k + mul_k + 2 * fma_k); // + (add_inv + mul_inv + 2*fma_inv)
                                   //+ (intermediateSize * 2) + matsize;
  // simpleFFTcost = intermediateSize * 2 * 3;
  simpleFFTcost *= 10;
  double parallelFFTcost = simpleFFTcost / 4;
  // std::cout << "simpleFFTcost: " << simpleFFTcost/FLOPMS << "
  // parallelFFTConvCost: " << parallelFFTcost/FLOPMS << std::endl;

  // cleanup
  FFT<TPrecision>::fft_destroy_plan(plan_image);
  FFT<TPrecision>::fft_destroy_plan(plan_kernel);
  FFT<TPrecision>::fft_destroy_plan(plan_inverse);
  FFT<TPrecision>::fft_free(kernel);
  FFT<TPrecision>::fft_free(image_fft);
  FFT<TPrecision>::fft_free(kernel_fft);
  FFT<TPrecision>::fft_cleanup_threads();
}

bool estimateConvolutionExecutionTime(int width, int height, int padWidth,
                                      int padHeight, int k_w, int k_h,
                                      bool linear) {
  // assumed 4 GFLOPS per core
  const double FLOPMS = 4000000; // FLOP per MS
  const double THREADS = 4;

  double N1 = padWidth;
  double N2 = padHeight;
  // for power of 2
  // double N1cost = 2*N1 * log2(N1) /*additions*/ + 4*N1 * log2(N1)
  // /*multiplications*/; double N2cost = 2*N2 * log2(N2) /*additions*/ + 4*N2 *
  // log2(N2) /*multiplications*/; double singleFFTcost1 = N1 * N2cost + N2 *
  // N1cost; double singleFFTcost2 = 6.0*N1*N2*log2(N1*N2)
  // + 2.0*N1*N2*log2(N1*N2); // 6 for complex mul and 2 for complex add

  double singleFFTcost = 7.0 * N1 * N2 * log2(N1 * N2);

  // double FFTConvolutionCost1 = (singleFFTcost1 * 3 + (N1*N2*(3+1))) / THREADS;
  // double FFTConvolutionCost2 = (singleFFTcost2 * 3 + (N1*N2*(3+1))) / THREADS;

  double FFTConvolutionCost = (singleFFTcost * 3 + (N1 * N2 * (3 + 1))) / THREADS;
  std::cout << "FFTConvolutionCost: " << FFTConvolutionCost / FLOPMS << std::endl;

  double flopPerPixelInHipaccKernel = 4; // depends on border condition
  double simpleHipaccCost =
      (height * width) * (k_w * k_h) * (2 + flopPerPixelInHipaccKernel);
  double parallelHipaccCost = simpleHipaccCost / THREADS;
  std::cout << "simpleHipaccCost: " << simpleHipaccCost / FLOPMS << std::endl;

  if (FFTConvolutionCost < simpleHipaccCost) {
    std::cout << "FFT Convolution will be faster" << std::endl;
    return true;
  } else {
    std::cout << "Hipacc Kernel will be faster" << std::endl;
    return false;
  }
}

template <class TPrecision, class T, class U, class V, size_t rows, size_t cols,
          class B = int>
void fftwConvolution(T *in, int width, int height, const V (&kernel)[rows][cols],
                     int k_w, int k_h, U *out, int alignment, bool linear,
                     B boundaryConstant = 0) {
  assert(rows == k_h && cols == k_w);

  int width_in = alignedWidth<T>(width, alignment);
  int width_out = alignedWidth<U>(width, alignment);

  int padWidth = width;
  int padHeight = height;
  if (linear) {
    // required padding for linear convolution
    padWidth = width + k_w / 2;
    padHeight = height + k_h / 2;
    // additional padding for performance
    padWidth = upperPowerOfTwo(padWidth); // width has more influence
    padHeight = nextDiv(padHeight, 8);
  }
  int padSize = padWidth * padHeight;

  estimateConvolutionExecutionTime(width, height, padWidth, padHeight, k_w, k_h,
                                   linear);

  // prepare input buffer
  TPrecision *input = new TPrecision[padSize];
  if (linear) {
    std::fill_n(input, padSize, (TPrecision)boundaryConstant);
  }
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      input[y * padWidth + x] = (TPrecision)in[y * width_in + x];
    }
  }
  // prepare output buffer
  TPrecision *output = new TPrecision[padSize];
  fftConvolve(input, padWidth, padHeight, (V *)(&kernel[0][0]), k_w, k_h, output);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      out[y * width_out + x] = (U)output[y * padWidth + x];
    }
  }

  free(input);
  free(output);
}

template <class TPrecision>
void fftw_transform(TPrecision *in, int width, int height, TPrecision *out,
                    bool forward = true) {
  typedef
      typename std::conditional<std::is_same<TPrecision, float>::value,
                                fftwf_complex, fftw_complex>::type fft_complex;
  typedef typename std::conditional<std::is_same<TPrecision, float>::value,
                                    fftwf_plan, fftw_plan>::type fft_plan;
  bool floatPrecision = false;
  if (std::is_same<TPrecision, float>::value) {
    floatPrecision = true;
  }

  FFT<TPrecision>::fft_init_threads();
  FFT<TPrecision>::fft_plan_with_nthreads(4);

  // setup buffers
  // create plans
  fft_plan plan_image;
  if (forward) {
    fft_complex *fft = reinterpret_cast<fft_complex *>(out);
    plan_image = FFT<TPrecision>::fft_plan_dft_r2c_2d(height, width, in, fft,
                                                      FFTW_ESTIMATE);
  } else {
    fft_complex *fft = reinterpret_cast<fft_complex *>(in);
    plan_image = FFT<TPrecision>::fft_plan_dft_c2r_2d(height, width, fft, out,
                                                      FFTW_ESTIMATE);
  }

  FFT<TPrecision>::fft_execute(plan_image);
  if (!forward) {
    // scale
#pragma omp parallel for
    for (int ind = 0; ind < width * height; ++ind) {
      out[ind] /= (width * height);
    }
  }

  // cleanup
  FFT<TPrecision>::fft_destroy_plan(plan_image);
  FFT<TPrecision>::fft_cleanup_threads();
}

// forward FFT
template <class T, class TPrecision> TPrecision *fft(HipaccImage &in) {
  int width = in->width;
  int height = in->height;
  int width_in = alignedWidth<T>(width, in->alignment);

  // prepare input buffer
  TPrecision *input = new TPrecision[width * height];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      input[y * width + x] = (TPrecision)((T *)(in->mem))[y * width_in + x];
    }
  }
  // prepare output buffer
  TPrecision *output = new TPrecision[2 * (width / 2 + 1) * height];

  fftw_transform(input, width, height, output);

  free(input);

  return output;
}

// inverse FFT
template <class T, class TPrecision> void ifft(TPrecision *in, HipaccImage &out) {
  int width = out->width;
  int height = out->height;
  int width_out = alignedWidth<T>(width, out->alignment);

  // prepare output buffer
  TPrecision *output = new TPrecision[width * height];

  fftw_transform(in, width, height, output, false);

  // truncate values outside of range 0-255
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      T val = output[y * width + x];
      if (output[y * width + x] < 0) {
        val = 0;
      } else if (output[y * width + x] > 255) {
        val = 255;
      }
      // convert output
      ((T *)(out->mem))[y * width_out + x] = (T)(val);
    }
  }

  free(output);
}

// create magnitude from fft
template <class T, class TPrecision>
void fftToMag(TPrecision *in, HipaccImage &mag) {
  int width = mag->width;
  int height = mag->height;
  int width_out = alignedWidth<T>(width, mag->alignment);

  T *tmp = new T[width * height];
  calcMagnitude(reinterpret_cast<std::complex<TPrecision> *>(in), tmp, width,
                height);

  shiftFFT(tmp, width, height);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      ((T *)(mag->mem))[y * width_out + x] = (T)tmp[y * width + x];
    }
  }
}

// apply mask mag to result of FFT in
// and ignore values for frequencies lower than r
template <class T, class TPrecision>
void magScaleFFT(TPrecision *in, HipaccImage &mag, float r) {
  int width = mag->width;
  int height = mag->height;
  int width_in = alignedWidth<T>(width, mag->alignment);

  T *scale = new T[width_in * height];
  memcpy(scale, mag->mem, sizeof(T) * width_in * height);

  magResetLowFreq(scale, width, height, mag->alignment, r, 10);
  magLowPassFilter(scale, width, height, mag->alignment,
                   min(width, height) * 0.25, 100);

  iShiftFFT(scale, width, height, mag->alignment);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < (width / 2 + 1); x++) {
      in[(y * (width / 2 + 1) + x) * 2] *=
          (TPrecision)scale[y * width_in + x] / 255;
      in[(y * (width / 2 + 1) + x) * 2 + 1] *=
          (TPrecision)scale[y * width_in + x] / 255;
    }
  }

  free(scale);
}
