#include <fftw3.h>
#include <omp.h>

#include "hipacc_fft_helper.hpp"

#include <assert.h>
#include <iostream>
#include <type_traits>

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

  if (floatPrecision) {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(4);
  } else {
    fftw_init_threads();
    fftw_plan_with_nthreads(4);
  }

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

  // std::cout << "plan: " << image_height << " " << image_width << std::endl;
  auto start = std::chrono::system_clock::now();

  // create plans
  fft_plan plan_image = FFT<TPrecision>::fft_plan_dft_r2c_2d(
      image_height, image_width, in, image_fft, FFTW_ESTIMATE);

  auto end = std::chrono::system_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  // std::cout << (float)elapsed.count()/1000.0 << " ms" << std::endl;

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
    reinterpret_cast<std::complex<TPrecision> *>(image_fft)[ind] =
        reinterpret_cast<std::complex<TPrecision> *>(image_fft)[ind] *
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
  std::cout << (float)elapsed_time.count() / 1000.0 << " ms" << std::endl;

  // cleanup
  FFT<TPrecision>::fft_destroy_plan(plan_image);
  FFT<TPrecision>::fft_destroy_plan(plan_kernel);
  FFT<TPrecision>::fft_destroy_plan(plan_inverse);
  FFT<TPrecision>::fft_free(kernel);
  FFT<TPrecision>::fft_free(image_fft);
  FFT<TPrecision>::fft_free(kernel_fft);
  if (floatPrecision) {
    fftwf_cleanup_threads();
  } else {
    fftw_cleanup_threads();
  }
}

template <class V>
void fftConvolvePadded(float *in, int width, int height, V *kernel, int k_w,
                       int k_h, float *out, int padX, int padY) {
  int padWidth = width + padX;
  int padHeight = height + padY;
  int padMatsize = padWidth * padHeight;

  float *in_pad = (float *)fftwf_malloc(sizeof(float) * padMatsize);
  memset(in_pad, 0, sizeof(float) * padMatsize);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      in_pad[linearize(padWidth, x, y)] = in[y * (width) + x];
    }
  }

  float *out_pad = (float *)fftwf_malloc(sizeof(float) * padMatsize);

  fftConvolve(in_pad, padWidth, padHeight, kernel, k_w, k_h, out_pad);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      out[linearize(width, x, y)] = out_pad[y * (padWidth) + x];
    }
  }

  fftwf_free(in_pad);
  fftwf_free(out_pad);
}

template <class V>
void fftConvolvePadded(double *in, int width, int height, V *kernel, int k_w,
                       int k_h, double *out, int padX, int padY) {
  int padWidth = width + padX;
  int padHeight = height + padY;
  int padMatsize = padWidth * padHeight;

  double *in_pad = (double *)fftw_malloc(sizeof(double) * padMatsize);
  memset(in_pad, 0, sizeof(double) * padMatsize);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      in_pad[linearize(padWidth, x, y)] = in[y * (width) + x];
    }
  }

  double *out_pad = (double *)fftw_malloc(sizeof(double) * padMatsize);

  fftConvolve(in_pad, padWidth, padHeight, kernel, k_w, k_h, out_pad);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      out[linearize(width, x, y)] = out_pad[y * (padWidth) + x];
    }
  }

  fftw_free(in_pad);
  fftw_free(out_pad);
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
    padWidth = width + k_w / 2;
    padHeight = height + k_h / 2;
  }
  int padSize = padWidth * padHeight;

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
