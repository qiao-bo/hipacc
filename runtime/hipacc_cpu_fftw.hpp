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
  template <typename... TArgs> static fftwf_plan fft_plan_r2r_2d(TArgs... args) {
    return fftwf_plan_r2r_2d(args...);
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
  template <typename... TArgs> static void fft_set_timelimit(TArgs... args) {
    fftwf_set_timelimit(args...);
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
  template <typename... TArgs> static fftw_plan fft_plan_r2r_2d(TArgs... args) {
    return fftw_plan_r2r_2d(args...);
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
  template <typename... TArgs> static void fft_set_timelimit(TArgs... args) {
    fftw_set_timelimit(args...);
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
  FFT<TPrecision>::fft_plan_with_nthreads(2);
  const unsigned int PLANNER_FLAG = FFTW_MEASURE; // FFTW_ESTIMATE or FFTW_MEASURE
  FFT<TPrecision>::fft_set_timelimit(0.001);

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
      image_height, image_width, in, image_fft, PLANNER_FLAG);
  fft_plan plan_kernel = FFT<TPrecision>::fft_plan_dft_r2c_2d(
      image_height, image_width, kernel, kernel_fft, PLANNER_FLAG);
  fft_plan plan_inverse = FFT<TPrecision>::fft_plan_dft_c2r_2d(
      image_height, image_width, image_fft, out, PLANNER_FLAG);

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

  double singleFFTcost = 7.0 * N1 * N2 * log2(N1 * N2);

  // double FFTConvolutionCost1 = (singleFFTcost1 * 3 + (N1*N2*(3+1))) / THREADS;
  // double FFTConvolutionCost2 = (singleFFTcost2 * 3 + (N1*N2*(3+1))) / THREADS;

  double FFTConvolutionCost = (singleFFTcost * 3 + (N1 * N2 * (3 + 1))) / THREADS;
  std::cout << "FFTConvolutionCost: " << FFTConvolutionCost / FLOPMS << std::endl;

  double flopPerPixelInHipaccKernel; // depends on border condition
  if (linear) {
    flopPerPixelInHipaccKernel = 4;
  } else {
    flopPerPixelInHipaccKernel = 12;
  }
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

  /*
  estimateConvolutionExecutionTime(width, height, padWidth, padHeight, k_w, k_h,
                                   linear);
  */

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
void fftTransform(TPrecision *in, int width, int height, TPrecision *out,
                  bool forward = true, bool scale = false) {
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
  FFT<TPrecision>::fft_plan_with_nthreads(2);
  const unsigned int PLANNER_FLAG = FFTW_MEASURE; // FFTW_ESTIMATE or FFTW_MEASURE
  FFT<TPrecision>::fft_set_timelimit(0.001);

  // create plans
  fft_plan plan_image;
  if (forward) {
    fft_complex *fft = reinterpret_cast<fft_complex *>(out);
    plan_image = FFT<TPrecision>::fft_plan_dft_r2c_2d(height, width, in, fft,
                                                      PLANNER_FLAG);
  } else {
    fft_complex *fft = reinterpret_cast<fft_complex *>(in);
    plan_image = FFT<TPrecision>::fft_plan_dft_c2r_2d(height, width, fft, out,
                                                      PLANNER_FLAG);
  }

  FFT<TPrecision>::fft_execute(plan_image);

  if (!forward && scale) {
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

template <class TPrecision>
void fftTransformDevice(TPrecision *in, int width, int height, TPrecision *out,
                        bool forward = true, bool scale = false) {
  // static_assert(false, "fftTransformDevice not available for cpu!");
  std::cerr << "fftTransformDevice is not available for cpu! Using fftTransform "
               "instead."
            << std::endl;
  fftTransform(in, width, height, out, forward, scale);
}

template <class TPrecision>
void dctTransformDevice(TPrecision *in, int width, int height, TPrecision *out,
                        bool forward = true) {
  // static_assert(false, "dctTransformDevice not available for cpu!");
  std::cerr << "dctTransformDevice is not available for cpu! Using dctTransform "
               "instead."
            << std::endl;
  dctTransform(in, width, height, out, forward);
}

template <class TPrecision>
void dctTransform(TPrecision *in, int width, int height, TPrecision *out,
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
  FFT<TPrecision>::fft_plan_with_nthreads(2);
  const unsigned int PLANNER_FLAG = FFTW_MEASURE; // FFTW_ESTIMATE or FFTW_MEASURE
  FFT<TPrecision>::fft_set_timelimit(0.001);

  // setup buffers
  // create plans
  fft_plan plan_image;
  if (forward) {
    TPrecision *fft = reinterpret_cast<TPrecision *>(out);
    plan_image = FFT<TPrecision>::fft_plan_r2r_2d(
        height, width, in, fft, FFTW_REDFT10, FFTW_REDFT10, PLANNER_FLAG);
  } else {
    TPrecision *fft = reinterpret_cast<TPrecision *>(in);
    plan_image = FFT<TPrecision>::fft_plan_r2r_2d(
        height, width, fft, out, FFTW_REDFT01, FFTW_REDFT01, PLANNER_FLAG);
  }

  FFT<TPrecision>::fft_execute(plan_image);

  if (!forward) {
    // scale
#pragma omp parallel for
    for (int ind = 0; ind < width * height; ++ind) {
      out[ind] /= (2 * width * 2 * height);
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

  fftTransform(input, width, height, output);

  free(input);

  return output;
}

// inverse FFT
template <class T, class TPrecision> void ifft(TPrecision *in, HipaccImage &out, bool scaled = true) {
  int width = out->width;
  int height = out->height;
  int width_out = alignedWidth<T>(width, out->alignment);

  // prepare output buffer
  TPrecision *output = new TPrecision[width * height];

  fftTransform(in, width, height, output, false, scaled);

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
void fftToMagnitude(TPrecision *in, HipaccImage &mag) {
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

  free(tmp);
}

// create magnitude from dct
template <class T, class TPrecision>
void dctToMagnitude(TPrecision *in, HipaccImage &mag) {
  int width = mag->width;
  int height = mag->height;
  int width_out = alignedWidth<T>(width, mag->alignment);

  T *tmp = new T[width * height];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      tmp[y * width + x] = std::abs(in[y * width + x]);
    }
  }

  float max = 0.0;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (tmp[y * width + x] > max)
        max = tmp[y * width + x];
    }
  }
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      tmp[y * width + x] =
          (T)(255.0 * pow(tmp[y * width + x] * (1.0 / max), 1.0 / 4.0));
    }
  }

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      ((T *)(mag->mem))[y * width_out + x] = (T)tmp[y * width + x];
    }
  }

  free(tmp);
}

// apply mask mag to result of FFT in
// and ignore values for frequencies lower than r
template <class T, class TPrecision>
void fftScaleMagnitude(TPrecision *in, HipaccImage &mag) {
  int width = mag->width;
  int height = mag->height;
  int width_in = alignedWidth<T>(width, mag->alignment);

  T *scale = new T[width_in * height];
  memcpy(scale, mag->mem, sizeof(T) * width_in * height);

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

// forward DCT
template <class T, class TPrecision> TPrecision *dct(HipaccImage &in) {
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
  TPrecision *output = new TPrecision[width * height];

  dctTransform(input, width, height, output);

  free(input);

  return output;
}

// inverse DCT
template <class T, class TPrecision> void idct(TPrecision *in, HipaccImage &out) {
  int width = out->width;
  int height = out->height;
  int width_out = alignedWidth<T>(width, out->alignment);

  // prepare output buffer
  TPrecision *output = new TPrecision[width * height];

  dctTransform(in, width, height, output, false);

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
