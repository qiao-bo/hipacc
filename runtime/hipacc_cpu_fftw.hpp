#include <fftw3.h>
#include <omp.h>

#include "hipacc_fft_helper.hpp"

#include <assert.h>
#include <iostream>
#include <type_traits>

void fftConvolve(double *in, int width, int height,
                 float *k, int k_w, int k_h, double *out) {
  fftw_init_threads();
  fftw_plan_with_nthreads(4);

  int image_width = width;
  int image_height = height;
  int matsize = image_width * image_height;
  int intermediateSize = image_height * (image_width / 2 + 1);

  // setup buffers
  double *kernel = (double *)fftw_malloc(sizeof(double) * matsize);
  fftw_complex *image_fft =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * intermediateSize);
  fftw_complex *kernel_fft =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * intermediateSize);

  memset(kernel, 0, sizeof(double) * matsize);

  // prepare kernel
  putKernel((float *)k, kernel, k_w, k_h, image_width, image_height);

  // std::cout << "plan: " << image_height << " " << image_width << std::endl;
  auto start = std::chrono::system_clock::now();

  // create plans
  fftw_plan plan_image = fftw_plan_dft_r2c_2d(
      image_height, image_width, in, image_fft, FFTW_ESTIMATE);

  auto end = std::chrono::system_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  // std::cout << (float)elapsed.count()/1000.0 << " ms" << std::endl;

  fftw_plan plan_kernel = fftw_plan_dft_r2c_2d(image_height, image_width, kernel,
                                               kernel_fft, FFTW_ESTIMATE);
  fftw_plan plan_inverse = fftw_plan_dft_c2r_2d(image_height, image_width,
                                                image_fft, out, FFTW_ESTIMATE);

  // start
  auto start_time = std::chrono::system_clock::now();

  //#pragma omp parallel sections
  {
    //#pragma omp section
    { fftw_execute(plan_image); }
    //#pragma omp section
    { fftw_execute(plan_kernel); }
  }

// Pointwise Multiplication in Frequency Domain
#pragma omp parallel for
  for (int ind = 0; ind < intermediateSize; ++ind) {
    reinterpret_cast<std::complex<double> *>(image_fft)[ind] =
        reinterpret_cast<std::complex<double> *>(image_fft)[ind] *
        reinterpret_cast<std::complex<double> *>(kernel_fft)[ind];
  }

  fftw_execute(plan_inverse);

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
  fftw_destroy_plan(plan_image);
  fftw_destroy_plan(plan_kernel);
  fftw_destroy_plan(plan_inverse);
  fftw_free(kernel);
  fftw_free(image_fft);
  fftw_free(kernel_fft);
}

void fftConvolvePadded(double *in, int width, int height,
                       float *kernel, int k_w, int k_h,
                       double *out, int padX, int padY) {
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

template <class T, class U, size_t rows, size_t cols>
void fftwConvolution(T *in, int width, int height,
                     const float (&kernel)[rows][cols], int k_w, int k_h,
                     U *out) {
  assert(rows == k_h && cols == k_w);
  // prepare input buffer
  double *input;
  if (std::is_same<double, T>::value) {
    input = reinterpret_cast<double *>(in);
  } else {
    // convert to double
    input = new double[width * height];
    for (int i = 0; i < width * height; i++) {
      input[i] = (double)in[i];
    }
  }
  // prepare output buffer
  double *output;
  if (std::is_same<double, U>::value) {
    output = reinterpret_cast<double *>(out);
  } else {
    output = new double[width * height];
  }
  // prepare mask
  float *mask = new float[k_w * k_h];
  for (size_t r = 0; r < rows; r++) {
    for (size_t c = 0; c < cols; c++) {
      mask[r * cols + c] = kernel[r][c];
    }
  }
  fftConvolve(input, width, height, mask, k_w, k_h, output);
  if (!std::is_same<double, U>::value) {
    // convert output
    for (int i = 0; i < width * height; i++) {
      out[i] = (U)output[i];
    }
  }
}