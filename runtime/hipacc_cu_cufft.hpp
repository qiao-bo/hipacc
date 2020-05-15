#include <cufft.h>
#include <cufftXt.h>

#include "hipacc_fft_helper.hpp"

#include <iostream>

// Complex data type
struct d2 {
  double x;
  double y;
};
typedef d2 DoubleComplex;
// typedef float2 Complex;
struct f2 {
  float x;
  float y;
};
typedef f2 Complex;

// Mul
__device__ cufftComplex C_Mul(cufftComplex a, cufftComplex b) {
  cufftComplex temp;
  temp.x = a.x * b.x - a.y * b.y;
  temp.y = a.y * b.x + b.y * a.x;
  return temp;
}
// Mul double
__device__ cufftDoubleComplex C_Mul(cufftDoubleComplex a, cufftDoubleComplex b) {
  cufftDoubleComplex temp;
  temp.x = a.x * b.x - a.y * b.y;
  temp.y = a.y * b.x + b.y * a.x;
  return temp;
}
// Scale
__device__ cufftComplex C_Scale(cufftComplex a, float s) {
  cufftComplex c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}
// Scale double
__device__ cufftDoubleComplex C_Scale(cufftDoubleComplex a, double s) {
  cufftDoubleComplex c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}
// Mul and scale
__global__ void MulScalePw(cufftDoubleComplex *image, cufftDoubleComplex *filter,
                           int Nx, int Ny, double scale) {
  int tix = threadIdx.x + blockIdx.x * blockDim.x;
  int tiy = threadIdx.y + blockIdx.y * blockDim.y;
  int tid = tix + Nx * tiy;

  if (tid >= Nx * Ny)
    return;

  if ((tix < Nx) && (tiy < Ny)) {
    image[tid] = C_Scale(C_Mul(image[tid], filter[tid]), scale);
  }
}
// Mul and scale float
__global__ void MulScalePw(cufftComplex *image, cufftComplex *filter, int Nx,
                           int Ny, float scale) {
  int tix = threadIdx.x + blockIdx.x * blockDim.x;
  int tiy = threadIdx.y + blockIdx.y * blockDim.y;
  int tid = tix + Nx * tiy;

  if (tid >= Nx * Ny)
    return;

  if ((tix < Nx) && (tiy < Ny)) {
    image[tid] = C_Scale(C_Mul(image[tid], filter[tid]), scale);
  }
}

template <class TPrecision, class V>
void fftConvolve(TPrecision *in, int width, int height, V *k, int k_w, int k_h,
                 TPrecision *out) {
  typedef typename std::conditional<std::is_same<TPrecision, float>::value,
                                    cufftComplex, cufftDoubleComplex>::type
      cufft_Complex;
  typedef typename std::conditional<std::is_same<TPrecision, float>::value,
                                    cufftReal, cufftDoubleReal>::type cufft_Real;
  bool floatPrecision = false;
  if (std::is_same<TPrecision, float>::value) {
    floatPrecision = true;
  }

  int image_width = width;
  int image_height = height;

  int matsize = image_width * image_height;
  int matsize2 = image_height * image_width;
  int intermediateHeight = image_height / 2 + 1;
  int intermediateSize = image_width * intermediateHeight;

  // setup buffers
  TPrecision *kernel = (TPrecision *)malloc(sizeof(TPrecision) * matsize2);

  memset(kernel, 0, sizeof(TPrecision) * matsize2);

  // prepare kernel
  putKernel(k, kernel, k_w, k_h, image_width, image_height);

  // Allocate device memory
  TPrecision *d_in, *d_kernel, *d_out;
  cufft_Complex *d_image_fft, *d_kernel_fft;
  cudaMalloc(reinterpret_cast<void **>(&d_in), sizeof(TPrecision) * matsize2);
  cudaMalloc(reinterpret_cast<void **>(&d_kernel), sizeof(TPrecision) * matsize2);
  cudaMalloc(reinterpret_cast<void **>(&d_image_fft),
             sizeof(cufft_Complex) * intermediateSize);
  cudaMalloc(reinterpret_cast<void **>(&d_kernel_fft),
             sizeof(cufft_Complex) * intermediateSize);
  cudaMalloc(reinterpret_cast<void **>(&d_out), sizeof(TPrecision) * matsize2);
  // Copy host memory to device
  cudaMemcpy(d_in, in, sizeof(TPrecision) * matsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel, sizeof(TPrecision) * matsize,
             cudaMemcpyHostToDevice);

  // create plans
  cufftHandle plan_forward_many, plan_reverse_many, plan_forward_kernel;
  int n[2] = {image_height, image_width};
  if (floatPrecision) {
    cufftPlanMany(&plan_forward_many, 2, n, nullptr, 1, 1, nullptr, 1, 1,
                  CUFFT_R2C, 1);
    cufftPlanMany(&plan_reverse_many, 2, n, nullptr, 1, 1, nullptr, 1, 1,
                  CUFFT_C2R, 1);
  } else {
    cufftPlanMany(&plan_forward_many, 2, n, nullptr, 1, 1, nullptr, 1, 1,
                  CUFFT_D2Z, 1);
    cufftPlanMany(&plan_reverse_many, 2, n, nullptr, 1, 1, nullptr, 1, 1,
                  CUFFT_Z2D, 1);
  }

  cudaDeviceSynchronize(); // only for time measure

  // start
  auto start_time = std::chrono::system_clock::now();

  // Transform signal and kernel
  if (floatPrecision) {
    cufftExecR2C(plan_forward_many, reinterpret_cast<cufftReal *>(d_in),
                 reinterpret_cast<cufftComplex *>(d_image_fft));
    cufftExecR2C(plan_forward_many, reinterpret_cast<cufftReal *>(d_kernel),
                 reinterpret_cast<cufftComplex *>(d_kernel_fft));
  } else {
    cufftExecD2Z(plan_forward_many, reinterpret_cast<cufftDoubleReal *>(d_in),
                 reinterpret_cast<cufftDoubleComplex *>(d_image_fft));
    cufftExecD2Z(plan_forward_many, reinterpret_cast<cufftDoubleReal *>(d_kernel),
                 reinterpret_cast<cufftDoubleComplex *>(d_kernel_fft));
  }

  // define dimensions
  dim3 threadsPerBlock(32, 32); // 1024 threads
  dim3 numBlocks(image_width / threadsPerBlock.x,
                 image_height / threadsPerBlock.y);

  // Pointwise Multiplication in Frequency Domain
  MulScalePw<<<numBlocks, threadsPerBlock>>>(
      reinterpret_cast<cufft_Complex *>(d_image_fft),
      reinterpret_cast<cufft_Complex *>(d_kernel_fft), image_width,
      intermediateHeight, 1.0 / (image_height * image_width));
  // HANDLE_ERROR(cudaGetLastError());

  if (floatPrecision) {
    cufftExecC2R(plan_reverse_many, reinterpret_cast<cufftComplex *>(d_image_fft),
                 reinterpret_cast<cufftReal *>(d_out));
  } else {
    cufftExecZ2D(plan_reverse_many,
                 reinterpret_cast<cufftDoubleComplex *>(d_image_fft),
                 reinterpret_cast<cufftDoubleReal *>(d_out));
  }

  // Copy device memory to host
  cudaMemcpy(out, d_out, sizeof(TPrecision) * matsize, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize(); // only for time measure

  // stop
  auto end_time = std::chrono::system_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  std::cout << (float)elapsed_time.count() / 1000.0 << " ms" << std::endl;

  // cleanup
  cufftDestroy(plan_forward_many);
  cufftDestroy(plan_reverse_many);
  free(kernel);
  cudaFree(d_in);
  cudaFree(d_kernel);
  cudaFree(d_image_fft);
  cudaFree(d_kernel_fft);
  cudaFree(d_out);
}

template <class TPrecision, class T, class U, class V, size_t rows, size_t cols>
void cufftConvolution(T *in, int width, int height, const V (&kernel)[rows][cols],
                      int k_w, int k_h, U *out, int alignment, bool linear) {
  assert(rows == k_h && cols == k_w);

  int width_in = paddedWidth<T>(width, alignment);
  int width_out = paddedWidth<U>(width, alignment);

  // prepare input buffer
  TPrecision *input = new TPrecision[width * height];
  T *input_d = new T[width_in * height];
  // convert input
  cudaMemcpy(input_d, in, sizeof(T) * width_in * height, cudaMemcpyDeviceToHost);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      input[y * width + x] = (TPrecision)input_d[y * width_in + x];
    }
  }
  // prepare output buffer
  TPrecision *output = new TPrecision[width * height];
  fftConvolve(input, width, height, (V *)(&kernel[0][0]), k_w, k_h, output);
  // convert output
  U *out_d = new U[width_out * height];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      out_d[y * width_out + x] = (U)output[y * width + x];
    }
  }
  cudaMemcpy(out, out_d, sizeof(U) * width_out * height, cudaMemcpyHostToDevice);
}