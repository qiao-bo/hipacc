#include <cufft.h>
#include <cufftXt.h>

#include "hipacc_fft_helper.hpp"

#include <iostream>

// Complex data type
struct f2 {
  double x;
  double y;
};
typedef f2 Complex;
// typedef float2 Complex;

__device__ cufftComplex C_Mul(cufftComplex a, cufftComplex b) {
  cufftComplex temp;
  temp.x = a.x * b.x - a.y * b.y;
  temp.y = a.y * b.x + b.y * a.x;
  return temp;
}
// Mulps
__global__ void Mulps(cufftComplex *image, cufftComplex *filter, int Nx, int Ny) {
  int tix = threadIdx.x + blockIdx.x * blockDim.x;
  int tiy = threadIdx.y + blockIdx.y * blockDim.y;
  int tid = tix + Nx * tiy;

  if ((tix < Nx) && (tiy < Ny)) {
    image[tid] = C_Mul(image[tid], filter[tid]);
  }
}
// Double
__device__ cufftDoubleComplex C_Mul(cufftDoubleComplex a, cufftDoubleComplex b) {
  cufftDoubleComplex temp;
  temp.x = a.x * b.x - a.y * b.y;
  temp.y = a.y * b.x + b.y * a.x;
  return temp;
}
// Mulps
__global__ void Mulps(cufftDoubleComplex *image, cufftDoubleComplex *filter,
                      int Nx, int Ny) {
  int tix = threadIdx.x + blockIdx.x * blockDim.x;
  int tiy = threadIdx.y + blockIdx.y * blockDim.y;
  int tid = tix + Nx * tiy;

  if (tid >= Nx * Ny)
    return;

  if ((tix < Nx) && (tiy < Ny)) {
    image[tid] = C_Mul(image[tid], filter[tid]);
  }
}
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

void fftConvolve(double *in, int width, int height, float *k, int k_w, int k_h,
                 double *out) {
  int image_width = width;
  int image_height = height;

  int matsize = image_width * image_height;
  int intermediateHeight = image_height / 2 + 1;
  int intermediateSize = image_width * intermediateHeight;

  // setup buffers
  double *kernel = (double *)malloc(sizeof(double) * matsize);

  memset(kernel, 0, sizeof(double) * matsize);

  // prepare kernel
  putKernel((float *)k, kernel, k_w, k_h, image_width, image_height);

  // Allocate device memory
  double *d_in, *d_kernel, *d_out;
  Complex *d_image_fft, *d_kernel_fft;
  cudaMalloc(reinterpret_cast<void **>(&d_in), sizeof(double) * matsize);
  cudaMalloc(reinterpret_cast<void **>(&d_kernel), sizeof(double) * matsize);
  cudaMalloc(reinterpret_cast<void **>(&d_image_fft),
             sizeof(Complex) * intermediateSize);
  cudaMalloc(reinterpret_cast<void **>(&d_kernel_fft),
             sizeof(Complex) * intermediateSize);
  cudaMalloc(reinterpret_cast<void **>(&d_out), sizeof(double) * matsize);
  // Copy host memory to device
  cudaMemcpy(d_in, in, sizeof(double) * matsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel, sizeof(double) * matsize, cudaMemcpyHostToDevice);

  // create plans
  cufftHandle plan_forward_many, plan_reverse_many;
  int n[2] = {image_height, image_width};
  cufftPlanMany(&plan_forward_many, 2, n, nullptr, 1, 1, nullptr, 1, 1, CUFFT_D2Z,
                1);
  cufftPlanMany(&plan_reverse_many, 2, n, nullptr, 1, 1, nullptr, 1, 1, CUFFT_Z2D,
                1);

  cudaDeviceSynchronize(); // only for time measure

  // start
  auto start_time = std::chrono::system_clock::now();

  // Transform signal and kernel
  cufftExecD2Z(plan_forward_many, reinterpret_cast<cufftDoubleReal *>(d_in),
               reinterpret_cast<cufftDoubleComplex *>(d_image_fft));
  cufftExecD2Z(plan_forward_many, reinterpret_cast<cufftDoubleReal *>(d_kernel),
               reinterpret_cast<cufftDoubleComplex *>(d_kernel_fft));

  // define dimensions
  dim3 threadsPerBlock(32, 32); // 1024 threads
  dim3 numBlocks(image_width / threadsPerBlock.x,
                 image_height / threadsPerBlock.y);

  // Pointwise Multiplication in Frequency Domain
  MulScalePw<<<numBlocks, threadsPerBlock>>>(
      reinterpret_cast<cufftDoubleComplex *>(d_image_fft),
      reinterpret_cast<cufftDoubleComplex *>(d_kernel_fft), image_width,
      intermediateHeight, 1.0 / (image_height * image_width));
  // HANDLE_ERROR(cudaGetLastError());

  cufftExecZ2D(plan_reverse_many,
               reinterpret_cast<cufftDoubleComplex *>(d_image_fft),
               reinterpret_cast<cufftDoubleReal *>(d_out));

  // Copy device memory to host
  cudaMemcpy(out, d_out, sizeof(double) * matsize, cudaMemcpyDeviceToHost);

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

void cufftConvolution(double *in, int width, int height, float *kernel, int k_w,
                      int k_h, double *out) {
  fftConvolve(in, width, height, kernel, k_w, k_h, out);
}