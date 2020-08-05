#include <cuComplex.h>
#include <cufft.h>
#include <cufftXt.h>

#include "cuda_complex.hpp"
#include "hipacc_base_standalone.hpp"
#include "hipacc_fft_helper.hpp"

#include <iostream>
#include <complex>

#define gpuErrchk(ans)                                                           \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

void Handle_Error(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    std::string cu = std::string(cudaGetErrorString(err)) + " in " +
                     std::string(file) + " at line: " + std::to_string(line) +
                     "\n";
    std::cout << cu;
  }
}
#define HANDLE_ERROR(err) (Handle_Error(err, __FILE__, __LINE__))

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
template <class C> __device__ C C_Scale(C a, float s) {
  C c;
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

// scale
template <class C>
__global__ void ScalePw(C *image, int Nx, int Ny, double scale) {
  int tix = threadIdx.x + blockIdx.x * blockDim.x;
  int tiy = threadIdx.y + blockIdx.y * blockDim.y;
  int tid = tix + Nx * tiy;

  if (tid >= Nx * Ny)
    return;

  if ((tix < Nx) && (tiy < Ny)) {
    image[tid] = C_Scale(image[tid], scale);
  }
}

// scale element-wise
template <class T, class TScale>
__global__ void ScaleE(T *image, int Nx, int Ny, TScale scale) {
  int tix = threadIdx.x + blockIdx.x * blockDim.x;
  int tiy = threadIdx.y + blockIdx.y * blockDim.y;
  int tid = tix + Nx * tiy;

  if (tid >= Nx * Ny)
    return;

  if ((tix < Nx) && (tiy < Ny)) {
    image[tid] = image[tid] * scale;
  }
}

// Makhoul reorder
template <class T, class U>
__global__ void Reorder(T *image, U *out, int width, int height, int width_in,
                        int blockX, int blockY) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int tid = x + width_in * y;

  if (tid >= width_in * height)
    return;

  if ((x < width) && (y < height)) {
    int dstX;
    if (x % 2 == 0) {
      dstX = x / 2;
    } else {
      dstX = (width - (x / 2) - 1);
    }
    int dstY;
    if (y % 2 == 0) {
      dstY = y / 2;
    } else {
      dstY = (width - (y / 2) - 1);
    }
    out[dstY * width + dstX] = (U)image[x + width_in * y];
  }
}

// Makhoul inverse reorder
template <class T, class U>
__global__ void iReorder(T *in, U *image, int width, int height, int width_out,
                         int blockX, int blockY) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int tid = x + width_out * y;

  if (tid >= width_out * height)
    return;

  if ((x < width) && (y < height)) {
    int dstX;
    if (x % 2 == 0) {
      dstX = x / 2;
    } else {
      dstX = (width - (x / 2) - 1);
    }
    int dstY;
    if (y % 2 == 0) {
      dstY = y / 2;
    } else {
      dstY = (width - (y / 2) - 1);
    }
    image[x + width_out * y] = (T)in[dstY * width + dstX];
  }
}

// Makhoul factor multiplication
template <class T>
__global__ void Factoring(complex<T> *in, T *out, int width, int height,
                          int blockX, int blockY, int out_width) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int tid = x + width * y;

  if (tid >= width * height)
    return;

  const complex<T> i(0.0, 1.0);
  complex<T> factor = exp(-i * complex<T>(M_PI * y / (2 * height)));
  complex<T> factor2 = exp(-i * complex<T>(M_PI * x / (2 * width)));
  complex<T> factor3 = exp(-i * complex<T>(M_PI * (-x) / (2 * width)));
  complex<T> factor4 = exp(-i * complex<T>(M_PI * (-y) / (2 * height)));

  if ((x < width / 2 + 1) && (y < height)) {
    if (x == 0 && y == 0) {
      out[y * width + x] =
          4.0 * (factor * (factor2 * in[y * out_width + x])).real();
    } else if (x == 0) {
      out[y * width + x] =
          2.0 * (factor2 * (factor * in[y * out_width + x] +
                            factor4 * in[(height - y) * out_width + x]))
                    .real();
    } else if (y == 0) {
      out[y * width + x] =
          2.0 * (factor * (factor2 * in[y * out_width + x] +
                           factor3 * conj(in[y * out_width + x])))
                    .real();
    } else {
      out[y * width + x] =
          2.0 * (factor * (factor2 * in[y * out_width + x] +
                           factor3 * conj(in[(height - y) * out_width + x])))
                    .real();
    }
  } else if ((x < width) && (y < height)) {
    if (x == 0 && y == 0) {
      out[y * width + x] =
          4.0 * (factor * (factor2 * conj(in[y * out_width + width - x]))).real();
    } else if (x == 0) {
      out[y * width + x] =
          2.0 *
          (factor2 * (factor * conj(in[(height - y) * out_width + width - x]) +
                      factor4 * conj(in[y * out_width + width - x])))
              .real();
    } else if (y == 0) {
      out[y * width + x] =
          2.0 * (factor * (factor2 * conj(in[y * out_width + width - x]) +
                           factor3 * in[y * out_width + width - x]))
                    .real();
    } else {
      out[y * width + x] =
          2.0 *
          (factor * (factor2 * conj(in[(height - y) * out_width + width - x]) +
                     factor3 * in[y * out_width + width - x]))
              .real();
    }
  }
}

// Makhoul inverse factor multiplication
template <class T>
__global__ void iFactoring(T *in, complex<T> *out, int width, int height,
                           int blockX, int blockY) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int tid = x + width * y;

  if (tid >= width * height)
    return;

  int in_width = width / 2 + 1;

  const complex<T> i(0.0, 1.0);

  if ((x < in_width) && (y < height)) {
    complex<T> factor3 = exp(-i * complex<T>(M_PI * (-x) / (2 * width)));
    complex<T> factor4 = exp(-i * complex<T>(M_PI * (-y) / (2 * height)));
    if (x == 0 && y == 0) {
      out[y * in_width + x] = complex<T>(0.25, 0.0) * factor4 * factor3 *
                              complex<T>(in[y * width + x]);
    } else if (y == 0) {
      out[y * in_width + x] = complex<T>(0.25, 0.0) * factor4 * factor3 *
                              (complex<T>(in[y * width + x]) -
                               i * complex<T>(in[y * width + width - x]));
    } else if (x == 0) {
      out[y * in_width + x] = complex<T>(0.25, 0.0) * factor4 * factor3 *
                              (complex<T>(in[y * width + x]) -
                               i * complex<T>(in[(height - y) * width + x]));
    } else {
      out[y * in_width + x] =
          complex<T>(0.25, 0.0) * factor4 * factor3 *
          ((complex<T>(in[y * width + x]) -
            complex<T>(in[(height - y) * width + width - x])) -
           i * (complex<T>(in[(height - y) * width + x]) +
                complex<T>(in[y * width + width - x])));
    }
  }
}

template <class T, class U>
__global__ void CopyAligned(T *in, U *out, int width, int height, int width_in,
                            int width_out) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int tid = x + width * y;

  if (tid >= width_in * height)
    return;

  if ((x < width) && (y < height)) {
    out[y * width_out + x] = (U)in[x + width_in * y];
  }
}

template <class T, class U>
__global__ void FillMem(T *mem, U val, int width, int height) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int tid = x + width * y;

  if (tid >= width * height)
    return;

  if ((x < width) && (y < height)) {
    mem[y * width + x] = (T)val;
  }
}

template <class T, class U>
__global__ void PutKernel(T *kernel, U *dest, int kw, int kh, int nx, int ny) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int tid = x + kw * y;

  if (tid >= kw * kh)
    return;

  if ((x < kw) && (y < kh)) {
    int kx = x - kw / 2;
    int ky = y - kh / 2;

    dest[nx * ((ny + ky) % ny) + ((nx + kx) % nx)] =
        (U)kernel[kw * (kh / 2 + ky) + (kw / 2 + kx)];
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
  int intermediateWidth = image_width / 2 + 1;
  int intermediateSize = image_height * intermediateWidth;

  // setup buffers
  TPrecision *kernel = (TPrecision *)malloc(sizeof(TPrecision) * matsize);

  memset(kernel, 0, sizeof(TPrecision) * matsize);

  // prepare kernel
  putKernel(k, kernel, k_w, k_h, image_width, image_height);

  // Allocate device memory
  TPrecision *d_in, *d_kernel, *d_out;
  cufft_Complex *d_image_fft, *d_kernel_fft;
  cudaMalloc(reinterpret_cast<void **>(&d_in), sizeof(TPrecision) * matsize);
  cudaMalloc(reinterpret_cast<void **>(&d_kernel), sizeof(TPrecision) * matsize);
  cudaMalloc(reinterpret_cast<void **>(&d_image_fft),
             sizeof(cufft_Complex) * intermediateSize);
  cudaMalloc(reinterpret_cast<void **>(&d_kernel_fft),
             sizeof(cufft_Complex) * intermediateSize);
  cudaMalloc(reinterpret_cast<void **>(&d_out), sizeof(TPrecision) * matsize);
  // Copy host memory to device
  cudaMemcpy(d_in, in, sizeof(TPrecision) * matsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel, sizeof(TPrecision) * matsize,
             cudaMemcpyHostToDevice);

  cudaDeviceSynchronize(); // only for time measure
  // start
  auto start_time = std::chrono::system_clock::now();

  // create plans
  cufftHandle plan_forward_many, plan_reverse_many;
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
  dim3 numBlocks(image_width / threadsPerBlock.x + 1,
                 image_height / threadsPerBlock.y + 1);

  // Pointwise Multiplication in Frequency Domain
  MulScalePw<<<numBlocks, threadsPerBlock>>>(
      reinterpret_cast<cufft_Complex *>(d_image_fft),
      reinterpret_cast<cufft_Complex *>(d_kernel_fft), intermediateWidth,
      image_height, 1.0 / (image_height * image_width));
  HANDLE_ERROR(cudaGetLastError());

  if (floatPrecision) {
    cufftExecC2R(plan_reverse_many, reinterpret_cast<cufftComplex *>(d_image_fft),
                 reinterpret_cast<cufftReal *>(d_out));
  } else {
    cufftExecZ2D(plan_reverse_many,
                 reinterpret_cast<cufftDoubleComplex *>(d_image_fft),
                 reinterpret_cast<cufftDoubleReal *>(d_out));
  }

  cudaDeviceSynchronize(); // only for time measure
  // stop
  auto end_time = std::chrono::system_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  std::cout << (float)elapsed_time.count() / 1000.0 << " ms" << std::endl;

  // Copy device memory to host
  cudaMemcpy(out, d_out, sizeof(TPrecision) * matsize, cudaMemcpyDeviceToHost);

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

template <class TPrecision>
void fftConvolve_device(TPrecision *in, int width, int height, TPrecision *k,
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

  std::cout << width << " x " << height << std::endl;

  int intermediateWidth = image_width / 2 + 1;
  int intermediateSize = image_height * intermediateWidth;

  // Allocate device memory
  cufft_Complex *d_image_fft, *d_kernel_fft;
  cudaMalloc(reinterpret_cast<void **>(&d_image_fft),
             sizeof(cufft_Complex) * intermediateSize);
  cudaMalloc(reinterpret_cast<void **>(&d_kernel_fft),
             sizeof(cufft_Complex) * intermediateSize);

  // create plans
  cufftHandle plan_forward_many, plan_reverse_many;
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
    cufftExecR2C(plan_forward_many, reinterpret_cast<cufftReal *>(in),
                 reinterpret_cast<cufftComplex *>(d_image_fft));
    cufftExecR2C(plan_forward_many, reinterpret_cast<cufftReal *>(k),
                 reinterpret_cast<cufftComplex *>(d_kernel_fft));
  } else {
    cufftExecD2Z(plan_forward_many, reinterpret_cast<cufftDoubleReal *>(in),
                 reinterpret_cast<cufftDoubleComplex *>(d_image_fft));
    cufftExecD2Z(plan_forward_many, reinterpret_cast<cufftDoubleReal *>(k),
                 reinterpret_cast<cufftDoubleComplex *>(d_kernel_fft));
  }

  // define dimensions
  dim3 threadsPerBlock(32, 32); // 1024 threads
  dim3 numBlocks(image_width / threadsPerBlock.x + 1,
                 image_height / threadsPerBlock.y + 1);

  // Pointwise Multiplication in Frequency Domain
  MulScalePw<<<numBlocks, threadsPerBlock>>>(
      reinterpret_cast<cufft_Complex *>(d_image_fft),
      reinterpret_cast<cufft_Complex *>(d_kernel_fft), intermediateWidth,
      image_height, 1.0 / (image_height * image_width));
  HANDLE_ERROR(cudaGetLastError());

  if (floatPrecision) {
    cufftExecC2R(plan_reverse_many, reinterpret_cast<cufftComplex *>(d_image_fft),
                 reinterpret_cast<cufftReal *>(out));
  } else {
    cufftExecZ2D(plan_reverse_many,
                 reinterpret_cast<cufftDoubleComplex *>(d_image_fft),
                 reinterpret_cast<cufftDoubleReal *>(out));
  }

  cudaDeviceSynchronize(); // only for time measure
  // stop
  auto end_time = std::chrono::system_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  std::cout << (float)elapsed_time.count() / 1000.0 << " ms" << std::endl;

  // cleanup
  cufftDestroy(plan_forward_many);
  cufftDestroy(plan_reverse_many);
  cudaFree(d_image_fft);
  cudaFree(d_kernel_fft);
}

bool estimateConvolutionExecutionTime(int width, int height, int padWidth,
                                      int padHeight, int k_w, int k_h,
                                      bool linear) {
  // assumed:
  // 2496 CUDA cores
  // 3.52 TFLOPS single, 1.17 TFLOPS double
  const double FLOPMS = 3500000000; // FLOP per MS

  double N1 = padWidth;
  double N2 = padHeight;

  double singleFFTcost = 8.0 * N1 * N2 * log2(N1 * N2);

  double FFTConvolutionCost = (singleFFTcost * 3 + (N1 * N2 * (3 + 1))) * 80;
  std::cout << "FFTConvolutionCost: " << FFTConvolutionCost / FLOPMS << std::endl;

  double simpleHipaccCost = (height * width) * (k_w * k_h) * 2 * 22;
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
void cufftConvolution(T *in, int width, int height, const V (&kernel)[rows][cols],
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

  /*
  estimateConvolutionExecutionTime(width, height, padWidth, padHeight, k_w, k_h,
                                   linear);
  */

  // prepare input buffer
  TPrecision *input = new TPrecision[padSize];
  if (linear) {
    std::fill_n(input, padSize, (TPrecision)boundaryConstant);
  }
  T *input_d = new T[width_in * height];
  // convert input
  gpuErrchk(cudaMemcpy(input_d, in, sizeof(T) * width_in * height,
                       cudaMemcpyDeviceToHost));
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      input[y * padWidth + x] = (TPrecision)input_d[y * width_in + x];
    }
  }
  // prepare output buffer
  TPrecision *output = new TPrecision[padSize];
  fftConvolve(input, padWidth, padHeight, (V *)(&kernel[0][0]), k_w, k_h, output);
  // convert output
  U *out_d = new U[width_out * height];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      out_d[y * width_out + x] = (U)output[y * padWidth + x];
    }
  }
  cudaMemcpy(out, out_d, sizeof(U) * width_out * height, cudaMemcpyHostToDevice);

  free(input);
  free(input_d);
  free(output);
  free(out_d);
}

template <class TPrecision, class T, class U, class V, size_t rows, size_t cols,
          class B = int>
void cufftConvolution_device(T *in, int width, int height,
                             const V (&kernel)[rows][cols], int k_w, int k_h,
                             U *out, int alignment, bool linear,
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

  /*
  estimateConvolutionExecutionTime(width, height, padWidth, padHeight, k_w, k_h,
                                   linear);
  */

  V *kernel_in;
  cudaMalloc(reinterpret_cast<void **>(&kernel_in), sizeof(V) * k_w * k_h);
  gpuErrchk(cudaMemcpy(kernel_in, (V *)(&kernel[0][0]), sizeof(V) * k_w * k_h,
                       cudaMemcpyHostToDevice));

  TPrecision *d_in, *d_kernel, *d_out;
  cudaMalloc(reinterpret_cast<void **>(&d_in), sizeof(TPrecision) * padSize);
  cudaMalloc(reinterpret_cast<void **>(&d_kernel), sizeof(TPrecision) * padSize);
  cudaMalloc(reinterpret_cast<void **>(&d_out), sizeof(TPrecision) * padSize);

  dim3 threadsPerBlock(32, 32); // 1024 threads
  dim3 numBlocks(width / threadsPerBlock.x + 1, height / threadsPerBlock.y + 1);
  dim3 threadsPerBlock2(32, 32); // 1024 threads
  dim3 numBlocks2(padWidth / threadsPerBlock.x + 1,
                  padHeight / threadsPerBlock.y + 1);

  // prepare image buffer
  if (linear) {
    FillMem<<<numBlocks2, threadsPerBlock2>>>(d_in, (TPrecision)boundaryConstant,
                                              padWidth, padHeight);
    HANDLE_ERROR(cudaGetLastError());
  }
  CopyAligned<<<numBlocks, threadsPerBlock>>>(in, d_in, width, height, width_in,
                                              padWidth);
  HANDLE_ERROR(cudaGetLastError());

  // prepare kernel buffer
  cudaMemset(d_kernel, 0, sizeof(TPrecision) * padSize);
  PutKernel<<<numBlocks, threadsPerBlock>>>(kernel_in, d_kernel, k_w, k_h,
                                            padWidth, padHeight);

  // prepare output buffer
  // TPrecision *output = new TPrecision[padSize];
  fftConvolve_device(d_in, padWidth, padHeight, d_kernel, d_out);

  // convert output

  CopyAligned<<<numBlocks, threadsPerBlock>>>(d_out, out, width, height, padWidth,
                                              width_out);
  HANDLE_ERROR(cudaGetLastError());

  // clean up
  cudaFree(d_in);
  cudaFree(d_kernel);
  cudaFree(d_out);
}

template <class TPrecision>
void fftTransform(TPrecision *in, int width, int height, TPrecision *out,
                  bool forward = true, bool scale = false) {
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
  int intermediateWidth = image_width / 2 + 1;
  int intermediateSize = image_height * intermediateWidth;

  // Allocate device memory
  cufft_Real *d_in;
  cufft_Complex *d_image_fft;
  gpuErrchk(
      cudaMalloc(reinterpret_cast<void **>(&d_in), sizeof(cufft_Real) * matsize));
  gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_image_fft),
                       sizeof(cufft_Complex) * intermediateSize));
  // Copy host memory to device
  if (forward) {
    gpuErrchk(cudaMemcpy(d_in, in, sizeof(cufft_Real) * matsize,
                         cudaMemcpyHostToDevice));
  } else {
    gpuErrchk(cudaMemcpy(d_image_fft, in,
                         sizeof(cufft_Complex) * intermediateSize,
                         cudaMemcpyHostToDevice));
  }
  // create plans
  cufftHandle plan_forward_many;
  int n[2] = {image_height, image_width};
  if (floatPrecision) {
    if (forward) {
      cufftPlanMany(&plan_forward_many, 2, n, nullptr, 1, 1, nullptr, 1, 1,
                    CUFFT_R2C, 1);
    } else {
      cufftPlanMany(&plan_forward_many, 2, n, nullptr, 1, 1, nullptr, 1, 1,
                    CUFFT_C2R, 1);
    }
  } else {
    if (forward) {
      cufftPlanMany(&plan_forward_many, 2, n, nullptr, 1, 1, nullptr, 1, 1,
                    CUFFT_D2Z, 1);
    } else {
      cufftPlanMany(&plan_forward_many, 2, n, nullptr, 1, 1, nullptr, 1, 1,
                    CUFFT_Z2D, 1);
    }
  }

  // define dimensions
  dim3 threadsPerBlock(32, 32); // 1024 threads
  dim3 numBlocks(image_width / threadsPerBlock.x + 1,
                 image_height / threadsPerBlock.y + 1);

  // Transform signal
  if (forward) {
    if (floatPrecision) {
      cufftExecR2C(plan_forward_many, reinterpret_cast<cufftReal *>(d_in),
                   reinterpret_cast<cufftComplex *>(d_image_fft));
    } else {
      cufftExecD2Z(plan_forward_many, reinterpret_cast<cufftDoubleReal *>(d_in),
                   reinterpret_cast<cufftDoubleComplex *>(d_image_fft));
    }
  } else {
    if (floatPrecision) {
      cufftExecC2R(plan_forward_many,
                   reinterpret_cast<cufftComplex *>(d_image_fft),
                   reinterpret_cast<cufftReal *>(d_in));
    } else {
      cufftExecZ2D(plan_forward_many,
                   reinterpret_cast<cufftDoubleComplex *>(d_image_fft),
                   reinterpret_cast<cufftDoubleReal *>(d_in));
    }
  }

  if (!forward && scale) {
    // Pointwise Scale
    ScaleE<<<numBlocks, threadsPerBlock>>>(
        reinterpret_cast<cufft_Real *>(d_in), image_width,
        image_height, 1.0 / (image_height * image_width));
    HANDLE_ERROR(cudaGetLastError());
  }

  // Copy device memory to host
  if (forward) {
    gpuErrchk(cudaMemcpy(out, d_image_fft,
                         sizeof(cufft_Complex) * intermediateSize,
                         cudaMemcpyDeviceToHost));
  } else {
    gpuErrchk(cudaMemcpy(out, d_in, sizeof(cufft_Real) * matsize,
                         cudaMemcpyDeviceToHost));
  }

  // cleanup
  cufftDestroy(plan_forward_many);
  cudaFree(d_in);
  cudaFree(d_image_fft);
}

template <class TPrecision>
void fftTransformDevice(TPrecision *in, int width, int height, TPrecision *out,
                        bool forward = true, bool scale = false,
                        int alignment = 0) {
  if (alignment == 0)
    alignment = width;
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

  // int matsize = image_width * image_height;

  // create plans
  cufftHandle plan_forward_many;
  int n[2] = {image_height, image_width};
  int inembed[2] = {height, alignment};
  int onembed[2] = {height, width / 2 + 1};
  int idist = alignment * height;
  int odist = (width / 2 + 1) * height;

  if (floatPrecision) {
    if (forward) {
      cufftPlanMany(&plan_forward_many, 2, n, inembed, 1, idist, onembed, 1,
                    odist, CUFFT_R2C, 1);
    } else {
      cufftPlanMany(&plan_forward_many, 2, n, onembed, 1, odist, inembed, 1,
                    idist, CUFFT_C2R, 1);
    }
  } else {
    if (forward) {
      cufftPlanMany(&plan_forward_many, 2, n, inembed, 1, idist, onembed, 1,
                    odist, CUFFT_D2Z, 1);
    } else {
      cufftPlanMany(&plan_forward_many, 2, n, onembed, 1, odist, inembed, 1,
                    idist, CUFFT_Z2D, 1);
    }
  }

  // define dimensions
  dim3 threadsPerBlock(32, 32); // 1024 threads
  dim3 numBlocks(alignment / threadsPerBlock.x + 1,
                 image_height / threadsPerBlock.y + 1);

  // Transform signal
  if (forward) {
    if (floatPrecision) {
      cufftExecR2C(plan_forward_many, reinterpret_cast<cufftReal *>(in),
                   reinterpret_cast<cufftComplex *>(out));
    } else {
      cufftExecD2Z(plan_forward_many, reinterpret_cast<cufftDoubleReal *>(in),
                   reinterpret_cast<cufftDoubleComplex *>(out));
    }
  } else {
    if (floatPrecision) {
      cufftExecC2R(plan_forward_many,
                   reinterpret_cast<cufftComplex *>(in),
                   reinterpret_cast<cufftReal *>(out));
    } else {
      cufftExecZ2D(plan_forward_many,
                   reinterpret_cast<cufftDoubleComplex *>(in),
                   reinterpret_cast<cufftDoubleReal *>(out));
    }
  }

  if (!forward && scale) {
    // Pointwise Scale
    ScaleE<<<numBlocks, threadsPerBlock>>>(reinterpret_cast<cufft_Real *>(out),
                                           alignment, image_height,
                                           1.0 / (image_height * image_width));
    HANDLE_ERROR(cudaGetLastError());
  }

  // cleanup
  cufftDestroy(plan_forward_many);
}

// old fft part without device functions
/*
// forward FFT
template <class T, class TPrecision> TPrecision *fft(HipaccImage &in) {
  int width = in->width;
  int height = in->height;
  int width_in = alignedWidth<T>(width, in->alignment);

  // prepare input buffer
  TPrecision *input = new TPrecision[width * height];
  T *input_d = new T[width_in * height];
  // convert input
  gpuErrchk(cudaMemcpy(input_d, in->mem, sizeof(T) * width_in * height,
                       cudaMemcpyDeviceToHost));
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      input[y * width + x] = (TPrecision)input_d[y * width_in + x];
    }
  }

  // prepare output buffer
  TPrecision *output = new TPrecision[2 * (width / 2 + 1) * height];
  fftTransform(input, width, height, output, true);

  free(input);
  free(input_d);

  return output;
}

// inverse FFT
template <class T, class TPrecision> void ifft(TPrecision *in, HipaccImage &out) {
  int width = out->width;
  int height = out->height;
  int width_out = alignedWidth<T>(width, out->alignment);

  // prepare output buffer
  TPrecision *output = new TPrecision[width * height];

  fftTransform(in, width, height, output, false, true);

  // truncate values outside of range 0-255
  T *out_d = new T[width_out * height];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      T val = output[y * width + x];
      if (output[y * width + x] < 0) {
        val = 0;
      } else if (output[y * width + x] > 255) {
        val = 255;
      }
      // convert output
      out_d[y * width_out + x] = (T)(val);
    }
  }

  gpuErrchk(cudaMemcpy(out->mem, out_d, sizeof(T) * width_out * height,
                       cudaMemcpyHostToDevice));

  free(output);
  free(out_d);
}
*/

// new fft part

// forward FFT
template <class T, class TPrecision> TPrecision *fft(HipaccImage &in) {
  int width = in->width;
  int height = in->height;
  int width_in = alignedWidth<T>(width, in->alignment);

  // prepare output buffer
  TPrecision *output;
  cudaMalloc((void **)&output, sizeof(TPrecision) * 2 * (width / 2 + 1) * height);
  TPrecision *input;
  if (std::is_same<T, TPrecision>::value) {
    input = (TPrecision *)(in->mem);
  } else {
    gpuErrchk(
        cudaMalloc((void **)&input, sizeof(TPrecision) * width_in * height));
    // define dimensions
    dim3 threadsPerBlock(32, 32); // 1024 threads
    dim3 numBlocks(width / threadsPerBlock.x + 1, height / threadsPerBlock.y + 1);

    // Pointwise Multiplication in Frequency Domain
    CopyAligned<<<numBlocks, threadsPerBlock>>>((T *)(in->mem), input, width,
                                                height, width_in, width_in);
    HANDLE_ERROR(cudaGetLastError());
  }
  fftTransformDevice(input, width, height, output, true, false, width_in);
  HANDLE_ERROR(cudaGetLastError());
  return output;
}

// inverse FFT
template <class T, class TPrecision> void ifft(TPrecision *in, HipaccImage &out) {
  int width = out->width;
  int height = out->height;
  int width_out = alignedWidth<T>(width, out->alignment);

  // prepare output buffer
  TPrecision *output;
  if (std::is_same<T, TPrecision>::value) {
    output = (TPrecision *)(out->mem);
  } else {
    gpuErrchk(
        cudaMalloc((void **)&output, sizeof(TPrecision) * width_out * height));
  }
  fftTransformDevice(in, width, height, output, false, true, width_out);

  if (!std::is_same<T, TPrecision>::value) {
    // define dimensions
    dim3 threadsPerBlock(32, 32); // 1024 threads
    dim3 numBlocks(width / threadsPerBlock.x + 1, height / threadsPerBlock.y + 1);

    // Pointwise Multiplication in Frequency Domain
    CopyAligned<<<numBlocks, threadsPerBlock>>>(output, (T *)(out->mem), width,
                                                height, width_out, width_out);
    HANDLE_ERROR(cudaGetLastError());
  }
}

// create magnitude from fft
template <class T, class TPrecision>
void fftToMagnitude(TPrecision *in, HipaccImage &mag) {
  int width = mag->width;
  int height = mag->height;
  int width_out = alignedWidth<T>(width, mag->alignment);

  T *tmp = new T[width * height];
  std::complex<TPrecision> *input =
      new std::complex<TPrecision>[(width / 2 + 1) * height];
  gpuErrchk(cudaMemcpy(
      input, in, sizeof(std::complex<TPrecision>) * (width / 2 + 1) * height,
      cudaMemcpyDeviceToHost));
  calcMagnitude(reinterpret_cast<std::complex<TPrecision> *>(input), tmp, width,
                height);

  shiftFFT(tmp, width, height);

  T *out = new T[width_out * height];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      out[y * width_out + x] = (T)tmp[y * width + x];
    }
  }

  gpuErrchk(cudaMemcpy(mag->mem, out, sizeof(T) * width_out * height,
                       cudaMemcpyHostToDevice));
  free(tmp);
  free(out);
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

  T *out = new T[width_out * height];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      out[y * width_out + x] = (T)tmp[y * width + x];
    }
  }

  gpuErrchk(cudaMemcpy(mag->mem, out, sizeof(T) * width_out * height,
                       cudaMemcpyHostToDevice));
  free(tmp);
  free(out);
}

// apply mask mag to result of FFT in
template <class T, class TPrecision>
void fftScaleMagnitude(TPrecision *in, HipaccImage &mag) {
  int width = mag->width;
  int height = mag->height;
  int width_in = alignedWidth<T>(width, mag->alignment);
  T *scale = new T[width_in * height];
  std::complex<TPrecision> *input =
      new std::complex<TPrecision>[(width / 2 + 1) * height];
  gpuErrchk(cudaMemcpy(
      input, in, sizeof(std::complex<TPrecision>) * (width / 2 + 1) * height,
      cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(scale, mag->mem, sizeof(T) * width_in * height,
                       cudaMemcpyDeviceToHost));

  iShiftFFT(scale, width, height, mag->alignment);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < (width / 2 + 1); x++) {
      input[(y * (width / 2 + 1) + x)] *=
          (TPrecision)scale[y * width_in + x] / 255;
    }
  }

  gpuErrchk(cudaMemcpy(
      in, input, sizeof(std::complex<TPrecision>) * (width / 2 + 1) * height,
      cudaMemcpyHostToDevice));

  free(scale);
}

template <class TPrecision>
void dct_forward_device(TPrecision *in, int width, int height, TPrecision *out,
                        int width_in = 0) {
  if (width_in == 0) {
    width_in = width;
  }
  // define dimensions
  dim3 threadsPerBlock(32, 32); // 1024 threads
  dim3 numBlocks(width / threadsPerBlock.x + 1, height / threadsPerBlock.y + 1);

  TPrecision *reorder;
  gpuErrchk(cudaMalloc((void **)&reorder, sizeof(TPrecision) * width * height));

  Reorder<<<numBlocks, threadsPerBlock>>>(in, reorder, width, height, width_in,
                                          width, height);
  HANDLE_ERROR(cudaGetLastError());

  // prepare output buffer
  int out_width = width / 2 + 1;
  std::complex<TPrecision> *output;
  gpuErrchk(cudaMalloc((void **)&output,
                       sizeof(std::complex<TPrecision>) * out_width * height));

  fftTransformDevice(reorder, width, height,
                     reinterpret_cast<TPrecision *>(output), true, false);

  Factoring<<<numBlocks, threadsPerBlock>>>(
      reinterpret_cast<complex<TPrecision> *>(output), out, width, height, width,
      height, out_width);
  cudaDeviceSynchronize();
  HANDLE_ERROR(cudaGetLastError());

  cudaFree(reorder);
  cudaFree(output);
}

template <class TPrecision, class T>
void dct_forward(T *in, int width, int height, TPrecision *out,
                 int width_in = 0) {
  if (width_in == 0) {
    width_in = width;
  }

  TPrecision *input = new TPrecision[width * height];

  for (int y = 0; y < height / 2; y++) {
    for (int x = 0; x < width / 2; x++) {
      input[y * width + x] = (TPrecision)in[(2 * y) * width_in + (2 * x)];
      input[y * width + (x + width / 2)] =
          (TPrecision)in[(2 * y) * width_in + (2 * (width / 2 - x - 1) + 1)];
      input[(y + height / 2) * width + x] =
          (TPrecision)in[(2 * (height / 2 - y - 1) + 1) * width_in + (2 * x)];
      input[(y + height / 2) * width + (x + width / 2)] =
          (TPrecision)in[(2 * (height / 2 - y - 1) + 1) * width_in +
                         (2 * (width / 2 - x - 1) + 1)];
    }
  }

  // prepare output buffer
  int out_width = width / 2 + 1;
  std::complex<TPrecision> *output =
      new std::complex<TPrecision>[out_width * height];

  fftTransform(input, width, height, reinterpret_cast<TPrecision *>(output),
               true);

  const std::complex<TPrecision> i(0.0, 1.0);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width / 2 + 1; x++) {
      std::complex<TPrecision> factor =
          std::exp(-i * std::complex<TPrecision>(M_PI * y / (2 * height)));
      std::complex<TPrecision> factor2 =
          std::exp(-i * std::complex<TPrecision>(M_PI * x / (2 * width)));
      std::complex<TPrecision> factor3 =
          std::exp(-i * std::complex<TPrecision>(M_PI * (-x) / (2 * width)));
      std::complex<TPrecision> factor4 =
          std::exp(-i * std::complex<TPrecision>(M_PI * (-y) / (2 * height)));
      if (x == 0 && y == 0) {
        out[y * width + x] =
            4.0 * (factor * (factor2 * output[y * out_width + x])).real();
      } else if (x == 0) {
        out[y * width + x] =
            2.0 * (factor2 * (factor * output[y * out_width + x] +
                              factor4 * output[(height - y) * out_width + x]))
                      .real();
      } else if (y == 0) {
        out[y * width + x] =
            2.0 * (factor * (factor2 * output[y * out_width + x] +
                             factor3 * std::conj(output[y * out_width + x])))
                      .real();
      } else {
        out[y * width + x] =
            2.0 *
            (factor * (factor2 * output[y * out_width + x] +
                       factor3 * std::conj(output[(height - y) * out_width + x])))
                .real();
      }
    }
  }

  for (int y = 0; y < height; y++) {
    for (int x = width / 2 + 1; x < width; x++) {
      std::complex<TPrecision> factor =
          std::exp(-i * std::complex<TPrecision>(M_PI * y / (2 * height)));
      std::complex<TPrecision> factor2 =
          std::exp(-i * std::complex<TPrecision>(M_PI * x / (2 * width)));
      std::complex<TPrecision> factor3 =
          std::exp(-i * std::complex<TPrecision>(M_PI * (-x) / (2 * width)));
      std::complex<TPrecision> factor4 =
          std::exp(-i * std::complex<TPrecision>(M_PI * (-y) / (2 * height)));
      if (x == 0 && y == 0) {
        out[y * width + x] =
            4.0 *
            (factor * (factor2 * std::conj(output[y * out_width + width - x])))
                .real();
      } else if (x == 0) {
        out[y * width + x] =
            2.0 *
            (factor2 *
             (factor * std::conj(output[(height - y) * out_width + width - x]) +
              factor4 * std::conj(output[y * out_width + width - x])))
                .real();
      } else if (y == 0) {
        out[y * width + x] =
            2.0 *
            (factor * (factor2 * std::conj(output[y * out_width + width - x]) +
                       factor3 * output[y * out_width + width - x]))
                .real();
      } else {
        out[y * width + x] =
            2.0 *
            (factor *
             (factor2 * std::conj(output[(height - y) * out_width + width - x]) +
              factor3 * output[y * out_width + width - x]))
                .real();
      }
    }
  }

  free(input);
  free(output);
}

template <class T, class TPrecision>
void dct_inverse_device(TPrecision *in, int width, int height, T *out,
                        int width_out = 0) {
  if (width_out == 0) {
    width_out = width;
  }

  int in_width = width / 2 + 1;

  // define dimensions
  dim3 threadsPerBlock(32, 32); // 1024 threads
  dim3 numBlocks(width_out / threadsPerBlock.x + 1,
                 height / threadsPerBlock.y + 1);

  // prepare buffers
  std::complex<TPrecision> *input;
  gpuErrchk(cudaMalloc((void **)&input,
                       sizeof(std::complex<TPrecision>) * in_width * height));
  TPrecision *output;
  gpuErrchk(cudaMalloc((void **)&output, sizeof(TPrecision) * width * height));

  iFactoring<<<numBlocks, threadsPerBlock>>>(
      in, reinterpret_cast<complex<TPrecision> *>(input), width, height, width,
      height);
  cudaDeviceSynchronize();
  HANDLE_ERROR(cudaGetLastError());

  fftTransformDevice(reinterpret_cast<TPrecision *>(input), width, height, output,
                     false, true);

  iReorder<<<numBlocks, threadsPerBlock>>>(output, out, width, height, width_out,
                                           width, height);
  cudaDeviceSynchronize();
  HANDLE_ERROR(cudaGetLastError());
}

template <class T, class TPrecision>
void dct_inverse(TPrecision *in, int width, int height, T *out,
                 int width_out = 0) {
  if (width_out == 0) {
    width_out = width;
  }

  int in_width = width / 2 + 1;

  std::complex<TPrecision> *input =
      new std::complex<TPrecision>[in_width * height];

  const std::complex<TPrecision> i(0.0, 1.0);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < in_width; x++) {
      std::complex<TPrecision> factor3 =
          std::exp(-i * std::complex<TPrecision>(M_PI * (-x) / (2 * width)));
      std::complex<TPrecision> factor4 =
          std::exp(-i * std::complex<TPrecision>(M_PI * (-y) / (2 * height)));
      if (x == 0 && y == 0) {
        input[y * in_width + x] = std::complex<TPrecision>(0.25, 0.0) * factor4 *
                                  factor3 *
                                  std::complex<TPrecision>(in[y * width + x]);
      } else if (y == 0) {
        input[y * in_width + x] =
            std::complex<TPrecision>(0.25, 0.0) * factor4 * factor3 *
            (std::complex<TPrecision>(in[y * width + x]) -
             i * std::complex<TPrecision>(in[y * width + width - x]));
      } else if (x == 0) {
        input[y * in_width + x] =
            std::complex<TPrecision>(0.25, 0.0) * factor4 * factor3 *
            (std::complex<TPrecision>(in[y * width + x]) -
             i * std::complex<TPrecision>(in[(height - y) * width + x]));
      } else {
        input[y * in_width + x] =
            std::complex<TPrecision>(0.25, 0.0) * factor4 * factor3 *
            ((std::complex<TPrecision>(in[y * width + x]) -
              std::complex<TPrecision>(in[(height - y) * width + width - x])) -
             i * (std::complex<TPrecision>(in[(height - y) * width + x]) +
                  std::complex<TPrecision>(in[y * width + width - x])));
      }
    }
  }

  // prepare output buffer
  TPrecision *output = new TPrecision[width * height];

  fftTransform(reinterpret_cast<TPrecision *>(input), width, height, output,
               false, true);

  for (int y = 0; y < height / 2; y++) {
    for (int x = 0; x < width / 2; x++) {
      out[(2 * y) * width_out + (2 * x)] = (T)output[y * width + x];
      out[(2 * y) * width_out + (2 * (width / 2 - x - 1) + 1)] =
          (T)output[y * width + (x + width / 2)];
      out[(2 * (height / 2 - y - 1) + 1) * width_out + (2 * x)] =
          (T)output[(y + height / 2) * width + x];
      out[(2 * (height / 2 - y - 1) + 1) * width_out +
          (2 * (width / 2 - x - 1) + 1)] =
          (T)output[(y + height / 2) * width + (x + width / 2)];
    }
  }

  free(output);
}

template <class TPrecision>
void dctTransform(TPrecision *in, int width, int height, TPrecision *out,
                  bool forward = true) {
  if (forward) {
    dct_forward(in, width, height, out);
  } else {
    dct_inverse(in, width, height, out);
  }
}

template <class TPrecision>
void dctTransformDevice(TPrecision *in, int width, int height, TPrecision *out,
                        bool forward = true) {
  if (forward) {
    dct_forward_device(in, width, height, out);
  } else {
    dct_inverse_device(in, width, height, out);
  }
}

// forward FCT
template <class T, class TPrecision> TPrecision *dct(HipaccImage &in) {
  int width = in->width;
  int height = in->height;
  int width_in = alignedWidth<T>(width, in->alignment);

  // prepare input buffer
  T *input_d = new T[width_in * height];
  // convert input
  gpuErrchk(cudaMemcpy(input_d, in->mem, sizeof(T) * width_in * height,
                       cudaMemcpyDeviceToHost));

  TPrecision *out = new TPrecision[width * height];
  dct_forward(input_d, width, height, out, width_in);

  free(input_d);

  return out;
}

// inverse FCT
template <class T, class TPrecision> void idct(TPrecision *in, HipaccImage &out) {
  int width = out->width;
  int height = out->height;
  int width_out = alignedWidth<T>(width, out->alignment);

  T *out_d = new T[width_out * height];

  dct_inverse(in, width, height, out_d, width_out);

  gpuErrchk(cudaMemcpy(out->mem, out_d, sizeof(T) * width_out * height,
                       cudaMemcpyHostToDevice));

  free(out_d);
}

/*
template <class T, class TPrecision> TPrecision *dct(HipaccImage &in) {
  int width = in->width;
  int height = in->height;
  int width_in = alignedWidth<T>(width, in->alignment);

  // prepare output buffer
  TPrecision *output;
  gpuErrchk(cudaMalloc((void **)&output, sizeof(TPrecision) * width * height));
  dct_forward_device((TPrecision *)(in->mem), width, height, output, width_in);

  return output;
}

// inverse FCT
template <class T, class TPrecision> void idct(TPrecision *in, HipaccImage &out) {
  int width = out->width;
  int height = out->height;
  int width_out = alignedWidth<T>(width, out->alignment);

  //dct_inverse_device(in, width, height, (TPrecision *)(out->mem), width_out);
  dct_inverse_device(in, 1000, 100, (TPrecision *)(out->mem));

}
*/

// function wrappers for images

template <class T> void fftShift(HipaccImage &mag) {
  int width = mag->width;
  int height = mag->height;
  int width_in = alignedWidth<T>(width, mag->alignment);
  T *input = new T[width_in * height];
  gpuErrchk(cudaMemcpy(input, mag->mem, sizeof(T) * width_in * height,
                       cudaMemcpyDeviceToHost));

  shiftFFT(input, width, height, width_in);

  gpuErrchk(cudaMemcpy(mag->mem, input, sizeof(T) * width_in * height,
                       cudaMemcpyHostToDevice));
}

template <class T> void ifftShift(HipaccImage &mag) {
  int width = mag->width;
  int height = mag->height;
  int width_in = alignedWidth<T>(width, mag->alignment);
  T *input = new T[width_in * height];
  gpuErrchk(cudaMemcpy(input, mag->mem, sizeof(T) * width_in * height,
                       cudaMemcpyDeviceToHost));

  iShiftFFT(input, width, height, width_in);

  gpuErrchk(cudaMemcpy(mag->mem, input, sizeof(T) * width_in * height,
                       cudaMemcpyHostToDevice));
}

template <class T>
void fftResetMask(HipaccImage &mag, int radius, bool low, int window = 0) {
  int width = mag->width;
  int height = mag->height;
  int width_in = alignedWidth<T>(width, mag->alignment);
  T *input = new T[width_in * height];
  gpuErrchk(cudaMemcpy(input, mag->mem, sizeof(T) * width_in * height,
                       cudaMemcpyDeviceToHost));

  magResetFreq(input, width, height, width_in, radius, window, low);

  gpuErrchk(cudaMemcpy(mag->mem, input, sizeof(T) * width_in * height,
                       cudaMemcpyHostToDevice));
}

template <class T>
void fftApplyPassFilter(HipaccImage &mag, int radius, bool low, int window = 0) {
  int width = mag->width;
  int height = mag->height;
  int width_in = alignedWidth<T>(width, mag->alignment);
  T *input = new T[width_in * height];
  gpuErrchk(cudaMemcpy(input, mag->mem, sizeof(T) * width_in * height,
                       cudaMemcpyDeviceToHost));

  magPassFilter(input, width, height, width_in, radius, window, low);

  gpuErrchk(cudaMemcpy(mag->mem, input, sizeof(T) * width_in * height,
                       cudaMemcpyHostToDevice));
}
