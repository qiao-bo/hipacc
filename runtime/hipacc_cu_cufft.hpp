#include <cufft.h>
#include <cufftXt.h>

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

  estimateConvolutionExecutionTime(width, height, padWidth, padHeight, k_w, k_h,
                                   linear);

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

template <class TPrecision>
void fft_transform(TPrecision *in, int width, int height, TPrecision *out,
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
void fft_transform_device(TPrecision *in, int width, int height, TPrecision *out,
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
    ScaleE<<<numBlocks, threadsPerBlock>>>(
        reinterpret_cast<cufft_Real *>(out), image_width,
        image_height, 1.0 / (image_height * image_width));
    HANDLE_ERROR(cudaGetLastError());
  }

  // cleanup
  cufftDestroy(plan_forward_many);
}

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
  fft_transform(input, width, height, output, true);

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

  fft_transform(in, width, height, output, false, true);

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
// and ignore values for frequencies lower than r
template <class T, class TPrecision>
void magScaleFFT(TPrecision *in, HipaccImage &mag, float r) {
  int width = mag->width;
  int height = mag->height;
  int width_in = alignedWidth<T>(width, mag->alignment);
  T *scale = new T[width_in * height];
  gpuErrchk(cudaMemcpy(scale, mag->mem, sizeof(T) * width_in * height,
                       cudaMemcpyDeviceToHost));

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
