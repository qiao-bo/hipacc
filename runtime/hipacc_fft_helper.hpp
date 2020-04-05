// This is a helper file for both hipacc_cpu_fftw.hpp and hipacc_cu_cufft.hpp

// DEBUG
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <string>

#include <sys/time.h>
struct timeval t0, t1;
void start() { gettimeofday(&t0, NULL); }
double stop() {
  gettimeofday(&t1, NULL);
  long long time = (t1.tv_sec - t0.tv_sec) * 1000000LL + t1.tv_usec - t0.tv_usec;
  printf("time: %.3f ms\n", (double)time / 1000);
  return (double)time / 1000;
}

// get mat type as string
std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch (depth) {
  case CV_8U:
    r = "8U";
    break;
  case CV_8S:
    r = "8S";
    break;
  case CV_16U:
    r = "16U";
    break;
  case CV_16S:
    r = "16S";
    break;
  case CV_32S:
    r = "32S";
    break;
  case CV_32F:
    r = "32F";
    break;
  case CV_64F:
    r = "64F";
    break;
  default:
    r = "User";
    break;
  }

  r += "C";
  r += (chans + '0');

  return r;
}
// END DEBUG

// https://stackoverflow.com/questions/8204645/implementing-gaussian-blur-how-to-calculate-convolution-matrix-kernel#8204880
void calcGauss(float *kernel, int W) {
  double sigma = 1;
  double mean = W / 2;
  double sum = 0.0; // For accumulating the kernel values
  for (int x = 0; x < W; ++x) {
    for (int y = 0; y < W; ++y) {
      kernel[x * W + y] = exp(-0.5 * (pow((x - mean) / sigma, 2.0) +
                                      pow((y - mean) / sigma, 2.0))) /
                          (2 * M_PI * sigma * sigma);

      // Accumulate the kernel values
      sum += kernel[x * W + y];
    }
  }
  // Normalize the kernel
  for (int x = 0; x < W; ++x)
    for (int y = 0; y < W; ++y)
      kernel[x * W + y] /= sum;
}
// boxblur
void calcBox(float *kernel, int W) {
  double count = W * W;
  for (int x = 0; x < count; ++x) {
    kernel[x] = 1.0 / count;
  }
}

// from HIPAcc
// Gaussian blur filter reference
void gaussian_filter(uchar *in, uchar *out, float *filter, int size_x, int size_y,
                     int width, int height) {
  int anchor_x = size_x >> 1;
  int anchor_y = size_y >> 1;
  int upper_x = width - anchor_x;
  int upper_y = height - anchor_y;

  for (int y = anchor_y; y < upper_y; ++y) {
    for (int x = anchor_x; x < upper_x; ++x) {
      float sum = 0.5f;

      for (int yf = -anchor_y; yf <= anchor_y; ++yf) {
        for (int xf = -anchor_x; xf <= anchor_x; ++xf) {
          sum += filter[(yf + anchor_y) * size_x + xf + anchor_x] *
                 in[(y + yf) * width + x + xf];
        }
      }
      out[y * width + x] = (uchar)(sum);
    }
  }
}

void putKernel(float *kernel, double *dest, int kx, int ky, int nx, int ny) {
  // centered
  /*for (int y = -ky/2; y <= ky/2; y++) {
      for (int x = -kx/2; x <= kx/2; x++) {
          dest[nx * (ny/2 + y) + (nx/2 + x)] = kernel[kx * (ky/2+y) + (kx/2+x)];
      }
  }*/

  // edges wrap-around
  for (int y = -ky / 2; y <= ky / 2; y++) {
    for (int x = -kx / 2; x <= kx / 2; x++) {
      dest[nx * ((ny + y) % ny) + ((nx + x) % nx)] =
          kernel[kx * (ky / 2 + y) + (kx / 2 + x)];
    }
  }
  // dest[0] = 1.0;
}

void putKernelComplex(float *kernel, std::complex<double> *dest, int kx, int ky,
                      int nx, int ny) {
  // edges wrap-around
  for (int y = -ky / 2; y <= ky / 2; y++) {
    for (int x = -kx / 2; x <= kx / 2; x++) {
      dest[nx * ((ny + y) % ny) + ((nx + x) % nx)].real(
          kernel[kx * (ky / 2 + y) + (kx / 2 + x)]);
      dest[nx * ((ny + y) % ny) + ((nx + x) % nx)].imag(0);
    }
  }
}

void calcMagnitude(std::complex<double> *in, uchar *out, int N) {
  float *magnitude = (float *)malloc(sizeof(float) * N);
  for (int i = 0; i < N; ++i) {
    magnitude[i] = std::abs(in[i]);
  }
  float max = 0.0;
  for (int i = 0; i < N; ++i) {
    if (magnitude[i] > max)
      max = magnitude[i];
  }
  for (int i = 0; i < N; ++i) {
    out[i] = (uchar)(255.0 * pow(magnitude[i] * (1.0 / max), 1.0 / 4.0));
  }
}

inline int linearize(int w, int x, int y) { return y * w + x; }

template <typename T> void shiftFFT(T *image, int width, int height) {
  int hw = width / 2;
  int hh = height / 2;
  int size_n = sizeof(T) * hw;
  T *temp = (T *)malloc(size_n);
  for (int y = 0; y < hh; y++) {
    memcpy(temp, &image[linearize(width, 0, y)], size_n);
    memcpy(&image[linearize(width, 0, y)], &image[linearize(width, hw, hh + y)],
           size_n);
    memcpy(&image[linearize(width, hw, hh + y)], temp, size_n);
  }
  for (int y = 0; y < hh; y++) {
    memcpy(temp, &image[linearize(width, hw, y)], size_n);
    memcpy(&image[linearize(width, hw, y)], &image[linearize(width, 0, hh + y)],
           size_n);
    memcpy(&image[linearize(width, 0, hh + y)], temp, size_n);
  }
  free(temp);
}