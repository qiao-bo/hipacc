// This is a helper file for both hipacc_cpu_fftw.hpp and hipacc_cu_cufft.hpp

#include <cmath>
#include <complex>
#include <string>

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

template <class T, class U>
void putKernel(T *kernel, U *dest, int kx, int ky, int nx, int ny) {
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

template <class T, class U>
void putKernelComplex(T *kernel, std::complex<U> *dest, int kx, int ky, int nx,
                      int ny) {
  // edges wrap-around
  for (int y = -ky / 2; y <= ky / 2; y++) {
    for (int x = -kx / 2; x <= kx / 2; x++) {
      dest[nx * ((ny + y) % ny) + ((nx + x) % nx)].real(
          kernel[kx * (ky / 2 + y) + (kx / 2 + x)]);
      dest[nx * ((ny + y) % ny) + ((nx + x) % nx)].imag(0);
    }
  }
}

// only unshifted
template <class T, class U>
void calcMagnitude(std::complex<T> *in, U *out, int width, int height) {
  // in is result of fft (r2c: only left half (width/2+1)*height)
  int width_in = width / 2 + 1;
  int N = width * height;
  float *magnitude = (float *)malloc(sizeof(float) * width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width_in; x++) {
      magnitude[y * width + x] = std::abs(in[y * width_in + x]);
      if (x < width / 2)
        magnitude[width * height - (y * width + x) - 1] =
            std::abs(in[y * width_in + x + 1]); // point mirror
    }
  }
  float max = 0.0;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width_in; x++) {
      if (magnitude[y * width + x] > max)
        max = magnitude[y * width + x];
    }
  }
  for (int i = 0; i < N; ++i) {
    out[i] = (U)(255.0 * pow(magnitude[i] * (1.0 / max), 1.0 / 4.0));
  }
  free(magnitude);
}

// 2d access to 1d array
inline int linearize(int w, int x, int y) { return y * w + x; }

// get next width that is aligned to a size of alignment of type T
template <class T> int alignedWidth(int width, int alignment) {
  alignment /= sizeof(T);
  int res = width;
  int rest = res % alignment;
  if (rest != 0) {
    res += alignment - (rest);
  }
  return res;
}

// get int greater or equal val that is divisible by div
int nextDiv(int val, int div) {
  int newVal = val;
  int rest = newVal % div;
  if (rest != 0) {
    newVal += div - (rest);
  }
  return newVal;
}

// https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2
unsigned long upperPowerOfTwo(unsigned long v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

// https://stackoverflow.com/questions/4424374/determining-if-a-number-is-prime
bool isPrime(int number) {
  if (number < 2)
    return false;
  if (number == 2)
    return true;
  if (number % 2 == 0)
    return false;
  for (int i = 3; (i * i) <= number; i += 2) {
    if (number % i == 0)
      return false;
  }
  return true;
}

// shift the zero-frequency component to the center of the image
template <class T>
void shiftFFT(T *image, int width, int height, int alignment = 0) {
  int width_align = width;
  if (alignment) {
    width_align = alignedWidth<T>(width, alignment);
  }
  T *tmp = new T[width_align * height];

  int y_inc = height / 2;
  int x_inc = width / 2;
  int cpy_len_l = sizeof(T) * (width / 2);
  int cpy_len_r = sizeof(T) * (width / 2);
  if (width % 2) {
    cpy_len_l += 1;
  }
  int src_x_l = 0;
  int src_x_r = width / 2;
  if (width % 2) {
    src_x_r += 1;
  }

  for (int y = 0; y < height; y++) {
    memcpy(&tmp[linearize(width_align, (src_x_l + x_inc) % width,
                          (y + y_inc) % height)],
           &image[linearize(width_align, src_x_l, y)], cpy_len_l);
    memcpy(&tmp[linearize(width_align, (src_x_r + x_inc) % width,
                          (y + y_inc) % height)],
           &image[linearize(width_align, src_x_r, y)], cpy_len_r);
  }

  memcpy(image, tmp, sizeof(T) * width_align * height);
  free(tmp);
}

// inverse shift the zero-frequency
template <class T>
void iShiftFFT(T *image, int width, int height, int alignment = 0) {
  int width_align = width;
  if (alignment) {
    width_align = alignedWidth<T>(width, alignment);
  }
  T *tmp = new T[width_align * height];

  int y_inc = height / 2;
  int x_inc = width / 2;
  if (width % 2) {
    y_inc += 1;
    x_inc += 1;
  }
  int cpy_len_l = sizeof(T) * (width / 2);
  int cpy_len_r = sizeof(T) * (width / 2);
  if (width % 2) {
    cpy_len_r += 1;
  }
  int src_x_l = 0;
  int src_x_r = width / 2;

  for (int y = 0; y < height; y++) {
    memcpy(&tmp[linearize(width_align, (src_x_l + x_inc) % width,
                          (y + y_inc) % height)],
           &image[linearize(width_align, src_x_l, y)], cpy_len_l);
    memcpy(&tmp[linearize(width_align, (src_x_r + x_inc) % width,
                          (y + y_inc) % height)],
           &image[linearize(width_align, src_x_r, y)], cpy_len_r);
  }

  memcpy(image, tmp, sizeof(T) * width_align * height);
  free(tmp);
}

// hann window of size N for cos weighted gradient
float hannWindow(float N, float x) {
  if (x > N)
    return 1;
  N *= 2;
  return 0.5 * (1.0 - cos(2.0 * M_PI * x / N));
}

// reset mask values for low frequencies below r with gradient blur of size w
template <class T>
void magResetLowFreq(T *image, int width, int height, int alignment, float r,
                     float w) {
  int width_align = alignedWidth<T>(width, alignment);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int xd = x - width / 2;
      int yd = y - height / 2;
      float d = sqrt(xd * xd + yd * yd);
      if (d < r) {
        image[y * width_align + x] =
            255 - (255 - image[y * width_align + x]) * (1 - hannWindow(w, r - d));
      }
    }
  }
}

// low pass filter for low frequencies below r with gradient blur of size w
template <class T>
void magLowPassFilter(T *image, int width, int height, int alignment, int r,
                      int w) {
  int width_align = alignedWidth<T>(width, alignment);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int xd = x - width / 2;
      int yd = y - height / 2;
      float d = sqrt(xd * xd + yd * yd);
      if (d > r) {
        image[y * width_align + x] *= (1 - hannWindow(w, d - r));
      }
    }
  }
}
