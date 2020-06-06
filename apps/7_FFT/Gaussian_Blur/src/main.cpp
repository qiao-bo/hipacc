//
// Copyright (c) 2012, University of Erlangen-Nuremberg
// Copyright (c) 2012, Siemens AG
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "hipacc.hpp"

#include <hipacc_helper.hpp>
#include <iostream>

#define SIZE_X 7
#define SIZE_Y 7

#define WIDTH 4032
#define HEIGHT 3024
#define IMAGE "../../common/img/fuerte_ship.jpg"
/*
#define WIDTH 700
#define HEIGHT 445
#define IMAGE "../../common/img/halftone-printing-process.jpg"
*/
#define TYPE uchar

using namespace hipacc;
using namespace hipacc::math;

// Gaussian blur filter in Hipacc
class GaussianBlur : public Kernel<TYPE> {
private:
  Accessor<TYPE> &input;
  Mask<float> &mask;

public:
  GaussianBlur(IterationSpace<TYPE> &iter, Accessor<TYPE> &input,
               Mask<float> &mask)
      : Kernel(iter), input(input), mask(mask) {
    add_accessor(&input);
  }

  void kernel() {
    output() = (TYPE)(convolve(mask, Reduce::SUM,
                               [&]() -> float { return mask() * input(mask); }) +
                      0.0f);
  }
};

// forward declaration of reference implementation
template <typename T>
void gaussian_filter(T *in, T *out, float *filter, int size_x, int size_y,
                     int width, int height);

// alternative compare to analyse precision and errors
template <typename T>
void compare(const T *cmp1, const T *cmp2, const unsigned int width,
             const unsigned int height, const unsigned int border_x = 0,
             const unsigned int border_y = 0) {
  int count = 0;
  double max = 0.0;
  double sum = 0.0;
  for (int i = border_y; i < height - border_y; i++) {
    for (int j = border_x; j < width - border_x; j++) {
      double diff = (double)std::abs(cmp1[i * width + j] - cmp2[i * width + j]);
      if (diff > 0.0) {
        count++;
        sum += diff;
        max = std::max(diff, max);
      }
    }
  }
  double avg = sum / ((width - border_x * 2) * (height - border_y * 2));
  std::cout << "different pixels:" << count << " avg error:" << avg
            << " max error:" << max << std::endl;
}

/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
  const int width = WIDTH;
  const int height = HEIGHT;
  const int size_x = SIZE_X;
  const int size_y = SIZE_Y;
  const int offset_x = size_x >> 1;
  const int offset_y = size_y >> 1;
  float timing = 0;

  // only filter kernel sizes 3x3, 5x5, and 7x7 implemented
  if (size_x != size_y || !(size_x == 3 || size_x == 5 || size_x == 7)) {
    std::cerr << "Wrong filter kernel size. "
              << "Currently supported values: 3x3, 5x5, and 7x7!" << std::endl;
    exit(EXIT_FAILURE);
  }

  // convolution filter mask
  const float coef[SIZE_Y][SIZE_X] = {
#if SIZE_X == 3
    {0.057118f, 0.124758f, 0.057118f},
    {0.124758f, 0.272496f, 0.124758f},
    {0.057118f, 0.124758f, 0.057118f}
#endif
#if SIZE_X == 5
    {0.005008f, 0.017300f, 0.026151f, 0.017300f, 0.005008f},
    {0.017300f, 0.059761f, 0.090339f, 0.059761f, 0.017300f},
    {0.026151f, 0.090339f, 0.136565f, 0.090339f, 0.026151f},
    {0.017300f, 0.059761f, 0.090339f, 0.059761f, 0.017300f},
    {0.005008f, 0.017300f, 0.026151f, 0.017300f, 0.005008f}
#endif
#if SIZE_X == 7
    {0.000841f, 0.003010f, 0.006471f, 0.008351f, 0.006471f, 0.003010f, 0.000841f},
    {0.003010f, 0.010778f, 0.023169f, 0.029902f, 0.023169f, 0.010778f, 0.003010f},
    {0.006471f, 0.023169f, 0.049806f, 0.064280f, 0.049806f, 0.023169f, 0.006471f},
    {0.008351f, 0.029902f, 0.064280f, 0.082959f, 0.064280f, 0.029902f, 0.008351f},
    {0.006471f, 0.023169f, 0.049806f, 0.064280f, 0.049806f, 0.023169f, 0.006471f},
    {0.003010f, 0.010778f, 0.023169f, 0.029902f, 0.023169f, 0.010778f, 0.003010f},
    {0.000841f, 0.003010f, 0.006471f, 0.008351f, 0.006471f, 0.003010f, 0.000841f}
#endif
  };

  // host memory for image of width x height pixels
  TYPE *input = load_data<TYPE>(width, height, 1, IMAGE);
  TYPE *ref_out = new TYPE[width * height];

  std::cout << "Calculating Hipacc Gaussian filter ..." << std::endl;

  //************************************************************************//

  // input and output image of width x height pixels
  Image<TYPE> in(width, height, input);
  Image<TYPE> out(width, height);

  // define Mask for Gaussian filter
  Mask<float> mask(coef);

  BoundaryCondition<TYPE> bound(in, mask, Boundary::CONSTANT, 0);
  Accessor<TYPE> acc(bound);

  IterationSpace<TYPE> iter(out);
  GaussianBlur filter(iter, acc, mask);

  Image<TYPE> outFFT(width, height);
  IterationSpace<TYPE> iterFFT(outFFT);
  GaussianBlur filterFFT(iterFFT, acc, mask);

  filter.execute();
  timing = hipacc_last_kernel_timing();

  filterFFT.convolveFFT();

  // get pointer to result data
  TYPE *output = out.data();
  TYPE *outputFFT = outFFT.data();

  //************************************************************************//

  std::cout << "Hipacc: " << timing << " ms, " << (width * height / timing) / 1000
            << " Mpixel/s" << std::endl;

  std::cout << "Calculating reference ..." << std::endl;
  double start = time_ms();
  gaussian_filter(input, ref_out, (float *)coef, size_x, size_y, width, height);
  double end = time_ms();
  std::cout << "Reference: " << end - start << " ms, "
            << (width * height / (end - start)) / 1000 << " Mpixel/s"
            << std::endl;

  // compare hipacc to reference
  compare(output, ref_out, width, height, offset_x, offset_y);
  // compare hipacc to fft convolution
  compare(outputFFT, output, width, height /*, offset_x, offset_y*/);

  save_data(width, height, 1, input, "input.jpg");
  save_data(width, height, 1, output, "output.jpg");
  save_data(width, height, 1, outputFFT, "outputFFT.jpg");
  show_data(width, height, 1, output, "output.jpg");

  // free memory
  delete[] input;
  delete[] ref_out;

  return EXIT_SUCCESS;
}

// Gaussian blur filter reference
template <typename T>
void gaussian_filter(T *in, T *out, float *filter, int size_x, int size_y,
                     int width, int height) {
  int anchor_x = size_x >> 1;
  int anchor_y = size_y >> 1;
  int upper_x = width - anchor_x;
  int upper_y = height - anchor_y;

  for (int y = anchor_y; y < upper_y; ++y) {
    for (int x = anchor_x; x < upper_x; ++x) {
      float sum = 0.0f;

      for (int yf = -anchor_y; yf <= anchor_y; ++yf) {
        for (int xf = -anchor_x; xf <= anchor_x; ++xf) {
          sum += filter[(yf + anchor_y) * size_x + xf + anchor_x] *
                 in[(y + yf) * width + x + xf];
        }
      }
      out[y * width + x] = (T)(sum);
    }
  }
}
