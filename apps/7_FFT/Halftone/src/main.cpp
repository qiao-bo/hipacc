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
/*
#define WIDTH 700
#define HEIGHT 445
#define IMAGE "../../common/img/halftone-printing-process.jpg"
*/
#define WIDTH 600
#define HEIGHT 450
#define IMAGE "../../common/img/halftone-face.jpg"

#define TYPE uchar

using namespace hipacc;
using namespace hipacc::math;

// Kernel description in Hipacc
class Dilate : public Kernel<TYPE> {
private:
  Accessor<TYPE> &in;
  Domain &dom;

public:
  Dilate(IterationSpace<TYPE> &iter, Accessor<TYPE> &in, Domain &dom)
      : Kernel(iter), in(in), dom(dom) {
    add_accessor(&in);
  }

  void kernel() {
    output() = reduce(dom, Reduce::MIN, [&]() -> int { return in(dom); });
  }
};

class ImproveFilter : public Kernel<TYPE> {
private:
  Accessor<TYPE> &in;
  Domain &dom;
  int size_x, size_y;
  Mask<float> &mask;

public:
  ImproveFilter(IterationSpace<TYPE> &iter, Accessor<TYPE> &in, Domain &dom,
                int size_x, int size_y, Mask<float> &mask)
      : Kernel(iter), in(in), dom(dom), size_x(size_x), size_y(size_y),
        mask(mask) {
    add_accessor(&in);
  }

  void kernel() {
    float w_avg =
        convolve(mask, Reduce::SUM, [&]() -> float { return mask() * in(mask); });
    float avg = reduce(dom, Reduce::SUM, [&]() -> int { return in(dom); }) /
                (float)(size_x * size_y);
    float max = reduce(dom, Reduce::MAX, [&]() -> int { return in(dom); });
    output() = ((w_avg - avg) > 1) || (max > (avg / 2 + 40)) ? 0 : 255;
  }
};

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

/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
  const int width = WIDTH;
  const int height = HEIGHT;
  const int size_x = SIZE_X;
  const int size_y = SIZE_Y;

  // only filter kernel sizes 3x3, 5x5, and 7x7 implemented
  if (size_x != size_y || !(size_x == 3 || size_x == 5 || size_x == 7)) {
    std::cerr << "Wrong filter kernel size. "
              << "Currently supported values: 3x3, 5x5, and 7x7!" << std::endl;
    exit(EXIT_FAILURE);
  }

  // convolution filter mask
  const float coef3[3][3] = {{0.057118f, 0.124758f, 0.057118f},
                             {0.124758f, 0.272496f, 0.124758f},
                             {0.057118f, 0.124758f, 0.057118f}};

  const float coef5[5][5] = {
      {0.005008f, 0.017300f, 0.026151f, 0.017300f, 0.005008f},
      {0.017300f, 0.059761f, 0.090339f, 0.059761f, 0.017300f},
      {0.026151f, 0.090339f, 0.136565f, 0.090339f, 0.026151f},
      {0.017300f, 0.059761f, 0.090339f, 0.059761f, 0.017300f},
      {0.005008f, 0.017300f, 0.026151f, 0.017300f, 0.005008f}};

  const float coef7[7][7] = {{0.000841f, 0.003010f, 0.006471f, 0.008351f,
                              0.006471f, 0.003010f, 0.000841f},
                             {0.003010f, 0.010778f, 0.023169f, 0.029902f,
                              0.023169f, 0.010778f, 0.003010f},
                             {0.006471f, 0.023169f, 0.049806f, 0.064280f,
                              0.049806f, 0.023169f, 0.006471f},
                             {0.008351f, 0.029902f, 0.064280f, 0.082959f,
                              0.064280f, 0.029902f, 0.008351f},
                             {0.006471f, 0.023169f, 0.049806f, 0.064280f,
                              0.049806f, 0.023169f, 0.006471f},
                             {0.003010f, 0.010778f, 0.023169f, 0.029902f,
                              0.023169f, 0.010778f, 0.003010f},
                             {0.000841f, 0.003010f, 0.006471f, 0.008351f,
                              0.006471f, 0.003010f, 0.000841f}};

  // host memory for image of width x height pixels
  TYPE *input = load_data<TYPE>(width, height, 1, IMAGE);

  // std::cout << "Calculating Hipacc Gaussian filter ..." << std::endl;

  //************************************************************************//

  // input and output image of width x height pixels
  Image<TYPE> in(width, height, input);
  Image<TYPE> out(width, height);

  // blur masks
  Mask<float> mask3(coef3);
  Mask<float> mask5(coef5);
  Mask<float> mask7(coef7);

  // images
  Image<TYPE> mag(width, height);
  Image<TYPE> mag_blur(width, height);
  Image<TYPE> mag_out(width, height);
  Image<TYPE> mag_out2(width, height);
  Image<TYPE> mag_out3(width, height);

  // kernel domains
  Domain domImp(SIZE_X, SIZE_Y);
  Domain domDil(5, 5);

  // declare all acessors and iterationSpaces for images
  BoundaryCondition<TYPE> bound_mag(mag, mask3, Boundary::CLAMP);
  Accessor<TYPE> acc_mag(bound_mag);
  IterationSpace<TYPE> iter_mag(mag);

  BoundaryCondition<TYPE> bound_mag_out(mag_out, domDil, Boundary::CLAMP);
  Accessor<TYPE> acc_mag_out(bound_mag_out);
  IterationSpace<TYPE> iter_mag_out(mag_out);

  BoundaryCondition<TYPE> bound_mag_out2(mag_out2, mask5, Boundary::CLAMP);
  Accessor<TYPE> acc_mag_out2(bound_mag_out2);
  IterationSpace<TYPE> iter_mag_out2(mag_out2);

  BoundaryCondition<TYPE> bound_mag_out3(mag_out3, domImp, Boundary::CLAMP);
  Accessor<TYPE> acc_mag_out3(bound_mag_out3);
  IterationSpace<TYPE> iter_mag_out3(mag_out3);

  BoundaryCondition<TYPE> bound_mag_blur(mag_blur, domImp, Boundary::CLAMP);
  Accessor<TYPE> acc_mag_blur(bound_mag_blur);
  IterationSpace<TYPE> iter_mag_blur(mag_blur);

  // write fft of in to fftResult
  float *fftResult = (float *)fft<TYPE, float>(in);

  // create magnitude from fft
  fftToMag<TYPE>(fftResult, mag);

  // blur magnitude
  Domain domBlur(7, 7);
  GaussianBlur blur(iter_mag_blur, acc_mag, mask3);
  blur.execute();

  // apply ImproveFilter
  ImproveFilter improve(iter_mag_out, acc_mag_blur, domImp, SIZE_X, SIZE_Y,
                        mask7);
  improve.execute();
  Dilate dilate(iter_mag_out2, acc_mag_out, domDil);
  dilate.execute();

  // blur magnitude mask to prevent ringing effect
  GaussianBlur blur_mask(iter_mag_out3, acc_mag_out2, mask5);
  blur_mask.execute();

  // apply mask to fftResult (with radius of highpass filter)
  magScaleFFT<TYPE>(fftResult, mag_out3, min(width, height) / 20);

  // visualize resulting magnitude
  Image<TYPE> back(width, height);
  fftToMag<TYPE>(fftResult, back);

  // get result image from inverse fft
  ifft<TYPE>(fftResult, out);

  // get pointer to result data
  TYPE *output = out.data();
  TYPE *Mag = mag.data();
  TYPE *Mag_out = mag_out3.data();
  TYPE *Back = back.data();

  save_data(width, height, 1, input, "input.jpg");
  save_data(width, height, 1, output, "output.jpg");
  save_data(width, height, 1, Mag, "Mag.png");
  save_data(width, height, 1, Mag_out, "Mag_mask.png");
  save_data(width, height, 1, Back, "Mag_new.png");
  show_data(width, height, 1, output, "output.jpg");

  // free memory
  delete[] input;

  return EXIT_SUCCESS;
}
