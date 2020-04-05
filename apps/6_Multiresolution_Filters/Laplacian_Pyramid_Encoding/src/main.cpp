//
// Copyright (c) 2020, University of Erlangen-Nuremberg
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
#include <iostream>
#include <hipacc_helper.hpp>

#define SIZE_X 3
#define SIZE_Y 3
#define WIDTH 1024
#define HEIGHT 1024

using namespace hipacc;
using namespace hipacc::math;


class Gaussian : public Kernel<char> {
  private:
    Accessor<char> &input;
    Mask<float> &mask;

  public:
    Gaussian(IterationSpace<char> &iter, Accessor<char> &input,
             Mask<float> &mask)
          : Kernel(iter), input(input), mask(mask) {
        add_accessor(&input);
    }

    void kernel() {
        output() = convolve(mask, Reduce::SUM, [&] () {
            return input(mask) * mask();
        });
    }

    void _operatePyramidReduce() {}
};

class DifferenceOfGaussian : public Kernel<char> {
  private:
    Accessor<char> &input1;
    Accessor<char> &input2;

  public:
    DifferenceOfGaussian(IterationSpace<char> &iter, Accessor<char> &input1,
                         Accessor<char> &input2)
          : Kernel(iter), input1(input1), input2(input2) {
        add_accessor(&input1);
        add_accessor(&input2);
    }

    void kernel() {
        output() = input1() - input2();
    }

    void _operatePyramidFilter() {}
};

class Blend : public Kernel<char> {
  private:
    Accessor<char> &input1;
    Accessor<char> &input2;

  public:
    Blend(IterationSpace<char> &iter, Accessor<char> &input1,
          Accessor<char> &input2)
        : Kernel(iter), input1(input1), input2(input2) {
      add_accessor(&input1);
      add_accessor(&input2);
    }

    void kernel() {
        output() = (short)input1() + (short)input2() / 2;
    }

    void _operatePyramidExpand() {}
};


/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    const int width = WIDTH;
    const int height = HEIGHT;
    const int size_x = SIZE_X;
    const int size_y = SIZE_Y;
    float timing = 0;

    // only filter kernel sizes 3x3, 5x5, and 7x7 implemented
    if (size_x != size_y || !(size_x == 3 || size_x == 5 || size_x == 7)) {
        std::cout << "Wrong filter kernel size. "
                  << "Currently supported values: 3x3, 5x5, and 7x7!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // convolution filter mask
    const float coef[SIZE_Y][SIZE_X] = {
#if SIZE_X == 3
        { 0.057118f, 0.124758f, 0.057118f },
        { 0.124758f, 0.272496f, 0.124758f },
        { 0.057118f, 0.124758f, 0.057118f }
#endif
#if SIZE_X == 5
        { 0.005008f, 0.017300f, 0.026151f, 0.017300f, 0.005008f },
        { 0.017300f, 0.059761f, 0.090339f, 0.059761f, 0.017300f },
        { 0.026151f, 0.090339f, 0.136565f, 0.090339f, 0.026151f },
        { 0.017300f, 0.059761f, 0.090339f, 0.059761f, 0.017300f },
        { 0.005008f, 0.017300f, 0.026151f, 0.017300f, 0.005008f }
#endif
#if SIZE_X == 7
        { 0.000841f, 0.003010f, 0.006471f, 0.008351f, 0.006471f, 0.003010f, 0.000841f },
        { 0.003010f, 0.010778f, 0.023169f, 0.029902f, 0.023169f, 0.010778f, 0.003010f },
        { 0.006471f, 0.023169f, 0.049806f, 0.064280f, 0.049806f, 0.023169f, 0.006471f },
        { 0.008351f, 0.029902f, 0.064280f, 0.082959f, 0.064280f, 0.029902f, 0.008351f },
        { 0.006471f, 0.023169f, 0.049806f, 0.064280f, 0.049806f, 0.023169f, 0.006471f },
        { 0.003010f, 0.010778f, 0.023169f, 0.029902f, 0.023169f, 0.010778f, 0.003010f },
        { 0.000841f, 0.003010f, 0.006471f, 0.008351f, 0.006471f, 0.003010f, 0.000841f }
#endif
    };

    // host memory for random generated image of width x height pixels
    char *input = new char[width * height];
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        input[y * width + x] = (char)((y * width + x) % 19);
      }
    }

    std::cout << "Calculating Hipacc Gaussian Laplacian pyramid ..." << std::endl;

    //************************************************************************//

    // input and output image of width x height pixels
    Image<char> gaus(width, height);
    Image<char> lap(width, height);
    Mask<float> mask(coef);

    const int depth = 10;
    Pyramid<char> pgaus(gaus, depth);
    Pyramid<char> plap(lap, depth);

    gaus = input;
    traverse(pgaus, plap, [&] () {
        if (!pgaus.is_top_level()) {
            // construct Gaussian pyramid
            BoundaryCondition<char> bound(pgaus(-1), mask, Boundary::CLAMP);
            Accessor<char> acc1(bound);
            IterationSpace<char> iter1(pgaus(0));
            Gaussian blur(iter1, acc1, mask);
            blur.execute();

            // construct Laplacian pyramid
            Accessor<char> acc3(pgaus(-1));
            Accessor<char> acc4(pgaus(0), Interpolate::LF);
            IterationSpace<char> iter3(plap(-1));
            DifferenceOfGaussian DoG(iter3, acc3, acc4);
            DoG.execute();
        }

        traverse();

        // collapse pyramids
        if (!pgaus.is_bottom_level()) {
            // blend final output image from Laplacian pyramid
            Accessor<char> acc3(plap(1), Interpolate::LF);
            Accessor<char> acc4(plap(0));
            IterationSpace<char> iter2(plap(0));
            Blend blend(iter2, acc3, acc4);
            blend.execute();
        }
    });

    // get pointer to result data
    char *output = lap.data();

    //************************************************************************//

    // convert to uchar for visualization
    for (int p = 0; p < width*height; ++p) {
        output[p] = (char)(output[p] + 127);
    }

    // free memory
    delete[] input;

    return EXIT_SUCCESS;
}