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

/*#define WIDTH 4032
#define HEIGHT 3024
#define IMAGE "../../common/img/fuerte_ship.jpg"*/
#define WIDTH 600
#define HEIGHT 450
#define IMAGE "../../common/img/halftone-face.jpg"

#define TYPE float
#define TYPE2 float

using namespace hipacc;
using namespace hipacc::math;

/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
  const int width = WIDTH;
  const int height = HEIGHT;

  // host memory for image of width x height pixels
  TYPE *input = load_data<TYPE>(width, height, 1, IMAGE);

  // std::cout << "Calculating Hipacc Gaussian filter ..." << std::endl;

  //************************************************************************//

  // input and output image of width x height pixels
  Image<TYPE> in(width, height, input);
  Image<TYPE> out_dct(width, height);
  Image<TYPE> out_fft(width, height);
  Image<TYPE> out_mag_dct(width, height);
  Image<TYPE> out_mag_fft(width, height);

  // write fft of in to fftResult
  float *fftResult = (float *)fft<TYPE, float>(in);
  // create magnitude from fft
  fftToMagnitude<TYPE>(fftResult, out_mag_fft);
  ifft<TYPE, float>(fftResult, out_fft);

  float *dctResult = (float *)dct<TYPE, float>(in);
  // create magnitude from fft
  dctToMagnitude<TYPE>(dctResult, out_mag_dct);
  idct<TYPE, float>(dctResult, out_dct);

  // get pointer to result data
  TYPE *output_dct = out_dct.data();
  TYPE *output_fft = out_fft.data();
  TYPE *output_mag_dct = out_mag_dct.data();
  TYPE *output_mag_fft = out_mag_fft.data();

  save_data(width, height, 1, input, "input.jpg");
  save_data(width, height, 1, output_dct, "output_dct.jpg");
  save_data(width, height, 1, output_fft, "output_fft.jpg");
  save_data(width, height, 1, output_mag_dct, "output_mag_dct.jpg");
  save_data(width, height, 1, output_mag_fft, "output_mag_fft.jpg");


  // manual transforms

  TYPE2 test[4][4] = {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}};
  TYPE2 fft[4][(4 / 2 + 1) * 2];
  TYPE2 dct[4][4];

  TYPE2 *test_d;
  TYPE2 *fft_d;
  TYPE2 *dct_d;

  /*cudaMalloc((void**)&test_d, sizeof(TYPE2) * 4*4);
  cudaMalloc((void**)&fft_d, sizeof(TYPE2) * 4*(4 / 2 + 1) * 2);
  cudaMalloc((void**)&dct_d, sizeof(TYPE2) * 4*4);
  cudaMemcpy(test_d, test, sizeof(TYPE2) * 4*4, cudaMemcpyHostToDevice);
  cudaMemcpy(fft_d, fft, sizeof(TYPE2) * 4*(4 / 2 + 1) * 2,
  cudaMemcpyHostToDevice); cudaMemcpy(dct_d, dct, sizeof(TYPE2) * 4*4,
  cudaMemcpyHostToDevice);

  fft_transform_device(test_d, 4, 4, fft_d);

  cudaMemcpy(fft, fft_d, sizeof(TYPE2) * 4*(4 / 2 + 1) * 2,
  cudaMemcpyDeviceToHost);

  for (int y = 0; y < 4; y++) {
    for (int x = 0; x < (4 / 2 + 1) * 2; x++) {
      std::cout << fft[y][x] << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  dct_transform_device((TYPE2*)test_d, 4, 4, (TYPE2*)dct_d);
  dct_transform_device((TYPE2*)dct_d, 4, 4, (TYPE2*)test_d, false);

  //cudaMemcpy(dct, dct_d, sizeof(TYPE2) * 4*4, cudaMemcpyDeviceToHost);
  cudaMemcpy(dct, test_d, sizeof(TYPE2) * 4*4, cudaMemcpyDeviceToHost);*/

  for (int y = 0; y < 4; y++) {
    for (int x = 0; x < 4; x++) {
      std::cout << dct[y][x] << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  // free memory
  delete[] input;

  return EXIT_SUCCESS;
}
