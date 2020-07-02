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

#ifndef __HIPACC_HPP__
#define __HIPACC_HPP__

#ifdef _WIN32
# define NOMINMAX
#endif

// include all other Hipacc headers
#include "types.hpp"
#include "math_functions.hpp"
#include "image.hpp"
#include "iterationspace.hpp"
#include "kernel.hpp"
#include "mask.hpp"
#include "pyramid.hpp"

namespace hipacc {

template <class TPrecision>
void fft_transform(TPrecision *in, int width, int height, TPrecision *out,
                     bool forward = true, bool scale = false) {

};
template <class TPrecision>
void fft_transform_device(TPrecision *in, int width, int height, TPrecision *out,
                          bool forward = true, bool scale = false){

};
template <class TPrecision>
void dct_transform(TPrecision *in, int width, int height, TPrecision *out,
                   bool forward = true){

};
template <class data_t, class TPrecision, class T> TPrecision *fft(Image<T> &in) {
  static_assert(std::is_same<T, data_t>::value,
                "Type of Image and Memory have to be the same!");
  return nullptr;
};
template <class data_t, class TPrecision, class T>
void ifft(TPrecision *in, Image<T> &out) {
  static_assert(std::is_same<T, data_t>::value,
                "Type of Image and Memory have to be the same!");
};
template <class data_t, class TPrecision, class T>
void fftToMag(TPrecision *in, Image<T> &mag) {
  static_assert(std::is_same<T, data_t>::value,
                "Type of Image and Memory have to be the same!");
};
template <class data_t, class TPrecision, class T>
void dctToMag(TPrecision *in, Image<T> &mag) {
  static_assert(std::is_same<T, data_t>::value,
                "Type of Image and Memory have to be the same!");
};
template <class data_t, class TPrecision, class T>
void magScaleFFT(TPrecision *in, Image<T> &mag, float r) {
  static_assert(std::is_same<T, data_t>::value,
                "Type of Image and Memory have to be the same!");
};

template <class data_t, class TPrecision, class T> TPrecision *dct(Image<T> &in) {
  static_assert(std::is_same<T, data_t>::value,
                "Type of Image and Memory have to be the same!");
  return nullptr;
};
template <class data_t, class TPrecision, class T>
void idct(TPrecision *in, Image<T> &out) {
  static_assert(std::is_same<T, data_t>::value,
                "Type of Image and Memory have to be the same!");
};

class HipaccEoP{};
} // end namespace hipacc

#endif // __HIPACC_HPP__

