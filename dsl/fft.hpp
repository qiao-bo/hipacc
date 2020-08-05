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

#ifndef __FFT_HPP__
#define __FFT_HPP__

namespace hipacc {

template <class TPrecision>
void fftTransform(TPrecision *in, int width, int height, TPrecision *out,
                  bool forward = true, bool scale = false){

};
template <class TPrecision>
void fftTransformDevice(TPrecision *in, int width, int height, TPrecision *out,
                        bool forward = true, bool scale = false){

};
template <class TPrecision>
void dctTransform(TPrecision *in, int width, int height, TPrecision *out,
                  bool forward = true){

};
template <class TPrecision>
void dctTransformDevice(TPrecision *in, int width, int height, TPrecision *out,
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
void fftToMagnitude(TPrecision *in, Image<T> &mag) {
  static_assert(std::is_same<T, data_t>::value,
                "Type of Image and Memory have to be the same!");
};
template <class data_t, class TPrecision, class T>
void dctToMagnitude(TPrecision *in, Image<T> &mag) {
  static_assert(std::is_same<T, data_t>::value,
                "Type of Image and Memory have to be the same!");
};
template <class data_t, class TPrecision, class T>
void fftScaleMagnitude(TPrecision *in, Image<T> &mag) {
  static_assert(std::is_same<T, data_t>::value,
                "Type of Image and Memory have to be the same!");
};
template <class data_t, class T>
void fftResetMask(Image<T> &mag, int radius, bool low, int window = 0) {
  static_assert(std::is_same<T, data_t>::value,
                "Type of Image and Memory have to be the same!");
};
template <class data_t, class T>
void fftApplyPassFilter(Image<T> &mag, int radius, bool low, int window = 0) {
  static_assert(std::is_same<T, data_t>::value,
                "Type of Image and Memory have to be the same!");
};
template <class data_t, class T> void fftShift(Image<T> &mag) {
  static_assert(std::is_same<T, data_t>::value,
                "Type of Image and Memory have to be the same!");
};
template <class data_t, class T> void ifftShift(Image<T> &mag) {
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

}

#endif // __FFT_HPP__