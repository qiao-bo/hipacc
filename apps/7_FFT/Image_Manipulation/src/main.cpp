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

#include <arpa/inet.h>
#include <fstream> // for png image size
#include <iostream>
#include <string>
#include <hipacc_helper.hpp>

#define WIDTH  4032
#define HEIGHT 3024
#define IMAGE  "../../common/img/fuerte_ship.png"


using namespace hipacc;
using namespace hipacc::math;


//  filter in Hipacc
/*class LBP : public Kernel<uchar> {
    private:
        Accessor<uchar> &input;
        Mask<int> &mask;

    public:
        LBP(IterationSpace<uchar> &iter, Accessor<uchar> &input,
                     Mask<int> &mask)
              : Kernel(iter), input(input), mask(mask) {
            add_accessor(&input);
        }

        void kernel() {
            int result = 0;
            for (int y = -1; y <= 1; ++y) {
                for (int x = -1; x <= 1; ++x) {
                    result += mask(x, y) * (input(x, y) > input(0, 0) ? 1 : 0);
                }
            }
            output() = (uchar) result;
            output() = (uchar)(convolve(mask, Reduce::SUM, [&] () -> float {
                     return mask() * (input(mask) > input(0, 0) ? 1 : 0);
                 }));
        }
};*/

class LBP : public Kernel<uchar> {
    private:
        Accessor<uchar> &input;
        Mask<int> &mask;

    public:
        LBP(IterationSpace<uchar> &iter, Accessor<uchar> &input,
                     Mask<int> &mask)
              : Kernel(iter), input(input), mask(mask) {
            add_accessor(&input);
        }

        void kernel() {
            output() = (uchar)(convolve(mask, Reduce::SUM, [&] () -> float {
                     return mask() * (input(mask) > input(0, 0) ? 1 : 0);
                 }));
        }
};

/*
class LTP : public Kernel<uchar4> {
    private:
        Accessor<uchar> &input;
        Mask<int> &mask;

    public:
        LTP(IterationSpace<uchar4> &iter, Accessor<uchar> &input,
                     Mask<int> &mask)
              : Kernel(iter), input(input), mask(mask) {
            add_accessor(&input);
        }

        void kernel() {
            int ltpu = 0;
            int ltpl = 0;
            const int th = 10;
            for (int y = -1; y <= 1; ++y) {
                for (int x = -1; x <= 1; ++x) {
                    if (input(x, y) >= input(0, 0) + th) {
                        ltpu += mask(x, y);
                    } else if (input(x, y) <= input(0, 0) - th) {
                        ltpl += mask(x, y);
                    }
                }
            }
            uchar4 result = (uchar4){(uchar)ltpu, (uchar)ltpl, 0, 0};
            output() = result;
        }
};
*/

// Kernel description in Hipacc
class ColorConversionY : public Kernel<uchar> {
    private:
        Accessor<uchar4> &in;

    public:
        ColorConversionY(IterationSpace<uchar> &iter, Accessor<uchar4> &acc)
              : Kernel(iter), in(acc) {
            add_accessor(&in);
        }

        void kernel() {
            uchar4 pixel = in();
            output() = .299f*pixel.x + .587f*pixel.y + .114f*pixel.z;
        }
};

// Kernel description in Hipacc
class ColorConversionCb : public Kernel<uchar> {
    private:
        Accessor<uchar4> &in;

    public:
        ColorConversionCb(IterationSpace<uchar> &iter, Accessor<uchar4> &acc)
              : Kernel(iter), in(acc) {
            add_accessor(&in);
        }

        void kernel() {
            uchar4 pixel = in();
            output() = -.168736f*pixel.x + -.331264f*pixel.y + .5f*pixel.z + 128;
        }
};

// Kernel description in Hipacc
class ColorConversionCr : public Kernel<uchar> {
    private:
        Accessor<uchar4> &in;

    public:
        ColorConversionCr(IterationSpace<uchar> &iter, Accessor<uchar4> &acc)
              : Kernel(iter), in(acc) {
            add_accessor(&in);
        }

        void kernel() {
            uchar4 pixel = in();
            output() = .5f*pixel.x + .418688f*pixel.y + -.081312f*pixel.z + 128;
        } 
};


/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    std::string imagePath = IMAGE;
    // get width and height for png format 
    const int width = WIDTH;
    const int height = HEIGHT;

    float timing = 0;

    // convolution filter mask
    const int weights[3][3] = {
        { 32,  64, 128 },
        { 16,   0,   1 },
        {  8,   4,   2 }
    };

    std::cout << "Calculating color Conversion filter ..." << std::endl;

    //************************************************************************// 


    // ColorConversion

    // host memory for image of width x height pixels
    uchar4 *input = (uchar4*)load_data<uchar>(width, height, 4, IMAGE);
    float *inputFloat = (float*)load_data<float>(width, height, 1, IMAGE);

    


    // input and output image of width x height pixels
    Image<uchar4> in(width, height, input);
    Image<uchar> outY(width, height);
    Image<uchar> outCb(width, height);
    Image<uchar> outCr(width, height);
    
    Accessor<uchar4> acc(in);

    IterationSpace<uchar> iterY(outY);
    ColorConversionY filterY(iterY, acc);

    IterationSpace<uchar> iterCb(outCb);
    ColorConversionCb filterCb(iterCb, acc);

    IterationSpace<uchar> iterCr(outCr);
    ColorConversionCr filterCr(iterCr, acc);

    filterY.execute();
    filterCb.execute();
    filterCr.execute();

    uchar *outputY = outY.data();
    uchar *outputCb = outCb.data();
    uchar *outputCr = outCr.data();

    save_data(width, height, 1, outputY, "outputY.png");
    save_data(width, height, 1, outputCb, "outputCb.png"); 
    save_data(width, height, 1, outputCr, "outputCr.png");


    // input and output image of width x height pixels
    Image<uchar> inLP(width, height, outputCr);
    Image<uchar> outLBP(width, height);
    //Image<uchar4> outLTP(width, height);
    
    // define Mask for filter
    Mask<int> mask(weights);

    BoundaryCondition<uchar> bound(inLP, mask, Boundary::CLAMP);
    Accessor<uchar> acc2(bound);

    IterationSpace<uchar> iterLBP(outLBP);
    LBP filterLBP(iterLBP, acc2, mask);

    std::cout << "Calculating LBP ..." << std::endl;
    filterLBP.execute();

    //IterationSpace<uchar4> iterLTP(outLTP);
    //LTP filterLTP(iterLTP, acc2, mask);

    //std::cout << "Calculating LTP ..." << std::endl;
    //filterLTP.execute();

    timing = hipacc_last_kernel_timing();

    Image<uchar> outDCT(width, height);

    float *dctResult = (float *)dct<uchar, float>(outLBP);
    dctToMagnitude<uchar>(dctResult, outDCT);


    // get pointer to result data
    uchar *output = outDCT.data();

    //************************************************************************//

    std::cout << "Hipacc: " << timing << " ms, "
              << (width*height/timing)/1000 << " Mpixel/s" << std::endl;

    save_data(width, height, 1, output, "outputDCT.png");
    //save_data(width, height, 4, outputLTP, "outputLTP.png");



    // free memory
    delete[] input;

    return EXIT_SUCCESS;
}

