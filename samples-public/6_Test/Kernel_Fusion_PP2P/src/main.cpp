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

#define WIDTH 512
#define HEIGHT 512
#define TYPE uchar

using namespace hipacc;
using namespace hipacc::math;

// Kernel description in Hipacc
class PointOperatorExample : public Kernel<TYPE> {
    private:
        Accessor<TYPE> &in;

    public:
        PointOperatorExample(IterationSpace<TYPE> &iter, Accessor<TYPE> &acc)
              : Kernel(iter), in(acc) {
            add_accessor(&in);
        }

        void kernel() {
            TYPE interm_pixel = in();
            interm_pixel += 3;
            output() = interm_pixel;
        }
};

class OutOperatorExample : public Kernel<TYPE> {
    private:
        Accessor<TYPE> &in1;
        Accessor<TYPE> &in2;

    public:
        OutOperatorExample(IterationSpace<TYPE> &iter, Accessor<TYPE> &acc1, Accessor<TYPE> &acc2)
              : Kernel(iter), in1(acc1), in2(acc2) {
            add_accessor(&in1);
            add_accessor(&in2);
        }

        void kernel() {
            TYPE interm_pixel1 = in1();
            TYPE interm_pixel2 = in2();
            output() = interm_pixel1 + interm_pixel2;
        }
};

// forward declaration of reference implementation
void kernel_fusion(TYPE *in, TYPE *out, int width, int height);

/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
HIPACC_CODEGEN int main(int argc, const char **argv) {
    int width_arg = WIDTH;
    int height_arg = HEIGHT;

    if(argc >= 2) {
        width_arg = std::stoi(argv[1]);
        height_arg = width_arg;
    }

    if(argc >= 3) {
        height_arg = std::stoi(argv[2]);
    }

    const int width = width_arg;
    const int height = height_arg;

    // host memory for image of width x height pixels, random
    TYPE *input = (TYPE*)load_data<TYPE>(width, height);
    TYPE *ref_out = new TYPE[width*height];

    std::cout << "Testing Hipacc kernel fusion ..." << std::endl;

    //************************************************************************//

    // input and output image of width x height pixels
    Image<TYPE> in(width, height, input);
    Image<TYPE> out(width, height);

    // test parallelpoint-point to point kernel fusion
    Accessor<TYPE> acc0(in);

    Image<TYPE> buf0(width, height);
    IterationSpace<TYPE> iter0(buf0);
    PointOperatorExample pointOp0(iter0, acc0);

    Image<TYPE> buf1(width, height);
    IterationSpace<TYPE> iter1(buf1);
    PointOperatorExample pointOp1(iter1, acc0);

    Accessor<TYPE> acc1(buf0);
    Accessor<TYPE> acc2(buf1);
    IterationSpace<TYPE> iter2(out);
    OutOperatorExample outOp(iter2, acc1, acc2);

    // execution after all decls
    pointOp0.execute();
    pointOp1.execute();
    outOp.execute();

    // get pointer to result data
    TYPE *output = out.data();

    //************************************************************************//
    std::cout << "Calculating reference ..." << std::endl;
    kernel_fusion(input, ref_out, width, height);
    compare_results(output, ref_out, width, height);

    // free memory
    delete[] input;
    delete[] ref_out;

    return EXIT_SUCCESS;
}

// kernel fusion reference
void point_kernel(TYPE *in, TYPE *out, int width, int height) {
    for (int p = 0; p < width*height; ++p) {
        TYPE interm_pixel = in[p];
        interm_pixel += 3;
        out[p] = interm_pixel;
    }
}

void out_kernel(TYPE *in1, TYPE *in2, TYPE *out, int width, int height) {
    for (int p = 0; p < width*height; ++p) {
        TYPE interm_pixel1 = in1[p];
        TYPE interm_pixel2 = in2[p];
        out[p] = interm_pixel1 + interm_pixel2;
    }
}

void kernel_fusion(TYPE *in, TYPE *out, int width, int height) {
    TYPE *ref_buf0 = new TYPE[width*height];
    TYPE *ref_buf1 = new TYPE[width*height];

    // left operator
    point_kernel(in, ref_buf0, width, height);

    // right operator
    point_kernel(in, ref_buf1, width, height);

    // out operator
    out_kernel(ref_buf0, ref_buf1, out, width, height);

    delete[] ref_buf0;
    delete[] ref_buf1;
}
