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

#include <iostream>
#include <hipacc_helper.hpp>


#define WIDTH  512
#define HEIGHT 512
#define IMAGE1 "../../common/img/q5_00164.jpg"
#define IMAGE2 "../../common/img/q5_00165.jpg"

#define WINDOW_SIZE_X 32
#define WINDOW_SIZE_Y 32
#define EPSILON       16
#define TILE_SIZE     8
#define STRIDE_X      ((WIDTH+TILE_SIZE-1)/TILE_SIZE)
#define STRIDE_Y      ((HEIGHT+TILE_SIZE-1)/TILE_SIZE)


using namespace hipacc;
using namespace hipacc::math;


class GaussianBlur : public Kernel<uchar> {
    private:
        Accessor<uchar> &input;
        Mask<float> &mask;

    public:
        GaussianBlur(IterationSpace<uchar> &iter, Accessor<uchar> &input,
                     Mask<float> &mask)
              : Kernel(iter), input(input), mask(mask) {
            add_accessor(&input);
        }

        void kernel() {
            output() = (uchar)(convolve(mask, Reduce::SUM, [&] () -> float {
                    return mask() * input(mask);
                }) + 0.5f);
        }
};

class SignatureKernel : public Kernel<uint> {
    private:
        Accessor<uchar> &input;
        Domain &dom;

    public:
        SignatureKernel(IterationSpace<uint> &iter, Accessor<uchar> &input,
                        Domain &dom)
              : Kernel(iter), input(input), dom(dom) {
            add_accessor(&input);
        }

        void kernel() {
            // Census Transformation
            uchar z = input();
            uint c = 0u;
            iterate(dom, [&] () {
                    uchar data = input(dom);
                    if (data > z + EPSILON) {
                        c = (c << 2) | 0x01;
                    } else if (data < z - EPSILON) {
                        c = (c << 2) | 0x02;
                    } else {
                        c = c << 2;
                    }
                });

            output() = c;
        }
};

class VectorKernel : public Kernel<int, float4> {
    private:
        Accessor<uint> &sig1, &sig2;
        Domain &dom;

    public:
        VectorKernel(IterationSpace<int> &iter, Accessor<uint> &sig1,
                     Accessor<uint> &sig2, Domain &dom)
              : Kernel(iter), sig1(sig1), sig2(sig2), dom(dom) {
            add_accessor(&sig1);
            add_accessor(&sig2);
        }

        void kernel() {
            int vec_found = 0;
            int mem_loc = 0;

            uint reference = sig1();

            iterate(dom, [&] () -> void {
                    if (sig2(dom) == reference) {
                        // BUG: ++operator is not recognized as assignment
                        vec_found = vec_found + 1;
                        // encode x and y as upper and lower half-word
                        mem_loc = (dom.x() << 16) | (dom.y() & 0xffff);
                    }
                });

            // save the vector, if exactly one was found
            if (vec_found!=1) {
                mem_loc = 0;
            }

            output() = mem_loc;
        }

        void binning(uint x, uint y, int vector) {
            if (vector != 0) {
                float4 result = { 0.0f, 0.0f, 0.0f, 0.0f };

                // Cartesian to polar
                int xi = vector >> 16;
                int yi = (vector & 0xffff);
                if (yi >> 15) yi |= 0xffff0000;
                float xf = (float)xi;
                float yf = (float)yi;
                float dist = sqrt(xf*xf+yf*yf);
                float angle = atan2(yf,xf);

                result.x = dist;
                result.y = angle;
                result.w = 1.0f;

                // target tile is at midway (xi/2 & yi/2) of vector
                uint xt = (x+xi/2) / TILE_SIZE;
                uint yt = (y+yi/2) / TILE_SIZE;
                bin((uint)(yt * STRIDE_X + xt)) = result;
            }
        }

        float4 reduce(float4 left, float4 right) const {
            if (left.w == 0.0f) {
                return right;
            } else if (right.w == 0.0f) {
                return left;
            } else {
                // average vectors
                float ws = left.w + right.w;
                float wl = left.w/ws;
                float wr = 1.0f-wl;
                float4 result = wl*left + wr*right;
                result.w = ws;
                return result;
            }
        }
};


class Assemble : public Kernel<uchar> {
    private:
        Accessor<uchar> &input;
        Accessor<float4> &vecs;

    public:
        Assemble(IterationSpace<uchar> &iter, Accessor<uchar> &input,
                 Accessor<float4> &vecs)
              : Kernel(iter), input(input), vecs(vecs) {
            add_accessor(&input);
            add_accessor(&vecs);
        }

        void kernel() {
            int x = 0;
            int y = 0;
            float4 vector = vecs();

            if (vector.w != 0.0f) {
                // polar to Cartesian
                float xf = vector.x * cosf(vector.y);
                float yf = vector.x * sinf(vector.y);

                // half distance and opposite direction
                xf *= -.5f;
                yf *= -.5f;

                // correct rounding
                x = (int)(xf + (xf < 0 ? -.5f : .5f));
                y = (int)(yf + (yf < 0 ? -.5f : .5f));
            }

            output() = input(x, y);
        }
};


/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    const int width = WIDTH;
    const int height = HEIGHT;
    float timing = 0;

    // filter mask
    const float filter_mask[3][3] = {
        { 0.057118f, 0.124758f, 0.057118f },
        { 0.124758f, 0.272496f, 0.124758f },
        { 0.057118f, 0.124758f, 0.057118f }
    };

    // domain for signature kernel
    const uchar sig_coef[9][9] = {
        { 1, 0, 0, 0, 1, 0, 0, 0, 1 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 1, 0, 1, 0, 1, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 1, 0, 0, 0, 1, 0, 0, 0, 1 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 1, 0, 1, 0, 1, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 1, 0, 0, 0, 1, 0, 0, 0, 1 }
    };

    // host memory for image of width x height pixels
    uchar *input1 = load_data<uchar>(width, height, 1, IMAGE1);
    uchar *input2 = load_data<uchar>(width, height, 1, IMAGE2);

    std::cout << "Calculating Hipacc motion interpolation ..." << std::endl;

    //************************************************************************//

    // input and output image of width x height pixels
    Image<uchar> in1(width, height, input1);
    Image<uchar> in2(width, height, input2);
    Image<uchar> tmp(width, height);
    Image<uint> in1_sig(width, height);
    Image<uint> in2_sig(width, height);
    Image<uchar> out(width, height);
    Image<int> img_vec(width, height);
    Image<float4> merged_vec(STRIDE_X, STRIDE_Y);

    // define Mask for Gaussian blur filter
    Mask<float> mask(filter_mask);

    // define Domain for signature kernel
    Domain sig_dom(sig_coef);

    // Domain for vector kernel
    Domain dom(WINDOW_SIZE_X/2, WINDOW_SIZE_Y/2);
    // do not process the center pixel
    dom(0,0) = 0;

    // filter first image
    BoundaryCondition<uchar> bound_in1(in1, mask, Boundary::CLAMP);
    Accessor<uchar> acc_in1(bound_in1);
    IterationSpace<uchar> iter_blur(tmp);
    GaussianBlur blur1(iter_blur, acc_in1, mask);
    blur1.execute();
    timing += hipacc_last_kernel_timing();

    // generate signature for first image
    BoundaryCondition<uchar> bound_tmp(tmp, sig_dom, Boundary::CLAMP);
    Accessor<uchar> acc_tmp(bound_tmp);
    IterationSpace<uint> iter_in1_sig(in1_sig);
    SignatureKernel sig1(iter_in1_sig, acc_tmp, sig_dom);
    sig1.execute();
    timing += hipacc_last_kernel_timing();

    // filter second image
    BoundaryCondition<uchar> bound_in2(in2, mask, Boundary::CLAMP);
    Accessor<uchar> acc_in2(bound_in2);
    GaussianBlur blur2(iter_blur, acc_in2, mask);
    blur2.execute();
    timing += hipacc_last_kernel_timing();

    // generate signature for second image
    IterationSpace<uint> iter_in2_sig(in2_sig);
    SignatureKernel sig2(iter_in2_sig, acc_tmp, sig_dom);
    sig2.execute();
    timing += hipacc_last_kernel_timing();

    // compute motion vectors
    BoundaryCondition<uint> bound_in2_sig(in2_sig, dom, Boundary::CONSTANT, 0);
    Accessor<uint> acc_in2_sig(bound_in2_sig);
    BoundaryCondition<uint> bound_in1_sig(in1_sig, dom, Boundary::CONSTANT, 0);
    Accessor<uint> acc_in1_sig(bound_in1_sig);
    IterationSpace<int> iter_vec(img_vec);
    VectorKernel vector_kernel(iter_vec, acc_in1_sig, acc_in2_sig, dom);
    vector_kernel.execute();
    timing += hipacc_last_kernel_timing();

    // merged vectors
    float4* vecs = vector_kernel.binned_data(STRIDE_X*STRIDE_Y);
    timing += hipacc_last_kernel_timing();

    // load vectors into image 'merged_vec'
    merged_vec = vecs;

    // assemble final image
    IterationSpace<uchar> iter_out(out);
    Accessor<float4> acc_merged_vec(merged_vec, Interpolate::NN);
    BoundaryCondition<uchar> bound_asm_in1(in1, dom, Boundary::CLAMP);
    Accessor<uchar> acc_asm_in1(bound_asm_in1);
    Assemble assemble(iter_out, acc_in1, acc_merged_vec);
    assemble.execute();
    timing += hipacc_last_kernel_timing();

    uchar *output = out.data();

    //************************************************************************//

    std::cout << "Hipacc: " << timing << " ms, "
              << (width*height/timing)/1000 << " Mpixel/s" << std::endl;

    save_data(width, height, 1, input1, "frame1.jpg");
    save_data(width, height, 1, output, "frame2.jpg");
    save_data(width, height, 1, input2, "frame3.jpg");
    show_data(width, height, 1, output, "frame2.jpg");

    // free memory
    delete[] input1;
    delete[] input2;

    return EXIT_SUCCESS;
}
