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
#include <vector>
#include <hipacc_helper.hpp>

#define WIDTH 512
#define HEIGHT 512
#define N_CONTROL_POINTS 4
#define MIN_VALUE static_cast<uchar>(0)
#define MAX_VALUE static_cast<uchar>(255)

#define CHANNEL_R 0
#define CHANNEL_G 1
#define CHANNEL_B 2

using namespace hipacc;
using namespace hipacc::math;

// Kernel description in Hipacc
class ColorCurves : public Kernel<uchar> {
    private:
        Accessor<uchar4> &in;
		int channel;
		uchar cp_y1, cp_y2, cp_y3, cp_y4;

    public:
        ColorCurves(IterationSpace<uchar> &iter, Accessor<uchar4> &acc, int channel, uchar cp_y1, uchar cp_y2, uchar cp_y3, uchar cp_y4)
              : Kernel(iter), in(acc), channel(channel), cp_y1(cp_y1), cp_y2(cp_y2), cp_y3(cp_y3), cp_y4(cp_y4) {
            add_accessor(&in);
        }

        void kernel() {
            uchar4 interm_pixel = in();
			int x = static_cast<int>(interm_pixel.w);
			if (channel == CHANNEL_R) {
				x = static_cast<int>(interm_pixel.x);
			} else if (channel == CHANNEL_G) {
				x = static_cast<int>(interm_pixel.y);
			} else if (channel == CHANNEL_B) {
				x = static_cast<int>(interm_pixel.z);
			}

			const int h = (static_cast<int>(MAX_VALUE) - static_cast<int>(MIN_VALUE)) / (static_cast<int>(N_CONTROL_POINTS) - 1);

			const int y1 = static_cast<int>(cp_y1);
			const int y2 = static_cast<int>(cp_y2);
			const int y3 = static_cast<int>(cp_y3);
			const int y4 = static_cast<int>(cp_y4);

			const int a = (6/(h*h)) * (-2 * y2 + y3);
			const int b = (6/(h*h)) * (y2 - 2 * y3); 

			const int m1 = 0;
			const int m2 = (4 * a - b) / 15;
			const int m3 = (-a + 4 * b) / 15;
			const int m4 = 0;

			const int a1 = (m2-m1) / (6*h);
			const int a2 = (m3-m2) / (6*h);
			const int a3 = (m4-m3) / (6*h);
			const int a4 = 0;

			const int b1 = m1 / 2;
			const int b2 = m2 / 2;
			const int b3 = m3 / 2;
			const int b4 = m4 / 2;

			const int c1 = (y2 - y1) / h - ((m2 + 2*m1) / 6) * h;
			const int c2 = (y3 - y2) / h - ((m3 + 2*m2) / 6) * h;
			const int c3 = (y4 - y3) / h - ((m4 + 2*m3) / 6) * h;
			const int c4 = 0;
			
			const int d1 = y1;
			const int d2 = y2;
			const int d3 = y3;
			const int d4 = y4;
			
			int ai = a1;
			int bi = b1;
			int ci = c1;
			int di = d1;

			if (x < h) {
				// nothing to do here
			} else if (x < 2 * h) {
				ai = a2;
				bi = b2;
				ci = c2;
				di = d2;
				x -= h;
			} else if (x < 3 * h) {
				ai = a3;
				bi = b3;
				ci = c3;
				di = d3;
				x -= 2 * h;
			} else {
				ai = a4;
				bi = b4;
				ci = c4;
				di = d4;
				x -= 3 * h;
			}

			const int x3 = x*x*x;
			const int x2 = x*x;

			output() = static_cast<uchar>(ai*x3 + bi*x2 + ci*x + di);
        }
};

// Kernel description in Hipacc
class CombineChannels : public Kernel<uchar4> {
    private:
        Accessor<uchar>& in_r;
		Accessor<uchar>& in_g;
		Accessor<uchar>& in_b;

    public:
        CombineChannels(IterationSpace<uchar4>& iter, Accessor<uchar>& acc_r, Accessor<uchar>& acc_g, Accessor<uchar>& acc_b)
              : Kernel(iter), in_r(acc_r), in_g(acc_g), in_b(acc_b) {
            add_accessor(&in_r);
			add_accessor(&in_g);
			add_accessor(&in_b);
        }

        void kernel() {
			uchar pixel_r = in_r();
			uchar pixel_g = in_g();
			uchar pixel_b = in_b();

			uchar4 out;
			out.x = pixel_r;
			out.y = pixel_g;
			out.z = pixel_b;
			out.w = MAX_VALUE;
			
			output() = out;
		}
};

// forward declaration of reference implementation
void kernel_fusion(uchar4 *in, uchar4 *out, int width, int height);

void compare_color(uchar4* output, uchar4* ref_out, int width, int height) {
	uchar *cmp_output_r = new uchar[width*height];
	uchar *cmp_output_g = new uchar[width*height];
	uchar *cmp_output_b = new uchar[width*height];
	uchar *cmp_output_a = new uchar[width*height];
	uchar *cmp_ref_out_r = new uchar[width*height];
	uchar *cmp_ref_out_g = new uchar[width*height];
	uchar *cmp_ref_out_b = new uchar[width*height];
	uchar *cmp_ref_out_a = new uchar[width*height];

	for (int i = 0; i < width*height; ++i) {
		cmp_output_r[i] = output[i].x;
		cmp_output_g[i] = output[i].y;
		cmp_output_b[i] = output[i].z;
		cmp_output_a[i] = output[i].w;
		cmp_ref_out_r[i] = ref_out[i].x;
		cmp_ref_out_g[i] = ref_out[i].y;
		cmp_ref_out_b[i] = ref_out[i].z;
		cmp_ref_out_a[i] = ref_out[i].w;
	}

    compare_results(cmp_output_r, cmp_ref_out_r, width, height);
    compare_results(cmp_output_g, cmp_ref_out_g, width, height);
    compare_results(cmp_output_b, cmp_ref_out_b, width, height);
    compare_results(cmp_output_a, cmp_ref_out_a, width, height);

	delete[] cmp_output_r;
	delete[] cmp_output_g;
	delete[] cmp_output_b;
	delete[] cmp_output_a;
	delete[] cmp_ref_out_r;
	delete[] cmp_ref_out_g;
	delete[] cmp_ref_out_b;
	delete[] cmp_ref_out_a;
}

void diff_color(uchar4* img1, uchar4* img2, int width, int height) {
	for (int i = 0; i < width*height; ++i) {
		uchar4 c1 = img1[i];
		uchar4 c2 = img2[i];

		const bool eq_r = c1.x == c2.x;
		const bool eq_g = c1.y == c2.y;
		const bool eq_b = c1.z == c2.z;
		const bool eq_a = c1.w == c2.w;

		const bool eq_all = eq_r && eq_g && eq_b && eq_a;
		
		if (eq_all) {
			std::cout << "[" << i << "] ";
			std::cout << "(" << c1.x << "," << c1.y << "," << c1.z << "," << c1.w << ")";
			std::cout << " <=> ";
			std::cout << "(" << c2.x << "," << c2.y << "," << c2.z << "," << c2.w << ")";
			std::cout << std::endl;
		}
	}
}

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
    uchar4 *input = (uchar4*)load_data<uchar>(width, height, 4);
    uchar4 *ref_out = new uchar4[width*height];

    std::cout << "Testing Hipacc kernel fusion ..." << std::endl;

    //************************************************************************//

    // input and output image of width x height pixels
    Image<uchar4> in(width, height, input);
    Image<uchar4> out(width, height);

    Accessor<uchar4> acc_in(in);

	// ****
	// * Red channel
    // ****
    Image<uchar> channel_r(width, height);
    IterationSpace<uchar> iter_channel_r(channel_r);
	Accessor<uchar> acc_channel_r(channel_r);

	ColorCurves curves_channel_r(iter_channel_r, acc_in, CHANNEL_R, 10, 50, 230, 20);

	// ****
	// * Green channel
    // ****
    Image<uchar> channel_g(width, height);
    IterationSpace<uchar> iter_channel_g(channel_g);
	Accessor<uchar> acc_channel_g(channel_g);

	ColorCurves curves_channel_g(iter_channel_g, acc_in, CHANNEL_G, 10, 50, 230, 20);

	// ****
	// * Blue channel
    // ****
    Image<uchar> channel_b(width, height);
    IterationSpace<uchar> iter_channel_b(channel_b);
	Accessor<uchar> acc_channel_b(channel_b);

	ColorCurves curves_channel_b(iter_channel_b, acc_in, CHANNEL_B, 10, 50, 230, 20);

	// ****
	// * Combining operator
    // ****
    IterationSpace<uchar4> iter_out(out);

	CombineChannels combine(iter_out, acc_channel_r, acc_channel_g, acc_channel_b);

    // execution after all decls
   	curves_channel_r.execute();
	curves_channel_g.execute();
	curves_channel_b.execute();
	combine.execute();

    // get pointer to result data
    uchar4 *output = out.data();

    //************************************************************************//
    std::cout << "Calculating reference ..." << std::endl;
    kernel_fusion(input, ref_out, width, height);

	compare_color(output, ref_out, width, height);	
	//diff_color(output, ref_out, width, height);

    // free memory
    delete[] input;
    delete[] ref_out;

    return EXIT_SUCCESS;
}

// kernel fusion reference
void color_curves_kernel(uchar4 *in, uchar *out, int width, int height, int channel, uchar cp_y1, uchar cp_y2, uchar cp_y3, uchar cp_y4) {
    for (int p = 0; p < width*height; ++p) {
        uchar4 interm_pixel = in[p];
		int x = static_cast<int>(interm_pixel.w);
		if (channel == CHANNEL_R) {
			x = static_cast<int>(interm_pixel.x);
		} else if (channel == CHANNEL_G) {
			x = static_cast<int>(interm_pixel.y);
		} else if (channel == CHANNEL_B) {
			x = static_cast<int>(interm_pixel.z);
		}
        
		const int h = (static_cast<int>(MAX_VALUE) - static_cast<int>(MIN_VALUE)) / (static_cast<int>(N_CONTROL_POINTS) - 1);

		const int y1 = static_cast<int>(cp_y1);
		const int y2 = static_cast<int>(cp_y2);
		const int y3 = static_cast<int>(cp_y3);
		const int y4 = static_cast<int>(cp_y4);

		const int a = (6/(h*h)) * (-2 * y2 + y3);
		const int b = (6/(h*h)) * (y2 - 2 * y3); 

		const int m1 = 0;
		const int m2 = (4 * a - b) / 15;
		const int m3 = (-a + 4 * b) / 15;
		const int m4 = 0;

		const int a1 = (m2-m1) / (6*h);
		const int a2 = (m3-m2) / (6*h);
		const int a3 = (m4-m3) / (6*h);
		const int a4 = 0;

		const int b1 = m1 / 2;
		const int b2 = m2 / 2;
		const int b3 = m3 / 2;
		const int b4 = m4 / 2;

		const int c1 = (y2 - y1) / h - ((m2 + 2*m1) / 6) * h;
		const int c2 = (y3 - y2) / h - ((m3 + 2*m2) / 6) * h;
		const int c3 = (y4 - y3) / h - ((m4 + 2*m3) / 6) * h;
		const int c4 = 0;
		
		const int d1 = y1;
		const int d2 = y2;
		const int d3 = y3;
		const int d4 = y4;
		
		int ai = a1;
		int bi = b1;
		int ci = c1;
		int di = d1;

		if (x < h) {
			// nothing to do here
		} else if (x < 2 * h) {
			ai = a2;
			bi = b2;
			ci = c2;
			di = d2;
			x -= h;
		} else if (x < 3 * h) {
			ai = a3;
			bi = b3;
			ci = c3;
			di = d3;
			x -= 2 * h;
		} else {
			ai = a4;
			bi = b4;
			ci = c4;
			di = d4;
			x -= 3 * h;
		}

		const int x3 = x*x*x;
		const int x2 = x*x;

		out[p] = static_cast<uchar>(ai*x3 + bi*x2 + ci*x + di);
    }
}

void combine_channels_kernel(uchar* in_r, uchar* in_g, uchar* in_b, uchar4* out, int width, int height) {
	for (int p = 0; p < width*height; ++p) {
		uchar pixel_r = in_r[p];
		uchar pixel_g = in_g[p];
		uchar pixel_b = in_b[p];

		uchar4 pixel_out;
		pixel_out.x = pixel_r;
		pixel_out.y = pixel_g;
		pixel_out.z = pixel_b;
		pixel_out.w = MAX_VALUE;

		out[p] = pixel_out;
	}
}

void kernel_fusion(uchar4 *in, uchar4 *out, int width, int height) {
 	uchar *ref_buf_r = new uchar[width*height];
 	uchar *ref_buf_g = new uchar[width*height];
 	uchar *ref_buf_b = new uchar[width*height];

 	color_curves_kernel(in, ref_buf_r, width, height, CHANNEL_R, 10, 50, 230, 20);
	color_curves_kernel(in, ref_buf_g, width, height, CHANNEL_G, 10, 50, 230, 20);
	color_curves_kernel(in, ref_buf_b, width, height, CHANNEL_B, 10, 50, 230, 20);
	combine_channels_kernel(ref_buf_r, ref_buf_g, ref_buf_b, out, width, height);

	delete[] ref_buf_r;
	delete[] ref_buf_g;
	delete[] ref_buf_b;
}

