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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <hipacc_helper.hpp>


#define WIDTH  4032
#define HEIGHT 3024
#define IMAGE  "../../common/img/fuerte_ship.jpg"

#define BILIN
#define PACK_INT

#ifdef PACK_INT
# define data_t uint
# define pack(x, y, z, w) \
    (uint)((uint)(x) | (uint)(y) << 8 | (uint)(z) << 16 | (uint)(w) << 24)
# define unpack(val, byte) \
    (((val) >> (byte*8)) & 0xff)
#else
# define data_t uchar4
# define pack(x, y, z, w) \
    ((uchar4){x, y, z, w})
# define unpack(val, byte) \
    (val.s ## byte)
#endif


using namespace hipacc;


class Chrome : public Kernel<data_t> {
    private:
        Accessor<data_t> &input;
        int width;
        int height;
        float rindex;
        float gindex;

    public:
        Chrome(IterationSpace<data_t> &iter, Accessor<data_t> &input,
               int width, int height, float rindex, float gindex)
              : Kernel(iter), input(input), width(width), height(height),
                rindex(rindex), gindex(gindex) {
            add_accessor(&input);
        }

        void kernel() {
            int gposx = x();
            int gposy = y();
            float xpos = gposx - width/2;
            float ypos = gposy - height/2;

            float rout = 0;
            float gout = 0;
            float bout = 0;
            
            { /* red */
              float xshift = (xpos * rindex) + width/2;
              float yshift = (ypos * rindex) + height/2;

              int bx = (int)xshift-gposx;
              int by = (int)yshift-gposy;
#ifdef BILIN
              // bilinear filtering
              float xp = xshift-(uint)xshift;
              float yp = yshift-(uint)yshift;

              float val1 = unpack(input(bx  , by  ), 2);
              float val2 = unpack(input(bx+1, by  ), 2);
              float val3 = unpack(input(bx  , by+1), 2);
              float val4 = unpack(input(bx+1, by+1), 2);

              rout = ((1.0f-yp) * (((1.0f-xp) * val1) + (xp * val2)))
                     +      (yp * (((1.0f-xp) * val3) + (xp * val4)));
#else

              rout = unpack(input(bx, by), 2);
#endif
            }

            { /* green */
              float xshift = (xpos * gindex) + width/2;
              float yshift = (ypos * gindex) + height/2;

              int bx = (int)xshift-gposx;
              int by = (int)yshift-gposy;

#ifdef BILIN
              // bilinear filtering
              float xp = xshift-(uint)xshift;
              float yp = yshift-(uint)yshift;

              float val1 = unpack(input(bx  , by  ), 1);
              float val2 = unpack(input(bx+1, by  ), 1);
              float val3 = unpack(input(bx  , by+1), 1);
              float val4 = unpack(input(bx+1, by+1), 1);

              gout = ((1.0f-yp) * (((1.0f-xp) * val1) + (xp * val2)))
                     +      (yp * (((1.0f-xp) * val3) + (xp * val4)));
#else
              gout = unpack(input(bx, by), 1);
#endif
            }

            { /* blue */
              bout = unpack(input(), 0);
            }

            output() = pack(bout, gout, rout, 255);
        }
};


/*************************************************************************
 * Main function                                                         *
 *************************************************************************/
int main(int argc, const char **argv) {
    float rindex = 0.9975f;
    float gindex = 0.995f;

    if (argc > 2) {
      rindex = atof(argv[1]);
      gindex = atof(argv[2]);
    }

    const int width = WIDTH;
    const int height = HEIGHT;
    data_t *input = (data_t*)load_data<uchar>(width, height, 4, IMAGE);

    //************************************************************************//

    Image<data_t> in(width, height, input);
    Image<data_t> out(width, height);

    IterationSpace<data_t> iter(out);

    Accessor<data_t> acc_at(in);

    Chrome chrome(iter, acc_at, width, height, rindex, gindex);
    chrome.execute();
    float timing = hipacc_last_kernel_timing();

    data_t *output = out.data(); 

    //************************************************************************//

    store_data(width, height, 4, (uchar*)input, "input.jpg");
    store_data(width, height, 4, (uchar*)output, "output.jpg");

    std::cout << "<HIPACC:> Overall time: " << timing << "(ms)" << std::endl;

    delete[] input;

    return EXIT_SUCCESS;
}

