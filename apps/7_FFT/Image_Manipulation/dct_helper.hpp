#ifndef DCT_HELPER_HPP
#define DCT_HELPER_HPP

#include <iostream>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <vector>

// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
*  The dimension of pixels block
*/
#define BLOCK_SIZE          8


/**
*  Square of dimension of pixels block
*/
#define BLOCK_SIZE2         64

/**
*  This unitary matrix performs DCT of rows of the matrix to the left
*/
const float DCTv8matrix[BLOCK_SIZE2] =
{
    0.3535533905932738f,  0.4903926402016152f,  0.4619397662556434f,  0.4157348061512726f,  0.3535533905932738f,  0.2777851165098011f,  0.1913417161825449f,  0.0975451610080642f,
    0.3535533905932738f,  0.4157348061512726f,  0.1913417161825449f, -0.0975451610080641f, -0.3535533905932737f, -0.4903926402016152f, -0.4619397662556434f, -0.2777851165098011f,
    0.3535533905932738f,  0.2777851165098011f, -0.1913417161825449f, -0.4903926402016152f, -0.3535533905932738f,  0.0975451610080642f,  0.4619397662556433f,  0.4157348061512727f,
    0.3535533905932738f,  0.0975451610080642f, -0.4619397662556434f, -0.2777851165098011f,  0.3535533905932737f,  0.4157348061512727f, -0.1913417161825450f, -0.4903926402016153f,
    0.3535533905932738f, -0.0975451610080641f, -0.4619397662556434f,  0.2777851165098009f,  0.3535533905932738f, -0.4157348061512726f, -0.1913417161825453f,  0.4903926402016152f,
    0.3535533905932738f, -0.2777851165098010f, -0.1913417161825452f,  0.4903926402016153f, -0.3535533905932733f, -0.0975451610080649f,  0.4619397662556437f, -0.4157348061512720f,
    0.3535533905932738f, -0.4157348061512727f,  0.1913417161825450f,  0.0975451610080640f, -0.3535533905932736f,  0.4903926402016152f, -0.4619397662556435f,  0.2777851165098022f,
    0.3535533905932738f, -0.4903926402016152f,  0.4619397662556433f, -0.4157348061512721f,  0.3535533905932733f, -0.2777851165098008f,  0.1913417161825431f, -0.0975451610080625f
};


/**
*  This unitary matrix performs DCT of columns of the matrix to the right
*/
const float DCTv8matrixT[BLOCK_SIZE2] =
{
    0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f,
    0.4903926402016152f,  0.4157348061512726f,  0.2777851165098011f,  0.0975451610080642f, -0.0975451610080641f, -0.2777851165098010f, -0.4157348061512727f, -0.4903926402016152f,
    0.4619397662556434f,  0.1913417161825449f, -0.1913417161825449f, -0.4619397662556434f, -0.4619397662556434f, -0.1913417161825452f,  0.1913417161825450f,  0.4619397662556433f,
    0.4157348061512726f, -0.0975451610080641f, -0.4903926402016152f, -0.2777851165098011f,  0.2777851165098009f,  0.4903926402016153f,  0.0975451610080640f, -0.4157348061512721f,
    0.3535533905932738f, -0.3535533905932737f, -0.3535533905932738f,  0.3535533905932737f,  0.3535533905932738f, -0.3535533905932733f, -0.3535533905932736f,  0.3535533905932733f,
    0.2777851165098011f, -0.4903926402016152f,  0.0975451610080642f,  0.4157348061512727f, -0.4157348061512726f, -0.0975451610080649f,  0.4903926402016152f, -0.2777851165098008f,
    0.1913417161825449f, -0.4619397662556434f,  0.4619397662556433f, -0.1913417161825450f, -0.1913417161825453f,  0.4619397662556437f, -0.4619397662556435f,  0.1913417161825431f,
    0.0975451610080642f, -0.2777851165098011f,  0.4157348061512727f, -0.4903926402016153f,  0.4903926402016152f, -0.4157348061512720f,  0.2777851165098022f, -0.0975451610080625f
};


/**
*  JPEG quality=0_of_12 quantization matrix
*/
float Q[BLOCK_SIZE2] =
{
    32.f,  33.f,  51.f,  81.f,  66.f,  39.f,  34.f,  17.f,
    33.f,  36.f,  48.f,  47.f,  28.f,  23.f,  12.f,  12.f,
    51.f,  48.f,  47.f,  28.f,  23.f,  12.f,  12.f,  12.f,
    81.f,  47.f,  28.f,  23.f,  12.f,  12.f,  12.f,  12.f,
    66.f,  28.f,  23.f,  12.f,  12.f,  12.f,  12.f,  12.f,
    39.f,  23.f,  12.f,  12.f,  12.f,  12.f,  12.f,  12.f,
    34.f,  12.f,  12.f,  12.f,  12.f,  12.f,  12.f,  12.f,
    17.f,  12.f,  12.f,  12.f,  12.f,  12.f,  12.f,  12.f
};


/**
**************************************************************************
*  Performs multiplication of two 8x8 matrices
*
* \param M1             [IN] - Pointer to the first matrix
* \param M1Stride       [IN] - Stride of the first matrix
* \param M2             [IN] - Pointer to the second matrix
* \param M2Stride       [IN] - Stride of the second matrix
* \param Mres           [OUT] - Pointer to the result matrix
* \param MresStride     [IN] - Stride of the result matrix
*
* \return None
*/
void mult8x8(const float *M1, int M1Stride, const float *M2, int M2Stride, float *Mres, int MresStride)
{
    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        for (int j = 0; j < BLOCK_SIZE; j++)
        {
            float accumul = 0;

            for (int k = 0; k < BLOCK_SIZE; k++)
            {
                accumul += M1[i*M1Stride+k] * M2[k*M2Stride+j];
            }

            Mres[i*MresStride+j] = accumul;
        }
    }
}


/**
**************************************************************************
*  Performs 8x8 block-wise Forward Discrete Cosine Transform of the given
*  image plane and outputs result to the plane of coefficients.
*  1st version.
*
* \param fSrc       [IN] - Source image plane
* \param fDst       [OUT] - Destination coefficients plane
* \param Stride     [IN] - Stride of both planes
* \param Size       [IN] - Size of planes
*
* \return None
*/

void computeDCT8x8Gold1(const float *fSrc, float *fDst, int width, int  height)
{
    int Stride = width;
    float tmpblock[BLOCK_SIZE2];

    //perform block wise DCT
    //DCT(A) = DCTv8matrixT * A * DCTv8matrix
    for (int i = 0; i + BLOCK_SIZE - 1 < height; i += BLOCK_SIZE)
    {
        for (int j = 0; j + BLOCK_SIZE - 1 < width; j += BLOCK_SIZE)
        {
            // tmpblock = DCTv8matrixT * A
            mult8x8(DCTv8matrixT, BLOCK_SIZE, fSrc + i * Stride + j, Stride, tmpblock, BLOCK_SIZE);
            // DCT(A) = tmpblock * DCTv8matrix
            mult8x8(tmpblock, BLOCK_SIZE, DCTv8matrix, BLOCK_SIZE, fDst + i * Stride + j, Stride);
        }
    }
}


#endif // DCT_HELPER_HPP
