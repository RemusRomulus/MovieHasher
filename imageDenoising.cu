/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



/*
 * This sample demonstrates two adaptive image denoising techniques:
 * KNN and NLM, based on computation of both geometric and color distance
 * between texels. While both techniques are already implemented in the
 * DirectX SDK using shaders, massively speeded up variation
 * of the latter technique, taking advantage of shared memory, is implemented
 * in addition to DirectX counterparts.
 * See supplied whitepaper for more explanations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "imageDenoising.h"


////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
float Max(float x, float y)
{
    return (x > y) ? x : y;
}

float Min(float x, float y)
{
    return (x < y) ? x : y;
}

int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__device__ float lerpf(float a, float b, float c)
{
    return a + (b - a) * c;
}

__device__ float vecLen(float4 a, float4 b)
{
    return (
               (b.x - a.x) * (b.x - a.x) +
               (b.y - a.y) * (b.y - a.y) +
               (b.z - a.z) * (b.z - a.z)
           );
}

__device__ TColor make_color(float r, float g, float b, float a)
{
    return
        ((int)(a * 255.0f) << 24) |
        ((int)(b * 255.0f) << 16) |
        ((int)(g * 255.0f) <<  8) |
        ((int)(r * 255.0f) <<  0);
}

__device__ void uv_wrap(const int currX, const int currY, int &x, int &y, const int w, const int h)
{
	x = (currX >= w) ? currX - w : currX ;
	y = (currY >= w) ? currY - h : currY ;
}

__device__ TColor make_color_int(unsigned int r, unsigned int g, unsigned int b, unsigned int a)
{
	return(a << 24 | b << 16 | g << 8 | r);
}

//__device__ TColor operator+(const TColor &a, const TColor &b)
//{
//	unsigned int amask = 0xff000000;
//	unsigned int bmask = 0x00ff0000;
//	unsigned int gmask = 0x0000ff00;
//	unsigned int rmask = 0x000000ff;
//	unsigned int aa = (int)(amask & a) >> 24;
//	unsigned int ab = (int)(bmask & a) >> 16;
//	unsigned int ag = (int)(gmask & a) >> 8;
//	unsigned int ar = (int)(rmask & a) >> 0;
//
//	unsigned int ba = (int)(amask & b) >> 24;
//	unsigned int bb = (int)(bmask & b) >> 16;
//	unsigned int bg = (int)(gmask & b) >> 8;
//	unsigned int br = (int)(rmask & b) >> 0;
//
//	return make_color(ar + br, ag + bg, ab + bb, aa + ba);
//}

__device__ TColor tcolor_plus_tcolor(const TColor &a, const TColor &b)
{
	unsigned int amask = 0xff000000;
	unsigned int bmask = 0x00ff0000;
	unsigned int gmask = 0x0000ff00;
	unsigned int rmask = 0x000000ff;
	unsigned int aa = (int)(amask & a) >> 24;
	unsigned int ab = (int)(bmask & a) >> 16;
	unsigned int ag = (int)(gmask & a) >> 8;
	unsigned int ar = (int)(rmask & a) >> 0;

	unsigned int ba = (int)(amask & b) >> 24;
	unsigned int bb = (int)(bmask & b) >> 16;
	unsigned int bg = (int)(gmask & b) >> 8;
	unsigned int br = (int)(rmask & b) >> 0;

	return make_color_int(ar + br, ag + bg, ab + bb, aa + ba);
}




////////////////////////////////////////////////////////////////////////////////
// Global data handlers and parameters
////////////////////////////////////////////////////////////////////////////////
//Texture reference and channel descriptor for image texture
texture<uchar4, 2, cudaReadModeNormalizedFloat> texImage;
cudaChannelFormatDesc uchar4tex = cudaCreateChannelDesc<uchar4>();

texture<uchar4, 2, cudaReadModeNormalizedFloat> tex_next_Image;
cudaChannelFormatDesc uchar4_next_tex = cudaCreateChannelDesc<uchar4>();

texture<uchar4, 2, cudaReadModeNormalizedFloat> hashImage;
cudaChannelFormatDesc uchar4hash = cudaCreateChannelDesc<uchar4>();

//CUDA array descriptor
cudaArray *a_Src;
cudaArray *next_Src;
cudaArray *hash_Src;

////////////////////////////////////////////////////////////////////////////////
// Filtering kernels
////////////////////////////////////////////////////////////////////////////////
#include "imageDenoising_copy_kernel.cuh"
#include "imageDenoising_knn_kernel.cuh"
#include "imageDenoising_nlm_kernel.cuh"
#include "imageDenoising_nlm2_kernel.cuh"
#include "imageDenoising_hasher_kernel.cuh"

extern "C"
cudaError_t CUDA_Bind2TextureArray()
{
	cudaError_t out = cudaBindTextureToArray(texImage, a_Src);
    if(out !=0)
		return out;

	out = cudaBindTextureToArray(tex_next_Image, next_Src);
	if (out !=0)
		return out;

	return cudaBindTextureToArray(hashImage, hash_Src);
}

extern "C"
cudaError_t CUDA_UnbindTexture()
{
	cudaError_t out = cudaUnbindTexture(texImage);
	if (out != 0)
		return out;

	out = cudaUnbindTexture(tex_next_Image);
	if(out!=0)
		return out;

	return cudaUnbindTexture(hashImage);
}

extern "C"
cudaError_t CUDA_MallocArray(uchar4 **h_Src, uchar4 **h_next_Src, uchar4 **hashHost_Src, int imageW, int imageH,
	int hashW, int hashH)
{
    cudaError_t error;

	// Init current frame
    error = cudaMallocArray(&a_Src, &uchar4tex, imageW, imageH);
    error = cudaMemcpyToArray(a_Src, 0, 0,
                              *h_Src, imageW * imageH * sizeof(uchar4),
                              cudaMemcpyHostToDevice
                             );

	// init next frame
	error = cudaMallocArray(&next_Src, &uchar4_next_tex, imageW, imageH);
    error = cudaMemcpyToArray(next_Src, 0, 0,
                              *h_next_Src, imageW * imageH * sizeof(uchar4),
                              cudaMemcpyHostToDevice
                             );

	// init hash tex
	error = cudaMallocArray(&hash_Src, &uchar4hash, hashW, hashH);
	error = cudaMemcpyToArray(hash_Src, 0, 0,
							*hashHost_Src, hashW * hashH * sizeof(uchar4),
							cudaMemcpyHostToDevice
							);

    return error;
}


extern "C"
cudaError_t CUDA_FreeArray()
{
	cudaError_t out = cudaFreeArray(a_Src);
	if (out != 0)
		return out;
	
	out = cudaFreeArray(next_Src);
	if (out != 0)
		return out;
	
	return cudaFreeArray(hash_Src);
}

