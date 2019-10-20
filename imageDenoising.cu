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

__device__ void tcolor_to_int(TColor color, unsigned int &r, unsigned int &g, unsigned int &b, unsigned int &a)
{
	unsigned int amask = 0xff000000;
	unsigned int bmask = 0x00ff0000;
	unsigned int gmask = 0x0000ff00;
	unsigned int rmask = 0x000000ff;
	a = (int)(amask & color) >> 24;
	b = (int)(bmask & color) >> 16;
	g = (int)(gmask & color) >> 8;
	r = (int)(rmask & color) >> 0;
}

__device__ TColor tcolor_plus_tcolor(const TColor &a, const TColor &b)
{
	unsigned int aa, ab, ag, ar, ba, bb, bg, br;

	tcolor_to_int(a, ar, ag, ab, aa);
	tcolor_to_int(b, br, bg, bb, ba);

	return make_color_int(ar + br, ag + bg, ab + bb, aa + ba);
}

__device__ void THashAffineLookup(int4 input, THash *lookup, unsigned short orientation, float &r, float &g, float &b)
{
	if (orientation == 0 || orientation > 3)
	{
		THash mask = THash(0x0000000000000001);
		int channel_size = 4;
		int channel_mult = 0;
		int iter = input.x & 3;
		THash sub_iter = input.x - (iter * 64);
		THash mask_r = mask << sub_iter;
		r = (mask_r & lookup[iter + (channel_mult*channel_size)]) * 255.0f;

		mask = THash(0x0000000000000001);
		channel_mult = 1;
		iter = input.y & 3;
		sub_iter = input.y - (iter * 64);
		mask_r = mask << sub_iter;
		g = (mask_r & lookup[iter + (channel_mult*channel_size)]) * 255.0f;

		mask = THash(0x0000000000000001);
		channel_mult = 2;
		iter = input.z & 3;
		sub_iter = input.z - (iter * 64);
		mask_r = mask << sub_iter;
		b = (mask_r & lookup[iter + (channel_mult*channel_size)]) * 255.0f;
	}
	else if (orientation == 1)
	{
		THash mask = THash(0x0000000000000001);

		//remap inputs
		int3 _in;
		_in.x = (16 * (input.x & 15)) + (input.x & 15);
		_in.y = (16 * (input.y & 15)) + (input.y & 15);
		_in.z = (16 * (input.z & 15)) + (input.z & 15);

		int channel_size = 4;
		int channel_mult = 0;
		int iter = _in.x & 3;
		THash sub_iter = _in.x - (iter * 64);
		THash mask_r = mask << sub_iter;
		r = (mask_r & lookup[iter + (channel_mult*channel_size)]) * 255.0f;

		mask = THash(0x0000000000000001);
		channel_mult = 1;
		iter = _in.y & 3;
		sub_iter = _in.y - (iter * 64);
		mask_r = mask << sub_iter;
		g = (mask_r & lookup[iter + (channel_mult*channel_size)]) * 255.0f;

		mask = THash(0x0000000000000001);
		channel_mult = 2;
		iter = _in.z & 3;
		sub_iter = _in.z - (iter * 64);
		mask_r = mask << sub_iter;
		b = (mask_r & lookup[iter + (channel_mult*channel_size)]) * 255.0f;

		
	}
	else if (orientation == 2)
	{
		THash mask = THash(0x8000000000000000);
		int channel_size = 4;
		int channel_mult = 0;
		int iter = 3 - (input.x & 3);
		THash sub_iter = 64 - (input.x - (iter * 64));
		THash mask_r = mask >> sub_iter;
		r = (mask_r & lookup[iter + (channel_mult*channel_size)]) * 255.0f;

		mask = THash(0x8000000000000000);
		channel_mult = 1;
		iter = 3 - (input.y & 3);
		sub_iter = 64 - (input.y - (iter * 64));
		mask_r = mask >> sub_iter;
		g = (mask_r & lookup[iter + (channel_mult*channel_size)]) * 255.0f;

		mask = THash(0x8000000000000000);
		channel_mult = 2;
		iter = 3 - (input.z & 3);
		sub_iter = 64 - (input.z - (iter * 64));
		mask_r = mask >> sub_iter;
		b = (mask_r & lookup[iter + (channel_mult*channel_size)]) * 255.0f;
	}
	else
	{
		THash mask = THash(0x8000000000000000);

		//remap inputs
		int3 _in;
		_in.x = (16 * (input.x & 15)) + (input.x & 15);
		_in.y = (16 * (input.y & 15)) + (input.y & 15);
		_in.z = (16 * (input.z & 15)) + (input.z & 15);

		int channel_size = 4;
		int channel_mult = 0;
		int iter = _in.x & 3;
		THash sub_iter = _in.x - (iter * 64);
		THash mask_r = mask >> sub_iter;
		r = (mask_r & lookup[iter + (channel_mult*channel_size)]) * 255.0f;

		mask = THash(0x8000000000000000);
		channel_mult = 1;
		iter = _in.y & 3;
		sub_iter = _in.y - (iter * 64);
		mask_r = mask >> sub_iter;
		g = (mask_r & lookup[iter + (channel_mult*channel_size)]) * 255.0f;

		mask = THash(0x8000000000000000);
		channel_mult = 2;
		iter = _in.z & 3;
		sub_iter = _in.z - (iter * 64);
		mask_r = mask >> sub_iter;
		b = (mask_r & lookup[iter + (channel_mult*channel_size)]) * 255.0f;
	}
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

