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



__global__ void HASH(
	TColor *dst,
	int imageW,
	int imageH
)
{
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;
	//Add half of a texel to always address exact texel centers
	const float x = (float)ix + 0.5f;
	const float y = (float)iy + 0.5f;

	if (ix < imageW && iy < imageH)
	{
		// Calling tex2D x*x times is slow
		// TODO: learn to load image sections as pointer to array
		float4 fresult = tex2D(texImage, x, y);
		//int4 fIntResult = int4(fresult) * 255;
		int4 fIntResult;
		fIntResult.x = __float2int_rz(fresult.x * 255.0f);
		fIntResult.y = __float2int_rz(fresult.y * 255.0f);
		fIntResult.z = __float2int_rz(fresult.z * 255.0f);
		float4 hR, hG, hB;
		int mod = fIntResult.x & 16 - 1;
		int rem = fIntResult.x - (mod * 16);
		hR = tex2D(hashImage, float(mod + 0.5f), float(rem + 0.5f));
		mod = fIntResult.y & 16 - 1;
		rem = fIntResult.y - (mod * 16);
		hG = tex2D(hashImage, float(mod + 0.5f), float(rem + 0.5f));
		mod = fIntResult.z & 16 - 1;
		rem = fIntResult.z - (mod * 16);
		hB = tex2D(hashImage, float(mod + 0.5f), float(rem + 0.5f));
		dst[imageW * iy + ix] = make_color(hR.x/255.0f, hG.y / 255.0f, hB.z / 255.0f, 0);
	}
}

extern "C" void
cuda_HASH(TColor *d_dst, int imageW, int imageH)
{
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	HASH << <grid, threads >> >(d_dst, imageW, imageH);
}
