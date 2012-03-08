/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef _BOXFILTER_KERNEL_H_
#define _BOXFILTER_KERNEL_H_

#include <shrUtils.h>
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cutil_math.h>
#include <stdio.h>

#define PIXEL	186

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#define printf(f, ...) ((void) (f, __VA_ARGS__), 0)
#endif

texture<float, 2> tex;
texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaTex;

cudaArray* d_array, *d_tempArray;

__device__ __forceinline__ uint bitCount(uint v)
{
	uint c;
	for (c = 0; v; ++c)
	{
		v &= v - 1;
	}
	return c;
}

__device__ __forceinline__ uint rgbToyuv(float4 rgba)
{
	float4 yuv;
	yuv.x = 0.29900f*rgba.x + 0.58700f*rgba.y+0.11400f*rgba.z;
	yuv.y = 0.71300f*(rgba.x - yuv.x) + 0.500f;
	yuv.z = 0.56400f*(rgba.z - yuv.x) + 0.500f;
	yuv.x = __saturatef(yuv.x);
	yuv.y = __saturatef(yuv.y);
	yuv.z = __saturatef(yuv.z);
	return (uint(255)<<24) | (uint(yuv.z*255.0f) << 16) | (uint(yuv.y*255.0f) << 8) | uint(yuv.x*255.0f);
}

// If two node's YUV difference is larger than either 48 for Y, 7 for U or 6 for V.
// We consider the two node is not connected
__device__ __forceinline__ bool isConnected(uint lnode, uint rnode)
{
	int ly = lnode & 0xff;
	int lu = ((lnode>>8) & 0xff);
	int lv = ((lnode>>16) & 0xff);
	int ry = rnode & 0xff;
	int ru = ((rnode>>8) & 0xff);
	int rv = ((rnode>>16) & 0xff);
	//if ( center == PIXEL )
	//{
	//	printf("center other: %d %d src: %d %d %d, dst: %d %d %d ",center, other, ly, lu, lv, ry, ru, rv);
	//printf("%d\n", ((48 - abs(ly-ry) >= 0) && (7 - abs(lu-ru) >= 0) && (6 - abs(lv-rv) >= 0)));
	//}
	return !((abs(ly-ry) > 48) || (abs(lu-ru) > 7) || (abs(lv-rv) > 6));
}

// Pass One, check the connectivities.
	__global__ void
d_check_connect(uint *od, uint *connect, int w, int h)
{
	unsigned int center = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int row = center/w;
	int column = center%w;
	int neibor_row, neibor_column;
	unsigned char con = 0;
	uint yuv_c = rgbToyuv(tex2D(rgbaTex, column, row));

	//check 8 neiboughrs of one node for their connectivities.

	//upper left
	neibor_row = (row>0&&column>0)?(row-1):row;
	neibor_column = (column>0&&row>0)?(column-1):column;
	uint yuv_ul = rgbToyuv(tex2D(rgbaTex, neibor_column, neibor_row));
	con += (uint)(!((row==neibor_row) && (column==neibor_column)) && isConnected(yuv_c, yuv_ul));

	//up
	neibor_row = (row>0)?(row-1):row;
	neibor_column = column;
	uint yuv_up = rgbToyuv(tex2D(rgbaTex, neibor_column, neibor_row));
	con += (uint)(!((row==neibor_row) && (column==neibor_column)) && isConnected(yuv_c, yuv_up))<<1;

	//upper right
	neibor_row = (row>0&&column<(w-1))?(row-1):row;
	neibor_column = (column<(w-1)&&row>0)?(column+1):column;
	uint yuv_ur = rgbToyuv(tex2D(rgbaTex, neibor_column, neibor_row));
	con += (uint)(!((row==neibor_row) && (column==neibor_column)) && isConnected(yuv_c, yuv_ur))<<2;

	//right
	neibor_row = row;
	neibor_column = (column<(w-1))?(column+1):column;
	uint yuv_rt = rgbToyuv(tex2D(rgbaTex, neibor_column, neibor_row));
	con += (uint)(!((row==neibor_row) && (column==neibor_column)) && isConnected(yuv_c, yuv_rt))<<3;

	//lower right
	neibor_row = (row<(h-1)&&column<(w-1))?(row+1):row;
	neibor_column = (column<(w-1)&&row<(h-1))?(column+1):column;
	uint yuv_lr = rgbToyuv(tex2D(rgbaTex, neibor_column, neibor_row));
	con += (uint)(!((row==neibor_row) && (column==neibor_column)) && isConnected(yuv_c, yuv_lr))<<4;

	//low
	neibor_row = (row<(h-1))?(row+1):row;
	neibor_column = column;
	uint yuv_lw = rgbToyuv(tex2D(rgbaTex, neibor_column, neibor_row));
	con += (uint)(!((row==neibor_row) && (column==neibor_column)) && isConnected(yuv_c, yuv_lw))<<5;


	//lower left
	neibor_row = (row<(h-1)&&column>0)?(row+1):row;
	neibor_column = (column>0&&row<(h-1))?(column-1):column;
	uint yuv_ll = rgbToyuv(tex2D(rgbaTex, neibor_column, neibor_row));
	con += (uint)(!((row==neibor_row) && (column==neibor_column)) && isConnected(yuv_c, yuv_ll))<<6;


	//left
	neibor_row = row;
	neibor_column = (column>0)?(column-1):column;
	uint yuv_lt = rgbToyuv(tex2D(rgbaTex, neibor_column, neibor_row));
	con += (uint)(!((row==neibor_row) && (column==neibor_column)) && isConnected(yuv_c, yuv_lt))<<7;


	//if ( center == PIXEL )
	//	printf("%d %d\n", center, con);
	//if ( center == 67 || center == 66 )
	//	printf("%u\n", yuv_c);
	//test
	//	__syncthreads();

	connect[center] = (yuv_c>>16&0xFF)<<24 | (yuv_c>>8&0xFF)<<16 | (yuv_c&0xFF)<<8 | con;
	//test end
}


//TODO: Pass Two, find and eliminate crosses.
	__global__ void
d_eliminate_crosses( uint *id, uint *od, int w, int h )
{
	unsigned int center = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int row = center/w;
	int column = center%w;
	int start_row = (row > 2)?row-3:0;
	int start_column = (column > 2)?column-3:0;
	int end_row = (row < w-4)?row+4:w-1;
	int end_column = (column < h-4)?column+4:h-1;
	int weight_l = 0;	//weight for left diagonal	
	int weight_r = 0;	//weight for right diagonal
	od[center] = 0;
	if ((row<h-1) && (column<w-1))
	{
		od[center] = id[center]&0x08 | id[center]&0x20 | id[center+w+1]&0x02 | id[center+w+1]&0x80;

		if ((id[center]&0x10 && id[center+1]&0x40))
		{
			//if fully connected
			if (id[center]&0x28 && id[center+1]&0xA0)
			{
				//eliminate cross (no cross needs to be added)
				return;
			}

			//island
			if (id[center] == 0x10)
			{
				//island 1
				//accumulate weight
				weight_l += 5;
			}
			if (id[center+1] == 0x40)
			{
				//island 2
				//accumulate weight
				weight_r += 5;
			}

			//sparse judge
			int sum_l = 0;
			int sum_r = 0;
			for ( int i = start_row; i <= end_row; ++i )
			{
				for ( int j = start_column; j <= end_column; ++j )
				{
					//compute connectivity
					//accumulate weight
					if (i*w+j!=center && i*w+j!=center+1)
					{
						sum_l += isConnected(id[center]>>8, id[i*w+j]>>8);
						sum_r += isConnected(id[center+1]>>8, id[i*w+j]>>8);
					}
				}
			}

			weight_r += (sum_l > sum_r)?(sum_l-sum_r):0;
			weight_l += (sum_l < sum_r)?(sum_r-sum_l):0;


			//curve judge
			int c_row = row;
			int c_column = column;
			uint curve_l = id[c_row*w+c_column]&0xFF;
			uint edge_l = 16;
			sum_l = 1;
			while(bitCount(curve_l) == 2 && sum_l < w*h)
			{
				edge_l = curve_l - edge_l;
				switch (edge_l)
				{
					case 1:
						c_row -= 1;
						c_column -= 1;
						break;
					case 2:
						c_row -= 1;
						break;
					case 4:
						c_row -= 1;
						c_column += 1;
						break;
					case 8:
						c_column += 1;
						break;
						case 16:
						c_row += 1;
						c_column += 1;
						break;
					case 32:
						c_row += 1;
						break;
					case 64:
						c_row += 1;
						c_column -= 1;
						break;
					case 128:
						c_column -= 1;
						break;
				}
				edge_l = (edge_l > 8)?edge_l>>4:edge_l<<4;
				curve_l = id[c_row*w+c_column]&0xFF;
				++sum_l;
			}
			c_row = row+1;
			c_column = column+1;
			curve_l = id[c_row*w+c_column]&0xFF;
			edge_l = 1;
			while(bitCount(curve_l) == 2 && sum_l < w*h)
			{
				edge_l = curve_l - edge_l;
				switch (edge_l)
				{
					case 1:
					c_row -= 1;
					c_column -= 1;
					break;
					case 16:
						c_row += 1;
						c_column += 1;
						break;
					case 2:
						c_row -= 1;
						break;
					case 4:
						c_row -= 1;
						c_column += 1;
						break;
					case 8:
						c_column += 1;
						break;
					case 32:
						c_row += 1;
						break;
					case 64:
						c_row += 1;
						c_column -= 1;
						break;
					case 128:
						c_column -= 1;
						break;
				}
				edge_l = (edge_l > 8)?edge_l>>4:edge_l<<4;
				curve_l = id[c_row*w+c_column]&0xFF;
				++sum_l;
			}
			c_row = row;
			c_column = column + 1;
			uint curve_r = id[c_row*w+c_column]&0xFF;
			uint edge_r = 64;
			sum_r = 1;
			while(bitCount(curve_r) == 2 && sum_r < w*h)
			{
				edge_r = curve_r - edge_r;
				switch (edge_r)
				{
					case 64:
					c_row += 1;
					c_column -= 1;
					case 1:
						c_row -= 1;
						c_column -= 1;
						break;
					case 2:
						c_row -= 1;
						break;
					case 4:
						c_row -= 1;
						c_column += 1;
						break;
					case 8:
						c_column += 1;
						break;
					case 32:
						c_row += 1;
						break;
					case 16:
						c_row += 1;
						c_column += 1;
						break;
					case 128:
						c_column -= 1;
						break;
				}
				edge_r = (edge_r > 8)?edge_r>>4:edge_r<<4;
				curve_r = id[c_row*w+c_column]&0xFF;
				++sum_r;
			}
			c_row = row+1;
			c_column = column;
			curve_r = id[c_row*w+c_column]&0xFF;
			edge_r = 4;
			while(bitCount(curve_r) == 2 && sum_r < w*h)
			{	
				edge_r = curve_r - edge_r;
				switch (edge_r)
				{
					case 4:
					c_row -= 1;
					c_column += 1;
					break;
					case 16:
						c_row += 1;
						c_column += 1;
						break;
					case 2:
						c_row -= 1;
						break;
					case 1:
						c_row -= 1;
						c_column -= 1;
						break;
					case 8:
						c_column += 1;
						break;
					case 32:
						c_row += 1;
						break;
					case 64:
						c_row += 1;
						c_column -= 1;
						break;
					case 128:
						c_column -= 1;
						break;
				}
				edge_r = (edge_r > 8)?edge_r>>4:edge_r<<4;
				curve_r = id[c_row*w+c_column]&0xFF;
				++sum_r;
			}

			weight_l += (sum_l > sum_r)?(sum_l-sum_r):0;
			weight_r += (sum_l < sum_r)?(sum_r-sum_l):0;


			//eliminate cross according to weight
			if (weight_l > weight_r)
			{
				//add left diagonal
				od[center] |= 0x10;
				return;
			}
			else
			{
				if(weight_r > weight_l)
				{
					//add right diagonal
					od[center] |= 0x20;
					return;
				}
			}
		}
		od[center] = od[center] | id[center]&0x10 | id[center+1]&0x40;
	}
	//if ( center > 0 )
	//	printf("%d %u \n", center, od[center]);
}


//TODO: Pass Three, Voronoi Graph

//TODO: Pass Four, Curve Extraction


	extern "C" 
void initTexture(int width, int height, void *pImage)
{
	int size = width * height * sizeof(unsigned int);

	// copy image data to array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	cutilSafeCall( cudaMallocArray  ( &d_array, &channelDesc, width, height )); 
	cutilSafeCall( cudaMemcpyToArray( d_array, 0, 0, pImage, size, cudaMemcpyHostToDevice));
	cutilSafeCall( cudaMallocArray  ( &d_tempArray,   &channelDesc, width, height )); 

	// set texture parameters
	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.filterMode = cudaFilterModePoint;
	tex.normalized = true;

	// Bind the array to the texture
	cutilSafeCall( cudaBindTextureToArray(tex, d_array, channelDesc) );
}

	extern "C"
void freeTextures()
{
	cutilSafeCall(cudaFreeArray(d_array));
	cutilSafeCall(cudaFreeArray(d_tempArray));
}

	extern "C"
double connectivityDetection(uint *d_temp, unsigned int *d_dest, int width, int height, int nthreads)
{
	cutilSafeCall( cudaBindTextureToArray(rgbaTex, d_array));

	// var for kernel computation timing
	double dKernelTime;

	// sync host and start kernel computation timer
	dKernelTime = 0.0;
	cutilSafeCall(cutilDeviceSynchronize());
	shrDeltaT(0);

	d_check_connect<<<height*width/nthreads, nthreads, 0>>>(d_temp, d_dest, width, height);
	d_eliminate_crosses<<<height*width/nthreads, nthreads, 0>>>(d_dest, d_temp, width, height);

	// sync host and stop computation timer
	cutilSafeCall( cutilDeviceSynchronize() );
	dKernelTime += shrDeltaT(0);

	// copy result back from global memory to array
	cutilSafeCall( cudaMemcpyToArray( d_tempArray, 0, 0, d_temp, width * height * sizeof(float), cudaMemcpyDeviceToDevice));
	cutilSafeCall( cudaBindTextureToArray(rgbaTex, d_tempArray) );

	return (dKernelTime);
}


#endif // #ifndef _BOXFILTER_KERNEL_H_
