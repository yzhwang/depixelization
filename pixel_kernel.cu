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


//#define PIXEL	16*16-1

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#define printf(f, ...) ((void) (f, __VA_ARGS__), 0)
#endif

//texture<float, 2> tex;
texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaTex;

cudaArray* d_array, *d_tempArray;

__device__ void setNodeValues2(float2* pt,
	    float x1, float y1,
		float x2, float y2)
{
	pt[0].x = x1;
	pt[0].y = y1;
	pt[1].x = x2;
	pt[1].y = y2;
}

__device__ void setNodeValues(float2* pt,
		float x1, float y1,
		float x2, float y2,
		float x3, float y3,
		float x4, float y4,
		float x5, float y5,
		float x6, float y6,
		float x7, float y7,
		float x8, float y8)
{
	//if (abs(pt[0].x-0.0f) < 0.001f || abs(pt[0].x -1.0f) < 0.001f || abs(pt[0].x + 1.0f) < 0.001f)
	{
		pt[0].x = x1;
		pt[0].y = y1;
	}
	//if (abs(pt[1].x-0.0f) < 0.001f || abs(pt[1].x -1.0f) < 0.001f || abs(pt[1].x + 1.0f) < 0.001f)
	{
		pt[1].x = x2;
		pt[1].y = y2;
	}
	//if (abs(pt[2].x-0.0f) < 0.001f || abs(pt[2].x -1.0f) < 0.001f || abs(pt[2].x + 1.0f) < 0.001f)
	{
		pt[2].x = x3;
		pt[2].y = y3;
	}
	//if (abs(pt[3].x-0.0f) < 0.001f || abs(pt[3].x -1.0f) < 0.001f || abs(pt[3].x + 1.0f) < 0.001f)
	{
		pt[3].x = x4;
		pt[3].y = y4;
	}
	//if (abs(pt[4].x-0.0f) < 0.001f || abs(pt[4].x -1.0f) < 0.001f || abs(pt[4].x + 1.0f) < 0.001f)
	{
		pt[4].x = x5;
		pt[4].y = y5;
	}
	//if (abs(pt[5].x-0.0f) < 0.001f || abs(pt[5].x -1.0f) < 0.001f || abs(pt[5].x + 1.0f) < 0.001f)
	{
		pt[5].x = x6;
		pt[5].y = y6;
	}
	{
		pt[6].x = x7;
		pt[6].y = y7;
		pt[7].x = x8;
		pt[7].y = y8;
	}

	/*
	   float2 unsorted1[6];
	   float2 unsorted2[6];
	   int index1 = 0;
	   int index2 = 0;
	   for ( int i = 0; i < 6; ++i )
	   {
	   if (pt[i].y < 0.5f)
	   unsorted1[index1++] = pt[i];
	   else
	   unsorted2[index2++] = pt[i];
	   }

	   for ( int i = 0; i < index1-1; ++i )
	   {
	   for ( int j = i+1; j < index1; ++j )
	   {
	   if (unsorted1[i].x > unsorted1[j].x)
	   {
	   float2 temp = unsorted1[i];
	   unsorted1[i] = unsorted1[j];
	   unsorted1[j] = temp;
	   }
	   }
	   }


	   for ( int i = 0; i < index2-1; ++i )
	   {
	   for ( int j = i+1; j < index2; ++j )
	   {
	   if (unsorted2[i].x < unsorted2[j].x)
	   {
	   float2 temp = unsorted2[i];
	   unsorted2[i] = unsorted2[j];
	   unsorted2[j] = temp;
	   }
	   }
	   }

	//for ( int i = 0; i < index1; ++i )
	//{
	//		printf("%f %f ", unsorted1[i].x, unsorted1[i].y);
	//	}
	//	printf("%f \n", 8000000.0f);


	int minus = 0;
	for ( int i = 0; i < index1; ++i )
	{
	while (unsorted1[i].x < -0.5f)
	{
	++minus;
	++i;
	}
	pt[i-minus] = unsorted1[i];
	}
	for ( int i = 0; i < index2; ++i )
	{
	pt[i+index1-minus] = unsorted2[i];
	}
	for ( int i = 0; i < minus; ++i )
	{
	pt[index1+index2-minus+i].x = -1.0f;
	pt[index1+index2-minus+i].y = -1.0f;
	}

	//printf(" center %f %f %f %f %f %f %f %f %f %f %f %f \n", pt[0].x, pt[0].y,
	//														 pt[1].x, pt[1].y,
	//														 pt[2].x, pt[2].y,
	//														 pt[3].x, pt[3].y,
	//														 pt[4].x, pt[4].y,
	//														 pt[5].x, pt[5].y);
	*/	
}

/*
   code from http://alienryderflex.com/polygon/
 */
__device__ bool isPointInPolygon(float2 src, float2* corners)
{

	//calculate number of corners
	int index[8] = {0};
	int n_corners = 0;
	for ( int i = 0; i < 8; ++i )
	{
		if (corners[i].x > -0.5f)
		{
			//printf("%d got here!\n", 1);
			index[n_corners++] = i;
		}
	}
	//printf("%d %f %f %d %f %f %d %f %f %d %f %f %d %f %f %d %f %f %d %f %f %d %f %f\n", index[0], corners[0].x, corners[0].y, index[1], corners[1].x, corners[1].y, index[2], corners[2].x, corners[2].y, index[3], corners[3].x, corners[3].y,
	  //                                                    index[4], corners[4].x, corners[4].y, index[5], corners[5].x, corners[5].y, index[6], corners[6].x, corners[6].y, index[7], corners[7].x, corners[7].y);

	int j = n_corners - 1;
	bool odd_nodes = false;
	for ( int i = 0; i < n_corners; ++i )
	{
		if ((corners[index[i]].y < src.y && corners[index[j]].y >= src.y
					|| corners[index[j]].y < src.y && corners[index[i]].y >= src.y)
				&& (corners[index[i]].x <= src.x || corners[index[j]].x <= src.x))
		{
			odd_nodes^=(corners[index[i]].x+(src.y-corners[index[i]].y)/(corners[index[j]].y-corners[index[i]].y)*(corners[index[j]].x-corners[index[i]].x)<src.x);
		}
		j = i;
	}
	return odd_nodes;
}

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

__device__ __forceinline__ uint yuvTorgba(uint yuvi)
{
	float4 yuv;
	yuv.x = (float)(yuvi&0xFF) * 0.003921568627f;
	yuv.y = (float)((yuvi>>8)&0xFF) * 0.003921568627f;
	yuv.z = (float)((yuvi>>16)&0xFF) * 0.003921568627f;
	float4 rgb;
	rgb.x = yuv.x + (yuv.y-0.5f) * 1.403f;
	rgb.y = yuv.x - 0.714f * yuv.y - 0.344f * yuv.z + 0.529f;
	rgb.z = yuv.x + (yuv.z-0.5f) * 1.773f;
	rgb.x = __saturatef(rgb.x);
	rgb.y = __saturatef(rgb.y);
	rgb.z = __saturatef(rgb.z);
	return (uint(255.0f) << 24) | (uint(rgb.z * 255.0f) << 16) | (uint(rgb.y * 255.0f) << 8) | uint(rgb.x * 255.0f);
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
		od[center] = (id[center]&0x08)>>3 | (((id[center+w+1]&0x02)>>1)<<1) | (((id[center+w+1]&0x80)>>7)<<2) | (((id[center]&0x20)>>5)<<3);
		//if ( center < 6 )
		//	printf("center %d od %u\n", center, od[center]);

		if ((id[center]&0x10 && id[center+1]&0x40))
		{
			//if fully connected
			if (id[center]&0x28 && id[center+1]&0xA0)
			{
				//eliminate cross (no cross needs to be added)
				od[center] = ((id[center]>>8)&0xFFFFFF)<<8 | od[center];
				//printf("center %d od %u \n", center, od[center]&0xFF);
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
				od[center] = ((id[center]>>8)&0xFFFFFF)<<8 | od[center];
				//if (center == 4)
				//	printf("%d %u \n", center, od[center]&0xFF);
				return;
			}
			else
			{
				if(weight_r > weight_l)
				{
					//add right diagonal
					od[center] |= 0x20;
					od[center] = ((id[center]>>8)&0xFFFFFF)<<8 | od[center];
					//if (center == 4)
					//	printf("%d %u \n", center, od[center]&0xFF);
					return;
				}
			}
		}
		od[center] = od[center] | (((id[center]&0x10)>>4)<<4) | (((id[center+1]&0x40)>>6)<<5);
	}
	od[center] = ((id[center]>>8)&0xFFFFFF)<<8 | od[center];
	//od[center] = od[center]&0xFF;
	//if (center == 4)
	//	printf("%d %u \n", center, od[center]&0xFF);
}

//TODO: Pass Three, Voronoi Graph Generation
	__global__ void
d_voronoi_generation(uint *id, float2 *pt, int w, int h, int scale)
{
	unsigned int center = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int row = center/w;
	int column = center%w;
	setNodeValues(&pt[center*8], 0, 0, -1, -1, 1, 0, -1, -1, 1, 1, -1, -1, 0, 1, -1, -1);
	if ((row<h-1) && (row>0) && (column>0) && (column<w-1))
	{
		//setNodeValues(&pt[(center+1)*8], 0, 0, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1);
		//setNodeValues(&pt[(center+2)*8], 0, 0, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1);
		//setNodeValues(&pt[(center+w)*8], 0, 0, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1);
		//setNodeValues(&pt[(center+w+1)*8], 0, 0, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1);
		//setNodeValues(&pt[(center+w+2)*8], 0, 0, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1);

		//printf("row %d column %d center %u center+1 %u\n", row, column, (id[center]&0x3F), (id[center+1]&0x3F));
		if ((id[center-w-1]&0x36)==32)
		{
			//case 1
			setNodeValues2(&pt[center*8], 0.25, 0.25, -1, -1);

		}
		if ((id[center-w-1]&0x33)==16)
		{
			setNodeValues2(&pt[center*8], -0.25, 0.25, 0.25, -0.25);
		}
		if ((id[center-w-1]&0x33)==19)
		{
			setNodeValues2(&pt[center*8], -0.25, -0.25, 0, 0);
		}
		if ((id[center-w-1]&0x3B)==28)
		{
			setNodeValues2(&pt[center*8], 0, 0, 0.25, -0.25);
		}

		if ((id[center-w]&0x3B)==16)
		{
			//case 2
			setNodeValues2(&pt[center*8+2], 0.75, 0.25, -1, -1);

		}
		if ((id[center-w]&0x33)==32)
		{
			setNodeValues2(&pt[center*8+2], 0.75, -0.25, 1.25, 0.25);
		}
		if ((id[center-w]&0x33)==41)
		{
			setNodeValues2(&pt[center*8+2], 1, 0, 1.25, 0.25);
		}
		if ((id[center-w]&0x3B)==38)
		{
			setNodeValues2(&pt[center*8+2], 0.75, -0.25, 1, 0);
		}

		if ((id[center]&0x39)==32)
		{
			//case 3
			setNodeValues2(&pt[center*8+4], 0.75, 0.75, -1, -1);

		}
		if ((id[center]&0x33)==16)
		{
			setNodeValues2(&pt[center*8+4], 1.25, 0.75, 0.75, 1.25);
		}
		if ((id[center]&0x33)==19)
		{
			setNodeValues2(&pt[center*8+4], 1, 1, 0.75, 1.25);
		}
		if ((id[center]&0x3B)==28)
		{
			setNodeValues2(&pt[center*8+4], 1.25, 0.75, 1, 1);
		}

		if ((id[center-1]&0x33)==16)
		{
			//case 4
			setNodeValues2(&pt[center*8+6], 0.25, 0.75, -1, -1);

		}
		if ((id[center-1]&0x33)==32)
		{
			setNodeValues2(&pt[center*8+6], 0.25, 1.25, -0.25, 0.75);
		}
		if ((id[center-1]&0x36)==38)
		{
			setNodeValues2(&pt[center*8+6], 0, 1, -0.25, 0.75);
		}
		if ((id[center-1]&0x39)==41)
		{
			setNodeValues2(&pt[center*8+6], 0.25, 1.25, 0, 1);
		}
	}

}

/*
   __global__ void
   d_voronoi_generation_r0c1(uint *id, float2 *pt, int w, int h, int scale)
   {
   unsigned int center = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
   int row = center/w;
   int column = center%w;
   if ((row<h-1) && (column<w-2))
   {

   if ((row%2==0) && (column%3==1))
   {
   if (((id[center]&0x23)==0) && ((id[center+1]&0x39)==32))
   {
//case 1
setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);

}
if (((id[center]&0x33)==16) && ((id[center+1]&0x19)==0))
{
// case 2
setNodeValues(&pt[center*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 1, 0, -1, -1);
setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
}
if (((id[center]&0x33)==16) && ((id[center+1]&0x39)==32))
{
// case 3
//printf("center %d\n", center);
setNodeValues(&pt[center*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -1, -1);
setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 0.75, 0.75, 0.25, 0.75, -1, -1, -1, -1);
setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
}
if (((id[center]&0x3F)==32) && ((id[center+1]&0x30)==0))
{
// case 4
setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.25, 1.25, -0.25, 0.75, -1, -1);
setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1);
}
if (((id[center]&0x3F)==32) && ((id[center+1]&0x39)==32))
{
// case 5
setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 0.75, 0.75, 0.25, 1.25, -0.25, 0.75, -1, -1);
setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
}
if (((id[center]&0x3F)==41) && ((id[center+1]&0x18)==0))
{
// case 6
setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.25, 1.25, 0, 1, -1, -1);
/setNodeValues(&pt[(center+w)*6], 0, 0, 1, 0, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1);
}
if (((id[center]&0x3F)==41) && ((id[center+1]&0x39)==32))
{
//case 7
setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 0.75, 0.75, 0.25, 1.25, 0, 1, -1, -1);
setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
setNodeValues(&pt[(center+w)*6], 0, 0, 1, 0, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
}
if (((id[center]&0x3F)==38) && ((id[center+1]&0x08)==8))
{
// case 8
setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
}
if (((id[center]&0x20)==0) && ((id[center+1]&0x3F)==16))
{
	// case 9
	setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 0.75, 1.25, 0, 1, -1, -1);
	setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
	setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
	setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
}
if (((id[center]&0x13)==16) && ((id[center+1]&0x3F)==16))
{
	// case 10
	setNodeValues(&pt[center*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -1, -1);
	setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 0.75, 1.25, 0.25, 0.75, -1, -1);
	setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
	setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.25, -0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1);
	setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
}
if (((id[center]&0x22)==0) && ((id[center+1]&0x3F)==19))
{
	// case 11
	setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.75, 1.25, 0, 1, -1, -1);
	setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
	setNodeValues(&pt[(center+w+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.25, -1, -1);
}
if (((id[center]&0x13)==16) && ((id[center+1]&0x3F)==19))
{
	// case 12
	setNodeValues(&pt[center*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -1, -1);
	setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.75, 1.25, 0.25, 0.75, -1, -1);
	setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.25, -0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1);
	setNodeValues(&pt[(center+w+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.25, -1, -1);
}
if (((id[center]&0x02)==2) && ((id[center+1]&0x3F)==28))
{
	// case 13
	setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -1, -1);
	setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
	setNodeValues(&pt[(center+w+2)*6], 0, 0, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
}
if (((id[center]&0x3F)==32) && ((id[center+1]&0x3F)==16))
{
	// case 14
	setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
	setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 0.75, 1.25, 0.25, 1.25, -0.25, 0.75);
	setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
	setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
	setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
	setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
}
if (((id[center]&0x3F)==38) && ((id[center+1]&0x3F)==28))
{
	// case 15
	setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
	setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -0.25, 0.75);
	setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
	setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
	setNodeValues(&pt[(center+w+2)*6], 0, 0, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);


}
if (((id[center]&0x3F)==41) && ((id[center+1]&0x3F)==16))
{
	// case 16
	setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 0.75, 1.25, 0.25, 1.25, 0, 1);
	setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 0.25, 0.75, 1, 1, -1, -1, -1, -1);
	setNodeValues(&pt[(center+w)*6], 0, 0, 1, 0, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
	setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
	setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);

}
if (((id[center]&0x3F)==32) && ((id[center+1]&0x3F)==19))
{
	// case 17
	setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
	setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.75, 1.25, 0.25, 1.25, -0.25, 0.75);
	setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
	setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.25, 0.75, 1, 1, 0, 1, -1, -1, -1, -1);
	setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0, 0, 1, 0, 1, 1, 0, 1, -1, -1);

}
if (((id[center]&0x3F)==41) && ((id[center+1]&0x3F)==19))
{
	// case 18
	setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.25, 1.25, 0.75, 1.25, 0, 1);
	setNodeValues(&pt[(center+w)*6], 0, 0, 1, 0, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
	setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
	setNodeValues(&pt[(center+w+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.25, -1, -1);
}

}

}
}
	__global__ void
d_voronoi_generation_r0c2(uint *id, float2 *pt, int w, int h, int scale)
{
	unsigned int center = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int row = center/w;
	int column = center%w;
	if ((row<h-1) && (column<w-2))
	{

		if ((row%2==0) && (column%3==2))
		{
			if (((id[center]&0x23)==0) && ((id[center+1]&0x39)==32))
			{
				//case 1
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);

			}
			if (((id[center]&0x33)==16) && ((id[center+1]&0x19)==0))
			{
				// case 2
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 1, 0, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x33)==16) && ((id[center+1]&0x39)==32))
			{
				// case 3
				//printf("center %d\n", center);
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 0.75, 0.75, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
			}
			if (((id[center]&0x3F)==32) && ((id[center+1]&0x30)==0))
			{
				// case 4
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.25, 1.25, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1);
			}
			if (((id[center]&0x3F)==32) && ((id[center+1]&0x39)==32))
			{
				// case 5
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 0.75, 0.75, 0.25, 1.25, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x3F)==41) && ((id[center+1]&0x18)==0))
			{
				// case 6
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.25, 1.25, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 1, 0, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1);
			}
			if (((id[center]&0x3F)==41) && ((id[center+1]&0x39)==32))
			{
				//case 7
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 0.75, 0.75, 0.25, 1.25, 0, 1, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 1, 0, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x3F)==38) && ((id[center+1]&0x08)==8))
			{
				// case 8
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x20)==0) && ((id[center+1]&0x3F)==16))
			{
				// case 9
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 0.75, 1.25, 0, 1, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x13)==16) && ((id[center+1]&0x3F)==16))
			{
				// case 10
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 0.75, 1.25, 0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.25, -0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x22)==0) && ((id[center+1]&0x3F)==19))
			{
				// case 11
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.75, 1.25, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.25, -1, -1);
			}
			if (((id[center]&0x13)==16) && ((id[center+1]&0x3F)==19))
			{
				// case 12
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.75, 1.25, 0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.25, -0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.25, -1, -1);
			}
			if (((id[center]&0x02)==2) && ((id[center+1]&0x3F)==28))
			{
				// case 13
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], 0, 0, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x3F)==32) && ((id[center+1]&0x3F)==16))
			{
				// case 14
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 0.75, 1.25, 0.25, 1.25, -0.25, 0.75);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x3F)==38) && ((id[center+1]&0x3F)==28))
			{
				// case 15
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -0.25, 0.75);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], 0, 0, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);


			}
			if (((id[center]&0x3F)==41) && ((id[center+1]&0x3F)==16))
			{
				// case 16
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 0.75, 1.25, 0.25, 1.25, 0, 1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 0.25, 0.75, 1, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 1, 0, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);

			}
			if (((id[center]&0x3F)==32) && ((id[center+1]&0x3F)==19))
			{
				// case 17
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.75, 1.25, 0.25, 1.25, -0.25, 0.75);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.25, 0.75, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0, 0, 1, 0, 1, 1, 0, 1, -1, -1);

			}
			if (((id[center]&0x3F)==41) && ((id[center+1]&0x3F)==19))
			{
				// case 18
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.25, 1.25, 0.75, 1.25, 0, 1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 1, 0, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.25, -1, -1);
			}

		}

	}
}

	__global__ void
d_voronoi_generation_r1c0(uint *id, float2 *pt, int w, int h, int scale)
{
	unsigned int center = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int row = center/w;
	int column = center%w;
	if ((row<h-1) && (column<w-2))
	{	//setNodeValues(&pt[center*6], 0, 0, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1);
		//setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1);
		//setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1);
		//setNodeValues(&pt[(center+w)*6], 0, 0, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1);
		//setNodeValues(&pt[(center+w+1)*6], 0, 0, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1);
		//setNodeValues(&pt[(center+w+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1);


		if ((row%2==1) && (column%3==0))
		{
			if (((id[center]&0x23)==0) && ((id[center+1]&0x39)==32))
			{
				//case 1
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);

			}
			if (((id[center]&0x33)==16) && ((id[center+1]&0x19)==0))
			{
				// case 2
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 1, 0, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x33)==16) && ((id[center+1]&0x39)==32))
			{
				// case 3
				//printf("center %d\n", center);
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 0.75, 0.75, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
			}
			if (((id[center]&0x3F)==32) && ((id[center+1]&0x30)==0))
			{
				// case 4
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.25, 1.25, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1);
			}
			if (((id[center]&0x3F)==32) && ((id[center+1]&0x39)==32))
			{
				// case 5
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 0.75, 0.75, 0.25, 1.25, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x3F)==41) && ((id[center+1]&0x18)==0))
			{
				// case 6
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.25, 1.25, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 1, 0, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1);
			}
			if (((id[center]&0x3F)==41) && ((id[center+1]&0x39)==32))
			{
				//case 7
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 0.75, 0.75, 0.25, 1.25, 0, 1, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 1, 0, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x3F)==38) && ((id[center+1]&0x08)==8))
			{
				// case 8
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x20)==0) && ((id[center+1]&0x3F)==16))
			{
				// case 9
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 0.75, 1.25, 0, 1, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x13)==16) && ((id[center+1]&0x3F)==16))
			{
				// case 10
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 0.75, 1.25, 0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.25, -0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x22)==0) && ((id[center+1]&0x3F)==19))
			{
				// case 11
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.75, 1.25, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.25, -1, -1);
			}
			if (((id[center]&0x13)==16) && ((id[center+1]&0x3F)==19))
			{
				// case 12
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.75, 1.25, 0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.25, -0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.25, -1, -1);
			}
			if (((id[center]&0x02)==2) && ((id[center+1]&0x3F)==28))
			{
				// case 13
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], 0, 0, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x3F)==32) && ((id[center+1]&0x3F)==16))
			{
				// case 14
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 0.75, 1.25, 0.25, 1.25, -0.25, 0.75);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x3F)==38) && ((id[center+1]&0x3F)==28))
			{
				// case 15
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -0.25, 0.75);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], 0, 0, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);


			}
			if (((id[center]&0x3F)==41) && ((id[center+1]&0x3F)==16))
			{
				// case 16
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 0.75, 1.25, 0.25, 1.25, 0, 1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 0.25, 0.75, 1, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 1, 0, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);

			}
			if (((id[center]&0x3F)==32) && ((id[center+1]&0x3F)==19))
			{
				// case 17
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.75, 1.25, 0.25, 1.25, -0.25, 0.75);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.25, 0.75, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0, 0, 1, 0, 1, 1, 0, 1, -1, -1);

			}
			if (((id[center]&0x3F)==41) && ((id[center+1]&0x3F)==19))
			{
				// case 18
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.25, 1.25, 0.75, 1.25, 0, 1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 1, 0, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.25, -1, -1);
			}

		}

	}
}

	__global__ void
d_voronoi_generation_r1c1(uint *id, float2 *pt, int w, int h, int scale)
{
	unsigned int center = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int row = center/w;
	int column = center%w;
	if ((row<h-1) && (column<w-2))
	{

		if ((row%2==1) && (column%3==1))
		{
			if (((id[center]&0x23)==0) && ((id[center+1]&0x39)==32))
			{
				//case 1
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);

			}
			if (((id[center]&0x33)==16) && ((id[center+1]&0x19)==0))
			{
				// case 2
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 1, 0, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x33)==16) && ((id[center+1]&0x39)==32))
			{
				// case 3
				//printf("center %d\n", center);
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 0.75, 0.75, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
			}
			if (((id[center]&0x3F)==32) && ((id[center+1]&0x30)==0))
			{
				// case 4
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.25, 1.25, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1);
			}
			if (((id[center]&0x3F)==32) && ((id[center+1]&0x39)==32))
			{
				// case 5
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 0.75, 0.75, 0.25, 1.25, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x3F)==41) && ((id[center+1]&0x18)==0))
			{
				// case 6
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.25, 1.25, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 1, 0, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1);
			}
			if (((id[center]&0x3F)==41) && ((id[center+1]&0x39)==32))
			{
				//case 7
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 0.75, 0.75, 0.25, 1.25, 0, 1, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 1, 0, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x3F)==38) && ((id[center+1]&0x08)==8))
			{
				// case 8
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x20)==0) && ((id[center+1]&0x3F)==16))
			{
				// case 9
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 0.75, 1.25, 0, 1, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x13)==16) && ((id[center+1]&0x3F)==16))
			{
				// case 10
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 0.75, 1.25, 0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.25, -0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x22)==0) && ((id[center+1]&0x3F)==19))
			{
				// case 11
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.75, 1.25, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.25, -1, -1);
			}
			if (((id[center]&0x13)==16) && ((id[center+1]&0x3F)==19))
			{
				// case 12
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.75, 1.25, 0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.25, -0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.25, -1, -1);
			}
			if (((id[center]&0x02)==2) && ((id[center+1]&0x3F)==28))
			{
				// case 13
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], 0, 0, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x3F)==32) && ((id[center+1]&0x3F)==16))
			{
				// case 14
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 0.75, 1.25, 0.25, 1.25, -0.25, 0.75);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x3F)==38) && ((id[center+1]&0x3F)==28))
			{
				// case 15
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -0.25, 0.75);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], 0, 0, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);


			}
			if (((id[center]&0x3F)==41) && ((id[center+1]&0x3F)==16))
			{
				// case 16
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 0.75, 1.25, 0.25, 1.25, 0, 1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 0.25, 0.75, 1, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 1, 0, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);

			}
			if (((id[center]&0x3F)==32) && ((id[center+1]&0x3F)==19))
			{
				// case 17
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.75, 1.25, 0.25, 1.25, -0.25, 0.75);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.25, 0.75, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0, 0, 1, 0, 1, 1, 0, 1, -1, -1);

			}
			if (((id[center]&0x3F)==41) && ((id[center+1]&0x3F)==19))
			{
				// case 18
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.25, 1.25, 0.75, 1.25, 0, 1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 1, 0, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.25, -1, -1);
			}

		}

	}
}
	__global__ void
d_voronoi_generation_r1c2(uint *id, float2 *pt, int w, int h, int scale)
{
	unsigned int center = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int row = center/w;
	int column = center%w;
	if ((row<h-1) && (column<w-2))
	{

		if ((row%2==1) && (column%3==2))
		{
			if (((id[center]&0x23)==0) && ((id[center+1]&0x39)==32))
			{
				//case 1
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);

			}
			if (((id[center]&0x33)==16) && ((id[center+1]&0x19)==0))
			{
				// case 2
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 1, 0, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x33)==16) && ((id[center+1]&0x39)==32))
			{
				// case 3
				//printf("center %d\n", center);
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 0.75, 0.75, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
			}
			if (((id[center]&0x3F)==32) && ((id[center+1]&0x30)==0))
			{
				// case 4
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.25, 1.25, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1);
			}
			if (((id[center]&0x3F)==32) && ((id[center+1]&0x39)==32))
			{
				// case 5
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 0.75, 0.75, 0.25, 1.25, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x3F)==41) && ((id[center+1]&0x18)==0))
			{
				// case 6
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.25, 1.25, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 1, 0, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 1, 0, 1, 1, 0, 1, -1, -1, -1, -1);
			}
			if (((id[center]&0x3F)==41) && ((id[center+1]&0x39)==32))
			{
				//case 7
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 0.75, 0.75, 0.25, 1.25, 0, 1, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 1, 0, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x3F)==38) && ((id[center+1]&0x08)==8))
			{
				// case 8
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x20)==0) && ((id[center+1]&0x3F)==16))
			{
				// case 9
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 0.75, 1.25, 0, 1, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x13)==16) && ((id[center+1]&0x3F)==16))
			{
				// case 10
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 0.75, 1.25, 0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.25, -0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x22)==0) && ((id[center+1]&0x3F)==19))
			{
				// case 11
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.75, 1.25, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.25, -1, -1);
			}
			if (((id[center]&0x13)==16) && ((id[center+1]&0x3F)==19))
			{
				// case 12
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.75, 1.25, 0.25, 0.75, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0, 0, 0.25, -0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.25, -1, -1);
			}
			if (((id[center]&0x02)==2) && ((id[center+1]&0x3F)==28))
			{
				// case 13
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], 0, 0, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x3F)==32) && ((id[center+1]&0x3F)==16))
			{
				// case 14
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 0.75, 1.25, 0.25, 1.25, -0.25, 0.75);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
			}
			if (((id[center]&0x3F)==38) && ((id[center+1]&0x3F)==28))
			{
				// case 15
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 1, 1, 0, 1, -0.25, 0.75);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 1, 1, 0.25, 0.75, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], 0, 0, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);


			}
			if (((id[center]&0x3F)==41) && ((id[center+1]&0x3F)==16))
			{
				// case 16
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1.25, 0.75, 0.75, 1.25, 0.25, 1.25, 0, 1);
				setNodeValues(&pt[(center+2)*6], 0, 0, 1, 0, 0.25, 0.75, 1, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 1, 0, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0.25, -0.25, 1, 0, 1, 1, 0, 1, -1, -1);

			}
			if (((id[center]&0x3F)==32) && ((id[center+1]&0x3F)==19))
			{
				// case 17
				setNodeValues(&pt[center*6], 0, 0, 1, 0, 0.75, 0.75, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.75, 1.25, 0.25, 1.25, -0.25, 0.75);
				setNodeValues(&pt[(center+w)*6], 0, 0, 0.75, -0.25, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.25, 0.75, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], -0.25, 0.25, 0, 0, 1, 0, 1, 1, 0, 1, -1, -1);

			}
			if (((id[center]&0x3F)==41) && ((id[center+1]&0x3F)==19))
			{
				// case 18
				setNodeValues(&pt[(center+1)*6], 0, 0, 1, 0, 1, 1, 0.25, 1.25, 0.75, 1.25, 0, 1);
				setNodeValues(&pt[(center+w)*6], 0, 0, 1, 0, 1.25, 0.25, 1, 1, 0, 1, -1, -1);
				setNodeValues(&pt[(center+w+1)*6], 0.25, 0.25, 0.75, 0.25, 1, 1, 0, 1, -1, -1, -1, -1);
				setNodeValues(&pt[(center+w+2)*6], 0, 0, 1, 0, 1, 1, 0, 1, -0.25, 0.25, -1, -1);
			}

		}

	}

}
*/
//All three passes finished the reshaping of the original pixel art.

//TODO: Pass Four, Curve Extraction

// Final Pass, Render to pbo

	__global__ void
d_render_to_pbo(uint *id, float2 *pt, uint *od, int w, int h, int scale)
{
	unsigned int center = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int row = center/w;
	int column = center%w;
	int center2 = (center/w)*scale*scale*w+(center%w)*scale;
	//int n_corners = 0;
	int target = 0;
	//if (row>0 && row<w-1 && column>0 && column<h-1)
	{

		for ( int i = 0; i < scale; ++i )
		{
			for ( int j = 0; j < scale; ++j )
			{
				target = 0;
				bool flag = false;
				for ( int k = -1; k < 2; ++k )
				{
					for ( int p = -1; p < 2; ++p )
					{
						if (row + k < 0 || row + k > h -1 || column + p < 0 || column + p > w - 1)
							continue;
						int index = center + k*w + p;
						//n_corners = 0;
						//while (pt[index*6+n_corners].x > -0.5 && n_corners < 6)
						//	++n_corners;
						float2 src;
						src.x = (((float)j+0.5f)/(float)scale)-(float)p;
						src.y = (((float)i+0.5f)/(float)scale)-(float)k;
						//if (center == 0)
						//printf("index %d i %d j %d: srcx %f srcy %f n_corners %d\n", index, i, j, src.x, src.y, n_corners);
						if (isPointInPolygon(src, &pt[index*8]))
						{
							//if (center == 0 && index == 8)
							//	printf("center %d, srcx %f, srcy %f, n_corners, %d index %d\n", center, src.x, src.y, n_corners, index);
							target = index;
							flag = true;
							break;
							//od[center2+i*w*scale+j] = id[index]>>8;
						}
						//if (n_corners != 4)
						//printf("row %d column %d id %d\n", row, column, n_corners);
					}
					if (flag)
						break;
				}
				od[center2+i*w*scale+j] = yuvTorgba(id[target]>>8);
				//if ((i%2)^(j%2))
				//	od[center2+i*w*scale+j] = 255;
				//else
				//	od[center2+i*w*scale+j] = 0;
			}
		}
	}
	//int fu = 48;
	//printf("%f %f %f %f %f %f %f %f %f %f %f %f\n", pt[fu+0].x, pt[fu+0].y, pt[fu+1].x, pt[fu+1].y,
	//									      pt[fu+2].x, pt[fu+2].y, pt[fu+3].x, pt[fu+3].y,
	//										  pt[fu+4].x, pt[fu+4].y, pt[fu+5].x, pt[fu+5].y);
}

	extern "C" 
void initTexture(int width, int height, void *pImage, void *pResult)
{
	int size = width * height * sizeof(unsigned int);

	// copy image data to array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	cutilSafeCall( cudaMallocArray  ( &d_array, &channelDesc, width, height )); 
	cutilSafeCall( cudaMemcpyToArray( d_array, 0, 0, pImage, size, cudaMemcpyHostToDevice));
	cutilSafeCall( cudaMallocArray  ( &d_tempArray,   &channelDesc, width, height )); 

}

	extern "C"
void freeTextures()
{
	cutilSafeCall(cudaFreeArray(d_array));
	cutilSafeCall(cudaFreeArray(d_tempArray));
}

	extern "C"
double connectivityDetection(uint *d_temp, unsigned int *d_dest, unsigned int * d_dest2, float2* d_point, int width, int height, int scale, int nthreads)
{
	cutilSafeCall( cudaBindTextureToArray(rgbaTex, d_array));

	// var for kernel computation timing
	double dKernelTime;

	// sync host and start kernel computation timer
	dKernelTime = 0.0;
	cutilSafeCall(cutilDeviceSynchronize());
	shrDeltaT(0);

	//ping-pong data while doing processing works
	d_check_connect<<<height*width/nthreads, nthreads, 0>>>(d_temp, d_dest, width, height);
	d_eliminate_crosses<<<height*width/nthreads, nthreads, 0>>>(d_dest, d_temp, width, height);

	//voronoi generation
	d_voronoi_generation<<<height*width/nthreads, nthreads, 0>>>(d_temp, d_point, width, height, scale);
	//d_voronoi_generation_r0c1<<<height*width/nthreads, nthreads, 0>>>(d_temp, d_point, width, height, scale);
	//d_voronoi_generation_r0c2<<<height*width/nthreads, nthreads, 0>>>(d_temp, d_point, width, height, scale);
	//d_voronoi_generation_r1c0<<<height*width/nthreads, nthreads, 0>>>(d_temp, d_point, width, height, scale);
	//d_voronoi_generation_r1c1<<<height*width/nthreads, nthreads, 0>>>(d_temp, d_point, width, height, scale);
	//d_voronoi_generation_r1c2<<<height*width/nthreads, nthreads, 0>>>(d_temp, d_point, width, height, scale);

	//render to a larger pbo
	d_render_to_pbo<<<height*width/nthreads, nthreads, 0>>>(d_temp, d_point, d_dest2, width, height, scale);
	// sync host and stop computation timer
	cutilSafeCall( cutilDeviceSynchronize() );
	dKernelTime += shrDeltaT(0);

	return (dKernelTime);
}


#endif // #ifndef _BOXFILTER_KERNEL_H_
