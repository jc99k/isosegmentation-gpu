#ifndef ISOSEGMENTATION_CUDA_KERNEL_CONSTANTS_H
#define ISOSEGMENTATION_CUDA_KERNEL_CONSTANTS_H

#define WARP_BITS 5
#define WARP_SIZE (1 << WARP_BITS)
#define MAX_THREADS_PER_BLOCK 1024
#define NUM_COLORPATTERN_SAMPLES WARP_SIZE*200
#define NUM_WARPS_FOR_COLORPATTERN 1
#define ALPHA 0.45f

#include "thrust_headers.h"
#include <stdio.h>
#include <curand_kernel.h>
#include <helper_math.h>

__constant__ int V;
__constant__ int E;
__constant__ int DV;
__constant__ int Img_width;
__constant__ int Img_height;
__constant__ int Img_depth;
__constant__ int Img_XYarea;
__constant__ int Vol;
__constant__ int Dim;
__constant__ int NCellVertices;
__constant__ int NCellNeighbors;
__constant__ int NFacetVertices;
__constant__ int NGraphColors;
__constant__ int Num_isovalues;
__constant__ float Isovalues[16];
__constant__ float Threshold;
__constant__ float Inv_width;
__constant__ float Inv_heightxwidth;

#endif // ISOSEGMENTATION_CUDA_KERNEL_CONSTANTS_H