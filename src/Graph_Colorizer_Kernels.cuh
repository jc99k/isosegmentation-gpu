#include "kernel_constants.h"
#include "Neighbor_kernels.cuh"
#include "Filter_Kernels.cuh"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
///* GRAPH COLORIZATION KERNELS */
//

/* FOR GRAPHS */
template<class XSpace>
__global__ void rng_generate_kernel(float *result)
{
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < V; i += blockDim.x*gridDim.x)
	{
		curandState localState;
		/* Each thread gets a different seed, no sequence number, no offset (a sacrifice necessary for speed)*/
		curand_init(i, 0, 0, &localState);
		/* Generate pseudo-random floats */
		result[i] = curand_uniform(&localState);
	}
}

template<class XSpace>
__global__ void color_jpl_kernel(int c, XSpace::NeighborsType *neighbors, const float *randoms, int *colors)
{
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < V; i += blockDim.x*gridDim.x) {
		bool f = true; // true iff you have max random

		// ignore nodes colored earlier
		if (colors[i] != -1) continue;

		// look at neighbors to check their random number
		/*int *p = &nodeedges[i].x;
		for (int k = 0; k < NCellNeighbors; k++, p++) {
			if (*p == -1) continue;
			int j = edgenodes[*p].x; if (i == j) j = edgenodes[*p].y;*/
			

		// look at neighbors to check their random number
		for (int k = 0; k < NCellNeighbors; ++k) {
			int j = get_neighbor_id<XSpace>(i, k, neighbors);
			if (j == -1) continue;

			// ignore nodes colored earlier (and yourself)
			int jc = colors[j];
			if (((jc != -1) && (jc != c)) || (i == j)) continue;
			if (randoms[i] <= randoms[j]) f = false;
		}

		// assign color if you have the maximum random number
		if (f) colors[i] = c;
	}
}

template<class XSpace>
__global__ void correct_color_kernel(int c, XSpace::NeighborsType *neighbors, int *colors)
{
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < V; i += blockDim.x*gridDim.x) {
		if (colors[i] != c) continue;

		for (int t = 0; t <= NCellNeighbors; ++t) {
			int mark = 1;

			for (int k = 0; k < NCellNeighbors; ++k) {
				int j = get_neighbor_id<XSpace>(i, k, neighbors);
				if (j == -1) continue;

				if (colors[j] == t) {
					mark = 0;
					break;
				}
			}

			if (mark) {
				colors[i] = t;
				break;
			}
		}
	}
}

//__global__ void correct_color_kernel_test(const int *Ao, const int *Ac, int *colors)
//{
//	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < V; i += blockDim.x*gridDim.x) {
//		for (int k = Ao[i]; k < Ao[i + 1]; k++) {
//			if (colors[Ac[k]] == colors[i]) {
//				printf("Cell %i is NOT okay\n", i);
//				break;
//			}
//		}
//	}
//}

__global__ void count_left(const int *src, int n, int val, int *ctr) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n) return;

	if (src[i] == val)
		atomicAggInc(ctr);
}

///////////////////////////////////////////////////////

/* FOR IMAGES */
template<class XSpace>
__global__ void colorize_Image_kernel(int *updatepatterns)
{
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < V; i += blockDim.x*gridDim.x)
	{
		int temp = i / Img_width;
		
		updatepatterns[i] = ((i % Img_width) & 1) + ((temp % Img_height) & 1)*2 + ((temp / Img_height) & 1)*4;

		//if (i == 345786) printf("cell = %i, updpattern = %i\n", i, updatepatterns[i]);
	}
}