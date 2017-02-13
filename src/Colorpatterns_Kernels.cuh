#include "kernel_constants.h"
#include "Texture_Kernels.cuh"

/*------------------------------------------*/
/* FLIP Y-AXIS */

template<class XSpace>
__global__ void flip_YAxis_Graph_kernel(XSpace::PointCoordsType *delaunay_vertices)
{
	float *p;
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < DV; i += blockDim.x*gridDim.x) {
		p = (float*)&delaunay_vertices[i];
		p[1] = Img_height - p[1];
	}
}

template<class XSpace>
__global__ void flip_YAxis_Image_kernel(XSpace::PointCoordsType *delaunay_vertices)
{
	int y_i, swap_pos;
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < DV; i += blockDim.x*gridDim.x) {
		y_i = (i / Img_width) % Img_height;
		swap_pos = i + Img_width*(Img_height - 1 - 2 * y_i);
		if (y_i < Img_height / 2){
			float temp = delaunay_vertices[i];
			delaunay_vertices[i] = delaunay_vertices[swap_pos];
			delaunay_vertices[swap_pos] = temp;
		}
	}
}


/*------------------------------------------*/
/* COMPUTE CELL SIZES */

template<class XSpace>
__global__ void compute_cellsizes_kernel(XSpace::CellVerticesType *cell_vertices, XSpace::PointCoordsType *delaunay_vertices, float *cell_sizes);

template<>
__global__ void compute_cellsizes_kernel<CDelaunay_2D_Cuda_XGraph_Adaptor>(int3 *cell_vertices, float2 *delaunay_vertices, float *cell_sizes)
{
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < V; i += blockDim.x*gridDim.x) {

		float2 *v[3];
		v[0] = &delaunay_vertices[cell_vertices[i].x];
		v[1] = &delaunay_vertices[cell_vertices[i].y];
		v[2] = &delaunay_vertices[cell_vertices[i].z];

		cell_sizes[i] = 0.5*fabsf((v[0]->x)*(v[1]->y - v[2]->y) + (v[1]->x)*(v[2]->y - v[0]->y) + (v[2]->x)*(v[0]->y - v[1]->y));
	}
}

template<>
__global__ void compute_cellsizes_kernel<CDelaunay_3D_Cuda_XGraph_Adaptor>(int4 *cell_vertices, float3 *delaunay_vertices, float *cell_sizes)
{
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < V; i += blockDim.x*gridDim.x) {

		float3 *v[4];
		v[0] = &delaunay_vertices[cell_vertices[i].x];
		v[1] = &delaunay_vertices[cell_vertices[i].y];
		v[2] = &delaunay_vertices[cell_vertices[i].z];
		v[3] = &delaunay_vertices[cell_vertices[i].w];

		float a00, a01, a02, a10, a11, a12, a20, a21, a22;
		a00 = v[1]->x - v[0]->x;	a01 = v[1]->y - v[0]->y;	a02 = v[1]->z - v[0]->z;
		a10 = v[2]->x - v[0]->x;	a11 = v[2]->y - v[0]->y;	a12 = v[2]->z - v[0]->z;
		a20 = v[3]->x - v[0]->x;	a21 = v[3]->y - v[0]->y;	a22 = v[3]->z - v[0]->z;

		cell_sizes[i] = fabsf((a00*a11 - a10*a01)*a22 - (a00*a21 - a20*a01)*a12 + (a10*a21 - a20*a11)*a02) / 6.0;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
///* COLOR PATTERNS COMPUTATION KERNELS */
//

/*------------------------------------------*/
/* GENERATE A UNIFORM RANDOM DISTRIBUTION OF POINTS INSIDE A CELL (BARYCENTRIC COORDINATES) */
template<class XSpace>
__global__ void generate_sample_points_kernel(XSpace::BaryCoordsType *sampleBaryCoords)
{
	float s, t, u;
	float *sample;
	curandState randomState;
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < NUM_COLORPATTERN_SAMPLES; i += blockDim.x*gridDim.x){
		curand_init(i, 0, 0, &randomState);
		sample = &sampleBaryCoords[i].x;

		switch (Dim) {
		case 2:
			s = sqrtf(curand_uniform(&randomState));
			t = curand_uniform(&randomState);

			sample[0] = 1 - s;
			sample[1] = s*(1 - t);
			sample[2] = s*t;
			//printf("random[%i] = %.2f, %.2f, %.2f\n", i, sample[0], sample[1], sample[2]);
			break;

		case 3:
			//s = curand_uniform(&randomState);
			//t = curand_uniform(&randomState);
			//u = curand_uniform(&randomState);

			//if (s + t > 1.0) { // cut'n fold the cube into a prism
			//	s = 1.0 - s;
			//	t = 1.0 - t;
			//}

			//if (t + u > 1.0) { // cut'n fold the prism into a tetrahedron
			//	float tmp = u;
			//	u = 1.0 - s - t;
			//	t = 1.0 - tmp;
			//}
			//else if (s + t + u > 1.0) {
			//	float tmp = u;
			//	u = s + t + u - 1.0;
			//	s = 1 - t - tmp;
			//}

			//sample[0] = s;
			//sample[1] = t;
			//sample[2] = u;
			//sample[3] = 1 - s - t - u;

			s = cbrtf(curand_uniform(&randomState));
			t = sqrtf(curand_uniform(&randomState));
			u = curand_uniform(&randomState);

			sample[0] = 1 - s;
			sample[1] = s*(1 - t);
			sample[2] = s*t*(1 - u);
			sample[3] = s*t*u;

			//printf("random[%i] = %.2f, %.2f, %.2f, %.2f\n", i, *sample, *(sample + 1), *(sample + 2), *(sample + 3));
			break;
		}
	}
}

/*------------------------------------------*/
/* COMPUTE COLOR PATTERNS */
template<class XSpace>
__global__ void compute_colorpatterns_kernel(cudaTextureObject_t texObj, float *colorpatterns, XSpace::CellVerticesType *cell_vertices,
	XSpace::PointCoordsType *delaunay_vertices, XSpace::BaryCoordsType *sampleBaryCoords, float *cell_sizes);

///*--------2D VERSION--------*/
template<>
__global__ void compute_colorpatterns_kernel<CDelaunay_2D_Cuda_XGraph_Adaptor>(cudaTextureObject_t texObj, float *colorpatterns, CDelaunay_2D_Cuda_XGraph_Adaptor::CellVerticesType *cell_vertices,
	CDelaunay_2D_Cuda_XGraph_Adaptor::PointCoordsType *delaunay_vertices, CDelaunay_2D_Cuda_XGraph_Adaptor::BaryCoordsType *sampleBaryCoords, float *cell_sizes)
{
	__shared__ float shmem_colorpatterns[WARP_SIZE];

	int warp_id = threadIdx.x >> WARP_BITS;
	int lane_id = threadIdx.x & (WARP_SIZE - 1);
	float2* v[3];
	float3* bary;
	float cp = 0;
	float2 p;

	if (warp_id == 0) shmem_colorpatterns[lane_id] = 0;
	v[0] = &delaunay_vertices[cell_vertices[blockIdx.x].x];
	v[1] = &delaunay_vertices[cell_vertices[blockIdx.x].y];
	v[2] = &delaunay_vertices[cell_vertices[blockIdx.x].z];

	//int num_samples_adjusted = ((ceilf(cell_sizes[blockIdx.x]) - 1) / WARP_SIZE + 1)*WARP_SIZE;
	//int num_samples_adjusted = fminf(((ceilf(cell_sizes[blockIdx.x]) - 1) / (WARP_SIZE) + 1) * (WARP_SIZE), NUM_COLORPATTERN_SAMPLES);
	int num_samples_adjusted = fminf(ceilf(cell_sizes[blockIdx.x])*32, NUM_COLORPATTERN_SAMPLES);
	//if (threadIdx.x == 0) printf("%i %i\n", blockIdx.x, *sample, *(sample + 1), *(sample + 2), *(sample + 3));
	//int num_samples_adjusted = NUM_COLORPATTERN_SAMPLES;
	//for (int i = threadIdx.x; i < NUM_COLORPATTERN_SAMPLES; i += blockDim.x){
	for (int i = threadIdx.x; i < num_samples_adjusted; i += blockDim.x){
		bary = &sampleBaryCoords[i];
		p.x = (bary->x)*(v[0]->x) + (bary->y)*(v[1]->x) + (bary->z)*(v[2]->x);
		p.y = (bary->x)*(v[0]->y) + (bary->y)*(v[1]->y) + (bary->z)*(v[2]->y);
		cp += tex<CDelaunay_2D_Cuda_XGraph_Adaptor>(texObj, p);
		//printf("%.2f, %.2f\n", p.x, p.y);
	}

	__syncthreads();
	for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
		cp += __shfl_down(cp, offset);

	if (NUM_WARPS_FOR_COLORPATTERN > 1)
	{
		if (lane_id == 0)
			shmem_colorpatterns[warp_id] = cp;
		__syncthreads();

		if (warp_id == 0) {
			cp = shmem_colorpatterns[lane_id];
			for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
				cp += __shfl_down(cp, offset);

			cp /= NUM_COLORPATTERN_SAMPLES;
			if (lane_id == 0) colorpatterns[blockIdx.x] = cp;
		}
	}
	else
	{
		cp /= num_samples_adjusted;
		//cp /= NUM_COLORPATTERN_SAMPLES;
		if (lane_id == 0) colorpatterns[blockIdx.x] = cp;
	}
}

///*--------3D VERSION--------*/
template<>
__global__ void compute_colorpatterns_kernel<CDelaunay_3D_Cuda_XGraph_Adaptor>(cudaTextureObject_t texObj, float *colorpatterns, CDelaunay_3D_Cuda_XGraph_Adaptor::CellVerticesType *cell_vertices,
	CDelaunay_3D_Cuda_XGraph_Adaptor::PointCoordsType *delaunay_vertices, CDelaunay_3D_Cuda_XGraph_Adaptor::BaryCoordsType *sampleBaryCoords, float *cell_sizes)
{
	__shared__ float shmem_colorpatterns[WARP_SIZE];

	int warp_id = threadIdx.x >> WARP_BITS;
	int lane_id = threadIdx.x & (WARP_SIZE - 1);
	float3* v[4];
	float4* bary;
	float cp = 0;
	float3 p;

	if (warp_id == 0) shmem_colorpatterns[lane_id] = 0;
	v[0] = &delaunay_vertices[cell_vertices[blockIdx.x].x];
	v[1] = &delaunay_vertices[cell_vertices[blockIdx.x].y];
	v[2] = &delaunay_vertices[cell_vertices[blockIdx.x].z];
	v[3] = &delaunay_vertices[cell_vertices[blockIdx.x].w];

	//int num_samples_adjusted = fminf(((ceilf(cell_sizes[blockIdx.x]) - 1) / WARP_SIZE + 1) * WARP_SIZE, NUM_COLORPATTERN_SAMPLES);
	int num_samples_adjusted = fminf(ceilf(cell_sizes[blockIdx.x]) * 1, NUM_COLORPATTERN_SAMPLES);
	for (int i = threadIdx.x; i < num_samples_adjusted; i += blockDim.x){
	//for (int i = threadIdx.x; i < NUM_COLORPATTERN_SAMPLES; i += blockDim.x){
		bary = &sampleBaryCoords[i];
		p.x = (bary->x)*(v[0]->x) + (bary->y)*(v[1]->x) + (bary->z)*(v[2]->x) + (bary->w)*(v[3]->x);
		p.y = (bary->x)*(v[0]->y) + (bary->y)*(v[1]->y) + (bary->z)*(v[2]->y) + (bary->w)*(v[3]->y);
		p.z = (bary->x)*(v[0]->z) + (bary->y)*(v[1]->z) + (bary->z)*(v[2]->z) + (bary->w)*(v[3]->z);
		cp += tex<CDelaunay_3D_Cuda_XGraph_Adaptor>(texObj, p);
	}

	__syncthreads();
	for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
		cp += __shfl_down(cp, offset);

	if (NUM_WARPS_FOR_COLORPATTERN > 1)
	{
		if (lane_id == 0)
			shmem_colorpatterns[warp_id] = cp;
		__syncthreads();

		if (warp_id == 0) {
			cp = shmem_colorpatterns[lane_id];
			for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
				cp += __shfl_down(cp, offset);

			cp /= NUM_COLORPATTERN_SAMPLES;
			if (lane_id == 0) colorpatterns[blockIdx.x] = cp;
		}
	}
	else
	{
		cp /= num_samples_adjusted;
		//cp /= NUM_COLORPATTERN_SAMPLES;
		if (lane_id == 0) colorpatterns[blockIdx.x] = cp;
	}
}

/*------------------------------------------*/
