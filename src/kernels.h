
#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <helper_math.h>
//#include "cutil_math.h"

__constant__ int V;
__constant__ int E;
__constant__ int DV;
__constant__ int Img_width;
__constant__ int Img_height;
__constant__ float Isovalues[64];
__constant__ int Num_isovalues;

///////////////////////////////////////////////////////

__device__ float similarity(float i, float j) { 
	//for (int k = 0; k<Num_isovalues; k++)
	//{
	//	if( (i - Isovalues[k])*(j - Isovalues[k]) <= 0 )
	//		return 1.0f;
	//}
	return 1.0f-expf(-1*abs(i-j)/50); 
} 

///////////////////////////////////////////////////////

/* GPU RANDOM NUMBER GENERATION KERNELS */

__global__ void rng_generate_kernel(float *result)
{
	for (int i = threadIdx.x+blockIdx.x*blockDim.x ; i < V ; i += blockDim.x*gridDim.x)
	{
		curandState localState;
		/* Each thread gets same seed, a different sequence number, no offset */
		curand_init(clock64(), i, 0, &localState);
		/* Generate pseudo-random floats */
		result[i] = curand_uniform(&localState);
	}
}

///////////////////////////////////////////////////////

///* COLORING KERNEL */

__global__ void color_jpl_kernel(int c, const int *Ao, const int *Ac, const float *randoms, int *colors)
{
	for (int i = threadIdx.x+blockIdx.x*blockDim.x ; i < V ; i += blockDim.x*gridDim.x) {
		bool f=true; // true iff you have max random

		// ignore nodes colored earlier
		if (colors[i] != -1) continue;

		float ir = randoms[i];

		// look at neighbors to check their random number
		for (int k = Ao[i]; k < Ao[i+1]; k++) {
			// ignore nodes colored earlier (and yourself)
			int j = Ac[k];
			int jc = colors[j];
			if (((jc != -1) && (jc != c)) || (i == j)) continue;
			float jr = randoms[j];
			if (ir <= jr) f=false;
		}

		// assign color if you have the maximum random number
		if (f) colors[i] = c;
	}
}

/////////////////////////////////////////////////////////

/* GEOMETRIC ADJUSTMENTS KERNELS */

//template<class XSpace>
__global__ void flip_verticality_kernel(float2 *delaunay_vertices)
{
	// Cuda-style for
	for (int i = threadIdx.x+blockIdx.x*blockDim.x ; i < DV ; i += blockDim.x*gridDim.x) {
		delaunay_vertices[i].y = Img_height - delaunay_vertices[i].y;
	}
}

__global__ void flip_verticality_kernel_3D(float3 *delaunay_vertices)
{
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < DV; i += blockDim.x*gridDim.x) {
		delaunay_vertices[i].y = Img_height - delaunay_vertices[i].y;
	}
}

//
//template<class G, kernel K>
__global__ void compute_cellcentroids_kernel(const int *Ao, const int *Ac, int3 *cell_vertices, float2 *delaunay_vertices, float2 *cell_centroids)
{
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < V; i += blockDim.x*gridDim.x) {

		// Cell Centroid
		cell_centroids[i] = (delaunay_vertices[cell_vertices[i].x] + delaunay_vertices[cell_vertices[i].y] + delaunay_vertices[cell_vertices[i].z]) / 3.0f;
		//printf("Ccentroid[%i] = (%.2f,%.2f)\n", i, cell_centroids[i].x, cell_centroids[i].y);
	}
}

__global__ void compute_cellcentroids_kernel_3D(const int *Ao, const int *Ac, int4 *cell_vertices, float3 *delaunay_vertices, float3 *cell_centroids)
{
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < V; i += blockDim.x*gridDim.x) {

		// Cell Centroid
		cell_centroids[i] = (delaunay_vertices[cell_vertices[i].x] + delaunay_vertices[cell_vertices[i].y] + delaunay_vertices[cell_vertices[i].z] + delaunay_vertices[cell_vertices[i].w]) / 4.0f;
		//printf("Ccentroid[%i] = (%.2f,%.2f,%.2f)\n", i, cell_centroids[i].x, cell_centroids[i].y, cell_centroids[i].z);
	}
}

__global__ void compute_facetcentroids_kernel(const int *Ao, const int *Ac, int3 *cell_vertices, float2 *delaunay_vertices, float2 *facet_centroids)
{
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < V; i += blockDim.x*gridDim.x) {

		// Facet Centroids, for each neighbor
		for (int k = Ao[i]; k < Ao[i+1]; ++k) {

			float2 fcentroid;
			fcentroid.x = fcentroid.y = 0.0f;
			int* ptr1 = &cell_vertices[i].x;
			int* ptr2 = &cell_vertices[Ac[k]].x;

			for(int m=0; m<3; m++) {
				for( int n=0; n<3; n++) {
					if (*(ptr1 + m) == *(ptr2 + n)) {
						fcentroid.x += delaunay_vertices[*(ptr1 + m)].x;
						fcentroid.y += delaunay_vertices[*(ptr1 + m)].y;
					}
				}
			}

			facet_centroids[k] = fcentroid/2;

			//printf("Facet centroid[%i,%i] = (%.2f,%.2f)\n", i, Ac[k], facet_centroids[k].x, facet_centroids[k].y);
		}
	}
}

__global__ void compute_facetcentroids_kernel_3D(const int *Ao, const int *Ac, int4 *cell_vertices, float3 *delaunay_vertices, float3 *facet_centroids)
{
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < V; i += blockDim.x*gridDim.x) {

		// Facet Centroids, for each neighbor
		for (int k = Ao[i]; k < Ao[i + 1]; ++k) {

			float3 fcentroid;
			fcentroid.x = fcentroid.y = fcentroid.z = 0.0f;
			int* ptr1 = &cell_vertices[i].x;
			int* ptr2 = &cell_vertices[Ac[k]].x;

			for (int m = 0; m<4; m++) {
				for (int n = 0; n<4; n++) {
					if (*(ptr1 + m) == *(ptr2 + n)) {
						fcentroid.x += delaunay_vertices[*(ptr1 + m)].x;
						fcentroid.y += delaunay_vertices[*(ptr1 + m)].y;
						fcentroid.z += delaunay_vertices[*(ptr1 + m)].z;
					}
				}
			}

			facet_centroids[k] = fcentroid / 3;

			//printf("Facet centroid[%i,%i] = (%.2f,%.2f,%.2f)\n", i, Ac[k], facet_centroids[k].x, facet_centroids[k].y, facet_centroids[k].z);
		}
	}
}
//
//template<class G, kernel K>
__global__ void compute_gammas_kernel(const int *Ao, const int *Ac, const int3 *cell_vertices, const float2 *delaunay_vertices, float *cell_sizes, float *cell_gammas)
{
	// Cuda-style for
	for (int i = threadIdx.x+blockIdx.x*blockDim.x ; i < V ; i += blockDim.x*gridDim.x) {

		float2 a_2 = delaunay_vertices[cell_vertices[i].x] - delaunay_vertices[cell_vertices[i].y];
		float2 b_2 = delaunay_vertices[cell_vertices[i].x] - delaunay_vertices[cell_vertices[i].z];

		float3 a, b;
		a.x = a_2.x; a.y = a_2.y; a.z = 0;
		b.x = b_2.x; b.y = b_2.y; b.z = 0;

		float area_i = 0.5*length(cross(a, b));
		
		cell_sizes[i] = area_i;
		cell_gammas[i] = (1 / V) + (area_i / (Img_width*Img_height));

		printf("[%i] size = %.2f, gamma = %.2f\n", i, cell_sizes[i], cell_gammas[i]);
	}
}

///////////////////////////////////////////////////////

//
///* SEGMENTATION KERNELS */
//
//template<class G, kernel K>
__global__ void segmentation_NCG_kernel(int c, const int *Ao, const int *Ac, const int *colors,
	const float *similarities, int *labels, float threshold)
{
	// Cuda-style for
	for (int i = threadIdx.x+blockIdx.x*blockDim.x;i < V;i += blockDim.x*gridDim.x)	{
		if (colors[i] != c) continue;
		
		// Here starts the segmentation 
		for (int s_i = Ao[i]; s_i < Ao[i+1]; s_i++) {
			int test = (similarities[s_i] <= threshold);
			labels[i] = (((test << 31) >> 31) & min(labels[i], labels[Ac[s_i]])) | (((!test << 31) >> 31) & labels[i]);
			//printf("test = %i , labels[%i] = %i\n", test, i, labels[i]);
		}

		
	}
}

//template<class G, kernel K>
__global__ void path_compression_kernel(int *labels)
{
	// Cuda-style for
	for (int i = threadIdx.x+blockIdx.x*blockDim.x;i < V;i += blockDim.x*gridDim.x)	{
		do { 
			labels[i] = labels[labels[i]];
		} while(labels[i] != labels[labels[i]]);
	}
}
//
//template<class G, kernel K>
//__global__ void relabel_kernel(int *cell_indices, int *sizes, int *offsets, int *labels, int *region_labels)
//{
//	// Cuda-style for
//	for (int i = offsets[blockIdx.x] + threadIdx.x; i < offsets[blockIdx.x] + sizes[blockIdx.x]; i += blockDim.x) {
//		labels[cell_indices[i]] = region_labels[blockIdx.x];
//	}
//}
//
//template<class G, kernel K>
//__global__ void segmentation_NRG_kernel(int nR, int *cell_indices, int *sizes, int *offsets, int *labels, int *Ao, int *Ac, float *region_colorpatterns, float *shared_similarities)
//{
//    extern __shared__ float s[];
//
//	for (int i = threadIdx.x; i < nR; i += blockDim.x)
//	    s[i] = 1.0f;
//
//    // Cuda-style for
//	for (int i = offsets[nR + blockIdx.x] + threadIdx.x; i < offsets[nR + blockIdx.x] + sizes[nR + blockIdx.x]; i += blockDim.x) {
//	    for (int s_i = Ao[cell_indices[i]]; s_i < Ao[cell_indices[i]+1]; s_i++) {
//	    	int neigh = labels[Ac[s_i]];
//	    	if(neigh < nR) {
//	    		s[neigh] = similarity<K>(region_colorpatterns[labels[cell_indices[i]]],region_colorpatterns[neigh]);
//	    	}
//	    }
//	}
//
//	__syncthreads();
//
//	for (int i = threadIdx.x; i < nR; i += blockDim.x)
//	    shared_similarities[nR*blockIdx.x + i] = s[i];
//}
//
//template<class G, kernel K>
//__global__ void min_element_kernel(int nR, int *region_labels, float *shared_similarities, int *min_element_output)
//{
//	int min_element = nR + blockIdx.x;
//	float min_similarity = 1.0f;
//
//	for (int i = 0; i < nR; ++i) {
//		if( shared_similarities[nR*blockIdx.x + i] < min_similarity ) {
//			min_similarity = shared_similarities[nR*blockIdx.x + i];
//			min_element = i;
//		}
//
//	}
//
//	min_element_output[blockIdx.x] = min_element;
//
//	// printf("Region %i, Min_element = %i\n",nR+blockIdx.x,min_element);
//}
//
/////////////////////////////////////////////////////////
//
///* COLOR PATTERNS COMPUTATION KERNEL */
//
//__device__ bool PointInTriangle(float2 p, float2 a, float2 b, float2 c)
//{
//	// Compute vectors        
//	float2 v0 = c - a;
//	float2 v1 = b - a;
//	float2 v2 = p - a;
//
//	// Compute dot products
//	float dot00 = dot(v0,v0);
//	float dot01 = dot(v0,v1);
//	float dot02 = dot(v0,v2);
//	float dot11 = dot(v1,v1);
//	float dot12 = dot(v1,v2);
//
//	// Compute barycentric coordinates
//	float invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
//	float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
//	float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
//
//	// Check if point is in triangle
//	return (u >= 0) && (v >= 0) && (u + v < 1);
//}
//
//template<class G, kernel K>
__global__ void basic_colorpatterns_kernel(cudaTextureObject_t texObj, float *colorpatterns, float2 *cell_centroids)
{
	// Cuda-style for
	for (int i = threadIdx.x+blockIdx.x*blockDim.x ; i < V ; i += blockDim.x*gridDim.x) {
		colorpatterns[i] = tex2D<float>(texObj,cell_centroids[i].x, cell_centroids[i].y);
		//printf("colorpatterns[%i] = %.2f\n", i, colorpatterns[i]);
	}
}

__global__ void basic_colorpatterns_kernel_3D(cudaTextureObject_t texObj, float *colorpatterns, float3 *cell_centroids)
{
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < V; i += blockDim.x*gridDim.x) {
		colorpatterns[i] = tex3D<float>(texObj, cell_centroids[i].x, cell_centroids[i].y, cell_centroids[i].z);
		//colorpatterns[i] = ((float) i / V);
		//printf("centroid[%i] = (%.2f,%.2f,%.2f) ;colorpatterns[%i] = %.2f\n", i, cell_centroids[i].x, cell_centroids[i].y, cell_centroids[i].z, i,colorpatterns[i]);
	}
}

//template<class G, kernel K>
//__global__ void compute_colorpatterns_kernel(cudaTextureObject_t texObj, int3 *cell_vertices, 
//	float2 *delaunay_vertices, float *colorpatterns, int *cell_sizes, float2 *cell_centroids)
//{
//	// Cuda-style for
//	for (int i = threadIdx.x+blockIdx.x*blockDim.x ; i < V ; i += blockDim.x*gridDim.x) {
//
//		float pattern = 0;
//		int numcells = 0;
//		float2 cmin, cmax;
//
//		cmin.x = fminf(delaunay_vertices[cell_vertices[i].x].x, fminf(delaunay_vertices[cell_vertices[i].y].x, delaunay_vertices[cell_vertices[i].z].x));
//		cmin.y = fminf(delaunay_vertices[cell_vertices[i].x].y, fminf(delaunay_vertices[cell_vertices[i].y].y, delaunay_vertices[cell_vertices[i].z].y));
//		cmax.x = fmaxf(delaunay_vertices[cell_vertices[i].x].x, fmaxf(delaunay_vertices[cell_vertices[i].y].x, delaunay_vertices[cell_vertices[i].z].x));
//		cmax.y = fmaxf(delaunay_vertices[cell_vertices[i].x].y, fmaxf(delaunay_vertices[cell_vertices[i].y].y, delaunay_vertices[cell_vertices[i].z].y));
//
//		for(int ci = (int)roundf(cmin.x); ci <= (int)roundf(cmax.x); ci++) {
//			for(int cj = (int)roundf(cmin.y); cj <= (int)roundf(cmax.y); cj++) {
//				if(PointInTriangle( make_float2(ci,cj), delaunay_vertices[cell_vertices[i].x], delaunay_vertices[cell_vertices[i].y], delaunay_vertices[cell_vertices[i].z] )) {
//					pattern += tex2D<float>(texObj,ci,cj);
//					numcells++;
//				}
//			}
//		}
//
//		colorpatterns[i] = pattern/numcells;
//		cell_sizes[i] = numcells;
//
//		if(!numcells){
//			colorpatterns[i] = tex2D<float>(texObj,cell_centroids[i].x, Img_height - cell_centroids[i].y);
//			numcells=1;
//		}
//		// 	printf("Cell = %i, Triangle = [(%i,%i);(%i,%i);(%i,%i)]\n",i
//		// 		,delaunay_vertices[cell_vertices[i].x].x, delaunay_vertices[cell_vertices[i].x].y
//		// 		,delaunay_vertices[cell_vertices[i].y].x, delaunay_vertices[cell_vertices[i].y].y
//		// 		,delaunay_vertices[cell_vertices[i].z].x, delaunay_vertices[cell_vertices[i].z].y);
//	}
//}
//

__device__ bool SameSide(float3 &p, float3 &v1, float3 &v2, float3 &v3, float3 &v4)
{
	//float3 normal = cross(v2 - v1, v3 - v1);
	//float dotV4 = dot(normal, v4 - v1);
	//float dotP = dot_product(normal, p - v1);
	//return ((dotV4 < 0) && (dotP < 0));
	return true;
}

__device__ bool PointInTetrahedron(float3 &p, float3 &v1, float3 &v2, float3 &v3, float3 &v4)
{
	return	(SameSide(p, v1, v2, v3, v4) && SameSide(p, v2, v3, v4, v1) && SameSide(p, v3, v4, v1, v2) && SameSide(p, v4, v1, v2, v3));
}

__global__ void compute_colorpatterns_kernel_3D(cudaTextureObject_t texObj, float *colorpatterns, int4 *cell_vertices,
	float3 *delaunay_vertices, float3 *cell_centroids)
{
	// Cuda-style for
	for (int i = threadIdx.x+blockIdx.x*blockDim.x ; i < V ; i += blockDim.x*gridDim.x) {

		float pattern = 0;
		int numcells = 0;
		float3 cmin, cmax, citer;

		cmin.x = rintf(fminf(delaunay_vertices[cell_vertices[i].x].x, fminf(delaunay_vertices[cell_vertices[i].y].x, fminf(delaunay_vertices[cell_vertices[i].z].x, delaunay_vertices[cell_vertices[i].w].x))));
		cmin.y = rintf(fminf(delaunay_vertices[cell_vertices[i].x].y, fminf(delaunay_vertices[cell_vertices[i].y].y, fminf(delaunay_vertices[cell_vertices[i].z].y, delaunay_vertices[cell_vertices[i].w].y))));
		cmin.z = rintf(fminf(delaunay_vertices[cell_vertices[i].x].z, fminf(delaunay_vertices[cell_vertices[i].y].z, fminf(delaunay_vertices[cell_vertices[i].z].z, delaunay_vertices[cell_vertices[i].w].z))));
		
		cmax.x = rintf(fmaxf(delaunay_vertices[cell_vertices[i].x].x, fmaxf(delaunay_vertices[cell_vertices[i].y].x, fmaxf(delaunay_vertices[cell_vertices[i].z].x, delaunay_vertices[cell_vertices[i].w].x))));
		cmax.y = rintf(fmaxf(delaunay_vertices[cell_vertices[i].x].y, fmaxf(delaunay_vertices[cell_vertices[i].y].y, fmaxf(delaunay_vertices[cell_vertices[i].z].y, delaunay_vertices[cell_vertices[i].w].y))));
		cmax.z = rintf(fmaxf(delaunay_vertices[cell_vertices[i].x].z, fmaxf(delaunay_vertices[cell_vertices[i].y].z, fmaxf(delaunay_vertices[cell_vertices[i].z].z, delaunay_vertices[cell_vertices[i].w].z))));

		

		for(citer.x = cmin.x; citer.x <= cmax.x; citer.x += 1.0) {
			for(citer.y = cmin.y; citer.y <= cmax.y; citer.y += 1.0) {
				for (citer.z = cmin.z; citer.z <= cmax.z; citer.z += 1.0) {
					if (PointInTetrahedron(citer, delaunay_vertices[cell_vertices[i].x], delaunay_vertices[cell_vertices[i].y], delaunay_vertices[cell_vertices[i].z], delaunay_vertices[cell_vertices[i].w])) {
						pattern += tex3D<float>(texObj, citer.x, citer.y, citer.z);
						numcells++;
					}
				}
			}
		}

		colorpatterns[i] = pattern/numcells;
		//printf("[%i] pattern = %.2f, numcells = %i, colorpattern = %.2f\n", i, pattern, numcells, colorpatterns[i]);

		if(!numcells){
			colorpatterns[i] = tex3D<float>(texObj, cell_centroids[i].x, cell_centroids[i].y, cell_centroids[i].z);
			numcells=1;
		}
		// 	printf("Cell = %i, Triangle = [(%i,%i);(%i,%i);(%i,%i)]\n",i
		// 		,delaunay_vertices[cell_vertices[i].x].x, delaunay_vertices[cell_vertices[i].x].y
		// 		,delaunay_vertices[cell_vertices[i].y].x, delaunay_vertices[cell_vertices[i].y].y
		// 		,delaunay_vertices[cell_vertices[i].z].x, delaunay_vertices[cell_vertices[i].z].y);
	}
}

//template<class G, kernel K>
//__global__ void get_boundingbox_tex2D_kernel(cudaTextureObject_t texObj, int3 *cell_vertices, float2 *delaunay_vertices,
//	float2 *minCoords, float2 *diffCoords)
//{
//	// Cuda-style for
//	for (int i = threadIdx.x+blockIdx.x*blockDim.x ; i < V ; i += blockDim.x*gridDim.x) {
//		float2 minC, maxC;
//		minC.x = fminf(delaunay_vertices[cell_vertices[i].x].x, fminf(delaunay_vertices[cell_vertices[i].y].x, delaunay_vertices[cell_vertices[i].z].x));
//		minC.y = fminf(delaunay_vertices[cell_vertices[i].x].y, fminf(delaunay_vertices[cell_vertices[i].y].y, delaunay_vertices[cell_vertices[i].z].y));
//		maxC.x = fmaxf(delaunay_vertices[cell_vertices[i].x].x, fmaxf(delaunay_vertices[cell_vertices[i].y].x, delaunay_vertices[cell_vertices[i].z].x));
//		maxC.y = fmaxf(delaunay_vertices[cell_vertices[i].x].y, fmaxf(delaunay_vertices[cell_vertices[i].y].y, delaunay_vertices[cell_vertices[i].z].y));
//
//		minCoords[i] = minC;
//		diffCoords[i] = maxC - minC;
//
//		// printf("Block %i, Thread %i, minC = (%.2f,%.2f) ; maxC = (%.2f,%.2f) \n",blockIdx.x,threadIdx.x,minC.x,minC.y,maxC.x,maxC.y);
//	}
//}
//
//template<class G, kernel K>
//__global__ void compute_colorpatterns_tex2D_kernel(cudaTextureObject_t texObj, int3 *cell_vertices, float2 *delaunay_vertices,
//	float2 *minCoords, float2 *diffCoords, float *colorpatterns)
//{
//	// extern __shared__ float s[];
// //    float *shared_colorpatterns = &s[0];
// //    int *shared_counts = (int*)&s[blockDim.x];
//
// //    shared_colorpatterns[threadIdx.x] = 0;
//	// shared_counts[threadIdx.x] = 0;
//
//	// // for (int i = threadIdx.x+blockIdx.x*blockDim.x ; i < V ; i += blockDim.x*gridDim.x) {
//
//	// 	float2 sample = minCoords[blockIdx.x];
//	// 	float dx = diffCoords[blockIdx.x].x/blockDim.x;
//	// 	float dy = diffCoords[blockIdx.x].y/512;
//	// 	sample.x += threadIdx.x*dx + dx/2;
//
//	// 	for( int step = 0; step < 512; step++, sample.y+=dy ) {
//	// 		if( PointInTriangle(sample,
//	// 			delaunay_vertices[cell_vertices[blockIdx.x].x], 
//	// 			delaunay_vertices[cell_vertices[blockIdx.x].y], 
//	// 			delaunay_vertices[cell_vertices[blockIdx.x].z]) ) {
//	// 				shared_colorpatterns[threadIdx.x] += tex2D<float>(texObj,sample.x,sample.y);;
//	// 				shared_counts[threadIdx.x]++;
//	// 		}
//	// 	}
//
//	// // }
//	// __syncthreads();
//
//	// // Reduction
//
//	// if (blockDim.x >= 128) { 
//	// 	if (threadIdx.x < 64) {
//	// 		shared_colorpatterns[threadIdx.x] += shared_colorpatterns[threadIdx.x + 64];
//	// 		shared_counts[threadIdx.x] += shared_counts[threadIdx.x + 64];
//	// 	}
//	// 	__syncthreads(); 
//	// }
//	
//	// if (threadIdx.x < 32) {
//	// 	if (blockDim.x >= 64) {
//	// 		shared_colorpatterns[threadIdx.x] += shared_colorpatterns[threadIdx.x + 32];
//	// 		shared_counts[threadIdx.x] += shared_counts[threadIdx.x + 32];
//	// 	}
//	// 	if (blockDim.x >= 32) {
//	// 		shared_colorpatterns[threadIdx.x] += shared_colorpatterns[threadIdx.x + 16];
//	// 		shared_counts[threadIdx.x] += shared_counts[threadIdx.x + 16];
//	// 	}
//	// 	if (blockDim.x >= 16) {
//	// 		shared_colorpatterns[threadIdx.x] += shared_colorpatterns[threadIdx.x + 8];
//	// 		shared_counts[threadIdx.x] += shared_counts[threadIdx.x + 8];
//	// 	}
//	// 	if (blockDim.x >= 8) {
//	// 		shared_colorpatterns[threadIdx.x] += shared_colorpatterns[threadIdx.x + 4];
//	// 		shared_counts[threadIdx.x] += shared_counts[threadIdx.x + 4];
//	// 	}
//	// 	if (blockDim.x >= 4) {
//	// 		shared_colorpatterns[threadIdx.x] += shared_colorpatterns[threadIdx.x + 2];
//	// 		shared_counts[threadIdx.x] += shared_counts[threadIdx.x + 2];
//	// 	}
//	// 	if (blockDim.x >= 2) {
//	// 		shared_colorpatterns[threadIdx.x] += shared_colorpatterns[threadIdx.x + 1];
//	// 		shared_counts[threadIdx.x] += shared_counts[threadIdx.x + 1];
//	// 	}
//	// }
//	
//	// if (threadIdx.x == 0) {
//	// 	colorpatterns[blockIdx.x] = shared_colorpatterns[0];
//	// }
//
//}
//
//template<class G, kernel K>
//__global__ void get_boundingbox_kernel(cudaTextureObject_t texObj, int3 *cell_vertices, int2 *delaunay_vertices,
//	int2 *minCoords, int2 *maxCoords, int *minDims, int *maxDims, int2 *spread_dir, int2 *sweep_dir)
//{
//	// Cuda-style for
//	for (int i = threadIdx.x+blockIdx.x*blockDim.x ; i < V ; i += blockDim.x*gridDim.x) {
//		int2 minC, maxC, diffC;
//		minC.x = min(delaunay_vertices[cell_vertices[i].x].x, min(delaunay_vertices[cell_vertices[i].y].x, delaunay_vertices[cell_vertices[i].z].x));
//		minC.y = min(delaunay_vertices[cell_vertices[i].x].y, min(delaunay_vertices[cell_vertices[i].y].y, delaunay_vertices[cell_vertices[i].z].y));
//		maxC.x = max(delaunay_vertices[cell_vertices[i].x].x, max(delaunay_vertices[cell_vertices[i].y].x, delaunay_vertices[cell_vertices[i].z].x));
//		maxC.y = max(delaunay_vertices[cell_vertices[i].x].y, max(delaunay_vertices[cell_vertices[i].y].y, delaunay_vertices[cell_vertices[i].z].y));
//		diffC = maxC - minC;
//
//		minCoords[i] = minC;
//		maxCoords[i] = maxC;
//
//		minDims[i] = min(diffC.x,diffC.y);
//		maxDims[i] = max(diffC.x,diffC.y);
//
//		if( minDims[i] == diffC.y) {
//			spread_dir[i].x = 1;
//			spread_dir[i].y = 0;
//		}
//		else {
//			spread_dir[i].x = 0;
//			spread_dir[i].y = 1;
//		}
//
//		sweep_dir[i].x = spread_dir[i].y;
//		sweep_dir[i].y = spread_dir[i].x;
//
//		// printf("Block %i, Thread %i, minC = %i,%i ; maxC = %i,%i \n",blockIdx.x,threadIdx.x,minC.x,minC.y,maxC.x,maxC.y);
//	}
//}
//
//template<class G, kernel K>
//__global__ void pixelcell_map_kernel(cudaTextureObject_t texObj, int3 *cell_vertices, int2 *delaunay_vertices, int *pixelcells,
//	int2 *minCoords, int2 *maxCoords)
//{
//	// for (int i =  minCoords[blockIdx.x].x + threadIdx.x; i < maxCoords[blockIdx.x].x ; i += blockDim.x) {
//	// 	for (int j = minCoords[blockIdx.x].y + threadIdx.y ; j < maxCoords[blockIdx.x].y  ; j += blockDim.y) {
//
//	// 		if(PointInTriangle( make_int2(i,j), delaunay_vertices[cell_vertices[blockIdx.x].x], delaunay_vertices[cell_vertices[blockIdx.x].y], delaunay_vertices[cell_vertices[blockIdx.x].z] )) {
//	// 			pixelcells[i+Img_width*j] = blockIdx.x;
//	// 		}
//
//	// 	}
//	// }
//}
//
//template<class G, kernel K>
//__global__ void trianglescan_kernel(cudaTextureObject_t texObj, int *pixelcells,
//	int2 *minCoords, int *minDims, int2 *spread_dir, int2 *sweep_dir, float *aggregate_colorpatterns, int *aggregate_counts)
//{
//	extern __shared__ float s[];
//    float *shared_colorpatterns = &s[0];
//    int *shared_counts = (int*)&s[blockDim.x];
//
//    shared_colorpatterns[threadIdx.x] = 0;
//	shared_counts[threadIdx.x] = 0;
//
//	int2 seed = minCoords[blockIdx.x] + threadIdx.x*spread_dir[blockIdx.x];
//	for (int i = 0; i < minDims[blockIdx.x] ; ++i) {
//		if(pixelcells[seed.x+Img_width*(seed.y)] == blockIdx.x) {
//			shared_colorpatterns[threadIdx.x] += tex2D<float>(texObj,seed.x,seed.y);
//			shared_counts[threadIdx.x]++;
//		}
//		seed += sweep_dir[blockIdx.x];
//	}
//
//	aggregate_colorpatterns[blockDim.x*blockIdx.x+threadIdx.x] = shared_colorpatterns[threadIdx.x];
//	aggregate_counts[blockDim.x*blockIdx.x+threadIdx.x] = shared_counts[threadIdx.x];
//
//}
//
//template<class G, kernel K>
//__global__ void reducescan_kernel(int globalmaxDim, float *colorpatterns, 
//	float *aggregate_colorpatterns, int *aggregate_counts)
//{
//	float cp_result = 0;
//	int count_result = 0;
//	for(int i = 0; i < globalmaxDim; ++i) {
//		cp_result 		+= aggregate_colorpatterns[globalmaxDim*blockIdx.x + i];
//		count_result 	+= aggregate_counts[globalmaxDim*blockIdx.x + i];
//	}
//	colorpatterns[blockIdx.x] = cp_result/count_result;
//
//	if(!count_result)
//		printf("Block %i, colorpattern = %.2f, count = %i\n",blockIdx.x,cp_result,count_result);
//}
//
/////////////////////////////////////////////////////////
//
///* SIMILARITIES COMPUTATION KERNEL */

__device__ bool FindIsovalue(float2 p1, float2 p2, cudaTextureObject_t texObj)
{
	float2 v = p2 - p1;
	int n = rintf(length(v));

	float2 q1 = p1;
	float2 q2 = q1 + (v / n);
	for (int t = 0; t < n; ++t){
		float q1_cp = tex2D<float>(texObj, q1.x, q1.y);
		float q2_cp = tex2D<float>(texObj, q2.x, q2.y);
		for (int k = 0; k<Num_isovalues; k++)
		{
			if ((q1_cp - Isovalues[k])*(q2_cp - Isovalues[k]) <= 0)
				return true;
		}
		q1 = q2;
		q2 += (v / n);
	}
	return false;
}

__device__ bool FindIsovalue_3D(float3 p1, float3 p2, cudaTextureObject_t texObj)
{
	float3 v = p2 - p1;
	int n = rintf(length(v));

	float3 q1 = p1;
	float3 q2 = q1 + (v / n);
	for (int t = 0; t < n; ++t){
		float q1_cp = tex3D<float>(texObj, q1.x, q1.y, q1.z);
		float q2_cp = tex3D<float>(texObj, q2.x, q2.y, q2.z);
		for (int k = 0; k<Num_isovalues; k++)
		{
			if ((q1_cp - Isovalues[k])*(q2_cp - Isovalues[k]) <= 0)
				return true;
		}
		q1 = q2;
		q2 += (v / n);
	}
	return false;
}

//template<class G, kernel K>
__global__ void compute_similarity_kernel(const int *Ao, const int *Ac, cudaTextureObject_t texObj, float2 *cell_centroids, float2 *facet_centroids, const float *colorpatterns, float *similarities)
{
	// Cuda-style for
	for (int i = threadIdx.x+blockIdx.x*blockDim.x ; i < V ; i += blockDim.x*gridDim.x) {
		// look at neighbors
		for (int j = Ao[i]; j < Ao[i+1]; j++) {
   			if( FindIsovalue(cell_centroids[i], facet_centroids[j], texObj) || FindIsovalue(facet_centroids[j], cell_centroids[Ac[j]], texObj) )
				similarities[j] = 1.0f;
			else
				similarities[j] = similarity(colorpatterns[i],colorpatterns[Ac[j]]);

			//printf("Similarity[%i,%i] = %.2f\n", i, Ac[j], similarities[j]);
		}
	}
}

__global__ void compute_similarity_kernel_3D(const int *Ao, const int *Ac, cudaTextureObject_t texObj, float3 *cell_centroids, float3 *facet_centroids, const float *colorpatterns, float *similarities)
{
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < V; i += blockDim.x*gridDim.x) {
		// look at neighbors
		for (int j = Ao[i]; j < Ao[i + 1]; j++) {
			if (FindIsovalue_3D(cell_centroids[i], facet_centroids[j], texObj) || FindIsovalue_3D(facet_centroids[j], cell_centroids[Ac[j]], texObj))
				similarities[j] = 1.0f;
			else
				similarities[j] = similarity(colorpatterns[i], colorpatterns[Ac[j]]);

			//printf("Similarity[%i,%i] = %.2f\n", i, Ac[j], similarities[j]);
		}
	}
}
//
/////////////////////////////////////////////////////////