#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <helper_math.h>

#include "class.h"
#include "cudagraph.cuh"

__constant__ int V;
__constant__ int E;
__constant__ int DV;
__constant__ int Img_width;
__constant__ int Img_height;
__constant__ float Isovalues[64];
__constant__ int Num_isovalues;

///////////////////////////////////////////////////////
//
//__device__ float Ker1(float i, float j) { return 1-expf(-1*abs(i-j)/50); } 
//
//typedef float (*pSimilarityDevKernel_t)(float,float);
//__device__ pSimilarityDevKernel_t Similarity_Kernels[1] = { Ker1 };
//
//template<kernel K>
//__device__ float similarity(float i, float j){
//	return Similarity_Kernels[K](i,j);
//}

__device__ float similarity(float i, float j) { 
	return 1-expf(-1*abs(i-j)/50); 
} 

///////////////////////////////////////////////////////

/* TEST KERNELS */

__global__ void test_texture_kernel(float* image, cudaTextureObject_t texObj)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
  	int y = threadIdx.y + blockIdx.y * blockDim.y;
  	int id = x + y * blockDim.x * gridDim.x;

	image[id] = 255 - tex2D<float>(texObj,x,y);
}

__global__ void test_constants_kernel()
{
	printf("Isovalues: ");
	for(int i=0; i<Num_isovalues; i++)
		printf("%.2f ", Isovalues[i]);
	printf("\n");
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
//
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
//
/////////////////////////////////////////////////////////
//
///* GEOMETRIC ADJUSTMENTS KERNELS */
//
//template<class G, kernel K>
//__global__ void flip_verticality_kernel(float2 *delaunay_vertices)
//{
//	// Cuda-style for
//	for (int i = threadIdx.x+blockIdx.x*blockDim.x ; i < DV ; i += blockDim.x*gridDim.x) {
//		delaunay_vertices[i].y = Img_height - delaunay_vertices[i].y;
//	}
//}
//
//template<class G, kernel K>
__global__ void compute_centroids_kernel(const int *Ao, const int *Ac, int3 *cell_vertices, float2 *delaunay_vertices, float2 *cell_centroids)
{
	// Cuda-style for
	for (int i = threadIdx.x+blockIdx.x*blockDim.x ; i < V ; i += blockDim.x*gridDim.x) {

		// Cell Centroid
		cell_centroids[i] = (delaunay_vertices[cell_vertices[i].x] + delaunay_vertices[cell_vertices[i].y] + delaunay_vertices[cell_vertices[i].z]) / 3.0f;
		
//
//		// Facet Centroids, for each neighbor
//		for (int k = Ao[i]; k < Ao[i+1]; k++) {
//
//			float2 fcentroid;
//			int* ptr1 = &cell_vertices[i].x;
//			int* ptr2 = &cell_vertices[Ac[k]].x;
//
//			for(int m=0; m<3; m++) {
//				for( int n=0; n<3; n++) {
//					if( *(ptr1+m) == *(ptr2+n) ) fcentroid += make_float2(delaunay_vertices[*(ptr1+m)]);
//				}
//			}
//
//			facet_centroids[k] = make_int2(fcentroid/2);
//		}
	}
}
//
//template<class G, kernel K>
//__global__ void compute_sizes_kernel(const int *Ao, const int *Ac, const int3 *cell_vertices, const int2 *delaunay_vertices, float *cell_sizes)
//{
//	// Cuda-style for
//	for (int i = threadIdx.x+blockIdx.x*blockDim.x ; i < V ; i += blockDim.x*gridDim.x) {
//		
//
//		int2 a_i2 = delaunay_vertices[cell_vertices[i].x] - delaunay_vertices[cell_vertices[i].y];
//		int2 b_i2 = delaunay_vertices[cell_vertices[i].x] - delaunay_vertices[cell_vertices[i].z];
//
//		float3 a = make_float3(a_i2.x,a_i2.y,0.0f);
//		float3 b = make_float3(b_i2.x,b_i2.y,0.0f);
//
//		cell_sizes[i] = 0.5*length(cross(a,b));
//	}
//}
//
/////////////////////////////////////////////////////////
//
///* SEGMENTATION KERNELS */
//
//template<class G, kernel K>
//__global__ void segmentation_NCG_kernel(int c, const int *Ao, const int *Ac, const int *colors,
//	const float *similarities, int *labels, float threshold)
//{
//	// Cuda-style for
//	for (int i = threadIdx.x+blockIdx.x*blockDim.x;i < V;i += blockDim.x*gridDim.x)	{
//		if (colors[i] != c) continue;
//		
//		// Here starts the segmentation 
//		for (int s_i = Ao[i]; s_i < Ao[i+1]; s_i++) {
//			int test = (similarities[s_i] <= threshold);
//			labels[i] = ( ((test<<31)>>31) & rmin(labels[i],labels[Ac[s_i]]) ) | ( ((!test<<31)>>31) & labels[i] );
//		}
//
//	}
//}
//
//template<class G, kernel K>
//__global__ void path_compression_kernel(int *labels)
//{
//	// Cuda-style for
//	for (int i = threadIdx.x+blockIdx.x*blockDim.x;i < V;i += blockDim.x*gridDim.x)	{
//		do { 
//			labels[i] = labels[labels[i]];
//		} while(labels[i] != labels[labels[i]]);
//	}
//}
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
		colorpatterns[i] = tex2D<float>(texObj,cell_centroids[i].x, Img_height - cell_centroids[i].y);
		//printf("colorpattern[%i] = %.2f\n",i,colorpatterns[i]);
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
//
//__device__ bool FindIsovalue(float2 p1, float2 p2, cudaTextureObject_t texObj)
//{
//    int x = (int)p1.x, y = Img_height - (int)p1.y;
//    int x2 = (int)p2.x, y2 = Img_height - (int)p2.y;
//
//    bool yLonger=false;
//    int shortLen=y2-y;
//    int longLen=x2-x;
//    if (abs(shortLen)>abs(longLen))
//    {
//        int swap=shortLen;
//        shortLen=longLen;
//        longLen=swap;
//        yLonger=true;
//    }
//    int decInc;
//    if (longLen==0) decInc=0;
//    else decInc = (shortLen << 16) / longLen;
//
//    if (yLonger)
//    {
//        if (longLen>0)
//        {
//            longLen+=y;
//            y2=y;
//            for (int j=0x8000+(x<<16);y<=longLen;++y)
//            {
//				float imgdata1 = tex2D<float>(texObj,x,y2);
//				float imgdata2 = tex2D<float>(texObj,j>>16,y);
//				for(int k=0; k<Num_isovalues; k++)
//				{
//					if( (imgdata1 - Isovalues[k])*(imgdata2 - Isovalues[k]) <= 0 )
//						return true;
//				}
//
//                x=j>>16;
//                y2=y;
//                j+=decInc;
//            }
//            return false;
//        }
//        longLen+=y;
//        y2=y;
//        for (int j=0x8000+(x<<16);y>=longLen;--y)
//        {
//			float imgdata1 = tex2D<float>(texObj,x,y2);
//			float imgdata2 = tex2D<float>(texObj,j>>16,y);
//			for(int k=0; k<Num_isovalues; k++)
//			{
//				if( (imgdata1 - Isovalues[k])*(imgdata2 - Isovalues[k]) <= 0 )
//					return true;
//			}
//
//            x=j>>16;
//            y2=y;
//            j-=decInc;
//        }
//        return false;
//    }
//
//    if (longLen>0)
//    {
//        longLen+=x;
//        x2=x;
//        for (int j=0x8000+(y<<16);x<=longLen;++x)
//        {
//			float imgdata1 = tex2D<float>(texObj,x2,y);
//			float imgdata2 = tex2D<float>(texObj,x,j>>16);
//			for(int k=0; k<Num_isovalues; k++)
//			{
//				if( (imgdata1 - Isovalues[k])*(imgdata2 - Isovalues[k]) <= 0 )
//					return true;
//			}
//
//
//            y=j>>16;
//            x2=x;
//            j+=decInc;
//        }
//        return false;
//    }
//    longLen+=x;
//    x2=x;
//    for (int j=0x8000+(y<<16);x>=longLen;--x)
//    {
//		float imgdata1 = tex2D<float>(texObj,x2,y);
//		float imgdata2 = tex2D<float>(texObj,x,j>>16);
//		for(int k=0; k<Num_isovalues; k++)
//		{
//			if( (imgdata1 - Isovalues[k])*(imgdata2 - Isovalues[k]) <= 0 )
//				return true;
//		}
//
//
//        y=j>>16;
//        x2=x;
//        j-=decInc;
//    }
//    return false;
//}
//
//template<class G, kernel K>
//__global__ void compute_similarity_kernel(const int *Ao, const int *Ac, cudaTextureObject_t texObj, float2 *cell_centroids, float2 *facet_centroids, const float *colorpatterns, float *similarities)
//{
//	// Cuda-style for
//	for (int i = threadIdx.x+blockIdx.x*blockDim.x ; i < V ; i += blockDim.x*gridDim.x) {
//		// look at neighbors
//		for (int j = Ao[i]; j < Ao[i+1]; j++) {
//   			if( FindIsovalue(cell_centroids[i], facet_centroids[j], texObj) || FindIsovalue(facet_centroids[j], cell_centroids[Ac[j]], texObj) )
//				similarities[j] = 1.0f;
//			else
//				similarities[j] = similarity(colorpatterns[i],colorpatterns[Ac[j]]);
//		}
//	}
//}
//
/////////////////////////////////////////////////////////