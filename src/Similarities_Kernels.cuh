#include "kernel_constants.h"
#include "SimilarityFunction_Kernels.cuh"
#include "Filter_Kernels.cuh"
#include "Texture_Kernels.cuh"

__constant__ int Neighbor_steps[26];
__constant__ int3 Neighbor_stepcoords[26];

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
///* SIMILARITIES COMPUTATION KERNELS */
//

/* FOR GRAPHS */

/*------------------------------------------*/
/* COMPUTE CELL CENTROIDS & FACET CENTROIDS */

template <class XSpace>
__global__ void reset_similarity_Graph_kernel(const XSpace::NeighborsType *neighbors, XSpace::SimilarityType *similarities)
{
	int simil = 0;

	if (threadIdx.x < NCellNeighbors){
		int j = *((int*)&neighbors[blockIdx.x] + threadIdx.x);
		if (j == -1)
			simil |= (1 << threadIdx.x);
	}

	// Warp Or-Reduce
	__syncthreads();
	for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
		simil |= __shfl_down(simil, offset);

	// Write result
	if (threadIdx.x == 0) {
		// if (blockIdx.x == 280280) printf("cell = %i, simil = %i\n", blockIdx.x, simil);
		similarities[blockIdx.x] = simil;
	}
}

template <class XSpace>
__global__ void filter_edges_kernel(int2 *dst, const XSpace::NeighborsType *src, int *ctr) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int i = tid / NCellNeighbors;
	if (i >= V) return;

	int k = tid % NCellNeighbors;
	int j = *( (int*)&src[i] + k );
	//printf("i = %i, j = %i\n", i, j);

	if (i < j) {
		dst[atomicAggInc(ctr)] = make_int2(i,j);
		/*int j = atomicAggInc(ctr);
		dst[j] = src[i];
		printf("Head %i: %i\n", j,i);*/
	}
}

template<class XSpace>
__global__ void compute_cellcentroids_kernel(XSpace::CellVerticesType *cell_vertices, XSpace::PointCoordsType *delaunay_vertices, XSpace::PointCoordsType *cell_centroids)
{
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < V; i += blockDim.x*gridDim.x) {
		// Cell Centroid
		typename XSpace::PointCoordsType c;
		float *q, *p = (float*)&c;

		for (int j = 0; j < Dim; ++j)	*(p + j) = 0;

		for (int k = 0; k < NCellVertices; ++k){
			q = (float*)&delaunay_vertices[*(&cell_vertices[i].x + k)];
			for (int j = 0; j < Dim; ++j)	*(p + j) += *(q + j);
		}

		for (int j = 0; j < Dim; ++j)	*(p + j) /= NCellVertices;

		cell_centroids[i] = c;

		//float *cc1 = (float*)&cell_centroids[i];
		//if (i < 10) printf("# %i, c = (%.2f, %.2f, %.2f)\n", i, *(cc1), *(cc1 + 1), *(cc1 + 2));
	}
}

template<class XSpace>
__global__ void compute_facets_kernel(int2 *edgenodes, XSpace::CellVerticesType *cell_vertices, XSpace::PointCoordsType *delaunay_vertices, XSpace::FacetsType *facet_vertices)
{
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < E; i += blockDim.x*gridDim.x) {

		typename XSpace::FacetsType fc;
		float *q, *p = (float*)&fc;
		int *cv1 = &cell_vertices[edgenodes[i].x].x;
		int *cv2 = &cell_vertices[edgenodes[i].y].x;

		for (int j = 0; j < Dim; ++j)	*(p + j) = 0;

		int c = 0;
		for (int j = 0; j < NCellVertices; ++j){
			for (int k = 0; k < NCellVertices; ++k) {
				if (*(cv1 + j) == *(cv2 + k)){
					((int*)&facet_vertices[i])[c++] = *(cv1 + j);
				}
			}
		}
	}
}

template<class XSpace>
__global__ void compute_facetcentroids_kernel(int2 *edgenodes, XSpace::CellVerticesType *cell_vertices, XSpace::PointCoordsType *delaunay_vertices, XSpace::PointCoordsType *facet_centroids)
{
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < E; i += blockDim.x*gridDim.x) {
		// Facet Centroid

		//typename XSpace::PointCoordsType fc;
		//float *q, *p = (float*)&fc;
		//for (int j = 0; j < Dim; ++j)	*(p + j) = 0;
		//for (int k = 0; k < NFacetVertices; ++k){
		//	q = (float*)&delaunay_vertices[*(&facet_vertices[i].x + k)];
		//	for (int j = 0; j < Dim; ++j)	*(p + j) += *(q + j);
		//}
		//for (int j = 0; j < Dim; ++j)	*(p + j) /= NFacetVertices;
		//facet_centroids[i] = fc;

		typename XSpace::PointCoordsType fc;
		float *q, *p = (float*)&fc;
		int *cv1 = &cell_vertices[edgenodes[i].x].x;
		int *cv2 = &cell_vertices[edgenodes[i].y].x;

		for (int j = 0; j < Dim; ++j)	*(p + j) = 0;

		for (int j = 0; j < NCellVertices; ++j){
			for (int k = 0; k < NCellVertices; ++k) {
				if (*(cv1 + j) == *(cv2 + k)){
					q = (float*)&delaunay_vertices[*(cv1 + j)];
					for (int d = 0; d < Dim; ++d)	*(p + d) += *(q + d);
				}
			}
		}

		for (int j = 0; j < Dim; ++j)	*(p + j) /= NFacetVertices;
		facet_centroids[i] = fc;

		//float *cc1 = &fc.x;
		//printf("Edge %i (%i, %i) : fc = (%.2f, %.2f)\n", i, edgenodes[i].x, edgenodes[i].y, *(cc1), *(cc1 + 1));
		//if (i == 0)
		//	printf("Edge %i (%i, %i) : fc = (%.2f, %.2f)\n", i, edgenodes[i].x, edgenodes[i].y, fc.x, fc.y);
	}
}

template<class XSpace>
__global__ void compute_facetcentroids_kernel2(int2 *edgenodes, XSpace::FacetsType *facet_vertices, XSpace::PointCoordsType *delaunay_vertices, XSpace::PointCoordsType *facet_centroids)
{
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < E; i += blockDim.x*gridDim.x) {
		// Facet Centroid

		typename XSpace::PointCoordsType fc;
		float *q, *p = (float*)&fc;
		int *fv = &facet_vertices[i].x;

		for (int j = 0; j < Dim; ++j) *(p + j) = 0;

		for (int j = 0; j < Dim; ++j) {
			q = (float*)&delaunay_vertices[*(fv + j)];
			for (int d = 0; d < Dim; ++d)	*(p + d) += *(q + d);
		}
		
		for (int j = 0; j < Dim; ++j)	*(p + j) /= NFacetVertices;
		facet_centroids[i] = fc;

		//float *cc1 = &fc.x;
		//printf("Edge %i (%i, %i) : fc = (%.2f, %.2f)\n", i, edgenodes[i].x, edgenodes[i].y, *(cc1), *(cc1 + 1));
		//if (i == 0)
		//	printf("Edge %i (%i, %i) : fc = (%.2f, %.2f)\n", i, edgenodes[i].x, edgenodes[i].y, fc.x, fc.y);
	}
}

 template<class XSpace>
 __global__ void compute_maxfacets_kernel(XSpace::CellVerticesType *cell_vertices, XSpace::PointCoordsType *delaunay_vertices, float *cell_maxfacets);

template<>
 __global__ void compute_maxfacets_kernel <CDelaunay_2D_Cuda_XGraph_Adaptor>(int3 *cell_vertices, float2 *delaunay_vertices, float *cell_maxfacets)
 {
 	// Cuda-style for
 	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < V; i += blockDim.x*gridDim.x) {

 		float2 *v[3];
 		v[0] = &delaunay_vertices[cell_vertices[i].x];
 		v[1] = &delaunay_vertices[cell_vertices[i].y];
 		v[2] = &delaunay_vertices[cell_vertices[i].z];

 		float facetsizes[3];
 		float maxfacetsize = 0;

 		facetsizes[0] = (v[0]->x - v[1]->x)*(v[0]->x - v[1]->x) + (v[0]->y - v[1]->y)*(v[0]->y - v[1]->y);
 		facetsizes[1] = (v[0]->x - v[2]->x)*(v[0]->x - v[2]->x) + (v[0]->y - v[2]->y)*(v[0]->y - v[2]->y);
 		facetsizes[2] = (v[1]->x - v[2]->x)*(v[1]->x - v[2]->x) + (v[1]->y - v[2]->y)*(v[1]->y - v[2]->y);

 		for(int j=0; j < 3; ++j)
 			maxfacetsize = fmaxf(maxfacetsize, facetsizes[j]);

 		cell_maxfacets[i] = sqrtf(maxfacetsize);

 	}
 }

 template<>
 __global__ void compute_maxfacets_kernel <CDelaunay_3D_Cuda_XGraph_Adaptor>(int4 *cell_vertices, float3 *delaunay_vertices, float *cell_maxfacets)
 {
 	// Cuda-style for
 	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < V; i += blockDim.x*gridDim.x) {

 		float3 *v[4];
 		v[0] = &delaunay_vertices[cell_vertices[i].x];
 		v[1] = &delaunay_vertices[cell_vertices[i].y];
 		v[2] = &delaunay_vertices[cell_vertices[i].z];
 		v[3] = &delaunay_vertices[cell_vertices[i].w];

 		float facetsizes[4];
 		float maxfacetsize = 0;

 		facetsizes[0] = fabsf((v[0]->x)*(v[1]->y - v[2]->y) + (v[1]->x)*(v[2]->y - v[0]->y) + (v[2]->x)*(v[0]->y - v[1]->y));
 		facetsizes[1] = fabsf((v[0]->x)*(v[1]->y - v[3]->y) + (v[1]->x)*(v[3]->y - v[0]->y) + (v[3]->x)*(v[0]->y - v[1]->y));
 		facetsizes[2] = fabsf((v[0]->x)*(v[2]->y - v[3]->y) + (v[2]->x)*(v[3]->y - v[0]->y) + (v[3]->x)*(v[0]->y - v[2]->y));
 		facetsizes[3] = fabsf((v[1]->x)*(v[2]->y - v[3]->y) + (v[2]->x)*(v[3]->y - v[1]->y) + (v[3]->x)*(v[1]->y - v[2]->y));

 		for(int j=0; j < 4; ++j)
 			maxfacetsize = fmaxf(maxfacetsize, facetsizes[j]);

 		cell_maxfacets[i] = 0.5*(maxfacetsize);

 	}
 }

 //template<class XSpace>
 //__global__ void compute_aspectratio_kernel(int2 *edgenodes, XSpace::CellVerticesType *cell_vertices, XSpace::PointCoordsType *delaunay_vertices, XSpace::PointCoordsType *facet_centroids)
 //{
 //	// Cuda-style for
 //	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < E; i += blockDim.x*gridDim.x) {
 //		// Facet Size

 //		typename XSpace::PointCoordsType fc;
 //		float *q, *p = (float*)&fc;
 //		int *cv1 = &cell_vertices[edgenodes[i].x].x;
 //		int *cv2 = &cell_vertices[edgenodes[i].y].x;

 //		for (int j = 0; j < Dim; ++j)	*(p + j) = 0;

 //		for (int j = 0; j < NCellVertices; ++j){
 //			for (int k = 0; k < NCellVertices; ++k) {
 //				if (*(cv1 + j) == *(cv2 + k)){
 //					q = (float*)&delaunay_vertices[*(cv1 + j)];
 //					for (int d = 0; d < Dim; ++d)	*(p + d) += *(q + d);
 //				}
 //			}
 //		}

 //		for (int j = 0; j < Dim; ++j)	*(p + j) /= NFacetVertices;
 //		facet_centroids[i] = fc;

 //		//float *cc1 = &fc.x;
 //		//printf("Edge %i (%i, %i) : fc = (%.2f, %.2f)\n", i, edgenodes[i].x, edgenodes[i].y, *(cc1), *(cc1 + 1));
 //		//if (i == 0)
 //		//	printf("Edge %i (%i, %i) : fc = (%.2f, %.2f)\n", i, edgenodes[i].x, edgenodes[i].y, fc.x, fc.y);
 //	}
 //}

__device__ inline int find_pos(int x, int* a, int n)
{
	int res = -1;
	for (int i = 0; i < n; ++i){
		if (a[i] == x) {
			res = i;
			break;
		}
	}
	return res;
}

template<class XSpace>
__global__ void compute_neighborpositions_kernel(int2 *edgenodes, XSpace::NeighborsType *neighbors, int2 *neighbor_positions)
{
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < E; i += blockDim.x*gridDim.x) {
		neighbor_positions[i].x = find_pos(edgenodes[i].y, (int*)&neighbors[edgenodes[i].x], NCellNeighbors);
		neighbor_positions[i].y = find_pos(edgenodes[i].x, (int*)&neighbors[edgenodes[i].y], NCellNeighbors);

		//printf("Edge %i (%i, %i) : pos1 = %i, pos2 = %i\n", i, edgenodes[i].x, edgenodes[i].y, neighbor_positions[i].x, neighbor_positions[i].y);
	}
}

template<class XSpace>
__global__ void compute_similarity_Graph_kernel(cudaTextureObject_t texObj, int2 *edgenodes, int2 *neighbor_positions, XSpace::PointCoordsType *cell_centroids, XSpace::PointCoordsType *facet_centroids, float *colorpatterns, XSpace::SimilarityType *similarities)
{
	__shared__ int flag;
	flag = 0;

	typename XSpace::PointCoordsType *c1, *c2, *fc;
	c1 = &cell_centroids[edgenodes[blockIdx.x].x];
	c2 = &cell_centroids[edgenodes[blockIdx.x].y];
	fc = &facet_centroids[blockIdx.x];

	typename XSpace::PointCoordsType v1, v2, k1, k2;
	v1 = *fc - *c1;
	v2 = *c2 - *fc;

	k1 = *fc - ALPHA*(Dim)*(v1);
	k2 = *fc + ALPHA*(Dim)*(v2);
	v1 = *fc - k1;
	v2 = k2 - *fc;

	//if (edgenodes[blockIdx.x].x == 3){
	//	printf("Edge %i (%i, %i) : fc = (%.2f, %.2f)\n", blockIdx.x, edgenodes[blockIdx.x].x, edgenodes[blockIdx.x].y, fc->x, fc->y);
	//	printf("Edge %i (%i, %i) : c1 = (%.2f, %.2f), c2 = (%.2f, %.2f)\n", blockIdx.x, edgenodes[blockIdx.x].x, edgenodes[blockIdx.x].y, c1->x, c1->y, c2->x, c2->y);
	//	printf("Edge %i (%i, %i) : v1 = (%.2f, %.2f), v2 = (%.2f, %.2f)\n", blockIdx.x, edgenodes[blockIdx.x].x, edgenodes[blockIdx.x].y, v1.x, v1.y, v2.x, v2.y);
	//}

	int s1, s2;
	s1 = fmaxf(1, rintf(length(v1)));
	s2 = fmaxf(1, rintf(length(v2)));

	v1 /= s1;
	v2 /= s2;

	////if (blockIdx.x == 0){
	//if ((edgenodes[blockIdx.x].x == 30108/* && edgenodes[blockIdx.x].y == 155949*/) && threadIdx.x == 0)
	//	printf("Edge %i (%i, %i) : s1 = %i, s2 = %i\n", blockIdx.x, edgenodes[blockIdx.x].x, edgenodes[blockIdx.x].y, s1, s2);
	////	printf("Edge %i (%i, %i) : v1 = (%.2f, %.2f), v2 = (%.2f, %.2f)\n", blockIdx.x, edgenodes[blockIdx.x].x, edgenodes[blockIdx.x].y, v1.x, v1.y, v2.x, v2.y);
	////}


	typename XSpace::PointCoordsType p1, p2;
	float p1_cp, p2_cp;
	for (int i = threadIdx.x; i < s1 + s2; i += blockDim.x) {
		p1 = (k1 + v1*(threadIdx.x))*(threadIdx.x < s1) + (*fc + v2*(threadIdx.x - s1))*(threadIdx.x >= s1);
		p2 = (k1 + v1*(threadIdx.x + 1))*(threadIdx.x < s1) + (*fc + v2*(threadIdx.x - s1 + 1))*(threadIdx.x >= s1);

		p1_cp = tex<XSpace>(texObj, p1);
		p2_cp = tex<XSpace>(texObj, p2);

		//if (blockIdx.x == 20656)
		//if (edgenodes[blockIdx.x].x == 30108/* && edgenodes[blockIdx.x].y == 155949*/){
		//	//printf("Edge %i (%i, %i) : p1 = (%.2f, %.2f), p2 = (%.2f, %.2f)\n", blockIdx.x, edgenodes[blockIdx.x].x, edgenodes[blockIdx.x].y, p1.x, p1.y, p2.x, p2.y);
		//	float *q1 = (float*)&p1;
		//	float *q2 = (float*)&p2;
		//	printf("Edge %i (%i, %i) : p1 = (%.2f, %.2f, %.2f) [%.2f], p2 = (%.2f, %.2f, %.2f) [%.2f]\n", blockIdx.x, edgenodes[blockIdx.x].x, edgenodes[blockIdx.x].y, q1[0], Img_height - q1[1], q1[2], p1_cp, q2[0], Img_height - q2[1], q2[2], p2_cp);
		//}

		if (similarity<true>(p1_cp, p2_cp) == 1.0) {
			//if (edgenodes[blockIdx.x].x == 19 && edgenodes[blockIdx.x].y == 155949) printf("Edge %i (%i, %i) : INTERCEPTION p1 = (%.2f, %.2f)[%.2f], p2 = (%.2f, %.2f)[%.2f]\n", blockIdx.x, edgenodes[blockIdx.x].x, edgenodes[blockIdx.x].y, p1.x, p1.y, p1_cp, p2.x, p2.y, p2_cp);
			flag = 1;
			break;
		}

		/*if (similarity<false>(p1_cp, p2_cp) >= Threshold/2.5) {
			//if (edgenodes[blockIdx.x].x == 19 && edgenodes[blockIdx.x].y == 155949) printf("Edge %i (%i, %i) : INTERCEPTION p1 = (%.2f, %.2f)[%.2f], p2 = (%.2f, %.2f)[%.2f]\n", blockIdx.x, edgenodes[blockIdx.x].x, edgenodes[blockIdx.x].y, p1.x, p1.y, p1_cp, p2.x, p2.y, p2_cp);
			if (edgenodes[blockIdx.x].x == 30108) printf("Edge %i (%i, %i) : p1 = (%.2f, %.2f)[%.2f], p2 = (%.2f, %.2f)[%.2f], SIMILARITY = %.2f\n", blockIdx.x, edgenodes[blockIdx.x].x, edgenodes[blockIdx.x].y, p1.x, p1.y, p1_cp, p2.x, p2.y, p2_cp, similarity<false>(p1_cp, p2_cp));
			flag = 1;
			break;
		}*/
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		//if (flag) similarities[blockIdx.x] = 1.0;
		//else similarities[blockIdx.x] = similarity<true>(colorpatterns[edgenodes[blockIdx.x].x], colorpatterns[edgenodes[blockIdx.x].y]);

		float simil;
		if (flag) simil = 1.0;
		else simil = similarity<true>(colorpatterns[edgenodes[blockIdx.x].x], colorpatterns[edgenodes[blockIdx.x].y]);
		if (simil >= Threshold){
			atomicOr(&similarities[edgenodes[blockIdx.x].x], (1 << neighbor_positions[blockIdx.x].x));
			atomicOr(&similarities[edgenodes[blockIdx.x].y], (1 << neighbor_positions[blockIdx.x].y));
		}

		/*simil = similarity<true>(colorpatterns[edgenodes[blockIdx.x].x], colorpatterns[edgenodes[blockIdx.x].y]);
		if (simil >= Threshold || flag == 1) 
		{
			atomicOr(&similarities[edgenodes[blockIdx.x].x], (1 << neighbor_positions[blockIdx.x].x));
			atomicOr(&similarities[edgenodes[blockIdx.x].y], (1 << neighbor_positions[blockIdx.x].y));
		}*/

		//if (blockIdx.x == 0) {
		//	printf("Edge %i (%i, %i) : simil = %.2f\n", blockIdx.x, edgenodes[blockIdx.x].x, edgenodes[blockIdx.x].y, simil);
		//	printf("Edge %i (%i, %i) : pos1 = %i, pos2 = %i\n", blockIdx.x, edgenodes[blockIdx.x].x, edgenodes[blockIdx.x].y, neighbor_positions[blockIdx.x].x, neighbor_positions[blockIdx.x].y);
		//}

		//if (edgenodes[blockIdx.x].x == 3) {
		//	printf("Edge %i (%i, %i) : pos1 = %i, pos2 = %i, simil = %.2f\n", blockIdx.x, edgenodes[blockIdx.x].x, edgenodes[blockIdx.x].y, neighbor_positions[blockIdx.x].x, neighbor_positions[blockIdx.x].y, simil);
		//}
	}

}

template<class XSpace>
__device__ float facet_size(XSpace::FacetsType *facet_vertices, XSpace::PointCoordsType *delaunay_vertices);

template<>
__device__ float facet_size <CDelaunay_2D_Cuda_XGraph_Adaptor>(int2 *facet_vertices, float2 *delaunay_vertices)
{
	float2 *v[2];
	v[0] = &delaunay_vertices[facet_vertices->x];
	v[1] = &delaunay_vertices[facet_vertices->y];

	return sqrtf( (v[0]->x - v[1]->x)*(v[0]->x - v[1]->x) + (v[0]->y - v[1]->y)*(v[0]->y - v[1]->y) );
}

template<>
__device__ float facet_size <CDelaunay_3D_Cuda_XGraph_Adaptor>(int3 *facet_vertices, float3 *delaunay_vertices)
{
	float3 *v[3];
	v[0] = &delaunay_vertices[facet_vertices->x];
	v[1] = &delaunay_vertices[facet_vertices->y];
	v[2] = &delaunay_vertices[facet_vertices->z];

	return 0.5*fabsf((v[0]->x)*(v[1]->y - v[2]->y) + (v[1]->x)*(v[2]->y - v[0]->y) + (v[2]->x)*(v[0]->y - v[1]->y));
}

template<class XSpace>
__global__ void compute_geometric_similarity_kernel(int2 *edgenodes, int2 *neighbor_positions, XSpace::PointCoordsType *delaunay_vertices, XSpace::FacetsType *facet_vertices, float *cell_maxfacets, XSpace::SimilarityType *similarities)
{
	float geothreshold = 0.35;
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < E; i += blockDim.x*gridDim.x) {
		
		float facetsize = facet_size<XSpace>(&facet_vertices[i], delaunay_vertices);
		float phi = 0.5*((facetsize / cell_maxfacets[edgenodes[i].x]) + (facetsize / cell_maxfacets[edgenodes[i].y]));

		//printf("edge %i, phi = %.2f\n", i, phi);

		if (phi < geothreshold){
			atomicOr(&similarities[edgenodes[i].x], (1 << neighbor_positions[i].x));
			atomicOr(&similarities[edgenodes[i].y], (1 << neighbor_positions[i].y));
		}

	}
}

///////////////////////////////////////////////////////

/* FOR IMAGES */

template<bool Flag>
__global__ void filter_Image_borders(int *dst, int *ctr) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= V) return;

	int3 p;
	p.x = i % Img_width;
	p.y = ((int)(i * Inv_width)) % Img_height;

	bool test;
	test = ((p.x == 0) || (p.x == Img_width - 1) || (p.y == 0) || (p.y == Img_height - 1));

	if (Img_depth > 0) {
		p.z = (int)(i * Inv_heightxwidth);
		test |= ((p.z == 0) || (p.z == Img_depth - 1));
	}

	if (test == Flag) {
		dst[atomicAggInc(ctr)] = i;
	}
}

template<class XSpace>
__global__ void reset_similarity_Image_kernel(int *src, int n, XSpace::SimilarityType *similarities, int3 *neighbor_stepcoords)
{
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < n; i += blockDim.x*gridDim.x)	{
		int id = src[i];

		int simil = 0;

		int3 p;
		p.x = id % Img_width;
		p.y = ((int)(id * Inv_width)) % Img_height;
		p.z = (int)(id * Inv_heightxwidth);

		int3 n;
		for (int k = 0; k < NCellNeighbors; ++k){
			n.x = p.x + neighbor_stepcoords[k].x;
			n.y = p.y + neighbor_stepcoords[k].y;
			n.z = p.z + neighbor_stepcoords[k].z;

			if ((n.x < 0) || (n.x >= Img_width) || (n.y < 0) || (n.y >= Img_height) || (n.z < 0) || (n.z >= Img_depth))
				simil |= (1 << k);
		}

		// Write result
		similarities[id] = simil;
	}
}

template<class XSpace>
__global__ void compute_similarity_Image_kernel(int i, const float *colorpatterns, XSpace::SimilarityType *similarities, int *neighbor_steps, int3 *neighbor_stepcoords)
{
	int simil = 0;

	int id = blockIdx.x + i * gridDim.x;
	if (id >= V) return;

	if (threadIdx.x < NCellNeighbors){
		int3 n;
		n = neighbor_stepcoords[threadIdx.x];
		n.x += id % Img_width;
		n.y += ((int)(id * Inv_width)) % Img_height;
		n.z += (int)(id * Inv_heightxwidth);
		int neighbor = id + neighbor_steps[threadIdx.x];

		if ((n.x < 0) || (n.x >= Img_width) || (n.y < 0) || (n.y >= Img_height) || (n.z < 0) || (n.z >= Img_depth))
			simil |= (1 << threadIdx.x);
		else if (similarity<true>(colorpatterns[id], colorpatterns[neighbor]) >= Threshold)
			simil |= (1 << threadIdx.x);
	}

	// Warp Or-Reduce
	__syncthreads();
	//#pragma unroll 5
	for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
		simil |= __shfl_down(simil, offset);

	// Write result
	if (threadIdx.x == 0) {
		// if (id == 280280) printf("cell = %i, simil = %i\n", blockIdx.x, simil);
		similarities[id] = simil;
	}
}

template<class XSpace>
__global__ void compute_similarity_Image_kernel_v3(const float *colorpatterns, XSpace::SimilarityType *similarities, int *neighbor_steps, int3 *neighbor_stepcoords)
{
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < V; i += blockDim.x*gridDim.x)	{
		int simil = 0;
		int3 p;
		p.x = i % Img_width;
		p.y = ((int)(i * Inv_width)) % Img_height;
		p.z = (int)(i * Inv_heightxwidth);

		int3 n;
		int neighbor;
		for (int k = 0; k < NCellNeighbors; ++k){
			n.x = p.x + neighbor_stepcoords[k].x;
			n.y = p.y + neighbor_stepcoords[k].y;
			n.z = p.z + neighbor_stepcoords[k].z;
			neighbor = i + neighbor_steps[k];

			if ((n.x < 0) || (n.x >= Img_width) || (n.y < 0) || (n.y >= Img_height) || (n.z < 0) || (n.z >= Img_depth))
				simil |= (1 << k);
			else if (similarity<true>(colorpatterns[i], colorpatterns[neighbor]) >= Threshold)
				simil |= (1 << k);
		}

		// Write result
		similarities[i] = simil;
	}
}

template<class XSpace, bool Border>
__global__ void compute_similarity_Image_kernel_v4(int *src, int n, const float *colorpatterns, XSpace::SimilarityType *similarities, int *neighbor_steps, int3 *neighbor_stepcoords)
{
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < n; i += blockDim.x*gridDim.x)	{
		int id = src[i];

		int simil = 0;
		int neighbor;

		if (Border) {
			int3 p;
			p.x = id % Img_width;
			p.y = ((int)(id * Inv_width)) % Img_height;
			p.z = (int)(id * Inv_heightxwidth);

			int3 n;
			for (int k = 0; k < NCellNeighbors; ++k){
				n.x = p.x + neighbor_stepcoords[k].x;
				n.y = p.y + neighbor_stepcoords[k].y;
				n.z = p.z + neighbor_stepcoords[k].z;
				neighbor = id + neighbor_steps[k];

				if ((n.x < 0) || (n.x >= Img_width) || (n.y < 0) || (n.y >= Img_height) || (n.z < 0) || (n.z >= Img_depth))
					simil |= (1 << k);
				else if (similarity<true>(colorpatterns[id], colorpatterns[neighbor]) >= Threshold)
					simil |= (1 << k);
			}
		}
		else {
			for (int k = 0; k < NCellNeighbors; ++k){
				neighbor = id + neighbor_steps[k];
				if (similarity<true>(colorpatterns[id], colorpatterns[neighbor]) >= Threshold)
					simil |= (1 << k);
			}
		}

		// Write result
		similarities[id] = simil;
	}
}
