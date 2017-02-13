#include "kernel_constants.h"
#include "Filter_Kernels.cuh"
#include "Neighbor_kernels.cuh"
#include "SimilarityFunction_kernels.cuh"
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////
/////* SEGMENTATION KERNELS */
////
//
/////////////////////////////////////////////////////////

/*------------------------------------------*/
/* OPERATOR # 1 : NEIGHBOR CELL GROUPING KERNELS */

//template<class XSpace>
//__global__ void segmentation_NCG_kernel(int c, XSpace::NodeEdgesType *nodeedges, XSpace::EdgeNodesType *edgenodes, const int *colors,
//	float *similarities, int *labels, float threshold)
//{
//	// Cuda-style for
//	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < V; i += blockDim.x*gridDim.x)	{
//		if (colors[i] != c) continue;
//
//		// Here starts the segmentation 
//		//for (int k = 0; k < NCellNeighbors; k++, p++) {
//		for (int k = 0; k < NCellNeighbors; k++) {
//			int j = get_neighbor_id<XSpace>(i, k, nodeedges, edgenodes);
//			if (j == -1) continue;
//			int test = (get_similarity<XSpace>(i, k, nodeedges, similarities) <= threshold);
//			labels[i] = (((test << 31) >> 31) & min(labels[i], labels[j])) | (((!test << 31) >> 31) & labels[i]);
//			//printf("test = %i , labels[%i] = %i\n", test, i, labels[i]);
//			//if(i == 95) printf("i= %i, j= %i, similarity= %.2f, test= %i, labels[%i] = %i\n", i, j, get_similarity<XSpace>(i, k, nodeedges, similarities), test, i, labels[i]);
//		}
//	}
//}
//
//template<class XSpace>
//__global__ void segmentation_NCG_kernel2(int c, int *lower_bounds, int *sorted_cells, XSpace::NodeEdgesType *nodeedges, XSpace::EdgeNodesType *edgenodes, 
//	float *similarities, int *labels)
//{
//	// Cuda-style for
//	for (int id = lower_bounds[c] + threadIdx.x + blockIdx.x*blockDim.x; id < lower_bounds[c + 1]; id += blockDim.x*gridDim.x) {
//
//		int i = sorted_cells[id];
//		// Here starts the segmentation 
//		for (int k = 0; k < NCellNeighbors; ++k) {
//			int j = get_neighbor_id<XSpace>(i, k, nodeedges, edgenodes);
//			if (j == -1) continue;
//			int test = (get_similarity<XSpace>(i, k, nodeedges, similarities) <= Threshold);
//			labels[i] = (((test << 31) >> 31) & min(labels[i], labels[j])) | (((!test << 31) >> 31) & labels[i]);
//			//printf("test = %i , labels[%i] = %i\n", test, i, labels[i]);
//			if(i == 0) printf("i= %i, j= %i, similarity= %.2f, test= %i, labels[%i] = %i\n", i, j, get_similarity<XSpace>(i, k, nodeedges, similarities), test, i, labels[i]);
//		}
//		//labels[i] = c;
//	}
//}

template<class XSpace>
__global__ void segmentation_NCG_kernel3(int c, int *lower_bounds, int *sorted_cells, XSpace::NeighborsType *neighbors, XSpace::SimilarityType *similarities, int *labels)
{
	// Cuda-style for
	for (int id = lower_bounds[c] + threadIdx.x + blockIdx.x*blockDim.x; id < lower_bounds[c + 1]; id += blockDim.x*gridDim.x) {

		int i = sorted_cells[id];
		
		//printf("cell %i, parent = %i\n", i, labels[labels[i]]);
		
		if (labels[i] == 0) continue;
		
		// Here starts the segmentation 
		int lab = labels[labels[i]];

		// if Similarity[neighbor] == 0 request neighbor id
		// So, to avoid boundary operations, make sure out-of-bounds neighbors are verboten!
		for (int k = 0, j; k < NCellNeighbors; ++k) {
			if (!(similarities[i] & (1 << k))) {
				j = get_neighbor_id<XSpace>(i, k, neighbors);
				lab = min(lab, labels[labels[j]]);
				//if (i == 420) printf("cell %i, parent = %i, neighbor = %i, neighborparent = %i, lab = %i\n", i, labels[labels[i]], j, labels[labels[j]], lab);
			}
		}

		//if (labels[labels[i]] == 148) printf("cell %i, parent = %i, lab = %i\n", i, labels[labels[i]], lab);

		//labels[labels[i]] = min(lab, labels[labels[i]]);
		
		atomicMin(&labels[labels[i]], lab);
		//labels[labels[i]] = lab;
		labels[i] = labels[labels[i]]; // Path compression

		//if (i == 420) printf("cell %i, parent = %i, lab = %i\n", i, labels[labels[i]], lab);
		//labels[i] = lab;
	}
}

template<class XSpace>
__global__ void segmentation_NCG_kernel4(int c, int *lower_bounds, int *sorted_cells, XSpace::NeighborsType *neighbors, XSpace::SimilarityType *similarities, int *labels)
{
	// Cuda-style for
	for (int id = lower_bounds[c] + blockIdx.x; id < lower_bounds[c + 1]; id += gridDim.x) {
		
		// Cell index, Lane index
		int i = sorted_cells[id];
		int k = threadIdx.x/* & (WARP_SIZE - 1)*/;
		
		if (labels[i] == 0) continue;
		// Here starts the segmentation 

		int lab = labels[labels[i]];

		// if Similarity[neighbor] == 0 request neighbor id
		// So, to avoid boundary operations, make sure out-of-bounds neighbors are verboten!
		if ( k < NCellNeighbors && !(similarities[i] & (1 << k))) {
			int j = get_neighbor_id<XSpace>(i, k, neighbors);
			lab = labels[labels[j]];
			
			//if (i == 5960) printf("cell = %i, neighbor = %i\n", i, j);
		}

		// Warp Min-Reduce
		__syncthreads();
		for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
			lab = min(lab, __shfl_down(lab, offset));
		
		//if (i == 5960 && k == 0) printf("cell = %i, lab = %i\n", i, lab);
		if (k == 0) {
			labels[labels[i]] = lab;
			labels[i] = labels[labels[i]]; // Path compression
			//labels[i] = lab;
		}
	}
}

__global__ void path_compression_kernel(int *labels)
{
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < V; i += blockDim.x*gridDim.x)	{
		do {
			labels[i] = labels[labels[i]];
			//if (i == 3912) printf("labels[%i] = %i\n", i, labels[i]);
		} while (labels[i] != labels[labels[i]]);
	}
}

/*------------------------------------------*/
/* REGION DETECTION KERNELS */

__global__ void heads_kernel(int *labels, int *heads)
{
	__shared__ int sh[1024];
	
	// Initialize to avoid garbage!
	sh[threadIdx.x] = 0;

	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < V; i += blockDim.x*gridDim.x)	{
		//if (i == labels[i] /*&& blockIdx.x > 200*/) printf("%i is head\n", i);
		sh[threadIdx.x] = (labels[i] == i);
	}

	__syncthreads();
	for (unsigned int s = 512; s > 0; s >>= 1) {
		if (threadIdx.x < s)
			sh[threadIdx.x] += sh[threadIdx.x + s];
		__syncthreads();
	}

	if (threadIdx.x == 0) heads[blockIdx.x] = sh[0];
}

__global__ void filter_k(int *dst, const int *src, int n, int *ctr) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n)
		return;
	if (src[i] == i) {
		dst[atomicAggInc(ctr)] = src[i];
		/*int j = atomicAggInc(ctr);
		dst[j] = src[i];
		printf("Head %i: %i\n", j,i);*/
	}
}

template<class XSpace>
__global__ void filter_k_frontiers(int *dst, const int *src, int n, int *ctr, XSpace::NodeEdgesType *nodeedges, XSpace::EdgeNodesType *edgenodes, float *similarities, int *labels) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= n) return;
	
	int i = src[id];
	bool isFrontier = false;
	for (int k = 0; k < NCellNeighbors; ++k) {
		int j = get_neighbor_id<XSpace>(i, k, nodeedges, edgenodes);
		if (j == -1) continue;
		if (labels[i] != labels[j]) {
			isFrontier = true;
			break;
		}
	}
	
	if (isFrontier) {
		dst[atomicAggInc(ctr)] = i;
		/*int j = atomicAggInc(ctr);
		dst[j] = src[i];
		printf("Head %i: %i\n", j,i);*/
	}
}

__global__ void filter_k_c(int c, int *dst, const int *src, int n, int *ctr) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n)
		return;
	if (src[i] == c) {
		dst[atomicAggInc(ctr)] = i;
		/*int j = atomicAggInc(ctr);
		dst[j] = src[i];
		printf("Head %i: %i\n", j,i);*/
	}
}

__global__ void filter_k_not_c(int c, int *dst, const int *src, int n, int *ctr) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n)
		return;
	if (src[i] != c) {
		dst[atomicAggInc(ctr)] = i;
		/*int j = atomicAggInc(ctr);
		dst[j] = src[i];
		printf("Head %i: %i\n", j,i);*/
	}
}

/*------------------------------------------*/
/* BITONIC SORT KERNELS */

template<typename V>
__global__ void bitonic_sort_step(V *values, int n, int j, int k)
{
	unsigned int i, ixj, sort; /* Sorting partners: i and ixj */
	i = threadIdx.x + blockDim.x * blockIdx.x;
	ixj = i^j;
	sort = (!!(i&k)) ^ (!!(k&(n - 1)));

	//printf("k = %i, j = %i, i = %i, ixj = %i, values[i=%i] = %.2f, values[ixj=%i] = %.2f, sort = %i\n", k,j,i,ixj, i, values[i], ixj, values[ixj], sort);

	/* The threads with the lowest ids sort the array. */
	if ((ixj)>i && ixj < n) {
		if (sort == 0) {
			/* Sort ascending */
			if (values[i]>values[ixj]) {
				/* exchange(i,ixj); */
				V temp = values[i];
				values[i] = values[ixj];
				values[ixj] = temp;
				//printf("(1) k = %i, j = %i, i = %i, ixj = %i - Exchanged %.2f, %.2f\n", k, j, i, ixj, values[ixj], values[i]);
			}
		}
		if (sort != 0) {
			/* Sort descending */
			if (values[i]<values[ixj]) {
				/* exchange(i,ixj); */
				V temp = values[i];
				values[i] = values[ixj];
				values[ixj] = temp;
				//printf("(2) k = %i, j = %i, i = %i, ixj = %i - Exchanged %.2f, %.2f\n", k, j, i, ixj, values[ixj], values[i]);
			}
		}
	}
}

template<typename K, typename V>
__global__ void bitonic_sort_by_key_step(K *keys, V *values, int n, int j, int k)
{
	unsigned int i, ixj, sort; /* Sorting partners: i and ixj */
	i = threadIdx.x + blockDim.x * blockIdx.x;
	ixj = i^j;
	sort = (!!(i&k)) ^ (!!(k&(n - 1)));

	//printf("k = %i, j = %i, i = %i, ixj = %i, values[i=%i] = %.2f, values[ixj=%i] = %.2f, sort = %i\n", k,j,i,ixj, i, values[i], ixj, values[ixj], sort);

	/* The threads with the lowest ids sort the array. */
	if ((ixj)>i && ixj < n) {
		if (sort == 0) {
			/* Sort ascending */
			if (keys[i]>keys[ixj]) {
				/* exchange(i,ixj); */
				K temp = keys[i]; keys[i] = keys[ixj]; keys[ixj] = temp;
				V temp1 = values[i]; values[i] = values[ixj]; values[ixj] = temp1;
				//printf("(1) k = %i, j = %i, i = %i, ixj = %i - Exchanged %.2f, %.2f\n", k, j, i, ixj, values[ixj], values[i]);
			}
		}
		if (sort != 0) {
			/* Sort descending */
			if (keys[i]<keys[ixj]) {
				/* exchange(i,ixj); */
				K temp = keys[i]; keys[i] = keys[ixj]; keys[ixj] = temp;
				V temp1 = values[i]; values[i] = values[ixj]; values[ixj] = temp1;
				//printf("(2) k = %i, j = %i, i = %i, ixj = %i - Exchanged %.2f, %.2f\n", k, j, i, ixj, values[ixj], values[i]);
			}
		}
	}
}

template<typename K, typename V>
__global__ void bitonic_sort_by_composition_key_step(K* keys, V *values, int n, int j, int k)
{
	unsigned int i, ixj, sort; /* Sorting partners: i and ixj */
	i = threadIdx.x + blockDim.x * blockIdx.x;
	ixj = i^j;
	sort = (!!(i&k)) ^ (!!(k&(n - 1)));

	//printf("k = %i, j = %i, i = %i, ixj = %i, values[i=%i] = %.2f, values[ixj=%i] = %.2f, sort = %i\n", k,j,i,ixj, i, values[i], ixj, values[ixj], sort);

	/* The threads with the lowest ids sort the array. */
	if ((ixj)>i && ixj < n) {
		if (sort == 0) {
			/* Sort ascending */
			if (keys[values[i]]>keys[values[ixj]]) {
				/* exchange(i,ixj); */
				V temp = values[i];
				values[i] = values[ixj];
				values[ixj] = temp;
				//printf("(1) k = %i, j = %i, i = %i, ixj = %i - Exchanged %.2f, %.2f\n", k, j, i, ixj, values[ixj], values[i]);
			}
		}
		if (sort != 0) {
			/* Sort descending */
			if (keys[values[i]]<keys[values[ixj]]) {
				/* exchange(i,ixj); */
				V temp = values[i];
				values[i] = values[ixj];
				values[ixj] = temp;
				//printf("(2) k = %i, j = %i, i = %i, ixj = %i - Exchanged %.2f, %.2f\n", k, j, i, ixj, values[ixj], values[i]);
			}
		}
	}
}

template<typename K, typename V1, typename V2, typename V3, typename V4>
__global__ void bitonic_quadruple_sort_by_key_step(K *keys, V1 *values1, V2 *values2, V3 *values3, V4 *values4, int n, int j, int k)
{
	unsigned int i, ixj, sort; /* Sorting partners: i and ixj */
	i = threadIdx.x + blockDim.x * blockIdx.x;
	ixj = i^j;
	sort = (!!(i&k)) ^ (!!(k&(n - 1)));

	/* The threads with the lowest ids sort the array. */
	if ((ixj)>i && ixj < n) {
		if (sort == 0) {
			/* Sort ascending */
			if (keys[i]<keys[ixj]) {
				/* exchange(i,ixj); */
				K	temp = keys[i];	keys[i] = keys[ixj]; keys[ixj] = temp;
				V1 temp1 = values1[i];	values1[i] = values1[ixj];	values1[ixj] = temp1;
				V2 temp2 = values2[i];	values2[i] = values2[ixj];	values2[ixj] = temp2;
				V3 temp3 = values3[i];	values3[i] = values3[ixj];	values3[ixj] = temp3;
				V4 temp4 = values4[i];	values4[i] = values4[ixj];	values4[ixj] = temp4;
				//printf("(1) k = %i, j = %i, i = %i, ixj = %i - Exchanged %.2f, %.2f\n", k, j, i, ixj, values[ixj], values[i]);
			}
		}
		if (sort != 0) {
			/* Sort descending */
			if (keys[i]>keys[ixj]) {
				/* exchange(i,ixj); */
				K	temp = keys[i];	keys[i] = keys[ixj]; keys[ixj] = temp;
				V1 temp1 = values1[i];	values1[i] = values1[ixj];	values1[ixj] = temp1;
				V2 temp2 = values2[i];	values2[i] = values2[ixj];	values2[ixj] = temp2;
				V3 temp3 = values3[i];	values3[i] = values3[ixj];	values3[ixj] = temp3;
				V4 temp4 = values4[i];	values4[i] = values4[ixj];	values4[ixj] = temp4;
				//printf("(2) k = %i, j = %i, i = %i, ixj = %i - Exchanged %.2f, %.2f\n", k, j, i, ixj, values[ixj], values[i]);
			}
		}
	}
}

/*------------------------------------------*/
/* BOUNDS KERNEL */

__device__ inline int find_position2(int x, int* a, int r)
{
	int l = 0, m;
	while (l <= r)
	{
		m = l + (r - l) / 2;

		// Check if x is present at mid
		if (a[m] == x)
			return m;

		// If x greater, ignore left half  
		if (a[m] < x)
			l = m + 1;

		// If x is smaller, ignore right half 
		else
			r = m - 1;
	}
	// if we reach here, then element was not present
	return -1;
}

__global__ void bounds_kernel(int *region_bounds, int *sorted_cells, int *cell_labels, int *region_labels, int num_regions) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= V - 1) return;
	
	if (cell_labels[sorted_cells[i]] != cell_labels[sorted_cells[i-1]]) {
		int j = find_position2(cell_labels[sorted_cells[i]], region_labels, num_regions);
		region_bounds[j] = i + 1;
		//printf("cell = %i, label = %i, region_bounds[%i] = %i\n", sorted_cells[i], cell_labels[sorted_cells[i]], j, region_bounds[j]);
	}
}

__global__ void inv_prefixsum_kernel(int *region_numcells, int *region_bounds, int num_regions) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_regions) return;

	region_numcells[i] = region_bounds[i + 1] - region_bounds[i];
}

/*------------------------------------------*/
/* CELL REDUCTION BY KEY (GIVEN SORTED CELLS BY KEY) */

template<class BaseClass> 
__device__ inline float get_cell_size(int id, float* cell_sizes);

template<> 
__device__ inline float get_cell_size<CDelaunay>(int id, float* cell_sizes) 
{	
	return cell_sizes[id];	
}

template<> 
__device__ inline float get_cell_size<CImage>(int id, float* cell_sizes)
{	
	return 1.0f;	
}

template<class BaseClass>
__global__ void cell_reduction_kernel(int* lower_bounds, int* sorted_cells, float* cell_sizes, float* cell_colorpatterns,
	/*int* region_numcells,*/ float* region_sizes, float* region_colorpatterns)
{
	__shared__ float colorpatterns[1024];
	__shared__ float sizes[1024];
	//__shared__ int numcells[1024];

	colorpatterns[threadIdx.x] = 0.0f;
	sizes[threadIdx.x] = 0.0f;
	//numcells[threadIdx.x] = 0;

	for (int i = lower_bounds[blockIdx.x] + threadIdx.x; i < lower_bounds[blockIdx.x + 1]; i += blockDim.x) {
		int id = sorted_cells[i];
		float denom = sizes[threadIdx.x] + get_cell_size<BaseClass>(id, cell_sizes);
		colorpatterns[threadIdx.x] = (colorpatterns[threadIdx.x] * sizes[threadIdx.x] + cell_colorpatterns[id] * get_cell_size<BaseClass>(id, cell_sizes)) / (denom + !denom);
		sizes[threadIdx.x] += get_cell_size<BaseClass>(id, cell_sizes);
		//numcells[threadIdx.x] += 1;
	}
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			float denom = sizes[threadIdx.x] + sizes[threadIdx.x + s];
			colorpatterns[threadIdx.x] = (colorpatterns[threadIdx.x] * sizes[threadIdx.x] + colorpatterns[threadIdx.x + s] * sizes[threadIdx.x + s]) / (denom + !denom);
			sizes[threadIdx.x] += sizes[threadIdx.x + s];
			//numcells[threadIdx.x] += numcells[threadIdx.x + s];
		}
		__syncthreads();
	}

	//region_numcells[blockIdx.x] = numcells[0];
	region_sizes[blockIdx.x] = sizes[0];
	region_colorpatterns[blockIdx.x] = colorpatterns[0];
}

/*------------------------------------------*/
/* COMPUTE REGION GAMMAS */

__global__ void compute_gammas_kernel(int *region_numcells, float *region_cellsizes, float *region_gammas, int num_regions)
{
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < num_regions; i += blockDim.x*gridDim.x) {
		region_gammas[i] = (float(region_numcells[i]) / V) + (region_cellsizes[i] / Vol);
	}
}

/////////////////////////////////////////////////////////
/*------------------------------------------*/
/* OPERATOR # 2 : NEIGHBOR REGION GROUPING KERNELS *///

template<class XSpace>
__global__ void segmentation_NRG_kernel(int nR, int *lower_bounds, int *old_labels, int *sorted_cells, int *cell_labels, XSpace::NeighborsType *neighbors, XSpace::SimilarityType *similarities, float *region_colorpatterns, int *region_labels)
{
	extern __shared__ float shared_similarities[];

	if (region_labels[nR + blockIdx.x] < nR)
		return;

	for (int i = threadIdx.x; i < nR; i += blockDim.x)
		shared_similarities[i] = 1.0f;

	/*//if (nR + blockIdx.x == 176) 
	//if (threadIdx.x == 0)
		//printf("block = %i, nR + blockIdx.x = %i, lower_bound[min] = %i, lower_bound[max] = %i, diff = %i\n", blockIdx.x, nR + blockIdx.x, lower_bounds[region_labels[nR + blockIdx.x]], lower_bounds[region_labels[nR + blockIdx.x] + 1], lower_bounds[region_labels[nR + blockIdx.x] + 1] - lower_bounds[region_labels[nR + blockIdx.x]]);*/
	
	int old_label = old_labels[region_labels[nR + blockIdx.x]];
	// Cuda-style for
	//for (int i = lower_bounds[region_labels[nR + blockIdx.x]] + threadIdx.x; i < lower_bounds[region_labels[nR + blockIdx.x] + 1]; i += blockDim.x) {
	for (int i = lower_bounds[old_label] + threadIdx.x; i < lower_bounds[old_label + 1]; i += blockDim.x) {
		
		//if (nR + blockIdx.x == 176) printf("nR + blockIdx.x = %i, block = %i, i = %i\n", nR + blockIdx.x, blockIdx.x, i);
		int id = sorted_cells[i];
		int reg = region_labels[cell_labels[id]];
		//if (nR + blockIdx.x == 176) printf("block = %i, i = %i, id = %i, reg = %i\n", blockIdx.x, i, id, region_labels[cell_labels[id]]);
		
		for (int k = 0; k < NCellNeighbors; k++) {
			if (!(similarities[id] & (1 << k))){

				int j = get_neighbor_id<XSpace>(id, k, neighbors);
				////if (j < 0) printf("block = %i, i = %i, id = %i, reg = %i\n", blockIdx.x, i, id, region_labels[cell_labels[id]]);
				int neigh = region_labels[cell_labels[j]];

				//if (nR + blockIdx.x == 64) printf("id = %i, neigh = %i, neighreg = %i\n", id, j, neigh);
				if (neigh < nR) {
					shared_similarities[neigh] = similarity<false>(region_colorpatterns[reg], region_colorpatterns[neigh]);
					//if (nR + blockIdx.x == 64) printf("neighbor %i, similarity %.4f\n", neigh, shared_similarities[neigh]);
				}
			}
		}
	}
	
	__syncthreads();

	if (threadIdx.x == 0)
	{
		//if (nR + blockIdx.x == 64)
		//for (int i = 0; i < nR; ++i) {
		//	printf("Region %i, shared_similarities[%i] = %.2f\n", nR + blockIdx.x, i, shared_similarities[i]);
		//}

		int min_element = region_labels[nR + blockIdx.x];
		float min_similarity = 1.0f;

		for (int i = 0; i < nR; ++i) {
			if (shared_similarities[i] < min_similarity) {
				min_similarity = shared_similarities[i];
				min_element = i;
			}

		}

		//if (nR + blockIdx.x == 176) printf("Region %i, Min_element = %i\n", nR + blockIdx.x, min_element);
		region_labels[nR + blockIdx.x] = min_element;
	}
	
}

template<class XSpace>
__global__ void mark_neighbors(int current_reg, int *lower_bounds, int *sorted_cells, int *cell_labels, XSpace::NodeEdgesType *nodeedges, int2 *edgenodes, int *region_labels, int *region_neighborhoods)
{
	// Cuda-style for
	for (int id = lower_bounds[current_reg] + threadIdx.x + blockIdx.x*blockDim.x; id < lower_bounds[current_reg + 1]; id += blockDim.x*gridDim.x) {
	
		int i = sorted_cells[id];
		int j, neigh;
		//printf("block %i, reg %i, id %i, i %i\n", blockIdx.x, current_reg, id, i);
		// Here starts the segmentation 
		for (int k = 0; k < NCellNeighbors; ++k) {
			j = get_neighbor_id<XSpace>(i, k, nodeedges, edgenodes);
			if (j == -1) continue;
			neigh = region_labels[cell_labels[j]];
			if(neigh != current_reg) region_neighborhoods[neigh] = 1;
		}
	}
}

__global__ void get_region_similarities_kernel(int current_reg, int *neighbors, int num_neighbors, float* region_colorpatterns, float *neighbor_similarities)
{
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < num_neighbors; i += blockDim.x*gridDim.x)	{

		int neigh = neighbors[i];
		neighbor_similarities[i] = similarity<false>(region_colorpatterns[current_reg], region_colorpatterns[neigh]);
		//printf("simil %i (%.4f) %i (%.4f) = %.4f \n", current_reg, region_colorpatterns[current_reg], neigh, region_colorpatterns[neigh], neighbor_similarities[i]);
	}
}

/////////////////////////////////////////////////////////
/*------------------------------------------*/
/* OPERATOR # 3 : SIMILAR REGION GROUPING KERNELS *///

__global__ void all_similarities_kernel(int nR, int *region_labels, float* region_colorpatterns, float *all_similarities, int *all_similarities_coords)
{
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < nR*nR; i += blockDim.x*gridDim.x)	{
		int y = i / nR;
		int x = i % nR;

		all_similarities[i] = fmaxf(similarity<true>(region_colorpatterns[x], region_colorpatterns[y]),float((x==y)));
		all_similarities_coords[i] = i;
		//all_similarities_coords[i].x = fminf(x,y);
		//all_similarities_coords[i].y = fmaxf(x,y);

		//printf("simil %i (%.4f) %i (%.4f) = %.4f \n", region_labels[x], region_colorpatterns[x], region_labels[y], region_colorpatterns[y], all_similarities[i]);
	}
}


/*------------------------------------------*/
/* RELABELS: for CELLS and for REGIONS */

__global__ void relabel_kernel_1(int n, int num_regions, int *labels, int *region_labels)
{
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < n; i += blockDim.x*gridDim.x)	{
		labels[i] = find_position2(labels[i], region_labels, num_regions);
	}
}

__global__ void relabel_kernel_2(int *labels, int *region_labels)
{
	// Cuda-style for
	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < V; i += blockDim.x*gridDim.x)	{
		labels[i] = region_labels[labels[i]];
	}
}

__global__ void relabel_kernel_3(int* lower_bounds, int* region_labels, int* cell_labels, int* sorted_cells)
{
	for (int i = lower_bounds[blockIdx.x] + threadIdx.x; i < lower_bounds[blockIdx.x + 1]; i += blockDim.x) {
		cell_labels[sorted_cells[i]] = blockIdx.x;
	}
}

__global__ void relabel_kernel_4(int* lower_bounds, int* region_labels, int* cell_labels, int* sorted_cells)
{
	for (int i = lower_bounds[region_labels[blockIdx.x]] + threadIdx.x; i < lower_bounds[region_labels[blockIdx.x] + 1]; i += blockDim.x) {
		cell_labels[sorted_cells[i]] = blockIdx.x;
	}
}

/////////////////////////////////////////////////////////