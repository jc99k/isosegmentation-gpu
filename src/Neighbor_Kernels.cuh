#ifndef ISOSEGMENTATION_CUDA_NEIGHBOR_KERNELS_H
#define ISOSEGMENTATION_CUDA_NEIGHBOR_KERNELS_H

//#include "kernel_constants.h"

__constant__ int ImageNeighborSteps[26];

/*------------------------------------------*/
/* GENERIC NEIGHBOR FUNCTIONS (2D/3D) */

/* GET NEIGHBOR ID */
template<class XSpace>
__device__ inline int get_neighbor_id(int i, int k, XSpace::NeighborsType *neighbors)
{
	return *((int*)&neighbors[i] + k);
}

template<>
__device__ inline int get_neighbor_id<CImage_2D_Cuda_XGraph_Adaptor>(int i, int k, CImage_2D_Cuda_XGraph_Adaptor::NeighborsType *neighbors)
{
	return i + ImageNeighborSteps[k];
}

template<>
__device__ inline int get_neighbor_id<CImage_3D_Cuda_XGraph_Adaptor>(int i, int k, CImage_3D_Cuda_XGraph_Adaptor::NeighborsType *neighbors)
{
	return i + ImageNeighborSteps[k];
}

///////////////////////////////////////////
///* GET SIMILARITY */
//
//template<class XSpace>
//__device__ inline float get_similarity(int i, int k, XSpace::NodeEdgesType *nodeedges, float *similarities)
//{
//	int *p = &nodeedges[i].x + k;
//	return similarities[*p];
//}
//
//
//template<>
//__device__ inline float get_similarity<CImage_2D_Cuda_XGraph_Adaptor>(int i, int k, CImage_2D_Cuda_XGraph_Adaptor::NodeEdgesType *nodeedges, float *similarities)
//{
//	float simil;
//	switch (k) {
//	case 0:
//		simil = similarities[i*2];
//		break;
//	case 1:
//		simil = similarities[(i-1) * 2];
//		break;
//	case 2:
//		simil = similarities[i*2 + 1];
//		break;
//	case 3:
//		simil = similarities[(i - Img_width) * 2 + 1];
//		break;
//	}
//	return simil;
//}
//
//template<>
//__device__ inline float get_similarity<CImage_3D_Cuda_XGraph_Adaptor>(int i, int k, CImage_3D_Cuda_XGraph_Adaptor::NodeEdgesType *nodeedges, float *similarities)
//{
//	float simil;
//	switch (k) {
//	case 0:
//		simil = similarities[i * 3];
//		break;
//	case 1:
//		simil = similarities[(i - 1) * 3];
//		break;
//	case 2:
//		simil = similarities[i * 3 + 1];
//		break;
//	case 3:
//		simil = similarities[(i - Img_width) * 3 + 1];
//		break;
//	case 4:
//		simil = similarities[i * 3 + 2];
//		break;
//	case 5:
//		simil = similarities[(i - Img_width*Img_height) * 3 + 2];
//		break;
//	}
//	return simil;
//}

#endif //ISOSEGMENTATION_CUDA_NEIGHBOR_KERNELS_H