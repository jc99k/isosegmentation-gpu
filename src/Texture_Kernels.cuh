#ifndef ISOSEGMENTATION_CUDA_TEXTURE_KERNELS_H
#define ISOSEGMENTATION_CUDA_TEXTURE_KERNELS_H

/*------------------------------------------*/
/* GENERIC TEXTURE FETCH (2D/3D) */

template<class XSpace>
__device__ inline float tex(cudaTextureObject_t &texObj, XSpace::PointCoordsType &point);

template<>
__device__ inline float tex<CDelaunay_2D_Cuda_XGraph_Adaptor>(cudaTextureObject_t &texObj, float2 &p)
{
	return tex2D<float>(texObj, p.x, p.y);
}

template<>
__device__ inline float tex<CDelaunay_3D_Cuda_XGraph_Adaptor>(cudaTextureObject_t &texObj, float3 &p)
{
	return tex3D<float>(texObj, p.x, p.y, p.z);
}

#endif //ISOSEGMENTATION_CUDA_TEXTURE_KERNELS_H
