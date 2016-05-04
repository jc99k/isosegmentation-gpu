#include "Delaunay3D.cuh"

CDelaunay_3D_Cuda_XGraph_Adaptor::~CDelaunay_3D_Cuda_XGraph_Adaptor()
{
	cudaFree(m_xadj);
	cudaFree(m_adjncy);
	cudaFree(m_updatepatterns);
	cudaFree(m_labels);
	cudaFree(m_colorpatterns);
	cudaFree(m_similarities);
	cudaFree(m_cellvertices);
	cudaFree(m_delaunayvertices);

	cudaDestroyTextureObject(m_texObj);
	if(d_volumeArray) cudaFreeArray(d_volumeArray);
}

void CDelaunay_3D_Cuda_XGraph_Adaptor::UnifiedMalloc()
{
	cudaMallocManaged(&m_xadj, (v + 1)*sizeof(int));
	cudaMallocManaged(&m_adjncy, (e * 2)*sizeof(int));
	cudaMallocManaged(&m_updatepatterns, (v)*sizeof(int));
	cudaMallocManaged(&m_labels, (v)*sizeof(int));
	cudaMallocManaged(&m_colorpatterns, (v)*sizeof(float));
	cudaMallocManaged(&m_similarities, (e * 2)*sizeof(float));
	cudaMallocManaged(&m_cellvertices, (v)*sizeof(int4));
	cudaMallocManaged(&m_delaunayvertices, (dv)*sizeof(float3));
}

void CDelaunay_3D_Cuda_XGraph_Adaptor::Synchronize()
{
	cudaDeviceSynchronize();
}

void CDelaunay_3D_Cuda_XGraph_Adaptor::SetTexture(ImgType* imgdata, int width, int height, int depth)
{
	m_image_dimensions = make_int3(width, height, depth);

	// CUDA MALLOC 3D ARRAY

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<ImgType>();
	cudaExtent volumeSize = make_cudaExtent(width, height, depth);
	cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize);

	// CUDA MEMCPY3D : Pitched-imgdata (Host) -> d_volumeArray (Device CUDA Array)

	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(imgdata, width*sizeof(ImgType), height, depth);
	copyParams.dstArray = d_volumeArray;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

	// Texture (new way) : Create texture object
	// 1. Resource Descriptor (resource to texture with)
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = d_volumeArray;

	// 2. Texture Descriptor (texturing settings)
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.normalizedCoords = false;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.addressMode[2] = cudaAddressModeClamp;
	texDesc.readMode = cudaReadModeElementType;

	// 3. Create Texture Object (with descriptors)
	cudaCreateTextureObject(&m_texObj, &resDesc, &texDesc, NULL);

	printf("FINISHED TEXTURE ALLOC\n");
}

const int CDelaunay_3D_Cuda_XGraph_Adaptor::Dim = 3;
const int CDelaunay_3D_Cuda_XGraph_Adaptor::NCellVertices = 4;
const int CDelaunay_3D_Cuda_XGraph_Adaptor::NCellNeighbors = 4;