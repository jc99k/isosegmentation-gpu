#include "Delaunay3D.h"

CDelaunay_3D_Cuda_XGraph_Adaptor::~CDelaunay_3D_Cuda_XGraph_Adaptor()
{
	cudaFree(m_pointcoords);
	cudaFree(m_cellvertices);
	//cudaFree(m_nodeedges);
	//cudaFree(m_updatepatterns);
	cudaFree(m_sortedcells);
	cudaFree(m_colorpatterns);
	cudaFree(m_labels);
	cudaFree(m_sizes);
	//cudaFree(m_facetvertices);
	//cudaFree(m_edgenodes);
	//cudaFree(m_similarities);
	
	cudaFree(m_neighbors);
	cudaFree(m_similarities);
	FreeTexture();
}

void CDelaunay_3D_Cuda_XGraph_Adaptor::UnifiedMalloc()
{
	cudaMallocManaged(&m_pointcoords, (dv)*sizeof(PointCoordsType));
	cudaMallocManaged(&m_cellvertices, (v)*sizeof(CellVerticesType));
	//cudaMallocManaged(&m_nodeedges, (v)*sizeof(NodeEdgesType));
	//cudaMallocManaged(&m_updatepatterns, (v)*sizeof(int));
	cudaMallocManaged(&m_sortedcells, (v)*sizeof(int));
	cudaMallocManaged(&m_colorpatterns, (v)*sizeof(float));
	cudaMallocManaged(&m_labels, (v)*sizeof(int));
	cudaMallocManaged(&m_sizes, (v)*sizeof(float));
	//cudaMallocManaged(&m_facetvertices, (e)*sizeof(FacetsType));
	//cudaMallocManaged(&m_edgenodes, (e)*sizeof(EdgeNodesType));
	//cudaMallocManaged(&m_similarities, (e)*sizeof(float));

	cudaMallocManaged(&m_neighbors, (v)*sizeof(NeighborsType));
	cudaMallocManaged(&m_similarities, (v)*sizeof(SimilarityType));
}

void CDelaunay_3D_Cuda_XGraph_Adaptor::Synchronize()
{
	cudaDeviceSynchronize();
}

void CDelaunay_3D_Cuda_XGraph_Adaptor::SetTexture(ImgType* imgdata, int width, int height, int depth)
{
	m_image_dimensions = make_int3(width, height, depth);
	m_vol = width*height*depth;
	// CUDA MALLOC 3D ARRAY

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<ImgType>();
	cudaExtent volumeSize = make_cudaExtent(width, height, depth);
	cudaMalloc3DArray(&m_texdata, &channelDesc, volumeSize);

	// CUDA MEMCPY3D : Pitched-imgdata (Host) -> m_texdata (Device CUDA Array)

	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(imgdata, width*sizeof(ImgType), width, height);
	copyParams.dstArray = m_texdata;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	
	cudaMemcpy3D(&copyParams);

	// Texture (new way) : Create texture object
	// 1. Resource Descriptor (resource to texture with)
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = m_texdata;

	// 2. Texture Descriptor (texturing settings)
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.normalizedCoords = false;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.addressMode[0] = cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaAddressModeBorder;
	texDesc.addressMode[2] = cudaAddressModeBorder;
	texDesc.readMode = cudaReadModeElementType;

	// 3. Create Texture Object (with descriptors)
	cudaCreateTextureObject(&m_texObj, &resDesc, &texDesc, NULL);

	//printf("Texture memory ready!\n");
}

void CDelaunay_3D_Cuda_XGraph_Adaptor::FreeTexture()
{
	cudaDestroyTextureObject(m_texObj);
	cudaFreeArray(m_texdata);
}

const int CDelaunay_3D_Cuda_XGraph_Adaptor::Dim = 3;
const int CDelaunay_3D_Cuda_XGraph_Adaptor::NCellVertices = 4;
const int CDelaunay_3D_Cuda_XGraph_Adaptor::NCellNeighbors = 4;
const int CDelaunay_3D_Cuda_XGraph_Adaptor::NFacetVertices = 3;
const int CDelaunay_3D_Cuda_XGraph_Adaptor::NGraphColors = 5;