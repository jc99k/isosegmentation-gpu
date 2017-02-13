#include "Delaunay2D.h"

CDelaunay_2D_Cuda_XGraph_Adaptor::~CDelaunay_2D_Cuda_XGraph_Adaptor()
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

void CDelaunay_2D_Cuda_XGraph_Adaptor::UnifiedMalloc()
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

void CDelaunay_2D_Cuda_XGraph_Adaptor::Synchronize()
{
	cudaDeviceSynchronize();
}

void CDelaunay_2D_Cuda_XGraph_Adaptor::SetTexture(ImgType* imgdata, int width, int height)
{
	m_image_dimensions = make_int3(width, height, 1);
	m_vol = width*height;
	size_t pitch, widthInBytes = width*sizeof(ImgType);

	// Allocate pitched (padded) memory in device for texture memory
	// Then copy image data from host into pitched memory, respecting the padding
	cudaMallocPitch(&m_texdata, &pitch, widthInBytes, height);
	cudaMemcpy2D(m_texdata, pitch, imgdata, widthInBytes, widthInBytes, height, cudaMemcpyHostToDevice);

	// Texture (new way) : Create texture object
	// 1. Resource Descriptor (resource to texture with)
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = m_texdata;
	resDesc.res.pitch2D.desc = cudaCreateChannelDesc<ImgType>();
	resDesc.res.pitch2D.width = width;
	resDesc.res.pitch2D.height = height;
	resDesc.res.pitch2D.pitchInBytes = pitch;

	// 2. Texture Descriptor (texturing settings)
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	texDesc.filterMode = cudaFilterModeLinear;      // linear interpolation
	texDesc.addressMode[0] = cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaAddressModeBorder;
	texDesc.readMode = cudaReadModeElementType;

	// 3. Create Texture Object (with descriptors)
	cudaCreateTextureObject(&m_texObj, &resDesc, &texDesc, NULL);

	//printf("Texture memory ready!\n");
}

void CDelaunay_2D_Cuda_XGraph_Adaptor::FreeTexture()
{
	cudaDestroyTextureObject(m_texObj);
	cudaFree(m_texdata);
}

const int CDelaunay_2D_Cuda_XGraph_Adaptor::Dim = 2;
const int CDelaunay_2D_Cuda_XGraph_Adaptor::NCellVertices = 3;
const int CDelaunay_2D_Cuda_XGraph_Adaptor::NCellNeighbors = 3;
const int CDelaunay_2D_Cuda_XGraph_Adaptor::NFacetVertices = 2;
const int CDelaunay_2D_Cuda_XGraph_Adaptor::NGraphColors = 4;