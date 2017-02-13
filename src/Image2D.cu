#include "Image2D.h"

CImage_2D_Cuda_XGraph_Adaptor::~CImage_2D_Cuda_XGraph_Adaptor()
{
	//cudaFree(m_updatepatterns);
	cudaFree(m_sortedcells);
	cudaFree(m_labels);
	cudaFree(m_colorpatterns);
	cudaFree(m_similarities);
}

void CImage_2D_Cuda_XGraph_Adaptor::UnifiedMalloc()
{
	//cudaMallocManaged(&m_updatepatterns, (v)*sizeof(int));
	cudaMallocManaged(&m_sortedcells, (v)*sizeof(int));
	cudaMallocManaged(&m_labels, (v)*sizeof(int));
	cudaMallocManaged(&m_colorpatterns, (v)*sizeof(float));
	//cudaMallocManaged(&m_similarities, (e)*sizeof(float));

	cudaMallocManaged(&m_similarities, (v)*sizeof(SimilarityType));
}

void CImage_2D_Cuda_XGraph_Adaptor::Synchronize()
{
	cudaDeviceSynchronize();
}

void CImage_2D_Cuda_XGraph_Adaptor::SetTexture(ImgType* imgdata, int width, int height)
{
	m_image_dimensions = make_int3(width, height, 1);
	cudaMemcpy(m_colorpatterns, imgdata, m_vol*sizeof(ImgType), cudaMemcpyHostToHost);
	m_pointcoords = m_colorpatterns;
}

void CImage_2D_Cuda_XGraph_Adaptor::FreeTexture()
{
	//cudaDestroyTextureObject(m_texObj);
}

const int CImage_2D_Cuda_XGraph_Adaptor::Dim = 2;
const int CImage_2D_Cuda_XGraph_Adaptor::NCellVertices = 1;
const int CImage_2D_Cuda_XGraph_Adaptor::NCellNeighbors = 8;
const int CImage_2D_Cuda_XGraph_Adaptor::NFacetVertices = 2;
const int CImage_2D_Cuda_XGraph_Adaptor::NGraphColors = 4;