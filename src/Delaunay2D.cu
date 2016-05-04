#include "Delaunay2D.cuh"

CDelaunay_2D_Cuda_XGraph_Adaptor::~CDelaunay_2D_Cuda_XGraph_Adaptor()
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
	cudaFree(m_imagedata);
}

void CDelaunay_2D_Cuda_XGraph_Adaptor::UnifiedMalloc()
{
	cudaMallocManaged(&m_xadj, (v + 1)*sizeof(int));
	cudaMallocManaged(&m_adjncy, (e * 2)*sizeof(int));
	cudaMallocManaged(&m_updatepatterns, (v)*sizeof(int));
	cudaMallocManaged(&m_labels, (v)*sizeof(int));
	cudaMallocManaged(&m_colorpatterns, (v)*sizeof(float));
	cudaMallocManaged(&m_similarities, (e * 2)*sizeof(float));
	cudaMallocManaged(&m_cellvertices, (v)*sizeof(int3));
	cudaMallocManaged(&m_delaunayvertices, (dv)*sizeof(float2));
}

void CDelaunay_2D_Cuda_XGraph_Adaptor::Synchronize()
{
	cudaDeviceSynchronize();
}

void CDelaunay_2D_Cuda_XGraph_Adaptor::SetTexture(ImgType* imgdata, int width, int height)
{
	m_image_dimensions = make_int3(width, height, 1);
	size_t pitch, widthInBytes = width*sizeof(ImgType);

	cudaMallocPitch(&m_imagedata, &pitch, widthInBytes, height);
	cudaMemcpy2D(m_imagedata, pitch, imgdata, widthInBytes, widthInBytes, height, cudaMemcpyHostToDevice);

	// Texture (new way) : Create texture object
	// 1. Resource Descriptor (resource to texture with)
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = m_imagedata;
	resDesc.res.pitch2D.desc = cudaCreateChannelDesc<ImgType>();
	resDesc.res.pitch2D.width = width;
	resDesc.res.pitch2D.height = height;
	resDesc.res.pitch2D.pitchInBytes = pitch;

	// 2. Texture Descriptor (texturing settings)
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	texDesc.filterMode = cudaFilterModeLinear;      // linear interpolation
	texDesc.addressMode[0] = cudaAddressModeWrap;   // wrap texture coordinates
	texDesc.addressMode[1] = cudaAddressModeWrap;

	// 3. Create Texture Object (with descriptors)
	cudaCreateTextureObject(&m_texObj, &resDesc, &texDesc, NULL);
}

const int CDelaunay_2D_Cuda_XGraph_Adaptor::Dim = 2;
const int CDelaunay_2D_Cuda_XGraph_Adaptor::NCellVertices = 3;
const int CDelaunay_2D_Cuda_XGraph_Adaptor::NCellNeighbors = 3;