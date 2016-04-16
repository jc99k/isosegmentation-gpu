#include "img_reader.h"

using namespace std;
using namespace cimg_library;

template<class G>
CImageReader<G>::CImageReader(float* imgdata, cudaTextureObject_t& texObj) : m_imagedata(imgdata), m_ptexObject(&texObj) {}

template<class G>
void CImageReader<G>::load_file()
{
	string filename;
	cout << "Enter Image filename: ";
	cin >> filename;
	ifstream is(filename, ifstream::in);
	CImg<float> image(filename.c_str());
	//image = image.RGBtoLab().channel(0).normalize(0, 255);

	m_imageextent = make_int3(image.width(), image.height(), image.depth());
	size_t pitch, widthInBytes = image.width()*sizeof(float);

	cudaMallocPitch(&m_imagedata, &pitch, widthInBytes, image.height());
	cudaMemcpy2D(m_imagedata, pitch, image.data(), widthInBytes, widthInBytes, image.height(), cudaMemcpyHostToDevice);

	// Texture (new way) : Create texture object
	// 1. Resource Descriptor (resource to texture with)
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = m_imagedata;
	resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
	resDesc.res.pitch2D.width = image.width();
	resDesc.res.pitch2D.height = image.height();
	resDesc.res.pitch2D.pitchInBytes = pitch;

	// 2. Texture Descriptor (texturing settings)
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	texDesc.filterMode = cudaFilterModeLinear;      // linear interpolation
	texDesc.addressMode[0] = cudaAddressModeWrap;   // wrap texture coordinates
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.addressMode[2] = cudaAddressModeWrap; // 3D
	
	// 3. Create Texture Object (with descriptors)
	cudaCreateTextureObject(m_ptexObject, &resDesc, &texDesc, NULL);
}

template<class G>
int3& CImageReader<G>::image_extent() { return m_imageextent; }


template class CImageReader<int>;