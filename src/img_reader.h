#ifndef CUDA_IMG_READER_H
#define CUDA_IMG_READER_H

#include <iostream>
#include <fstream>
#include <string>
#include "CImg.h"
#include "vector_types.h"

template<class G>
class CImageReader
{

public:

	CImageReader(float* imgdata, cudaTextureObject_t& texObj);
	void load_file();
	int3& image_extent();

private:
	cudaTextureObject_t*	m_ptexObject;
	float* m_imagedata;
	int3 m_imageextent;
};

#endif //CUDA_IMG_READER_H
