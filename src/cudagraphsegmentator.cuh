#include "cudagraph.cuh"

#include "csr_reader.h"
#include "csr_writer.h"
#include "img_reader.h"
#include "isovaluelist.h"

template<class XSpace>
class CCudaCSRGraphSegmentator
{
	typedef CCudaCSRGraph<XSpace> cudaCSRGraph_t;
	typedef CGraphReader<cudaCSRGraph_t> cudaCSRGraphReader_t;
    typedef CGraphWriter<cudaCSRGraph_t> cudaCSRGraphWriter_t;
	typedef CImageReader<XSpace> cudaImageReader_t;
	typedef CIsoValueList<XSpace> CIsovalueList_t;

public:
	CCudaCSRGraphSegmentator();
	~CCudaCSRGraphSegmentator();
	void colorize();
	void compute_colorpatterns();
    void compute_similarities();
	void write();

private:
	cudaCSRGraph_t			m_csrGraph;
	cudaTextureObject_t		m_texObject;
	float* d_imagedata;
	CIsovalueList_t			m_isovaluelist;

	int &v, &e, &dv, &num_isovalues, num_colors;
	dim3 nt, nb;
};
