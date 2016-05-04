#ifndef ISOSEGMENTATION_CUDA_DELAUNAY3D_H
#define ISOSEGMENTATION_CUDA_DELAUNAY3D_H

#include <string>
#include <helper_math.h>
#include "vector_types.h"

class CDelaunay_3D_Cuda_XGraph_Adaptor
{
public:
	typedef float ImgType;
	typedef int4 CellVerticesType;
	typedef float3 DelaunayVerticesType;

	~CDelaunay_3D_Cuda_XGraph_Adaptor();
	void UnifiedMalloc();
	void Synchronize();
	void SetTexture(ImgType *imgdata, int width, int height, int depth);

	inline int* xadj()					{ return m_xadj; }
	inline int* adjncy()				{ return m_adjncy; }
	inline int* updatepatterns()		{ return m_updatepatterns; }
	inline int* labels()				{ return m_labels; }
	inline float* colorpatterns()		{ return m_colorpatterns; }
	inline float* similarities()		{ return m_similarities; }
	inline CellVerticesType* cellvertices()			{ return m_cellvertices; }
	inline DelaunayVerticesType* delaunayvertices()	{ return m_delaunayvertices; }

	inline int& num_vertices()				{ return v; }
	inline int& num_edges()					{ return e; }
	inline int& num_delaunayvertices()		{ return dv; }
	inline int& xadj(int i)					{ return m_xadj[i]; }
	inline int& adjncy(int i)				{ return m_adjncy[i]; }
	inline int& updatepatterns(int i)		{ return m_updatepatterns[i]; }
	inline int& labels(int i)				{ return m_labels[i]; }
	inline float& colorpatterns(int i)		{ return m_colorpatterns[i]; }
	inline float& similarities(int i)		{ return m_similarities[i]; }
	inline CellVerticesType& cellvertices(int i)		{ return m_cellvertices[i]; }
	inline DelaunayVerticesType& delaunayvertices(int i)	{ return m_delaunayvertices[i]; }

	inline int3& image_dimensions()			{ return m_image_dimensions; }
//	inline ImgType* image_data()				{ return m_imagedata; }
	inline cudaTextureObject_t& texObject()	{ return m_texObj; }

	inline std::string& filename() { return m_filename; }

	static const int Dim, NCellVertices, NCellNeighbors;

private:
	int v, e, dv;
	int *m_xadj, *m_adjncy, *m_updatepatterns, *m_labels;
	float *m_colorpatterns, *m_similarities;
	CellVerticesType *m_cellvertices;
	DelaunayVerticesType  *m_delaunayvertices;

	int3 m_image_dimensions;
	cudaArray * d_volumeArray;
	cudaTextureObject_t m_texObj;

	std::string m_filename;
};


#endif //ISOSEGMENTATION_CUDA_DELAUNAY3D_H