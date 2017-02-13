#ifndef ISOSEGMENTATION_CUDA_DELAUNAY2D_H
#define ISOSEGMENTATION_CUDA_DELAUNAY2D_H

#include "XBases.h"

class CDelaunay_2D_Cuda_XGraph_Adaptor : public CDelaunay
{
public:
	typedef CDelaunay BaseClass;
	typedef float ImgType;

	typedef int2 EdgeNodesType;
	typedef int3 NodeEdgesType;
	typedef int2 FacetsType;
	typedef int3 CellVerticesType;
	typedef float2 PointCoordsType;
	typedef float3 BaryCoordsType;
	
	typedef int3 NeighborsType;
	typedef int SimilarityType;

	typedef ImgType TexDataType;

	~CDelaunay_2D_Cuda_XGraph_Adaptor();
	void UnifiedMalloc();
	void Synchronize();
	void SetTexture(ImgType *imgdata, int width, int height);
	void FreeTexture();

	inline int* updatepatterns()		{ return m_updatepatterns; }
	inline int* labels()				{ return m_labels; }
	inline int* sortedcells()			{ return m_sortedcells; }
	inline float* sizes()				{ return m_sizes; }
	inline float* colorpatterns()		{ return m_colorpatterns; }
	inline SimilarityType* similarities()		{ return m_similarities; }
	inline CellVerticesType* cellvertices()			{ return m_cellvertices; }
	inline PointCoordsType* pointcoords()	{ return m_pointcoords; }
	inline EdgeNodesType* edgenodes()	{ return m_edgenodes; }
	inline FacetsType* facetvertices()		{ return m_facetvertices; }
	inline NodeEdgesType* nodeedges()		{ return m_nodeedges; }
	inline NeighborsType* neighbors()		{ return m_neighbors; }

	inline int& num_cells()					{ return v; }
	inline int& num_edges()					{ return e; }
	inline int& num_points()				{ return dv; }
	inline int& updatepatterns(int i)		{ return m_updatepatterns[i]; }
	inline int& labels(int i)				{ return m_labels[i]; }
	inline int& sortedcells(int i)			{ return m_sortedcells[i]; }
	inline float& sizes(int i)				{ return m_sizes[i]; }
	inline float& colorpatterns(int i)		{ return m_colorpatterns[i]; }
	inline SimilarityType& similarities(int i)		{ return m_similarities[i]; }
	inline CellVerticesType& cellvertices(int i)		{ return m_cellvertices[i]; }
	inline PointCoordsType& pointcoords(int i)	{ return m_pointcoords[i]; }
	inline EdgeNodesType& edgenodes(int i)	{ return m_edgenodes[i]; }
	inline FacetsType& facetvertices(int i)		{ return m_facetvertices[i]; }
	inline NodeEdgesType& nodeedges(int i)		{ return m_nodeedges[i]; }
	inline NeighborsType& neighbors(int i)		{ return m_neighbors[i]; }

	inline int3& image_dimensions()			{ return m_image_dimensions; }
	inline int& image_vol()					{ return m_vol; }
	inline cudaTextureObject_t& texObject()	{ return m_texObj; }

	static const int Dim, NCellVertices, NCellNeighbors, NFacetVertices, NGraphColors;

private:
	int v, e, dv;
	int *m_updatepatterns, *m_labels, *m_sortedcells;
	float *m_colorpatterns, *m_sizes;
	CellVerticesType *m_cellvertices;
	PointCoordsType  *m_pointcoords;
	EdgeNodesType *m_edgenodes;
	FacetsType *m_facetvertices;
	NodeEdgesType *m_nodeedges;

	NeighborsType *m_neighbors;
	SimilarityType *m_similarities;

	int3 m_image_dimensions;
	int m_vol;
	cudaTextureObject_t m_texObj;
	TexDataType *m_texdata;

};


#endif //ISOSEGMENTATION_CUDA_DELAUNAY2D_H