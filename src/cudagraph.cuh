#ifndef CUDA_CSR_GRAPH_H
#define CUDA_CSR_GRAPH_H

#include "class.h"
#include "vector_types.h"

template<class XSpace>
class CCudaCSRGraph
{
public:
	~CCudaCSRGraph();
	void UnifiedMalloc();
	void Synchronize();
	void Print();

	inline int* xadj()					{ return m_xadj; }
	inline int* adjncy()				{ return m_adjncy; }
	inline int* updatepatterns()		{ return m_updatepatterns; }
	inline int* labels()				{ return m_labels; }
	inline float* colorpatterns()		{ return m_colorpatterns; }
	inline float* similarities()		{ return m_similarities; }
	inline int3* cellvertices()			{ return m_cellvertices; }
	inline float2* delaunayvertices()	{ return m_delaunayvertices; }
	
	inline int& num_vertices()				{ return v;						}
	inline int& num_edges()					{ return e; }
	inline int& num_delaunayvertices()		{ return dv; }
	inline int& xadj(int i)					{ return m_xadj[i];				}
	inline int& adjncy(int i)				{ return m_adjncy[i];			}
	inline int& updatepatterns(int i)		{ return m_updatepatterns[i];	}
	inline int& labels(int i)				{ return m_labels[i];			}
	inline float& colorpatterns(int i)		{ return m_colorpatterns[i];	}
	inline float& similarities(int i)		{ return m_similarities[i];		}
	inline int3& cellvertices(int i)		{ return m_cellvertices[i];		}
	inline float2& delaunayvertices(int i)	{ return m_delaunayvertices[i]; }

private:
    int v,e,dv,num_updatepatterns, num_regions;
    int *m_xadj, *m_adjncy, *m_updatepatterns, *m_labels;
	float *m_colorpatterns, *m_similarities;
	int3 *m_cellvertices;
	float2  *m_delaunayvertices;
};

#endif //CUDA_CSR_GRAPH_H