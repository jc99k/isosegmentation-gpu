#ifndef CUDA_XGRAPH_SEGMENTATOR_H
#define CUDA_XGRAPH_SEGMENTATOR_H

#include "Isovalue_List.h"

template<class X>
class CCudaXGraphSegmentator
{
	typedef typename X::XSpace XSpace;
	typedef typename XSpace::CellVerticesType CellVerticesType;
	typedef typename XSpace::DelaunayVerticesType DelaunayVerticesType;
	typedef float IsovalueType;
	typedef CIsovalueList<IsovalueType> CIsovalueList_t;

public:
	CCudaXGraphSegmentator(X& x);

	void colorize();
	void compute_colorpatterns_and_similarities();
    //void compute_similarities();
	void segmentate();
	void segmentate2();

private:
	X& m_X;
	//int repeat;
	float threshold;
	dim3 BlocksInGrid, ThreadsPerBlock;
	int v, e, dv, num_isovalues, num_colors, num_regions;
	CIsovalueList_t	 m_isovaluelist;
};



#endif //CUDA_XGRAPH_SEGMENTATOR_H