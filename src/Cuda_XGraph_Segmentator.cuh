#ifndef CUDA_XGRAPH_SEGMENTATOR_H
#define CUDA_XGRAPH_SEGMENTATOR_H

#include "thrust_headers.h"

#include "Delaunay2D.h"
#include "Delaunay3D.h"
#include "Image2D.h"
#include "Image3D.h"
#include "Cuda_XGraph.cuh"
#include "Graph_Colorizer.cuh"
#include "Similarities_Calculator.cuh"

/////////////////////////////////////////////////////////////

template<class X>
class CCudaXGraphSegmentator
{
public:
	typedef typename X::XSpace XSpace;
	typedef typename XSpace::BaseClass BaseClass;
	typedef typename XSpace::CellVerticesType CellVerticesType;
	typedef typename XSpace::PointCoordsType PointCoordsType;
	typedef typename XSpace::BaryCoordsType BaryCoordsType;
	typedef float IsovalueType;
	typedef CGraphColorizer<X, BaseClass> CGraphColorizerType;
	typedef CSimilaritiesCalculator<X, BaseClass> CSimilaritiesCalculatorType;

	CCudaXGraphSegmentator(X& x);

	void segmentate();
	void segmentate2();
	void segmentate3();

private:
	int nextpowerof2(int v);
	
	X& m_X;
	float threshold;
	dim3 BlocksInGrid, ThreadsPerBlock;
	int v, e, dv, num_isovalues, num_colors, num_regions, num_repeat_seg1, num_repeat_seg2, nR_2, nR_3;
	CGraphColorizerType m_graphcolorizer;
	CSimilaritiesCalculatorType m_similaritiescalculator;

	thrust::device_vector<int> region_labels;
	thrust::device_vector<int> region_numcells;
	thrust::device_vector<float> region_sizes;
	thrust::device_vector<float> region_colorpatterns;
	thrust::device_vector<int> region_bounds;
	thrust::device_vector<int> region_boundlabels;
	thrust::device_vector<int> sorted_cells;
};

/////////////////////////////////////////////////////////////

#endif //CUDA_XGRAPH_SEGMENTATOR_H