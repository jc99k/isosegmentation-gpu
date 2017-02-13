#ifndef ISOSEGMENTATION_CUDA_GRAPHCOLORIZER_H
#define ISOSEGMENTATION_CUDA_GRAPHCOLORIZER_H

#include "Delaunay2D.h"
#include "Delaunay3D.h"
#include "Image2D.h"
#include "Image3D.h"
#include "Cuda_XGraph.cuh"

/////////////////////////////////////////////////////////////
// Base Class Template

template<class X, class B = X::XSpace::BaseClass>
class CGraphColorizer {};

///////////////////////////////////////////////////////////////
// Partial Spec.1 : Delaunay contexts (Delaunay2D/ Delaunay3D)

template<class X>
class CGraphColorizer<X, CDelaunay>
{
public:
	typedef typename X::XSpace XSpace;
	typedef typename XSpace::CellVerticesType CellVerticesType;
	typedef typename XSpace::NodeEdgesType NodeEdgesType;
	typedef typename XSpace::EdgeNodesType EdgeNodesType;

	CGraphColorizer(X& x);
	void colorize();

private:
	X& m_X;
	int num_colors;
};

/////////////////////////////////////////////////////////////
// Partial Spec.2 : Image contexts (Image2D/ Image3D)

template<class X>
class CGraphColorizer<X, CImage>
{
public:
	typedef typename X::XSpace XSpace;
	typedef typename XSpace::PointCoordsType PointCoordsType;

	CGraphColorizer(X& x);
	void colorize();

private:
	X& m_X;
	int num_colors;
};

/////////////////////////////////////////////////////////////	

#endif // ISOSEGMENTATION_CUDA_GRAPHCOLORIZER_H