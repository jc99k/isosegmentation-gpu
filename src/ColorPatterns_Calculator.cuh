#ifndef ISOSEGMENTATION_CUDA_COLORPATTERNCALCULATOR_H
#define ISOSEGMENTATION_CUDA_COLORPATTERNCALCULATOR_H

#include "Delaunay2D.h"
#include "Delaunay3D.h"
#include "Image2D.h"
#include "Image3D.h"
#include "Cuda_XGraph.cuh"

/////////////////////////////////////////////////////////////
// Base Class Template

template<class X, class B = typename X::XSpace::BaseClass>
class CColorPatternsCalculator {};

///////////////////////////////////////////////////////////////
// Partial Spec.1 : Delaunay contexts (Delaunay2D/ Delaunay3D)

template<class X>
class CColorPatternsCalculator<X, CDelaunay>
{
public:
	typedef typename X::XSpace XSpace;
	typedef typename XSpace::CellVerticesType CellVerticesType;
	typedef typename XSpace::PointCoordsType PointCoordsType;
	typedef typename XSpace::BaryCoordsType BaryCoordsType;

	CColorPatternsCalculator(X& x);
	void compute_colorpatterns();

private:
	X& m_X;
};

/////////////////////////////////////////////////////////////
// Partial Spec.2 : Image contexts (Image2D/ Image3D)

template<class X>
class CColorPatternsCalculator<X, CImage>
{
public:
	typedef typename X::XSpace XSpace;
	typedef typename XSpace::PointCoordsType PointCoordsType;

	CColorPatternsCalculator(X& x);
	void compute_colorpatterns();

private:
	X& m_X;
};

/////////////////////////////////////////////////////////////

#endif // ISOSEGMENTATION_CUDA_COLORPATTERNCALCULATOR_H
