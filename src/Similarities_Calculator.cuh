#ifndef ISOSEGMENTATION_CUDA_SIMILARITIESCALCULATOR_H
#define ISOSEGMENTATION_CUDA_SIMILARITIESCALCULATOR_H

#include "thrust_headers.h"

#include "Delaunay2D.h"
#include "Delaunay3D.h"
#include "Image2D.h"
#include "Image3D.h"
#include "Cuda_XGraph.cuh"
#include "Isovalue_List.h"

/////////////////////////////////////////////////////////////
// Base Class Template

template<class X, class B = X::XSpace::BaseClass>
class CSimilaritiesCalculator {};

///////////////////////////////////////////////////////////////
// Partial Spec.1 : Delaunay contexts (Delaunay2D/ Delaunay3D)

template<class X>
class CSimilaritiesCalculator<X, CDelaunay>
{
public:
	typedef typename X::XSpace XSpace;
	typedef typename XSpace::CellVerticesType CellVerticesType;
	typedef typename XSpace::FacetsType FacetsType;
	typedef typename XSpace::PointCoordsType PointCoordsType;
	typedef typename XSpace::NeighborsType NeighborsType;
	typedef float IsovalueType;
	typedef CIsovalueList<IsovalueType> CIsovalueListType;

	CSimilaritiesCalculator(X& x);
	void compute_similarities();
	void reset_similarities();

	void compute_neighbors();
	
	int& num_isovalues() { return m_numisovalues; };
	IsovalueType* isovalues_data() { return m_isovaluelist->data(); };

	float& threshold() { return m_threshold; };

private:
	X& m_X;
	CIsovalueListType m_isovaluelist;
	int m_numisovalues;
	float m_threshold;
};

/////////////////////////////////////////////////////////////
// Partial Spec.2 : Image contexts (Image2D/ Image3D)

template<class X>
class CSimilaritiesCalculator<X, CImage>
{
public:
	typedef typename X::XSpace XSpace;
	typedef typename XSpace::PointCoordsType PointCoordsType;
	typedef float IsovalueType;
	typedef CIsovalueList<IsovalueType> CIsovalueListType;

	CSimilaritiesCalculator(X& x);
	void compute_similarities();
	void reset_similarities();

	int& num_isovalues() { return m_numisovalues; };
	IsovalueType* isovalues_data() { return m_isovaluelist->data(); };

	float& threshold() { return m_threshold; };

private:
	X& m_X;
	CIsovalueListType m_isovaluelist;
	int m_numisovalues;
	float m_threshold;
	thrust::device_vector<int> neighbor_steps;
	thrust::device_vector<int3> neighbor_stepcoords;
};

/////////////////////////////////////////////////////////////	

#endif // ISOSEGMENTATION_CUDA_SIMILARITIESCALCULATOR_H