#include <cstdio>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "cudagraph.cuh"
#include "thrust_headers.h"

///////////////////////////////////////////////////////

template<class XSpace>
void CCudaCSRGraph<XSpace>::UnifiedMalloc()
{
    cudaMallocManaged(&m_xadj,				(v+1)*sizeof(int)		);
	cudaMallocManaged(&m_adjncy,			(e * 2)*sizeof(int)		);
	cudaMallocManaged(&m_updatepatterns,	(v)*sizeof(int)			);
	cudaMallocManaged(&m_labels,			(v)*sizeof(int)			);
	cudaMallocManaged(&m_colorpatterns,		(v)*sizeof(float)		);
	cudaMallocManaged(&m_similarities,		(e * 2)*sizeof(float)	);
	cudaMallocManaged(&m_cellvertices,		(v)*sizeof(int3)		);
	cudaMallocManaged(&m_delaunayvertices,	(dv)*sizeof(float2)		);
}

template<class XSpace>
CCudaCSRGraph<XSpace>::~CCudaCSRGraph()
{
	cudaFree(m_xadj);
	cudaFree(m_adjncy);
	cudaFree(m_updatepatterns);
	cudaFree(m_labels);
	cudaFree(m_colorpatterns);
	cudaFree(m_similarities);
	cudaFree(m_cellvertices);
	cudaFree(m_delaunayvertices);
}

template<class XSpace>
inline void CCudaCSRGraph<XSpace>::Synchronize()
{
	cudaDeviceSynchronize();
}

template<class XSpace>
void CCudaCSRGraph<XSpace>::Print()
{
	Synchronize();
	for (int i = 0; i<v + 1; ++i) printf("%i ", m_xadj[i]); printf("\n");
	for (int i = 0; i<e*2  ; ++i) printf("%i ", m_adjncy[i]); printf("\n");
	for (int i = 0; i < v; ++i)   printf("%i %i %i ", m_cellvertices[i].x, m_cellvertices[i].y, m_cellvertices[i].z); printf("\n");
	for (int i = 0; i < dv; ++i)  printf("%.2f %.2f ", m_delaunayvertices[i].x, m_delaunayvertices[i].y); printf("\n");
}

template<class XSpace>
void CCudaCSRGraph<XSpace>::Print2()
{
    Synchronize();
    //std::ofstream os("out.txt", std::ofstream::out);
//    for (int i = 0; i < v; ++i)   printf("%.2f\n", m_colorpatterns[i]); printf("\n");
    //for (int i = 0; i < v; ++i) os << m_colorpatterns[i] << "\n";
    for (int i = 0; i < v; ++i) std::cout << m_colorpatterns[i] << std::endl;
}


//////////////////////////////////////////////

template class CCudaCSRGraph<int>;

//////////////////////////////////////////////
