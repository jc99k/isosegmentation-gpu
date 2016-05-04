#ifndef XREADER_H
#define XREADER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "CImg.h"

using namespace std;
using namespace cimg_library;

///////////////////////////////////////////////////////

template<class X>
class CXReader
{
	typedef typename X::XSpace::ImgType ImgType;
public:
	CXReader(X& x) : m_X(x) {}
	void load_file();

private:
	X& m_X;
};

///////////////////////////////////////////////////////

template<>
void CXReader< CCudaXGraph< CDelaunay_2D_Cuda_XGraph_Adaptor > >::load_file()
{
	
	cout << "Enter CSR Graph filename: ";
	cin >> m_X->filename();
	ifstream is(m_X->filename(), ifstream::in);

	is >> m_X->num_vertices() >> m_X->num_edges() >> m_X->num_delaunayvertices();
	m_X->UnifiedMalloc();
	for (int i = 0; i< m_X->num_vertices() + 1; ++i)		is >> m_X->xadj(i);
	for (int i = 0; i< m_X->num_edges() * 2; ++i)			is >> m_X->adjncy(i);
	for (int i = 0; i< m_X->num_vertices(); ++i)			is >> m_X->cellvertices(i).x >> m_X->cellvertices(i).y >> m_X->cellvertices(i).z;
	for (int i = 0; i< m_X->num_delaunayvertices(); ++i)	is >> m_X->delaunayvertices(i).x >> m_X->delaunayvertices(i).y;

	string img_filename;
	cout << "Enter Image filename: ";
	cin >> img_filename;
	CImg<ImgType> image(img_filename.c_str());
	m_X->SetTexture(image.data(), image.width(), image.height());
}

///////////////////////////////////////////////////////

template<>
void CXReader< CCudaXGraph< CDelaunay_3D_Cuda_XGraph_Adaptor > >::load_file()
{

	cout << "Enter CSR Graph filename: ";
	cin >> m_X->filename();
	ifstream is(m_X->filename(), ifstream::in);

	is >> m_X->num_vertices() >> m_X->num_edges() >> m_X->num_delaunayvertices();
	m_X->UnifiedMalloc();
	for (int i = 0; i< m_X->num_vertices() + 1; ++i)		is >> m_X->xadj(i);
	for (int i = 0; i< m_X->num_edges() * 2; ++i)			is >> m_X->adjncy(i);
	for (int i = 0; i< m_X->num_vertices(); ++i)			is >> m_X->cellvertices(i).x >> m_X->cellvertices(i).y >> m_X->cellvertices(i).z >> m_X->cellvertices(i).w;
	for (int i = 0; i< m_X->num_delaunayvertices(); ++i)	is >> m_X->delaunayvertices(i).x >> m_X->delaunayvertices(i).y >> m_X->delaunayvertices(i).z;
	
	string img_filename;
	int width, height, depth;
	cout << "Enter Image filename and dimensions: ";
	cin >> img_filename >> width >> height >> depth;

	ifstream input(img_filename.c_str(), ios::binary);
	vector<unsigned char> buffer((istreambuf_iterator<char>(input)), istreambuf_iterator<char>());
	CImg<ImgType> image(buffer.data(), width, height, depth, 1, false);

	//image.display();
	m_X->SetTexture(image.data(), image.width(), image.height(), image.depth());

	//int index;
	//while (1) {
	//	cout << "Cell to evaluate: ";
	//	cin >> index;
	//	if (index == -1) break;
	//	
	//	cout << "Color pattern : " << (int)                                                                                                        m_X->colorpatterns(index) << endl;
	//	for (int j = m_X->xadj(index); j < m_X->xadj(index + 1); ++j)
	//		cout << m_X->adjncy(j) << " (" << m_X->similarities(j) << ")\n" ;
	//}
}

///////////////////////////////////////////////////////

#endif //XREADER_H