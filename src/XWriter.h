#ifndef XWRITER_H
#define XWRITER_H

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

///////////////////////////////////////////////////////

template<class X>
class CXWriter
{
public:
	CXWriter(X& x) : m_X(x) {}
	void save_file();

private:
	X& m_X;

};

///////////////////////////////////////////////////////

template<>
void CXWriter< CCudaXGraph< CDelaunay_2D_Cuda_XGraph_Adaptor > >::save_file()
{
	cout << "Synchronizing..." << endl;
	m_X->Synchronize();
	cout << "Synchronized." << endl;
	
	cout << "Opening output file..." << endl;
	string out_filename = m_X->filename();
	out_filename.replace(out_filename.end() - 3, out_filename.end(), "out");
	ofstream os(out_filename, ofstream::out);
	
	for (int i = 0; i< m_X->num_delaunayvertices(); ++i)	os << m_X->delaunayvertices(i).x << ' ' << m_X->delaunayvertices(i).y << ' '; os << endl;
	for (int i = 0; i< m_X->num_vertices(); ++i)			os << m_X->cellvertices(i).x << ' ' << m_X->cellvertices(i).y << ' ' << m_X->cellvertices(i).z << ' '; os << endl;
	for (int i = 0; i< m_X->num_edges(); ++i)				os << m_X->edgenodes(i).x << ' ' << m_X->edgenodes(i).y << ' '; os << endl;
	for (int i = 0; i< m_X->num_vertices(); ++i)			os << m_X->colorpatterns(i)/255.0f << ' '; os << endl;
	for (int i = 0; i< m_X->num_edges(); ++i)				os << m_X->similarities(i) << ' '; os << endl;
	for (int i = 0; i< m_X->num_vertices(); ++i)			os << m_X->updatepatterns(i) << ' '; os << endl;
	for (int i = 0; i< m_X->num_vertices(); ++i)			os << m_X->labels(i) << ' '; os << endl;

	//ifstream  src(m_X->filename(), std::ios::binary);
	//ofstream  dst(out_filename, std::ios::binary);

	//dst << src.rdbuf();
}

///////////////////////////////////////////////////////
//
//template<>
//void CXWriter< CCudaXGraph< CDelaunay_3D_Cuda_XGraph_Adaptor > >::save_file()
//{
//	cout << "Synchronizing..." << endl;
//	m_X->Synchronize();
//	cout << "Synchronized." << endl;
//
//	cout << "Opening output file..." << endl;
//	string out_filename = m_X->filename();
//	out_filename.replace(out_filename.end() - 3, out_filename.end(), "out");
//	ofstream os(out_filename, ofstream::out);
//
//	cout << "Writing..." << endl;
//
//	//for (int i = 0; i < m_X->num_vertices() + 1; ++i)		os << m_X->xadj(i) << ' '; os << endl;
//	//for (int i = 0; i< m_X->num_edges() * 2; ++i)			os << m_X->adjncy(i) << ' '; os << endl;
//	//for (int i = 0; i< m_X->num_vertices(); ++i)			os << m_X->cellvertices(i).x << ' ' << m_X->cellvertices(i).y << ' ' << m_X->cellvertices(i).z << ' ' << m_X->cellvertices(i).w << ' '; os << endl;
//	//for (int i = 0; i< m_X->num_delaunayvertices(); ++i)	os << m_X->delaunayvertices(i).x << ' ' << m_X->delaunayvertices(i).y << ' ' << m_X->delaunayvertices(i).z << ' '; os << endl;
//	for (int i = 0; i< m_X->num_vertices(); ++i)			os << m_X->colorpatterns(i) / 255.0f << ' '; os << endl;
//	//for (int i = 0; i< m_X->num_vertices(); ++i)			os << m_X->sizes(i) << ' '; os << endl;
//	//for (int i = 0; i< m_X->num_vertices(); ++i)			os << m_X->updatepatterns(i) << ' '; os << endl;
//	//for (int i = 0; i< m_X->num_vertices(); ++i)			os << i << ' '; os << endl;
//	for (int i = 0; i< m_X->num_edges(); ++i)			os << m_X->similarities(i) << ' '; os << endl;
//	for (int i = 0; i< m_X->num_vertices(); ++i)			os << m_X->labels(i) << ' '; os << endl;
//
//	cout << "Done." << endl;
//}
//
/////////////////////////////////////////////////////////

template<>
void CXWriter< CCudaXGraph< CImage_2D_Cuda_XGraph_Adaptor > >::save_file()
{
	cout << "Synchronizing..." << endl;
	m_X->Synchronize();
	cout << "Synchronized." << endl;

	cout << "Opening output file..." << endl;
	string out_filename = m_X->filename();
	out_filename.replace(out_filename.end() - 3, out_filename.end(), "out");
	ofstream os(out_filename, ofstream::out);

	for (int i = 0; i< m_X->num_vertices(); ++i)			os << m_X->colorpatterns(i) / 255.0f << ' '; os << endl;
	for (int i = 0; i< m_X->num_edges(); ++i)				os << m_X->similarities(i) << ' '; os << endl;
	for (int i = 0; i< m_X->num_vertices(); ++i)			os << m_X->labels(i) << ' '; os << endl;

	cout << "Done." << endl;
}

///////////////////////////////////////////////////////

#endif //XWRITER_H