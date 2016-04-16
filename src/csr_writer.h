#ifndef CSR_WRITER_H
#define CSR_WRITER_H

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

template<class G>
class CGraphWriter
{

public:

	CGraphWriter(G& Graph) : m_pGraph(&Graph) {}
	
	void save_file()
	{
        string filename;
        cout << "Enter Save filename: ";
        cin >> filename;
        ofstream os(filename,ofstream::out);
		
		m_pGraph->Synchronize();
		for (int i = 0; i< m_pGraph->num_vertices() + 1; ++i)		os << m_pGraph->xadj(i);
		for (int i = 0; i< m_pGraph->num_edges() * 2; ++i)			os << m_pGraph->adjncy(i);
		for (int i = 0; i< m_pGraph->num_vertices(); ++i)			os << m_pGraph->cellvertices(i).x << m_pGraph->cellvertices(i).y << m_pGraph->cellvertices(i).z;
		for (int i = 0; i< m_pGraph->num_delaunayvertices(); ++i)	os << m_pGraph->delaunayvertices(i).x << m_pGraph->delaunayvertices(i).y;
    }

private:
    G* m_pGraph;
};

#endif //CSR_WRITER_H