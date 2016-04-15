#ifndef CSR_READER_H
#define CSR_READER_H

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

template<class G>
class CGraphReader
{

public:

	CGraphReader(G& Graph) : m_pGraph(&Graph) {}
	
	void load_file()
	{
        string filename;
        cout << "Enter CSR Graph filename: ";
        cin >> filename;
        ifstream is(filename,ifstream::in);
		
		is >> m_pGraph->num_vertices() >> m_pGraph->num_edges() >> m_pGraph->num_delaunayvertices();
		m_pGraph->UnifiedMalloc();
		for (int i = 0; i< m_pGraph->num_vertices() + 1; ++i)		is >> m_pGraph->xadj(i);
		for (int i = 0; i< m_pGraph->num_edges() * 2; ++i)			is >> m_pGraph->adjncy(i);
		for (int i = 0; i< m_pGraph->num_vertices(); ++i)			is >> m_pGraph->cellvertices(i).x >> m_pGraph->cellvertices(i).y >> m_pGraph->cellvertices(i).z;
		for (int i = 0; i< m_pGraph->num_delaunayvertices(); ++i)	is >> m_pGraph->delaunayvertices(i).x >> m_pGraph->delaunayvertices(i).y;
    }

private:
    G* m_pGraph;
};

#endif //CSR_READER_H
