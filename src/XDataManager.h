#ifndef XREADER_H
#define XREADER_H

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include "CImg.h"

using namespace std;
using namespace cimg_library;

///////////////////////////////////////////////////////

template<class X>
class CXDataManager
{
	typedef typename X::XSpace::ImgType ImgType;
public:
	CXDataManager(X& x) : m_X(x) {}
	void load_file();
	void save_file();

	void load_file2();
	void save_file2();

private:
	X& m_X;
	string input_mesh, input_dual, input_image, output_mesh, output_dual, output_image;
	int points_begin, points_end, cells_begin, cells_end, facets_begin, facets_end, nodeedges_begin, nodeedges_end, edgenodes_begin, edgenodes_end;
	int num_points, num_cells, num_edges;
	int nbits_cells, nbits_edges, cell_field_data, edge_field_data;

	vector<unsigned char> mesh_buffer;
	vector<unsigned char> raw_buffer;

	string s, sub;
	istringstream iss;
	ifstream is;
};

///////////////////////////////////////////////////////
// Spec.1 : Delaunay2D

template<>
void CXDataManager< CCudaXGraph< CDelaunay_2D_Cuda_XGraph_Adaptor > >::load_file2()
{
	///* INPUT FILENAMES */
	cout << "Enter Delaunay Mesh (Binary .vtk file): \n";
	cin >> input_mesh;
	cout << endl;

	cout << "Mesh File : " << input_mesh << '\n';

	// Read whole files into memory
	is.open(input_mesh, ios::in | ios::binary);
	is.seekg(0, ios::end);
	mesh_buffer.resize(is.tellg());
	is.seekg(0, ios::beg);
	is.read((char*)&mesh_buffer[0], mesh_buffer.size());
	is.close();

	// Read Parameters from Mesh File
	int p = 78;
	points_begin = p, num_points = 0;
	do{
		num_points = num_points * 10 + (mesh_buffer[p++] - '0');
	} while (mesh_buffer[p] != ' ');
	m_X->num_points() = num_points;

	points_begin = p + 7;
	points_end = points_begin + 4 * 3 * num_points; // 4 = sizeof(float), 3 = x,y,z

	p = points_end + 7;
	cells_begin = p, num_cells = 0;
	do{
		num_cells = num_cells * 10 + (mesh_buffer[p++] - '0');
	} while (mesh_buffer[p] != ' ');
	m_X->num_cells() = num_cells;

	nbits_cells = p - cells_begin;
	facets_begin = cells_begin;

	do {} while (mesh_buffer[p++] != '\n');
	cells_begin = p;
	cells_end = cells_begin + 4 * 4 * num_cells; // 4 = sizeof(int), 4 = 3,a,b,c

	cout << "Points : " << m_X->num_points() << '\n';
	cout << "Cells : " << m_X->num_cells() << '\n';

	///* ALLOCATE UNIFIED MEMORY (CUDA) */
	m_X->UnifiedMalloc();

	/* LOAD MESH FILE */
	cout << "Reading : " << input_mesh << '\n';
	char temp[4];
	// READ MESH POINTS
	float *f;
	f = (float*)temp;
	for (int i = points_begin, j = 0; i < points_end; i += 4 * 3, j++) {
		//cout << j << " : ";
		// Swap order because of Endianness
		// X
		temp[0] = mesh_buffer[i + 3];
		temp[1] = mesh_buffer[i + 2];
		temp[2] = mesh_buffer[i + 1];
		temp[3] = mesh_buffer[i];
		//cout << *f << ' ';
		m_X->pointcoords(j).x = *f;

		// Y
		temp[0] = mesh_buffer[i + 7];
		temp[1] = mesh_buffer[i + 6];
		temp[2] = mesh_buffer[i + 5];
		temp[3] = mesh_buffer[i + 4];
		//cout << *f << ' ';
		m_X->pointcoords(j).y = *f;

		//cout << endl;
	}

	//READ VERTEX INDICES FOR EACH CELL
	int *d;
	d = (int*)temp;
	set<int> sorted_vertices;
	int k;
	for (int i = cells_begin, j = 0; i < cells_end; i += 4 * 4, j++) {
		sorted_vertices.clear();
		//cout << j << " : ";
		// Swap order because of Endianness
		// v[0]
		temp[0] = mesh_buffer[i + 7];
		temp[1] = mesh_buffer[i + 6];
		temp[2] = mesh_buffer[i + 5];
		temp[3] = mesh_buffer[i + 4];
		//cout << *d << ' ';
		//m_X->cellvertices(j).x = *d;
		sorted_vertices.insert(*d);

		// v[1]
		temp[0] = mesh_buffer[i + 11];
		temp[1] = mesh_buffer[i + 10];
		temp[2] = mesh_buffer[i + 9];
		temp[3] = mesh_buffer[i + 8];
		//cout << *d << ' ';
		//m_X->cellvertices(j).y = *d;
		sorted_vertices.insert(*d);

		// v[2]
		temp[0] = mesh_buffer[i + 15];
		temp[1] = mesh_buffer[i + 14];
		temp[2] = mesh_buffer[i + 13];
		temp[3] = mesh_buffer[i + 12];
		//cout << *d << ' ';
		//m_X->cellvertices(j).z = *d;
		sorted_vertices.insert(*d);

		k = 0;
		for (auto it = sorted_vertices.begin(); it != sorted_vertices.end(); ++it, ++k){
			*((int*)&m_X->cellvertices(j) + k) = *it;
		}
		//cout << j << ": " << m_X->cellvertices(j).x << ' ' << m_X->cellvertices(j).y << ' ' << m_X->cellvertices(j).z << endl;

		//cout << endl;
	}

}

template<>
void CXDataManager< CCudaXGraph< CDelaunay_2D_Cuda_XGraph_Adaptor > >::load_file()
{
	//int dummy_int;
	//string s, sub, dummy_str;
	//istringstream iss;
	//ifstream is;

	///* INPUT FILENAMES */
	cout << "Enter Delaunay Mesh (Binary .vtk file): \n";
	cin >> input_mesh;
	cout << "Enter Image filename (.png file): \n";
	cin >> input_image;
	cout << endl;

	cout << "Mesh File : " << input_mesh << '\n';
	cout << "Image File: " << input_image << '\n';

	// Read whole files into memory
	is.open(input_mesh, ios::in | ios::binary);
	is.seekg(0, ios::end);
	mesh_buffer.resize(is.tellg());
	is.seekg(0, ios::beg);
	is.read((char*)&mesh_buffer[0], mesh_buffer.size());
	is.close();

	// Read Parameters from Mesh File
	int p = 78;
	points_begin = p, num_points = 0;
	do{
		num_points = num_points * 10 + (mesh_buffer[p++] - '0');
	} while (mesh_buffer[p] != ' ');
	m_X->num_points() = num_points;

	points_begin = p + 7;
	points_end = points_begin + 4 * 3 * num_points; // 4 = sizeof(float), 3 = x,y,z

	p = points_end + 7;
	cells_begin = p, num_cells = 0;
	do{
		num_cells = num_cells * 10 + (mesh_buffer[p++] - '0');
	} while (mesh_buffer[p] != ' ');
	m_X->num_cells() = num_cells;

	nbits_cells = p - cells_begin;
	facets_begin = cells_begin;

	do {} while (mesh_buffer[p++] != '\n');
	cells_begin = p;
	cells_end = cells_begin + 4 * 4 * num_cells; // 4 = sizeof(int), 4 = 3,a,b,c

	cout << "Points : " << m_X->num_points() << '\n';
	cout << "Cells : " << m_X->num_cells() << '\n';
	//cout << "Edges : " << m_X->num_edges() << "(Ascii: " << nbits_edges << " bits)\n";

	///* ALLOCATE UNIFIED MEMORY (CUDA) */
	m_X->UnifiedMalloc();

	/* LOAD MESH FILE */
	cout << "Reading : " << input_mesh << '\n';
	char temp[4];
	// READ MESH POINTS
	float *f;
	f = (float*)temp;
	for (int i = points_begin, j = 0; i < points_end; i += 4 * 3, j++) {
		//cout << j << " : ";
		// Swap order because of Endianness
		// X
		temp[0] = mesh_buffer[i + 3];
		temp[1] = mesh_buffer[i + 2];
		temp[2] = mesh_buffer[i + 1];
		temp[3] = mesh_buffer[i];
		//cout << *f << ' ';
		m_X->pointcoords(j).x = *f;

		// Y
		temp[0] = mesh_buffer[i + 7];
		temp[1] = mesh_buffer[i + 6];
		temp[2] = mesh_buffer[i + 5];
		temp[3] = mesh_buffer[i + 4];
		//cout << *f << ' ';
		m_X->pointcoords(j).y = *f;

		//cout << endl;
	}

	//READ VERTEX INDICES FOR EACH CELL
	int *d;
	d = (int*)temp;
	for (int i = cells_begin, j = 0; i < cells_end; i += 4 * 4, j++) {
		//cout << j << " : ";
		// Swap order because of Endianness
		// v[0]
		temp[0] = mesh_buffer[i + 7];
		temp[1] = mesh_buffer[i + 6];
		temp[2] = mesh_buffer[i + 5];
		temp[3] = mesh_buffer[i + 4];
		//cout << *d << ' ';
		m_X->cellvertices(j).x = *d;

		// v[1]
		temp[0] = mesh_buffer[i + 11];
		temp[1] = mesh_buffer[i + 10];
		temp[2] = mesh_buffer[i + 9];
		temp[3] = mesh_buffer[i + 8];
		//cout << *d << ' ';
		m_X->cellvertices(j).y = *d;

		// v[2]
		temp[0] = mesh_buffer[i + 15];
		temp[1] = mesh_buffer[i + 14];
		temp[2] = mesh_buffer[i + 13];
		temp[3] = mesh_buffer[i + 12];
		//cout << *d << ' ';
		m_X->cellvertices(j).z = *d;

		//cout << endl;
	}

	//READ NEIGHBOR INDICES FOR EACH CELL

	// Skip up to FieldData number
	p = cells_end + 1;
	do {} while (mesh_buffer[p++] != '\n');
	p += 4 * num_cells + 1;
	p += 9 + nbits_cells + 18;

	cell_field_data = p;

	// Skip up to start of neighbors
	p += 2;
	do {} while (mesh_buffer[p++] != '\n');

	//cout << " p = " << p << endl;
	//cout << buffer[p] << endl;

	for (int i = p, j = 0; j < num_cells; i += 4 * 3, j++) {
		//cout << j << " : ";
		// Swap order because of Endianness
		// edge[0]
		temp[0] = mesh_buffer[i + 3];
		temp[1] = mesh_buffer[i + 2];
		temp[2] = mesh_buffer[i + 1];
		temp[3] = mesh_buffer[i];
		//cout << *d << ' ';
		m_X->neighbors(j).x = *d;

		// edge[1]
		temp[0] = mesh_buffer[i + 7];
		temp[1] = mesh_buffer[i + 6];
		temp[2] = mesh_buffer[i + 5];
		temp[3] = mesh_buffer[i + 4];
		//cout << *d << ' ';
		m_X->neighbors(j).y = *d;

		// edge[2]
		temp[0] = mesh_buffer[i + 11];
		temp[1] = mesh_buffer[i + 10];
		temp[2] = mesh_buffer[i + 9];
		temp[3] = mesh_buffer[i + 8];
		//cout << *d << ' ';
		m_X->neighbors(j).z = *d;

		//cout << endl;
	}

	///////////////////////////////////////////////////////////////////////////

	/* LOAD IMAGE FILE */
	cout << "Reading : " << input_image << '\n';
	CImg<ImgType> image(input_image.c_str());
	//cout << "\nFloat image created!" << endl;
	m_X->SetTexture(image.data(), image.width(), image.height());
}

template<>
void CXDataManager< CCudaXGraph< CDelaunay_2D_Cuda_XGraph_Adaptor > >::save_file()
{
	cout << "Synchronizing... ";
	m_X->Synchronize();
	cout << "Synchronized!" << endl;

	cout << "Enter Output Mesh (.vtk file): \n";
	cin >> output_mesh;

	cout << "Writing : " << output_mesh << '\n';

	// ASCII
	//ofstream os(output_mesh, ofstream::out);
	//for (int i = 0; i< m_X->num_delaunayvertices(); ++i)	os << m_X->delaunayvertices(i).x << ' ' << m_X->delaunayvertices(i).y << ' '; os << endl;
	//for (int i = 0; i< m_X->num_vertices(); ++i)			os << m_X->cellvertices(i).x << ' ' << m_X->cellvertices(i).y << ' ' << m_X->cellvertices(i).z << ' '; os << endl;
	//for (int i = 0; i< m_X->num_edges(); ++i)				os << m_X->edgenodes(i).x << ' ' << m_X->edgenodes(i).y << ' '; os << endl;
	//for (int i = 0; i< m_X->num_vertices(); ++i)			os << m_X->sortedcells(i) << ' '; os << endl;
	//for (int i = 0; i< m_X->num_vertices(); ++i)			os << m_X->colorpatterns(i) << ' '; os << endl;
	//for (int i = 0; i< m_X->num_edges(); ++i)				os << m_X->similarities(i) << ' '; os << endl;
	//for (int i = 0; i< m_X->num_vertices(); ++i)			os << m_X->labels(i) << ' '; os << endl;
	//os.close();

	// BINARY (Copy and modify Mesh buffer file)
	// Modify NUMBER OF FIELDS
	mesh_buffer[cell_field_data] = '3';

	// Colorpatterns

	// Define header, resize buffer, append header
	string s1 = "colorpatterns 1 " + to_string(m_X->num_cells()) + " float\n";
	int mbuf_size = mesh_buffer.size();
	mesh_buffer.resize(mbuf_size + s1.size() + m_X->num_cells()* 4);
	for (int i = 0; i < s1.size(); ++i) mesh_buffer[mbuf_size + i] = (int)(s1[i]);

	// Append field
	unsigned char *c;
	unsigned char temp[4];
	int j;
	for (int i = 0; i < m_X->num_cells(); ++i)
	{
		j = i * 4;
		c = (unsigned char*)&m_X->colorpatterns(i);
		mesh_buffer[mbuf_size + s1.size() + j]		= c[3];
		mesh_buffer[mbuf_size + s1.size() + j + 1]	= c[2];
		mesh_buffer[mbuf_size + s1.size() + j + 2]	= c[1];
		mesh_buffer[mbuf_size + s1.size() + j + 3]	= c[0];
	}

	// Segmentation
	// Define header, resize buffer, append header
	string s2 = "imesh_segmentation 1 " + to_string(m_X->num_cells()) + " int\n";
	mbuf_size = mesh_buffer.size();
	mesh_buffer.resize(mbuf_size + s2.size() + m_X->num_cells() * 4);
	for (int i = 0; i < s2.size(); ++i) mesh_buffer[mbuf_size + i] = (int)(s2[i]);

	// Append field
	for (int i = 0; i < m_X->num_cells(); ++i)
	{
		j = i * 4;
		c = (unsigned char*)&m_X->labels(i);
		mesh_buffer[mbuf_size + s2.size() + j] = c[3];
		mesh_buffer[mbuf_size + s2.size() + j + 1] = c[2];
		mesh_buffer[mbuf_size + s2.size() + j + 2] = c[1];
		mesh_buffer[mbuf_size + s2.size() + j + 3] = c[0];
	}

	ofstream os(output_mesh, ofstream::out | ofstream::binary);
	os.write((char*)mesh_buffer.data(), mesh_buffer.size()*sizeof(unsigned char));
	os.close();
}

///////////////////////////////////////////////////////
// Spec.2 : Delaunay3D

template<>
void CXDataManager< CCudaXGraph< CDelaunay_3D_Cuda_XGraph_Adaptor > >::load_file()
{
	///* INPUT FILENAMES */
	cout << "Enter Delaunay Mesh (Binary .vtk file): \n";
	cin >> input_mesh;
	cout << "Enter 3D Image filename (.dat file): \n";
	cin >> input_image;
	cout << endl;

	cout << "Mesh File : " << input_mesh << '\n';
	cout << "Image File: " << input_image << '\n';

	// Read the whole Mesh file into memory
	is.open(input_mesh, ios::in | ios::binary);
	is.seekg(0, ios::end);
	mesh_buffer.resize(is.tellg());
	is.seekg(0, ios::beg);
	is.read((char*)&mesh_buffer[0], mesh_buffer.size());
	is.close();

	// Read Parameters from Mesh File
	int p = 78;
	points_begin = p, num_points = 0;
	do{
		num_points = num_points * 10 + (mesh_buffer[p++] - '0');
	} while (mesh_buffer[p] != ' ');
	m_X->num_points() = num_points;

	points_begin = p + 7;
	points_end = points_begin + 4 * 3 * num_points; // 4 = sizeof(float), 3 = x,y,z

	p = points_end + 7;
	cells_begin = p, num_cells = 0;
	do{
		num_cells = num_cells * 10 + (mesh_buffer[p++] - '0');
	} while (mesh_buffer[p] != ' ');
	m_X->num_cells() = num_cells;

	nbits_cells = p - cells_begin;
	facets_begin = cells_begin;

	do {} while (mesh_buffer[p++] != '\n');
	cells_begin = p;
	cells_end = cells_begin + 4 * 5 * num_cells; // 4 = sizeof(int), 5 = 4,a,b,c,d

	cout << "Points : " << m_X->num_points() << '\n';
	cout << "Cells : " << m_X->num_cells() << '\n';
	//cout << "Edges : " << m_X->num_edges() << "(Ascii: " << nbits_edges << " bits)\n";

	///* ALLOCATE UNIFIED MEMORY (CUDA) */
	m_X->UnifiedMalloc();

	/* LOAD MESH FILE */
	cout << "Reading : " << input_mesh << '\n';
	char temp[4];
	// READ MESH POINTS
	float *f;
	f = (float*)temp;
	for (int i = points_begin, j = 0; i < points_end; i += 4 * 3, j++) {
		//cout << j << " : ";
		// Swap order because of Endianness
		// X
		temp[0] = mesh_buffer[i + 3];
		temp[1] = mesh_buffer[i + 2];
		temp[2] = mesh_buffer[i + 1];
		temp[3] = mesh_buffer[i];
		//cout << *f << ' ';
		m_X->pointcoords(j).x = *f;

		// Y
		temp[0] = mesh_buffer[i + 7];
		temp[1] = mesh_buffer[i + 6];
		temp[2] = mesh_buffer[i + 5];
		temp[3] = mesh_buffer[i + 4];
		//cout << *f << ' ';
		m_X->pointcoords(j).y = *f;

		// Z
		temp[0] = mesh_buffer[i + 11];
		temp[1] = mesh_buffer[i + 10];
		temp[2] = mesh_buffer[i + 9];
		temp[3] = mesh_buffer[i + 8];
		//cout << *f << ' ';
		m_X->pointcoords(j).z = *f;

		//cout << endl;
	}

	//READ VERTEX INDICES FOR EACH CELL
	int *d;
	d = (int*)temp;
	for (int i = cells_begin, j = 0; i < cells_end; i += 4 * 5, j++) {
		//cout << j << " : ";
		// Swap order because of Endianness
		// v[0]
		temp[0] = mesh_buffer[i + 7];
		temp[1] = mesh_buffer[i + 6];
		temp[2] = mesh_buffer[i + 5];
		temp[3] = mesh_buffer[i + 4];
		//cout << *d << ' ';
		m_X->cellvertices(j).x = *d;

		// v[1]
		temp[0] = mesh_buffer[i + 11];
		temp[1] = mesh_buffer[i + 10];
		temp[2] = mesh_buffer[i + 9];
		temp[3] = mesh_buffer[i + 8];
		//cout << *d << ' ';
		m_X->cellvertices(j).y = *d;

		// v[2]
		temp[0] = mesh_buffer[i + 15];
		temp[1] = mesh_buffer[i + 14];
		temp[2] = mesh_buffer[i + 13];
		temp[3] = mesh_buffer[i + 12];
		//cout << *d << ' ';
		m_X->cellvertices(j).z = *d;

		// v[3]
		temp[0] = mesh_buffer[i + 19];
		temp[1] = mesh_buffer[i + 18];
		temp[2] = mesh_buffer[i + 17];
		temp[3] = mesh_buffer[i + 16];
		//cout << *d << ' ';
		m_X->cellvertices(j).w = *d;

		//cout << endl;
	}

	//READ NEIGHBOR INDICES FOR EACH CELL

	// Skip up to FieldData number
	p = cells_end + 1;
	do {} while (mesh_buffer[p++] != '\n');
	p += 4 * num_cells + 1;
	p += 9 + nbits_cells + 18;

	cell_field_data = p;

	//cout << " p = " << p << endl;
	//cout << mesh_buffer[p] << endl;

	// Skip up to start of neighbors
	p += 2;
	do {} while (mesh_buffer[p++] != '\n');

	//cout << " p = " << p << endl;
	//cout << mesh_buffer[p] << endl;

	for (int i = p, j = 0; j < num_cells; i += 4 * 4, j++) {
		//cout << j << " : ";
		// Swap order because of Endianness
		// neighbor[0]
		temp[0] = mesh_buffer[i + 3];
		temp[1] = mesh_buffer[i + 2];
		temp[2] = mesh_buffer[i + 1];
		temp[3] = mesh_buffer[i];
		//cout << *d << ' ';
		m_X->neighbors(j).x = *d;

		// neighbor[1]
		temp[0] = mesh_buffer[i + 7];
		temp[1] = mesh_buffer[i + 6];
		temp[2] = mesh_buffer[i + 5];
		temp[3] = mesh_buffer[i + 4];
		//cout << *d << ' ';
		m_X->neighbors(j).y = *d;

		// neighbor[2]
		temp[0] = mesh_buffer[i + 11];
		temp[1] = mesh_buffer[i + 10];
		temp[2] = mesh_buffer[i + 9];
		temp[3] = mesh_buffer[i + 8];
		//cout << *d << ' ';
		m_X->neighbors(j).z = *d;

		// neighbor[3]
		temp[0] = mesh_buffer[i + 15];
		temp[1] = mesh_buffer[i + 14];
		temp[2] = mesh_buffer[i + 13];
		temp[3] = mesh_buffer[i + 12];
		//cout << *d << ' ';
		m_X->neighbors(j).w = *d;

		//cout << endl;
	}

	/////////////////////////////////////////////////////////////////////////////

	/* LOAD IMAGE FILE */
	int width, height, depth;
	is.open(input_image, ifstream::in);
	getline(is, s);	iss.str(s);
	iss >> sub;
	getline(is, s);	iss.str(s);
	iss >> width >> height >> depth;
	iss.clear();
	is.close();

	// Store the image (file contents) into a host memory vector
	cout << "Reading Image : " << sub << '(' << width << ',' << height << ',' << depth << ')' << endl;
	is.open(sub.c_str(), ios::in | ios::binary);
	is.seekg(0, ios::end);
	raw_buffer.resize(is.tellg());
	is.seekg(0, ios::beg);
	is.read((char*)&raw_buffer[0], raw_buffer.size());
	is.close();

	// Create a Float type image, necessary for CUDA Texture Memory Interpolation and color operations
	CImg<ImgType> image(raw_buffer.data(), width, height, depth, 1, false);
	//cout << "Float image created!" << endl;

	m_X->SetTexture(image.data(), image.width(), image.height(), image.depth());

}

template<>
void CXDataManager< CCudaXGraph< CDelaunay_3D_Cuda_XGraph_Adaptor > >::save_file()
{
	cout << "Synchronizing... ";
	m_X->Synchronize();
	cout << "Synchronized!" << endl;

	cout << "Enter Output Mesh (.vtk file): \n";
	cin >> output_mesh;
	cout << "Writing : " << output_mesh << '\n';

	//ofstream os(output_mesh, ofstream::out);
	//for (int i = 0; i< m_X->num_delaunayvertices(); ++i)	os << m_X->delaunayvertices(i).x << ' ' << m_X->delaunayvertices(i).y << ' ' << m_X->delaunayvertices(i).z << ' '; os << endl;
	//for (int i = 0; i< m_X->num_vertices(); ++i)			os << m_X->cellvertices(i).x << ' ' << m_X->cellvertices(i).y << ' ' << m_X->cellvertices(i).z << ' ' << m_X->cellvertices(i).w << ' '; os << endl;
	//for (int i = 0; i< m_X->num_edges(); ++i)				os << m_X->edgenodes(i).x << ' ' << m_X->edgenodes(i).y << ' '; os << endl;
	//for (int i = 0; i< m_X->num_vertices(); ++i)			os << m_X->sortedcells(i) << ' '; os << endl;
	//for (int i = 0; i< m_X->num_cells(); ++i)			os << m_X->colorpatterns(i) << ' '; os << endl;
	//for (int i = 0; i< m_X->num_edges(); ++i)				os << m_X->similarities(i) << ' '; os << endl;
	//for (int i = 0; i< m_X->num_cells(); ++i)			os << m_X->labels(i) << ' '; os << endl;

	// BINARY (Copy and modify Mesh buffer file)
	// Modify NUMBER OF FIELDS
	mesh_buffer[cell_field_data] = '5';

	// Colorpatterns
	// Define header, resize buffer, append header
	string s1 = "colorpatterns 1 " + to_string(m_X->num_cells()) + " float\n";
	int mbuf_size = mesh_buffer.size();
	mesh_buffer.resize(mbuf_size + s1.size() + m_X->num_cells() * 4);
	for (int i = 0; i < s1.size(); ++i) mesh_buffer[mbuf_size + i] = (int)(s1[i]);

	// Append field
	unsigned char *c;
	unsigned char temp[4];
	int j;
	for (int i = 0; i < m_X->num_cells(); ++i)
	{
		j = i * 4;
		c = (unsigned char*)&m_X->colorpatterns(i);
		mesh_buffer[mbuf_size + s1.size() + j] = c[3];
		mesh_buffer[mbuf_size + s1.size() + j + 1] = c[2];
		mesh_buffer[mbuf_size + s1.size() + j + 2] = c[1];
		mesh_buffer[mbuf_size + s1.size() + j + 3] = c[0];
	}

	// Segmentation
	// Define header, resize buffer, append header
	string s2 = "imesh_segmentation 1 " + to_string(m_X->num_cells()) + " int\n";
	mbuf_size = mesh_buffer.size();
	mesh_buffer.resize(mbuf_size + s2.size() + m_X->num_cells() * 4);
	for (int i = 0; i < s2.size(); ++i) mesh_buffer[mbuf_size + i] = (int)(s2[i]);

	// Append field
	for (int i = 0; i < m_X->num_cells(); ++i)
	{
		j = i * 4;
		c = (unsigned char*)&m_X->labels(i);
		mesh_buffer[mbuf_size + s2.size() + j] = c[3];
		mesh_buffer[mbuf_size + s2.size() + j + 1] = c[2];
		mesh_buffer[mbuf_size + s2.size() + j + 2] = c[1];
		mesh_buffer[mbuf_size + s2.size() + j + 3] = c[0];
	}

	// Similarities
	// Define header, resize buffer, append header
	string s3 = "similarities 1 " + to_string(m_X->num_cells()) + " int\n";
	mbuf_size = mesh_buffer.size();
	mesh_buffer.resize(mbuf_size + s3.size() + m_X->num_cells() * 4);
	for (int i = 0; i < s3.size(); ++i) mesh_buffer[mbuf_size + i] = (int)(s3[i]);

	// Append field
	for (int i = 0; i < m_X->num_cells(); ++i)
	{
		j = i * 4;
		c = (unsigned char*)&m_X->similarities(i);
		mesh_buffer[mbuf_size + s3.size() + j] = c[3];
		mesh_buffer[mbuf_size + s3.size() + j + 1] = c[2];
		mesh_buffer[mbuf_size + s3.size() + j + 2] = c[1];
		mesh_buffer[mbuf_size + s3.size() + j + 3] = c[0];
	}

	// CellID
	// Define header, resize buffer, append header
	string s4 = "cell_ID 1 " + to_string(m_X->num_cells()) + " int\n";
	mbuf_size = mesh_buffer.size();
	mesh_buffer.resize(mbuf_size + s4.size() + m_X->num_cells() * 4);
	for (int i = 0; i < s4.size(); ++i) mesh_buffer[mbuf_size + i] = (int)(s4[i]);

	// Append field
	for (int i = 0; i < m_X->num_cells(); ++i)
	{
		j = i * 4;
		c = (unsigned char*)(&i);
		mesh_buffer[mbuf_size + s4.size() + j] = c[3];
		mesh_buffer[mbuf_size + s4.size() + j + 1] = c[2];
		mesh_buffer[mbuf_size + s4.size() + j + 2] = c[1];
		mesh_buffer[mbuf_size + s4.size() + j + 3] = c[0];
	}

	ofstream os(output_mesh, ofstream::out | ofstream::binary);
	os.write((char*)mesh_buffer.data(), mesh_buffer.size()*sizeof(unsigned char));
	os.close();
}

///////////////////////////////////////////////////////
// Spec.3 : Image2D

template<>
void CXDataManager< CCudaXGraph< CImage_2D_Cuda_XGraph_Adaptor > >::load_file()
{
	cout << "Enter Image filename (.png file): \n";
	cin >> input_image;
	CImg<ImgType> image(input_image.c_str());

	m_X->num_cells() = m_X->num_points() = m_X->image_vol() = image.width()*image.height();
	m_X->num_edges() = m_X->num_cells() * 8;

	m_X->UnifiedMalloc();
	m_X->SetTexture(image.data(), image.width(), image.height());
}

template<>
void CXDataManager< CCudaXGraph< CImage_2D_Cuda_XGraph_Adaptor > >::save_file()
{
	cout << "Synchronizing... ";
	m_X->Synchronize();
	cout << "Synchronized!" << endl;

	cout << "Enter Output File : \n";
	cin >> output_mesh;

	cout << "Writing : " << output_mesh << '\n';

	//ofstream os(output_mesh, ofstream::out);
	//for (int i = 0; i< m_X->num_cells(); ++i)			os << m_X->colorpatterns(i) << ' '; os << endl;
	//for (int i = 0; i< m_X->num_cells(); ++i)			os << m_X->sortedcells(i) << ' '; os << endl;
	//for (int i = 0; i< m_X->num_edges(); ++i)				os << m_X->similarities(i) << ' '; os << endl;
	//for (int i = 0; i< m_X->num_cells(); ++i)			os << m_X->labels(i) << ' '; os << endl;

	// Define header
	string h[5], header;
	h[0] = "# vtk DataFile Version 4.0\nvtk output\nBINARY\nDATASET STRUCTURED_POINTS\nDIMENSIONS ";
	h[1] = to_string(m_X->image_dimensions().x) + " " + to_string(m_X->image_dimensions().y) + " " + to_string(m_X->image_dimensions().z);
	h[2] = "\nSPACING 1 1 1\nORIGIN 0 0 0\nPOINT_DATA ";
	h[3] = to_string(m_X->image_vol());
	h[4] = "\nFIELD FieldData 2";
	for (int i = 0; i<5; ++i) header += h[i];

	// Resize buffer and append header
	int mbuf_size = mesh_buffer.size();
	mesh_buffer.resize(mbuf_size + header.size());
	for (int i = 0; i < header.size(); ++i) mesh_buffer[mbuf_size + i] = (int)(header[i]);

	// ... Now do the same with fields

	// Resize buffer and append header
	string field[2];
	field[0] = "\ncolorpatterns 1 " + to_string(m_X->image_vol()) + " float\n";
	mbuf_size = mesh_buffer.size();
	mesh_buffer.resize(mbuf_size + field[0].size());
	for (int i = 0; i < field[0].size(); ++i) mesh_buffer[mbuf_size + i] = (int)(field[0][i]);

	// Append field
	mbuf_size = mesh_buffer.size();
	mesh_buffer.resize(mbuf_size + m_X->image_vol() * 4);
	unsigned char *c;
	unsigned char temp[4];
	int j;
	for (int i = 0; i < m_X->image_vol(); ++i)
	{
		j = i * 4;
		c = (unsigned char*)&m_X->colorpatterns(i);
		mesh_buffer[mbuf_size + j] = c[3];
		mesh_buffer[mbuf_size + j + 1] = c[2];
		mesh_buffer[mbuf_size + j + 2] = c[1];
		mesh_buffer[mbuf_size + j + 3] = c[0];
	}

	// Resize buffer and append header
	field[1] = "\nimesh_segmentation 1 " + to_string(m_X->image_vol()) + " int\n";
	mbuf_size = mesh_buffer.size();
	mesh_buffer.resize(mbuf_size + field[1].size());
	for (int i = 0; i < field[1].size(); ++i) mesh_buffer[mbuf_size + i] = (int)(field[1][i]);

	// Append field
	mbuf_size = mesh_buffer.size();
	mesh_buffer.resize(mbuf_size + m_X->image_vol() * 4);
	for (int i = 0; i < m_X->image_vol(); ++i)
	{
		j = i * 4;
		c = (unsigned char*)&m_X->labels(i);
		mesh_buffer[mbuf_size + j] = c[3];
		mesh_buffer[mbuf_size + j + 1] = c[2];
		mesh_buffer[mbuf_size + j + 2] = c[1];
		mesh_buffer[mbuf_size + j + 3] = c[0];
	}

	ofstream os(output_mesh, ofstream::out | ofstream::binary);
	os.write((char*)mesh_buffer.data(), mesh_buffer.size()*sizeof(unsigned char));
	os.close();
}

///////////////////////////////////////////////////////
// Spec.4 : Image3D

template<>
void CXDataManager< CCudaXGraph< CImage_3D_Cuda_XGraph_Adaptor > >::load_file()
{
	/* LOAD IMAGE FILE */
	cout << "Enter 3D Image filename (.dat file): \n";
	cin >> input_image;

	int width, height, depth;
	is.open(input_image, ifstream::in);
	getline(is, s);	iss.str(s);
	iss >> sub;
	getline(is, s);	iss.str(s);
	iss >> width >> height >> depth;
	iss.clear();
	is.close();

	// Store the image (file contents) into a host memory vector
	cout << "Reading Image : " << sub << '(' << width << ',' << height << ',' << depth << ')' << endl;
	is.open(sub.c_str(), ios::in | ios::binary);
	is.seekg(0, ios::end);
	raw_buffer.resize(is.tellg());
	is.seekg(0, ios::beg);
	is.read((char*)&raw_buffer[0], raw_buffer.size());
	is.close();

	// Create a Float type image...
	CImg<ImgType> image(raw_buffer.data(), width, height, depth, 1, false);
	//cout << "Float image created!" << endl;

	m_X->num_cells() = m_X->num_points() = m_X->image_vol() = image.width()*image.height()*image.depth();
	m_X->num_edges() = m_X->num_cells() * 26;

	cout << "Num cells = " << m_X->num_cells() << endl;

	m_X->UnifiedMalloc();
	m_X->SetTexture(image.data(), image.width(), image.height(), image.depth());
}

template<>
void CXDataManager< CCudaXGraph< CImage_3D_Cuda_XGraph_Adaptor > >::save_file()
{
	/* SAVE IMAGE */
	cout << "Synchronizing... ";
	m_X->Synchronize();
	cout << "Synchronized!" << endl;

	cout << "Enter Output File : \n";
	cin >> output_image;

	cout << "Writing : " << output_image << '\n';

	// Colorpatterns
	string colorpatterns_out(output_image);
	colorpatterns_out.replace(colorpatterns_out.end() - 4, colorpatterns_out.end(), "_colorpattern.raw");
	cout << "Writing : " << colorpatterns_out << '\n';
	ofstream os1(colorpatterns_out, ofstream::out | ofstream::binary);
	os1.write((char*)m_X->colorpatterns(), m_X->num_cells()*sizeof(float));
	os1.close();

	////Sorted cells
	//string sortedcells_out(output_image);
	//sortedcells_out.replace(sortedcells_out.end() - 4, sortedcells_out.end(), "_sortedcells.raw");
	//cout << "Writing : " << sortedcells_out << '\n';
	//ofstream os2(sortedcells_out, ofstream::out | ofstream::binary);
	//os2.write((char*)m_X->sortedcells(), m_X->num_cells()*sizeof(int));
	//os2.close();

	////Similarities
	//string similarities_out(output_image);
	//similarities_out.replace(similarities_out.end() - 4, similarities_out.end(), "_similarities.raw");
	//cout << "Writing : " << similarities_out << '\n';
	//ofstream os3(similarities_out, ofstream::out | ofstream::binary);
	//os3.write((char*)m_X->similarities(), m_X->num_edges()*sizeof(float));
	//os3.close();

	//Labels
	string labels_out(output_image);
	labels_out.replace(labels_out.end() - 4, labels_out.end(), "_labels.raw");
	cout << "Writing : " << labels_out << '\n';
	ofstream os4(labels_out, ofstream::out | ofstream::binary);
	os4.write((char*)m_X->labels(), m_X->num_cells()*sizeof(int));
	os4.close();

	// Old, slow way...
	//ofstream os(output_image, ofstream::out);
	//for (int i = 0; i< m_X->num_cells(); ++i)			os << m_X->colorpatterns(i) << ' '; os << endl;
	//for (int i = 0; i< m_X->num_cells(); ++i)			os << m_X->updatepatterns(i) << ' '; os << endl;
	//for (int i = 0; i< m_X->num_edges(); ++i)				os << m_X->similarities(i) << ' '; os << endl;
	//for (int i = 0; i< m_X->num_vertices(); ++i)			os << m_X->labels(i) << ' '; os << endl;
}

#endif //XREADER_H
