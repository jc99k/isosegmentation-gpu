#include "Delaunay2D.cuh"
#include "Delaunay3D.cuh"
#include "Cuda_XGraph.cuh"
#include "XReader.h"
#include "XWriter.h"
#include "Cuda_XGraph_Segmentator.cuh"

typedef CDelaunay_3D_Cuda_XGraph_Adaptor XSpace;
typedef CCudaXGraph<XSpace> XGraph;
typedef CXReader<XGraph> XReader;
typedef CXWriter<XGraph> XWriter;
typedef CCudaXGraphSegmentator<XGraph> XSegmentator;

int main()
{
	XGraph x;
	XReader Rx(x);
	XWriter Wx(x);

	Rx.load_file();

	XSegmentator Sx(x);

	/*Colorize graph*/
	Sx.colorize();

	/*Compute Color Patterns*/
	Sx.compute_colorpatterns_and_similarities();

	/*Segmentate graph*/
	Sx.segmentate();


//////////////////////////////////
	
	
	Wx.save_file();

    return 0;
}