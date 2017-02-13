#include "Delaunay2D.h"
#include "Delaunay3D.h"
#include "Image2D.h"
#include "Image3D.h"
#include "Cuda_XGraph.cuh"
#include "ColorPatterns_Calculator.cuh"
#include "XDataManager.h"
#include "Cuda_XGraph_Segmentator.cuh"

// #include "Similarities_Calculator.cuh"

//typedef CDelaunay_2D_Cuda_XGraph_Adaptor	XSpace;
typedef CDelaunay_3D_Cuda_XGraph_Adaptor	XSpace;
//typedef CImage_2D_Cuda_XGraph_Adaptor		XSpace;
//typedef CImage_3D_Cuda_XGraph_Adaptor		XSpace;

typedef CCudaXGraph<XSpace>					XGraph;
typedef CXDataManager<XGraph>				XDataManager;
typedef CColorPatternsCalculator<XGraph>	XColorPatternsCalculator;
 typedef CCudaXGraphSegmentator<XGraph>		XSegmentator;
 //typedef CSimilaritiesCalculator<XGraph>		XSimilaritiesCalculator;

int main()
{
	XGraph x;
	XDataManager Dx(x);
	Dx.load_file();

	// XSimilaritiesCalculator Simx(x);
	// Simx.compute_neighbors();

	/*Compute Color Patterns*/
	XColorPatternsCalculator Cx(x);
	Cx.compute_colorpatterns();

	/*Compute Similarities and Update Patterns*/
	XSegmentator Sx(x);

	/*Segmentate graph*/
	Sx.segmentate();
	//Sx.segmentate2();
	//Sx.segmentate3();

	Dx.save_file();

	return 0;

}
