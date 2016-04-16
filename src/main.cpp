//#include "cudatemplate.h"
#include "cudagraphsegmentator.cuh"

using namespace std;

typedef CCudaCSRGraphSegmentator<int> CGraphSegmentator;

int main()
{
	/*GPU Segmentator Class*/
	CGraphSegmentator IsoSegmentator;

    IsoSegmentator.write();

	/*Colorize graph*/
	IsoSegmentator.colorize();

	/*Compute Color Patterns*/
	//IsoSegmentator.compute_colorpatterns();
	IsoSegmentator.write();

	/*Compute Cell Similarities*/
	// gs.compute_similarities();
    
	/*Segmentate graph*/
	// gs.segmentate();

	/*Return Data From GPU*/
	//     gs.GetDataFromGPU();


// // ////////////////////////////

//     // image.display();

	// /*Output data*/
	// ofstream os("graph_segmentation.txt", ifstream::out);

    return 0;
}