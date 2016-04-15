#include "class.h"
#include "thrust_headers.h"
#include <cuda.h>
#include <curand_kernel.h>

template<class G, kernel K>
class CCudaGraphSegmentator
{
public:
    void SendDataToGPU(int, int, int, 
        int *, int *, int *, int *, 
        float *, float *, float *, float *, 
        int *, float *, float *, int);
    void SendImageToGPU(float *, int, int, int);
    void GetDataFromGPU();
    void colorize();
    void compute_centroids();
    void compute_colorpatterns();
    void compute_colorpatterns_tex();
    void compute_similarities();
    void segmentate();
    void segmentate2();
    void segmentate3();
    void FreeGPU();
    void Transform();
    void Synchronize();

    int *hPtr_xadj, *hPtr_adjncy, *hPtr_coloring, *hPtr_labels;
    int *dPtr_xadj, *dPtr_adjncy, *dPtr_coloring, *dPtr_labels;
    float *hPtr_colorpatterns, *hPtr_randoms, *hPtr_similarities;
    float *dPtr_colorpatterns, *dPtr_randoms, *dPtr_similarities;
    int *dPtr_cell_sizes;
    int v,e,dv;
    int num_colors, num_regions;

    dim3 nt, nb;

    curandState *dPtr_curandStates;

    float *hPtr_cell_centroids, *hPtr_facet_centroids;
    float2 *dPtr_cell_centroids, *dPtr_facet_centroids;

    float *hPtr_image_data, *dPtr_image_data;
    int image_width, image_height, image_channels, image_dimension;
    size_t pitch, widthInBytes;
    cudaTextureObject_t texObj;
    

    int *hPtr_cell_vertices;
    int3 *dPtr_cell_vertices;
    
    float *hPtr_delaunay_vertices;
    float2 *dPtr_delaunay_vertices;

    int repeat;
    float threshold;

    thrust::device_vector<int>      dVector_xadj;
    thrust::device_vector<int>      dVector_adjncy;
    thrust::device_vector<int>      dVector_coloring;
    thrust::device_vector<int>      dVector_labels;
    thrust::device_vector<int>      dVector_cell_sizes;
    thrust::device_vector<float>    dVector_randoms;
    thrust::device_vector<float>    dVector_colorpatterns;
    thrust::device_vector<float>    dVector_similarities;
    thrust::device_vector<curandState> dVector_curandStates;
    thrust::device_vector<float2>      dVector_cell_centroids;
    thrust::device_vector<float2>      dVector_facet_centroids;
    thrust::device_vector<float2>      dVector_delaunay_vertices;
    thrust::device_vector<int3>      dVector_cell_vertices;
};
