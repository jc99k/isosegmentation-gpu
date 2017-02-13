#include "ColorPatterns_Calculator.cuh"
#include "Colorpatterns_Kernels.cuh"

/////////////////////////////
// Partial Spec.1 : Delaunay contexts (Delaunay2D/ Delaunay3D)

template<class X>
CColorPatternsCalculator<X, CDelaunay>::CColorPatternsCalculator(X &x) : m_X(x) 
{
	cudaMemcpyToSymbol(V, &(m_X->num_cells()), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(E, &(m_X->num_edges()), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(DV, &(m_X->num_points()), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Img_width, &(m_X->image_dimensions().x), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Img_height, &(m_X->image_dimensions().y), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Img_depth, &(m_X->image_dimensions().z), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Vol, &(m_X->image_vol()), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Dim, &(XSpace::Dim), sizeof(int), 0, cudaMemcpyHostToDevice);
}

template<class X>
void CColorPatternsCalculator<X, CDelaunay>::compute_colorpatterns()
{
	printf("\nColor patterns computing (DELAUNAY) ... ");
	float ms1;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	thrust::device_vector<BaryCoordsType> samplesBaryCoords(NUM_COLORPATTERN_SAMPLES);
	BaryCoordsType *dp_samplesBaryCoords = thrust::raw_pointer_cast(samplesBaryCoords.data());
	
	flip_YAxis_Graph_kernel <XSpace> << <(m_X->num_points() + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> >(m_X->pointcoords());
	generate_sample_points_kernel <XSpace> << <(NUM_COLORPATTERN_SAMPLES + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (dp_samplesBaryCoords);
	compute_cellsizes_kernel <XSpace> << <(m_X->num_cells() + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> >(m_X->cellvertices(), m_X->pointcoords(), m_X->sizes());
	compute_colorpatterns_kernel <XSpace> << <m_X->num_cells(), WARP_SIZE*NUM_WARPS_FOR_COLORPATTERN >> >(m_X->texObject(), m_X->colorpatterns(), m_X->cellvertices(), m_X->pointcoords(), dp_samplesBaryCoords, m_X->sizes());

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms1, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceSynchronize(); printf("Done! (%.2f ms)\n", ms1);
}

///////////////////////////
// Partial Spec.2 : Image contexts (Image2D/ Image3D)

template<class X>
CColorPatternsCalculator<X, CImage>::CColorPatternsCalculator(X &x) : m_X(x) 
{
	cudaMemcpyToSymbol(V, &(m_X->num_cells()), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(E, &(m_X->num_edges()), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(DV, &(m_X->num_points()), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Img_width, &(m_X->image_dimensions().x), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Img_height, &(m_X->image_dimensions().y), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Img_depth, &(m_X->image_dimensions().z), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Vol, &(m_X->image_vol()), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Dim, &(XSpace::Dim), sizeof(int), 0, cudaMemcpyHostToDevice);
}

template<class X>
void CColorPatternsCalculator<X, CImage>::compute_colorpatterns()
{
	printf("\nColor patterns computing (IMAGE) ... ");
	float ms1;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	flip_YAxis_Image_kernel <XSpace> << <(m_X->num_points() + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> >(m_X->pointcoords());

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms1, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceSynchronize(); printf("Done! (%.2f ms)\n", ms1);
}

//////////////////////////////////////////////////
// Instantiate to avoid Linker error
template class CColorPatternsCalculator< CCudaXGraph< CDelaunay_2D_Cuda_XGraph_Adaptor > >;
template class CColorPatternsCalculator< CCudaXGraph< CDelaunay_3D_Cuda_XGraph_Adaptor > >;
template class CColorPatternsCalculator< CCudaXGraph< CImage_2D_Cuda_XGraph_Adaptor > >;
template class CColorPatternsCalculator< CCudaXGraph< CImage_3D_Cuda_XGraph_Adaptor > >;
//////////////////////////////////////////////////