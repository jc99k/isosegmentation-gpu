#include "Graph_Colorizer.cuh"
#include "Graph_Colorizer_Kernels.cuh"

/////////////////////////////
// Partial Spec.1 : Delaunay contexts (Delaunay2D/ Delaunay3D)

template<class X>
CGraphColorizer<X, CDelaunay>::CGraphColorizer(X &x) : m_X(x) 
{
	cudaMemcpyToSymbol(V, &(m_X->num_cells()), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(E, &(m_X->num_edges()), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(DV, &(m_X->num_points()), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Img_width, &(m_X->image_dimensions().x), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Img_height, &(m_X->image_dimensions().y), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Img_depth, &(m_X->image_dimensions().z), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Vol, &(m_X->image_vol()), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Dim, &(XSpace::Dim), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(NCellVertices, &(XSpace::NCellVertices), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(NCellNeighbors, &(XSpace::NCellNeighbors), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(NFacetVertices, &(XSpace::NFacetVertices), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(NGraphColors, &(XSpace::NGraphColors), sizeof(int), 0, cudaMemcpyHostToDevice);
}

template<class X>
void CGraphColorizer<X, CDelaunay>::colorize()
{
	// REPLACE m_X->updatepatterns() with m_X->labels()
	printf("\nGraph Colorization (DELAUNAY) ... ");
	float ms1;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// device storage for the random numbers
	thrust::device_vector<float> randoms(m_X->num_cells());
	float* dp_randoms = thrust::raw_pointer_cast(randoms.data());

	// generate randoms
	rng_generate_kernel <XSpace> << <(m_X->num_cells() + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> >(dp_randoms);
	thrust::fill(thrust::cuda::par, m_X->labels(), m_X->labels() + m_X->num_cells(), -1);

	// colorize graph
	int c;
	thrust::device_vector<int> left(1);
	int* dp_left = thrust::raw_pointer_cast(left.data());

	for (c = 0; c < m_X->num_cells(); ++c) {
		left[0] = 0;
		color_jpl_kernel <XSpace> << <(m_X->num_cells() + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> >(c, m_X->neighbors(), dp_randoms, m_X->labels());
		count_left << <(m_X->num_cells() + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (m_X->labels(), m_X->num_cells(), -1, dp_left);
		if (left[0] == 0) break;
	}

	// fix extra colors
	for (; c > XSpace::NCellNeighbors; --c) {
		correct_color_kernel <XSpace> << <(m_X->num_cells() + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (c, m_X->neighbors(), m_X->labels());
	}

	//num_colors = ++c;
	//correct_color_kernel_test << <(m_X->num_vertices() + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK>> > (m_X->neighbors(), m_X->labels());
	
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
CGraphColorizer<X, CImage>::CGraphColorizer(X &x) : m_X(x) 
{
	cudaMemcpyToSymbol(V, &(m_X->num_cells()), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(E, &(m_X->num_edges()), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(DV, &(m_X->num_points()), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Img_width, &(m_X->image_dimensions().x), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Img_height, &(m_X->image_dimensions().y), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Img_depth, &(m_X->image_dimensions().z), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Vol, &(m_X->image_vol()), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Dim, &(XSpace::Dim), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(NCellVertices, &(XSpace::NCellVertices), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(NCellNeighbors, &(XSpace::NCellNeighbors), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(NFacetVertices, &(XSpace::NFacetVertices), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(NGraphColors, &(XSpace::NGraphColors), sizeof(int), 0, cudaMemcpyHostToDevice);
}

template<class X>
void CGraphColorizer<X, CImage>::colorize()
{
	printf("\nGraph Colorization (IMAGE) ... ");
	float ms1;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	colorize_Image_kernel <XSpace> << <(m_X->num_cells() + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (m_X->labels());

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms1, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceSynchronize(); printf("Done! (%.2f ms)\n", ms1);
}

//////////////////////////////////////////////////
// Instantiate to avoid Linker error
template class CGraphColorizer< CCudaXGraph< CDelaunay_2D_Cuda_XGraph_Adaptor > >;
template class CGraphColorizer< CCudaXGraph< CDelaunay_3D_Cuda_XGraph_Adaptor > >;
template class CGraphColorizer< CCudaXGraph< CImage_2D_Cuda_XGraph_Adaptor > >;
template class CGraphColorizer< CCudaXGraph< CImage_3D_Cuda_XGraph_Adaptor > >;
//////////////////////////////////////////////////