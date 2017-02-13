#include "Similarities_Calculator.cuh"
#include "Similarities_Kernels.cuh"

/////////////////////////////
// Partial Spec.1 : Delaunay contexts (Delaunay2D/ Delaunay3D)

template<class X>
CSimilaritiesCalculator<X, CDelaunay>::CSimilaritiesCalculator(X &x) : m_X(x)
{
	std::cout << "\nInput Similarity Threshold: \n";
	std::cin >> m_threshold;

	m_isovaluelist->load_values();
	m_numisovalues = m_isovaluelist->size();

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
	cudaMemcpyToSymbol(Num_isovalues, &m_numisovalues, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Isovalues, m_isovaluelist->data(), sizeof(IsovalueType)*m_numisovalues, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Threshold, &m_threshold, sizeof(float), 0, cudaMemcpyHostToDevice);
}

template<class X>
void CSimilaritiesCalculator<X, CDelaunay>::compute_neighbors()
{
	// thrust::device_vector<NeighborsType> face_records_vertices(m_X->num_cells() * 4);
	// thrust::device_vector<int> face_records_cells(m_X->num_cells() * 4);
	// thrust::device_vector<int> face_records_neighbors(m_X->num_cells() * 4);
	//
	// NeighborsType* dp_face_records_vertices = thrust::raw_pointer_cast(face_records_vertices.data());
	// int* dp_face_records_cells = thrust::raw_pointer_cast(face_records_cells.data());
	// int* dp_face_records_neighbors = thrust::raw_pointer_cast(face_records_neighbors.data());
	//
	// generate_face_records << < (m_X->num_cells() + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (dp_face_records_vertices, dp_face_records_cells, dp_face_records_neighbors);

	// Bitonic Sorts by Bin


	// Find connectivity
	//find_connectivity << < >> >();
}

template<class X>
void CSimilaritiesCalculator<X, CDelaunay>::compute_similarities()
{
	printf("\nSimilarities computing (DELAUNAY) ... ");
	float ms1;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	reset_similarity_Graph_kernel <XSpace> << < m_X->num_cells(), WARP_SIZE >> > (m_X->neighbors(), m_X->similarities());

	thrust::device_vector<int2> finite_edges(m_X->num_cells()*XSpace::NCellNeighbors);
	thrust::device_vector<int> edge_counter(1);

	int2* dp_finite_edges = thrust::raw_pointer_cast(finite_edges.data());
	int* dp_edge_counter = thrust::raw_pointer_cast(edge_counter.data());

	filter_edges_kernel <XSpace> << <(m_X->num_cells()*XSpace::NCellNeighbors + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> >(dp_finite_edges, m_X->neighbors(), dp_edge_counter);
	m_X->num_edges() = edge_counter[0];
	finite_edges.resize(m_X->num_edges());
	cudaMemcpyToSymbol(E, &(m_X->num_edges()), sizeof(int), 0, cudaMemcpyHostToDevice);

	//thrust::host_vector<int2> h0(finite_edges);
	//printf("Num. Edges = %i\n", m_X->num_edges());
	//for (int i = 0; i < h0.size(); ++i) printf("(%i, %i) ", h0[i].x, h0[i].y); printf("\n");

	thrust::device_vector<FacetsType> facet_vertices(m_X->num_edges());
	thrust::device_vector<PointCoordsType> cell_centroids(m_X->num_cells());
	thrust::device_vector<PointCoordsType> facet_centroids(m_X->num_edges());
	thrust::device_vector<int2> neighbor_positions(m_X->num_edges());
	FacetsType* dp_facet_vertices = thrust::raw_pointer_cast(facet_vertices.data());
	PointCoordsType* dp_cell_centroids = thrust::raw_pointer_cast(cell_centroids.data());
	PointCoordsType* dp_facet_centroids = thrust::raw_pointer_cast(facet_centroids.data());
	int2* dp_neighbor_positions = thrust::raw_pointer_cast(neighbor_positions.data());

	compute_facets_kernel <XSpace> << < (m_X->num_edges() + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (dp_finite_edges, m_X->cellvertices(), m_X->pointcoords(), dp_facet_vertices);

	compute_cellcentroids_kernel <XSpace> << <(m_X->num_cells() + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> >(m_X->cellvertices(), m_X->pointcoords(), dp_cell_centroids);
	//compute_facetcentroids_kernel <XSpace> << <(m_X->num_edges() + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (dp_finite_edges, m_X->cellvertices(), m_X->pointcoords(), dp_facet_centroids);
	compute_facetcentroids_kernel2 <XSpace> << <(m_X->num_edges() + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (dp_finite_edges, dp_facet_vertices, m_X->pointcoords(), dp_facet_centroids);
	compute_neighborpositions_kernel <XSpace> << <(m_X->num_edges() + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (dp_finite_edges, m_X->neighbors(), dp_neighbor_positions);
	compute_similarity_Graph_kernel <XSpace> << <m_X->num_edges(), WARP_SIZE >> >(m_X->texObject(), dp_finite_edges, dp_neighbor_positions, dp_cell_centroids, dp_facet_centroids, m_X->colorpatterns(), m_X->similarities());

	/*thrust::device_vector<float> cell_maxfacets(m_X->num_cells());
	float* dp_cell_maxfacets = thrust::raw_pointer_cast(cell_maxfacets.data());
	compute_maxfacets_kernel <XSpace> << <(m_X->num_cells() + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (m_X->cellvertices(), m_X->pointcoords(), dp_cell_maxfacets);
	compute_geometric_similarity_kernel <XSpace> << <(m_X->num_edges() + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (dp_finite_edges, dp_neighbor_positions, m_X->pointcoords(), dp_facet_vertices, dp_cell_maxfacets, m_X->similarities());*/


	//m_X->Synchronize();
	//for (int i = 0; i < m_X->num_cells(); ++i) printf("(%i) %i - ", i, m_X->similarities(i)); printf("\n");

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms1, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceSynchronize(); printf("Done! (%.2f ms)\n", ms1);
}

template<class X>
void CSimilaritiesCalculator<X, CDelaunay>::reset_similarities()
{
	reset_similarity_Graph_kernel <XSpace> << < m_X->num_cells(), WARP_SIZE >> > (m_X->neighbors(), m_X->similarities());
}

///////////////////////////
// Partial Spec.2 : Image contexts (Image2D/ Image3D)

template<class X>
CSimilaritiesCalculator<X, CImage>::CSimilaritiesCalculator(X &x) : m_X(x)
{
	std::cout << "\nInput Similarity Threshold: \n";
	std::cin >> m_threshold;

	m_isovaluelist->load_values();
	m_numisovalues = m_isovaluelist->size();

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
	cudaMemcpyToSymbol(Num_isovalues, &m_numisovalues, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Isovalues, m_isovaluelist->data(), sizeof(IsovalueType)*m_numisovalues, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Threshold, &m_threshold, sizeof(float), 0, cudaMemcpyHostToDevice);

	float inv_width = 1.0 / m_X->image_dimensions().x;
	float inv_heightxwidth = 1.0 / (m_X->image_dimensions().x * m_X->image_dimensions().y);

	cudaMemcpyToSymbol(Inv_width, &(inv_width), sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Inv_heightxwidth, &(inv_heightxwidth), sizeof(float), 0, cudaMemcpyHostToDevice);

	neighbor_steps.resize(26);

	// Z
	neighbor_steps[0] = 1;
	neighbor_steps[1] = 1 + m_X->image_dimensions().x;
	neighbor_steps[2] = m_X->image_dimensions().x;
	neighbor_steps[3] = -1 + m_X->image_dimensions().x;
	neighbor_steps[4] = -1;
	neighbor_steps[5] = -1 - m_X->image_dimensions().x;
	neighbor_steps[6] = -m_X->image_dimensions().x;
	neighbor_steps[7] = 1 - m_X->image_dimensions().x;

	// Z - 1
	neighbor_steps[8] = -m_X->image_dimensions().x*m_X->image_dimensions().y;
	neighbor_steps[9] = neighbor_steps[8] + neighbor_steps[0];
	neighbor_steps[10] = neighbor_steps[8] + neighbor_steps[1];
	neighbor_steps[11] = neighbor_steps[8] + neighbor_steps[2];
	neighbor_steps[12] = neighbor_steps[8] + neighbor_steps[3];
	neighbor_steps[13] = neighbor_steps[8] + neighbor_steps[4];
	neighbor_steps[14] = neighbor_steps[8] + neighbor_steps[5];
	neighbor_steps[15] = neighbor_steps[8] + neighbor_steps[6];
	neighbor_steps[16] = neighbor_steps[8] + neighbor_steps[7];

	// Z + 1
	neighbor_steps[17] = m_X->image_dimensions().x*m_X->image_dimensions().y;
	neighbor_steps[18] = neighbor_steps[17] + neighbor_steps[0];
	neighbor_steps[19] = neighbor_steps[17] + neighbor_steps[1];
	neighbor_steps[20] = neighbor_steps[17] + neighbor_steps[2];
	neighbor_steps[21] = neighbor_steps[17] + neighbor_steps[3];
	neighbor_steps[22] = neighbor_steps[17] + neighbor_steps[4];
	neighbor_steps[23] = neighbor_steps[17] + neighbor_steps[5];
	neighbor_steps[24] = neighbor_steps[17] + neighbor_steps[6];
	neighbor_steps[25] = neighbor_steps[17] + neighbor_steps[7];

	neighbor_stepcoords.resize(26);

	// Z
	neighbor_stepcoords[0] = make_int3(1, 0, 0);
	neighbor_stepcoords[1] = make_int3(1, 1, 0);
	neighbor_stepcoords[2] = make_int3(0, 1, 0);
	neighbor_stepcoords[3] = make_int3(-1, 1, 0);
	neighbor_stepcoords[4] = make_int3(-1, 0, 0);
	neighbor_stepcoords[5] = make_int3(-1, -1, 0);
	neighbor_stepcoords[6] = make_int3(0, -1, 0);
	neighbor_stepcoords[7] = make_int3(1, -1, 0);

	// Z - 1
	neighbor_stepcoords[8] = make_int3(0, 0, -1);
	neighbor_stepcoords[9] = neighbor_stepcoords[8] + neighbor_stepcoords[0];
	neighbor_stepcoords[10] = neighbor_stepcoords[8] + neighbor_stepcoords[1];
	neighbor_stepcoords[11] = neighbor_stepcoords[8] + neighbor_stepcoords[2];
	neighbor_stepcoords[12] = neighbor_stepcoords[8] + neighbor_stepcoords[3];
	neighbor_stepcoords[13] = neighbor_stepcoords[8] + neighbor_stepcoords[4];
	neighbor_stepcoords[14] = neighbor_stepcoords[8] + neighbor_stepcoords[5];
	neighbor_stepcoords[15] = neighbor_stepcoords[8] + neighbor_stepcoords[6];
	neighbor_stepcoords[16] = neighbor_stepcoords[8] + neighbor_stepcoords[7];

	// Z + 1
	neighbor_stepcoords[17] = make_int3(0, 0, 1);
	neighbor_stepcoords[18] = neighbor_stepcoords[17] + neighbor_stepcoords[0];
	neighbor_stepcoords[19] = neighbor_stepcoords[17] + neighbor_stepcoords[1];
	neighbor_stepcoords[20] = neighbor_stepcoords[17] + neighbor_stepcoords[2];
	neighbor_stepcoords[21] = neighbor_stepcoords[17] + neighbor_stepcoords[3];
	neighbor_stepcoords[22] = neighbor_stepcoords[17] + neighbor_stepcoords[4];
	neighbor_stepcoords[23] = neighbor_stepcoords[17] + neighbor_stepcoords[5];
	neighbor_stepcoords[24] = neighbor_stepcoords[17] + neighbor_stepcoords[6];
	neighbor_stepcoords[25] = neighbor_stepcoords[17] + neighbor_stepcoords[7];


}

template<class X>
void CSimilaritiesCalculator<X, CImage>::compute_similarities()
{
	printf("\nSimilarities computing (IMAGE) ... ");
	float ms1;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	int *dp_neighbor_steps = thrust::raw_pointer_cast(neighbor_steps.data());
	int3 *dp_neighbor_stepcoords = thrust::raw_pointer_cast(neighbor_stepcoords.data());

	thrust::device_vector<int> borders(m_X->num_cells());
	thrust::device_vector<int> border_counter(1);
	int *dp_borders = thrust::raw_pointer_cast(borders.data());
	int *dp_border_counter = thrust::raw_pointer_cast(border_counter.data());
	int n_borderelems;

	filter_Image_borders <true> << <(m_X->num_cells() + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> >(dp_borders, dp_border_counter);
	compute_similarity_Image_kernel_v4 <XSpace, true> << < (border_counter[0] + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> >(dp_borders, border_counter[0], m_X->colorpatterns(), m_X->similarities(), dp_neighbor_steps, dp_neighbor_stepcoords);
	n_borderelems = border_counter[0];
	filter_Image_borders <false> << <(m_X->num_cells() + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> >(dp_borders, dp_border_counter);
	compute_similarity_Image_kernel_v4 <XSpace, false> << < (m_X->num_cells() - n_borderelems + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> >(dp_borders + n_borderelems, m_X->num_cells() - n_borderelems, m_X->colorpatterns(), m_X->similarities(), dp_neighbor_steps, dp_neighbor_stepcoords);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms1, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceSynchronize(); printf("Done! (%.2f ms)\n", ms1);

	//printf("Border elements = %i\n", n_borderelems);
}

template<class X>
void CSimilaritiesCalculator<X, CImage>::reset_similarities()
{
	int3 *dp_neighbor_stepcoords = thrust::raw_pointer_cast(neighbor_stepcoords.data());
	thrust::device_vector<int> borders(m_X->num_cells());
	thrust::device_vector<int> border_counter(1);
	int *dp_borders = thrust::raw_pointer_cast(borders.data());
	int *dp_border_counter = thrust::raw_pointer_cast(border_counter.data());
	int n_borderelems;

	thrust::fill(thrust::cuda::par, m_X->similarities(), m_X->similarities() + m_X->num_cells(), 0);
	filter_Image_borders <true> << <(m_X->num_cells() + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> >(dp_borders, dp_border_counter);
	reset_similarity_Image_kernel <XSpace> << < (border_counter[0] + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> >(dp_borders, border_counter[0], m_X->similarities(), dp_neighbor_stepcoords);
}

//////////////////////////////////////////////////
// Instantiate to avoid Linker error
template class CSimilaritiesCalculator< CCudaXGraph< CDelaunay_2D_Cuda_XGraph_Adaptor > >;
template class CSimilaritiesCalculator< CCudaXGraph< CDelaunay_3D_Cuda_XGraph_Adaptor > >;
template class CSimilaritiesCalculator< CCudaXGraph< CImage_2D_Cuda_XGraph_Adaptor > >;
template class CSimilaritiesCalculator< CCudaXGraph< CImage_3D_Cuda_XGraph_Adaptor > >;
//////////////////////////////////////////////////
