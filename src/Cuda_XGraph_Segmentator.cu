#include "Delaunay2D.cuh"
#include "Delaunay3D.cuh"
#include "Cuda_XGraph.cuh"
#include "Cuda_XGraph_Segmentator.cuh"
#include "kernels.h"
#include <cstdio>

#include <curand_kernel.h>
#include "thrust_custom_predicates.h"

template<class X>
CCudaXGraphSegmentator<X>::CCudaXGraphSegmentator(X& x) : m_X(x)
{
	v = x->num_vertices();
	e = x->num_edges();
	dv = x->num_delaunayvertices();

	std::cout << "Threshold = ";
	std::cin >> threshold;

	m_isovaluelist->load_values();
	num_isovalues = m_isovaluelist->size();

	printf("Copy to constant symbols...\n");

	cudaMemcpyToSymbol(V, &v, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(E, &e, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(DV, &dv, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Num_isovalues, &num_isovalues, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Isovalues, m_isovaluelist->data(), sizeof(IsovalueType)*num_isovalues, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Img_width, &(m_X->image_dimensions().x), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Img_height, &(m_X->image_dimensions().y), sizeof(int), 0, cudaMemcpyHostToDevice);
	
	printf("Done.\n");

	ThreadsPerBlock = dim3(1024, 1, 1);
	BlocksInGrid = dim3(min((v + ThreadsPerBlock.x - 1) / ThreadsPerBlock.x, 65535), 1, 1);
}

template<class X>
void CCudaXGraphSegmentator<X>::colorize()
{
	printf("Colorization...\n");
	
	// device storage for the random numbers
	thrust::device_vector<float> randoms(v);
	float* rawPtr_randoms = thrust::raw_pointer_cast(randoms.data());

	// generate randoms
	rng_generate_kernel << <BlocksInGrid, ThreadsPerBlock >> >(rawPtr_randoms);
	thrust::fill(thrust::cuda::par, m_X->updatepatterns(), m_X->updatepatterns() + v, -1);

	for (int c = 0; c < v; ++c) {
		color_jpl_kernel << <BlocksInGrid, ThreadsPerBlock >> >(c, m_X->xadj(), m_X->adjncy(), rawPtr_randoms, m_X->updatepatterns());
		int left = thrust::count(thrust::cuda::par, m_X->updatepatterns(), m_X->updatepatterns() + v, -1);
		if (left == 0)
		{
			num_colors = c + 1;
			break;
		}
	}
	printf("Done.\n");
}

template<class X>
void CCudaXGraphSegmentator<X>::compute_colorpatterns_and_similarities()
{
	
	thrust::device_vector<DelaunayVerticesType> cell_centroids(v);
	thrust::device_vector<DelaunayVerticesType> facet_centroids(e * 2);
	DelaunayVerticesType* rawPtr_cell_centroids = thrust::raw_pointer_cast(cell_centroids.data());
	DelaunayVerticesType* rawPtr_facet_centroids = thrust::raw_pointer_cast(facet_centroids.data());

	//flip_verticality_kernel << <dv, 1 >> >(m_X->delaunayvertices());
	//compute_cellcentroids_kernel << <nb, nt >> >(m_X->xadj(), m_X->adjncy(), m_X->cellvertices(), m_X->delaunayvertices(), rawPtr_cell_centroids);
	//basic_colorpatterns_kernel << <nb, nt >> > (m_X->texObject(), m_X->colorpatterns(), rawPtr_cell_centroids);
	//compute_facetcentroids_kernel << <nb, nt >> > (m_X->xadj(), m_X->adjncy(), m_X->cellvertices(), m_X->delaunayvertices(), rawPtr_facet_centroids);
	//compute_similarity_kernel << <nb, nt >> >(m_X->xadj(), m_X->adjncy(), m_X->texObject(), rawPtr_cell_centroids, rawPtr_facet_centroids, m_X->colorpatterns(), m_X->similarities());

	printf("Color pattern computation...\n");
	flip_verticality_kernel_3D << <dv, 1 >> >(m_X->delaunayvertices());
	compute_cellcentroids_kernel_3D << <BlocksInGrid, ThreadsPerBlock >> >(m_X->xadj(), m_X->adjncy(), m_X->cellvertices(), m_X->delaunayvertices(), rawPtr_cell_centroids);
	//basic_colorpatterns_kernel_3D << <nb, nt >> > (m_X->texObject(), m_X->colorpatterns(), rawPtr_cell_centroids);
	compute_colorpatterns_kernel_3D << <BlocksInGrid, ThreadsPerBlock >> > (m_X->texObject(), m_X->colorpatterns(), m_X->cellvertices(), m_X->delaunayvertices(), rawPtr_cell_centroids);
	printf("Done.\n");

	printf("Similarity computation...\n");
	compute_facetcentroids_kernel_3D << <BlocksInGrid, ThreadsPerBlock >> > (m_X->xadj(), m_X->adjncy(), m_X->cellvertices(), m_X->delaunayvertices(), rawPtr_facet_centroids);
	compute_similarity_kernel_3D << <BlocksInGrid, ThreadsPerBlock >> >(m_X->xadj(), m_X->adjncy(), m_X->texObject(), rawPtr_cell_centroids, rawPtr_facet_centroids, m_X->colorpatterns(), m_X->similarities());
	printf("Done.\n");
}

template<class X>
void CCudaXGraphSegmentator<X>::segmentate()
{
	int repeat = 30;
	//float threshold = 0.02f;
	thrust::sequence(thrust::cuda::par, m_X->labels(), m_X->labels() + v);
	for (int i = 0; i < repeat; ++i) {
		for (int c = 0; c < num_colors; ++c)
			segmentation_NCG_kernel << <BlocksInGrid, ThreadsPerBlock >> >(c, m_X->xadj(), m_X->adjncy(), m_X->updatepatterns(), m_X->similarities(), m_X->labels(), threshold);
		path_compression_kernel << <BlocksInGrid, ThreadsPerBlock >> >(m_X->labels());
	}
}

template<class X>
void CCudaXGraphSegmentator<X>::segmentate2()
{
	///* CELL VECTORS */
	//thrust::device_vector<float> cell_sizes(v);
	//thrust::device_vector<float> cell_gammas(v);
	//thrust::device_vector<int> cell_sortedlabels(m_X->labels(), m_X->labels() + v);
	//thrust::device_vector<float> cell_colorpatterns(m_X->colorpatterns(), m_X->colorpatterns() + v);
	//thrust::device_vector<int> cell_offsets(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(v));

	//compute_gammas_kernel << <nb, nt >> >(m_X->xadj(), m_X->adjncy(), m_X->cellvertices(), m_X->delaunayvertices(), thrust::raw_pointer_cast(cell_sizes.data()), thrust::raw_pointer_cast(cell_gammas.data()));

	///* SORT CELL VECTORS BY KEY (KEY = CELL LABEL) */

	//thrust::sort_by_key(cell_sortedlabels.begin(), cell_sortedlabels.end(),
	//	thrust::make_zip_iterator(thrust::make_tuple(cell_sizes.begin(), cell_gammas.begin(), cell_offsets.begin(), cell_colorpatterns.begin())));

	///* GET NUMBER OF REGIONS */
	//thrust::device_vector<int> region_labels(v);
	//thrust::device_vector<int>::iterator uniquelabels_end = thrust::unique_copy(cell_sortedlabels.begin(), cell_sortedlabels.end(), region_labels.begin());
	//num_regions = uniquelabels_end - region_labels.begin();

	///* RESIZE OUTPUT VECTORS */

	//region_labels.resize(num_regions);
	//thrust::device_vector<float>    region_sizes(num_regions);
	//thrust::device_vector<float>    region_gammas(num_regions);
	//thrust::device_vector<float>  region_colorpatterns(num_regions);
	//thrust::device_vector<int>    region_offsets(num_regions);
	//
	// //// print
	// //printf("number of cells = %i\n",v);
	// //thrust::host_vector<int> h01(cell_sortedlabels);
	// //thrust::host_vector<float> h02(cell_colorpatterns);
	// //for(int i = 0; i < v; i++) printf("%i ", h01[i]); printf("\n");
	// //for(int i = 0; i < v; i++) printf("%.2f ", h02[i]); printf("\n");

	// /* GROUP REDUCTION: (Tuple = {int,int,float}) */
	// thrust::reduce_by_key(cell_sortedlabels.begin(), cell_sortedlabels.end(),
	//	 thrust::make_zip_iterator(thrust::make_tuple(cell_gammas.begin(), cell_sizes.begin(), cell_colorpatterns.begin())),
	//	 region_labels.begin(),
	//	 thrust::make_zip_iterator(thrust::make_tuple(region_gammas.begin(), region_sizes.begin(), region_colorpatterns.begin())),
	//	 thrust::equal_to<int>(),
	//	 TuplePlus<thrust::tuple<int, int, float>>());

	// cell_sizes.clear();
	// cell_gammas.clear();
	// cell_sortedlabels.clear();
	// cell_colorpatterns.clear();

	// /* REGION OFFSETS */
	// thrust::exclusive_scan(region_numcells.begin(), region_numcells.end(), region_offsets.begin());

	// /* SORT REGIONS DECREMENTALLY BY GAMMA */
	// thrust::sort_by_key(region_gammas.begin(), region_gammas.end(),
	//	 thrust::make_zip_iterator(thrust::make_tuple(region_labels.begin(), region_sizes.begin(), region_colorpatterns.begin(), region_offsets.begin())),
	//	 thrust::greater<int>());

	// /** SEGMENTATION **/

	// /* RELABEL FROM 0 */
	// int *dPtr_cell_offsets, *dPtr_region_numcells, *dPtr_region_offsets, *dPtr_region_labels;
	// dPtr_cell_offsets = thrust::raw_pointer_cast(cell_offsets.data());
	// dPtr_region_labels = thrust::raw_pointer_cast(region_labels.data());
	// dPtr_region_numcells = thrust::raw_pointer_cast(region_numcells.data());
	// dPtr_region_offsets = thrust::raw_pointer_cast(region_offsets.data());

	// thrust::copy(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(num_regions), region_labels.begin());
	// relabel_kernel<G, K> << <num_regions, min(region_numcells[0], 1024) >> >(dPtr_cell_offsets, dPtr_region_numcells, dPtr_region_offsets, dPtr_labels, dPtr_region_labels);


	// // print                                            
	// printf("Number of regions = %i\n", num_regions);
	// thrust::host_vector<int> h1(region_labels);
	// thrust::host_vector<float> h2(region_sizes);
	// thrust::host_vector<float> h3(region_gammas);
	// thrust::host_vector<float> h4(region_colorpatterns);
	// for (int i = 0; i < num_regions; i++) printf("%i ", h1[i]); printf("\n");
	// for (int i = 0; i < num_regions; i++) printf("%.2f ", h2[i]); printf("\n");
	// for (int i = 0; i < num_regions; i++) printf("%.2f ", h3[i]); printf("\n");
	// for (int i = 0; i < num_regions; i++) printf("%.2f ", h4[i]); printf("\n");
}

////////////////////////////////////////////////
//template class CCudaXGraphSegmentator< CCudaXGraph< CDelaunay_2D_Cuda_XGraph_Adaptor > >;
template class CCudaXGraphSegmentator< CCudaXGraph< CDelaunay_3D_Cuda_XGraph_Adaptor > >;
////////////////////////////////////////////////