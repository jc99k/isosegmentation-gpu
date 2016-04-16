#include <curand_kernel.h>
#include "cudagraphsegmentator.cuh"
#include "thrust_custom_predicates.h"
#include "kernels.h"

template<class XSpace>
CCudaCSRGraphSegmentator<XSpace>::CCudaCSRGraphSegmentator() : v(m_csrGraph.num_vertices()), e(m_csrGraph.num_edges()), dv(m_csrGraph.num_delaunayvertices()), num_isovalues(m_isovaluelist.size())
{
    cudaCSRGraphReader_t csrReader(m_csrGraph);
	csrReader.load_file();

    printf("%i %i %i\n", v,e,dv);

	cudaImageReader_t imgReader(d_imagedata, m_texObject);
	imgReader.load_file();

	m_isovaluelist.load_values();

	cudaMemcpyToSymbol(V, &v, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(E, &e, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(DV, &dv, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Num_isovalues, &num_isovalues, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Isovalues, &m_isovaluelist[0], sizeof(float)*num_isovalues, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Img_width, &(imgReader.image_extent().x), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Img_height, &(imgReader.image_extent().y), sizeof(int), 0, cudaMemcpyHostToDevice);
	nt = dim3(1024, 1, 1);
	nb = dim3(min((v + nt.x - 1) / nt.x, 65535), 1, 1);

    printf("%i %i\n", nb.x,nt.x);
}

template<class XSpace>
CCudaCSRGraphSegmentator<XSpace>::~CCudaCSRGraphSegmentator()
{
	cudaDestroyTextureObject(m_texObject);
	cudaFree(d_imagedata);
}

template<class XSpace>
void CCudaCSRGraphSegmentator<XSpace>::colorize()
{
	printf("Colorize...\n");
    // device storage for the random numbers
	thrust::device_vector<float> randoms(v);

    printf("Generate randoms...\n");

	// generate randoms
	rng_generate_kernel << <nb, nt >> >(thrust::raw_pointer_cast(randoms.data()));
	thrust::fill(thrust::cuda::par, m_csrGraph.updatepatterns(), m_csrGraph.updatepatterns() + v, -1);

    printf("Iterate...\n");

	for (int c = 0; c < v; ++c) {
		color_jpl_kernel << <nb, nt >> >(c, m_csrGraph.xadj(), m_csrGraph.adjncy(), thrust::raw_pointer_cast(randoms.data()), m_csrGraph.updatepatterns());
		int left = thrust::count(thrust::cuda::par, m_csrGraph.updatepatterns(), m_csrGraph.updatepatterns() + v, -1);
		if (left == 0)
		{
			num_colors = c + 1;
			break;
		}
	}

    printf("Colorized!\n");
}

template<class XSpace>
void CCudaCSRGraphSegmentator<XSpace>::compute_colorpatterns()
{
	thrust::device_vector<float2> cell_centroids(v);
    printf("v = %i\n", v);
	compute_centroids_kernel << <nb, nt >> >(m_csrGraph.xadj(), m_csrGraph.adjncy(), m_csrGraph.cellvertices(), m_csrGraph.delaunayvertices(), thrust::raw_pointer_cast(cell_centroids.data()));
	basic_colorpatterns_kernel << <nb, nt >> > (m_texObject, m_csrGraph.colorpatterns(), thrust::raw_pointer_cast(cell_centroids.data()));
}

template<class XSpace>
void CCudaCSRGraphSegmentator<XSpace>::compute_similarities()
{
	// Calcular facet_centroids
	//thrust::device_vector<float2> facet_centroids(e * 2);


	// Lanzar V blocks, con 3 threads c/u
	// Paralelismo dinamico

	


	//compute_similarity_kernel<< <nb, nt >> >(dPtr_xadj, dPtr_adjncy, texObj, dPtr_cell_centroids, dPtr_facet_centroids, dPtr_colorpatterns, dPtr_similarities);
}

template<class XSpace>
void CCudaCSRGraphSegmentator<XSpace>::write()
{
    printf("Create writer");
    cudaCSRGraphWriter_t csrWriter(m_csrGraph);
    printf("Save file");
    csrWriter.save_file();
    printf("Saved");
}

//////////////////////////////////////////////

template class CCudaCSRGraphSegmentator<int>;

//////////////////////////////////////////////