#include "Cuda_XGraph_Segmentator.cuh"
#include "Cuda_XGraph_Segmentator_Kernels.cuh"

/////////////////////////////

template<class X>
inline int CCudaXGraphSegmentator<X>::nextpowerof2(int v)
{
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}

template<class X>
CCudaXGraphSegmentator<X>::CCudaXGraphSegmentator(X& x) : m_X(x), m_graphcolorizer(x), m_similaritiescalculator(x)
{
	std::cout << "Op.Seg#1 : Num.Repeats = \n";
	std::cin >> num_repeat_seg1;
	std::cout << "Op.Seg#2 : Target #regions & Num.Repeats = \n";
	std::cin >> nR_2 >> num_repeat_seg2;
	std::cout << "Op.Seg#3 : Target #regions = \n";
	std::cin >> nR_3;

	printf("Copying constant symbols...\n");

	cudaMemcpyToSymbol(V, &(m_X->num_cells()), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(E, &(m_X->num_edges()), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(DV, &(m_X->num_points()), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Img_width, &(m_X->image_dimensions().x), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Img_height, &(m_X->image_dimensions().y), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Img_depth, &(m_X->image_dimensions().z), sizeof(int), 0, cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol(Img_XYarea, &(m_X->image_xyarea()), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Vol, &(m_X->image_vol()), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Dim, &(XSpace::Dim), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(NCellVertices, &(XSpace::NCellVertices), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(NCellNeighbors, &(XSpace::NCellNeighbors), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(NFacetVertices, &(XSpace::NFacetVertices), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(NGraphColors, &(XSpace::NGraphColors), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Num_isovalues, &(m_similaritiescalculator.num_isovalues()), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Isovalues, m_similaritiescalculator.isovalues_data(), sizeof(IsovalueType)*m_similaritiescalculator.num_isovalues(), 0, cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol(Threshold, &threshold, sizeof(int), 0, cudaMemcpyHostToDevice);

	int imageneighborsteps_h[26];
	// Z
	imageneighborsteps_h[0] = 1;
	imageneighborsteps_h[1] = 1 + m_X->image_dimensions().x;
	imageneighborsteps_h[2] = m_X->image_dimensions().x;
	imageneighborsteps_h[3] = -1 + m_X->image_dimensions().x;
	imageneighborsteps_h[4] = -1;
	imageneighborsteps_h[5] = -1 - m_X->image_dimensions().x;
	imageneighborsteps_h[6] = -m_X->image_dimensions().x;
	imageneighborsteps_h[7] = 1 - m_X->image_dimensions().x;

	// Z - 1
	imageneighborsteps_h[8] = -m_X->image_dimensions().x*m_X->image_dimensions().y;
	imageneighborsteps_h[9] = imageneighborsteps_h[8] + imageneighborsteps_h[0];
	imageneighborsteps_h[10] = imageneighborsteps_h[8] + imageneighborsteps_h[1];
	imageneighborsteps_h[11] = imageneighborsteps_h[8] + imageneighborsteps_h[2];
	imageneighborsteps_h[12] = imageneighborsteps_h[8] + imageneighborsteps_h[3];
	imageneighborsteps_h[13] = imageneighborsteps_h[8] + imageneighborsteps_h[4];
	imageneighborsteps_h[14] = imageneighborsteps_h[8] + imageneighborsteps_h[5];
	imageneighborsteps_h[15] = imageneighborsteps_h[8] + imageneighborsteps_h[6];
	imageneighborsteps_h[16] = imageneighborsteps_h[8] + imageneighborsteps_h[7];

	// Z + 1
	imageneighborsteps_h[17] = m_X->image_dimensions().x*m_X->image_dimensions().y;
	imageneighborsteps_h[18] = imageneighborsteps_h[17] + imageneighborsteps_h[0];
	imageneighborsteps_h[19] = imageneighborsteps_h[17] + imageneighborsteps_h[1];
	imageneighborsteps_h[20] = imageneighborsteps_h[17] + imageneighborsteps_h[2];
	imageneighborsteps_h[21] = imageneighborsteps_h[17] + imageneighborsteps_h[3];
	imageneighborsteps_h[22] = imageneighborsteps_h[17] + imageneighborsteps_h[4];
	imageneighborsteps_h[23] = imageneighborsteps_h[17] + imageneighborsteps_h[5];
	imageneighborsteps_h[24] = imageneighborsteps_h[17] + imageneighborsteps_h[6];
	imageneighborsteps_h[25] = imageneighborsteps_h[17] + imageneighborsteps_h[7];
	
	cudaMemcpyToSymbol(ImageNeighborSteps, imageneighborsteps_h, sizeof(int) * 26, 0, cudaMemcpyHostToDevice);

	ThreadsPerBlock = dim3(MAX_THREADS_PER_BLOCK, 1, 1);
	BlocksInGrid = dim3(min((m_X->num_cells() + ThreadsPerBlock.x - 1) / ThreadsPerBlock.x, 65535), 1, 1);

	m_graphcolorizer.colorize();
	m_similaritiescalculator.compute_similarities();
	region_labels.resize(m_X->num_cells());

	printf("Done.\n");
}

template<class X>
void CCudaXGraphSegmentator<X>::segmentate()
{
	cudaDeviceSynchronize(); printf("************* SEGMENTATION: START! *************\n");
	float ms1;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	/* PERFORM SEGMENTATION */
	/* SORT CELLS BY UPDATEPATTERN */
	/* USE FILTERS INSTEAD OF BITONIC SORT (WAY FASTER!) */

	thrust::device_vector<int> updatepattern_bounds(XSpace::NGraphColors + 1, 0);
	thrust::device_vector<int> upd_count_helper(1);

	int* dp_updatepattern_bounds = thrust::raw_pointer_cast(updatepattern_bounds.data());
	int* dp_upd_count_helper = thrust::raw_pointer_cast(upd_count_helper.data());

	for (int c = 0; c < XSpace::NGraphColors; ++c){
		filter_k_c << <BlocksInGrid, ThreadsPerBlock >> >(c, m_X->sortedcells(), m_X->labels(), m_X->num_cells(), dp_upd_count_helper);
		updatepattern_bounds[c + 1] = upd_count_helper[0];
	}
	
	///* PERFORM NCG ITERATIONS */
	thrust::sequence(thrust::cuda::par, m_X->labels(), m_X->labels() + m_X->num_cells());
	for (int i = 0; i < num_repeat_seg1; ++i) {
		for (int c = 0; c < XSpace::NGraphColors; ++c) {
			//printf("---------------------------------------------------------\n");
			segmentation_NCG_kernel3 <XSpace> << < ( updatepattern_bounds[c+1] - updatepattern_bounds[c] + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (c, dp_updatepattern_bounds, m_X->sortedcells(), m_X->neighbors(), m_X->similarities(), m_X->labels());
			//cudaDeviceSynchronize();
		}
			//segmentation_NCG_kernel3 <XSpace> << < ((int)updatepattern_sizes[c] + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (c, dp_updatepattern_bounds, m_X->sortedcells(), m_X->neighbors(), m_X->similarities(), m_X->labels());
			//segmentation_NCG_kernel4 <XSpace> << < (int)updatepattern_sizes[c], WARP_SIZE >> > (c, dp_updatepattern_bounds, m_X->sortedcells(), m_X->neighbors(), m_X->similarities(), m_X->labels());
	//	printf("Finished iteration #%i\n", i);
	}
	
	////////////////////////////////////////////////

	/* OBTAIN REGION LABELS */
	thrust::device_vector<int> region_count_helper(1);
	int* dp_region_count_helper = thrust::raw_pointer_cast(region_count_helper.data());
	int* dp_region_labels = thrust::raw_pointer_cast(region_labels.data());
	
	filter_k << <BlocksInGrid, ThreadsPerBlock >> >(dp_region_labels, m_X->labels(), m_X->num_cells(), dp_region_count_helper);

	num_regions = region_count_helper[0];
	region_labels.resize(num_regions);
	region_numcells.resize(num_regions);
	region_sizes.resize(num_regions);
	region_colorpatterns.resize(num_regions);
	dp_region_labels = thrust::raw_pointer_cast(region_labels.data());
	int*	dp_region_numcells = thrust::raw_pointer_cast(region_numcells.data());
	float*	dp_region_sizes = thrust::raw_pointer_cast(region_sizes.data());
	float*	dp_region_colorpatterns = thrust::raw_pointer_cast(region_colorpatterns.data());

	/* Sort region labels: Bitonic Sort*/
	for (int k = 2; k <= nextpowerof2(num_regions); k <<= 1) {
		for (int j = k >> 1; j>0; j = j >> 1) {
			bitonic_sort_step<int> << <(num_regions + ThreadsPerBlock.x - 1) / ThreadsPerBlock.x, ThreadsPerBlock >> >(dp_region_labels, num_regions, j, k);
		}
	}
	
	region_bounds.resize(num_regions + 1, 0);
	region_bounds[num_regions] = m_X->num_cells();
	int* dp_region_bounds = thrust::raw_pointer_cast(region_bounds.data());
	
	thrust::device_vector<int> rbound_count_helper(1);
	int* dp_rbound_count_helper = thrust::raw_pointer_cast(rbound_count_helper.data());

	filter_k_c << <BlocksInGrid, ThreadsPerBlock >> >(0, m_X->sortedcells(), m_X->labels(), m_X->num_cells(), dp_rbound_count_helper);
	region_bounds[1] = rbound_count_helper[0];
	filter_k_not_c << <BlocksInGrid, ThreadsPerBlock >> >(0, m_X->sortedcells(), m_X->labels(), m_X->num_cells(), dp_rbound_count_helper);

	
	/* Bitonic Sort */
	for (int k = 2; k <= nextpowerof2(m_X->num_cells() - region_bounds[1]); k <<= 1) {
		for (int j = k >> 1; j>0; j = j >> 1) {
			bitonic_sort_by_composition_key_step<int, int> << <BlocksInGrid, ThreadsPerBlock >> >(m_X->labels(), m_X->sortedcells() + region_bounds[1], m_X->num_cells() - region_bounds[1], j, k);
		}
	}
	
	
	///* SORT CELLS BY REGION */
	//thrust::sequence(thrust::cuda::par, m_X->sortedcells(), m_X->sortedcells() + m_X->num_cells());
	///* Bitonic Sort */
	//for (int k = 2; k <= nextpowerof2(m_X->num_cells()); k <<= 1) {
	//	for (int j = k >> 1; j>0; j = j >> 1) {
	//		bitonic_sort_by_composition_key_step<int, int> << <BlocksInGrid, ThreadsPerBlock >> >(m_X->labels(), m_X->sortedcells(), m_X->num_cells(), j, k);
	//	}
	//}
	/* OBTAIN REGION BOUNDS (For reduction of every region in parallel) */
	//region_bounds.resize(num_regions + 1, 0);
	//region_bounds[num_regions] = m_X->num_cells();
	//int* dp_region_bounds = thrust::raw_pointer_cast(region_bounds.data());

	bounds_kernel << <BlocksInGrid, ThreadsPerBlock >> >(dp_region_bounds, m_X->sortedcells() + 1, m_X->labels(), dp_region_labels, num_regions);
	inv_prefixsum_kernel << <(num_regions + ThreadsPerBlock.x - 1) / ThreadsPerBlock.x, ThreadsPerBlock >> >(dp_region_numcells, dp_region_bounds, num_regions);
	
	/* Relabel : The next steps are easier if region labels are contiguous */
	relabel_kernel_3 << <num_regions, ThreadsPerBlock >> >(dp_region_bounds, dp_region_labels, m_X->labels(), m_X->sortedcells());
	thrust::sequence(thrust::device, region_labels.begin(), region_labels.end());

	///* REDUCE REGIONS */
	cell_reduction_kernel <BaseClass><< <num_regions, 1024 >> >
		(dp_region_bounds, m_X->sortedcells(), m_X->sizes(), m_X->colorpatterns(),
		/*dp_region_numcells,*/ dp_region_sizes, dp_region_colorpatterns);

	
	/* SORT REGIONS DECREMENTALLY BY GAMMA */
	thrust::device_vector<float> region_gammas(num_regions);
	float *dp_region_gammas = thrust::raw_pointer_cast(region_gammas.data());
	
	compute_gammas_kernel << <(num_regions + ThreadsPerBlock.x - 1) / ThreadsPerBlock.x, ThreadsPerBlock >> >(dp_region_numcells, dp_region_sizes, dp_region_gammas, num_regions);


	/* Quadruple Bitonic Sort by Key : To move a region along with its properties */
	for (int k = 2; k <= nextpowerof2(num_regions); k <<= 1) {
		for (int j = k >> 1; j>0; j = j >> 1) {
			bitonic_quadruple_sort_by_key_step<float, int, int, float, float><<<(num_regions + ThreadsPerBlock.x - 1) / ThreadsPerBlock.x, ThreadsPerBlock >>>
				(dp_region_gammas, dp_region_labels, dp_region_numcells, dp_region_sizes, dp_region_colorpatterns, num_regions, j, k);
		}
	}
	
	region_boundlabels = region_labels;
	relabel_kernel_4 << <num_regions, ThreadsPerBlock >> >(dp_region_bounds, dp_region_labels, m_X->labels(), m_X->sortedcells());
	thrust::sequence(thrust::device, region_labels.begin(), region_labels.end());

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms1, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceSynchronize(); printf("Segmentation Operator #1: %i regions (%.2f ms)\n", num_regions, ms1);

	///* OBTAIN NEW REGION BOUNDS : for the next segmentation operator */

	

	//for (int i = 0; i < num_regions; ++i) region_bounds[i + 1] = region_bounds[i] + region_numcells[i];

	///* Bitonic Sort */
	//for (int k = 2; k <= nextpowerof2(m_X->num_cells()); k <<= 1) {
	//	for (int j = k >> 1; j>0; j = j >> 1) {
	//		bitonic_sort_by_composition_key_step<int, int> << <BlocksInGrid, ThreadsPerBlock >> >(m_X->labels(), m_X->sortedcells(), m_X->num_cells(), j, k);
	//	}
	//}

	//bounds_kernel << <BlocksInGrid, ThreadsPerBlock >> >(dp_region_bounds, m_X->sortedcells() + 1, m_X->labels(), dp_region_labels, num_regions);
	
}

template<class X>
void CCudaXGraphSegmentator<X>::segmentate2()
{
	/* SEGMENTATION OPERATOR #2 */
	if (num_regions <= nR_2) return;
	printf("About to segmentate2, #Current Regions = %i, #Target Regions = %i\n", num_regions, nR_2);
	
	float ms1;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	int* dp_sorted_cells = m_X->sortedcells();
	int* dp_region_boundlabels = thrust::raw_pointer_cast(region_boundlabels.data());
	int* dp_region_bounds = thrust::raw_pointer_cast(region_bounds.data());
	int *dp_region_labels = thrust::raw_pointer_cast(region_labels.data());
	float *dp_region_colorpatterns = thrust::raw_pointer_cast(region_colorpatterns.data());
	
	m_similaritiescalculator.reset_similarities();
	cudaDeviceSynchronize();

	for (int k = 0; k < num_repeat_seg2; ++k) 
		segmentation_NRG_kernel <XSpace> << <num_regions - nR_2, ThreadsPerBlock.x/2, nR_2*sizeof(float) >> > (nR_2, dp_region_bounds, dp_region_boundlabels, dp_sorted_cells, m_X->labels(), m_X->neighbors(), m_X->similarities(), dp_region_colorpatterns, dp_region_labels);
	relabel_kernel_2 << <BlocksInGrid, ThreadsPerBlock >> >(m_X->labels(), dp_region_labels);
	num_regions = nR_2;
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms1, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceSynchronize(); printf("Segmentation Operator #2: %i regions (%.2f ms)\n", num_regions, ms1);
	
}

template<class X>
void CCudaXGraphSegmentator<X>::segmentate3()
{
	/* SEGMENTATION OPERATOR #3 */
	if (num_regions <= nR_3) return;
	printf("About to segmentate3, nR = %i\n", nR_3);
	
	float ms1;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	thrust::device_vector<int> region_keeper(region_labels);
	thrust::device_vector<float> all_similarities;
	thrust::device_vector<int> all_similarities_coords;

	int *dp_region_keeper = thrust::raw_pointer_cast(region_keeper.data());
	int *dp_region_labels = thrust::raw_pointer_cast(region_labels.data());
	float *dp_region_colorpatterns = thrust::raw_pointer_cast(region_colorpatterns.data());
	float *dp_all_similarities;
	int *dp_all_similarities_coords;
	int nR = nR_2; 
	int nR_sq;

	//int count = 0;
	do {
		//printf("Segmentate3, iteration #%i\n", count++);
		nR_sq = nR*nR;
		all_similarities.resize(nR_sq);
		all_similarities_coords.resize(nR_sq);
		
		dp_all_similarities = thrust::raw_pointer_cast(all_similarities.data());
		dp_all_similarities_coords = thrust::raw_pointer_cast(all_similarities_coords.data());

		all_similarities_kernel << <BlocksInGrid, ThreadsPerBlock >> >(nR, dp_region_labels, dp_region_colorpatterns, dp_all_similarities, dp_all_similarities_coords);

		/* BITONIC SORT Major step */
		for (int k = 2; k <= nextpowerof2(nR_sq); k <<= 1) {
			/* Minor step */
			for (int j = k >> 1; j>0; j = j >> 1) {
				bitonic_sort_by_key_step<float, int> << <(nR*nR + ThreadsPerBlock.x - 1) / ThreadsPerBlock.x, ThreadsPerBlock >> >(dp_all_similarities, dp_all_similarities_coords, nR_sq, j, k);
			}
		}

		int position = all_similarities_coords[0];
		int x = position % nR;
		int y = position / nR;

		int a = min(x, y);
		int b = max(x, y);
		//printf("a = %i, b = %i\n", a, b);

		region_colorpatterns[a] = (region_colorpatterns[a] * region_sizes[a] + region_colorpatterns[b] * region_sizes[b]) / (region_sizes[a] + region_sizes[b]);
		region_sizes[a] += region_sizes[b];

		thrust::replace(thrust::device, region_keeper.begin(), region_keeper.end(), region_labels[b], region_labels[a]);

		region_labels.erase(region_labels.begin() + b);
		region_sizes.erase(region_sizes.begin() + b);
		region_colorpatterns.erase(region_colorpatterns.begin() + b);
		
		nR--;
	} while (nR > nR_3);

	relabel_kernel_1 <<< nR_2, 1 >>>(nR_2, nR_3, dp_region_keeper, dp_region_labels);
	relabel_kernel_2 <<<BlocksInGrid, ThreadsPerBlock >>>(m_X->labels(), dp_region_keeper);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms1, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceSynchronize(); printf("Segmentation Operator #3: %i regions (%.2f ms)\n", nR, ms1);
	
}

//////////////////////////////////////////////////
template class CCudaXGraphSegmentator< CCudaXGraph< CDelaunay_2D_Cuda_XGraph_Adaptor > >;
template class CCudaXGraphSegmentator< CCudaXGraph< CDelaunay_3D_Cuda_XGraph_Adaptor > >;
template class CCudaXGraphSegmentator< CCudaXGraph< CImage_2D_Cuda_XGraph_Adaptor	 > >;
template class CCudaXGraphSegmentator< CCudaXGraph< CImage_3D_Cuda_XGraph_Adaptor	 > >;
//////////////////////////////////////////////////
