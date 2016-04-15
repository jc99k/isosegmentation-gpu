#include "thrust_custom_predicates.h"
#include "kernel.h"
#include "cudatemplate.h"
#include <stdio.h>
#include <stdlib.h>

///////////////////////////////////////////////////////

template<class G, kernel K>
void CCudaGraphSegmentator<G,K>::Transform()
{
    // thrust::transform(thrust::cuda::par, xadj, xadj + (100));
}

template<class G, kernel K>
inline void CCudaGraphSegmentator<G,K>::Synchronize()
{
    cudaDeviceSynchronize();
}

/* SEND DATA TO GPU */
template<class G, kernel K>
void CCudaGraphSegmentator<G,K>::SendDataToGPU(int h_v, int h_e, int h_dv, 
    int *xadj, int *adjncy, int *coloring, int *labels, 
    float *colorpatterns, float *cell_centroids, float *facet_centroids, float *similarities, 
    int *cell_vertices, float *delaunay_vertices, float *isovalues, int num_isovalues)
{
    hPtr_xadj               = xadj;
    hPtr_adjncy             = adjncy;
    hPtr_coloring           = coloring;
    hPtr_labels             = labels;
    hPtr_colorpatterns      = colorpatterns;
    hPtr_cell_centroids     = cell_centroids;
    hPtr_facet_centroids    = facet_centroids;
    hPtr_similarities       = similarities;
    hPtr_cell_vertices      = cell_vertices;
    hPtr_delaunay_vertices  = delaunay_vertices;

    v = h_v;
    e = h_e;
    dv = h_dv;
    cudaMemcpyToSymbol(V,&v,sizeof(int),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(E,&e,sizeof(int),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(DV,&dv,sizeof(int),0,cudaMemcpyHostToDevice);

    nt = dim3(1024,1,1);
    nb = dim3(min((v + nt.x - 1)/nt.x,65535),1,1);

    repeat = 30, threshold = 0.99;

    dVector_xadj.resize(v+1);
    dVector_adjncy.resize(e*2);
    dVector_coloring.resize(v);
    dVector_labels.resize(v);
    dVector_randoms.resize(v);
    dVector_cell_sizes.resize(v);
    dVector_colorpatterns.resize(v);
    dVector_similarities.resize(e*2);
    dVector_curandStates.resize(v);
    dVector_cell_centroids.resize(v);
    dVector_facet_centroids.resize(e*2);
    dVector_cell_vertices.resize(v);
    dVector_delaunay_vertices.resize(dv);

    dPtr_xadj = thrust::raw_pointer_cast(dVector_xadj.data());
    dPtr_adjncy = thrust::raw_pointer_cast(dVector_adjncy.data());
    dPtr_coloring = thrust::raw_pointer_cast(dVector_coloring.data());
    dPtr_labels = thrust::raw_pointer_cast(dVector_labels.data());
    dPtr_randoms = thrust::raw_pointer_cast(dVector_randoms.data());
    dPtr_cell_sizes = thrust::raw_pointer_cast(dVector_cell_sizes.data());
    dPtr_colorpatterns = thrust::raw_pointer_cast(dVector_colorpatterns.data());
    dPtr_similarities = thrust::raw_pointer_cast(dVector_similarities.data());
    dPtr_curandStates = thrust::raw_pointer_cast(dVector_curandStates.data());
    dPtr_cell_centroids = thrust::raw_pointer_cast(dVector_cell_centroids.data());
    dPtr_facet_centroids = thrust::raw_pointer_cast(dVector_facet_centroids.data());
    dPtr_cell_vertices = thrust::raw_pointer_cast(dVector_cell_vertices.data());
    dPtr_delaunay_vertices = thrust::raw_pointer_cast(dVector_delaunay_vertices.data());
    
    thrust::copy(hPtr_xadj      ,hPtr_xadj + v+1        ,dVector_xadj.begin());
    thrust::copy(hPtr_adjncy    ,hPtr_adjncy + e*2      ,dVector_adjncy.begin());
    thrust::copy(hPtr_colorpatterns ,hPtr_colorpatterns + v   ,dVector_colorpatterns.begin());
    cudaMemcpy(dPtr_cell_vertices, hPtr_cell_vertices, v*sizeof(int)*3, cudaMemcpyHostToDevice);
    cudaMemcpy(dPtr_delaunay_vertices, hPtr_delaunay_vertices, dv*sizeof(float)*2, cudaMemcpyHostToDevice);
    
    cudaMemcpy(dPtr_cell_centroids, hPtr_cell_centroids, v*sizeof(float)*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dPtr_facet_centroids, hPtr_facet_centroids, e*2*sizeof(float)*2, cudaMemcpyHostToDevice);

    
    thrust::sequence(dVector_labels.begin(), dVector_labels.end());
    thrust::fill(dVector_coloring.begin(), dVector_coloring.end(), -1);
    rng_setup_kernel<<<v,1>>>(time(NULL), dPtr_curandStates);     /* Setup prng states */


    cudaMemcpyToSymbol(Num_isovalues,&num_isovalues,sizeof(int),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Isovalues,isovalues,sizeof(float)*num_isovalues,0,cudaMemcpyHostToDevice);

}

/* SEND IMAGE TO GPU */
template<class G, kernel K>
void CCudaGraphSegmentator<G,K>::SendImageToGPU(float *h_imagedata, int h_imgw, int h_imgh, int h_imgc)
{
    hPtr_image_data = h_imagedata;
    image_width = h_imgw;
    image_height = h_imgh;
    image_channels = h_imgc;
    widthInBytes = image_width*sizeof(float);
    image_dimension = image_width*image_height;
    
    cudaMemcpyToSymbol(Img_width,&image_width,sizeof(int),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Img_height,&image_height,sizeof(int),0,cudaMemcpyHostToDevice);

    cudaMallocPitch(&dPtr_image_data,&pitch,widthInBytes, image_height);
    cudaMemcpy2D(dPtr_image_data,pitch,hPtr_image_data,widthInBytes,widthInBytes,image_height,cudaMemcpyHostToDevice);

    printf("(Image sent.)\n");

    // Texture (new way) : Create texture object
    // 1. Resource Descriptor (resource to texture with)
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = dPtr_image_data;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    resDesc.res.pitch2D.width = image_width;
    resDesc.res.pitch2D.height = image_height;
    resDesc.res.pitch2D.pitchInBytes =  pitch;

    // 2. Texture Descriptor (texturing settings)
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.filterMode = cudaFilterModeLinear;      // linear interpolation
    // texDesc.addressMode[0] = cudaAddressModeWrap;   // wrap texture coordinates
    // texDesc.addressMode[1] = cudaAddressModeWrap;
    // texDesc.addressMode[2] = cudaAddressModeWrap; // 3D

    // 3. Create texture object
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    printf("(Texture created.)\n");
}

/* GET BACK DATA FROM GPU */
template<class G, kernel K>
void CCudaGraphSegmentator<G,K>::GetDataFromGPU()
{
        //  /* TEST TEXTURE */
        // dim3 grid(image_width,image_height);
        // test_texture_kernel<<<grid,1>>>(dPtr_image_data, texObj);


    cudaMemcpy(hPtr_labels, dPtr_labels, v*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hPtr_coloring, dPtr_coloring, v*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hPtr_colorpatterns, dPtr_colorpatterns, v*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hPtr_similarities, dPtr_similarities, e*2*sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy2D(hPtr_image_data,widthInBytes,dPtr_image_data,pitch,widthInBytes,image_height,cudaMemcpyDeviceToHost);    

    printf("(Copied back info.)\n");
}

/* FREE GPU DATA */
template<class G, kernel K>
void CCudaGraphSegmentator<G,K>::FreeGPU()
{
    // cudaFree(xadj);
    cudaDestroyTextureObject(texObj);
    cudaFree(dPtr_image_data);
}



///////////////////////////////////////////////////////

/* COLORING ALGORITHM */
template<class G, kernel K>
void CCudaGraphSegmentator<G,K>::colorize()
{
    float ms1,ms2;
    cudaEvent_t start,stop,end;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    rng_generate_kernel<<<nb,nt>>>(dPtr_curandStates, dPtr_randoms);

    cudaEventRecord(stop, 0);    
    cudaEventSynchronize(stop);

	for(int c=0; c < v; c++) {
		color_jpl_kernel<<<nb,nt>>>(c, dPtr_xadj, dPtr_adjncy, dPtr_randoms, dPtr_coloring);
		int left = thrust::count(dVector_coloring.begin(), dVector_coloring.end(), -1);
		if (left == 0) 
		{
			num_colors = c+1;
			break;
		}
	}

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms1, start, stop);
    cudaEventElapsedTime(&ms2, stop, end);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(end);

    printf("RNG Kernel : %f, Colorization : %f\n",ms1,ms2);
}

/* COMPUTE CENTROIDS */
template<class G, kernel K>
void CCudaGraphSegmentator<G,K>::compute_centroids()
{
    // int nv = min((dv + nt.x - 1)/nt.x,65535);
    // flip_verticality_kernel<G,K> <<<nv,nt>>> (dPtr_delaunay_vertices);
    // compute_centroids_kernel<G,K> <<<nb,nt>>> (dPtr_xadj, dPtr_adjncy, dPtr_cell_vertices, dPtr_delaunay_vertices, dPtr_cell_centroids, dPtr_facet_centroids);
}

/* COMPUTE COLOR PATTERNS */
template<class G, kernel K>
void CCudaGraphSegmentator<G,K>::compute_colorpatterns()
{

/* PARALLEL SCAN 2D (Fix) */
/*    thrust::device_vector<int2>     minCoords(v);
    thrust::device_vector<int2>     maxCoords(v);
    thrust::device_vector<int>      minDims(v);
    thrust::device_vector<int>      maxDims(v);
    thrust::device_vector<int2>     spread_dir(v);
    thrust::device_vector<int2>     sweep_dir(v);
    thrust::device_vector<int>    pixelcells(image_width*image_height);
    dim3 img_nt(32,32);
    int globalmaxDim;

    get_boundingbox_kernel<G,K> <<<v,1>>>(texObj, dPtr_cell_vertices, dPtr_delaunay_vertices, 
        thrust::raw_pointer_cast(minCoords.data()), thrust::raw_pointer_cast(maxCoords.data()),
        thrust::raw_pointer_cast(minDims.data()), thrust::raw_pointer_cast(maxDims.data()),
        thrust::raw_pointer_cast(spread_dir.data()), thrust::raw_pointer_cast(sweep_dir.data()));

    pixelcell_map_kernel<G,K> <<<v,img_nt>>>(texObj, dPtr_cell_vertices, dPtr_delaunay_vertices, thrust::raw_pointer_cast(pixelcells.data()), thrust::raw_pointer_cast(minCoords.data()), thrust::raw_pointer_cast(maxCoords.data()));

    globalmaxDim = thrust::reduce(maxDims.begin(), maxDims.end(),-1, thrust::maximum<int>());
    printf("globalmaxDim = %i\n", globalmaxDim);
    thrust::device_vector<float> aggregate_colorpatterns(globalmaxDim*v);
    thrust::device_vector<int> aggregate_counts(globalmaxDim*v);
    trianglescan_kernel<G,K> <<<v,globalmaxDim,globalmaxDim*(sizeof(float)+sizeof(int))>>>(texObj,thrust::raw_pointer_cast(pixelcells.data()),
        thrust::raw_pointer_cast(minCoords.data()), thrust::raw_pointer_cast(minDims.data()),
        thrust::raw_pointer_cast(spread_dir.data()), thrust::raw_pointer_cast(sweep_dir.data()),
        thrust::raw_pointer_cast(aggregate_colorpatterns.data()), thrust::raw_pointer_cast(aggregate_counts.data())
        );

    reducescan_kernel<G,K> <<<v,1>>>(globalmaxDim, dPtr_colorpatterns,
        thrust::raw_pointer_cast(aggregate_colorpatterns.data()), thrust::raw_pointer_cast(aggregate_counts.data())
        );*/

// float ms1,ms2;
//     cudaEvent_t start,stop,end;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventCreate(&end);
//     cudaEventRecord(start, 0);

/* SEQUENTIAL SCAN */
    // int nv = min((dv + nt.x - 1)/nt.x,65535);
    // flip_verticality_kernel<G,K> <<<nv,nt>>> (dPtr_delaunay_vertices);

    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);

    basic_colorpatterns_kernel<G,K> <<<nb,nt>>> (texObj, dPtr_colorpatterns, dPtr_cell_centroids);
    // compute_colorpatterns_kernel<G,K> <<<nb,nt>>> (texObj, dPtr_cell_vertices, dPtr_delaunay_vertices, dPtr_colorpatterns, dPtr_cell_sizes, dPtr_cell_centroids);

        /* TEST TEXTURE */
        // dim3 grid(image_width,image_height);
        // test_texture_kernel<<<grid,1>>>(dPtr_image_data, texObj);

    //     cudaEventRecord(end, 0);
    // cudaEventSynchronize(end);
    // cudaEventElapsedTime(&ms1, start, stop);
    // cudaEventElapsedTime(&ms2, stop, end);
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    // cudaEventDestroy(end);
    // printf("Flip Kernel : %f, Comp.ColorPattern : %f\n",ms1,ms2);
}

template<class G, kernel K>
void CCudaGraphSegmentator<G,K>::compute_colorpatterns_tex()
{
    // thrust::device_vector<float2>     minCoords(v);
    // thrust::device_vector<float2>     diffCoords(v);

    // get_boundingbox_tex2D_kernel<G,K> <<<v,1>>>(texObj, dPtr_cell_vertices, dPtr_delaunay_vertices, 
    //     thrust::raw_pointer_cast(minCoords.data()), thrust::raw_pointer_cast(diffCoords.data()));

    // compute_colorpatterns_tex2D_kernel<G,K> <<<v,128,128*(sizeof(float)+sizeof(int))>>>(texObj, dPtr_cell_vertices, dPtr_delaunay_vertices, 
    //         thrust::raw_pointer_cast(minCoords.data()), thrust::raw_pointer_cast(diffCoords.data()));
};

/* COMPUTE SIMILARITIES */
template<class G, kernel K>
void CCudaGraphSegmentator<G,K>::compute_similarities()
{
    compute_similarity_kernel<G,K> <<<nb,nt>>>(dPtr_xadj, dPtr_adjncy, texObj, dPtr_cell_centroids, dPtr_facet_centroids, dPtr_colorpatterns, dPtr_similarities);
}

/* SEGMENTATION : 1st Operator */
template<class G, kernel K>
void CCudaGraphSegmentator<G,K>::segmentate()
{
    float ms1;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for(int i=0; i < repeat; i++) {
        for(int c=0; c < num_colors; c++)
            segmentation_NCG_kernel<G,K> <<<nb,nt>>>(c, dPtr_xadj, dPtr_adjncy, dPtr_coloring, dPtr_similarities, dPtr_labels, threshold);
            path_compression_kernel<G,K> <<<nb,nt>>>(dPtr_labels);
    }

    cudaEventRecord(stop, 0);    
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms1, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Segmentation1 : %f\n",ms1);
}

/* SEGMENTATION : 2nd Operator */
template<class G, kernel K>
void CCudaGraphSegmentator<G,K>::segmentate2()
{
    /* TYPEDEFs */

    typedef thrust::device_vector<int>::iterator   IntIterator;
    typedef typename thrust::tuple<int,int,float> Tuple;

    /* CELL VECTORS */

    thrust::device_vector<int> cell_sortedlabels(dVector_labels);
    thrust::device_vector<float> cell_colorpatterns(dVector_colorpatterns);
    thrust::device_vector<int> cell_offsets(thrust::counting_iterator<int>(0),thrust::counting_iterator<int>(v));

    /* REGION VECTORS */

    thrust::device_vector<int>    region_labels(v);
    thrust::device_vector<int>    region_numcells;
    thrust::device_vector<int>    region_sizes;
    thrust::device_vector<float>  region_colorpatterns;
    thrust::device_vector<int>    region_offsets;
    thrust::device_vector<float>  region_gammas;

    /* SORT CELL VECTORS BY KEY (KEY = CELL LABEL) */

    thrust::sort_by_key(cell_sortedlabels.begin(), cell_sortedlabels.end(), 
        thrust::make_zip_iterator(thrust::make_tuple(dVector_cell_sizes.begin(), cell_offsets.begin(), cell_colorpatterns.begin())));

    /* GET NUMBER OF REGIONS */

    thrust::device_vector<int>::iterator uniquelabels_end = thrust::unique_copy(cell_sortedlabels.begin(), cell_sortedlabels.end(), region_labels.begin());
    num_regions = uniquelabels_end - region_labels.begin();

    /* RESIZE OUTPUT VECTORS */

    region_labels.resize(num_regions);
    region_numcells.resize(num_regions);
    region_sizes.resize(num_regions);
    region_colorpatterns.resize(num_regions);
    region_offsets.resize(num_regions);
    region_gammas.resize(num_regions);

    // // print
    // printf("Number of cells = %i\n",v);
    // thrust::host_vector<int> h01(cell_sortedlabels);
    // thrust::host_vector<float> h02(cell_colorpatterns);
    // for(int i = 0; i < v; i++) printf("%i ", h01[i]); printf("\n");
    // for(int i = 0; i < v; i++) printf("%.2f ", h02[i]); printf("\n");

    /* GROUP REDUCTION: (Tuple = {int,int,float}) */
    thrust::reduce_by_key(cell_sortedlabels.begin(), cell_sortedlabels.end(),
        thrust::make_zip_iterator(thrust::make_tuple(thrust::constant_iterator<int>(1), dVector_cell_sizes.begin(), cell_colorpatterns.begin())),
        region_labels.begin(),
        thrust::make_zip_iterator(thrust::make_tuple(region_numcells.begin(), region_sizes.begin(), region_colorpatterns.begin())),
        thrust::equal_to<int>(),
        TuplePlus<Tuple>());

    /* REGION OFFSETS */
    thrust::exclusive_scan(region_numcells.begin(), region_numcells.end(), region_offsets.begin());

    //  /* GET REGIONS PROPORTIONAL SIZES */
    // thrust::transform(region_gammas.begin(), region_gammas.end(), region_sizes.begin(), region_colorpatterns.begin(), thrust::divides<float>());

    /* SORT REGIONS DECREMENTALLY BY SIZE */
    thrust::sort_by_key(region_sizes.begin(), region_sizes.end(), 
        thrust::make_zip_iterator(thrust::make_tuple(region_labels.begin(),region_numcells.begin(),region_colorpatterns.begin(), region_offsets.begin())), 
        thrust::greater<int>());

    /** SEGMENTATION **/

    /* RELABEL FROM 0 */
    int *dPtr_cell_offsets, *dPtr_region_numcells, *dPtr_region_offsets, *dPtr_region_labels;
    dPtr_cell_offsets = thrust::raw_pointer_cast(cell_offsets.data());
    dPtr_region_labels = thrust::raw_pointer_cast(region_labels.data());
    dPtr_region_numcells = thrust::raw_pointer_cast(region_numcells.data());
    dPtr_region_offsets = thrust::raw_pointer_cast(region_offsets.data());
    
    thrust::copy(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(num_regions), region_labels.begin());
    relabel_kernel<G,K> <<<num_regions,min(region_numcells[0],1024)>>>(dPtr_cell_offsets, dPtr_region_numcells, dPtr_region_offsets, dPtr_labels, dPtr_region_labels);

    /* SEGMENTATION OPERATOR #2 */
    int nR = 6;
    int nRt = region_numcells[nR];

    thrust::device_vector<float> shared_similarities;
    thrust::device_vector<int> min_element_labels;
    float *dPtr_shared_similarities = thrust::raw_pointer_cast(shared_similarities.data());
    float *dPtr_region_colorpatterns = thrust::raw_pointer_cast(region_colorpatterns.data());
    int *dPtr_min_element_labels = thrust::raw_pointer_cast(min_element_labels.data());
    

    shared_similarities.resize(nR*(num_regions-nR));
    segmentation_NRG_kernel<G,K> <<<num_regions-nR,min(nRt,1024), nR*(num_regions-nR)*sizeof(float)>>>(nR, dPtr_cell_offsets, dPtr_region_numcells, dPtr_region_offsets, dPtr_labels, dPtr_xadj, dPtr_adjncy, dPtr_region_colorpatterns, dPtr_shared_similarities);

    // // print
    // printf("%i %i\n", num_regions-nR, nRt);
    // thrust::host_vector<float> h6(shared_similarities);
    // for(int i = 0; i < h6.size(); i++)
    //     printf("%.2f ", h6[i]);
    // printf("\n");

    min_element_labels.resize(num_regions-nR);
    min_element_kernel<G,K> <<<num_regions-nR,1>>>(nR, dPtr_region_labels, dPtr_shared_similarities, dPtr_min_element_labels);

    thrust::copy(min_element_labels.begin(), min_element_labels.end(), region_labels.begin()+nR);
    relabel_kernel<G,K> <<<num_regions-nR,min(region_numcells[nR],1024)>>>(dPtr_cell_offsets, dPtr_region_numcells + nR, dPtr_region_offsets + nR, dPtr_labels, dPtr_region_labels + nR);
    
    int excess_regions = thrust::count_if(region_labels.begin()+nR, region_labels.end(), is_greater_than(nR));
    num_regions = nR + excess_regions;

    /*SORT AND REDUCE RESULTS*/
    thrust::sort_by_key(region_labels.begin(), region_labels.end(), 
        thrust::make_zip_iterator(thrust::make_tuple(region_sizes.begin(),region_numcells.begin(),region_colorpatterns.begin(), region_offsets.begin())));

    thrust::device_vector<int> region_labels2(num_regions);
    thrust::device_vector<int> region_numcells2(num_regions);
    thrust::device_vector<int> region_sizes2(num_regions);
    thrust::device_vector<float> region_colorpatterns2(num_regions);

    region_labels2.resize(num_regions);
    region_numcells2.resize(num_regions);
    region_sizes2.resize(num_regions);
    region_colorpatterns2.resize(num_regions);

    thrust::reduce_by_key(region_labels.begin(), region_labels.end(),
        thrust::make_zip_iterator(thrust::make_tuple(region_numcells.begin(), region_sizes.begin(), region_colorpatterns.begin())),
        region_labels2.begin(),
        thrust::make_zip_iterator(thrust::make_tuple(region_numcells2.begin(), region_sizes2.begin(), region_colorpatterns2.begin())),
        thrust::equal_to<int>(),
        TuplePlus<Tuple>());


    // print                                            
    printf("Number of regions = %i\n",num_regions);
    thrust::host_vector<int> h1(region_labels2);
    thrust::host_vector<int> h2(region_numcells2);
    thrust::host_vector<int> h3(region_sizes2);
    thrust::host_vector<float> h4(region_colorpatterns2);
    for(int i = 0; i < num_regions; i++) printf("%i ", h1[i]); printf("\n");
    for(int i = 0; i < num_regions; i++) printf("%i ", h2[i]); printf("\n");
    for(int i = 0; i < num_regions; i++) printf("%i ", h3[i]); printf("\n");
    for(int i = 0; i < num_regions; i++) printf("%.2f ", h4[i]); printf("\n");

}

/* SEGMENTATION : 3rd Operator */
template<class G, kernel K>
void CCudaGraphSegmentator<G,K>::segmentate3()
{
}

//////////////////////////////////////////////

template class CCudaGraphSegmentator<G,K1>;
template class CCudaGraphSegmentator<G,K2>;

//////////////////////////////////////////////
