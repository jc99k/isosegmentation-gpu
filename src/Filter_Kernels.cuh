#ifndef ISOSEGMENTATION_CUDA_FILTER_KERNELS_H
#define ISOSEGMENTATION_CUDA_FILTER_KERNELS_H

/*------------------------------------------*/
// warp-aggregated atomic increment
__device__ inline int atomicAggInc(int *ctr)
{
	int mask = __ballot(1);
	// select the leader
	int leader = __ffs(mask) - 1;
	// leader does the update
	int res;
	//if (lane_id() == leader)
	if ((threadIdx.x % WARP_SIZE) == leader)
		res = atomicAdd(ctr, __popc(mask));
	// broadcast result
	//res = warp_bcast(res, leader);
	res = __shfl(res, leader);;
	// each thread computes its own value
	//res += __popc(mask & ((1 << lane_id()) - 1));
	res += __popc(mask & ((1 << (threadIdx.x % WARP_SIZE)) - 1));
	return res;
} // atomicAggInc

#endif //ISOSEGMENTATION_CUDA_FILTER_KERNELS_H