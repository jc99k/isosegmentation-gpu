#ifndef ISOSEGMENTATION_CUDA_SIMILARITYFUNCTION_KERNELS_H
#define ISOSEGMENTATION_CUDA_SIMILARITYFUNCTION_KERNELS_H

//#include "kernel_constants.h"
#define DELTA 50.0f

///* SIMILARITY FUNCTION */
template<bool Iso>
__device__ /*__forceinline__*/ float similarity(float i, float j) {
	if (Iso) {
		for (int k = 0; k < Num_isovalues; k++)
		{
			if ((i - Isovalues[k])*(j - Isovalues[k]) <= 0)
			//if ((i <= Isovalues[k])^(j <= Isovalues[k]))
			//if (signbit(i - Isovalues[k]) ^ signbit(j - Isovalues[k]))
			{
				//printf("i = %.2f , j = %.2f CROSS ISOVAL of %.2f\n", i, j, Isovalues[k]);
				return 1.0f;
			}

		}
	}
	return 1.0f - expf(-1 * abs(i - j) / DELTA);
}

#endif
