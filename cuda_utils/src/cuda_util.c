#include <THC/THC.h>
#include <math.h>
#include <assert.h>
#include "cuda_util_kernel.h"
#include "cuda_util.h"

extern THCState *state;


THCudaTensor *new_tensor_like(THCState *state, THCudaTensor *x)
{
	THCudaTensor *y = THCudaTensor_new(state);
	THCudaTensor_resizeAs(state, y, x);
	return y;
}


int pathcmatch_r(THCudaTensor* input, THCudaTensor* target,
                THCudaTensor* output, int patch, int stride)
{
    int c1 = THCudaTensor_size(state, input, 0);
	int h1 = THCudaTensor_size(state, input, 1);
	int w1 = THCudaTensor_size(state, input, 2);

	int c2 = THCudaTensor_size(state, target, 0);
	int h2 = THCudaTensor_size(state, target, 1);
	int w2 = THCudaTensor_size(state, target, 2);

    THCudaTensor *conv = THCudaTensor_new(state);
    THCudaTensor_resize2d(state, conv, h1*w1, h2*w2);
	THCudaTensor_zero(state, conv);

    assert(c1 == c2);

	patchmatch_r_conv_kernel_L(
	    THCudaTensor_data(state, input),
		THCudaTensor_data(state, target),
		THCudaTensor_data(state, conv),
		patch, stride,
		c1,
		h1, w1,
		h2, w2
		);

    THCudaTensor *match = new_tensor_like(state, input);
	THCudaTensor_zero(state, match);

	THCudaIntTensor *correspondence = THCudaIntTensor_new(state);
	THCudaIntTensor_resize3d(state, correspondence, h1, w1, 2);
	THCudaIntTensor_zero(state, correspondence);


	patchmatch_r_argmax_kernel_L(
		THCudaTensor_data(state, conv),
		THCudaTensor_data(state, target),
		THCudaTensor_data(state, match),
		THCudaIntTensor_data(state, correspondence),
		c1,
		h1, w1,
		h2, w2
	);

    output = match;
    THCudaTensor_free(state, conv);

//    err = cudaGetLastError();
//    if (cudaSuccess != err)
//    {
//        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
//        exit(-1);
//    }

    return 1;
}
