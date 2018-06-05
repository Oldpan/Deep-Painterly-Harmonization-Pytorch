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


int upsample_corr(THCudaIntTensor *curr_corrAB, int next_h, int next_w, THCudaIntTensor *next_corrAB)
{
    next_corrAB = THCudaIntTensor_new(state);

	THCudaIntTensor_resize3d(state, next_corrAB, next_h, next_w, 2);
	THCudaIntTensor_zero(state, next_corrAB);

	int curr_h = THCudaIntTensor_size(state, curr_corrAB, 0);
	int curr_w = THCudaIntTensor_size(state, curr_corrAB, 1);

	upsample_corr_kernel_L(
		THCudaIntTensor_data(state, curr_corrAB),
		THCudaIntTensor_data(state, next_corrAB),
		curr_h, curr_w, next_h, next_w
	);

	return 1;
}


int refineNNF(THCudaTensor* N_A, THCudaTensor *N_BP, THCudaIntTensor *init_corr,
              THCudaTensor* guide, THCudaIntTensor *tmask, THCudaIntTensor* corr, int patch, int niter)
{
	int c = THCudaTensor_size(state, N_BP, 0);
	int h = THCudaTensor_size(state, N_BP, 1);
	int w = THCudaTensor_size(state, N_BP, 2);

	corr = THCudaIntTensor_new(state);
	THCudaIntTensor_resize3d(state, corr, h, w, 2);
	THCudaIntTensor_zero(state, corr);

	for (int iter = 0; iter < niter; iter++) {
		printf("  iter=%d\n", iter);

		refineNNF_kernel_L(
        THCudaTensor_data   (state, N_A),
        THCudaTensor_data   (state, N_BP),
        THCudaIntTensor_data(state, init_corr),
        THCudaTensor_data   (state, guide),
        THCudaIntTensor_data(state, tmask),
        THCudaIntTensor_data(state, corr),
        patch, c, h, w);

//		checkCudaError(L);

		cudaMemcpy(
			THCudaIntTensor_data(state, init_corr),
			THCudaIntTensor_data(state, corr),
			sizeof(int) * h * w * 2,
			cudaMemcpyDeviceToDevice
		);
	}

	return 1;
}


int patchmatch(THCudaTensor* input, THCudaTensor* target,
                THCudaIntTensor* correspondence, int patch)
{
	int c1 = THCudaTensor_size(state, input, 0);
	int h1 = THCudaTensor_size(state, input, 1);
	int w1 = THCudaTensor_size(state, input, 2);

//	int c2 = THCudaTensor_size(state, target, 0);
	int h2 = THCudaTensor_size(state, target, 1);
	int w2 = THCudaTensor_size(state, target, 2);

	THCudaTensor *conv = THCudaTensor_new(state);
	THCudaTensor_resize2d(state, conv, h1*w1, h2*w2);
	THCudaTensor_zero(state, conv);

	assert(c1 == c2);

	patchmatch_conv_kernel_L(
	    THCudaTensor_data(state, input),
		THCudaTensor_data(state, target),
		THCudaTensor_data(state, conv),
		patch,
		c1,
		h1, w1,
		h2, w2
	);

	correspondence = THCudaIntTensor_new(state);
	THCudaIntTensor_resize3d(state, correspondence, h1, w1, 2);
	THCudaIntTensor_zero(state, correspondence);


    patchmatch_argmax_kernel_L(
    	THCudaTensor_data(state, conv),
		THCudaIntTensor_data(state, correspondence),
		patch,
		c1,
		h1, w1,
		h2, w2
    );

	THCudaTensor_free(state, conv);

	return 1;
}


int patchmatch_r(THCudaTensor* input, THCudaTensor* target,
                THCudaTensor* output, int patch, int stride)
{
    int c1 = THCudaTensor_size(state, input, 0);
	int h1 = THCudaTensor_size(state, input, 1);
	int w1 = THCudaTensor_size(state, input, 2);

//	int c2 = THCudaTensor_size(state, target, 0);
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

//    THCudaTensor *match = new_tensor_like(state, input);
	THCudaTensor_zero(state, output);

	THCudaIntTensor *correspondence = THCudaIntTensor_new(state);
	THCudaIntTensor_resize3d(state, correspondence, h1, w1, 2);
	THCudaIntTensor_zero(state, correspondence);


	patchmatch_r_argmax_kernel_L(
		THCudaTensor_data(state, conv),
		THCudaTensor_data(state, target),
		THCudaTensor_data(state, output),   // match -> output
		THCudaIntTensor_data(state, correspondence),
		c1,
		h1, w1,
		h2, w2
	);

    THCudaTensor_free(state, conv);

    return 1;
}


int Ring2(THCudaTensor *A, THCudaTensor *BP, THCudaIntTensor *corrAB, THCudaIntTensor *m, int ring, THCudaIntTensor *mask)
{

	int c = THCudaTensor_size(state, A, 0);
	int h = THCudaTensor_size(state, A, 1);
	int w = THCudaTensor_size(state, A, 2);

	m = THCudaIntTensor_new(state);
	THCudaIntTensor_resize2d(state, m, h, w);
	THCudaIntTensor_zero(state, m);

	Ring2_kernel_L(
		THCudaTensor_data(state, A),
		THCudaTensor_data(state, BP),
		THCudaIntTensor_data(state, corrAB),
		THCudaIntTensor_data(state, mask),
		THCudaIntTensor_data(state, m),
		ring, c, h, w
	);

//	checkCudaError(L);

	return 1;
}

int hist_remap2(THCudaTensor *I, int nI, THCudaTensor *mI, THCudaTensor *histJ, THCudaTensor *cumJ, THCudaTensor *minJ,
                THCudaTensor* maxJ,
                int nbins,THCudaTensor *sortI, THCudaIntTensor *idxI, THCudaTensor *R)
{

	int c = THCudaTensor_size(state, I, 0);
	int h = THCudaTensor_size(state, I, 1);
	int w = THCudaTensor_size(state, I, 2);

	hist_remap2_kernel_L(
		THCudaTensor_data(state, I),
		nI,
		THCudaTensor_data(state, mI),
		THCudaTensor_data(state, histJ),
		THCudaTensor_data(state, cumJ),
		THCudaTensor_data(state, minJ),
		THCudaTensor_data(state, maxJ),
		nbins,
		THCudaTensor_data(state, sortI),
		THCudaIntTensor_data(state, idxI),
		THCudaTensor_data(state, R),
		c, h, w
	);


	return 0;
}


int histogram(THCudaTensor *I, int nbins, THCudaTensor *minI, THCudaTensor *maxI,
              THCudaTensor *mask, THCudaTensor *hist)
{
	int c = THCudaTensor_size(state, I, 0);
	int h = THCudaTensor_size(state, I, 1);
	int w = THCudaTensor_size(state, I, 2);

	hist = THCudaTensor_new(state);
	THCudaTensor_resize2d(state, hist, c, nbins);
	THCudaTensor_zero(state, hist);

	histogram_kernel_L(
		THCudaTensor_data(state, I),
		THCudaTensor_data(state, minI),
		THCudaTensor_data(state, maxI),
		THCudaTensor_data(state, mask),
		nbins, c, h, w,
		THCudaTensor_data(state, hist)
	);

	return 1;
}
