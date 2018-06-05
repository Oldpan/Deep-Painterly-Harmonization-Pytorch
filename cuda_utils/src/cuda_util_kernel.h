#ifndef _CUDA_UTIL_KERNEL_
#define _CUDA_UTIL_KERNEL_

#ifdef __cplusplus
extern "C" {
#endif

__global__ void histogram_kernel(
	float *I, float *minI, float *maxI, float *mask,
	int nbins, int c, int h, int w, float *hist
);

__global__ void hist_remap2_kernel(
	float *I, int nI, float *mI, float *histJ, float *cumJ,
	float *_minJ, float *_maxJ, int nbins,
	float *_sortI, int *_idxI, float *R, int c, int h, int w
);

__global__ void patchmatch_r_conv_kernel(
	float *input, float *target, float *conv,
	int patch, int stride,
	int c1, int h1, int w1, int h2, int w2
);


__global__ void patchmatch_r_argmax_kernel(
	float *conv, float *target, float *match, int *correspondence,
	int c1, int h1, int w1, int h2, int w2
);


__global__ void Ring2_kernel(
	float *A, float *BP, int *corrAB, int *mask, int *m,
	int ring, int c, int h, int w
);

__global__ void patchmatch_argmax_kernel(
	float *conv, int *correspondence, int patch,
	int c1, int h1, int w1, int h2, int w2
);

__global__ void patchmatch_conv_kernel(
	float *input, float *target, float *conv,
	int patch, int c1, int h1, int w1, int h2, int w2
//	int *mask = NULL
);

__global__ void refineNNF_kernel(
	float *N_A, float *N_BP,
	int *init_corr, float *guide,
	int *tmask, int *corr,
	int patch, int c, int h, int w
);

__global__ void upsample_corr_kernel(
	int *curr_corr, int *next_corr,
	int curr_h, int curr_w, int next_h, int next_w
);

int patchmatch_r_conv_kernel_L(float *input, float *target, float *conv,
                                int patch, int stride,
                                int c1, int h1, int w1, int h2, int w2);

int patchmatch_r_argmax_kernel_L(
	float *conv, float *target, float *match, int *correspondence,
	int c1, int h1, int w1, int h2, int w2
);


int Ring2_kernel_L(
    float *A, float *BP, int *corrAB, int *mask, int *m,
	int ring, int c, int h, int w);

int patchmatch_argmax_kernel_L(float* conv, int* correspondence,int patch,
                                int c1, int h1, int w1, int h2, int w2);

int patchmatch_conv_kernel_L(float *input, float *target, float *conv,
                                int patch,
                                int c1, int h1, int w1, int h2, int w2);

int refineNNF_kernel_L(float *N_A, float *N_BP, int *init_corr,
                     float *guide, int *tmask, int *corr, int patch, int c, int h, int w);

int upsample_corr_kernel_L(int *curr_corrAB, int *next_corrAB,
	                       int curr_h, int curr_w, int next_h, int next_w);

int hist_remap2_kernel_L
(
	float *I, int nI, float *mI, float *histJ, float *cumJ,
	float *_minJ, float *_maxJ, int nbins,
	float *_sortI, int *_idxI, float *R, int c, int h, int w
);

int histogram_kernel_L(
	float *I, float *minI, float *maxI, float *mask,
	int nbins, int c, int h, int w, float *hist
);


#ifdef __cplusplus
}
#endif

#endif