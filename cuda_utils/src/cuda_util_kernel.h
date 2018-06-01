#ifndef _CUDA_UTIL_KERNEL_
#define _CUDA_UTIL_KERNEL_

#ifdef __cplusplus
extern "C" {
#endif


__global__ void patchmatch_r_conv_kernel(
	float *input, float *target, float *conv,
	int patch, int stride,
	int c1, int h1, int w1, int h2, int w2
);


__global__ void patchmatch_r_argmax_kernel(
	float *conv, float *target, float *match, int *correspondence,
	int c1, int h1, int w1, int h2, int w2
);


int patchmatch_r_conv_kernel_L(float *input, float *target, float *conv,
                                int patch, int stride,
                                int c1, int h1, int w1, int h2, int w2);

int patchmatch_r_argmax_kernel_L(
	float *conv, float *target, float *match, int *correspondence,
	int c1, int h1, int w1, int h2, int w2
);


#ifdef __cplusplus
}
#endif

#endif