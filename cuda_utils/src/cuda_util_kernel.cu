#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <math_constants.h>
#include <math_functions.h>
#include <stdint.h>
#include <unistd.h>
#include <omp.h>
#include <getopt.h>
#include "curand_kernel.h"
#include "cuda_util_kernel.h"


#define TB 256
#define EPS 0.1

#undef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#undef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))


__global__ void patchmatch_r_conv_kernel(
	float *input, float *target, float *conv,
	int patch, int stride,
	int c1, int h1, int w1, int h2, int w2
)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int size1 = h1 * w1, size2 = h2 * w2;
	int N = size1 * size2;

	if (id < N) {
		int id1 = id / size2, id2 = id % size2;

		int x1 = id1 % w1, y1 = id1 / w1;
		int x2 = id2 % w2, y2 = id2 / w2;

		int kernel_radius = (patch - 1) / 2;

		double conv_result = 0, norm_1 = 0, norm_2 = 0;
		for (int dy = -kernel_radius; dy <= kernel_radius; dy+=stride) {
			for (int dx = -kernel_radius; dx <= kernel_radius; dx+=stride) {
				int xx1 = x1 + dx, yy1 = y1 + dy;
				int xx2 = x2 + dx, yy2 = y2 + dy;
				if (0 <= xx1 && xx1 < w1 && 0 <= yy1 && yy1 < h1 &&
					0 <= xx2 && xx2 < w2 && 0 <= yy2 && yy2 < h2)
				{
					int _id1 = yy1 * w1 + xx1, _id2 = yy2 * w2 + xx2;
					for (int c = 0; c < c1; c++) {
						float term1 = input[c * size1 + _id1];
						float term2 = target[c * size2 + _id2];
						conv_result += term1 * term2;
						norm_1      += term1 * term1;
						norm_2      += term2 * term2;
					}

				}
			}
		}

		norm_1 = sqrt(norm_1);
		norm_2 = sqrt(norm_2);

		conv[id] = conv_result / (norm_1 * norm_2 + 1e-9);
	}

	return ;
}

__global__ void patchmatch_r_argmax_kernel(
	float *conv, float *target, float *match, int *correspondence,
	int c1, int h1, int w1, int h2, int w2
)
{
	int id1 = blockIdx.x * blockDim.x + threadIdx.x;
	int size1 = h1 * w1, size2 = h2 * w2;

	if (id1 < size1) {
		//int x1 = id1 % w1, y1 = id1 / w1;
		double conv_max = -1e20;

		for (int y2 = 0; y2 < h2; y2++) {
			for (int x2 = 0; x2 < w2; x2++) {
				int id2 = y2 * w2 + x2;

				int id = id1 * size2 + id2;
				float conv_result = conv[id];

				if (conv_result > conv_max) {
					conv_max = conv_result;
					correspondence[id1 * 2 + 0] = x2;
					correspondence[id1 * 2 + 1] = y2;
					for (int c = 0; c < c1; c++) {
						match[c * size1 + id1] = target[c * size2 + id2];
					}
				}
			}
		}

	}

	return ;
}


int patchmatch_r_conv_kernel_L(float *input, float *target, float *conv,
                                int patch, int stride,
                                int c1, int h1, int w1, int h2, int w2)
{
	int N = h1*w1*h2*w2;

	patchmatch_r_conv_kernel<<<(N-1)/TB+1, TB>>>(
		input,
		target,
		conv,
		patch, stride,
		c1,
		h1, w1,
		h2, w2
	);

    return 1;
}


int patchmatch_r_argmax_kernel_L(
	float *conv, float *target, float *match, int *correspondence,
	int c1, int h1, int w1, int h2, int w2
)
{
	patchmatch_r_argmax_kernel<<<(h1*w1-1)/TB+1, TB>>>(
		conv,
		target,
		match,
		correspondence,
		c1,
		h1, w1,
		h2, w2
	);

	return 1;

}

