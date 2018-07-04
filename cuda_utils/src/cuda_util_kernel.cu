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
// EPS:0.1 --> EPS:0.01
#define EPS 0.01

#undef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#undef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))


__global__ void histogram_kernel(
	float *I, float *minI, float *maxI, float *mask,
	int nbins, int c, int h, int w, float *hist
)
{
	int _id = blockIdx.x * blockDim.x + threadIdx.x;
	int size = h * w;

	if (_id < c * size) {
		int id = _id % size, dc = _id / size;

		if (mask[id] < EPS)
			return ;

		float val  = I[_id];

		float _minI = minI[dc];
		float _maxI = maxI[dc];


		if (_minI == _maxI) {
			_minI -= 1;
			_maxI += 1;
		}

		if (_minI <= val && val <= _maxI) {
			int idx = MIN((val - _minI) / (_maxI - _minI) * nbins, nbins-1);
			int index = dc * nbins + idx;
			atomicAdd(&hist[index], 1.0f);
		}

	}

	return ;
}


__global__ void hist_remap2_kernel(
	float *I, int nI, float *mI, float *histJ, float *cumJ,
	float *_minJ, float *_maxJ, int nbins,
	float *_sortI, int *_idxI, float *R, int c, int h, int w
)
{
	int _id = blockIdx.x * blockDim.x + threadIdx.x;
	int size = h * w;

	if (_id < c * size) {
		// _id = dc * size + id
		int id = _id % size, dc = _id / size;

		float minJ  = _minJ[dc];
		float maxJ  = _maxJ[dc];
		float stepJ = (maxJ - minJ) / nbins;

		int idxI = _idxI[_id] - 1;
		if (mI[idxI] < EPS)
			return ;
		int offset = h * w - nI;

		int cdf = id - offset;

		int s = 0;
		int e = nbins - 1;
		int m = (s + e) / 2;
		int binIdx = -1;

		while (s <= e) {
			// special handling for range boundary
			float cdf_e = m == nbins - 1 ?
						  cumJ[dc * nbins + m] + 0.5f :
						  cumJ[dc * nbins + m];
			float cdf_s = m == 0  ?
						  -0.5f :
						  cumJ[dc * nbins + m - 1];

			if (cdf >= cdf_e) {
				s = m + 1;
				m = (s + e) / 2;
			} else if (cdf < cdf_s) {
				e = m - 1;
				m = (s + e) / 2;
			} else {
				binIdx = m;    break;
			}
		}

		float hist  = histJ[dc * nbins + binIdx];
		float cdf_e = cumJ[dc * nbins + binIdx];
		float cdf_s = cdf_e - hist;
		float ratio = MIN(MAX((cdf - cdf_s) / (hist + 1e-8), 0.0f), 1.0f);
		float activation = minJ + (static_cast<float>(binIdx) + ratio) * stepJ;
		R[dc * size + idxI] = activation;
	}

	return ;
}


__global__ void upsample_corr_kernel(
	int *curr_corr, int *next_corr,
	int curr_h, int curr_w, int next_h, int next_w
)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < next_h * next_w) {
		int next_x = id % next_w, next_y = id / next_w;

		float w_ratio = (float)next_w / (float)curr_w;
		float h_ratio = (float)next_h / (float)curr_h;

		int curr_x = (next_x + 0.5) / w_ratio;
		int curr_y = (next_y + 0.5) / h_ratio;

		curr_x = MAX(MIN(curr_x, curr_w-1), 0);
		curr_y = MAX(MIN(curr_y, curr_h-1), 0);

		int curr_id = curr_y * curr_w + curr_x;

		int curr_x2 = curr_corr[2 * curr_id + 0];
		int curr_y2 = curr_corr[2 * curr_id + 1];

		int next_x2 = next_x + (curr_x2 - curr_x) * w_ratio + 0.5;
		int next_y2 = next_y + (curr_y2 - curr_y) * h_ratio + 0.5;

		next_x2 = MAX(MIN(next_x2, next_w-1), 0);
		next_y2 = MAX(MIN(next_y2, next_h-1), 0);

		next_corr[2 * id + 0] = next_x2;
		next_corr[2 * id + 1] = next_y2;
	}

	return ;
}


__global__ void refineNNF_kernel(
	float *N_A, float *N_BP,
	int *init_corr, float *guide,
	int *tmask, int *corr,
	int patch, int c, int h, int w
)
{
	int id1 = blockIdx.x * blockDim.x + threadIdx.x;
	int size = h * w;
	int r = (patch - 1) / 2;
	if (id1 < size) {
		int x1 = id1 % w, y1 = id1 / w;
		int x2 = init_corr[2*id1 + 0];
		int y2 = init_corr[2*id1 + 1];

		corr[2*id1 + 0] = x2;
		corr[2*id1 + 1] = y2;

		if (tmask[id1] < EPS)
			return ;

		float best_d = FLT_MAX;
		int best_x2 = x2, best_y2 = y2;

		for (int dx = -r; dx <= r; dx++)
		for (int dy = -r; dy <= r; dy++)
		{
			int new_x1 = x1 + dx;
			int new_y1 = y1 + dy;
			int new_id1 = new_y1 * w + new_x1;
			if (new_x1 >= 0 && new_x1 < w && new_y1 >= 0 && new_y1 < h) {
				int new_x2 = init_corr[2*new_id1 + 0] - dx;
				int new_y2 = init_corr[2*new_id1 + 1] - dy;
				int new_id2 = new_y2 * w + new_x2;
				if (new_x2 >= r && new_x2 < w - r - 1 && new_y2 >= r && new_y2 < h - r - 1) {

					float dist = 0;
					int cnt = 0;

					for (int _dx = -r; _dx <= r; _dx++)
					for (int _dy = -r; _dy <= r; _dy++)
					{
						int _new_x1 = x1 + _dx;
						int _new_y1 = y1 + _dy;
						int _new_id1 = _new_y1 * w + _new_x1;
						if (_new_x1 >= 0 && _new_x1 < w && _new_y1 >= 0 && _new_y1 < h) {
							int _new_x2 = init_corr[2*_new_id1 + 0] - _dx;
							int _new_y2 = init_corr[2*_new_id1 + 1] - _dy;
							int _new_id2 = _new_y2 * w + _new_x2;
							if (_new_x2 >= 0 && _new_x2 < w && _new_y2 >= 0 && _new_y2 < h) {
								float d = 0;
								for (int dc = 0; dc < 3; dc++) {
									float diff = guide[dc * size + new_id2] - guide[dc * size + _new_id2];
									d += diff * diff;
								}
								d = sqrt(d);
								dist += d;
								cnt++;
							}
						}
					}

					dist = dist / cnt;

					if (dist < best_d) {
						best_d = dist;
						best_x2 = new_x2;
						best_y2 = new_y2;
					}


				}
			}
		}

		corr[2*id1 + 0] = best_x2;
		corr[2*id1 + 1] = best_y2;

	}
	return ;
}

__global__ void patchmatch_conv_kernel
(
	float *input, float *target, float *conv,
	int patch, int c1, int h1, int w1, int h2, int w2
//	int *mask = NULL
)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int size1 = h1 * w1, size2 = h2 * w2;
	int N = size1 * size2;

	if (id < N) {
		conv[id] = -1;
		// id = id1 * size2 + id2
		int id1 = id / size2, id2 = id % size2;

//		if (mask && mask[id1] == 0)
//			return ;

		int x1 = id1 % w1, y1 = id1 / w1;
		int x2 = id2 % w2, y2 = id2 / w2;
		int kernel_radius  = (patch - 1) / 2;
		double conv_result = 0;
		// double sigma       = 0.5;
		// double sum_weight  = 0;
		// int cnt            = 0;
		for (int dy = -kernel_radius; dy <= kernel_radius; dy++) {
			for (int dx = -kernel_radius; dx <= kernel_radius; dx++) {
				int xx1 = x1 + dx, yy1 = y1 + dy;
				int xx2 = x2 + dx, yy2 = y2 + dy;
				if (0 <= xx1 && xx1 < w1 && 0 <= yy1 && yy1 < h1 &&
					0 <= xx2 && xx2 < w2 && 0 <= yy2 && yy2 < h2)
				{
					int _id1 = yy1 * w1 + xx1, _id2 = yy2 * w2 + xx2;
					// float weight = exp(-(dx*dx + dy*dy) / (2*sigma*sigma));
					for (int c = 0; c < c1; c++) {
						float term1 = input[c * size1 + _id1];
						float term2 = target[c * size2 + _id2];
						conv_result += term1 * term2;
						// conv_result += (term1 - term2) * (term1 - term2) * weight;
					}
					// cnt++;
					// sum_weight += weight;
				}
			}
		}

		// conv[id] = conv_result / cnt;
		// conv[id] = conv_result / sum_weight;
		conv[id] = conv_result;
	}

	return ;
}

__global__ void patchmatch_argmax_kernel(
	float *conv, int *correspondence, int patch,
	int c1, int h1, int w1, int h2, int w2
)
{
	int id1 = blockIdx.x * blockDim.x + threadIdx.x;
	int size1 = h1 * w1, size2 = h2 * w2;
	int kernel_radius = (patch - 1) / 2;
	if (id1 < size1) {
		float conv_max = -FLT_MAX;
		int y1 = id1 / w1, x1 = id1 % w1;

		for (int y2 = 0; y2 < h2; y2++) {
			for (int x2 = 0; x2 < w2; x2++) {
				int id2 = y2 * w2 + x2;
				int id = id1 * size2 + id2;
				float conv_result = conv[id];

				if (x2 < kernel_radius && !(x1 < kernel_radius))
					continue;
				if (x2 > w2 - 1 - kernel_radius && !(x1 > w1 - 1 - kernel_radius))
					continue;
				if (y2 < kernel_radius && !(y1 < kernel_radius))
					continue;
				if (y2 > h2 - 1 - kernel_radius && !(y1 > h1 - 1 - kernel_radius))
					continue;

				if (conv_result > conv_max) {
					conv_max = conv_result;
					correspondence[id1 * 2 + 0] = x2;
					correspondence[id1 * 2 + 1] = y2;
				}
				// if (conv_result < conv_min) {
				// 	conv_min = conv_result;
				// 	correspondence[id1 * 2 + 0] = x2;
				// 	correspondence[id1 * 2 + 1] = y2;
				// }
			}
		}

	}

	return ;
}


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

__global__ void Ring2_kernel(
	float *A, float *BP, int *corrAB, int *mask, int *m,
	int ring, int c, int h, int w
)
{
	int id1 = blockIdx.x * blockDim.x + threadIdx.x;
	int size = h * w;
	if (id1 < size) {
		// int y1 = id1 / w, x1 = id1 % w;
		if (mask[id1] != 0) {

			int y2 = corrAB[2 * id1 + 1], x2 = corrAB[2 * id1 + 0];
			for (int dx = -ring; dx <= ring; dx++)
				for (int dy = -ring; dy <= ring; dy++)
				{
					int _x2 = x2 + dx, _y2 = y2 + dy;
					if (_x2 >= 0 && _x2 < w && _y2 >= 0 && _y2 < h)
					{
						m[_y2 * w + _x2] = 1;
					}
				}
		}
	}

	return ;
}

__global__ void add(int *a, int *b, int *output)
{
    int i = threadIdx.x;
    output[i] = a[i] + b[i];

    return;
}

/*------------------------.cu and .c interface-----------------------------*/
int upsample_corr_kernel_L(int *curr_corrAB, int *next_corrAB,
	                       int curr_h, int curr_w, int next_h, int next_w)
{
	upsample_corr_kernel<<<(next_h*next_w-1)/TB+1, TB>>>(
		curr_corrAB,
		next_corrAB,
		curr_h, curr_w, next_h, next_w
	);

    return 1;
}


int refineNNF_kernel_L(float *N_A, float *N_BP, int *init_corr,
                     float *guide, int *tmask, int *corr, int patch, int c, int h, int w)
{
    refineNNF_kernel<<<(h*w-1)/TB+1, TB>>>(
        N_A,
        N_BP,
        init_corr,
        guide,
        tmask,
        corr,
        patch, c, h, w
    );

    return 1;
}

int patchmatch_conv_kernel_L(float *input, float *target, float *conv,
                                int patch,
                                int c1, int h1, int w1, int h2, int w2)
{
	int N = h1*w1*h2*w2;

    patchmatch_conv_kernel<<<(N-1)/TB+1, TB>>>(
    input,
    target,
    conv,
    patch,
    c1,
    h1, w1,
    h2, w2
    );

    return 1;
}

int patchmatch_argmax_kernel_L(float* conv, int* init_corr,int patch,
                                int c1, int h1, int w1, int h2, int w2)
{
	cudaError_t err;

    patchmatch_argmax_kernel<<<(h1*w1-1)/TB+1, TB>>>(
		conv,
		init_corr,
		patch,
		c1,
		h1, w1,
		h2, w2
	);

	err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

	return 1;
}


int patchmatch_r_conv_kernel_L(float *input, float *target, float *conv,
                                int patch, int stride,
                                int c1, int h1, int w1, int h2, int w2,
                                cudaStream_t stream)
{
	int N = h1*w1*h2*w2;

	cudaError_t err;

	patchmatch_r_conv_kernel<<<(N-1)/TB+1, TB, 0, stream>>>(
		input,
		target,
		conv,
		patch, stride,
		c1,
		h1, w1,
		h2, w2
	);

	err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


int patchmatch_r_argmax_kernel_L(
	float *conv, float *target, float *match, int *correspondence,
	int c1, int h1, int w1, int h2, int w2, cudaStream_t stream
)
{
	cudaError_t err;

	patchmatch_r_argmax_kernel<<<(h1*w1-1)/TB+1, TB, 0, stream>>>(
		conv,
		target,
		match,
		correspondence,
		c1,
		h1, w1,
		h2, w2
	);

	err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

	return 1;

}

int Ring2_kernel_L(
    float *A, float *BP, int *corrAB, int *mask, int *m,
	int ring, int c, int h, int w)
{
	Ring2_kernel<<<(h*w-1)/TB+1, TB>>>(
		A,
		BP,
		corrAB,
		mask,
		m,
		ring, c, h, w
	);

	return 1;
}


int hist_remap2_kernel_L
(
	float *I, int nI, float *mI, float *histJ, float *cumJ,
	float *minJ, float *maxJ, int nbins,
	float *sortI, int *idxI, float *R, int c, int h, int w
)
{

	cudaError_t err;

    hist_remap2_kernel<<<(c*h*w-1)/TB+1, TB>>>(
		I,
		nI,
		mI,
		histJ,
		cumJ,
		minJ,
		maxJ,
		nbins,
		sortI,
		idxI,
		R,
		c, h, w
	);

	err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

	return 1;
}


int histogram_kernel_L(
	float *I, float *minI, float *maxI, float *mask,
	int nbins, int c, int h, int w, float *hist
)
{
	histogram_kernel<<<(c*h*w-1)/TB+1, TB>>>(
		I,
		minI,
		maxI,
		mask,
		nbins, c, h, w,
		hist
	);

    return 1;
}


int add_L(int *a, int *b, int *output,int N)
{

    add<<<1,N>>>(
    a,
    b,
    output
    );

    return 1;
}
/*----------------------------------------------------------------------*/