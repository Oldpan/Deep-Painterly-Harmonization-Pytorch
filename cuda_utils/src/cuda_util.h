int patchmatch_r(THCudaTensor* input, THCudaTensor* target,
                THCudaTensor* output, int patch, int stride);

int Ring2(THCudaTensor *A, THCudaTensor *BP, THCudaIntTensor *corrAB, THCudaIntTensor *m, int ring, THCudaIntTensor *mask);

int patchmatch(THCudaTensor* input, THCudaTensor* target,
                THCudaIntTensor* correspondence, int patch);

int refineNNF(THCudaTensor* N_A, THCudaTensor *N_BP, THCudaIntTensor *init_corr,
              THCudaTensor* guide, THCudaIntTensor *tmask, THCudaIntTensor* corr, int patch, int niter);

int upsample_corr(THCudaIntTensor *curr_corrAB, int next_h, int next_w, THCudaIntTensor *next_corrAB);

int hist_remap2(THCudaTensor *I, int nI, THCudaTensor *mI, THCudaTensor *histJ, THCudaTensor *cumJ, THCudaTensor *minJ,
                THCudaTensor* maxJ,
                int nbins, THCudaTensor *sortI, THCudaIntTensor *idxI, THCudaTensor *R);

int histogram(THCudaTensor *I, int nbins, THCudaTensor *minI, THCudaTensor *maxI,
              THCudaTensor *mask, THCudaTensor *hist);