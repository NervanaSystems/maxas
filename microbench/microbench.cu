
// Note this file isn't configured to automatically compile

#include <device_functions.h>
#include <device_launch_parameters.h>

// nvcc -arch sm_50 -m 32 -cubin microbench.cu
// maxas.pl -e microbench.cubin microbench.sass
// maxas.pl -i microbench.sass microbench.cubin

// Use extern C so C++ doesn't mangle our kernel name
extern "C" __global__ void  microbench(int *out, int *clocks, int *in)
{
	__shared__ int share[4096];

	int tid = threadIdx.x;
	int blkdim = blockDim.x;

	int start = clock(); 

	share[tid] = in[tid + blkdim];

	__syncthreads();

	int end = clock();

	clocks[tid] = end - start;

	out[tid] = share[tid ^ 1];
}


