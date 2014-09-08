// microbench.cpp : Defines the entry point for the console application.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

CUcontext hContext = 0;

#define CUDA_CHECK( fn ) do { \
		CUresult status = (fn); \
		if ( CUDA_SUCCESS != status ) { \
			const char* errstr; \
			cuGetErrorString(status, &errstr); \
			printf("CUDA Driver Failure (line %d of file %s):\n\t%s returned 0x%x (%s)\n", __LINE__, __FILE__, #fn, status, errstr); \
			if (hContext) cuCtxDestroy(hContext); \
			exit(EXIT_FAILURE); \
		} \
	} while (0)


int main(int argc, char* argv[])
{
	char deviceName[32];
	int devCount, ordinal, major, minor;
	CUdevice  hDevice;

	// Initialize the Driver API and find a device
	CUDA_CHECK( cuInit(0) );
	CUDA_CHECK( cuDeviceGetCount(&devCount) );
	for (ordinal = 0; ordinal < devCount; ordinal++)
	{
		CUDA_CHECK( cuDeviceGet(&hDevice, ordinal) );
		CUDA_CHECK( cuDeviceGetAttribute (&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, hDevice) );
		CUDA_CHECK( cuDeviceGetAttribute (&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, hDevice) );
		CUDA_CHECK( cuDeviceGetName(deviceName, sizeof(deviceName), hDevice) );
		if (major >= 5)
		{
			printf("Using: Id:%d %s (%d.%d)\n\n", ordinal, deviceName, major, minor);
			break;
		}
	}
	if (ordinal == devCount)
	{
		printf("No compute 5.0 device found, exiting.\n");
		exit(EXIT_FAILURE);
	}

	// First command line arg is the number of threads in multiples of 128
	int thread128 = 1;
	if (argc > 1)
		thread128 = atoi(argv[1]);
	if (thread128 > 8 || thread128 < 1)
		thread128 = 1;

	// Second command line arg is the number of lanes to print for each warp
	int lanes = 1;
	if (argc > 2)
		lanes = atoi(argv[2]);
	if (lanes > 32 || lanes < 1)
		lanes = 1;

	// threads = total number of threads
	int threads = thread128 * 128;
	size_t size = sizeof(int) * threads;

	// Setup our input and output buffers
	int* dataIn  = (int*)malloc(size);
	int* dataOut = (int*)malloc(size);
	int* clocks  = (int*)malloc(size);
	memset(dataIn, 0, size);

	CUmodule hModule;
	CUfunction hKernel;
	CUdeviceptr devIn, devOut, devClocks;

	// Init our context and device memory buffers
	CUDA_CHECK( cuCtxCreate(&hContext, 0, hDevice) );
	CUDA_CHECK( cuMemAlloc(&devIn, size) );
	CUDA_CHECK( cuMemAlloc(&devOut, size) );
	CUDA_CHECK( cuMemAlloc(&devClocks, size) );
	CUDA_CHECK( cuMemcpyHtoD(devIn, dataIn, size) );
	CUDA_CHECK( cuMemsetD8(devOut, 0, size) );
	CUDA_CHECK( cuMemsetD8(devClocks, 0, size) );

	// Load our kernel
	CUDA_CHECK( cuModuleLoad(&hModule, "microbench.cubin") );
	CUDA_CHECK( cuModuleGetFunction(&hKernel, hModule, "microbench") );

	// Setup the params
	void* params[] = { &devOut, &devClocks, &devIn };

	// Launch the kernel
	CUDA_CHECK( cuLaunchKernel(hKernel, 1, 1, 1, threads, 1, 1, 0, 0, params, 0) );
	CUDA_CHECK( cuCtxSynchronize() );

	// Get back our results from each kernel
	CUDA_CHECK( cuMemcpyDtoH(dataOut, devOut, size) );
	CUDA_CHECK( cuMemcpyDtoH(clocks, devClocks, size) );

	// Cleanup and shutdown of cuda
	CUDA_CHECK( cuModuleUnload(hModule) );
	CUDA_CHECK( cuMemFree(devIn) );
	CUDA_CHECK( cuMemFree(devOut) );
	CUDA_CHECK( cuMemFree(devClocks) );
	CUDA_CHECK( cuCtxDestroy(hContext) );
	hContext = 0;

	// Loop over and print results
	int count = 0, total = 0, min = 999999, max = 0;
	for(int tid = 0; tid < threads; tid += 32)
	{
		// Sometimes we want data on each thread, sometimes just one sample per warp is fine
		for (int lane = 0; lane < lanes; lane++)
			printf("w:%03d t:%04d l:%02d clocks:%04u out:%04u\n", tid/32, tid, lane, clocks[tid+lane], dataOut[tid+lane]);

		count++;
		total += clocks[tid];
		if (clocks[tid] < min) min = clocks[tid];
		if (clocks[tid] > max) max = clocks[tid];
	}
	printf("average: %.3f, min %d, max: %d\n", (float)total/count, min, max);

	// And free up host memory
	free(dataIn); free(dataOut); free(clocks);

	return 0;
}
